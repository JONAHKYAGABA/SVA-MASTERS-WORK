#!/usr/bin/env python3
"""
Pre-build dataset cache using ThreadPoolExecutor.

Uses threads (not processes) - no BrokenPipe errors!
Perfect for I/O-bound file reading.

Run ONCE before training:
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml
"""

import os
import sys
import argparse
import logging
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CacheBuilder:
    """Thread-safe cache builder using shared memory."""
    
    def __init__(
        self,
        mimic_cxr_path: Path,
        mimic_qa_path: Path,
        valid_studies: Set[Tuple[int, int]],
        num_threads: int = 48
    ):
        self.mimic_cxr_path = mimic_cxr_path
        self.sg_dir = mimic_qa_path / 'scene_data'
        self.valid_studies = valid_studies
        self.num_threads = num_threads
        
        # Counters (thread-safe)
        self.lock = Lock()
        self.processed = 0
        self.skipped_split = 0
        self.skipped_img = 0
    
    def find_image(self, subject_id: int, study_id: int) -> Tuple[Optional[str], Optional[str]]:
        """Find image file."""
        p_prefix = f"p{str(subject_id)[:2]}"
        study_dir = self.mimic_cxr_path / 'files' / p_prefix / f"p{subject_id}" / f"s{study_id}"
        if not study_dir.exists():
            return None, None
        jpg_files = list(study_dir.glob('*.jpg'))
        if not jpg_files:
            return None, None
        return str(jpg_files[0]), jpg_files[0].stem
    
    def find_scene_graph(self, subject_id: int, study_id: int) -> Optional[str]:
        """Find scene graph file."""
        p_prefix = f"p{str(subject_id)[:2]}"
        sg_path = self.sg_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
        return str(sg_path) if sg_path.exists() else None
    
    def process_file(self, qa_file: Path) -> List[Dict[str, Any]]:
        """Process a single QA file. Thread-safe."""
        samples = []
        try:
            subject_id = int(qa_file.parent.name[1:])
            study_id = int(qa_file.stem.split('.')[0][1:])
            
            # Check split
            if self.valid_studies and (subject_id, study_id) not in self.valid_studies:
                with self.lock:
                    self.skipped_split += 1
                return []
            
            # Find image
            img_path, dicom_id = self.find_image(subject_id, study_id)
            if not img_path:
                with self.lock:
                    self.skipped_img += 1
                return []
            
            # Find scene graph
            sg_path = self.find_scene_graph(subject_id, study_id)
            
            # Load QA (this is the slow I/O part - threads help here!)
            with open(qa_file) as f:
                qa_data = json.load(f)
            
            for q in qa_data.get('questions', []):
                samples.append({
                    'subject_id': subject_id,
                    'study_id': study_id,
                    'dicom_id': dicom_id,
                    'image_path': img_path,
                    'scene_graph_path': sg_path,
                    'question_id': q.get('question_id', ''),
                    'question_type': q.get('question_type', 'unknown'),
                    'question_strategy': q.get('question_strategy', ''),
                    'question': q.get('question', ''),
                    'answers': q.get('answers', []),
                    'obs_ids': q.get('obs_ids', []),
                })
        except Exception:
            pass
        
        with self.lock:
            self.processed += 1
        
        return samples


def load_valid_studies(mimic_cxr_path: Path, split: str) -> Set[Tuple[int, int]]:
    """Load valid studies for split."""
    import pandas as pd
    
    split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
    if not split_file.exists():
        split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
    
    if not split_file.exists():
        return set()
    
    compression = 'gzip' if str(split_file).endswith('.gz') else None
    df = pd.read_csv(split_file, compression=compression)
    split_name = 'validate' if split == 'val' else split
    df = df[df['split'] == split_name]
    
    valid = set(zip(df['subject_id'].astype(int), df['study_id'].astype(int)))
    logger.info(f"Found {len(valid):,} studies in '{split_name}' split")
    return valid


def build_cache(
    mimic_cxr_path: Path,
    mimic_qa_path: Path,
    split: str,
    num_threads: int,
) -> List[Dict[str, Any]]:
    """Build cache using ThreadPoolExecutor."""
    
    qa_dir = mimic_qa_path / 'qa'
    
    # Load valid studies into RAM (fast set lookup)
    valid_studies = load_valid_studies(mimic_cxr_path, split)
    
    # Collect all QA file paths
    logger.info("Collecting QA files...")
    qa_files = []
    for p_group in sorted(qa_dir.iterdir()):
        if p_group.is_dir() and p_group.name.startswith('p'):
            for patient_dir in p_group.iterdir():
                if patient_dir.is_dir() and patient_dir.name.startswith('p'):
                    qa_files.extend(list(patient_dir.glob('s*.qa.json')))
    
    total = len(qa_files)
    logger.info(f"Found {total:,} QA files")
    logger.info(f"Using {num_threads} threads")
    
    # Create builder (shares state across threads)
    builder = CacheBuilder(mimic_cxr_path, mimic_qa_path, valid_studies, num_threads)
    
    # Process with threads
    all_samples = []
    start = time.time()
    last_log = start
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {executor.submit(builder.process_file, f): f for f in qa_files}
        
        # Collect results as they complete
        for future in as_completed(futures):
            samples = future.result()
            all_samples.extend(samples)
            
            # Progress logging
            now = time.time()
            if now - last_log >= 5:
                elapsed = now - start
                pct = builder.processed / total
                eta = (elapsed / pct * (1 - pct)) / 60 if pct > 0 else 0
                rate = builder.processed / elapsed
                
                logger.info(
                    f"  [{builder.processed:,}/{total:,}] {len(all_samples):,} samples | "
                    f"{pct*100:.1f}% | {rate:.0f} files/sec | ETA: {eta:.1f}min"
                )
                last_log = now
    
    elapsed = time.time() - start
    logger.info(f"Complete: {len(all_samples):,} samples in {elapsed/60:.1f} min ({total/elapsed:.0f} files/sec)")
    logger.info(f"  Skipped: {builder.skipped_split:,} (wrong split), {builder.skipped_img:,} (no image)")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pretrain_config.yaml')
    parser.add_argument('--mimic_cxr_path', type=str)
    parser.add_argument('--mimic_qa_path', type=str)
    parser.add_argument('--splits', nargs='+', default=['train'])
    parser.add_argument('--cache_dir', default='.cache/dataset_samples')
    parser.add_argument('--num_threads', type=int, default=48, help='Number of I/O threads')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("  DATASET CACHE BUILDER (ThreadPoolExecutor)")
    logger.info(f"  Using {args.num_threads} threads for parallel I/O")
    logger.info("=" * 60)
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        try:
            from configs.mimic_cxr_config import load_config_from_file
            config = load_config_from_file(args.config)
            logger.info(f"  Config: {args.config}")
        except Exception as e:
            logger.warning(f"Config error: {e}")
    
    mimic_cxr_path = Path(args.mimic_cxr_path or config.data.mimic_cxr_jpg_path)
    mimic_qa_path = Path(args.mimic_qa_path or config.data.mimic_ext_cxr_qba_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  CXR: {mimic_cxr_path}")
    logger.info(f"  QA:  {mimic_qa_path}")
    logger.info("=" * 60)
    
    for split in args.splits:
        logger.info(f"\nBuilding cache for: {split.upper()}")
        
        cache_key = hashlib.md5(
            f"{mimic_cxr_path}|{mimic_qa_path}|{split}".encode()
        ).hexdigest()[:12]
        cache_path = cache_dir / f"samples_{split}_{cache_key}.pkl"
        
        if cache_path.exists() and not args.force:
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            logger.info(f"Cache exists: {len(samples):,} samples")
            logger.info(f"Use --force to rebuild")
            continue
        
        samples = build_cache(
            mimic_cxr_path, 
            mimic_qa_path, 
            split, 
            args.num_threads
        )
        
        logger.info(f"Saving cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {cache_path} ({mb:.1f} MB)")
    
    logger.info("\n" + "=" * 60)
    logger.info("CACHE COMPLETE! Now run training:")
    logger.info("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
    logger.info("    --config configs/pretrain_config.yaml \\")
    logger.info("    --deepspeed_config configs/deepspeed_config.json")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
