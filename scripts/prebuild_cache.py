#!/usr/bin/env python3
"""
Pre-build dataset cache before distributed training.

PARALLEL VERSION - Uses all CPU cores efficiently!

Run this ONCE before starting distributed training:
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
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for worker processes
_MIMIC_CXR_PATH = None
_SG_DIR = None
_VALID_STUDIES = None
_QUALITY_GRADE = None


def init_worker(mimic_cxr_path: str, sg_dir: str, valid_studies_list: List, quality_grade: str):
    """Initialize global variables in worker processes."""
    global _MIMIC_CXR_PATH, _SG_DIR, _VALID_STUDIES, _QUALITY_GRADE
    _MIMIC_CXR_PATH = Path(mimic_cxr_path)
    _SG_DIR = Path(sg_dir)
    _VALID_STUDIES = set(tuple(x) for x in valid_studies_list) if valid_studies_list else None
    _QUALITY_GRADE = quality_grade


def find_image(subject_id: int, study_id: int) -> Tuple[Optional[str], Optional[str]]:
    """Find the image file for a given subject and study."""
    p_prefix = f"p{str(subject_id)[:2]}"
    study_dir = _MIMIC_CXR_PATH / 'files' / p_prefix / f"p{subject_id}" / f"s{study_id}"
    
    if not study_dir.exists():
        return None, None
    
    jpg_files = list(study_dir.glob('*.jpg'))
    if not jpg_files:
        return None, None
    
    return str(jpg_files[0]), jpg_files[0].stem


def find_scene_graph(subject_id: int, study_id: int) -> Optional[str]:
    """Find the scene graph file for a given subject and study."""
    p_prefix = f"p{str(subject_id)[:2]}"
    sg_path = _SG_DIR / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
    return str(sg_path) if sg_path.exists() else None


def meets_quality_grade(actual_grade: str, required_grade: str) -> bool:
    """Check if actual quality grade meets or exceeds required grade."""
    grade_order = {'A++': 5, 'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'U': 0}
    return grade_order.get(actual_grade, 0) >= grade_order.get(required_grade, 0)


def process_qa_file(qa_file_str: str) -> List[Dict[str, Any]]:
    """Process a single QA file and return samples. Runs in worker process."""
    samples = []
    try:
        qa_file = Path(qa_file_str)
        patient_dir = qa_file.parent
        subject_id = int(patient_dir.name[1:])
        study_id_str = qa_file.stem.split('.')[0]
        study_id = int(study_id_str[1:])
        
        # Check split first (fast set lookup)
        if _VALID_STUDIES and (subject_id, study_id) not in _VALID_STUDIES:
            return []
        
        # Find image
        image_path, dicom_id = find_image(subject_id, study_id)
        if image_path is None:
            return []
        
        # Find scene graph
        sg_path = find_scene_graph(subject_id, study_id)
        
        # Load QA
        with open(qa_file) as f:
            qa_data = json.load(f)
        
        for q in qa_data.get('questions', []):
            # Quality filter
            if _QUALITY_GRADE and _QUALITY_GRADE.lower() not in ('', 'all', 'none'):
                q_quality = q.get('question_quality', q.get('quality', {}))
                if isinstance(q_quality, dict):
                    grade = q_quality.get('overall', 'B')
                else:
                    grade = str(q_quality) if q_quality else 'B'
                
                if not meets_quality_grade(grade, _QUALITY_GRADE):
                    continue
            
            samples.append({
                'subject_id': subject_id,
                'study_id': study_id,
                'dicom_id': dicom_id,
                'image_path': image_path,
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
    
    return samples


def load_valid_studies(mimic_cxr_path: Path, split: str) -> Optional[Set[Tuple[int, int]]]:
    """Load valid studies for the given split."""
    import pandas as pd
    
    split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
    if not split_file.exists():
        split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
    
    if not split_file.exists():
        return None
    
    compression = 'gzip' if str(split_file).endswith('.gz') else None
    splits_df = pd.read_csv(split_file, compression=compression)
    split_name = 'validate' if split == 'val' else split
    splits_df = splits_df[splits_df['split'] == split_name]
    
    valid_studies = set(zip(
        splits_df['subject_id'].astype(int), 
        splits_df['study_id'].astype(int)
    ))
    logger.info(f"Found {len(valid_studies):,} studies in '{split_name}' split")
    return valid_studies


def build_cache_parallel(
    mimic_cxr_path: Path,
    mimic_qa_path: Path,
    split: str,
    quality_grade: str,
    valid_studies: Set[Tuple[int, int]],
    num_workers: int,
) -> List[Dict[str, Any]]:
    """
    Build sample cache using parallel processing.
    Processes QA files across all CPU cores.
    """
    qa_dir = mimic_qa_path / 'qa'
    sg_dir = mimic_qa_path / 'scene_data'
    
    if not qa_dir.exists():
        logger.error(f"QA directory not found: {qa_dir}")
        return []
    
    # Collect all QA file paths (this is fast)
    logger.info("Collecting QA file paths...")
    qa_files = []
    for p_group in sorted(qa_dir.iterdir()):
        if p_group.is_dir() and p_group.name.startswith('p'):
            for patient_dir in p_group.iterdir():
                if patient_dir.is_dir() and patient_dir.name.startswith('p'):
                    qa_files.extend([str(f) for f in patient_dir.glob('s*.qa.json')])
    
    total_files = len(qa_files)
    logger.info(f"Found {total_files:,} QA files to process")
    logger.info(f"Using {num_workers} CPU workers")
    
    # Convert valid_studies set to list for pickling
    valid_studies_list = list(valid_studies) if valid_studies else None
    
    # Process in parallel
    all_samples = []
    start_time = time.time()
    processed = 0
    last_log_time = start_time
    
    # Use imap_unordered for better load balancing
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(str(mimic_cxr_path), str(sg_dir), valid_studies_list, quality_grade)
    ) as pool:
        # Process in chunks for progress reporting
        chunk_size = max(100, total_files // (num_workers * 10))
        
        for samples in pool.imap_unordered(process_qa_file, qa_files, chunksize=chunk_size):
            all_samples.extend(samples)
            processed += 1
            
            # Progress logging every 10 seconds
            current_time = time.time()
            if current_time - last_log_time >= 10:
                elapsed = current_time - start_time
                progress = processed / total_files
                eta = (elapsed / progress * (1 - progress)) if progress > 0 else 0
                rate = processed / elapsed
                
                logger.info(
                    f"  [{processed:,}/{total_files:,}] "
                    f"{len(all_samples):,} samples | "
                    f"{progress*100:.1f}% | "
                    f"{rate:.0f} files/sec | "
                    f"ETA: {eta/60:.1f}min"
                )
                last_log_time = current_time
    
    total_time = time.time() - start_time
    rate = total_files / total_time
    logger.info(f"Complete: {len(all_samples):,} samples in {total_time/60:.1f} minutes ({rate:.0f} files/sec)")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Pre-build dataset cache (PARALLEL)')
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml')
    parser.add_argument('--mimic_cxr_path', type=str)
    parser.add_argument('--mimic_qa_path', type=str)
    parser.add_argument('--splits', nargs='+', default=['train'])
    parser.add_argument('--cache_dir', type=str, default='.cache/dataset_samples')
    parser.add_argument('--quality_grade', type=str, default='all')
    parser.add_argument('--num_workers', type=int, default=0, help='0 = auto-detect')
    parser.add_argument('--force', action='store_true')
    
    args = parser.parse_args()
    
    # Auto-detect workers
    num_workers = args.num_workers if args.num_workers > 0 else max(1, cpu_count() - 2)
    
    logger.info("=" * 70)
    logger.info("  PARALLEL DATASET CACHE PRE-BUILDER")
    logger.info(f"  Using {num_workers} CPU cores (of {cpu_count()} available)")
    logger.info("=" * 70)
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        try:
            from configs.mimic_cxr_config import load_config_from_file
            config = load_config_from_file(args.config)
            logger.info(f"  Loaded config: {args.config}")
        except Exception as e:
            logger.warning(f"Config error: {e}")
    
    # Get paths
    mimic_cxr_path = args.mimic_cxr_path or (config.data.mimic_cxr_jpg_path if config else None)
    mimic_qa_path = args.mimic_qa_path or (config.data.mimic_ext_cxr_qba_path if config else None)
    quality_grade = args.quality_grade or (config.data.quality_grade if config else 'all')
    
    if not mimic_cxr_path or not mimic_qa_path:
        logger.error("Need --mimic_cxr_path and --mimic_qa_path")
        sys.exit(1)
    
    mimic_cxr_path = Path(mimic_cxr_path)
    mimic_qa_path = Path(mimic_qa_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  CXR: {mimic_cxr_path}")
    logger.info(f"  QA:  {mimic_qa_path}")
    logger.info(f"  Quality: {quality_grade}")
    logger.info("=" * 70)
    
    for split in args.splits:
        logger.info(f"\n  Building cache for: {split.upper()}")
        
        # Load valid studies first
        valid_studies = load_valid_studies(mimic_cxr_path, split)
        
        # Cache key
        cache_key = hashlib.md5(
            f"{mimic_cxr_path}|{mimic_qa_path}|{split}|{quality_grade}".encode()
        ).hexdigest()[:12]
        cache_path = cache_dir / f"samples_{split}_{cache_key}.pkl"
        
        if cache_path.exists() and not args.force:
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            logger.info(f"  Cache exists: {len(samples):,} samples")
            logger.info(f"  Use --force to rebuild")
            continue
        
        # Build samples in parallel
        samples = build_cache_parallel(
            mimic_cxr_path=mimic_cxr_path,
            mimic_qa_path=mimic_qa_path,
            split=split,
            quality_grade=quality_grade,
            valid_studies=valid_studies,
            num_workers=num_workers,
        )
        
        # Save cache
        logger.info(f"Saving {len(samples):,} samples to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        cache_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved: {cache_path} ({cache_mb:.1f} MB)")
    
    logger.info("\n" + "=" * 70)
    logger.info("  CACHE COMPLETE!")
    logger.info("  Now run training:")
    logger.info("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
    logger.info("    --config configs/pretrain_config.yaml \\")
    logger.info("    --deepspeed_config configs/deepspeed_config.json")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
