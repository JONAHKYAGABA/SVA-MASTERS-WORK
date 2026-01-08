#!/usr/bin/env python3
"""
Pre-build dataset cache before distributed training.

FAST VERSION: Parallelizes across PATIENT directories (thousands)
instead of patient groups (only 10).

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
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables (shared via fork)
_VALID_STUDIES = None
_MIMIC_CXR_PATH = None
_QUALITY_GRADE = None
_SG_BASE_DIR = None


def init_worker(valid_studies: Set, mimic_cxr_path: str, quality_grade: str, sg_base_dir: str):
    """Initialize worker process with shared data."""
    global _VALID_STUDIES, _MIMIC_CXR_PATH, _QUALITY_GRADE, _SG_BASE_DIR
    _VALID_STUDIES = valid_studies
    _MIMIC_CXR_PATH = Path(mimic_cxr_path)
    _QUALITY_GRADE = quality_grade
    _SG_BASE_DIR = Path(sg_base_dir)


def process_patient_dir(patient_dir_str: str) -> List[Dict[str, Any]]:
    """Process a single patient directory. Fast - handles one patient."""
    global _VALID_STUDIES, _MIMIC_CXR_PATH, _QUALITY_GRADE, _SG_BASE_DIR
    
    samples = []
    patient_dir = Path(patient_dir_str)
    
    try:
        subject_id = int(patient_dir.name[1:])  # Remove 'p' prefix
        p_prefix = f"p{str(subject_id)[:2]}"
        
        for qa_file in patient_dir.glob('s*.qa.json'):
            try:
                study_id_str = qa_file.stem.split('.')[0]
                study_id = int(study_id_str[1:])
                
                # Check split
                if _VALID_STUDIES and (subject_id, study_id) not in _VALID_STUDIES:
                    continue
                
                # Find image
                study_dir = _MIMIC_CXR_PATH / 'files' / p_prefix / f"p{subject_id}" / f"s{study_id}"
                if not study_dir.exists():
                    continue
                
                jpg_files = list(study_dir.glob('*.jpg'))
                if not jpg_files:
                    continue
                
                image_path = jpg_files[0]
                dicom_id = image_path.stem
                
                # Scene graph path
                sg_path = _SG_BASE_DIR / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
                sg_path_str = str(sg_path) if sg_path.exists() else None
                
                # Load QA
                with open(qa_file) as f:
                    qa_data = json.load(f)
                
                for q in qa_data.get('questions', []):
                    # Quality filter (skip for 'all')
                    if _QUALITY_GRADE and _QUALITY_GRADE.lower() not in ('', 'all', 'none'):
                        q_quality = q.get('question_quality', q.get('quality', {}))
                        if isinstance(q_quality, dict):
                            grade = q_quality.get('overall', 'B')
                        else:
                            grade = str(q_quality) if q_quality else 'B'
                        
                        grade_order = {'A++': 5, 'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'U': 0}
                        if grade_order.get(grade, 0) < grade_order.get(_QUALITY_GRADE, 0):
                            continue
                    
                    samples.append({
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'dicom_id': dicom_id,
                        'image_path': str(image_path),
                        'scene_graph_path': sg_path_str,
                        'question_id': q.get('question_id', ''),
                        'question_type': q.get('question_type', 'unknown'),
                        'question_strategy': q.get('question_strategy', ''),
                        'question': q.get('question', ''),
                        'answers': q.get('answers', []),
                        'obs_ids': q.get('obs_ids', []),
                    })
                    
            except Exception:
                continue
                
    except Exception:
        pass
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Pre-build dataset cache (FAST)')
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml')
    parser.add_argument('--mimic_cxr_path', type=str)
    parser.add_argument('--mimic_qa_path', type=str)
    parser.add_argument('--splits', nargs='+', default=['train'])
    parser.add_argument('--cache_dir', type=str, default='.cache/dataset_samples')
    parser.add_argument('--quality_grade', type=str, default='all')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    
    args = parser.parse_args()
    
    # Auto-detect workers
    num_cpus = mp.cpu_count()
    num_workers = args.num_workers or min(num_cpus - 4, 40)  # Leave 4 for system
    
    logger.info("=" * 70)
    logger.info("  FAST DATASET CACHE PRE-BUILDER")
    logger.info("=" * 70)
    logger.info(f"  CPUs: {num_cpus} | Workers: {num_workers}")
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        try:
            from configs.mimic_cxr_config import load_config_from_file
            config = load_config_from_file(args.config)
            logger.info(f"  Config: {args.config}")
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
    logger.info("=" * 70)
    
    for split in args.splits:
        logger.info(f"\n  Building cache for: {split.upper()}")
        
        # Cache key
        cache_key = hashlib.md5(
            f"{mimic_cxr_path}|{mimic_qa_path}|{split}|{quality_grade}".encode()
        ).hexdigest()[:12]
        cache_path = cache_dir / f"samples_{split}_{cache_key}.pkl"
        
        if cache_path.exists() and not args.force:
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            logger.info(f"  Cache exists: {len(samples):,} samples")
            continue
        
        # Load split info
        import pandas as pd
        split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
        if not split_file.exists():
            split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
        
        valid_studies = None
        if split_file.exists():
            splits_df = pd.read_csv(split_file, compression='gzip' if str(split_file).endswith('.gz') else None)
            split_name = 'validate' if split == 'val' else split
            splits_df = splits_df[splits_df['split'] == split_name]
            valid_studies = set(zip(splits_df['subject_id'].astype(int), splits_df['study_id'].astype(int)))
            logger.info(f"  Valid studies: {len(valid_studies):,}")
        
        # Collect ALL patient directories
        qa_dir = mimic_qa_path / 'qa'
        sg_dir = mimic_qa_path / 'scene_data'
        
        logger.info("  Collecting patient directories...")
        patient_dirs = []
        for p_group in qa_dir.iterdir():
            if p_group.is_dir() and p_group.name.startswith('p'):
                for patient_dir in p_group.iterdir():
                    if patient_dir.is_dir() and patient_dir.name.startswith('p'):
                        patient_dirs.append(str(patient_dir))
        
        logger.info(f"  Found {len(patient_dirs):,} patient directories")
        logger.info(f"  Processing with {num_workers} workers...")
        
        # Process in parallel
        start_time = time.time()
        all_samples = []
        
        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(valid_studies, str(mimic_cxr_path), quality_grade, str(sg_dir))
        ) as pool:
            # Process in batches for progress
            batch_size = max(100, len(patient_dirs) // 20)
            
            for i in range(0, len(patient_dirs), batch_size):
                batch = patient_dirs[i:i + batch_size]
                results = pool.map(process_patient_dir, batch)
                
                for samples in results:
                    all_samples.extend(samples)
                
                elapsed = time.time() - start_time
                progress = (i + len(batch)) / len(patient_dirs)
                eta = (elapsed / progress * (1 - progress)) if progress > 0 else 0
                
                logger.info(f"  [{i + len(batch):,}/{len(patient_dirs):,}] "
                           f"{len(all_samples):,} samples | "
                           f"ETA: {eta/60:.1f}min")
        
        total_time = time.time() - start_time
        
        # Save cache
        logger.info(f"  Saving {len(all_samples):,} samples to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        cache_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Done: {len(all_samples):,} samples in {total_time/60:.1f}min ({cache_mb:.1f}MB)")
    
    logger.info("\n" + "=" * 70)
    logger.info("  CACHE COMPLETE! Now run:")
    logger.info("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
    logger.info("    --config configs/pretrain_config.yaml \\")
    logger.info("    --deepspeed_config configs/deepspeed_config.json")
    logger.info("=" * 70)


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
