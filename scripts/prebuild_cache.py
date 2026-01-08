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
_VALID_STUDIES: Optional[Set[Tuple[int, int]]] = None
_MIMIC_CXR_PATH: Optional[Path] = None
_QUALITY_GRADE: Optional[str] = None
_SG_BASE_DIR: Optional[Path] = None


def init_worker(valid_studies: Set, mimic_cxr_path: str, quality_grade: str, sg_base_dir: str):
    """Initialize worker process with shared data."""
    global _VALID_STUDIES, _MIMIC_CXR_PATH, _QUALITY_GRADE, _SG_BASE_DIR
    _VALID_STUDIES = valid_studies
    _MIMIC_CXR_PATH = Path(mimic_cxr_path)
    _QUALITY_GRADE = quality_grade
    _SG_BASE_DIR = Path(sg_base_dir)


def find_image(mimic_cxr_path: Path, subject_id: int, study_id: int) -> Tuple[Optional[Path], Optional[str]]:
    """
    Find the image file for a given subject and study.
    
    MIMIC-CXR-JPG structure:
        files/p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        
    Returns:
        (image_path, dicom_id) or (None, None) if not found
    """
    p_prefix = f"p{str(subject_id)[:2]}"
    study_dir = mimic_cxr_path / 'files' / p_prefix / f"p{subject_id}" / f"s{study_id}"
    
    if not study_dir.exists():
        return None, None
    
    # Find .jpg files (prefer frontal views if multiple)
    jpg_files = list(study_dir.glob('*.jpg'))
    if not jpg_files:
        return None, None
    
    # Return first image found
    image_path = jpg_files[0]
    dicom_id = image_path.stem
    
    return image_path, dicom_id


def find_scene_graph(sg_base_dir: Path, subject_id: int, study_id: int) -> Optional[Path]:
    """
    Find the scene graph file for a given subject and study.
    
    MIMIC-Ext-CXR-QBA structure:
        scene_data/p{XX}/p{subject_id}/s{study_id}.scene_graph.json
        
    Returns:
        scene_graph_path or None if not found
    """
    p_prefix = f"p{str(subject_id)[:2]}"
    sg_path = sg_base_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
    
    if sg_path.exists():
        return sg_path
    return None


def meets_quality_grade(actual_grade: str, required_grade: str) -> bool:
    """
    Check if actual quality grade meets or exceeds required grade.
    
    Quality hierarchy: A++ > A+ > A > B > C > U
    """
    grade_order = {'A++': 5, 'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'U': 0}
    actual_val = grade_order.get(actual_grade, 0)
    required_val = grade_order.get(required_grade, 0)
    return actual_val >= required_val


def load_valid_studies(mimic_cxr_path: Path, split: str) -> Optional[Set[Tuple[int, int]]]:
    """
    Load valid studies for the given split from MIMIC-CXR split file.
    
    Returns:
        Set of (subject_id, study_id) tuples for the split
    """
    import pandas as pd
    
    split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
    if not split_file.exists():
        split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
    
    if not split_file.exists():
        logger.warning("Split file not found, using all data")
        return None
    
    compression = 'gzip' if str(split_file).endswith('.gz') else None
    splits_df = pd.read_csv(split_file, compression=compression)
    
    # Map split names (MIMIC uses 'validate' not 'val')
    split_name = 'validate' if split == 'val' else split
    splits_df = splits_df[splits_df['split'] == split_name]
    
    valid_studies = set(zip(
        splits_df['subject_id'].astype(int), 
        splits_df['study_id'].astype(int)
    ))
    
    logger.info(f"Found {len(valid_studies):,} studies in '{split_name}' split")
    return valid_studies


def process_patient_dir(patient_dir_str: str) -> List[Dict[str, Any]]:
    """
    Process a single patient directory and extract all QA samples.
    
    This function is called in parallel across many patient directories.
    
    Args:
        patient_dir_str: Path to patient directory (e.g., qa/p10/p10000032)
        
    Returns:
        List of sample dictionaries for this patient
    """
    global _VALID_STUDIES, _MIMIC_CXR_PATH, _QUALITY_GRADE, _SG_BASE_DIR
    
    samples = []
    patient_dir = Path(patient_dir_str)
    
    try:
        # Parse subject_id from directory name (p10000032 -> 10000032)
        subject_id = int(patient_dir.name[1:])
        
        # Process each QA file for this patient
        for qa_file in patient_dir.glob('s*.qa.json'):
            try:
                # Parse study_id from filename (s50414267.qa.json -> 50414267)
                study_id_str = qa_file.stem.split('.')[0]
                study_id = int(study_id_str[1:])
                
                # Check if this study is in the valid split
                if _VALID_STUDIES and (subject_id, study_id) not in _VALID_STUDIES:
                    continue
                
                # Find corresponding image
                image_path, dicom_id = find_image(_MIMIC_CXR_PATH, subject_id, study_id)
                if image_path is None:
                    continue
                
                # Find scene graph
                sg_path = find_scene_graph(_SG_BASE_DIR, subject_id, study_id)
                
                # Load QA data
                with open(qa_file) as f:
                    qa_data = json.load(f)
                
                # Process each question
                for q in qa_data.get('questions', []):
                    # Quality filter (skip if quality_grade is 'all')
                    if _QUALITY_GRADE and _QUALITY_GRADE.lower() not in ('', 'all', 'none'):
                        q_quality = q.get('question_quality', q.get('quality', {}))
                        if isinstance(q_quality, dict):
                            grade = q_quality.get('overall', q_quality.get('grade', 'B'))
                        elif isinstance(q_quality, str):
                            grade = q_quality
                        else:
                            grade = 'B'
                        
                        if not meets_quality_grade(grade, _QUALITY_GRADE):
                            continue
                    
                    # Add sample
                    samples.append({
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'dicom_id': dicom_id,
                        'image_path': str(image_path),
                        'scene_graph_path': str(sg_path) if sg_path else None,
                        'question_id': q.get('question_id', ''),
                        'question_type': q.get('question_type', 'unknown'),
                        'question_strategy': q.get('question_strategy', ''),
                        'question': q.get('question', ''),
                        'answers': q.get('answers', []),
                        'obs_ids': q.get('obs_ids', []),
                    })
                    
            except Exception as e:
                # Skip problematic QA files
                continue
                
    except Exception as e:
        # Skip problematic patient directories
        pass
    
    return samples


def collect_patient_directories(qa_dir: Path) -> List[str]:
    """
    Collect all patient directories from the QA folder.
    
    Structure: qa/p{XX}/p{subject_id}/
    
    Returns:
        List of patient directory paths (strings for pickling)
    """
    patient_dirs = []
    
    for p_group in sorted(qa_dir.iterdir()):
        if not p_group.is_dir() or not p_group.name.startswith('p'):
            continue
        
        for patient_dir in p_group.iterdir():
            if patient_dir.is_dir() and patient_dir.name.startswith('p'):
                patient_dirs.append(str(patient_dir))
    
    return patient_dirs


def get_cache_key(
    mimic_cxr_path: str,
    mimic_qa_path: str,
    split: str,
    quality_grade: str,
) -> str:
    """Generate a unique cache key based on configuration."""
    config_str = f"{mimic_cxr_path}|{mimic_qa_path}|{split}|{quality_grade}"
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def parallel_build_cache(
    mimic_cxr_path: Path,
    mimic_qa_path: Path,
    split: str,
    quality_grade: str,
    num_workers: int,
) -> List[Dict[str, Any]]:
    """
    Build sample cache using parallel processing across patient directories.
    
    This is MUCH faster than the original which parallelized across only 10
    patient groups. Now we parallelize across ~65,000 patient directories.
    """
    # Load valid studies for this split
    valid_studies = load_valid_studies(mimic_cxr_path, split)
    
    # Paths
    qa_dir = mimic_qa_path / 'qa'
    sg_dir = mimic_qa_path / 'scene_data'
    
    if not qa_dir.exists():
        logger.error(f"QA directory not found: {qa_dir}")
        return []
    
    # Collect all patient directories
    logger.info("Collecting patient directories...")
    patient_dirs = collect_patient_directories(qa_dir)
    logger.info(f"Found {len(patient_dirs):,} patient directories to process")
    
    # Process in parallel
    logger.info(f"Processing with {num_workers} workers...")
    start_time = time.time()
    all_samples = []
    
    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(valid_studies, str(mimic_cxr_path), quality_grade, str(sg_dir))
    ) as pool:
        # Process in batches for progress updates
        batch_size = max(100, len(patient_dirs) // 20)
        
        for i in range(0, len(patient_dirs), batch_size):
            batch = patient_dirs[i:i + batch_size]
            results = pool.map(process_patient_dir, batch)
            
            for samples in results:
                all_samples.extend(samples)
            
            # Progress update
            elapsed = time.time() - start_time
            progress = (i + len(batch)) / len(patient_dirs)
            eta = (elapsed / progress * (1 - progress)) if progress > 0 else 0
            
            logger.info(
                f"  [{i + len(batch):,}/{len(patient_dirs):,}] "
                f"{len(all_samples):,} samples | "
                f"ETA: {eta/60:.1f}min"
            )
    
    total_time = time.time() - start_time
    logger.info(f"Processing complete: {len(all_samples):,} samples in {total_time/60:.1f}min")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(
        description='Pre-build dataset cache for distributed training (FAST)'
    )
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml',
                        help='Config file path')
    parser.add_argument('--mimic_cxr_path', type=str,
                        help='Path to MIMIC-CXR-JPG (overrides config)')
    parser.add_argument('--mimic_qa_path', type=str,
                        help='Path to MIMIC-Ext-CXR-QBA (overrides config)')
    parser.add_argument('--splits', nargs='+', default=['train'],
                        help='Splits to cache (default: train)')
    parser.add_argument('--cache_dir', type=str, default='.cache/dataset_samples',
                        help='Cache directory')
    parser.add_argument('--quality_grade', type=str, default='all',
                        help='Quality grade filter (all for pretraining)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of CPU workers (default: auto)')
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild even if cache exists')
    
    args = parser.parse_args()
    
    # Auto-detect workers (leave some for system)
    num_cpus = mp.cpu_count()
    num_workers = args.num_workers or min(num_cpus - 4, 40)
    
    logger.info("=" * 70)
    logger.info("  FAST DATASET CACHE PRE-BUILDER")
    logger.info("=" * 70)
    logger.info(f"  Available CPUs:  {num_cpus}")
    logger.info(f"  Using workers:   {num_workers}")
    logger.info("=" * 70)
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        try:
            from configs.mimic_cxr_config import load_config_from_file
            config = load_config_from_file(args.config)
            logger.info(f"  Loaded config: {args.config}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    # Get paths from args or config
    mimic_cxr_path = args.mimic_cxr_path
    mimic_qa_path = args.mimic_qa_path
    quality_grade = args.quality_grade
    
    if config:
        if not mimic_cxr_path:
            mimic_cxr_path = config.data.mimic_cxr_jpg_path
        if not mimic_qa_path:
            mimic_qa_path = config.data.mimic_ext_cxr_qba_path
        if not quality_grade or quality_grade == 'all':
            quality_grade = config.data.quality_grade or 'all'
    
    if not mimic_cxr_path or not mimic_qa_path:
        logger.error("Please provide --mimic_cxr_path and --mimic_qa_path or a valid config")
        sys.exit(1)
    
    mimic_cxr_path = Path(mimic_cxr_path)
    mimic_qa_path = Path(mimic_qa_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  MIMIC-CXR-JPG:     {mimic_cxr_path}")
    logger.info(f"  MIMIC-Ext-CXR-QBA: {mimic_qa_path}")
    logger.info(f"  Quality grade:     {quality_grade}")
    logger.info(f"  Cache directory:   {cache_dir}")
    logger.info("=" * 70)
    
    total_samples = 0
    
    for split in args.splits:
        logger.info(f"\n{'='*70}")
        logger.info(f"  Building cache for split: {split.upper()}")
        logger.info(f"{'='*70}")
        
        # Generate cache path
        cache_key = get_cache_key(str(mimic_cxr_path), str(mimic_qa_path), split, quality_grade)
        cache_path = cache_dir / f"samples_{split}_{cache_key}.pkl"
        
        # Check if cache exists
        if cache_path.exists() and not args.force:
            logger.info(f"Cache already exists: {cache_path}")
            logger.info("Use --force to rebuild")
            
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            logger.info(f"  Cached samples: {len(samples):,}")
            total_samples += len(samples)
            continue
        
        # Build samples
        samples = parallel_build_cache(
            mimic_cxr_path=mimic_cxr_path,
            mimic_qa_path=mimic_qa_path,
            split=split,
            quality_grade=quality_grade,
            num_workers=num_workers,
        )
        
        # Save to cache
        logger.info(f"Saving {len(samples):,} samples to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Split '{split}': {len(samples):,} samples ({cache_size_mb:.1f} MB)")
        total_samples += len(samples)
    
    logger.info(f"\n{'='*70}")
    logger.info("  CACHE PRE-BUILD COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Cache dir:     {cache_dir}")
    logger.info("")
    logger.info("  Now run distributed training:")
    logger.info("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
    logger.info("    --config configs/pretrain_config.yaml \\")
    logger.info("    --deepspeed_config configs/deepspeed_config.json")
    logger.info("=" * 70)


if __name__ == '__main__':
    # Use fork method for efficient memory sharing on Linux
    mp.set_start_method('fork', force=True)
    main()
