#!/usr/bin/env python3
"""
Pre-build dataset cache before distributed training.

This script scans all QA files and builds a cache of samples
that can be instantly loaded by all GPU processes during training.

Run this ONCE before starting distributed training:
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml

This prevents NCCL timeout issues caused by slow dataset loading
across multiple GPU processes.
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
from functools import partial

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable for valid studies (shared across processes via fork)
_VALID_STUDIES = None
_MIMIC_CXR_PATH = None
_QUALITY_GRADE = None
_QUESTION_TYPES = None


def init_worker(valid_studies_path: str, mimic_cxr_path: str, quality_grade: str, question_types: Optional[List[str]]):
    """Initialize worker process with shared data."""
    global _VALID_STUDIES, _MIMIC_CXR_PATH, _QUALITY_GRADE, _QUESTION_TYPES
    
    # Load valid studies from file (more efficient than pickling large sets)
    if valid_studies_path and os.path.exists(valid_studies_path):
        with open(valid_studies_path, 'rb') as f:
            _VALID_STUDIES = pickle.load(f)
    else:
        _VALID_STUDIES = None
    
    _MIMIC_CXR_PATH = Path(mimic_cxr_path)
    _QUALITY_GRADE = quality_grade
    _QUESTION_TYPES = question_types


def process_patient_group(p_group_path: str) -> List[Dict[str, Any]]:
    """
    Process a single patient group directory (p10, p11, etc.)
    This function runs in parallel across multiple CPU cores.
    """
    global _VALID_STUDIES, _MIMIC_CXR_PATH, _QUALITY_GRADE, _QUESTION_TYPES
    
    samples = []
    p_group = Path(p_group_path)
    
    try:
        for patient_dir in p_group.iterdir():
            if not patient_dir.is_dir() or not patient_dir.name.startswith('p'):
                continue
            
            for qa_file in patient_dir.glob('s*.qa.json'):
                try:
                    # Parse IDs from path
                    subject_id = int(patient_dir.name[1:])
                    study_id_str = qa_file.stem.split('.')[0]
                    study_id = int(study_id_str[1:])
                    
                    # Check if in valid split
                    if _VALID_STUDIES and (subject_id, study_id) not in _VALID_STUDIES:
                        continue
                    
                    # Load QA data
                    with open(qa_file) as f:
                        qa_data = json.load(f)
                    
                    # Find corresponding image
                    image_path, dicom_id = find_image(_MIMIC_CXR_PATH, subject_id, study_id)
                    if image_path is None:
                        continue
                    
                    # Find scene graph
                    sg_dir = p_group.parent.parent / 'scene_data'
                    sg_path = find_scene_graph(sg_dir, subject_id, study_id)
                    
                    # Process each question
                    questions = qa_data.get('questions', [])
                    
                    for q in questions:
                        # Quality filter
                        if _QUALITY_GRADE and _QUALITY_GRADE.lower() not in ('', 'all', 'none'):
                            q_quality = q.get('question_quality', q.get('quality', {}))
                            if isinstance(q_quality, dict):
                                quality_rating = q_quality.get('overall', q_quality.get('grade', 'B'))
                            elif isinstance(q_quality, str):
                                quality_rating = q_quality
                            else:
                                quality_rating = 'B'
                            
                            if not meets_quality_grade(quality_rating, _QUALITY_GRADE):
                                continue
                        
                        # Question type filter
                        q_type = q.get('question_type', 'unknown')
                        if _QUESTION_TYPES and q_type not in _QUESTION_TYPES:
                            continue
                        
                        samples.append({
                            'subject_id': subject_id,
                            'study_id': study_id,
                            'dicom_id': dicom_id,
                            'image_path': str(image_path),
                            'scene_graph_path': str(sg_path) if sg_path else None,
                            'question_id': q.get('question_id', ''),
                            'question_type': q_type,
                            'question_strategy': q.get('question_strategy', ''),
                            'question': q.get('question', ''),
                            'answers': q.get('answers', []),
                            'obs_ids': q.get('obs_ids', []),
                        })
                        
                except Exception as e:
                    continue
    except Exception as e:
        logger.warning(f"Error processing {p_group_path}: {e}")
    
    return samples


def find_image(mimic_cxr_path: Path, subject_id: int, study_id: int) -> Tuple[Optional[Path], Optional[str]]:
    """Find the image file for a given subject and study."""
    # Format: files/p{XX}/p{subject_id}/s{study_id}/*.jpg
    p_prefix = f"p{str(subject_id)[:2]}"
    study_dir = mimic_cxr_path / 'files' / p_prefix / f"p{subject_id}" / f"s{study_id}"
    
    if study_dir.exists():
        # Find first .jpg file (prefer frontal views)
        jpg_files = list(study_dir.glob('*.jpg'))
        if jpg_files:
            return jpg_files[0], jpg_files[0].stem
    
    return None, None


def find_scene_graph(sg_dir: Path, subject_id: int, study_id: int) -> Optional[Path]:
    """Find the scene graph file for a given subject and study."""
    p_prefix = f"p{str(subject_id)[:2]}"
    sg_path = sg_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
    
    if sg_path.exists():
        return sg_path
    return None


def meets_quality_grade(actual_grade: str, required_grade: str) -> bool:
    """Check if actual quality grade meets or exceeds required grade."""
    grade_order = {'A++': 5, 'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'U': 0}
    actual_val = grade_order.get(actual_grade, 0)
    required_val = grade_order.get(required_grade, 0)
    return actual_val >= required_val


def load_valid_studies(mimic_cxr_path: Path, split: str) -> Optional[Set[Tuple[int, int]]]:
    """Load valid studies for the given split."""
    import pandas as pd
    
    split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
    if not split_file.exists():
        split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
    
    if split_file.exists():
        if str(split_file).endswith('.gz'):
            splits_df = pd.read_csv(split_file, compression='gzip')
        else:
            splits_df = pd.read_csv(split_file)
        
        # Map split names
        split_name = 'validate' if split == 'val' else split
        splits_df = splits_df[splits_df['split'] == split_name]
        
        valid_studies = set(zip(
            splits_df['subject_id'].astype(int), 
            splits_df['study_id'].astype(int)
        ))
        logger.info(f"Found {len(valid_studies)} studies in '{split_name}' split")
        return valid_studies
    
    return None


def parallel_build_cache(
    mimic_cxr_path: str,
    mimic_qa_path: str,
    split: str = 'train',
    quality_grade: str = 'all',
    question_types: Optional[List[str]] = None,
    num_workers: Optional[int] = None,
    cache_dir: str = '.cache/dataset_samples',
) -> List[Dict[str, Any]]:
    """
    Build sample cache using parallel processing across CPUs.
    """
    mimic_cxr_path = Path(mimic_cxr_path)
    mimic_qa_path = Path(mimic_qa_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers (use fewer to avoid memory issues)
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)  # Cap at 16 to avoid memory issues
    else:
        num_workers = min(num_workers, 24)  # Cap at 24 max
    
    logger.info(f"Using {num_workers} CPU workers for parallel processing")
    
    # Load valid studies for this split
    valid_studies = load_valid_studies(mimic_cxr_path, split)
    
    # Save valid_studies to temp file (more efficient than pickling in each worker)
    valid_studies_path = cache_dir / f".valid_studies_{split}.pkl"
    if valid_studies:
        with open(valid_studies_path, 'wb') as f:
            pickle.dump(valid_studies, f)
    else:
        valid_studies_path = None
    
    # Find QA directory
    qa_dir = mimic_qa_path / 'qa'
    if not qa_dir.exists():
        logger.error(f"QA directory not found: {qa_dir}")
        return []
    
    # Get all patient group directories
    p_groups = sorted([str(p) for p in qa_dir.iterdir() if p.is_dir() and p.name.startswith('p')])
    logger.info(f"Found {len(p_groups)} patient groups to process")
    
    # Process in parallel using Pool with initializer
    all_samples = []
    start_time = time.time()
    
    try:
        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(str(valid_studies_path) if valid_studies_path else '', str(mimic_cxr_path), quality_grade, question_types)
        ) as pool:
            # Use imap_unordered for better progress tracking
            results = pool.imap_unordered(process_patient_group, p_groups, chunksize=1)
            
            completed = 0
            for samples in results:
                completed += 1
                all_samples.extend(samples)
                
                # Progress logging
                if completed % max(1, len(p_groups) // 10) == 0 or completed == len(p_groups):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(p_groups) - completed) / rate if rate > 0 else 0
                    logger.info(f"  Progress: {completed}/{len(p_groups)} groups ({100*completed/len(p_groups):.1f}%) - "
                               f"{len(all_samples):,} samples - ETA: {eta:.0f}s")
    
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        logger.info("Falling back to sequential processing...")
        
        # Fallback to sequential
        init_worker(str(valid_studies_path) if valid_studies_path else '', str(mimic_cxr_path), quality_grade, question_types)
        
        for i, p_group in enumerate(p_groups):
            samples = process_patient_group(p_group)
            all_samples.extend(samples)
            
            if (i + 1) % max(1, len(p_groups) // 10) == 0:
                logger.info(f"  Progress: {i+1}/{len(p_groups)} - {len(all_samples):,} samples")
    
    finally:
        # Cleanup temp file
        if valid_studies_path and valid_studies_path.exists():
            valid_studies_path.unlink()
    
    total_time = time.time() - start_time
    logger.info(f"Scan complete: {len(all_samples):,} samples in {total_time:.1f}s")
    
    return all_samples


def get_cache_key(
    mimic_cxr_path: str,
    mimic_qa_path: str,
    split: str,
    quality_grade: str,
    question_types: Optional[List[str]] = None,
) -> str:
    """Generate a unique cache key based on configuration."""
    config_str = (
        f"cxr:{mimic_cxr_path}|"
        f"qa:{mimic_qa_path}|"
        f"split:{split}|"
        f"quality:{quality_grade}|"
        f"qtypes:{sorted(question_types) if question_types else 'all'}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(
        description='Pre-build dataset cache for distributed training'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/pretrain_config.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--mimic_cxr_path',
        type=str,
        help='Path to MIMIC-CXR-JPG (overrides config)'
    )
    parser.add_argument(
        '--mimic_qa_path',
        type=str,
        help='Path to MIMIC-Ext-CXR-QBA (overrides config)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train'],
        help='Splits to cache (default: train)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='.cache/dataset_samples',
        help='Cache directory'
    )
    parser.add_argument(
        '--quality_grade',
        type=str,
        default='all',
        help='Quality grade filter (all for pretraining)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='Number of CPU workers (default: 16, max: 24)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if cache exists'
    )
    
    args = parser.parse_args()
    
    # Get number of CPUs
    num_cpus = mp.cpu_count()
    num_workers = min(args.num_workers, 24)  # Cap to avoid memory issues
    
    logger.info("=" * 70)
    logger.info("  DATASET CACHE PRE-BUILDER")
    logger.info("=" * 70)
    logger.info(f"  Available CPUs:    {num_cpus}")
    logger.info(f"  Using workers:     {num_workers}")
    logger.info("=" * 70)
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        try:
            from configs.mimic_cxr_config import load_config_from_file
            config = load_config_from_file(args.config)
            logger.info(f"Loaded config from: {args.config}")
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
        if not quality_grade:
            quality_grade = config.data.quality_grade or 'all'
    
    if not mimic_cxr_path or not mimic_qa_path:
        logger.error("Please provide --mimic_cxr_path and --mimic_qa_path or a valid config")
        sys.exit(1)
    
    logger.info(f"  MIMIC-CXR-JPG:     {mimic_cxr_path}")
    logger.info(f"  MIMIC-Ext-CXR-QBA: {mimic_qa_path}")
    logger.info(f"  Splits:            {args.splits}")
    logger.info(f"  Quality grade:     {quality_grade}")
    logger.info(f"  Cache directory:   {args.cache_dir}")
    logger.info(f"  Force rebuild:     {args.force}")
    logger.info("=" * 70)
    
    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    start_time = time.time()
    
    for split in args.splits:
        logger.info(f"\n{'='*70}")
        logger.info(f"  Building cache for split: {split.upper()}")
        logger.info(f"{'='*70}")
        
        # Generate cache path
        cache_key = get_cache_key(mimic_cxr_path, mimic_qa_path, split, quality_grade, None)
        cache_path = cache_dir / f"samples_{split}_{cache_key}.pkl"
        
        # Check if cache exists
        if cache_path.exists() and not args.force:
            logger.info(f"Cache already exists: {cache_path}")
            logger.info("Use --force to rebuild")
            
            # Load to get count
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            logger.info(f"  Cached samples: {len(samples):,}")
            total_samples += len(samples)
            continue
        
        split_start = time.time()
        
        # Build samples using parallel processing
        samples = parallel_build_cache(
            mimic_cxr_path=mimic_cxr_path,
            mimic_qa_path=mimic_qa_path,
            split=split,
            quality_grade=quality_grade,
            question_types=None,
            num_workers=num_workers,
            cache_dir=args.cache_dir,
        )
        
        # Save to cache
        logger.info(f"Saving {len(samples):,} samples to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        split_time = time.time() - split_start
        total_samples += len(samples)
        
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"  Split '{split}': {len(samples):,} samples")
        logger.info(f"  Cache size: {cache_size_mb:.1f} MB")
        logger.info(f"  Time: {split_time:.1f}s ({split_time/60:.1f} minutes)")
    
    total_time = time.time() - start_time
    
    logger.info(f"\n{'='*70}")
    logger.info("  CACHE PRE-BUILD COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Total time:    {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"  Cache dir:     {args.cache_dir}")
    logger.info("")
    logger.info("  Now run distributed training:")
    logger.info("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
    logger.info("    --config configs/pretrain_config.yaml \\")
    logger.info("    --deepspeed_config configs/deepspeed_config.json")
    logger.info("=" * 70)


if __name__ == '__main__':
    # Use fork method for multiprocessing (more efficient on Linux)
    mp.set_start_method('fork', force=True)
    main()
