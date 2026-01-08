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
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        '--force',
        action='store_true',
        help='Force rebuild even if cache exists'
    )
    
    args = parser.parse_args()
    
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
    
    if config:
        if not mimic_cxr_path:
            mimic_cxr_path = config.data.mimic_cxr_jpg_path
        if not mimic_qa_path:
            mimic_qa_path = config.data.mimic_ext_cxr_qba_path
        if not args.quality_grade:
            args.quality_grade = config.data.quality_grade or 'all'
    
    if not mimic_cxr_path or not mimic_qa_path:
        logger.error("Please provide --mimic_cxr_path and --mimic_qa_path or a valid config")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("DATASET CACHE PRE-BUILDER")
    logger.info("=" * 60)
    logger.info(f"  MIMIC-CXR-JPG:     {mimic_cxr_path}")
    logger.info(f"  MIMIC-Ext-CXR-QBA: {mimic_qa_path}")
    logger.info(f"  Splits:            {args.splits}")
    logger.info(f"  Quality grade:     {args.quality_grade}")
    logger.info(f"  Cache directory:   {args.cache_dir}")
    logger.info(f"  Force rebuild:     {args.force}")
    logger.info("=" * 60)
    
    # Import dataset
    from data.mimic_cxr_dataset import MIMICCXRVQADataset
    
    total_samples = 0
    start_time = time.time()
    
    for split in args.splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Building cache for split: {split.upper()}")
        logger.info(f"{'='*60}")
        
        split_start = time.time()
        
        # Create dataset (this will build and cache samples)
        dataset = MIMICCXRVQADataset(
            mimic_cxr_path=mimic_cxr_path,
            mimic_qa_path=mimic_qa_path,
            split=split,
            quality_grade=args.quality_grade,
            cache_dir=args.cache_dir,
            use_cache=True,
            force_rebuild_cache=args.force,
        )
        
        split_time = time.time() - split_start
        total_samples += len(dataset)
        
        logger.info(f"  Split '{split}': {len(dataset):,} samples")
        logger.info(f"  Time: {split_time:.1f}s ({split_time/60:.1f} minutes)")
    
    total_time = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info("CACHE PRE-BUILD COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Total time:    {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"  Cache dir:     {args.cache_dir}")
    logger.info("")
    logger.info("You can now run distributed training!")
    logger.info("The cache will be loaded instantly on all GPUs.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

