#!/usr/bin/env python3
"""
Pre-build dataset cache using MapReduce for parallel processing.

MapReduce approach:
- MAP: Each worker processes a chunk of QA files independently
- REDUCE: Main process concatenates all sample lists

Run:
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --num_workers 24
"""

import os
import sys
import argparse
import json
import pickle
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# MAPREDUCE FUNCTIONS (Top-level for pickling)
# =============================================================================

def _map_qa_file(args_tuple):
    """
    MAP function: Process a single QA file and return samples.
    
    Args:
        args_tuple: (qa_file_path, valid_studies_set, mimic_cxr_path, sg_dir_path)
    
    Returns:
        List of sample dicts, or empty list on failure
    """
    qa_file_str, valid_studies_frozen, mimic_cxr_str, sg_dir_str = args_tuple
    
    try:
        qa_file = Path(qa_file_str)
        mimic_cxr_path = Path(mimic_cxr_str)
        sg_dir = Path(sg_dir_str) if sg_dir_str else None
        
        # Parse IDs from path: .../p10/p10000032/s50414267.qa.json
        subject_id = int(qa_file.parent.name[1:])  # p10000032 -> 10000032
        study_id = int(qa_file.stem.split('.')[0][1:])  # s50414267.qa -> 50414267
        
        # Check if in valid split
        if (subject_id, study_id) not in valid_studies_frozen:
            return []
        
        # Find image
        p_prefix = f"p{str(subject_id)[:2]}"
        study_dir = mimic_cxr_path / 'files' / p_prefix / f"p{subject_id}" / f"s{study_id}"
        
        if not study_dir.exists():
            return []
        
        jpg_files = list(study_dir.glob('*.jpg'))
        if not jpg_files:
            return []
        
        img_path = str(jpg_files[0])
        dicom_id = jpg_files[0].stem
        
        # Find scene graph
        sg_path = None
        if sg_dir:
            sg_file = sg_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
            if sg_file.exists():
                sg_path = str(sg_file)
        
        # Load QA data
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        
        # Build samples for each question
        samples = []
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
        
        return samples
    
    except Exception:
        return []


def _map_qa_chunk(chunk_args):
    """
    MAP function for a CHUNK of QA files.
    Processes multiple files and returns combined samples.
    This reduces IPC overhead significantly.
    
    Args:
        chunk_args: (list_of_qa_file_paths, valid_studies_frozen, mimic_cxr_path, sg_dir_path)
    
    Returns:
        dict with 'samples' list and stats
    """
    file_paths, valid_studies_frozen, mimic_cxr_str, sg_dir_str = chunk_args
    
    all_samples = []
    files_processed = 0
    files_skipped_split = 0
    files_skipped_img = 0
    
    for qa_file_str in file_paths:
        result = _map_qa_file((qa_file_str, valid_studies_frozen, mimic_cxr_str, sg_dir_str))
        
        if result:
            all_samples.extend(result)
            files_processed += 1
        else:
            # Determine skip reason (simplified - count as split skip if file was readable)
            try:
                qa_file = Path(qa_file_str)
                subject_id = int(qa_file.parent.name[1:])
                study_id = int(qa_file.stem.split('.')[0][1:])
                if (subject_id, study_id) not in valid_studies_frozen:
                    files_skipped_split += 1
                else:
                    files_skipped_img += 1
            except:
                files_skipped_img += 1
    
    return {
        'samples': all_samples,
        'files_processed': files_processed,
        'files_skipped_split': files_skipped_split,
        'files_skipped_img': files_skipped_img,
    }


def _chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pre-build dataset cache with MapReduce')
    parser.add_argument('--config', default='configs/pretrain_config.yaml')
    parser.add_argument('--mimic_cxr_path', type=str)
    parser.add_argument('--mimic_qa_path', type=str)
    parser.add_argument('--split', default='train')
    parser.add_argument('--cache_dir', default='.cache/dataset_samples')
    parser.add_argument('--num_workers', type=int, default=None, 
                        help='Number of worker processes (default: CPU count - 2)')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='Files per chunk (default: 500)')
    parser.add_argument('--force', action='store_true', help='Rebuild even if cache exists')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MAPREDUCE CACHE BUILDER")
    print("=" * 70)
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        try:
            from configs.mimic_cxr_config import load_config_from_file
            config = load_config_from_file(args.config)
            print(f"  Config: {args.config}")
        except Exception as e:
            print(f"  Warning: Could not load config: {e}")
    
    # Get paths
    mimic_cxr_path = Path(args.mimic_cxr_path or (config.data.mimic_cxr_jpg_path if config else None))
    mimic_qa_path = Path(args.mimic_qa_path or (config.data.mimic_ext_cxr_qba_path if config else None))
    
    if not mimic_cxr_path or not mimic_qa_path:
        print("ERROR: Must provide --mimic_cxr_path and --mimic_qa_path or valid config")
        sys.exit(1)
    
    print(f"  CXR: {mimic_cxr_path}")
    print(f"  QA:  {mimic_qa_path}")
    
    # Cache path
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{mimic_cxr_path}|{mimic_qa_path}|{args.split}".encode()).hexdigest()[:12]
    cache_path = cache_dir / f"samples_{args.split}_{cache_key}.pkl"
    
    print(f"  Cache: {cache_path}")
    
    if cache_path.exists() and not args.force:
        with open(cache_path, 'rb') as f:
            samples = pickle.load(f)
        print(f"\n✓ Cache exists: {len(samples):,} samples")
        print(f"  Use --force to rebuild")
        return
    
    # Determine workers
    num_workers = args.num_workers or max(1, cpu_count() - 2)
    print(f"  Workers: {num_workers}")
    print(f"  Chunk size: {args.chunk_size}")
    
    # =========================================================================
    # STEP 1: Load split info
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Loading split info...")
    print("-" * 70)
    
    import pandas as pd
    split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
    if not split_file.exists():
        split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
    
    if not split_file.exists():
        print(f"ERROR: Split file not found: {split_file}")
        sys.exit(1)
    
    df = pd.read_csv(split_file, compression='gzip' if str(split_file).endswith('.gz') else None)
    split_name = 'validate' if args.split == 'val' else args.split
    df = df[df['split'] == split_name]
    
    # Create frozenset for efficient lookup (hashable for multiprocessing)
    valid_studies = frozenset(zip(df['subject_id'].astype(int), df['study_id'].astype(int)))
    print(f"  {len(valid_studies):,} studies in '{split_name}' split")
    
    # =========================================================================
    # STEP 2: Collect QA file paths
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Collecting QA file paths...")
    print("-" * 70)
    
    qa_dir = mimic_qa_path / 'qa'
    sg_dir = mimic_qa_path / 'scene_data'
    
    if not qa_dir.exists():
        print(f"ERROR: QA directory not found: {qa_dir}")
        sys.exit(1)
    
    # Collect all QA file paths as strings (for pickling)
    qa_files = []
    for p_group in sorted(qa_dir.iterdir()):
        if p_group.is_dir() and p_group.name.startswith('p'):
            for patient_dir in p_group.iterdir():
                if patient_dir.is_dir() and patient_dir.name.startswith('p'):
                    qa_files.extend([str(f) for f in patient_dir.glob('s*.qa.json')])
    
    print(f"  {len(qa_files):,} QA files found")
    
    if not qa_files:
        print("ERROR: No QA files found")
        sys.exit(1)
    
    # =========================================================================
    # STEP 3: MapReduce processing
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: MapReduce processing...")
    print("-" * 70)
    
    # Create chunks
    chunks = _chunk_list(qa_files, args.chunk_size)
    print(f"  Created {len(chunks)} chunks (~{args.chunk_size} files each)")
    
    # Prepare chunk arguments (each chunk gets the same shared data)
    mimic_cxr_str = str(mimic_cxr_path)
    sg_dir_str = str(sg_dir) if sg_dir.exists() else None
    
    chunk_args_list = [
        (chunk, valid_studies, mimic_cxr_str, sg_dir_str)
        for chunk in chunks
    ]
    
    # ============ MAP PHASE ============
    print(f"\n  MAP phase: Processing {len(chunks)} chunks with {num_workers} workers...")
    
    all_samples = []
    total_processed = 0
    total_skipped_split = 0
    total_skipped_img = 0
    
    start_time = time.time()
    
    try:
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for better performance and progress tracking
            for i, result in enumerate(pool.imap_unordered(_map_qa_chunk, chunk_args_list)):
                all_samples.extend(result['samples'])
                total_processed += result['files_processed']
                total_skipped_split += result['files_skipped_split']
                total_skipped_img += result['files_skipped_img']
                
                # Progress update every 10 chunks or at end
                if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
                    
                    print(f"    [{i+1}/{len(chunks)}] "
                          f"Samples: {len(all_samples):,} | "
                          f"Rate: {rate:.1f} chunks/s | "
                          f"ETA: {eta:.0f}s")
    
    except Exception as e:
        print(f"\n  ⚠ Multiprocessing failed: {e}")
        print("  Falling back to sequential processing...")
        
        # Sequential fallback with progress
        from tqdm import tqdm
        
        all_samples = []
        total_processed = 0
        total_skipped_split = 0
        total_skipped_img = 0
        
        for chunk_args in tqdm(chunk_args_list, desc="Processing", unit="chunk"):
            result = _map_qa_chunk(chunk_args)
            all_samples.extend(result['samples'])
            total_processed += result['files_processed']
            total_skipped_split += result['files_skipped_split']
            total_skipped_img += result['files_skipped_img']
    
    elapsed = time.time() - start_time
    
    # ============ REDUCE PHASE (already done - just concatenation) ============
    print(f"\n  REDUCE phase: Combined {len(all_samples):,} samples")
    
    # =========================================================================
    # STEP 4: Summary and save
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Saving cache...")
    print("-" * 70)
    
    print(f"  Total samples: {len(all_samples):,}")
    print(f"  Files processed: {total_processed:,}")
    print(f"  Skipped (wrong split): {total_skipped_split:,}")
    print(f"  Skipped (no image): {total_skipped_img:,}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    if all_samples:
        with open(cache_path, 'wb') as f:
            pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"\n  ✓ Saved: {cache_path}")
        print(f"    Size: {mb:.1f} MB")
    else:
        print("\n  ⚠ No samples to save!")
        sys.exit(1)
    
    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 70)
    print("✓ CACHE BUILD COMPLETE!")
    print("=" * 70)
    print("\nNow run training:")
    print("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
    print("    --config configs/pretrain_config.yaml \\")
    print("    --deepspeed_config configs/deepspeed_config.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
