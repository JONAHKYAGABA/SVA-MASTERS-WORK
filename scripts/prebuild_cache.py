#!/usr/bin/env python3
"""
Pre-build dataset cache - SIMPLE VERSION THAT WORKS.

Just iterates through files with tqdm. No parallelism BS.

Run ONCE:
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml
"""

import os
import sys
import argparse
import json
import pickle
import hashlib
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pretrain_config.yaml')
    parser.add_argument('--mimic_cxr_path', type=str)
    parser.add_argument('--mimic_qa_path', type=str)
    parser.add_argument('--split', default='train')
    parser.add_argument('--cache_dir', default='.cache/dataset_samples')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  SIMPLE CACHE BUILDER (just works)")
    print("=" * 60)
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        try:
            from configs.mimic_cxr_config import load_config_from_file
            config = load_config_from_file(args.config)
        except:
            pass
    
    mimic_cxr_path = Path(args.mimic_cxr_path or config.data.mimic_cxr_jpg_path)
    mimic_qa_path = Path(args.mimic_qa_path or config.data.mimic_ext_cxr_qba_path)
    
    print(f"  CXR: {mimic_cxr_path}")
    print(f"  QA:  {mimic_qa_path}")
    
    # Cache path
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{mimic_cxr_path}|{mimic_qa_path}|{args.split}".encode()).hexdigest()[:12]
    cache_path = cache_dir / f"samples_{args.split}_{cache_key}.pkl"
    
    if cache_path.exists() and not args.force:
        with open(cache_path, 'rb') as f:
            samples = pickle.load(f)
        print(f"\nCache exists: {len(samples):,} samples")
        print(f"Use --force to rebuild")
        return
    
    # Load split info
    print("\nLoading split info...")
    import pandas as pd
    split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
    if not split_file.exists():
        split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
    
    df = pd.read_csv(split_file, compression='gzip' if str(split_file).endswith('.gz') else None)
    split_name = 'validate' if args.split == 'val' else args.split
    df = df[df['split'] == split_name]
    valid_studies = set(zip(df['subject_id'].astype(int), df['study_id'].astype(int)))
    print(f"  {len(valid_studies):,} studies in '{split_name}' split")
    
    # Collect QA files
    print("\nCollecting QA files...")
    qa_dir = mimic_qa_path / 'qa'
    sg_dir = mimic_qa_path / 'scene_data'
    
    qa_files = []
    for p_group in sorted(qa_dir.iterdir()):
        if p_group.is_dir() and p_group.name.startswith('p'):
            for patient_dir in p_group.iterdir():
                if patient_dir.is_dir() and patient_dir.name.startswith('p'):
                    qa_files.extend(list(patient_dir.glob('s*.qa.json')))
    
    print(f"  {len(qa_files):,} QA files found")
    
    # Process with tqdm
    print("\nProcessing...")
    samples = []
    skipped_split = 0
    skipped_img = 0
    
    for qa_file in tqdm(qa_files, desc="Building cache", unit="file"):
        try:
            # Parse IDs from path
            subject_id = int(qa_file.parent.name[1:])
            study_id = int(qa_file.stem.split('.')[0][1:])
            
            # Check split
            if (subject_id, study_id) not in valid_studies:
                skipped_split += 1
                continue
            
            # Find image
            p_prefix = f"p{str(subject_id)[:2]}"
            study_dir = mimic_cxr_path / 'files' / p_prefix / f"p{subject_id}" / f"s{study_id}"
            if not study_dir.exists():
                skipped_img += 1
                continue
            jpg_files = list(study_dir.glob('*.jpg'))
            if not jpg_files:
                skipped_img += 1
                continue
            img_path = str(jpg_files[0])
            dicom_id = jpg_files[0].stem
            
            # Find scene graph
            sg_path = sg_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
            sg_str = str(sg_path) if sg_path.exists() else None
            
            # Load QA
            with open(qa_file) as f:
                qa_data = json.load(f)
            
            for q in qa_data.get('questions', []):
                samples.append({
                    'subject_id': subject_id,
                    'study_id': study_id,
                    'dicom_id': dicom_id,
                    'image_path': img_path,
                    'scene_graph_path': sg_str,
                    'question_id': q.get('question_id', ''),
                    'question_type': q.get('question_type', 'unknown'),
                    'question_strategy': q.get('question_strategy', ''),
                    'question': q.get('question', ''),
                    'answers': q.get('answers', []),
                    'obs_ids': q.get('obs_ids', []),
                })
        except Exception as e:
            continue
    
    print(f"\n  Total samples: {len(samples):,}")
    print(f"  Skipped (wrong split): {skipped_split:,}")
    print(f"  Skipped (no image): {skipped_img:,}")
    
    # Save
    print(f"\nSaving to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {mb:.1f} MB")
    
    print("\n" + "=" * 60)
    print("DONE! Now run training:")
    print("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
    print("    --config configs/pretrain_config.yaml \\")
    print("    --deepspeed_config configs/deepspeed_config.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
