#!/usr/bin/env python3
"""
Automatic Data Setup Script for MIMIC-CXR VQA

This script:
1. Validates dataset paths
2. Automatically extracts qa.zip and scene_data.zip
3. Verifies the extracted structure
4. Creates symlinks if needed

Usage:
    python setup_data.py --mimic_cxr_path /path/to/mimic-cxr-jpg --mimic_qa_path /path/to/mimic-ext-cxr-qba
    python setup_data.py --config configs/default_config.yaml
"""

import os
import sys
import zipfile
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSetup:
    """Automatic data setup for MIMIC-CXR VQA training."""
    
    def __init__(
        self,
        mimic_cxr_path: str,
        mimic_qa_path: str,
        use_exports: bool = False,
        export_grade: str = 'A_frontal'
    ):
        self.mimic_cxr_path = Path(mimic_cxr_path)
        self.mimic_qa_path = Path(mimic_qa_path)
        self.use_exports = use_exports
        self.export_grade = export_grade
        
    def run(self) -> bool:
        """Run complete data setup."""
        logger.info("=" * 60)
        logger.info("MIMIC-CXR VQA Data Setup")
        logger.info("=" * 60)
        
        # Step 1: Validate paths
        logger.info("\n[1/5] Validating paths...")
        if not self._validate_paths():
            return False
        
        # Step 2: Extract ZIP files
        logger.info("\n[2/5] Extracting ZIP files...")
        if not self._extract_zips():
            return False
        
        # Step 3: Verify structure
        logger.info("\n[3/5] Verifying extracted structure...")
        if not self._verify_structure():
            return False
        
        # Step 4: Check MIMIC-CXR files
        logger.info("\n[4/5] Checking MIMIC-CXR-JPG files...")
        self._check_mimic_cxr()
        
        # Step 5: Summary
        logger.info("\n[5/5] Setup complete!")
        self._print_summary()
        
        return True
    
    def _validate_paths(self) -> bool:
        """Validate that base paths exist."""
        valid = True
        
        if not self.mimic_cxr_path.exists():
            logger.error(f"✗ MIMIC-CXR-JPG path not found: {self.mimic_cxr_path}")
            valid = False
        else:
            logger.info(f"✓ MIMIC-CXR-JPG path: {self.mimic_cxr_path}")
        
        if not self.mimic_qa_path.exists():
            logger.error(f"✗ MIMIC-Ext-CXR-QBA path not found: {self.mimic_qa_path}")
            valid = False
        else:
            logger.info(f"✓ MIMIC-Ext-CXR-QBA path: {self.mimic_qa_path}")
        
        return valid
    
    def _extract_zips(self) -> bool:
        """Extract qa.zip and scene_data.zip if not already extracted."""
        success = True
        
        # Define ZIP files and their target directories
        zip_files = [
            ('qa.zip', 'qa'),
            ('scene_data.zip', 'scene_data'),
        ]
        
        for zip_name, target_dir in zip_files:
            zip_path = self.mimic_qa_path / zip_name
            target_path = self.mimic_qa_path / target_dir
            
            # Check if already extracted
            if target_path.exists() and any(target_path.iterdir()):
                logger.info(f"✓ {target_dir}/ already exists, skipping extraction")
                continue
            
            # Check if ZIP exists
            if not zip_path.exists():
                logger.warning(f"⚠ {zip_name} not found at {zip_path}")
                continue
            
            # Extract
            logger.info(f"  Extracting {zip_name} ({self._get_file_size(zip_path)})...")
            logger.info(f"  This may take several minutes...")
            
            try:
                # Try Python zipfile first
                if self._extract_with_python(zip_path, self.mimic_qa_path):
                    logger.info(f"✓ Extracted {zip_name} successfully")
                else:
                    # Fall back to system unzip (faster for large files)
                    if self._extract_with_unzip(zip_path, self.mimic_qa_path):
                        logger.info(f"✓ Extracted {zip_name} successfully")
                    else:
                        logger.error(f"✗ Failed to extract {zip_name}")
                        success = False
            except Exception as e:
                logger.error(f"✗ Error extracting {zip_name}: {e}")
                success = False
        
        return success
    
    def _extract_with_python(self, zip_path: Path, dest_path: Path) -> bool:
        """Extract ZIP using Python zipfile module."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Get total size for progress
                total_size = sum(info.file_size for info in zf.infolist())
                extracted_size = 0
                
                for info in zf.infolist():
                    zf.extract(info, dest_path)
                    extracted_size += info.file_size
                    
                    # Progress indicator every 10%
                    progress = (extracted_size / total_size) * 100
                    if extracted_size % (total_size // 10 + 1) < info.file_size:
                        logger.info(f"    Progress: {progress:.0f}%")
            
            return True
        except Exception as e:
            logger.warning(f"Python extraction failed: {e}, trying system unzip...")
            return False
    
    def _extract_with_unzip(self, zip_path: Path, dest_path: Path) -> bool:
        """Extract ZIP using system unzip command (Linux)."""
        try:
            result = subprocess.run(
                ['unzip', '-o', '-q', str(zip_path), '-d', str(dest_path)],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("unzip command not found, install with: sudo apt install unzip")
            return False
        except Exception as e:
            logger.warning(f"System unzip failed: {e}")
            return False
    
    def _verify_structure(self) -> bool:
        """Verify the extracted directory structure."""
        issues = []
        
        # Check QA directory
        qa_dir = self.mimic_qa_path / 'qa'
        if qa_dir.exists():
            # Check for patient directories
            p_dirs = list(qa_dir.glob('p*'))
            if p_dirs:
                logger.info(f"✓ QA directory: {len(p_dirs)} patient groups found")
                
                # Sample check
                sample_qa = list(qa_dir.rglob('*.qa.json'))[:1]
                if sample_qa:
                    logger.info(f"  Sample QA file: {sample_qa[0].name}")
            else:
                issues.append("QA directory exists but no patient folders found")
        else:
            issues.append("QA directory not found after extraction")
        
        # Check scene_data directory
        sg_dir = self.mimic_qa_path / 'scene_data'
        if sg_dir.exists():
            p_dirs = list(sg_dir.glob('p*'))
            if p_dirs:
                logger.info(f"✓ Scene data directory: {len(p_dirs)} patient groups found")
                
                # Sample check
                sample_sg = list(sg_dir.rglob('*.scene_graph.json'))[:1]
                if sample_sg:
                    logger.info(f"  Sample scene graph: {sample_sg[0].name}")
            else:
                issues.append("Scene data directory exists but no patient folders found")
        else:
            issues.append("Scene data directory not found after extraction")
        
        # Check metadata
        metadata_dir = self.mimic_qa_path / 'metadata'
        if metadata_dir.exists():
            dataset_info = metadata_dir / 'dataset_info.json'
            if dataset_info.exists():
                logger.info(f"✓ dataset_info.json found in metadata/")
            else:
                # Check root level
                root_info = self.mimic_qa_path / 'dataset_info.json'
                if root_info.exists():
                    logger.info(f"✓ dataset_info.json found in root")
        
        if issues:
            for issue in issues:
                logger.error(f"✗ {issue}")
            return False
        
        return True
    
    def _check_mimic_cxr(self):
        """Check MIMIC-CXR-JPG structure and files."""
        # Check files directory
        files_dir = self.mimic_cxr_path / 'files'
        if files_dir.exists():
            p_dirs = list(files_dir.glob('p*'))
            logger.info(f"✓ Images directory: {len(p_dirs)} patient groups")
        else:
            logger.warning("⚠ files/ directory not found in MIMIC-CXR-JPG")
        
        # Check metadata files
        metadata_files = [
            'mimic-cxr-2.0.0-split.csv.gz',
            'mimic-cxr-2.0.0-chexpert.csv.gz',
            'mimic-cxr-2.0.0-metadata.csv.gz',
        ]
        
        for mf in metadata_files:
            mf_path = self.mimic_cxr_path / mf
            if mf_path.exists():
                logger.info(f"✓ {mf} ({self._get_file_size(mf_path)})")
            else:
                # Try without .gz
                mf_path_nogz = self.mimic_cxr_path / mf.replace('.gz', '')
                if mf_path_nogz.exists():
                    logger.info(f"✓ {mf.replace('.gz', '')} ({self._get_file_size(mf_path_nogz)})")
                else:
                    logger.warning(f"⚠ {mf} not found")
    
    def _get_file_size(self, path: Path) -> str:
        """Get human-readable file size."""
        size = path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _print_summary(self):
        """Print setup summary and next steps."""
        logger.info("\n" + "=" * 60)
        logger.info("SETUP COMPLETE ✓")
        logger.info("=" * 60)
        
        logger.info(f"\nDataset Locations:")
        logger.info(f"  MIMIC-CXR-JPG:    {self.mimic_cxr_path}")
        logger.info(f"  MIMIC-Ext-CXR-QBA: {self.mimic_qa_path}")
        
        logger.info(f"\nNext Steps:")
        logger.info(f"  1. Update configs/default_config.yaml with these paths")
        logger.info(f"  2. Run: python analyze_data.py --mimic_cxr_path {self.mimic_cxr_path} --mimic_qa_path {self.mimic_qa_path}")
        logger.info(f"  3. Run: python train_mimic_cxr.py --config configs/default_config.yaml")


def main():
    parser = argparse.ArgumentParser(description='Setup MIMIC-CXR VQA Data')
    
    parser.add_argument('--mimic_cxr_path', type=str, required=True,
                       help='Path to MIMIC-CXR-JPG dataset')
    parser.add_argument('--mimic_qa_path', type=str, required=True,
                       help='Path to MIMIC-Ext-CXR-QBA dataset')
    parser.add_argument('--use_exports', action='store_true',
                       help='Use pre-filtered exports folder')
    parser.add_argument('--export_grade', type=str, default='A_frontal',
                       choices=['A_frontal', 'B_frontal'],
                       help='Export grade to use (A_frontal for fine-tuning, B_frontal for pre-training)')
    parser.add_argument('--config', type=str, default=None,
                       help='Load paths from config file')
    
    args = parser.parse_args()
    
    # Load from config if provided
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                config = yaml.safe_load(f)
            
            args.mimic_cxr_path = config.get('data', {}).get('mimic_cxr_jpg_path', args.mimic_cxr_path)
            args.mimic_qa_path = config.get('data', {}).get('mimic_ext_cxr_qba_path', args.mimic_qa_path)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    # Run setup
    setup = DataSetup(
        mimic_cxr_path=args.mimic_cxr_path,
        mimic_qa_path=args.mimic_qa_path,
        use_exports=args.use_exports,
        export_grade=args.export_grade
    )
    
    success = setup.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

