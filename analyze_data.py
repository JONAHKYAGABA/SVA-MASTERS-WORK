#!/usr/bin/env python3
"""
MIMIC-CXR VQA Data Analysis Script

This script analyzes and validates the datasets BEFORE training.
Run this first to ensure data is properly prepared.

Based on methodology.md requirements:
- Distribution of observation polarities (positive vs negative)
- Representation across anatomical regions
- Stratification of question complexity
- Scene graph quality assessment
- Data readiness checks

Usage:
    python analyze_data.py --mimic_cxr_path /path/to/MIMIC-CXR-JPG --mimic_qa_path /path/to/MIMIC-Ext-CXR-QBA
    python analyze_data.py --config configs/default_config.yaml
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataAnalysisReport:
    """Container for data analysis results."""
    # Dataset statistics
    total_images: int = 0
    total_qa_pairs: int = 0
    total_scene_graphs: int = 0
    
    # Split statistics
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    
    # Quality metrics
    qa_with_scene_graphs: int = 0
    images_found: int = 0
    images_missing: int = 0
    
    # Distribution analysis
    polarity_distribution: Dict[str, int] = field(default_factory=dict)
    region_distribution: Dict[str, int] = field(default_factory=dict)
    question_type_distribution: Dict[str, int] = field(default_factory=dict)
    answer_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Scene graph quality
    avg_observations_per_graph: float = 0.0
    avg_regions_per_observation: float = 0.0
    bbox_coverage: float = 0.0
    
    # Data quality issues
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Readiness
    is_ready: bool = False


class DataAnalyzer:
    """
    Comprehensive data analyzer for MIMIC-CXR VQA datasets.
    
    Performs:
    1. Dataset discovery and validation
    2. Distribution analysis (polarity, regions, question types)
    3. Scene graph quality assessment
    4. Bias detection
    5. Readiness checks
    """
    
    # Question type categories
    BINARY_TYPES = {'is_abnormal', 'is_normal', 'has_finding', 'has_device', 'yes_no'}
    CATEGORY_TYPES = {'describe_finding', 'what_finding', 'what_type', 'identify'}
    REGION_TYPES = {'where_is', 'locate', 'which_region', 'position'}
    SEVERITY_TYPES = {'how_severe', 'severity', 'grade'}
    COMPARISON_TYPES = {'compare', 'change', 'difference', 'progression'}
    
    # Anatomical regions
    ANATOMICAL_REGIONS = {
        'cardiac': ['heart', 'cardiac', 'cardiomegaly', 'cardiomediastinum', 'aorta'],
        'pulmonary': ['lung', 'pulmonary', 'bronchi', 'airway', 'lobe'],
        'pleural': ['pleura', 'pleural', 'effusion', 'pneumothorax'],
        'mediastinal': ['mediastinum', 'mediastinal', 'hilum', 'hilar'],
        'osseous': ['rib', 'spine', 'clavicle', 'bone', 'fracture', 'vertebra'],
    }
    
    def __init__(
        self,
        mimic_cxr_path: str,
        mimic_qa_path: str,
        chexpert_path: Optional[str] = None,
        output_dir: str = './analysis_output'
    ):
        self.mimic_cxr_path = Path(mimic_cxr_path)
        self.mimic_qa_path = Path(mimic_qa_path)
        self.chexpert_path = Path(chexpert_path) if chexpert_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report = DataAnalysisReport()
        
    def run_full_analysis(self) -> DataAnalysisReport:
        """Run complete data analysis pipeline."""
        logger.info("=" * 60)
        logger.info("MIMIC-CXR VQA Data Analysis")
        logger.info("=" * 60)
        
        # Step 1: Validate paths
        logger.info("\n[1/7] Validating dataset paths...")
        if not self._validate_paths():
            logger.error("Path validation failed. Cannot proceed.")
            return self.report
        
        # Step 2: Analyze splits
        logger.info("\n[2/7] Analyzing dataset splits...")
        self._analyze_splits()
        
        # Step 3: Analyze images
        logger.info("\n[3/7] Analyzing images...")
        self._analyze_images()
        
        # Step 4: Analyze QA pairs
        logger.info("\n[4/7] Analyzing QA pairs...")
        self._analyze_qa_pairs()
        
        # Step 5: Analyze scene graphs
        logger.info("\n[5/7] Analyzing scene graphs...")
        self._analyze_scene_graphs()
        
        # Step 6: Detect biases
        logger.info("\n[6/7] Detecting biases...")
        self._detect_biases()
        
        # Step 7: Generate report
        logger.info("\n[7/7] Generating report...")
        self._generate_report()
        
        return self.report
    
    def _validate_paths(self) -> bool:
        """Validate that required paths exist."""
        valid = True
        
        if not self.mimic_cxr_path.exists():
            self.report.issues.append(f"MIMIC-CXR path not found: {self.mimic_cxr_path}")
            logger.error(f"✗ MIMIC-CXR path not found: {self.mimic_cxr_path}")
            valid = False
        else:
            logger.info(f"✓ MIMIC-CXR path found: {self.mimic_cxr_path}")
        
        if not self.mimic_qa_path.exists():
            self.report.issues.append(f"MIMIC-Ext-CXR-QBA path not found: {self.mimic_qa_path}")
            logger.error(f"✗ MIMIC-Ext-CXR-QBA path not found: {self.mimic_qa_path}")
            valid = False
        else:
            logger.info(f"✓ MIMIC-Ext-CXR-QBA path found: {self.mimic_qa_path}")
        
        # Check for key subdirectories
        files_dir = self.mimic_cxr_path / 'files'
        if files_dir.exists():
            logger.info(f"✓ Images directory found: {files_dir}")
        else:
            self.report.warnings.append("Images 'files' directory not found")
            logger.warning(f"⚠ Images 'files' directory not found at {files_dir}")
        
        # Check QA directory - must be extracted from qa.zip
        qa_dir = self.mimic_qa_path / 'qa'
        qa_zip = self.mimic_qa_path / 'qa.zip'
        
        if qa_dir.exists():
            logger.info(f"✓ QA directory found: {qa_dir}")
        elif qa_zip.exists():
            logger.warning(f"⚠ qa.zip found but not extracted. Attempting automatic extraction...")
            if self._extract_zip(qa_zip, self.mimic_qa_path):
                logger.info(f"✓ qa.zip extracted successfully")
            else:
                self.report.issues.append("qa.zip found but extraction failed!")
                logger.error(f"✗ Failed to extract qa.zip")
                logger.error(f"  Please extract manually:")
                logger.error(f"  Linux:   unzip '{qa_zip}' -d '{self.mimic_qa_path}'")
                valid = False
        else:
            self.report.issues.append("QA directory not found (need to extract qa.zip)")
            logger.error(f"✗ QA directory not found at {qa_dir}")
            valid = False
        
        # Check scene_data directory - must be extracted from scene_data.zip
        scene_data_dir = self.mimic_qa_path / 'scene_data'
        scene_data_zip = self.mimic_qa_path / 'scene_data.zip'
        
        if scene_data_dir.exists():
            logger.info(f"✓ Scene data directory found: {scene_data_dir}")
        elif scene_data_zip.exists():
            logger.warning(f"⚠ scene_data.zip found but not extracted. Attempting automatic extraction...")
            if self._extract_zip(scene_data_zip, self.mimic_qa_path):
                logger.info(f"✓ scene_data.zip extracted successfully")
            else:
                self.report.issues.append("scene_data.zip found but extraction failed!")
                logger.error(f"✗ Failed to extract scene_data.zip")
                logger.error(f"  Please extract manually:")
                logger.error(f"  Linux:   unzip '{scene_data_zip}' -d '{self.mimic_qa_path}'")
                valid = False
        else:
            self.report.warnings.append("Scene data directory not found")
            logger.warning(f"⚠ Scene data directory not found at {scene_data_dir}")
        
        return valid
    
    def _extract_zip(self, zip_path: Path, dest_path: Path) -> bool:
        """Extract a ZIP file automatically."""
        import zipfile
        import subprocess
        
        logger.info(f"  Extracting {zip_path.name}... (this may take several minutes)")
        
        # Try system unzip first (faster for large files)
        try:
            result = subprocess.run(
                ['unzip', '-o', '-q', str(zip_path), '-d', str(dest_path)],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Fall back to Python zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest_path)
            return True
        except Exception as e:
            logger.error(f"  Extraction error: {e}")
            return False
    
    def _analyze_splits(self):
        """Analyze train/val/test splits."""
        split_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
        if not split_file.exists():
            split_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
        
        if split_file.exists():
            try:
                if str(split_file).endswith('.gz'):
                    splits_df = pd.read_csv(split_file, compression='gzip')
                else:
                    splits_df = pd.read_csv(split_file)
                
                split_counts = splits_df['split'].value_counts()
                self.report.train_samples = split_counts.get('train', 0)
                self.report.val_samples = split_counts.get('validate', 0)
                self.report.test_samples = split_counts.get('test', 0)
                
                logger.info(f"  Train: {self.report.train_samples:,} studies")
                logger.info(f"  Val:   {self.report.val_samples:,} studies")
                logger.info(f"  Test:  {self.report.test_samples:,} studies")
                
            except Exception as e:
                self.report.warnings.append(f"Could not parse split file: {e}")
                logger.warning(f"⚠ Could not parse split file: {e}")
        else:
            self.report.warnings.append("Split file not found")
            logger.warning("⚠ Split file not found")
    
    def _analyze_images(self):
        """Analyze image availability and statistics."""
        files_dir = self.mimic_cxr_path / 'files'
        if not files_dir.exists():
            return
        
        image_count = 0
        for p_group in files_dir.iterdir():
            if p_group.is_dir() and p_group.name.startswith('p'):
                for patient_dir in p_group.iterdir():
                    if patient_dir.is_dir():
                        for study_dir in patient_dir.iterdir():
                            if study_dir.is_dir():
                                images = list(study_dir.glob('*.jpg'))
                                image_count += len(images)
        
        self.report.total_images = image_count
        logger.info(f"  Total images found: {image_count:,}")
    
    def _analyze_qa_pairs(self):
        """Analyze QA pair statistics and distributions."""
        qa_dir = self.mimic_qa_path / 'qa'
        if not qa_dir.exists():
            logger.warning("QA directory not found, skipping QA analysis")
            return
        
        total_qa = 0
        question_types = Counter()
        answer_types = Counter()
        polarity = Counter()
        regions = Counter()
        
        # Sample QA files
        qa_files = list(qa_dir.rglob('*.qa.json'))[:10000]  # Sample for speed
        
        logger.info(f"  Sampling {len(qa_files)} QA files...")
        
        for qa_file in qa_files:
            try:
                with open(qa_file) as f:
                    qa_data = json.load(f)
                
                questions = qa_data.get('questions', [])
                total_qa += len(questions)
                
                for q in questions:
                    # Question type
                    q_type = q.get('question_type', 'unknown')
                    question_types[q_type] += 1
                    
                    # Categorize question type
                    q_type_lower = q_type.lower()
                    if any(t in q_type_lower for t in ['is_', 'has_', 'yes', 'no']):
                        answer_types['binary'] += 1
                    elif any(t in q_type_lower for t in ['where', 'locate', 'position']):
                        answer_types['region'] += 1
                    elif any(t in q_type_lower for t in ['severe', 'grade']):
                        answer_types['severity'] += 1
                    else:
                        answer_types['category'] += 1
                    
                    # Polarity (positive/negative findings)
                    answers = q.get('answers', [])
                    if answers:
                        answer_text = answers[0].get('text', '').lower()
                        if 'yes' in answer_text or 'present' in answer_text or 'positive' in answer_text:
                            polarity['positive'] += 1
                        elif 'no' in answer_text or 'absent' in answer_text or 'negative' in answer_text:
                            polarity['negative'] += 1
                        else:
                            polarity['neutral'] += 1
                    
                    # Anatomical regions
                    question_text = q.get('question', '').lower()
                    for region_cat, keywords in self.ANATOMICAL_REGIONS.items():
                        if any(kw in question_text for kw in keywords):
                            regions[region_cat] += 1
                            break
                    else:
                        regions['other'] += 1
                        
            except Exception as e:
                continue
        
        # Scale up estimates based on sampling
        scale_factor = len(list(qa_dir.rglob('*.qa.json'))) / max(len(qa_files), 1)
        
        self.report.total_qa_pairs = int(total_qa * scale_factor)
        self.report.question_type_distribution = dict(question_types.most_common(20))
        self.report.answer_type_distribution = dict(answer_types)
        self.report.polarity_distribution = dict(polarity)
        self.report.region_distribution = dict(regions)
        
        logger.info(f"  Estimated total QA pairs: {self.report.total_qa_pairs:,}")
        logger.info(f"  Question types found: {len(question_types)}")
        logger.info(f"  Top 5 question types: {question_types.most_common(5)}")
    
    def _analyze_scene_graphs(self):
        """Analyze scene graph quality and statistics."""
        # Try multiple possible scene graph locations
        sg_dirs = [
            self.mimic_qa_path / 'scene_graphs',
            self.mimic_qa_path / 'scene_data',
            self.mimic_qa_path / 'graphs',
        ]
        
        sg_dir = None
        for d in sg_dirs:
            if d.exists():
                sg_dir = d
                break
        
        if sg_dir is None:
            logger.warning("Scene graph directory not found")
            self.report.warnings.append("Scene graph directory not found")
            return
        
        # Sample scene graphs
        sg_files = list(sg_dir.rglob('*.json'))[:5000]
        
        if not sg_files:
            logger.warning("No scene graph files found")
            return
        
        logger.info(f"  Sampling {len(sg_files)} scene graph files...")
        
        total_observations = 0
        total_regions = 0
        total_bboxes = 0
        graphs_analyzed = 0
        
        for sg_file in sg_files:
            try:
                with open(sg_file) as f:
                    sg = json.load(f)
                
                observations = sg.get('observations', {})
                num_obs = len(observations)
                total_observations += num_obs
                
                for obs_id, obs in observations.items():
                    regions = obs.get('regions', [])
                    total_regions += len(regions)
                    
                    if 'localization' in obs and obs['localization']:
                        total_bboxes += 1
                
                graphs_analyzed += 1
                
            except Exception as e:
                continue
        
        if graphs_analyzed > 0:
            self.report.total_scene_graphs = int(len(list(sg_dir.rglob('*.json'))))
            self.report.avg_observations_per_graph = total_observations / graphs_analyzed
            self.report.avg_regions_per_observation = total_regions / max(total_observations, 1)
            self.report.bbox_coverage = total_bboxes / max(total_observations, 1)
            
            logger.info(f"  Total scene graphs: {self.report.total_scene_graphs:,}")
            logger.info(f"  Avg observations per graph: {self.report.avg_observations_per_graph:.2f}")
            logger.info(f"  Avg regions per observation: {self.report.avg_regions_per_observation:.2f}")
            logger.info(f"  BBox coverage: {self.report.bbox_coverage*100:.1f}%")
    
    def _detect_biases(self):
        """Detect potential biases in the dataset."""
        logger.info("  Checking for distribution biases...")
        
        # Check polarity balance
        if self.report.polarity_distribution:
            pos = self.report.polarity_distribution.get('positive', 0)
            neg = self.report.polarity_distribution.get('negative', 0)
            total = pos + neg
            
            if total > 0:
                pos_ratio = pos / total
                if pos_ratio < 0.3 or pos_ratio > 0.7:
                    self.report.warnings.append(
                        f"Polarity imbalance detected: {pos_ratio*100:.1f}% positive. "
                        "Consider stratified sampling for training."
                    )
                    logger.warning(f"  ⚠ Polarity imbalance: {pos_ratio*100:.1f}% positive")
                else:
                    logger.info(f"  ✓ Polarity balance OK: {pos_ratio*100:.1f}% positive")
        
        # Check region coverage
        if self.report.region_distribution:
            regions = self.report.region_distribution
            total = sum(regions.values())
            
            for region, count in regions.items():
                ratio = count / total if total > 0 else 0
                if ratio < 0.05:
                    self.report.warnings.append(
                        f"Low representation for {region} region ({ratio*100:.1f}%). "
                        "Consider augmentation or oversampling."
                    )
                    logger.warning(f"  ⚠ Low {region} representation: {ratio*100:.1f}%")
        
        # Check answer type balance
        if self.report.answer_type_distribution:
            types = self.report.answer_type_distribution
            total = sum(types.values())
            
            if total > 0:
                binary_ratio = types.get('binary', 0) / total
                if binary_ratio > 0.8:
                    self.report.warnings.append(
                        f"Dataset heavily skewed toward binary questions ({binary_ratio*100:.1f}%). "
                        "May need to balance training batches."
                    )
                    logger.warning(f"  ⚠ Binary question dominance: {binary_ratio*100:.1f}%")
    
    def _generate_report(self):
        """Generate final analysis report and determine readiness."""
        # Determine readiness
        critical_issues = len(self.report.issues)
        
        self.report.is_ready = (
            critical_issues == 0 and
            self.report.total_qa_pairs > 0 and
            self.report.total_images > 0
        )
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"\n📊 Dataset Statistics:")
        logger.info(f"   Images:       {self.report.total_images:,}")
        logger.info(f"   QA Pairs:     {self.report.total_qa_pairs:,}")
        logger.info(f"   Scene Graphs: {self.report.total_scene_graphs:,}")
        
        logger.info(f"\n📁 Split Distribution:")
        logger.info(f"   Train: {self.report.train_samples:,}")
        logger.info(f"   Val:   {self.report.val_samples:,}")
        logger.info(f"   Test:  {self.report.test_samples:,}")
        
        if self.report.question_type_distribution:
            logger.info(f"\n❓ Top Question Types:")
            for q_type, count in list(self.report.question_type_distribution.items())[:5]:
                logger.info(f"   {q_type}: {count:,}")
        
        if self.report.polarity_distribution:
            logger.info(f"\n⚖️ Polarity Distribution:")
            for pol, count in self.report.polarity_distribution.items():
                logger.info(f"   {pol}: {count:,}")
        
        if self.report.region_distribution:
            logger.info(f"\n🫁 Anatomical Region Distribution:")
            for region, count in self.report.region_distribution.items():
                logger.info(f"   {region}: {count:,}")
        
        if self.report.issues:
            logger.error(f"\n❌ Critical Issues ({len(self.report.issues)}):")
            for issue in self.report.issues:
                logger.error(f"   • {issue}")
        
        if self.report.warnings:
            logger.warning(f"\n⚠️ Warnings ({len(self.report.warnings)}):")
            for warning in self.report.warnings:
                logger.warning(f"   • {warning}")
        
        logger.info("\n" + "=" * 60)
        if self.report.is_ready:
            logger.info("✅ DATA IS READY FOR TRAINING")
            logger.info("   Run: python train_mimic_cxr.py --config configs/default_config.yaml")
        else:
            logger.error("❌ DATA NOT READY - Please resolve issues above")
        logger.info("=" * 60)
        
        # Save report to file
        self._save_report()
        
        # Generate plots if available
        if PLOTTING_AVAILABLE:
            self._generate_plots()
    
    def _save_report(self):
        """Save analysis report to JSON."""
        report_path = self.output_dir / 'analysis_report.json'
        
        report_dict = {
            'summary': {
                'is_ready': self.report.is_ready,
                'total_images': self.report.total_images,
                'total_qa_pairs': self.report.total_qa_pairs,
                'total_scene_graphs': self.report.total_scene_graphs,
            },
            'splits': {
                'train': self.report.train_samples,
                'val': self.report.val_samples,
                'test': self.report.test_samples,
            },
            'distributions': {
                'question_types': self.report.question_type_distribution,
                'answer_types': self.report.answer_type_distribution,
                'polarity': self.report.polarity_distribution,
                'regions': self.report.region_distribution,
            },
            'scene_graph_quality': {
                'avg_observations': self.report.avg_observations_per_graph,
                'avg_regions': self.report.avg_regions_per_observation,
                'bbox_coverage': self.report.bbox_coverage,
            },
            'issues': self.report.issues,
            'warnings': self.report.warnings,
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"\n📄 Report saved to: {report_path}")
    
    def _generate_plots(self):
        """Generate visualization plots."""
        if not PLOTTING_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Question type distribution
        if self.report.question_type_distribution:
            ax = axes[0, 0]
            types = list(self.report.question_type_distribution.keys())[:10]
            counts = [self.report.question_type_distribution[t] for t in types]
            ax.barh(types, counts, color='steelblue')
            ax.set_xlabel('Count')
            ax.set_title('Top Question Types')
        
        # Polarity distribution
        if self.report.polarity_distribution:
            ax = axes[0, 1]
            labels = list(self.report.polarity_distribution.keys())
            sizes = list(self.report.polarity_distribution.values())
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
            ax.set_title('Polarity Distribution')
        
        # Region distribution
        if self.report.region_distribution:
            ax = axes[1, 0]
            regions = list(self.report.region_distribution.keys())
            counts = list(self.report.region_distribution.values())
            ax.bar(regions, counts, color='coral')
            ax.set_xlabel('Anatomical Region')
            ax.set_ylabel('Count')
            ax.set_title('Anatomical Region Distribution')
            ax.tick_params(axis='x', rotation=45)
        
        # Answer type distribution
        if self.report.answer_type_distribution:
            ax = axes[1, 1]
            types = list(self.report.answer_type_distribution.keys())
            counts = list(self.report.answer_type_distribution.values())
            ax.bar(types, counts, color='mediumpurple')
            ax.set_xlabel('Answer Type')
            ax.set_ylabel('Count')
            ax.set_title('Answer Type Distribution')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'distribution_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Plots saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze MIMIC-CXR VQA Data')
    
    parser.add_argument('--mimic_cxr_path', type=str, required=True,
                       help='Path to MIMIC-CXR-JPG dataset')
    parser.add_argument('--mimic_qa_path', type=str, required=True,
                       help='Path to MIMIC-Ext-CXR-QBA dataset')
    parser.add_argument('--chexpert_path', type=str, default=None,
                       help='Path to CheXpert labels (optional)')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                       help='Directory to save analysis results')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (alternative to individual paths)')
    
    args = parser.parse_args()
    
    # Load paths from config if provided
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                config = yaml.safe_load(f)
            
            args.mimic_cxr_path = config.get('data', {}).get('mimic_cxr_jpg_path', args.mimic_cxr_path)
            args.mimic_qa_path = config.get('data', {}).get('mimic_ext_cxr_qba_path', args.mimic_qa_path)
            args.chexpert_path = config.get('data', {}).get('chexpert_labels_path', args.chexpert_path)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    # Run analysis
    analyzer = DataAnalyzer(
        mimic_cxr_path=args.mimic_cxr_path,
        mimic_qa_path=args.mimic_qa_path,
        chexpert_path=args.chexpert_path,
        output_dir=args.output_dir
    )
    
    report = analyzer.run_full_analysis()
    
    # Return exit code based on readiness
    sys.exit(0 if report.is_ready else 1)


if __name__ == '__main__':
    main()

