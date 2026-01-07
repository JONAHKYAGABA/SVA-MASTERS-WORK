#!/usr/bin/env python3
"""
MIMIC-CXR VQA Comprehensive Data Analysis Script

This script performs COMPREHENSIVE analysis of BOTH datasets:
1. MIMIC-CXR-JPG: Images, CheXpert labels, metadata
2. MIMIC-Ext-CXR-QBA: Scene graphs, QA pairs, observations

Analysis includes:
- Full metadata column analysis for all CSV files
- CheXpert structured label distribution
- Image metadata (ViewPosition, dimensions, procedures)
- Scene graph structure and quality
- QA pair distribution and complexity
- Cross-dataset correlation
- Patient-level comprehensive profiles
- Visual scene graph representations

Usage:
    python analyze_data.py --mimic_cxr_path /path/to/MIMIC-CXR-JPG --mimic_qa_path /path/to/MIMIC-Ext-CXR-QBA
    python analyze_data.py --config configs/default_config.yaml
"""

import os
import sys
import json
import argparse
import logging
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import warnings
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

import numpy as np
import pandas as pd

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    sns.set_style("whitegrid")
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Optional networkx for scene graph visualization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - CheXpert Labels
# ============================================================================
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Pleural Other',
    'Support Devices', 'No Finding'
]

# MIMIC-CXR-JPG Metadata columns
MIMIC_CXR_METADATA_COLS = [
    'dicom_id', 'PerformedProcedureStepDescription', 'ViewPosition',
    'Rows', 'Columns', 'StudyDate', 'StudyTime',
    'ProcedureCodeSequence_CodeMeaning', 'ViewCodeSequence_CodeMeaning',
    'PatientOrientationCodeSequence_CodeMeaning'
]


@dataclass
class DataAnalysisReport:
    """Container for comprehensive data analysis results."""
    # ===================== MIMIC-CXR-JPG Statistics =====================
    total_images: int = 0
    total_studies: int = 0
    total_patients: int = 0
    
    # Split statistics
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    
    # CheXpert Labels Distribution
    chexpert_label_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {label: {pos: X, neg: Y, uncertain: Z}}
    chexpert_studies_labeled: int = 0
    
    # Image Metadata Analysis
    view_position_distribution: Dict[str, int] = field(default_factory=dict)
    image_dimension_stats: Dict[str, Any] = field(default_factory=dict)
    procedure_distribution: Dict[str, int] = field(default_factory=dict)
    patient_orientation_distribution: Dict[str, int] = field(default_factory=dict)
    
    # ===================== MIMIC-Ext-CXR-QBA Statistics =====================
    total_qa_pairs: int = 0
    total_scene_graphs: int = 0
    total_observations: int = 0
    total_answers: int = 0
    
    # QA Distribution
    question_type_distribution: Dict[str, int] = field(default_factory=dict)
    question_strategy_distribution: Dict[str, int] = field(default_factory=dict)
    answer_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Scene Graph Quality
    avg_observations_per_graph: float = 0.0
    avg_regions_per_observation: float = 0.0
    avg_sentences_per_graph: float = 0.0
    bbox_coverage: float = 0.0
    
    # Observation Analysis
    polarity_distribution: Dict[str, int] = field(default_factory=dict)
    region_distribution: Dict[str, int] = field(default_factory=dict)
    entity_distribution: Dict[str, int] = field(default_factory=dict)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    subcategory_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality Metrics from QBA
    qa_with_scene_graphs: int = 0
    images_found: int = 0
    images_missing: int = 0
    
    # ===================== Metadata CSV Analysis =====================
    metadata_csv_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {filename: {rows, cols, columns, sample_values}}
    
    # ===================== Cross-Dataset Correlation =====================
    matched_studies: int = 0
    unmatched_studies_cxr: int = 0
    unmatched_studies_qba: int = 0
    
    # Data quality issues
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Readiness
    is_ready: bool = False


class DataAnalyzer:
    """
    Comprehensive data analyzer for MIMIC-CXR VQA datasets.
    
    Analyzes BOTH:
    1. MIMIC-CXR-JPG: Images, CheXpert labels, NegBio labels, metadata
    2. MIMIC-Ext-CXR-QBA: Scene graphs, QA pairs, observations, metadata
    
    Creates:
    - Full metadata column analysis
    - Distribution plots for all categories
    - Scene graph visualizations
    - Cross-dataset correlation analysis
    - Comprehensive patient profiles
    """
    
    # Question type categories from MIMIC-Ext-CXR-QBA
    QUESTION_STRATEGIES = ['indication', 'abnormal', 'region_abnormal', 'finding']
    
    # Anatomical regions for keyword matching
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
        output_dir: str = './analysis_output',
        num_workers: Optional[int] = None,
        prefetch_factor: int = 2,
        visualize_samples: int = 10,
    ):
        self.mimic_cxr_path = Path(mimic_cxr_path)
        self.mimic_qa_path = Path(mimic_qa_path)
        self.chexpert_path = Path(chexpert_path) if chexpert_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default to half the CPUs (at least 8) to utilize the machine
        if num_workers is None:
            cpu_count = os.cpu_count() or 8
            num_workers = max(8, cpu_count // 2)
        self.num_workers = max(1, num_workers)
        self.prefetch_factor = max(1, prefetch_factor)
        self.visualize_samples = max(0, visualize_samples)
        
        self.report = DataAnalysisReport()
        
        # Cache for loaded metadata
        self._split_df: Optional[pd.DataFrame] = None
        self._chexpert_df: Optional[pd.DataFrame] = None
        self._metadata_df: Optional[pd.DataFrame] = None
        
    def run_full_analysis(self) -> DataAnalysisReport:
        """Run complete comprehensive data analysis pipeline."""
        logger.info("=" * 70)
        logger.info("  COMPREHENSIVE MIMIC-CXR VQA DATA ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"Workers: {self.num_workers} | Prefetch: {self.prefetch_factor}")
        logger.info(f"MIMIC-CXR-JPG: {self.mimic_cxr_path}")
        logger.info(f"MIMIC-Ext-CXR-QBA: {self.mimic_qa_path}")
        logger.info("=" * 70)
        
        # Step 1: Validate paths
        logger.info("\n" + "━" * 60)
        logger.info("📁 [1/11] Validating dataset paths...")
        logger.info("━" * 60)
        if not self._validate_paths():
            logger.error("Path validation failed. Cannot proceed.")
            return self.report
        
        # Step 2: Analyze MIMIC-CXR-JPG splits
        logger.info("\n" + "━" * 60)
        logger.info("📊 [2/11] Analyzing MIMIC-CXR-JPG splits...")
        logger.info("━" * 60)
        self._analyze_splits()
        
        # Step 3: Analyze CheXpert labels
        logger.info("\n" + "━" * 60)
        logger.info("🏷️ [3/11] Analyzing CheXpert structured labels...")
        logger.info("━" * 60)
        self._analyze_chexpert_labels()
        
        # Step 4: Analyze image metadata
        logger.info("\n" + "━" * 60)
        logger.info("📷 [4/11] Analyzing image metadata...")
        logger.info("━" * 60)
        self._analyze_image_metadata()
        
        # Step 5: Count images
        logger.info("\n" + "━" * 60)
        logger.info("🖼️ [5/11] Counting images in filesystem...")
        logger.info("━" * 60)
        self._analyze_images()
        
        # Step 6: Analyze MIMIC-Ext-CXR-QBA metadata files
        logger.info("\n" + "━" * 60)
        logger.info("📑 [6/11] Analyzing MIMIC-Ext-CXR-QBA metadata CSVs...")
        logger.info("━" * 60)
        self._analyze_qba_metadata_files()
        
        # Step 7: Analyze QA pairs
        logger.info("\n" + "━" * 60)
        logger.info("❓ [7/11] Analyzing QA pairs...")
        logger.info("━" * 60)
        self._analyze_qa_pairs()
        
        # Step 8: Analyze scene graphs
        logger.info("\n" + "━" * 60)
        logger.info("🔗 [8/11] Analyzing scene graphs...")
        logger.info("━" * 60)
        self._analyze_scene_graphs()
        
        # Step 9: Cross-dataset correlation
        logger.info("\n" + "━" * 60)
        logger.info("🔀 [9/11] Cross-dataset correlation analysis...")
        logger.info("━" * 60)
        self._analyze_cross_dataset_correlation()
        
        # Step 10: Detect biases
        logger.info("\n" + "━" * 60)
        logger.info("⚖️ [10/11] Detecting biases and imbalances...")
        logger.info("━" * 60)
        self._detect_biases()
        
        # Step 11: Generate comprehensive report
        logger.info("\n" + "━" * 60)
        logger.info("📝 [11/11] Generating comprehensive report...")
        logger.info("━" * 60)
        self._generate_report()

        # Optional: Visualize samples with scene graphs
        if self.visualize_samples > 0:
            logger.info("\n" + "━" * 60)
            logger.info(f"🎨 [BONUS] Creating {self.visualize_samples} comprehensive patient visualizations...")
            logger.info("━" * 60)
            self._visualize_samples(self.visualize_samples)
            self._visualize_scene_graphs(min(5, self.visualize_samples))
        
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
                
                # Cache for later use
                self._split_df = splits_df
                
                # Count unique patients and studies
                self.report.total_patients = splits_df['subject_id'].nunique()
                self.report.total_studies = splits_df['study_id'].nunique()
                logger.info(f"  Total patients: {self.report.total_patients:,}")
                logger.info(f"  Total studies:  {self.report.total_studies:,}")
                
            except Exception as e:
                self.report.warnings.append(f"Could not parse split file: {e}")
                logger.warning(f"⚠ Could not parse split file: {e}")
        else:
            self.report.warnings.append("Split file not found")
            logger.warning("⚠ Split file not found")
    
    def _analyze_chexpert_labels(self):
        """Analyze CheXpert structured label distribution."""
        chexpert_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-chexpert.csv.gz'
        if not chexpert_file.exists():
            chexpert_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-chexpert.csv'
        
        if not chexpert_file.exists():
            logger.warning("⚠ CheXpert labels file not found")
            return
        
        try:
            logger.info(f"  Loading CheXpert labels from {chexpert_file.name}...")
            if str(chexpert_file).endswith('.gz'):
                df = pd.read_csv(chexpert_file, compression='gzip')
            else:
                df = pd.read_csv(chexpert_file)
            
            self._chexpert_df = df
            self.report.chexpert_studies_labeled = len(df)
            
            logger.info(f"  Studies with CheXpert labels: {len(df):,}")
            logger.info(f"  Columns: {list(df.columns)}")
            
            # Analyze each CheXpert label
            logger.info(f"\n  📊 CheXpert Label Distribution:")
            logger.info(f"  {'Label':<30} {'Positive':>10} {'Negative':>10} {'Uncertain':>10} {'Missing':>10}")
            logger.info(f"  {'-'*70}")
            
            for label in CHEXPERT_LABELS:
                if label in df.columns:
                    pos_count = (df[label] == 1.0).sum()
                    neg_count = (df[label] == 0.0).sum()
                    uncertain_count = (df[label] == -1.0).sum()
                    missing_count = df[label].isna().sum()
                    
                    self.report.chexpert_label_counts[label] = {
                        'positive': int(pos_count),
                        'negative': int(neg_count),
                        'uncertain': int(uncertain_count),
                        'missing': int(missing_count)
                    }
                    
                    logger.info(f"  {label:<30} {pos_count:>10,} {neg_count:>10,} {uncertain_count:>10,} {missing_count:>10,}")
            
            # Store metadata info
            self.report.metadata_csv_info['mimic-cxr-2.0.0-chexpert.csv.gz'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
        except Exception as e:
            logger.error(f"  Error loading CheXpert labels: {e}")
            self.report.warnings.append(f"CheXpert labels error: {e}")
    
    def _analyze_image_metadata(self):
        """Analyze MIMIC-CXR-JPG image metadata."""
        metadata_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-metadata.csv.gz'
        if not metadata_file.exists():
            metadata_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-metadata.csv'
        
        if not metadata_file.exists():
            logger.warning("⚠ Image metadata file not found")
            return
        
        try:
            logger.info(f"  Loading image metadata from {metadata_file.name}...")
            if str(metadata_file).endswith('.gz'):
                df = pd.read_csv(metadata_file, compression='gzip')
            else:
                df = pd.read_csv(metadata_file)
            
            self._metadata_df = df
            logger.info(f"  Total image records: {len(df):,}")
            logger.info(f"  Columns: {list(df.columns)}")
            
            # Store metadata info
            self.report.metadata_csv_info['mimic-cxr-2.0.0-metadata.csv.gz'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # View Position Distribution
            if 'ViewPosition' in df.columns:
                view_counts = df['ViewPosition'].value_counts()
                self.report.view_position_distribution = {str(k): int(v) for k, v in view_counts.items()}
                logger.info(f"\n  📷 View Position Distribution:")
                for view, count in view_counts.head(10).items():
                    pct = count / len(df) * 100
                    logger.info(f"    {view:<20} {count:>10,} ({pct:>5.1f}%)")
            
            # Image Dimensions
            if 'Rows' in df.columns and 'Columns' in df.columns:
                self.report.image_dimension_stats = {
                    'height_mean': float(df['Rows'].mean()),
                    'height_std': float(df['Rows'].std()),
                    'height_min': int(df['Rows'].min()),
                    'height_max': int(df['Rows'].max()),
                    'width_mean': float(df['Columns'].mean()),
                    'width_std': float(df['Columns'].std()),
                    'width_min': int(df['Columns'].min()),
                    'width_max': int(df['Columns'].max()),
                }
                logger.info(f"\n  📐 Image Dimension Statistics:")
                logger.info(f"    Height: {df['Rows'].mean():.0f} ± {df['Rows'].std():.0f} (range: {df['Rows'].min()}-{df['Rows'].max()})")
                logger.info(f"    Width:  {df['Columns'].mean():.0f} ± {df['Columns'].std():.0f} (range: {df['Columns'].min()}-{df['Columns'].max()})")
            
            # Procedure Distribution
            if 'PerformedProcedureStepDescription' in df.columns:
                proc_counts = df['PerformedProcedureStepDescription'].value_counts()
                self.report.procedure_distribution = {str(k): int(v) for k, v in proc_counts.head(10).items()}
                logger.info(f"\n  🏥 Procedure Distribution (top 5):")
                for proc, count in proc_counts.head(5).items():
                    pct = count / len(df) * 100
                    logger.info(f"    {str(proc)[:40]:<40} {count:>10,} ({pct:>5.1f}%)")
            
            # Patient Orientation
            if 'PatientOrientationCodeSequence_CodeMeaning' in df.columns:
                orient_counts = df['PatientOrientationCodeSequence_CodeMeaning'].value_counts(dropna=False)
                self.report.patient_orientation_distribution = {str(k): int(v) for k, v in orient_counts.items()}
                logger.info(f"\n  🧍 Patient Orientation Distribution:")
                for orient, count in orient_counts.items():
                    pct = count / len(df) * 100
                    logger.info(f"    {str(orient):<20} {count:>10,} ({pct:>5.1f}%)")
            
        except Exception as e:
            logger.error(f"  Error loading image metadata: {e}")
            self.report.warnings.append(f"Image metadata error: {e}")
    
    def _analyze_qba_metadata_files(self):
        """Analyze all MIMIC-Ext-CXR-QBA metadata CSV files."""
        metadata_dir = self.mimic_qa_path / 'metadata'
        
        if not metadata_dir.exists():
            logger.warning(f"⚠ QBA metadata directory not found: {metadata_dir}")
            return
        
        # List of expected metadata files
        metadata_files = [
            'patient_metadata.csv.gz',
            'study_metadata.csv.gz', 
            'image_metadata.csv.gz',
            'question_metadata.csv.gz',
            'answer_metadata.csv.gz',
        ]
        
        for filename in metadata_files:
            filepath = metadata_dir / filename
            parquet_path = metadata_dir / filename.replace('.csv.gz', '.parquet')
            
            # Try parquet first (faster)
            if parquet_path.exists():
                try:
                    df = pd.read_parquet(parquet_path)
                    self._log_metadata_file_info(filename.replace('.csv.gz', '.parquet'), df)
                    continue
                except Exception:
                    pass
            
            if filepath.exists():
                try:
                    logger.info(f"\n  Loading {filename}...")
                    df = pd.read_csv(filepath, compression='gzip', nrows=100000)  # Sample for large files
                    self._log_metadata_file_info(filename, df, sampled=True)
                except Exception as e:
                    logger.warning(f"    Error loading {filename}: {e}")
            else:
                logger.debug(f"    File not found: {filename}")
        
        # Check dataset_info.json
        dataset_info_path = metadata_dir / 'dataset_info.json'
        if dataset_info_path.exists():
            try:
                with open(dataset_info_path) as f:
                    dataset_info = json.load(f)
                logger.info(f"\n  📚 dataset_info.json:")
                logger.info(f"    Keys: {list(dataset_info.keys())}")
                if 'finding_entities' in dataset_info:
                    logger.info(f"    Finding entities: {len(dataset_info['finding_entities'])} types")
                if 'region_names' in dataset_info:
                    logger.info(f"    Region names: {len(dataset_info['region_names'])} types")
            except Exception as e:
                logger.warning(f"    Error loading dataset_info.json: {e}")
    
    def _log_metadata_file_info(self, filename: str, df: pd.DataFrame, sampled: bool = False):
        """Log detailed info about a metadata CSV/parquet file."""
        sample_note = " (sampled 100k rows)" if sampled else ""
        logger.info(f"    📄 {filename}{sample_note}")
        logger.info(f"       Rows: {len(df):,} | Columns: {len(df.columns)}")
        logger.info(f"       Columns: {list(df.columns)[:8]}{'...' if len(df.columns) > 8 else ''}")
        
        # Store in report
        self.report.metadata_csv_info[filename] = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'sampled': sampled,
            'sample_values': {}
        }
        
        # Show sample values for key columns
        for col in df.columns[:5]:
            try:
                if df[col].dtype == 'object':
                    unique_vals = df[col].dropna().unique()[:3]
                    self.report.metadata_csv_info[filename]['sample_values'][col] = [str(v)[:30] for v in unique_vals]
                else:
                    self.report.metadata_csv_info[filename]['sample_values'][col] = {
                        'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
                        'max': float(df[col].max()) if pd.notna(df[col].max()) else None,
                    }
            except Exception:
                pass
    
    def _analyze_cross_dataset_correlation(self):
        """Analyze correlation between MIMIC-CXR-JPG and MIMIC-Ext-CXR-QBA."""
        logger.info("  Checking cross-dataset alignment...")
        
        # Get study IDs from CXR-JPG
        cxr_study_ids = set()
        if self._split_df is not None:
            cxr_study_ids = set(self._split_df['study_id'].unique())
            logger.info(f"  MIMIC-CXR-JPG studies: {len(cxr_study_ids):,}")
        
        # Get study IDs from QBA
        qba_study_ids = set()
        qa_dir = self.mimic_qa_path / 'qa'
        if qa_dir.exists():
            # Sample QA files to get study IDs
            for qa_file in list(qa_dir.rglob('*.qa.json'))[:10000]:
                study_id = qa_file.stem.split('.')[0]
                if study_id.startswith('s'):
                    qba_study_ids.add(int(study_id[1:]))
                else:
                    try:
                        qba_study_ids.add(int(study_id))
                    except ValueError:
                        pass
            logger.info(f"  MIMIC-Ext-CXR-QBA studies (sampled): {len(qba_study_ids):,}")
        
        # Calculate overlap
        if cxr_study_ids and qba_study_ids:
            matched = cxr_study_ids & qba_study_ids
            only_cxr = cxr_study_ids - qba_study_ids
            only_qba = qba_study_ids - cxr_study_ids
            
            self.report.matched_studies = len(matched)
            self.report.unmatched_studies_cxr = len(only_cxr)
            self.report.unmatched_studies_qba = len(only_qba)
            
            match_pct = len(matched) / len(cxr_study_ids) * 100 if cxr_study_ids else 0
            logger.info(f"\n  🔗 Cross-Dataset Alignment:")
            logger.info(f"    Matched studies: {len(matched):,} ({match_pct:.1f}%)")
            logger.info(f"    Only in CXR-JPG: {len(only_cxr):,}")
            logger.info(f"    Only in QBA (sampled): {len(only_qba):,}")
    
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
        def process_qa_file(qa_file):
            nonlocal total_qa
            try:
                with open(qa_file) as f:
                    qa_data = json.load(f)
                
                questions = qa_data.get('questions', [])
                total_local = len(questions)
                local_qt = Counter()
                local_at = Counter()
                local_pol = Counter()
                local_regions = Counter()
                
                for q in questions:
                    # Question type
                    q_type = q.get('question_type', 'unknown')
                    local_qt[q_type] += 1
                    
                    # Categorize question type
                    q_type_lower = q_type.lower()
                    if any(t in q_type_lower for t in ['is_', 'has_', 'yes', 'no']):
                        local_at['binary'] += 1
                    elif any(t in q_type_lower for t in ['where', 'locate', 'position']):
                        local_at['region'] += 1
                    elif any(t in q_type_lower for t in ['severe', 'grade']):
                        local_at['severity'] += 1
                    else:
                        local_at['category'] += 1
                    
                    # Polarity (positive/negative findings)
                    answers = q.get('answers', [])
                    if answers:
                        answer_text = answers[0].get('text', '').lower()
                        if 'yes' in answer_text or 'present' in answer_text or 'positive' in answer_text:
                            local_pol['positive'] += 1
                        elif 'no' in answer_text or 'absent' in answer_text or 'negative' in answer_text:
                            local_pol['negative'] += 1
                        else:
                            local_pol['neutral'] += 1
                    
                    # Anatomical regions
                    question_text = q.get('question', '').lower()
                    for region_cat, keywords in self.ANATOMICAL_REGIONS.items():
                        if any(kw in question_text for kw in keywords):
                            local_regions[region_cat] += 1
                            break
                    else:
                        local_regions['other'] += 1
                
                return total_local, local_qt, local_at, local_pol, local_regions
            except Exception:
                return 0, Counter(), Counter(), Counter(), Counter()
        
        # Parallelize over QA files
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for total_local, l_qt, l_at, l_pol, l_regions in ex.map(process_qa_file, qa_files):
                total_qa += total_local
                question_types.update(l_qt)
                answer_types.update(l_at)
                polarity.update(l_pol)
                regions.update(l_regions)
        
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

        def process_sg_file(sg_file):
            try:
                with open(sg_file) as f:
                    sg = json.load(f)
                
                observations = sg.get('observations', {})
                num_obs = len(observations)
                
                local_regions = 0
                local_bboxes = 0
                
                for _, obs in observations.items():
                    regions = obs.get('regions', [])
                    local_regions += len(regions)
                    
                    if 'localization' in obs and obs['localization']:
                        local_bboxes += 1
                
                return num_obs, local_regions, local_bboxes
            except Exception:
                return 0, 0, 0
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for num_obs, local_regions, local_bboxes in ex.map(process_sg_file, sg_files):
                if num_obs == 0 and local_regions == 0:
                    continue
                graphs_analyzed += 1
                total_observations += num_obs
                total_regions += local_regions
                total_bboxes += local_bboxes
        
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
        """Generate comprehensive final analysis report and determine readiness."""
        # Determine readiness
        critical_issues = len(self.report.issues)
        
        self.report.is_ready = (
            critical_issues == 0 and
            self.report.total_qa_pairs > 0 and
            self.report.total_images > 0
        )
        
        # Print comprehensive summary
        logger.info("\n" + "=" * 70)
        logger.info("                    COMPREHENSIVE ANALYSIS SUMMARY                    ")
        logger.info("=" * 70)
        
        # MIMIC-CXR-JPG Statistics
        logger.info(f"\n{'─'*70}")
        logger.info(f"📁 MIMIC-CXR-JPG Dataset")
        logger.info(f"{'─'*70}")
        logger.info(f"   Total Images:           {self.report.total_images:>15,}")
        logger.info(f"   Total Studies:          {self.report.total_studies:>15,}")
        logger.info(f"   Total Patients:         {self.report.total_patients:>15,}")
        logger.info(f"   CheXpert Labeled:       {self.report.chexpert_studies_labeled:>15,}")
        
        logger.info(f"\n   📊 Train/Val/Test Split:")
        logger.info(f"      Train:      {self.report.train_samples:>12,}")
        logger.info(f"      Validate:   {self.report.val_samples:>12,}")
        logger.info(f"      Test:       {self.report.test_samples:>12,}")
        
        # CheXpert summary
        if self.report.chexpert_label_counts:
            logger.info(f"\n   🏷️ CheXpert Labels (top 5 by positive rate):")
            sorted_labels = sorted(
                self.report.chexpert_label_counts.items(),
                key=lambda x: x[1].get('positive', 0) / max(sum(x[1].values()), 1),
                reverse=True
            )[:5]
            for label, counts in sorted_labels:
                total = sum(counts.values())
                pos_rate = counts.get('positive', 0) / total * 100 if total > 0 else 0
                logger.info(f"      {label:<25} {counts.get('positive', 0):>8,} pos ({pos_rate:>5.1f}%)")
        
        # View Position summary
        if self.report.view_position_distribution:
            logger.info(f"\n   📷 View Positions (top 4):")
            for view, count in list(self.report.view_position_distribution.items())[:4]:
                pct = count / self.report.total_images * 100 if self.report.total_images > 0 else 0
                logger.info(f"      {view:<15} {count:>10,} ({pct:>5.1f}%)")
        
        # MIMIC-Ext-CXR-QBA Statistics
        logger.info(f"\n{'─'*70}")
        logger.info(f"🔗 MIMIC-Ext-CXR-QBA Dataset")
        logger.info(f"{'─'*70}")
        logger.info(f"   Total QA Pairs:         {self.report.total_qa_pairs:>15,}")
        logger.info(f"   Total Scene Graphs:     {self.report.total_scene_graphs:>15,}")
        logger.info(f"   Avg Obs/Graph:          {self.report.avg_observations_per_graph:>15.2f}")
        logger.info(f"   BBox Coverage:          {self.report.bbox_coverage*100:>14.1f}%")
        
        if self.report.question_type_distribution:
            logger.info(f"\n   ❓ Top Question Types:")
            for q_type, count in list(self.report.question_type_distribution.items())[:5]:
                logger.info(f"      {q_type:<30} {count:>10,}")
        
        if self.report.polarity_distribution:
            logger.info(f"\n   ⚖️ Finding Polarity:")
            total_pol = sum(self.report.polarity_distribution.values())
            for pol, count in self.report.polarity_distribution.items():
                pct = count / total_pol * 100 if total_pol > 0 else 0
                logger.info(f"      {pol:<15} {count:>10,} ({pct:>5.1f}%)")
        
        if self.report.region_distribution:
            logger.info(f"\n   🫁 Anatomical Regions:")
            total_reg = sum(self.report.region_distribution.values())
            for region, count in self.report.region_distribution.items():
                pct = count / total_reg * 100 if total_reg > 0 else 0
                logger.info(f"      {region:<15} {count:>10,} ({pct:>5.1f}%)")
        
        # Cross-dataset correlation
        if self.report.matched_studies > 0:
            logger.info(f"\n{'─'*70}")
            logger.info(f"🔀 Cross-Dataset Correlation")
            logger.info(f"{'─'*70}")
            logger.info(f"   Matched Studies:        {self.report.matched_studies:>15,}")
            logger.info(f"   CXR-JPG Only:           {self.report.unmatched_studies_cxr:>15,}")
            logger.info(f"   QBA Only (sampled):     {self.report.unmatched_studies_qba:>15,}")
        
        # Metadata files analyzed
        if self.report.metadata_csv_info:
            logger.info(f"\n{'─'*70}")
            logger.info(f"📑 Metadata Files Analyzed")
            logger.info(f"{'─'*70}")
            for filename, info in list(self.report.metadata_csv_info.items())[:8]:
                rows = info.get('rows', 0)
                cols = info.get('columns', 0)
                logger.info(f"   {filename:<40} {rows:>10,} rows × {cols:>3} cols")
        
        # Issues and warnings
        if self.report.issues:
            logger.error(f"\n{'─'*70}")
            logger.error(f"❌ Critical Issues ({len(self.report.issues)})")
            logger.error(f"{'─'*70}")
            for issue in self.report.issues:
                logger.error(f"   • {issue}")
        
        if self.report.warnings:
            logger.warning(f"\n{'─'*70}")
            logger.warning(f"⚠️ Warnings ({len(self.report.warnings)})")
            logger.warning(f"{'─'*70}")
            for warning in self.report.warnings:
                logger.warning(f"   • {warning}")
        
        # Final status
        logger.info("\n" + "=" * 70)
        if self.report.is_ready:
            logger.info("  ✅ DATA IS READY FOR TRAINING")
            logger.info("     Run: python train_mimic_cxr.py --config configs/gcp_server_config.yaml")
        else:
            logger.error("  ❌ DATA NOT READY - Please resolve issues above")
        logger.info("=" * 70)
        
        # Save report to file
        self._save_report()
        
        # Generate plots if available
        if PLOTTING_AVAILABLE:
            self._generate_plots()
    
    def _convert_to_native(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _save_report(self):
        """Save comprehensive analysis report to JSON."""
        report_path = self.output_dir / 'analysis_report.json'
        
        report_dict = {
            'summary': {
                'is_ready': self.report.is_ready,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
            },
            'mimic_cxr_jpg': {
                'total_images': int(self.report.total_images),
                'total_studies': int(self.report.total_studies),
                'total_patients': int(self.report.total_patients),
                'chexpert_studies_labeled': int(self.report.chexpert_studies_labeled),
                'splits': {
                    'train': int(self.report.train_samples),
                    'validate': int(self.report.val_samples),
                    'test': int(self.report.test_samples),
                },
                'chexpert_labels': self._convert_to_native(self.report.chexpert_label_counts),
                'view_positions': self._convert_to_native(self.report.view_position_distribution),
                'image_dimensions': self._convert_to_native(self.report.image_dimension_stats),
                'procedures': self._convert_to_native(self.report.procedure_distribution),
                'patient_orientations': self._convert_to_native(self.report.patient_orientation_distribution),
            },
            'mimic_ext_cxr_qba': {
                'total_qa_pairs': int(self.report.total_qa_pairs),
                'total_scene_graphs': int(self.report.total_scene_graphs),
                'scene_graph_quality': {
                    'avg_observations_per_graph': float(self.report.avg_observations_per_graph),
                    'avg_regions_per_observation': float(self.report.avg_regions_per_observation),
                    'avg_sentences_per_graph': float(self.report.avg_sentences_per_graph),
                    'bbox_coverage': float(self.report.bbox_coverage),
                },
                'distributions': {
                    'question_types': self._convert_to_native(self.report.question_type_distribution),
                    'answer_types': self._convert_to_native(self.report.answer_type_distribution),
                    'polarity': self._convert_to_native(self.report.polarity_distribution),
                    'regions': self._convert_to_native(self.report.region_distribution),
                    'entities': self._convert_to_native(self.report.entity_distribution),
                    'categories': self._convert_to_native(self.report.category_distribution),
                },
            },
            'cross_dataset': {
                'matched_studies': int(self.report.matched_studies),
                'unmatched_cxr_only': int(self.report.unmatched_studies_cxr),
                'unmatched_qba_only': int(self.report.unmatched_studies_qba),
            },
            'metadata_files': self._convert_to_native(self.report.metadata_csv_info),
            'issues': self.report.issues,
            'warnings': self.report.warnings,
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"\n📄 Comprehensive report saved to: {report_path}")
    
    def _generate_plots(self):
        """Generate comprehensive visualization plots."""
        if not PLOTTING_AVAILABLE:
            return
        
        # ============ Plot 1: QA Distribution ============
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Question type distribution
        if self.report.question_type_distribution:
            ax = axes[0, 0]
            types = list(self.report.question_type_distribution.keys())[:10]
            counts = [self.report.question_type_distribution[t] for t in types]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(types)))
            ax.barh(types, counts, color=colors)
            ax.set_xlabel('Count')
            ax.set_title('Top 10 Question Types', fontweight='bold')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # Polarity distribution
        if self.report.polarity_distribution:
            ax = axes[0, 1]
            labels = list(self.report.polarity_distribution.keys())
            sizes = list(self.report.polarity_distribution.values())
            colors = {'positive': '#e74c3c', 'negative': '#2ecc71', 'neutral': '#95a5a6'}
            pie_colors = [colors.get(l, '#3498db') for l in labels]
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                              colors=pie_colors, explode=[0.02]*len(labels))
            ax.set_title('Finding Polarity Distribution', fontweight='bold')
        
        # Region distribution
        if self.report.region_distribution:
            ax = axes[1, 0]
            regions = list(self.report.region_distribution.keys())
            counts = list(self.report.region_distribution.values())
            colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
            ax.bar(regions, counts, color=colors)
            ax.set_xlabel('Anatomical Region')
            ax.set_ylabel('Count')
            ax.set_title('Anatomical Region Distribution', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Answer type distribution
        if self.report.answer_type_distribution:
            ax = axes[1, 1]
            types = list(self.report.answer_type_distribution.keys())
            counts = list(self.report.answer_type_distribution.values())
            ax.bar(types, counts, color='mediumpurple')
            ax.set_xlabel('Answer Type')
            ax.set_ylabel('Count')
            ax.set_title('Answer Type Distribution', fontweight='bold')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plot_path = self.output_dir / 'qa_distribution_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 QA distribution plots saved to: {plot_path}")
        
        # ============ Plot 2: CheXpert Labels ============
        if self.report.chexpert_label_counts:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left: Stacked bar chart of label distribution
            ax = axes[0]
            labels = list(self.report.chexpert_label_counts.keys())
            pos_counts = [self.report.chexpert_label_counts[l].get('positive', 0) for l in labels]
            neg_counts = [self.report.chexpert_label_counts[l].get('negative', 0) for l in labels]
            unc_counts = [self.report.chexpert_label_counts[l].get('uncertain', 0) for l in labels]
            
            x = np.arange(len(labels))
            width = 0.25
            
            ax.barh(x - width, pos_counts, width, label='Positive (1.0)', color='#e74c3c', alpha=0.8)
            ax.barh(x, neg_counts, width, label='Negative (0.0)', color='#2ecc71', alpha=0.8)
            ax.barh(x + width, unc_counts, width, label='Uncertain (-1.0)', color='#f39c12', alpha=0.8)
            
            ax.set_yticks(x)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Number of Studies')
            ax.set_title('CheXpert Label Distribution', fontweight='bold', fontsize=12)
            ax.legend(loc='lower right')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
            # Right: Positive rate heatmap
            ax = axes[1]
            total_counts = [pos_counts[i] + neg_counts[i] + unc_counts[i] for i in range(len(labels))]
            pos_rates = [pos_counts[i] / total_counts[i] * 100 if total_counts[i] > 0 else 0 for i in range(len(labels))]
            
            colors = plt.cm.RdYlGn_r(np.array(pos_rates) / 100)
            ax.barh(labels, pos_rates, color=colors)
            ax.set_xlabel('Positive Rate (%)')
            ax.set_title('CheXpert Positive Finding Rate', fontweight='bold', fontsize=12)
            ax.set_xlim(0, 100)
            
            # Add percentage labels
            for i, (label, rate) in enumerate(zip(labels, pos_rates)):
                ax.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=8)
            
            plt.tight_layout()
            plot_path = self.output_dir / 'chexpert_distribution.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"📊 CheXpert distribution saved to: {plot_path}")
        
        # ============ Plot 3: Image Metadata ============
        if self.report.view_position_distribution:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            # View Position
            ax = axes[0]
            views = list(self.report.view_position_distribution.keys())[:8]
            counts = [self.report.view_position_distribution[v] for v in views]
            colors = plt.cm.Paired(np.linspace(0, 1, len(views)))
            wedges, texts, autotexts = ax.pie(counts, labels=views, autopct='%1.1f%%',
                                              colors=colors, pctdistance=0.75)
            ax.set_title('View Position Distribution', fontweight='bold')
            
            # Image dimensions histogram
            ax = axes[1]
            if self.report.image_dimension_stats:
                stats = self.report.image_dimension_stats
                dim_info = (f"Height: {stats['height_mean']:.0f}±{stats['height_std']:.0f}\n"
                           f"  Range: {stats['height_min']}-{stats['height_max']}\n\n"
                           f"Width: {stats['width_mean']:.0f}±{stats['width_std']:.0f}\n"
                           f"  Range: {stats['width_min']}-{stats['width_max']}")
                ax.text(0.5, 0.5, dim_info, ha='center', va='center', fontsize=12,
                       transform=ax.transAxes, fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                ax.set_title('Image Dimensions (pixels)', fontweight='bold')
                ax.axis('off')
            
            # Patient orientation
            ax = axes[2]
            if self.report.patient_orientation_distribution:
                orientations = list(self.report.patient_orientation_distribution.keys())
                counts = list(self.report.patient_orientation_distribution.values())
                ax.bar(orientations, counts, color='teal')
                ax.set_xlabel('Orientation')
                ax.set_ylabel('Count')
                ax.set_title('Patient Orientation', fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = self.output_dir / 'image_metadata_plots.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"📊 Image metadata plots saved to: {plot_path}")
        
        # ============ Plot 4: Dataset Summary ============
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        summary_text = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MIMIC-CXR VQA DATASET SUMMARY                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  📁 MIMIC-CXR-JPG Dataset                                             ║
║  ├─ Total Images:         {self.report.total_images:>12,}                            ║
║  ├─ Total Studies:        {self.report.total_studies:>12,}                            ║
║  ├─ Total Patients:       {self.report.total_patients:>12,}                            ║
║  └─ CheXpert Labeled:     {self.report.chexpert_studies_labeled:>12,}                            ║
║                                                                        ║
║  📊 Train/Val/Test Split                                              ║
║  ├─ Train:                {self.report.train_samples:>12,}                            ║
║  ├─ Validate:             {self.report.val_samples:>12,}                            ║
║  └─ Test:                 {self.report.test_samples:>12,}                            ║
║                                                                        ║
║  🔗 MIMIC-Ext-CXR-QBA Dataset                                         ║
║  ├─ Total QA Pairs:       {self.report.total_qa_pairs:>12,}                            ║
║  ├─ Total Scene Graphs:   {self.report.total_scene_graphs:>12,}                            ║
║  └─ Avg Obs/Graph:        {self.report.avg_observations_per_graph:>12.2f}                            ║
║                                                                        ║
║  ✓ Status: {'READY FOR TRAINING' if self.report.is_ready else 'NOT READY - Check Issues':^20}                            ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plot_path = self.output_dir / 'dataset_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"📊 Dataset summary saved to: {plot_path}")

    def _get_image_path(self, patient_id: str, study_id: str) -> Optional[Path]:
        """
        Locate an image path for a given patient/study.
        
        MIMIC-CXR-JPG structure:
        files/p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        Example: files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        """
        files_dir = self.mimic_cxr_path / 'files'
        if not files_dir.exists():
            return None
        
        # Handle patient_id format (may be "p10000032" or "10000032")
        if patient_id.startswith('p'):
            patient_num = patient_id[1:]
            patient_folder = patient_id
        else:
            patient_num = patient_id
            patient_folder = f"p{patient_id}"
        
        # First two digits determine the group folder (p10, p11, etc.)
        group_folder = f"p{patient_num[:2]}"
        
        # Handle study_id format (may be "s50414267" or "50414267")
        if study_id.startswith('s'):
            study_folder = study_id
        else:
            study_folder = f"s{study_id}"
        
        study_path = files_dir / group_folder / patient_folder / study_folder
        
        if study_path.exists():
            imgs = sorted(study_path.glob('*.jpg'))
            if imgs:
                return imgs[0]
        
        return None

    def _get_scene_graph_path(self, patient_id: str, study_id: str) -> Optional[Path]:
        """
        Locate scene graph json.
        
        MIMIC-Ext-CXR-QBA structure:
        scene_data/p{XX}/p{subject_id}/s{study_id}.scene_graph.json
        """
        # Handle patient_id format
        if patient_id.startswith('p'):
            patient_num = patient_id[1:]
            patient_folder = patient_id
        else:
            patient_num = patient_id
            patient_folder = f"p{patient_id}"
        
        group_folder = f"p{patient_num[:2]}"
        
        # Handle study_id format
        study_base = study_id[1:] if study_id.startswith('s') else study_id
        
        candidates = [
            self.mimic_qa_path / 'scene_data' / group_folder / patient_folder / f"s{study_base}.scene_graph.json",
            self.mimic_qa_path / 'scene_data' / group_folder / patient_folder / f"{study_base}.scene_graph.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _visualize_samples(self, num_samples: int = 10):
        """
        Visualize images + scene graphs for sample studies.
        
        Creates detailed visualizations showing:
        - Original chest X-ray image
        - Bounding boxes from scene graph observations
        - Sample QA pairs
        - Observation details
        """
        logger.info(f"\n[Visualization] Creating {num_samples} sample visualizations...")
        
        qa_dir = self.mimic_qa_path / 'qa'
        if not qa_dir.exists():
            logger.warning("QA directory missing; skipping visualization.")
            return

        qa_files = list(qa_dir.rglob('*.qa.json'))
        if not qa_files:
            logger.warning("No QA files found for visualization.")
            return

        # Sample more than needed in case some fail
        sample_pool = random.sample(qa_files, min(num_samples * 3, len(qa_files)))
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        
        for qa_file in sample_pool:
            if successful >= num_samples:
                break
                
            try:
                with open(qa_file) as f:
                    qa_data = json.load(f)
                
                patient_id = qa_data.get('patient_id', '')
                study_id = qa_data.get('study_id', '')
                
                # Fallback: extract from path if not in JSON
                if not patient_id:
                    # Path: .../p{XX}/p{patient_id}/s{study_id}.qa.json
                    patient_id = qa_file.parts[-2]  # p10000032
                if not study_id:
                    study_id = qa_file.stem.split('.')[0]  # s50414267

                img_path = self._get_image_path(patient_id, study_id)
                if img_path is None or not img_path.exists():
                    logger.debug(f"No image found for {patient_id}/{study_id}")
                    continue

                image = Image.open(img_path).convert('RGB')
                img_width, img_height = image.size

                # Load scene graph for bboxes
                sg_path = self._get_scene_graph_path(patient_id, study_id)
                observations_data = []
                all_bboxes = []
                
                if sg_path and sg_path.exists():
                    with open(sg_path) as f:
                        sg = json.load(f)
                    
                    observations = sg.get('observations', {})
                    
                    # Collect all observations with their bboxes
                    for obs_id, obs in observations.items():
                        obs_name = obs.get('name', 'unknown')
                        positiveness = obs.get('positiveness', 'unknown')
                        entities = obs.get('obs_entities', [])
                        regions = [r.get('region', '') if isinstance(r, dict) else r for r in obs.get('regions', [])]
                        
                        # Get bboxes from localization
                        locs = obs.get('localization', {})
                        obs_bboxes = []
                        for img_id, loc_data in locs.items():
                            bboxes = loc_data.get('bboxes', []) or []
                            obs_bboxes.extend(bboxes)
                        
                        if obs_bboxes:
                            observations_data.append({
                                'name': obs_name,
                                'positiveness': positiveness,
                                'entities': entities,
                                'regions': regions,
                                'bboxes': obs_bboxes
                            })
                            all_bboxes.extend(obs_bboxes)

                # Get sample questions
                questions = qa_data.get('questions', [])[:3]  # First 3 questions
                
                # Create visualization with 2 subplots
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
                # Left: Image with bounding boxes
                ax1 = axes[0]
                ax1.imshow(image)
                
                # Color-code by positiveness
                colors = {'pos': 'red', 'neg': 'lime', 'unknown': 'yellow'}
                
                for obs in observations_data[:5]:  # Limit to 5 observations for clarity
                    color = colors.get(obs['positiveness'], 'cyan')
                    for bb in obs['bboxes']:
                        if len(bb) == 4:
                            x1, y1, x2, y2 = bb
                            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                 linewidth=2, edgecolor=color, facecolor='none')
                            ax1.add_patch(rect)
                            # Add label
                            label = obs['name'][:20] + ('...' if len(obs['name']) > 20 else '')
                            ax1.text(x1, y1 - 5, label, fontsize=7, color=color,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                
                ax1.set_title(f"Patient: {patient_id} | Study: {study_id}\nImage: {img_path.name} ({img_width}x{img_height})",
                             fontsize=10)
                ax1.axis('off')
                
                # Add legend
                legend_elements = [
                    mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Positive finding'),
                    mpatches.Patch(facecolor='none', edgecolor='lime', linewidth=2, label='Negative finding'),
                ]
                ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
                
                # Right: QA pairs and observations info
                ax2 = axes[1]
                ax2.axis('off')
                
                text_content = f"📊 Scene Graph Summary\n{'='*40}\n"
                text_content += f"Observations: {len(observations_data)}\n"
                text_content += f"Total bboxes: {len(all_bboxes)}\n\n"
                
                text_content += f"🔬 Top Observations:\n{'-'*40}\n"
                for i, obs in enumerate(observations_data[:5], 1):
                    pos_marker = "🔴" if obs['positiveness'] == 'pos' else "🟢" if obs['positiveness'] == 'neg' else "🟡"
                    text_content += f"{i}. {pos_marker} {obs['name'][:40]}\n"
                    if obs['entities']:
                        text_content += f"   Entities: {', '.join(obs['entities'][:3])}\n"
                    if obs['regions']:
                        text_content += f"   Regions: {', '.join(obs['regions'][:3])}\n"
                
                text_content += f"\n❓ Sample Questions:\n{'-'*40}\n"
                for i, q in enumerate(questions, 1):
                    q_text = q.get('question', '')[:60]
                    q_type = q.get('question_type', 'unknown')
                    text_content += f"{i}. [{q_type}]\n   Q: {q_text}...\n"
                    
                    # Get first answer
                    answers = q.get('answers', [])
                    if answers:
                        a_text = answers[0].get('text', '')[:50]
                        text_content += f"   A: {a_text}...\n"
                    text_content += "\n"
                
                ax2.text(0.02, 0.98, text_content, transform=ax2.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                plt.tight_layout()
                out_path = viz_dir / f"sample_{successful+1:02d}_{patient_id}_{study_id}.png"
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                logger.info(f"  ✓ Saved: {out_path.name}")
                successful += 1
                
            except Exception as e:
                logger.debug(f"Visualization failed for {qa_file}: {e}")
                continue
        
        if successful > 0:
            logger.info(f"\n📸 {successful} visualizations saved to: {viz_dir}")

    def _visualize_scene_graphs(self, num_samples: int = 5):
        """
        Create visual scene graph representations showing:
        - Node structure (observations, regions, sentences)
        - Relationships between nodes
        - Observation properties and localizations
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, skipping scene graph visualization")
            return
        
        logger.info(f"  Creating scene graph structure visualizations...")
        
        scene_data_dir = self.mimic_qa_path / 'scene_data'
        if not scene_data_dir.exists():
            logger.warning("Scene data directory not found")
            return
        
        sg_files = list(scene_data_dir.rglob('*.scene_graph.json'))
        if not sg_files:
            logger.warning("No scene graph files found")
            return
        
        sample_files = random.sample(sg_files, min(num_samples * 2, len(sg_files)))
        viz_dir = self.output_dir / 'scene_graph_visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        
        for sg_file in sample_files:
            if successful >= num_samples:
                break
            
            try:
                with open(sg_file) as f:
                    sg = json.load(f)
                
                patient_id = sg.get('patient_id', sg_file.parts[-2])
                study_id = sg.get('study_id', sg_file.stem.split('.')[0])
                
                observations = sg.get('observations', {})
                regions = sg.get('regions', {})
                sentences = sg.get('sentences', {})
                
                if not observations:
                    continue
                
                # Create multi-panel visualization
                fig = plt.figure(figsize=(20, 12))
                gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
                
                # Panel 1: Observation summary table
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.axis('off')
                ax1.set_title(f"📋 Observations ({len(observations)})", fontsize=12, fontweight='bold')
                
                obs_text = ""
                for i, (obs_id, obs) in enumerate(list(observations.items())[:8]):
                    pos_marker = "🔴" if obs.get('positiveness') == 'pos' else "🟢" if obs.get('positiveness') == 'neg' else "🟡"
                    name = obs.get('name', 'unknown')[:35]
                    entities = ', '.join(obs.get('obs_entities', [])[:2])
                    obs_text += f"{pos_marker} {obs_id}: {name}\n"
                    obs_text += f"   Entities: {entities}\n"
                    obs_text += f"   Certainty: {obs.get('certainty', 'unknown')}\n\n"
                
                ax1.text(0.02, 0.98, obs_text, transform=ax1.transAxes,
                        fontsize=8, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
                
                # Panel 2: Regions summary
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.axis('off')
                ax2.set_title(f"🫁 Regions ({len(regions)})", fontsize=12, fontweight='bold')
                
                region_text = "Region Name             | Laterality\n" + "-" * 40 + "\n"
                for region_name, region_data in list(regions.items())[:12]:
                    lat = region_data.get('laterality', 'N/A')
                    region_text += f"{region_name[:24]:<24} | {lat}\n"
                
                ax2.text(0.02, 0.98, region_text, transform=ax2.transAxes,
                        fontsize=8, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
                
                # Panel 3: Report sentences
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.axis('off')
                ax3.set_title(f"📝 Report Sentences ({len(sentences)})", fontsize=12, fontweight='bold')
                
                sent_text = ""
                for sent_id, sent in list(sentences.items())[:6]:
                    section = sent.get('section_type', 'UNKNOWN')
                    text = sent.get('sentence', '')[:60]
                    sent_text += f"[{section}] {sent_id}:\n  \"{text}...\"\n\n"
                
                ax3.text(0.02, 0.98, sent_text, transform=ax3.transAxes,
                        fontsize=8, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9))
                
                # Panel 4: Category/Subcategory distribution
                ax4 = fig.add_subplot(gs[1, 0])
                categories = Counter()
                for obs in observations.values():
                    for cat in obs.get('obs_categories', []):
                        categories[cat] += 1
                
                if categories:
                    cats = list(categories.keys())[:6]
                    counts = [categories[c] for c in cats]
                    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
                    ax4.barh(cats, counts, color=colors)
                    ax4.set_xlabel('Count')
                    ax4.set_title('Observation Categories', fontsize=10)
                else:
                    ax4.text(0.5, 0.5, "No categories", ha='center', va='center')
                    ax4.set_title('Observation Categories', fontsize=10)
                
                # Panel 5: Polarity pie chart
                ax5 = fig.add_subplot(gs[1, 1])
                polarity = Counter()
                for obs in observations.values():
                    polarity[obs.get('positiveness', 'unknown')] += 1
                
                if polarity:
                    labels = list(polarity.keys())
                    sizes = list(polarity.values())
                    color_map = {'pos': '#e74c3c', 'neg': '#2ecc71', 'unknown': '#95a5a6'}
                    colors = [color_map.get(l, '#3498db') for l in labels]
                    ax5.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors,
                           explode=[0.05] * len(labels))
                    ax5.set_title('Finding Polarity', fontsize=10)
                
                # Panel 6: Indication/Clinical info
                ax6 = fig.add_subplot(gs[1, 2])
                ax6.axis('off')
                ax6.set_title("🏥 Clinical Context", fontsize=12, fontweight='bold')
                
                indication = sg.get('indication', {})
                clinical_text = ""
                if indication:
                    if indication.get('indication_summary'):
                        clinical_text += f"Summary:\n  {indication['indication_summary'][:100]}...\n\n"
                    if indication.get('patient_info'):
                        clinical_text += f"Patient Info:\n  {indication['patient_info'][:80]}\n\n"
                    if indication.get('evaluation'):
                        clinical_text += f"Evaluation:\n  {indication['evaluation'][:80]}\n"
                else:
                    clinical_text = "No indication information available"
                
                ax6.text(0.02, 0.98, clinical_text, transform=ax6.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.9))
                
                # Overall title
                fig.suptitle(f"Scene Graph: {patient_id} / {study_id}", fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                out_path = viz_dir / f"scene_graph_{successful+1:02d}_{patient_id}_{study_id}.png"
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                logger.info(f"  ✓ Saved: {out_path.name}")
                successful += 1
                
            except Exception as e:
                logger.debug(f"Scene graph visualization failed for {sg_file}: {e}")
                continue
        
        if successful > 0:
            logger.info(f"\n🔗 {successful} scene graph visualizations saved to: {viz_dir}")


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
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker threads for analysis (default: half of CPUs, min 8)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help='Prefetch factor for parallel loading')
    parser.add_argument('--visualize_samples', type=int, default=10,
                       help='Number of samples to visualize (images + scene graphs)')
    
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
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        visualize_samples=args.visualize_samples
    )
    
    report = analyzer.run_full_analysis()
    
    # Return exit code based on readiness
    sys.exit(0 if report.is_ready else 1)


if __name__ == '__main__':
    main()

