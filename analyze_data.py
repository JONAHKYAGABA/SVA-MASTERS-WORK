#!/usr/bin/env python3
"""
MIMIC-CXR VQA Comprehensive Data Analysis Script

This script performs COMPREHENSIVE analysis of BOTH datasets:
1. MIMIC-CXR-JPG: Images, CheXpert labels, metadata
2. MIMIC-Ext-CXR-QBA: Scene graphs, QA pairs, observations

    =============================================================================
MULTI-IMAGE PATIENT HANDLING
=============================================================================
MIMIC-CXR has a hierarchical structure: Patient -> Studies -> Images

- **Patients** can have multiple longitudinal **studies** (exams over time)
- **Studies** can have multiple **images** (different views: PA, AP, lateral)
- **Scene graphs** and **QA pairs** are at the STUDY level, not image level
- **Bounding boxes** in scene graphs are provided per-image within a study

Training Strategy (from methodology):
- Use PRIMARY FRONTAL images (PA or AP view) for each study
- This ensures consistent visual features while maintaining VQA coverage
- Lateral and other views are available but filtered for training

=============================================================================
MIMIC-Ext-CXR-QBA SCENE GRAPH STRUCTURE (Correct Field Names)
=============================================================================
Scene graph JSON files contain:

- **observations**: Dict of findings, each with:
  - `name`: Short name of finding
  - `summary_sentence`: Full sentence description
  - `regions`: [{"region": "lungs", "distances": []}] - CORRECT field name
  - `obs_entities`: ["consolidation", "opacity"] - list of entity strings
  - `obs_categories`: ["ANATOMICAL_FINDING", "DISEASE"]
  - `positiveness`: "pos" | "neg" | "unknown"
  - `localization`: {image_id: {"bboxes": [[x1,y1,x2,y2], ...]}}

- **located_at_relations**: Observation -> Region connections
- **obs_relations**: Parent-child observation relationships
- **obs_sent_relations**: Observation -> Sentence connections
- **region_region_relations**: Anatomical hierarchy (sub_region, left/right)

- **regions**: Dict of anatomical regions with bboxes
- **sentences**: Dict of report sentences with section info
- **indication**: Clinical indication with patient info

=============================================================================

Analysis includes:
- Full metadata column analysis for all CSV files
- CheXpert structured label distribution
- Image metadata (ViewPosition, dimensions, procedures)
- Scene graph structure and quality with CORRECT field parsing
- QA pair distribution and complexity
- Cross-dataset correlation
- Patient-level comprehensive profiles
- Visual scene graph representations (pipeline-style network graphs)

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

# Suppress noisy warnings
warnings.filterwarnings('ignore', message='.*categorical units.*')
warnings.filterwarnings('ignore', message='.*parsable as floats.*')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Optional visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    sns.set_style("whitegrid")
    
    # Suppress matplotlib logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
    
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
    
    # ===================== Multi-Image Patient Handling =====================
    # MIMIC-CXR structure: Patient -> Studies -> Images
    # - Each patient can have multiple studies (longitudinal exams)
    # - Each study can have multiple images (different views: PA, AP, lateral)
    # - Scene graphs and QA pairs are at the STUDY level, not image level
    # - Bounding boxes in scene graphs are provided per-image within a study
    multi_image_stats: Dict[str, Any] = field(default_factory=lambda: {
        'patients_single_study': 0,      # Patients with exactly 1 study
        'patients_multi_study': 0,       # Patients with >1 studies
        'studies_single_image': 0,       # Studies with exactly 1 image
        'studies_multi_image': 0,        # Studies with >1 images
        'avg_studies_per_patient': 0.0,
        'avg_images_per_study': 0.0,
        'max_studies_per_patient': 0,
        'max_images_per_study': 0,
        'handling_strategy': 'frontal_primary',  # Use primary frontal view
    })
    
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
            
            # ==================== MULTI-IMAGE PATIENT ANALYSIS ====================
            # MIMIC-CXR structure: Patient -> Studies -> Images
            # Scene graphs and QA pairs are at the STUDY level
            # Bounding boxes are provided per-image within each study
            logger.info(f"\n  📊 Multi-Image Patient Analysis:")
            logger.info(f"  " + "=" * 60)
            
            if 'subject_id' in df.columns and 'study_id' in df.columns and 'dicom_id' in df.columns:
                # Count studies per patient
                studies_per_patient = df.groupby('subject_id')['study_id'].nunique()
                
                # Count images per study
                images_per_study = df.groupby('study_id')['dicom_id'].nunique()
                
                # Compute statistics
                patients_single_study = int((studies_per_patient == 1).sum())
                patients_multi_study = int((studies_per_patient > 1).sum())
                studies_single_image = int((images_per_study == 1).sum())
                studies_multi_image = int((images_per_study > 1).sum())
                avg_studies_per_patient = float(studies_per_patient.mean())
                avg_images_per_study = float(images_per_study.mean())
                max_studies_per_patient = int(studies_per_patient.max())
                max_images_per_study = int(images_per_study.max())
                
                # Update report
                self.report.multi_image_stats = {
                    'patients_single_study': patients_single_study,
                    'patients_multi_study': patients_multi_study,
                    'studies_single_image': studies_single_image,
                    'studies_multi_image': studies_multi_image,
                    'avg_studies_per_patient': round(avg_studies_per_patient, 2),
                    'avg_images_per_study': round(avg_images_per_study, 2),
                    'max_studies_per_patient': max_studies_per_patient,
                    'max_images_per_study': max_images_per_study,
                    'handling_strategy': 'frontal_primary',
                }
                
                logger.info(f"  PATIENTS:")
                logger.info(f"    With single study:   {patients_single_study:>10,} ({patients_single_study/(patients_single_study+patients_multi_study)*100:.1f}%)")
                logger.info(f"    With multiple studies: {patients_multi_study:>10,} ({patients_multi_study/(patients_single_study+patients_multi_study)*100:.1f}%)")
                logger.info(f"    Avg studies/patient: {avg_studies_per_patient:>10.2f}")
                logger.info(f"    Max studies/patient: {max_studies_per_patient:>10}")
                
                logger.info(f"\n  STUDIES:")
                logger.info(f"    With single image:   {studies_single_image:>10,} ({studies_single_image/(studies_single_image+studies_multi_image)*100:.1f}%)")
                logger.info(f"    With multiple images: {studies_multi_image:>10,} ({studies_multi_image/(studies_single_image+studies_multi_image)*100:.1f}%)")
                logger.info(f"    Avg images/study:    {avg_images_per_study:>10.2f}")
                logger.info(f"    Max images/study:    {max_images_per_study:>10}")
                
                logger.info(f"\n  HANDLING STRATEGY:")
                logger.info(f"    Scene graphs and QA pairs are at the STUDY level")
                logger.info(f"    Training uses PRIMARY FRONTAL image (PA or AP) per study")
                logger.info(f"    Bounding boxes in scene graphs are provided per-image")
                
                # Distribution of images per study
                img_dist = images_per_study.value_counts().sort_index()
                logger.info(f"\n  Images per Study Distribution:")
                for num_imgs, count in list(img_dist.items())[:7]:
                    pct = count / len(images_per_study) * 100
                    logger.info(f"    {num_imgs} image(s): {count:>10,} studies ({pct:>5.1f}%)")
                if len(img_dist) > 7:
                    logger.info(f"    ...")
            
        except Exception as e:
            logger.error(f"  Error loading image metadata: {e}")
            self.report.warnings.append(f"Image metadata error: {e}")
    
    def _analyze_qba_metadata_files(self):
        """Analyze all MIMIC-Ext-CXR-QBA metadata CSV files with FULL column analysis."""
        metadata_dir = self.mimic_qa_path / 'metadata'
        
        if not metadata_dir.exists():
            logger.warning(f"[!] QBA metadata directory not found: {metadata_dir}")
            return
        
        # List of expected metadata files with descriptions
        metadata_files = {
            'patient_metadata': 'Patient-level aggregated information',
            'study_metadata': 'Study-level metadata including quality metrics', 
            'image_metadata': 'Image-level view positions and dimensions',
            'question_metadata': 'Question types, strategies, and quality ratings',
            'answer_metadata': 'Answer types, entities, and localization info',
        }
        
        for base_name, description in metadata_files.items():
            parquet_path = metadata_dir / f"{base_name}.parquet"
            csv_path = metadata_dir / f"{base_name}.csv.gz"
            
            df = None
            source = None
            sampled = False
            
            # Try parquet first (faster)
            if parquet_path.exists():
                try:
                    df = pd.read_parquet(parquet_path)
                    source = parquet_path.name
                except Exception:
                    pass
            
            # Fall back to CSV
            if df is None and csv_path.exists():
                try:
                    # Sample large files
                    df = pd.read_csv(csv_path, compression='gzip', nrows=500000)
                    source = csv_path.name
                    sampled = True
                except Exception as e:
                    logger.warning(f"    Error loading {csv_path.name}: {e}")
                    continue
            
            if df is not None:
                self._analyze_metadata_columns(source, df, description, sampled)
        
        # Analyze dataset_info.json comprehensively
        dataset_info_path = metadata_dir / 'dataset_info.json'
        if dataset_info_path.exists():
            self._analyze_dataset_info_json(dataset_info_path)
    
    def _analyze_metadata_columns(self, filename: str, df: pd.DataFrame, description: str, sampled: bool = False):
        """Perform comprehensive column-by-column analysis of a metadata file."""
        sample_note = " (sampled)" if sampled else ""
        
        logger.info(f"\n  {'='*60}")
        logger.info(f"  [METADATA] {filename}{sample_note}")
        logger.info(f"  {description}")
        logger.info(f"  {'='*60}")
        logger.info(f"  Shape: {len(df):,} rows x {len(df.columns)} columns")
        logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Store comprehensive info
        file_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'sampled': sampled,
            'column_analysis': {}
        }
        
        # Column-by-column analysis
        logger.info(f"\n  {'Column':<45} {'Type':<12} {'Non-Null':>10} {'Unique':>10} {'Analysis'}")
        logger.info(f"  {'-'*110}")
        
        for col in df.columns:
            col_analysis = self._analyze_single_column(df, col)
            file_info['column_analysis'][col] = col_analysis
            
            # Truncate column name for display
            col_display = col[:44] + '..' if len(col) > 44 else col
            
            logger.info(f"  {col_display:<45} {col_analysis['dtype']:<12} {col_analysis['non_null_count']:>10,} {col_analysis['unique_count']:>10,} {col_analysis['summary'][:40]}")
        
        self.report.metadata_csv_info[filename] = file_info
        
        # Show key distributions for important columns
        self._show_key_distributions(filename, df)
        
        # Generate HOLISTIC CSV visualization (not per-column)
        self._create_csv_overview_visualization(filename, df, description)
    
    def _analyze_single_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze a single column comprehensively based on its data type."""
        dtype = str(df[col].dtype)
        non_null = int(df[col].notna().sum())
        null_pct = (1 - non_null / len(df)) * 100 if len(df) > 0 else 0
        n_unique = int(df[col].nunique())
        
        analysis = {
            'dtype': dtype,
            'non_null_count': non_null,
            'null_count': int(len(df) - non_null),
            'null_percentage': round(null_pct, 2),
            'unique_count': n_unique,
            'unique_ratio': round(n_unique / non_null, 4) if non_null > 0 else 0,
            'summary': '',
            'statistics': {},
            'value_distribution': {},
            'data_quality': {}
        }
        
        try:
            # Determine column type and analyze accordingly
            if df[col].dtype == 'bool':
                # Boolean analysis
                true_count = int(df[col].sum())
                false_count = int((~df[col]).sum())
                true_pct = true_count / non_null * 100 if non_null > 0 else 0
                
                analysis['statistics'] = {
                    'true_count': true_count,
                    'false_count': false_count,
                    'true_percentage': round(true_pct, 2)
                }
                analysis['value_distribution'] = {'True': true_count, 'False': false_count}
                analysis['summary'] = f"True: {true_pct:.1f}%, False: {100-true_pct:.1f}%"
                analysis['column_type'] = 'boolean'
                
            elif df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                # Integer analysis
                stats = df[col].describe()
                analysis['statistics'] = {
                    'min': int(stats['min']),
                    'max': int(stats['max']),
                    'mean': round(float(stats['mean']), 2),
                    'std': round(float(stats['std']), 2),
                    'median': int(stats['50%']),
                    'q25': int(stats['25%']),
                    'q75': int(stats['75%']),
                }
                
                # Check if it's a small set of integers (categorical-like)
                if n_unique <= 20:
                    value_counts = df[col].value_counts().head(10)
                    analysis['value_distribution'] = {str(k): int(v) for k, v in value_counts.items()}
                    analysis['column_type'] = 'integer_categorical'
                    analysis['summary'] = f"min={stats['min']:.0f}, max={stats['max']:.0f}, {n_unique} unique"
                else:
                    analysis['column_type'] = 'integer_continuous'
                    analysis['summary'] = f"range=[{stats['min']:.0f}, {stats['max']:.0f}], mean={stats['mean']:.1f}"
                
            elif df[col].dtype in ['float64', 'float32', 'float16']:
                # Float analysis
                stats = df[col].describe()
                analysis['statistics'] = {
                    'min': round(float(stats['min']), 4),
                    'max': round(float(stats['max']), 4),
                    'mean': round(float(stats['mean']), 4),
                    'std': round(float(stats['std']), 4),
                    'median': round(float(stats['50%']), 4),
                    'q25': round(float(stats['25%']), 4),
                    'q75': round(float(stats['75%']), 4),
                }
                analysis['column_type'] = 'float_continuous'
                analysis['summary'] = f"range=[{stats['min']:.2f}, {stats['max']:.2f}], mean={stats['mean']:.2f}"
                
            elif df[col].dtype == 'object' or str(df[col].dtype).startswith('str') or str(df[col].dtype) == 'category':
                # String/categorical analysis
                value_counts = df[col].value_counts()
                top_n = min(20, n_unique)
                top_values = value_counts.head(top_n)
                
                analysis['value_distribution'] = {str(k): int(v) for k, v in top_values.items()}
                analysis['statistics'] = {
                    'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'most_common_pct': round(value_counts.iloc[0] / non_null * 100, 2) if non_null > 0 and len(value_counts) > 0 else 0,
                }
                
                # String length stats
                str_lengths = df[col].dropna().astype(str).str.len()
                if len(str_lengths) > 0:
                    analysis['statistics']['avg_string_length'] = round(float(str_lengths.mean()), 1)
                    analysis['statistics']['max_string_length'] = int(str_lengths.max())
                
                if n_unique <= 10:
                    analysis['column_type'] = 'string_categorical_low'
                    vals = ', '.join([f"'{str(k)[:15]}'" for k in list(value_counts.index)[:3]])
                    analysis['summary'] = f"{n_unique} values: {vals}"
                elif n_unique <= 50:
                    analysis['column_type'] = 'string_categorical_medium'
                    analysis['summary'] = f"{n_unique} unique, top='{str(value_counts.index[0])[:20]}'"
                else:
                    analysis['column_type'] = 'string_high_cardinality'
                    analysis['summary'] = f"{n_unique:,} unique values"
                    
            else:
                # Other types (datetime, etc.)
                analysis['column_type'] = 'other'
                analysis['summary'] = f"{dtype}, {n_unique:,} unique"
                
            # Data quality checks
            analysis['data_quality'] = {
                'has_nulls': null_pct > 0,
                'null_severity': 'high' if null_pct > 20 else 'medium' if null_pct > 5 else 'low',
                'is_constant': n_unique == 1,
                'is_unique': n_unique == non_null,
            }
                
        except Exception as e:
            analysis['summary'] = f"Error: {str(e)[:30]}"
            analysis['column_type'] = 'error'
        
        return analysis
    
    def _create_csv_overview_visualization(self, filename: str, df: pd.DataFrame, description: str):
        """Create ONE holistic visualization per CSV file showing all key comparisons.
        
        This creates a single, meaningful visualization that shows the entire dataset
        in context rather than fragmented per-column plots.
        """
        if not PLOTTING_AVAILABLE:
            return
        
        try:
            viz_dir = self.output_dir / 'csv_overviews'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            filename_lower = filename.lower()
            safe_name = filename.replace('.', '_').replace('/', '_')[:40]
            output_path = viz_dir / f'{safe_name}_overview.png'
            
            # Route to specialized holistic visualizations
            if 'patient' in filename_lower:
                self._create_patient_overview(df, output_path, description)
            elif 'study' in filename_lower:
                self._create_study_overview(df, output_path, description)
            elif 'question' in filename_lower:
                self._create_question_overview(df, output_path, description)
            elif 'answer' in filename_lower:
                self._create_answer_overview(df, output_path, description)
            elif 'image' in filename_lower:
                self._create_image_overview(df, output_path, description)
            elif 'chexpert' in filename_lower:
                self._create_chexpert_overview(df, output_path, description)
            elif 'split' in filename_lower:
                self._create_split_overview(df, output_path, description)
            elif 'metadata' in filename_lower and 'mimic-cxr' in filename_lower:
                self._create_mimic_metadata_overview(df, output_path, description)
            else:
                self._create_generic_overview(filename, df, output_path, description)
            
            logger.info(f"  [VIZ] CSV overview saved: {output_path.name}")
            
        except Exception as e:
            logger.warning(f"  [VIZ] Error creating CSV overview for {filename}: {e}")
    
    def _create_patient_overview(self, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Create meaningful patient-level insights visualization."""
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Data Split Distribution (Critical for ML)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'split' in df.columns:
            split_counts = df['split'].value_counts()
            colors = {'train': '#27ae60', 'validate': '#3498db', 'test': '#e74c3c'}
            bar_colors = [colors.get(s, '#95a5a6') for s in split_counts.index]
            bars = ax1.bar(split_counts.index, split_counts.values, color=bar_colors, edgecolor='white', linewidth=2)
            ax1.set_ylabel('Number of Patients', fontsize=11)
            ax1.set_title('Train / Validation / Test Split\n(Patient-Level)', fontsize=12, fontweight='bold')
            
            # Add count and percentage labels
            total = split_counts.sum()
            for bar, count in zip(bars, split_counts.values):
                pct = count / total * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                        f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax1.set_ylim(0, max(split_counts.values) * 1.15)
        else:
            ax1.text(0.5, 0.5, 'No split column', ha='center', va='center', fontsize=12)
            ax1.set_title('Data Split', fontsize=12, fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Studies per Patient Distribution (Important for understanding data)
        ax2 = fig.add_subplot(gs[0, 1])
        studies_col = [c for c in df.columns if 'total_studies' in c.lower() or 'study_count' in c.lower()]
        if studies_col:
            col = studies_col[0]
            study_data = df[col].dropna()
            
            # Create bins for cleaner histogram
            max_studies = min(study_data.max(), 30)  # Cap at 30 for readability
            bins = list(range(0, int(max_studies) + 2))
            
            ax2.hist(study_data.clip(upper=max_studies), bins=bins, color='#3498db', 
                    edgecolor='white', alpha=0.8)
            ax2.axvline(study_data.median(), color='#e74c3c', linestyle='--', linewidth=2, 
                       label=f'Median: {study_data.median():.0f}')
            ax2.axvline(study_data.mean(), color='#f39c12', linestyle=':', linewidth=2,
                       label=f'Mean: {study_data.mean():.1f}')
            ax2.set_xlabel('Number of Studies per Patient', fontsize=11)
            ax2.set_ylabel('Number of Patients', fontsize=11)
            ax2.set_title('Studies per Patient Distribution\n(How many X-rays per patient?)', 
                         fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            
            # Add insight text
            single_study = (study_data == 1).sum()
            multi_study = (study_data > 1).sum()
            insight = f"Single study: {single_study:,} ({single_study/len(study_data)*100:.1f}%)\n"
            insight += f"Multiple studies: {multi_study:,} ({multi_study/len(study_data)*100:.1f}%)"
            ax2.text(0.95, 0.95, insight, transform=ax2.transAxes, fontsize=9,
                    va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            ax2.text(0.5, 0.5, 'No study count column', ha='center', va='center', fontsize=12)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Key Statistics Summary
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        stats_text = "PATIENT DATASET SUMMARY\n" + "=" * 35 + "\n\n"
        stats_text += f"Total Patients: {len(df):,}\n\n"
        
        if 'split' in df.columns:
            for split in ['train', 'validate', 'test']:
                count = (df['split'] == split).sum()
                stats_text += f"  {split.title()}: {count:,}\n"
        
        stats_text += "\n"
        if studies_col:
            col = studies_col[0]
            stats_text += f"Studies per Patient:\n"
            stats_text += f"  Min: {df[col].min():.0f}\n"
            stats_text += f"  Max: {df[col].max():.0f}\n"
            stats_text += f"  Mean: {df[col].mean():.2f}\n"
            stats_text += f"  Median: {df[col].median():.0f}\n"
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=11,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        ax3.set_title('Key Statistics', fontsize=12, fontweight='bold')
        
        # 4. Data Quality Indicators
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Calculate quality metrics for each column
        quality_data = []
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            unique_ratio = df[col].nunique() / len(df)
            quality_data.append({
                'column': col.split('.')[-1][:20],
                'null_pct': null_pct,
                'unique_ratio': unique_ratio
            })
        
        quality_df = pd.DataFrame(quality_data)
        
        x = np.arange(len(quality_df))
        width = 0.6
        
        # Color by null percentage
        colors = ['#27ae60' if p < 1 else '#f39c12' if p < 10 else '#e74c3c' for p in quality_df['null_pct']]
        bars = ax4.bar(x, quality_df['null_pct'], width, color=colors, edgecolor='white')
        
        ax4.set_ylabel('Missing Values (%)', fontsize=11)
        ax4.set_title('Data Quality: Missing Values by Column\n(Green=<1%, Yellow=<10%, Red=>10%)', 
                     fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(quality_df['column'], rotation=45, ha='right', fontsize=9)
        ax4.axhline(1, color='green', linestyle='--', alpha=0.5)
        ax4.axhline(10, color='orange', linestyle='--', alpha=0.5)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Interpretation/Insights Panel
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        insights = "KEY INSIGHTS\n" + "=" * 30 + "\n\n"
        
        # Generate insights based on data
        if 'split' in df.columns:
            train_pct = (df['split'] == 'train').sum() / len(df) * 100
            if train_pct > 90:
                insights += "[!] Train set is >90% of data\n    Consider if this is intentional\n\n"
            else:
                insights += "[OK] Reasonable train/test split\n\n"
        
        if studies_col:
            col = studies_col[0]
            if df[col].max() > 50:
                insights += "[!] Some patients have 50+ studies\n    May need handling for imbalance\n\n"
            if df[col].median() == 1:
                insights += "[INFO] Most patients have single study\n    Limited longitudinal data\n\n"
        
        # Check for data quality issues
        high_null_cols = [c for c in df.columns if df[c].isnull().sum() / len(df) > 0.1]
        if high_null_cols:
            insights += f"[WARN] {len(high_null_cols)} columns have >10% nulls\n\n"
        else:
            insights += "[OK] Good data completeness\n\n"
        
        ax5.text(0.1, 0.9, insights, transform=ax5.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#f0ad4e'))
        ax5.set_title('Interpretation', fontsize=12, fontweight='bold')
        
        fig.suptitle('Patient Metadata Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_study_overview(self, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Create meaningful study-level insights visualization."""
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Quality Rating Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        quality_cols = [c for c in df.columns if 'quality' in c.lower() and 'rating' in c.lower()]
        if quality_cols:
            col = quality_cols[0]
            value_counts = df[col].value_counts().sort_index()
            
            # Color by quality level
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(value_counts)))
            bars = ax1.bar(value_counts.index.astype(str), value_counts.values, color=colors, edgecolor='white')
            ax1.set_xlabel('Quality Rating', fontsize=11)
            ax1.set_ylabel('Number of Studies', fontsize=11)
            ax1.set_title('Study Quality Distribution\n(Higher = Better)', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            total = value_counts.sum()
            for bar, count in zip(bars, value_counts.values):
                if count / total > 0.05:  # Only label if >5%
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                            f'{count/total*100:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No quality rating column', ha='center', va='center')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Observations per Study
        ax2 = fig.add_subplot(gs[0, 1])
        obs_cols = [c for c in df.columns if 'observation' in c.lower() and 'count' in c.lower()]
        if obs_cols:
            col = obs_cols[0]
            obs_data = df[col].dropna()
            
            ax2.hist(obs_data.clip(upper=30), bins=30, color='#9b59b6', edgecolor='white', alpha=0.8)
            ax2.axvline(obs_data.median(), color='#e74c3c', linestyle='--', linewidth=2,
                       label=f'Median: {obs_data.median():.0f}')
            ax2.set_xlabel('Observations per Study', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Clinical Observations per Study\n(Scene Graph Complexity)', fontsize=12, fontweight='bold')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No observation count column', ha='center', va='center')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. QA Pairs per Study
        ax3 = fig.add_subplot(gs[0, 2])
        qa_cols = [c for c in df.columns if 'qa' in c.lower() and 'count' in c.lower()]
        if qa_cols:
            col = qa_cols[0]
            qa_data = df[col].dropna()
            
            ax3.hist(qa_data.clip(upper=200), bins=50, color='#1abc9c', edgecolor='white', alpha=0.8)
            ax3.axvline(qa_data.median(), color='#e74c3c', linestyle='--', linewidth=2,
                       label=f'Median: {qa_data.median():.0f}')
            ax3.set_xlabel('QA Pairs per Study', fontsize=11)
            ax3.set_ylabel('Frequency', fontsize=11)
            ax3.set_title('VQA Pairs per Study\n(Training Data Density)', fontsize=12, fontweight='bold')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No QA count column', ha='center', va='center')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Image Count per Study
        ax4 = fig.add_subplot(gs[1, 0])
        img_cols = [c for c in df.columns if 'image' in c.lower() and 'count' in c.lower()]
        if img_cols:
            col = img_cols[0]
            img_data = df[col].value_counts().sort_index()
            
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(img_data)))
            bars = ax4.bar(img_data.index.astype(str), img_data.values, color=colors, edgecolor='white')
            ax4.set_xlabel('Number of Images', fontsize=11)
            ax4.set_ylabel('Number of Studies', fontsize=11)
            ax4.set_title('Images per Study\n(Multi-view Studies)', fontsize=12, fontweight='bold')
            
            total = img_data.sum()
            for bar, count in zip(bars, img_data.values):
                if count / total > 0.05:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{count/total*100:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'No image count column', ha='center', va='center')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Study Statistics Summary
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        
        summary = "STUDY DATASET SUMMARY\n" + "=" * 35 + "\n\n"
        summary += f"Total Studies: {len(df):,}\n\n"
        
        # Add statistics for key columns
        for col_type, pattern in [('Quality', 'quality'), ('Observations', 'observation'), ('QA Pairs', 'qa')]:
            matching = [c for c in df.columns if pattern in c.lower() and df[c].dtype in ['int64', 'float64']]
            if matching:
                col = matching[0]
                summary += f"{col_type}:\n"
                summary += f"  Mean: {df[col].mean():.1f}\n"
                summary += f"  Median: {df[col].median():.0f}\n"
                summary += f"  Range: [{df[col].min():.0f}, {df[col].max():.0f}]\n\n"
        
        ax5.text(0.1, 0.9, summary, transform=ax5.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        ax5.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        # 6. Training Implications
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        implications = "TRAINING IMPLICATIONS\n" + "=" * 30 + "\n\n"
        
        if quality_cols:
            high_quality = (df[quality_cols[0]].astype(str).str.contains('A|5|6', na=False)).sum()
            implications += f"[INFO] High-quality studies:\n       {high_quality:,} ({high_quality/len(df)*100:.1f}%)\n\n"
        
        if qa_cols:
            col = qa_cols[0]
            low_qa = (df[col] < 10).sum()
            if low_qa > len(df) * 0.1:
                implications += f"[WARN] {low_qa:,} studies have <10 QA pairs\n       Consider filtering or weighting\n\n"
        
        if obs_cols:
            col = obs_cols[0]
            if df[col].median() < 5:
                implications += "[INFO] Low observation density\n       Simpler scene graphs\n\n"
            else:
                implications += "[OK] Good observation density\n       Rich scene graph data\n\n"
        
        ax6.text(0.1, 0.9, implications, transform=ax6.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#f0ad4e'))
        ax6.set_title('Training Notes', fontsize=12, fontweight='bold')
        
        fig.suptitle('Study Metadata Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_question_overview(self, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Create meaningful question-level insights visualization."""
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. Question Type Distribution (Most Important)
        ax1 = fig.add_subplot(gs[0, :2])
        type_cols = [c for c in df.columns if 'question_type' in c.lower() or 'type' in c.lower()]
        if type_cols:
            col = type_cols[0]
            type_counts = df[col].value_counts().head(15)
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(type_counts)))
            bars = ax1.barh(type_counts.index[::-1], type_counts.values[::-1], color=colors[::-1], edgecolor='white')
            ax1.set_xlabel('Number of Questions', fontsize=11)
            ax1.set_title('Question Type Distribution\n(What kinds of questions are in the dataset?)', 
                         fontsize=12, fontweight='bold')
            
            # Add percentage labels
            total = df[col].count()
            for bar, count in zip(bars, type_counts.values[::-1]):
                pct = count / total * 100
                ax1.text(bar.get_width() + total*0.005, bar.get_y() + bar.get_height()/2,
                        f'{pct:.1f}%', va='center', fontsize=9)
            ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        else:
            ax1.text(0.5, 0.5, 'No question type column', ha='center', va='center')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Question Statistics
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        stats = "QUESTION STATISTICS\n" + "=" * 30 + "\n\n"
        stats += f"Total Questions: {len(df):,}\n\n"
        
        if type_cols:
            col = type_cols[0]
            stats += f"Question Types: {df[col].nunique()}\n\n"
            stats += "Top 5 Types:\n"
            for qtype, count in df[col].value_counts().head(5).items():
                stats += f"  {str(qtype)[:25]}: {count:,}\n"
        
        ax2.text(0.1, 0.9, stats, transform=ax2.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        ax2.set_title('Statistics', fontsize=12, fontweight='bold')
        
        # 3. Boolean Features (e.g., has_region, has_finding)
        ax3 = fig.add_subplot(gs[1, 0])
        bool_cols = [c for c in df.columns if df[c].dtype == 'bool'][:6]
        
        if bool_cols:
            true_pcts = [(df[c].sum() / len(df) * 100) for c in bool_cols]
            short_names = [c.split('.')[-1][:18] for c in bool_cols]
            
            colors = ['#27ae60' if p > 50 else '#e74c3c' for p in true_pcts]
            bars = ax3.barh(short_names, true_pcts, color=colors, edgecolor='white')
            ax3.set_xlabel('Percentage True (%)', fontsize=11)
            ax3.set_title('Question Features\n(What information is available?)', fontsize=12, fontweight='bold')
            ax3.set_xlim(0, 100)
            ax3.axvline(50, color='gray', linestyle='--', alpha=0.5)
            
            for bar, pct in zip(bars, true_pcts):
                ax3.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                        f'{pct:.1f}%', va='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No boolean columns', ha='center', va='center')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Quality Indicators
        ax4 = fig.add_subplot(gs[1, 1])
        quality_cols = [c for c in df.columns if 'quality' in c.lower()][:4]
        
        if quality_cols:
            quality_data = []
            for col in quality_cols:
                if df[col].dtype in ['int64', 'float64']:
                    quality_data.append({
                        'name': col.split('.')[-1][:15],
                        'mean': df[col].mean(),
                        'max': df[col].max()
                    })
            
            if quality_data:
                names = [d['name'] for d in quality_data]
                means = [d['mean'] for d in quality_data]
                maxes = [d['max'] for d in quality_data]
                
                x = np.arange(len(names))
                width = 0.4
                
                ax4.bar(x - width/2, means, width, label='Mean', color='#3498db', edgecolor='white')
                ax4.bar(x + width/2, maxes, width, label='Max', color='#95a5a6', edgecolor='white', alpha=0.7)
                
                ax4.set_ylabel('Value', fontsize=11)
                ax4.set_title('Quality Metrics\n(Higher = Better)', fontsize=12, fontweight='bold')
                ax4.set_xticks(x)
                ax4.set_xticklabels(names, rotation=45, ha='right')
                ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No quality columns', ha='center', va='center')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Training Recommendations
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        rec = "TRAINING RECOMMENDATIONS\n" + "=" * 30 + "\n\n"
        
        if type_cols:
            col = type_cols[0]
            type_dist = df[col].value_counts()
            max_type_pct = type_dist.iloc[0] / len(df) * 100
            
            if max_type_pct > 30:
                rec += f"[WARN] Dominant question type:\n       {type_dist.index[0][:20]}\n       ({max_type_pct:.1f}%)\n\n"
                rec += "       Consider balanced sampling\n\n"
            else:
                rec += "[OK] Good question type balance\n\n"
        
        if bool_cols:
            low_true = [c for c in bool_cols if df[c].sum() / len(df) < 0.1]
            if low_true:
                rec += f"[INFO] {len(low_true)} features are rare (<10%)\n"
                rec += "       May need oversampling\n\n"
        
        ax5.text(0.1, 0.9, rec, transform=ax5.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#f0ad4e'))
        ax5.set_title('Recommendations', fontsize=12, fontweight='bold')
        
        fig.suptitle('Question Metadata Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_answer_overview(self, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Create meaningful answer-level insights visualization."""
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. Answer Quality Rating Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        rating_cols = [c for c in df.columns if 'rating' in c.lower() and 'quality' in c.lower()]
        if rating_cols:
            col = rating_cols[0]
            rating_counts = df[col].value_counts().sort_index()
            
            # Color gradient from red to green
            n_ratings = len(rating_counts)
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n_ratings))
            
            bars = ax1.bar(rating_counts.index.astype(str), rating_counts.values, color=colors, edgecolor='white')
            ax1.set_xlabel('Quality Rating', fontsize=11)
            ax1.set_ylabel('Number of Answers', fontsize=11)
            ax1.set_title('Answer Quality Distribution\n(D=Low, A++=High)', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No rating column', ha='center', va='center')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Answer Type Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        type_cols = [c for c in df.columns if 'type' in c.lower() and 'answer' in c.lower()]
        if type_cols:
            col = type_cols[0]
            type_counts = df[col].value_counts()
            
            colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12'][:len(type_counts)]
            wedges, texts, autotexts = ax2.pie(
                type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                colors=colors, wedgeprops=dict(width=0.6, edgecolor='white'),
                textprops={'fontsize': 10}
            )
            ax2.set_title('Answer Type Distribution\n(Binary vs. Category)', fontsize=12, fontweight='bold')
            
            # Center text
            ax2.text(0, 0, f'{len(df):,}\nanswers', ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No answer type column', ha='center', va='center')
        
        # 3. Quality Sub-metrics
        ax3 = fig.add_subplot(gs[0, 2])
        quality_int_cols = [c for c in df.columns if 'quality' in c.lower() and df[c].dtype in ['int64', 'int32']][:5]
        
        if quality_int_cols:
            means = [df[c].mean() for c in quality_int_cols]
            short_names = [c.split('.')[-1].replace('_quality', '')[:12] for c in quality_int_cols]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(means)))
            bars = ax3.barh(short_names, means, color=colors, edgecolor='white')
            ax3.set_xlabel('Mean Quality Score', fontsize=11)
            ax3.set_title('Quality Sub-metrics\n(Detailed Quality Breakdown)', fontsize=12, fontweight='bold')
            
            for bar, mean in zip(bars, means):
                ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                        f'{mean:.2f}', va='center', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'No quality sub-metrics', ha='center', va='center')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Boolean Features Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        bool_cols = [c for c in df.columns if df[c].dtype == 'bool'][:6]
        
        if bool_cols:
            data = {'True': [], 'False': []}
            names = []
            for col in bool_cols:
                true_pct = df[col].sum() / len(df) * 100
                data['True'].append(true_pct)
                data['False'].append(100 - true_pct)
                names.append(col.split('.')[-1][:15])
            
            x = np.arange(len(names))
            width = 0.6
            
            ax4.barh(names, data['True'], width, label='True', color='#27ae60', edgecolor='white')
            ax4.barh(names, data['False'], width, left=data['True'], label='False', color='#e74c3c', edgecolor='white')
            
            ax4.set_xlabel('Percentage (%)', fontsize=11)
            ax4.set_title('Answer Features\n(True/False Split)', fontsize=12, fontweight='bold')
            ax4.legend(loc='lower right')
            ax4.set_xlim(0, 100)
        else:
            ax4.text(0.5, 0.5, 'No boolean features', ha='center', va='center')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Statistics Summary
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        
        stats = "ANSWER STATISTICS\n" + "=" * 30 + "\n\n"
        stats += f"Total Answers: {len(df):,}\n\n"
        
        if rating_cols:
            col = rating_cols[0]
            high_quality = df[col].astype(str).str.contains('A', na=False).sum()
            stats += f"High Quality (A+):\n  {high_quality:,} ({high_quality/len(df)*100:.1f}%)\n\n"
        
        if type_cols:
            col = type_cols[0]
            for atype, count in df[col].value_counts().items():
                stats += f"{atype}: {count:,}\n"
        
        ax5.text(0.1, 0.9, stats, transform=ax5.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        ax5.set_title('Summary', fontsize=12, fontweight='bold')
        
        # 6. Recommendations
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        rec = "RECOMMENDATIONS\n" + "=" * 30 + "\n\n"
        
        if rating_cols:
            col = rating_cols[0]
            low_quality = df[col].astype(str).str.contains('D|C|0_|1_', na=False).sum()
            low_pct = low_quality / len(df) * 100
            
            if low_pct > 10:
                rec += f"[WARN] {low_pct:.1f}% low-quality answers\n       Filter with quality_grade='A'\n\n"
            else:
                rec += "[OK] Good answer quality overall\n\n"
        
        if bool_cols:
            has_entity = [c for c in bool_cols if 'entit' in c.lower()]
            if has_entity:
                entity_pct = df[has_entity[0]].sum() / len(df) * 100
                rec += f"[INFO] {entity_pct:.1f}% have entities\n       Good for entity-aware models\n\n"
        
        ax6.text(0.1, 0.9, rec, transform=ax6.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#f0ad4e'))
        ax6.set_title('Recommendations', fontsize=12, fontweight='bold')
        
        fig.suptitle('Answer Metadata Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_image_overview(self, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Create meaningful image-level insights visualization."""
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. View Position Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        view_cols = [c for c in df.columns if 'view' in c.lower()]
        if view_cols:
            col = view_cols[0]
            view_counts = df[col].value_counts().head(6)
            
            colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']
            wedges, texts, autotexts = ax1.pie(
                view_counts.values, labels=view_counts.index, autopct='%1.1f%%',
                colors=colors[:len(view_counts)], 
                wedgeprops=dict(edgecolor='white', linewidth=2),
                textprops={'fontsize': 10}
            )
            ax1.set_title('X-Ray View Positions\n(AP, PA, Lateral, etc.)', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No view column', ha='center', va='center')
        
        # 2. Image Dimensions (Height)
        ax2 = fig.add_subplot(gs[0, 1])
        height_cols = [c for c in df.columns if 'height' in c.lower() or 'rows' in c.lower()]
        width_cols = [c for c in df.columns if 'width' in c.lower() or 'columns' in c.lower()]
        
        if height_cols:
            col = height_cols[0]
            height_data = df[col].dropna()
            
            ax2.hist(height_data, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
            ax2.axvline(height_data.median(), color='#e74c3c', linestyle='--', linewidth=2,
                       label=f'Median: {height_data.median():.0f}px')
            ax2.set_xlabel('Image Height (pixels)', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Image Height Distribution\n(Vertical Resolution)', fontsize=12, fontweight='bold')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No height column', ha='center', va='center')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Image Dimensions (Width)
        ax3 = fig.add_subplot(gs[0, 2])
        if width_cols:
            col = width_cols[0]
            width_data = df[col].dropna()
            
            ax3.hist(width_data, bins=50, color='#27ae60', edgecolor='white', alpha=0.8)
            ax3.axvline(width_data.median(), color='#e74c3c', linestyle='--', linewidth=2,
                       label=f'Median: {width_data.median():.0f}px')
            ax3.set_xlabel('Image Width (pixels)', fontsize=11)
            ax3.set_ylabel('Frequency', fontsize=11)
            ax3.set_title('Image Width Distribution\n(Horizontal Resolution)', fontsize=12, fontweight='bold')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No width column', ha='center', va='center')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Aspect Ratio / 2D Scatter
        ax4 = fig.add_subplot(gs[1, 0])
        if height_cols and width_cols:
            h_col = height_cols[0]
            w_col = width_cols[0]
            
            # Sample for performance
            sample_size = min(5000, len(df))
            sample_idx = np.random.choice(len(df), sample_size, replace=False)
            h_sample = df[h_col].iloc[sample_idx]
            w_sample = df[w_col].iloc[sample_idx]
            
            ax4.scatter(w_sample, h_sample, alpha=0.3, s=5, c='#3498db')
            ax4.set_xlabel('Width (pixels)', fontsize=11)
            ax4.set_ylabel('Height (pixels)', fontsize=11)
            ax4.set_title('Image Dimensions Scatter\n(Width vs Height)', fontsize=12, fontweight='bold')
            
            # Add 1:1 line
            max_dim = max(df[h_col].max(), df[w_col].max())
            ax4.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='1:1 ratio')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No dimension columns', ha='center', va='center')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Statistics Summary
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        
        stats = "IMAGE STATISTICS\n" + "=" * 30 + "\n\n"
        stats += f"Total Images: {len(df):,}\n\n"
        
        if view_cols:
            col = view_cols[0]
            stats += f"View Types: {df[col].nunique()}\n"
            frontal = df[col].isin(['AP', 'PA']).sum()
            stats += f"Frontal (AP/PA): {frontal:,}\n\n"
        
        if height_cols and width_cols:
            stats += f"Resolution:\n"
            stats += f"  Height: {df[height_cols[0]].mean():.0f} +/- {df[height_cols[0]].std():.0f}\n"
            stats += f"  Width: {df[width_cols[0]].mean():.0f} +/- {df[width_cols[0]].std():.0f}\n"
        
        ax5.text(0.1, 0.9, stats, transform=ax5.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        ax5.set_title('Summary', fontsize=12, fontweight='bold')
        
        # 6. Preprocessing Notes
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        notes = "PREPROCESSING NOTES\n" + "=" * 30 + "\n\n"
        
        if height_cols and width_cols:
            h_std = df[height_cols[0]].std()
            w_std = df[width_cols[0]].std()
            
            if h_std > 500 or w_std > 500:
                notes += "[INFO] High resolution variance\n       Resize to consistent size\n       (e.g., 224x224 for model)\n\n"
            else:
                notes += "[OK] Consistent image sizes\n\n"
        
        if view_cols:
            lateral = df[view_cols[0]].isin(['LATERAL', 'LL']).sum()
            lateral_pct = lateral / len(df) * 100
            
            if lateral_pct > 20:
                notes += f"[INFO] {lateral_pct:.1f}% lateral views\n       Consider view-specific models\n       or frontal-only filtering\n\n"
            else:
                notes += "[OK] Mostly frontal views\n\n"
        
        ax6.text(0.1, 0.9, notes, transform=ax6.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#f0ad4e'))
        ax6.set_title('Notes', fontsize=12, fontweight='bold')
        
        fig.suptitle('Image Metadata Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_generic_overview(self, filename: str, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Fallback visualization for unknown metadata files."""
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Column types
        ax1 = fig.add_subplot(gs[0, 0])
        dtype_counts = df.dtypes.astype(str).value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(dtype_counts)))
        ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Column Data Types', fontweight='bold')
        
        # 2. Missing values
        ax2 = fig.add_subplot(gs[0, 1])
        null_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(10)
        colors = ['#e74c3c' if p > 20 else '#f39c12' if p > 5 else '#27ae60' for p in null_pcts.values]
        ax2.barh([c[:20] for c in null_pcts.index], null_pcts.values, color=colors)
        ax2.set_xlabel('Missing Values (%)')
        ax2.set_title('Top 10 Columns with Missing Values', fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Summary stats
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        summary = f"FILE: {filename}\n" + "=" * 40 + "\n\n"
        summary += f"Rows: {len(df):,}\n"
        summary += f"Columns: {len(df.columns)}\n"
        summary += f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n\n"
        summary += f"Column Types:\n"
        for dtype, count in df.dtypes.astype(str).value_counts().items():
            summary += f"  {dtype}: {count}\n"
        
        ax3.text(0.1, 0.9, summary, transform=ax3.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1'))
        ax3.set_title('File Summary', fontweight='bold')
        
        # 4. Column names
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        cols_text = "COLUMNS\n" + "=" * 30 + "\n\n"
        for i, col in enumerate(df.columns[:15], 1):
            cols_text += f"{i:2d}. {col[:35]}\n"
        if len(df.columns) > 15:
            cols_text += f"\n... and {len(df.columns) - 15} more"
        
        ax4.text(0.1, 0.9, cols_text, transform=ax4.transAxes, fontsize=9,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax4.set_title('Column Names', fontweight='bold')
        
        fig.suptitle(f'{filename} Overview', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_chexpert_overview(self, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Create holistic CheXpert labels visualization - ALL 14 conditions in one comparative view."""
        # Get all CheXpert condition columns (exclude IDs)
        id_cols = ['subject_id', 'study_id']
        condition_cols = [c for c in df.columns if c not in id_cols]
        
        if not condition_cols:
            return
        
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], hspace=0.25, wspace=0.2)
        
        # 1. MAIN: Stacked bar chart comparing ALL conditions (Positive/Negative/Uncertain/Missing)
        ax1 = fig.add_subplot(gs[0, :])
        
        condition_stats = []
        for col in condition_cols:
            total = len(df)
            positive = (df[col] == 1.0).sum()
            negative = (df[col] == 0.0).sum()
            uncertain = (df[col] == -1.0).sum()
            missing = df[col].isna().sum()
            
            condition_stats.append({
                'condition': col,
                'positive': positive / total * 100,
                'negative': negative / total * 100,
                'uncertain': uncertain / total * 100,
                'missing': missing / total * 100,
                'pos_count': positive
            })
        
        # Sort by positive rate for better visualization
        condition_stats = sorted(condition_stats, key=lambda x: x['positive'], reverse=True)
        conditions = [s['condition'] for s in condition_stats]
        
        x = np.arange(len(conditions))
        width = 0.7
        
        positive = [s['positive'] for s in condition_stats]
        negative = [s['negative'] for s in condition_stats]
        uncertain = [s['uncertain'] for s in condition_stats]
        missing = [s['missing'] for s in condition_stats]
        
        # Stacked bars
        p1 = ax1.bar(x, positive, width, label='Positive (1.0)', color='#e74c3c', edgecolor='white')
        p2 = ax1.bar(x, negative, width, bottom=positive, label='Negative (0.0)', color='#27ae60', edgecolor='white')
        p3 = ax1.bar(x, uncertain, width, bottom=[p+n for p,n in zip(positive,negative)], 
                    label='Uncertain (-1.0)', color='#f39c12', edgecolor='white')
        p4 = ax1.bar(x, missing, width, bottom=[p+n+u for p,n,u in zip(positive,negative,uncertain)], 
                    label='Missing (NaN)', color='#95a5a6', edgecolor='white')
        
        ax1.set_ylabel('Percentage of Studies (%)', fontsize=12)
        ax1.set_title('CheXpert Label Distribution Across ALL 14 Conditions\n(Each bar = 100% of studies)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_ylim(0, 105)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add positive count labels on bars
        for i, (bar, stat) in enumerate(zip(p1, condition_stats)):
            if stat['positive'] > 5:
                ax1.text(bar.get_x() + bar.get_width()/2, stat['positive']/2, 
                        f"{stat['pos_count']:,}", ha='center', va='center', 
                        fontsize=8, color='white', fontweight='bold')
        
        # 2. Positive rate ranking (simpler view)
        ax2 = fig.add_subplot(gs[1, 0])
        pos_rates = [s['positive'] for s in condition_stats]
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(conditions)))
        
        bars = ax2.barh(conditions[::-1], pos_rates[::-1], color=colors[::-1], edgecolor='white')
        ax2.set_xlabel('Positive Rate (%)', fontsize=11)
        ax2.set_title('Conditions Ranked by Positive Rate\n(How common is each finding?)', fontsize=12, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, rate in zip(bars, pos_rates[::-1]):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{rate:.1f}%', va='center', fontsize=9)
        
        # 3. Summary and insights
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        summary = "CHEXPERT LABELS SUMMARY\n" + "=" * 40 + "\n\n"
        summary += f"Total Studies: {len(df):,}\n"
        summary += f"Conditions: {len(condition_cols)}\n\n"
        
        summary += "Top 5 Most Common Findings:\n"
        for s in condition_stats[:5]:
            summary += f"  {s['condition']}: {s['positive']:.1f}%\n"
        
        summary += "\nLeast Common Findings:\n"
        for s in condition_stats[-3:]:
            summary += f"  {s['condition']}: {s['positive']:.1f}%\n"
        
        # Calculate overall stats
        avg_positive = np.mean([s['positive'] for s in condition_stats])
        avg_missing = np.mean([s['missing'] for s in condition_stats])
        
        summary += f"\n--- Dataset Quality ---\n"
        summary += f"Avg Positive Rate: {avg_positive:.1f}%\n"
        summary += f"Avg Missing Rate: {avg_missing:.1f}%\n"
        
        if avg_missing > 50:
            summary += "\n[!] High missing rate - many unlabeled\n    Use uncertainty-aware training"
        
        ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        ax3.set_title('Summary & Insights', fontsize=12, fontweight='bold')
        
        fig.suptitle('CheXpert Labels: Complete 14-Condition Overview', fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_split_overview(self, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Create holistic train/val/test split visualization."""
        fig = plt.figure(figsize=(16, 8))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1])
        
        # Find split column
        split_col = None
        for col in df.columns:
            if 'split' in col.lower():
                split_col = col
                break
        
        if split_col is None:
            split_col = df.columns[0]  # Fallback
        
        split_counts = df[split_col].value_counts()
        
        # 1. Main bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        colors = {'train': '#27ae60', 'validate': '#3498db', 'test': '#e74c3c'}
        bar_colors = [colors.get(s.lower() if isinstance(s, str) else str(s), '#95a5a6') for s in split_counts.index]
        
        bars = ax1.bar(split_counts.index, split_counts.values, color=bar_colors, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
        
        # Add count and percentage labels
        total = split_counts.sum()
        for bar, (split, count) in zip(bars, split_counts.items()):
            pct = count / total * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                    f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax1.set_ylim(0, max(split_counts.values) * 1.15)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # 2. Summary
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        summary = "SPLIT SUMMARY\n" + "=" * 30 + "\n\n"
        summary += f"Total Samples: {total:,}\n\n"
        
        for split, count in split_counts.items():
            pct = count / total * 100
            summary += f"{split}: {count:,} ({pct:.1f}%)\n"
        
        summary += "\n--- Ratio ---\n"
        ratios = [f"{count/total*100:.0f}" for count in split_counts.values]
        summary += f"  {':'.join(ratios)}\n"
        
        ax2.text(0.1, 0.9, summary, transform=ax2.transAxes, fontsize=12,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1'))
        
        fig.suptitle('Dataset Split Overview', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_mimic_metadata_overview(self, df: pd.DataFrame, output_path: Path, description: str = ""):
        """Create holistic MIMIC-CXR metadata visualization showing views, dimensions, procedures together."""
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # 1. View Position Distribution (Main clinical info)
        ax1 = fig.add_subplot(gs[0, 0])
        view_col = 'ViewPosition' if 'ViewPosition' in df.columns else None
        
        if view_col:
            view_counts = df[view_col].value_counts().head(6)
            colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']
            
            wedges, texts, autotexts = ax1.pie(
                view_counts.values, labels=view_counts.index, autopct='%1.1f%%',
                colors=colors[:len(view_counts)], 
                wedgeprops=dict(edgecolor='white', linewidth=2),
                textprops={'fontsize': 10}
            )
            ax1.set_title('X-Ray View Positions\n(Clinical Perspective)', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No ViewPosition column', ha='center', va='center')
        
        # 2. Image Dimensions 2D Histogram (Shows resolution patterns)
        ax2 = fig.add_subplot(gs[0, 1])
        height_col = 'Rows' if 'Rows' in df.columns else None
        width_col = 'Columns' if 'Columns' in df.columns else None
        
        if height_col and width_col:
            # Sample for performance
            sample_size = min(10000, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df
            
            h = ax2.hist2d(sample_df[width_col], sample_df[height_col], bins=30, cmap='Blues')
            ax2.set_xlabel('Width (pixels)', fontsize=11)
            ax2.set_ylabel('Height (pixels)', fontsize=11)
            ax2.set_title('Image Resolution Distribution\n(Density Heatmap)', fontsize=12, fontweight='bold')
            plt.colorbar(h[3], ax=ax2, label='Count')
        else:
            ax2.text(0.5, 0.5, 'No dimension columns', ha='center', va='center')
        
        # 3. Procedure Types (What kind of X-rays)
        ax3 = fig.add_subplot(gs[0, 2])
        proc_col = 'ProcedureCodeSequence_CodeMeaning' if 'ProcedureCodeSequence_CodeMeaning' in df.columns else None
        
        if proc_col:
            proc_counts = df[proc_col].value_counts().head(6)
            short_names = [str(p)[:25] + '...' if len(str(p)) > 25 else str(p) for p in proc_counts.index]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(proc_counts)))
            bars = ax3.barh(short_names[::-1], proc_counts.values[::-1], color=colors[::-1], edgecolor='white')
            ax3.set_xlabel('Count', fontsize=11)
            ax3.set_title('Procedure Types\n(Why was X-ray taken?)', fontsize=12, fontweight='bold')
            ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        else:
            ax3.text(0.5, 0.5, 'No procedure column', ha='center', va='center')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Frontal vs Lateral (Important for model training)
        ax4 = fig.add_subplot(gs[1, 0])
        if view_col:
            frontal = df[view_col].isin(['AP', 'PA']).sum()
            lateral = df[view_col].isin(['LATERAL', 'LL']).sum()
            other = len(df) - frontal - lateral
            
            values = [frontal, lateral, other]
            labels = ['Frontal (AP/PA)', 'Lateral', 'Other']
            colors = ['#3498db', '#27ae60', '#95a5a6']
            
            wedges, texts, autotexts = ax4.pie(
                values, labels=labels, autopct='%1.1f%%',
                colors=colors, wedgeprops=dict(edgecolor='white', linewidth=2),
                textprops={'fontsize': 10}
            )
            ax4.set_title('Frontal vs Lateral Views\n(Training Data Split)', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No view data', ha='center', va='center')
        
        # 5. Resolution Statistics Box
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        
        summary = "IMAGE STATISTICS\n" + "=" * 35 + "\n\n"
        summary += f"Total Images: {len(df):,}\n\n"
        
        if height_col and width_col:
            summary += f"Height (pixels):\n"
            summary += f"  Mean: {df[height_col].mean():.0f}\n"
            summary += f"  Std:  {df[height_col].std():.0f}\n"
            summary += f"  Range: [{df[height_col].min()}, {df[height_col].max()}]\n\n"
            
            summary += f"Width (pixels):\n"
            summary += f"  Mean: {df[width_col].mean():.0f}\n"
            summary += f"  Std:  {df[width_col].std():.0f}\n"
            summary += f"  Range: [{df[width_col].min()}, {df[width_col].max()}]\n"
        
        ax5.text(0.1, 0.9, summary, transform=ax5.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        ax5.set_title('Resolution Stats', fontsize=12, fontweight='bold')
        
        # 6. Training Recommendations
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        rec = "PREPROCESSING NOTES\n" + "=" * 30 + "\n\n"
        
        if view_col:
            frontal_pct = df[view_col].isin(['AP', 'PA']).sum() / len(df) * 100
            if frontal_pct < 70:
                rec += f"[INFO] {frontal_pct:.1f}% frontal views\n"
                rec += "       Consider view-specific filtering\n\n"
            else:
                rec += f"[OK] {frontal_pct:.1f}% frontal views\n       Good for frontal-focused models\n\n"
        
        if height_col and width_col:
            std_h = df[height_col].std()
            if std_h > 500:
                rec += "[!] High resolution variance\n"
                rec += "    Resize to consistent size\n"
                rec += "    (e.g., 224x224 or 512x512)\n\n"
            else:
                rec += "[OK] Consistent image sizes\n\n"
        
        rec += "[TIP] Use transforms:\n"
        rec += "  - Resize to model input size\n"
        rec += "  - Normalize with ImageNet stats\n"
        rec += "  - Consider histogram equalization\n"
        
        ax6.text(0.1, 0.9, rec, transform=ax6.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#f0ad4e'))
        ax6.set_title('Preprocessing Tips', fontsize=12, fontweight='bold')
        
        fig.suptitle('MIMIC-CXR Image Metadata Overview', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    def _create_column_summary_plot(self, filename: str, df: pd.DataFrame, viz_dir: Path):
        """Create a summary overview of all columns in the file."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Data types distribution
        ax1 = axes[0, 0]
        dtype_counts = df.dtypes.astype(str).value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(dtype_counts)))
        ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Column Data Types', fontweight='bold')
        
        # 2. Null percentage per column
        ax2 = axes[0, 1]
        null_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
        if len(null_pcts) > 20:
            null_pcts = null_pcts.tail(20)  # Show top 20 with most nulls
        
        colors = ['#e74c3c' if p > 20 else '#f39c12' if p > 5 else '#2ecc71' for p in null_pcts.values]
        ax2.barh([c.split('.')[-1][:20] for c in null_pcts.index], null_pcts.values, color=colors)
        ax2.set_xlabel('Null Percentage (%)')
        ax2.set_title('Missing Values by Column', fontweight='bold')
        ax2.axvline(5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
        ax2.axvline(20, color='red', linestyle='--', alpha=0.7, label='20% threshold')
        ax2.legend(fontsize=8)
        
        # 3. Unique value counts (log scale)
        ax3 = axes[1, 0]
        unique_counts = df.nunique().sort_values(ascending=False)
        if len(unique_counts) > 15:
            unique_counts = unique_counts.head(15)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique_counts)))
        ax3.barh([c.split('.')[-1][:20] for c in unique_counts.index], unique_counts.values, color=colors)
        ax3.set_xlabel('Unique Values (log scale)')
        ax3.set_xscale('log')
        ax3.set_title('Unique Values per Column (Top 15)', fontweight='bold')
        ax3.invert_yaxis()
        
        # 4. Column overview table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        n_rows = len(df)
        n_cols = len(df.columns)
        n_nulls = df.isnull().sum().sum()
        n_bool = sum(1 for c in df.columns if df[c].dtype == 'bool')
        n_int = sum(1 for c in df.columns if df[c].dtype in ['int64', 'int32', 'int16', 'int8'])
        n_float = sum(1 for c in df.columns if df[c].dtype in ['float64', 'float32'])
        n_str = sum(1 for c in df.columns if df[c].dtype == 'object')
        
        summary_text = f"""
FILE SUMMARY
{'='*40}

Rows:                {n_rows:>15,}
Columns:             {n_cols:>15,}
Total Cells:         {n_rows * n_cols:>15,}
Total Nulls:         {n_nulls:>15,}
Null Percentage:     {n_nulls / (n_rows * n_cols) * 100:>14.2f}%

COLUMN TYPES
{'='*40}
Boolean:             {n_bool:>15}
Integer:             {n_int:>15}
Float:               {n_float:>15}
String/Object:       {n_str:>15}

MEMORY USAGE
{'='*40}
Total Memory:        {df.memory_usage(deep=True).sum() / 1024**2:>12.2f} MB
Avg per Column:      {df.memory_usage(deep=True).sum() / n_cols / 1024:>12.2f} KB
"""
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.suptitle(f'Column Analysis Summary: {filename}', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(viz_dir / 'column_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _show_key_distributions(self, filename: str, df: pd.DataFrame):
        """Show distributions for key columns based on file type."""
        
        # Question metadata specific analysis
        if 'question' in filename.lower():
            if 'question.question_type' in df.columns:
                logger.info(f"\n  [Distribution] question.question_type:")
                dist = df['question.question_type'].value_counts().head(10)
                for val, cnt in dist.items():
                    pct = cnt / len(df) * 100
                    logger.info(f"    {val:<40} {cnt:>12,} ({pct:>5.1f}%)")
            
            if 'question.question_strategy' in df.columns:
                logger.info(f"\n  [Distribution] question.question_strategy:")
                dist = df['question.question_strategy'].value_counts()
                for val, cnt in dist.items():
                    pct = cnt / len(df) * 100
                    logger.info(f"    {val:<40} {cnt:>12,} ({pct:>5.1f}%)")
            
            if 'question.quality.rating' in df.columns:
                logger.info(f"\n  [Distribution] question.quality.rating:")
                dist = df['question.quality.rating'].value_counts().sort_index()
                for val, cnt in dist.items():
                    pct = cnt / len(df) * 100
                    logger.info(f"    Rating {val:<35} {cnt:>12,} ({pct:>5.1f}%)")
        
        # Answer metadata specific analysis
        elif 'answer' in filename.lower():
            if 'answer.answer_type' in df.columns:
                logger.info(f"\n  [Distribution] answer.answer_type:")
                dist = df['answer.answer_type'].value_counts()
                for val, cnt in dist.items():
                    pct = cnt / len(df) * 100
                    logger.info(f"    {val:<40} {cnt:>12,} ({pct:>5.1f}%)")
            
            if 'answer.positiveness' in df.columns:
                logger.info(f"\n  [Distribution] answer.positiveness:")
                dist = df['answer.positiveness'].value_counts()
                for val, cnt in dist.items():
                    pct = cnt / len(df) * 100
                    logger.info(f"    {val:<40} {cnt:>12,} ({pct:>5.1f}%)")
        
        # Study metadata specific analysis
        elif 'study' in filename.lower():
            if 'study.procedure' in df.columns:
                logger.info(f"\n  [Distribution] study.procedure:")
                dist = df['study.procedure'].value_counts().head(5)
                for val, cnt in dist.items():
                    pct = cnt / len(df) * 100
                    logger.info(f"    {str(val)[:40]:<40} {cnt:>12,} ({pct:>5.1f}%)")
            
            if 'study.num_observations' in df.columns:
                logger.info(f"\n  [Statistics] study.num_observations:")
                stats = df['study.num_observations'].describe()
                logger.info(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                logger.info(f"    Min: {stats['min']:.0f}, Max: {stats['max']:.0f}, Median: {stats['50%']:.0f}")
        
        # Image metadata specific analysis
        elif 'image' in filename.lower():
            if 'img.view_position' in df.columns:
                logger.info(f"\n  [Distribution] img.view_position:")
                dist = df['img.view_position'].value_counts().head(6)
                for val, cnt in dist.items():
                    pct = cnt / len(df) * 100
                    logger.info(f"    {val:<40} {cnt:>12,} ({pct:>5.1f}%)")
            
            if 'img.height' in df.columns and 'img.width' in df.columns:
                logger.info(f"\n  [Statistics] Image Dimensions:")
                logger.info(f"    Height: {df['img.height'].mean():.0f} +/- {df['img.height'].std():.0f}")
                logger.info(f"    Width:  {df['img.width'].mean():.0f} +/- {df['img.width'].std():.0f}")
    
    def _analyze_dataset_info_json(self, path: Path):
        """Comprehensively analyze the dataset_info.json file."""
        try:
            with open(path) as f:
                info = json.load(f)
            
            logger.info(f"\n  {'='*60}")
            logger.info(f"  [REFERENCE] dataset_info.json")
            logger.info(f"  Defines all valid values for tags and categories")
            logger.info(f"  {'='*60}")
            
            for key, values in info.items():
                if isinstance(values, list):
                    logger.info(f"  {key}: {len(values)} types")
                    if len(values) <= 10:
                        logger.info(f"    Values: {values}")
                    else:
                        logger.info(f"    Sample: {values[:5]} ... {values[-3:]}")
                elif isinstance(values, dict):
                    logger.info(f"  {key}: {len(values)} entries (dict)")
            
            # Store in report
            self.report.metadata_csv_info['dataset_info.json'] = {
                'keys': list(info.keys()),
                'counts': {k: len(v) if isinstance(v, (list, dict)) else 1 for k, v in info.items()}
            }
            
        except Exception as e:
            logger.warning(f"    Error loading dataset_info.json: {e}")
    
    def _analyze_exports_directory(self):
        """Analyze pre-filtered exports if available."""
        exports_dir = self.mimic_qa_path / 'exports'
        if not exports_dir.exists():
            return
        
        logger.info(f"\n  [EXPORTS] Pre-filtered data exports:")
        
        for grade_dir in sorted(exports_dir.iterdir()):
            if grade_dir.is_dir() and grade_dir.name.startswith('grade_'):
                grade = grade_dir.name.replace('grade_', '')
                
                # Count files
                qa_files = list(grade_dir.glob('qa/*.json'))
                sg_files = list(grade_dir.glob('scene_graphs/*.json'))
                
                logger.info(f"    Grade {grade}: {len(qa_files):,} QA files, {len(sg_files):,} scene graphs")
    
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
        
        # ============ Plot 4: Dataset Summary Card ============
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        # Left panel: MIMIC-CXR-JPG stats
        ax1 = axes[0]
        ax1.axis('off')
        ax1.set_title('MIMIC-CXR-JPG Dataset', fontsize=14, fontweight='bold', pad=20)
        
        cxr_text = f"""
    IMAGES & STUDIES
    ================
    Total Images:      {self.report.total_images:>12,}
    Total Studies:     {self.report.total_studies:>12,}
    Total Patients:    {self.report.total_patients:>12,}
    CheXpert Labeled:  {self.report.chexpert_studies_labeled:>12,}
    
    DATA SPLITS
    ================
    Train:             {self.report.train_samples:>12,}
    Validate:          {self.report.val_samples:>12,}
    Test:              {self.report.test_samples:>12,}
    
    IMAGE DIMENSIONS
    ================
    Height: {self.report.image_dimension_stats.get('height_mean', 0):.0f} +/- {self.report.image_dimension_stats.get('height_std', 0):.0f}
    Width:  {self.report.image_dimension_stats.get('width_mean', 0):.0f} +/- {self.report.image_dimension_stats.get('width_std', 0):.0f}
"""
        ax1.text(0.1, 0.95, cxr_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#e8f4fd', alpha=0.9))
        
        # Middle panel: MIMIC-Ext-CXR-QBA stats
        ax2 = axes[1]
        ax2.axis('off')
        ax2.set_title('MIMIC-Ext-CXR-QBA Dataset', fontsize=14, fontweight='bold', pad=20)
        
        qba_text = f"""
    QA & SCENE GRAPHS
    ==================
    Total QA Pairs:    {self.report.total_qa_pairs:>12,}
    Scene Graphs:      {self.report.total_scene_graphs:>12,}
    Avg Obs/Graph:     {self.report.avg_observations_per_graph:>12.2f}
    BBox Coverage:     {self.report.bbox_coverage*100:>11.1f}%
    
    QUESTION TYPES
    ==================
    Total Types:       {len(self.report.question_type_distribution):>12,}
    
    FINDING POLARITY
    ==================
    Positive:          {self.report.polarity_distribution.get('positive', 0):>12,}
    Negative:          {self.report.polarity_distribution.get('negative', 0):>12,}
    Neutral:           {self.report.polarity_distribution.get('neutral', 0):>12,}
"""
        ax2.text(0.1, 0.95, qba_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f4fde8', alpha=0.9))
        
        # Right panel: Cross-dataset and status
        ax3 = axes[2]
        ax3.axis('off')
        ax3.set_title('Cross-Dataset & Status', fontsize=14, fontweight='bold', pad=20)
        
        status_color = '#d4edda' if self.report.is_ready else '#f8d7da'
        status_text = 'READY FOR TRAINING' if self.report.is_ready else 'NOT READY'
        
        cross_text = f"""
    DATASET ALIGNMENT
    ==================
    Matched Studies:   {self.report.matched_studies:>12,}
    CXR-JPG Only:      {self.report.unmatched_studies_cxr:>12,}
    QBA Only:          {self.report.unmatched_studies_qba:>12,}
    
    METADATA FILES
    ==================
    Files Analyzed:    {len(self.report.metadata_csv_info):>12,}
    
    ISSUES & WARNINGS
    ==================
    Critical Issues:   {len(self.report.issues):>12,}
    Warnings:          {len(self.report.warnings):>12,}
    
    ==================
    STATUS: {status_text}
    ==================
"""
        ax3.text(0.1, 0.95, cross_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.9))
        
        plt.suptitle('MIMIC-CXR VQA Dataset Summary', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_path = self.output_dir / 'dataset_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"[PLOT] Dataset summary saved to: {plot_path}")
        
        # ============ Plot 5: Comprehensive Statistics Dashboard ============
        self._generate_statistics_dashboard()

    def _generate_statistics_dashboard(self):
        """Generate a comprehensive statistics dashboard."""
        if not PLOTTING_AVAILABLE:
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: CheXpert heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        if self.report.chexpert_label_counts:
            labels = list(self.report.chexpert_label_counts.keys())
            data = []
            for label in labels:
                counts = self.report.chexpert_label_counts[label]
                total = sum(counts.values())
                row = [
                    counts.get('positive', 0) / total * 100 if total > 0 else 0,
                    counts.get('negative', 0) / total * 100 if total > 0 else 0,
                    counts.get('uncertain', 0) / total * 100 if total > 0 else 0,
                    counts.get('missing', 0) / total * 100 if total > 0 else 0,
                ]
                data.append(row)
            
            data_arr = np.array(data)
            im = ax1.imshow(data_arr, aspect='auto', cmap='RdYlGn_r')
            ax1.set_yticks(range(len(labels)))
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.set_xticks(range(4))
            ax1.set_xticklabels(['Positive', 'Negative', 'Uncertain', 'Missing'], fontsize=9)
            ax1.set_title('CheXpert Label Distribution (%)', fontweight='bold')
            plt.colorbar(im, ax=ax1, label='Percentage')
        
        # Row 1: View position pie
        ax2 = fig.add_subplot(gs[0, 2])
        if self.report.view_position_distribution:
            views = list(self.report.view_position_distribution.keys())[:5]
            counts = [self.report.view_position_distribution[v] for v in views]
            colors = plt.cm.Set2(np.linspace(0, 1, len(views)))
            ax2.pie(counts, labels=views, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 8})
            ax2.set_title('View Positions', fontweight='bold')
        
        # Row 1: Procedure distribution
        ax3 = fig.add_subplot(gs[0, 3])
        if self.report.procedure_distribution:
            procs = list(self.report.procedure_distribution.keys())[:5]
            counts = [self.report.procedure_distribution[p] for p in procs]
            short_procs = [p[:25] + '...' if len(p) > 25 else p for p in procs]
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(procs)))
            ax3.barh(short_procs, counts, color=colors)
            ax3.set_xlabel('Count')
            ax3.set_title('Procedures', fontweight='bold')
            ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # Row 2: Question types
        ax4 = fig.add_subplot(gs[1, :2])
        if self.report.question_type_distribution:
            types = list(self.report.question_type_distribution.keys())[:12]
            counts = [self.report.question_type_distribution[t] for t in types]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(types)))
            bars = ax4.barh(types, counts, color=colors)
            ax4.set_xlabel('Count')
            ax4.set_title('Question Type Distribution (Top 12)', fontweight='bold')
            ax4.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # Row 2: Polarity
        ax5 = fig.add_subplot(gs[1, 2])
        if self.report.polarity_distribution:
            labels = list(self.report.polarity_distribution.keys())
            sizes = list(self.report.polarity_distribution.values())
            colors = {'positive': '#e74c3c', 'negative': '#2ecc71', 'neutral': '#95a5a6'}
            pie_colors = [colors.get(l, '#3498db') for l in labels]
            ax5.pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, 
                   explode=[0.02]*len(labels), textprops={'fontsize': 9})
            ax5.set_title('Finding Polarity', fontweight='bold')
        
        # Row 2: Anatomical regions
        ax6 = fig.add_subplot(gs[1, 3])
        if self.report.region_distribution:
            regions = list(self.report.region_distribution.keys())
            counts = list(self.report.region_distribution.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
            ax6.bar(regions, counts, color=colors)
            ax6.set_ylabel('Count')
            ax6.set_title('Anatomical Regions', fontweight='bold')
            ax6.tick_params(axis='x', rotation=45)
            ax6.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Row 3: Split distribution
        ax7 = fig.add_subplot(gs[2, 0])
        splits = ['Train', 'Validate', 'Test']
        split_counts = [self.report.train_samples, self.report.val_samples, self.report.test_samples]
        colors = ['#3498db', '#9b59b6', '#e74c3c']
        ax7.bar(splits, split_counts, color=colors)
        ax7.set_ylabel('Count')
        ax7.set_title('Data Splits', fontweight='bold')
        ax7.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        for i, (split, count) in enumerate(zip(splits, split_counts)):
            ax7.text(i, count + max(split_counts)*0.02, f'{count:,}', ha='center', fontsize=8)
        
        # Row 3: Answer types
        ax8 = fig.add_subplot(gs[2, 1])
        if self.report.answer_type_distribution:
            types = list(self.report.answer_type_distribution.keys())
            counts = list(self.report.answer_type_distribution.values())
            colors = plt.cm.Paired(np.linspace(0, 1, len(types)))
            ax8.bar(types, counts, color=colors)
            ax8.set_ylabel('Count')
            ax8.set_title('Answer Types', fontweight='bold')
            ax8.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Row 3: Scene graph quality metrics
        ax9 = fig.add_subplot(gs[2, 2:])
        metrics = ['Avg Obs/Graph', 'Avg Regions/Obs', 'BBox Coverage (%)']
        values = [
            self.report.avg_observations_per_graph,
            self.report.avg_regions_per_observation,
            self.report.bbox_coverage * 100
        ]
        colors = ['#1abc9c', '#3498db', '#9b59b6']
        bars = ax9.bar(metrics, values, color=colors)
        ax9.set_ylabel('Value')
        ax9.set_title('Scene Graph Quality Metrics', fontweight='bold')
        for bar, val in zip(bars, values):
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}', ha='center', fontsize=10)
        
        plt.suptitle('MIMIC-CXR VQA Comprehensive Statistics Dashboard', 
                    fontsize=16, fontweight='bold', y=1.01)
        
        plot_path = self.output_dir / 'statistics_dashboard.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"[PLOT] Statistics dashboard saved to: {plot_path}")

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
                
                text_content = f"[SCENE GRAPH SUMMARY]\n{'='*40}\n"
                text_content += f"Observations: {len(observations_data)}\n"
                text_content += f"Total bboxes: {len(all_bboxes)}\n\n"
                
                text_content += f"[TOP OBSERVATIONS]\n{'-'*40}\n"
                for i, obs in enumerate(observations_data[:5], 1):
                    pos_marker = "[+]" if obs['positiveness'] == 'pos' else "[-]" if obs['positiveness'] == 'neg' else "[?]"
                    text_content += f"{i}. {pos_marker} {obs['name'][:40]}\n"
                    if obs['entities']:
                        text_content += f"   Entities: {', '.join(obs['entities'][:3])}\n"
                    if obs['regions']:
                        text_content += f"   Regions: {', '.join(obs['regions'][:3])}\n"
                
                text_content += f"\n[SAMPLE QUESTIONS]\n{'-'*40}\n"
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
                
                logger.info(f"  [OK] Saved: {out_path.name}")
                successful += 1
                
            except Exception as e:
                logger.debug(f"Visualization failed for {qa_file}: {e}")
                continue
        
        if successful > 0:
            logger.info(f"\n[IMG] {successful} visualizations saved to: {viz_dir}")
        
        # Now create combined image + network graph visualizations
        self._visualize_image_with_network(num_samples)

    def _visualize_image_with_network(self, num_samples: int = 10):
        """
        Create comprehensive visualization combining:
        - Chest X-ray image with bounding boxes
        - Network graph showing observation relationships
        - Clinical summary panel
        
        This provides the complete picture of a patient's data.
        """
        if not PLOTTING_AVAILABLE or not NETWORKX_AVAILABLE or not PIL_AVAILABLE:
            logger.warning("Required libraries not available for combined visualization")
            return
        
        logger.info(f"\n[COMBINED] Creating combined image + network graph visualizations...")
        
        qa_dir = self.mimic_qa_path / 'qa'
        if not qa_dir.exists():
            return
        
        qa_files = list(qa_dir.rglob('*.qa.json'))
        if not qa_files:
            return
        
        sample_pool = random.sample(qa_files, min(num_samples * 4, len(qa_files)))
        viz_dir = self.output_dir / 'combined_visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        
        for qa_file in sample_pool:
            if successful >= num_samples:
                break
            
            try:
                with open(qa_file) as f:
                    qa_data = json.load(f)
                
                patient_id = qa_data.get('patient_id', qa_file.parts[-2])
                study_id = qa_data.get('study_id', qa_file.stem.split('.')[0])
                
                # Get image
                img_path = self._get_image_path(patient_id, study_id)
                if img_path is None or not img_path.exists():
                    continue
                
                # Get scene graph
                sg_path = self._get_scene_graph_path(patient_id, study_id)
                if sg_path is None or not sg_path.exists():
                    continue
                
                with open(sg_path) as f:
                    sg = json.load(f)
                
                observations = sg.get('observations', {})
                if len(observations) < 3:
                    continue  # Need enough data for interesting graph
                
                image = Image.open(img_path).convert('RGB')
                img_width, img_height = image.size
                
                # ============ Create 3-panel visualization ============
                fig = plt.figure(figsize=(24, 12))
                gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.5, 1.5, 1.2, 0.8], 
                                       height_ratios=[2, 1], hspace=0.2, wspace=0.2)
                
                # ========== Panel 1: X-ray Image with Bounding Boxes ==========
                ax_img = fig.add_subplot(gs[:, 0])
                ax_img.imshow(image, cmap='gray')
                
                # Draw bounding boxes from observations
                COLOR_POS = '#e74c3c'
                COLOR_NEG = '#27ae60'
                COLOR_UNK = '#f39c12'
                
                bbox_count = 0
                for obs_id, obs in list(observations.items())[:8]:
                    positiveness = obs.get('positiveness', 'unknown')
                    if positiveness == 'pos':
                        color = COLOR_POS
                    elif positiveness == 'neg':
                        color = COLOR_NEG
                    else:
                        color = COLOR_UNK
                    
                    locs = obs.get('localization', {})
                    for img_id, loc_data in locs.items():
                        bboxes = loc_data.get('bboxes', []) or []
                        for bb in bboxes[:2]:  # Max 2 per observation
                            if len(bb) == 4:
                                x1, y1, x2, y2 = bb
                                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                     linewidth=2.5, edgecolor=color, 
                                                     facecolor=color, alpha=0.15)
                                ax_img.add_patch(rect)
                                
                                # Add border
                                rect_border = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                           linewidth=2, edgecolor=color, 
                                                           facecolor='none')
                                ax_img.add_patch(rect_border)
                                
                                # Compact label
                                obs_name = obs.get('name', obs_id)[:18]
                                ax_img.text(x1 + 2, y1 + 12, obs_name, fontsize=7, color='white',
                                           fontweight='bold',
                                           bbox=dict(boxstyle='round,pad=0.15', facecolor=color, 
                                                    alpha=0.85, edgecolor='none'))
                                bbox_count += 1
                
                ax_img.set_title(f'Chest X-Ray: {patient_id}/{study_id}\n{img_path.name} ({img_width}x{img_height})',
                                fontsize=11, fontweight='bold')
                ax_img.axis('off')
                
                # Add legend
                legend_patches = [
                    mpatches.Patch(color=COLOR_POS, label='Positive Finding', alpha=0.8),
                    mpatches.Patch(color=COLOR_NEG, label='Negative/Normal', alpha=0.8),
                    mpatches.Patch(color=COLOR_UNK, label='Uncertain', alpha=0.8),
                ]
                ax_img.legend(handles=legend_patches, loc='lower right', fontsize=8, 
                             framealpha=0.9, edgecolor='gray')
                
                # ========== Panel 2: Network Graph (with CORRECT field names) ==========
                ax_graph = fig.add_subplot(gs[:, 1])
                
                # Build network graph using CORRECT MIMIC-Ext-CXR-QBA field names
                G = nx.DiGraph()
                node_colors = []
                node_sizes = []
                node_labels = {}
                
                obs_list = list(observations.items())[:10]
                
                for obs_id, obs in obs_list:
                    short_name = obs.get('name', obs_id)[:18]
                    G.add_node(obs_id, node_type='obs')
                    node_labels[obs_id] = short_name
                    
                    polarity = obs.get('positiveness', 'unknown')
                    if polarity == 'pos':
                        node_colors.append(COLOR_POS)
                    elif polarity == 'neg':
                        node_colors.append(COLOR_NEG)
                    else:
                        node_colors.append(COLOR_UNK)
                    node_sizes.append(1800)
                    
                    # Add region connections - CORRECT field: "regions" contains [{"region": "...", ...}]
                    regions_list = obs.get('regions', [])
                    for reg_info in regions_list[:2]:
                        if isinstance(reg_info, dict):
                            region_name = reg_info.get('region', '')[:12]
                        else:
                            region_name = str(reg_info)[:12]
                        
                        if region_name:
                            region_node = f"R:{region_name}"
                            if not G.has_node(region_node):
                                G.add_node(region_node, node_type='region')
                                node_labels[region_node] = region_name
                                node_colors.append('#3498db')
                                node_sizes.append(1000)
                            G.add_edge(obs_id, region_node)
                    
                    # Add entity connections - field: "obs_entities"
                    for entity in obs.get('obs_entities', [])[:2]:
                        entity_node = f"E:{entity[:12]}"
                        if not G.has_node(entity_node):
                            G.add_node(entity_node, node_type='entity')
                            node_labels[entity_node] = entity[:12]
                            node_colors.append('#9b59b6')
                            node_sizes.append(800)
                        G.add_edge(obs_id, entity_node)
                
                if len(G.nodes()) >= 3:
                    try:
                        pos = nx.kamada_kawai_layout(G)
                    except:
                        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
                    
                    # Draw edges with arrows
                    nx.draw_networkx_edges(G, pos, ax=ax_graph, alpha=0.5, 
                                          edge_color='#7f8c8d', width=1.5,
                                          arrows=True, arrowsize=12,
                                          connectionstyle='arc3,rad=0.1')
                    
                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, ax=ax_graph,
                                          node_color=node_colors,
                                          node_size=node_sizes,
                                          alpha=0.9)
                    
                    # Draw labels
                    nx.draw_networkx_labels(G, pos, node_labels, ax=ax_graph,
                                           font_size=7, font_weight='bold')
                
                ax_graph.set_title(f'Scene Graph Network\n({len(G.nodes())} nodes, {len(G.edges())} edges)',
                                  fontsize=11, fontweight='bold')
                ax_graph.axis('off')
                
                # ========== Panel 3: Clinical Details ==========
                ax_clinical = fig.add_subplot(gs[0, 2:])
                ax_clinical.axis('off')
                
                # Build clinical summary
                indication = sg.get('indication', {})
                sentences = sg.get('sentences', {})
                
                clinical_text = "CLINICAL CONTEXT\n" + "=" * 50 + "\n\n"
                
                if indication.get('indication_summary'):
                    clinical_text += f"Indication:\n  {indication['indication_summary'][:120]}...\n\n"
                
                if indication.get('patient_info'):
                    clinical_text += f"Patient Info:\n  {indication['patient_info'][:100]}\n\n"
                
                # Add first report sentence
                if sentences:
                    first_sent = list(sentences.values())[0]
                    sent_text = first_sent.get('sentence', '')[:150]
                    section = first_sent.get('section_type', 'REPORT')
                    clinical_text += f"Report ({section}):\n  \"{sent_text}...\"\n"
                
                ax_clinical.text(0.02, 0.95, clinical_text, transform=ax_clinical.transAxes,
                                fontsize=10, verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                                         edgecolor='#dee2e6', alpha=0.95))
                
                # ========== Panel 4: Q&A Samples ==========
                ax_qa = fig.add_subplot(gs[1, 2:])
                ax_qa.axis('off')
                
                questions = qa_data.get('questions', [])[:4]
                
                qa_text = "SAMPLE VQA PAIRS\n" + "=" * 50 + "\n\n"
                
                for i, q in enumerate(questions, 1):
                    q_type = q.get('question_type', 'unknown')
                    q_text = q.get('question', '')[:55]
                    qa_text += f"Q{i} [{q_type}]:\n  {q_text}...\n"
                    
                    answers = q.get('answers', [])
                    if answers:
                        a_text = answers[0].get('text', '')[:45]
                        qa_text += f"  A: {a_text}\n\n"
                
                ax_qa.text(0.02, 0.95, qa_text, transform=ax_qa.transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                   edgecolor='#f0ad4e', alpha=0.95))
                
                # Overall title
                fig.suptitle(f'Comprehensive Patient Analysis: {patient_id} / Study: {study_id}',
                            fontsize=14, fontweight='bold', y=1.01)
                
                plt.tight_layout()
                out_path = viz_dir / f"combined_{successful+1:02d}_{patient_id}_{study_id}.png"
                plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                logger.info(f"  [OK] Combined visualization saved: {out_path.name}")
                successful += 1
                
            except Exception as e:
                logger.debug(f"Combined visualization failed: {e}")
                traceback.print_exc()
                continue
        
        if successful > 0:
            logger.info(f"\n[COMBINED] {successful} combined visualizations saved to: {viz_dir}")

    def _visualize_scene_graphs(self, num_samples: int = 5):
        """
        Create PROPER network graph visualizations using networkx that match
        the pipeline.png reference style.
        
        Uses CORRECT MIMIC-Ext-CXR-QBA field names:
        - observations: contains "regions" (list of {"region": "..."}), "obs_entities", etc.
        - located_at_relations: explicit observation->region relations
        - obs_relations: parent-child observation relations  
        - obs_sent_relations: observation->sentence relations
        - region_region_relations: anatomical region hierarchy
        
        Creates a DIRECTED graph with labeled edges like the SSG-VQA pipeline diagram.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, skipping scene graph visualization")
            return
        
        if not NETWORKX_AVAILABLE:
            logger.warning("networkx not available, skipping network graph visualization")
            return
        
        logger.info(f"  Creating PIPELINE-STYLE scene graph visualizations...")
        
        scene_data_dir = self.mimic_qa_path / 'scene_data'
        if not scene_data_dir.exists():
            logger.warning("Scene data directory not found")
            return
        
        sg_files = list(scene_data_dir.rglob('*.scene_graph.json'))
        if not sg_files:
            logger.warning("No scene graph files found")
            return
        
        sample_files = random.sample(sg_files, min(num_samples * 3, len(sg_files)))
        viz_dir = self.output_dir / 'scene_graph_networks'
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
                regions_dict = sg.get('regions', {})
                sentences = sg.get('sentences', {})
                
                # Get explicit relations from the scene graph
                located_at_rels = sg.get('located_at_relations', [])
                obs_relations = sg.get('obs_relations', [])
                obs_sent_rels = sg.get('obs_sent_relations', [])
                region_region_rels = sg.get('region_region_relations', [])
                
                if not observations or len(observations) < 2:
                    continue
                
                # Create DIRECTED graph (like pipeline.png)
                G = nx.DiGraph()
                
                # Define color scheme (matching pipeline.png style)
                COLOR_OBS_POS = '#e74c3c'      # Red for positive findings
                COLOR_OBS_NEG = '#27ae60'      # Green for negative findings
                COLOR_OBS_UNK = '#f39c12'      # Orange for uncertain
                COLOR_REGION = '#5dade2'       # Light blue for regions (like pipeline)
                COLOR_ENTITY = '#af7ac5'       # Light purple for entities
                COLOR_SENTENCE = '#48c9b0'     # Teal for sentences
                
                # Edge colors for different relation types
                EDGE_LOCATED = '#3498db'
                EDGE_HAS_ENTITY = '#9b59b6'
                EDGE_CHILD = '#e67e22'
                EDGE_REGION_REL = '#7f8c8d'
                
                # Limit observations to prevent overcrowding
                obs_list = list(observations.items())[:10]
                obs_ids_in_graph = set()
                
                # ===== Add observation nodes =====
                for obs_id, obs in obs_list:
                    obs_name = obs.get('name', 'unknown')
                    short_name = obs_name[:22] if len(obs_name) <= 22 else obs_name[:20] + '..'
                    
                    polarity = obs.get('positiveness', 'unknown')
                    if polarity == 'pos':
                        color = COLOR_OBS_POS
                    elif polarity == 'neg':
                        color = COLOR_OBS_NEG
                    else:
                        color = COLOR_OBS_UNK
                    
                    G.add_node(obs_id, 
                              node_type='observation',
                              label=short_name,
                              color=color,
                              size=2200,
                              polarity=polarity)
                    obs_ids_in_graph.add(obs_id)
                    
                    # Extract regions from observation's "regions" field
                    # CORRECT field name: "regions" contains [{"region": "lungs", "distances": []}]
                    obs_regions_list = obs.get('regions', [])
                    for reg_info in obs_regions_list[:2]:
                        if isinstance(reg_info, dict):
                            region_name = reg_info.get('region', '')
                        else:
                            region_name = str(reg_info)
                        
                        if region_name:
                            region_node = f"R:{region_name}"
                            if not G.has_node(region_node):
                                G.add_node(region_node,
                                          node_type='region',
                                          label=region_name[:15],
                                          color=COLOR_REGION,
                                          size=1400)
                            G.add_edge(obs_id, region_node, 
                                      relation='located_in',
                                      color=EDGE_LOCATED)
                    
                    # Add entity connections from "obs_entities" field (correct name)
                    obs_entities = obs.get('obs_entities', [])
                    for entity in obs_entities[:2]:
                        entity_node = f"E:{entity}"
                        if not G.has_node(entity_node):
                            G.add_node(entity_node,
                                      node_type='entity', 
                                      label=entity[:15],
                                      color=COLOR_ENTITY,
                                      size=1200)
                        G.add_edge(obs_id, entity_node,
                                  relation='finding',
                                  color=EDGE_HAS_ENTITY)
                
                # ===== Add explicit relations from scene graph =====
                # Add observation-observation relations (parent-child)
                for rel in obs_relations[:15]:
                    parent_id = rel.get('parent_observation_id', '')
                    child_id = rel.get('child_observation_id', '')
                    child_type = rel.get('child_type', 'related')
                    
                    if parent_id in obs_ids_in_graph and child_id in obs_ids_in_graph:
                        G.add_edge(parent_id, child_id,
                                  relation=child_type[:12],
                                  color=EDGE_CHILD)
                
                # Add region-region relations
                for rel in region_region_rels[:10]:
                    region1 = f"R:{rel.get('region', '')}"
                    region2 = f"R:{rel.get('related_region', '')}"
                    rel_type = rel.get('relation_type', 'related')
                    
                    if G.has_node(region1) and G.has_node(region2):
                        G.add_edge(region1, region2,
                                  relation=rel_type[:10],
                                  color=EDGE_REGION_REL)
                
                if len(G.nodes()) < 3:
                    continue
                
                # ===== Create figure with pipeline-style layout =====
                fig = plt.figure(figsize=(22, 16))
                gs = gridspec.GridSpec(2, 3, figure=fig, 
                                      width_ratios=[2.8, 1, 0.8], 
                                      height_ratios=[2.5, 1])
                
                # Panel 1: Main Network Graph (pipeline style)
                ax1 = fig.add_subplot(gs[:, 0])
                
                # Use hierarchical layout for better visualization
                try:
                    # Try shell layout - puts different node types in concentric circles
                    obs_nodes = [n for n in G.nodes() if not n.startswith(('R:', 'E:', 'S:'))]
                    region_nodes = [n for n in G.nodes() if n.startswith('R:')]
                    entity_nodes = [n for n in G.nodes() if n.startswith('E:')]
                    
                    shells = [obs_nodes]
                    if region_nodes:
                        shells.append(region_nodes)
                    if entity_nodes:
                        shells.append(entity_nodes)
                    
                    if len(shells) > 1 and all(len(s) > 0 for s in shells):
                        pos = nx.shell_layout(G, shells)
                    else:
                        pos = nx.kamada_kawai_layout(G)
                except:
                    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)
                
                # Collect node attributes for drawing
                node_colors = [G.nodes[n].get('color', '#888') for n in G.nodes()]
                node_sizes = [G.nodes[n].get('size', 1000) for n in G.nodes()]
                
                # Draw edges with arrows and labels (like pipeline.png)
                edge_colors = [G.edges[e].get('color', '#888') for e in G.edges()]
                
                # Draw curved edges with arrows
                nx.draw_networkx_edges(G, pos, ax=ax1,
                                       edge_color=edge_colors,
                                       width=2.0,
                                       alpha=0.7,
                                       arrows=True,
                                       arrowsize=15,
                                       arrowstyle='-|>',
                                       connectionstyle='arc3,rad=0.1',
                                       node_size=node_sizes)
                
                # Draw edge labels (relation names like in pipeline.png)
                edge_labels = {(u, v): d.get('relation', '')[:10] 
                              for u, v, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1,
                                            font_size=7, font_color='#555',
                                            label_pos=0.5,
                                            bbox=dict(boxstyle='round,pad=0.1',
                                                     facecolor='white', 
                                                     alpha=0.8,
                                                     edgecolor='none'))
                
                # Draw nodes as ellipses (like pipeline.png)
                nx.draw_networkx_nodes(G, pos, ax=ax1,
                                       node_color=node_colors,
                                       node_size=node_sizes,
                                       alpha=0.95,
                                       node_shape='o')
                
                # Draw labels inside nodes
                labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, ax=ax1,
                                        font_size=7, font_weight='bold',
                                        font_color='white')
                
                ax1.set_title(f'Scene Graph: {patient_id} / {study_id}\n'
                             f'(Directed Graph with {len(G.nodes())} nodes, {len(G.edges())} edges)',
                             fontsize=14, fontweight='bold', pad=20)
                ax1.axis('off')
                
                # Add a box around the graph area (like pipeline.png)
                ax1.set_xlim(ax1.get_xlim()[0] - 0.1, ax1.get_xlim()[1] + 0.1)
                ax1.set_ylim(ax1.get_ylim()[0] - 0.1, ax1.get_ylim()[1] + 0.1)
                
                # Panel 2: Legend (styled like pipeline.png)
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.axis('off')
                
                legend_y = 0.95
                ax2.text(0.1, legend_y, 'NODE TYPES', fontsize=12, fontweight='bold',
                        transform=ax2.transAxes)
                
                legend_items = [
                    (COLOR_OBS_POS, 'Positive Finding (+)', 'o'),
                    (COLOR_OBS_NEG, 'Negative Finding (-)', 'o'),
                    (COLOR_OBS_UNK, 'Uncertain Finding (?)', 'o'),
                    (COLOR_REGION, 'Anatomical Region', 's'),
                    (COLOR_ENTITY, 'Medical Entity', '^'),
                ]
                
                for i, (color, label, marker) in enumerate(legend_items):
                    y_pos = legend_y - 0.08 - (i * 0.07)
                    ax2.scatter([0.12], [y_pos], c=color, s=200, marker=marker,
                               transform=ax2.transAxes)
                    ax2.text(0.2, y_pos, label, fontsize=10, va='center',
                            transform=ax2.transAxes)
                
                # Edge legend
                edge_y = 0.45
                ax2.text(0.1, edge_y, 'EDGE TYPES', fontsize=12, fontweight='bold',
                        transform=ax2.transAxes)
                
                edge_items = [
                    (EDGE_LOCATED, 'located_in', '-->'),
                    (EDGE_HAS_ENTITY, 'finding/entity', '-->'),
                    (EDGE_CHILD, 'parent-child', '-->'),
                    (EDGE_REGION_REL, 'region relation', '-->'),
                ]
                
                for i, (color, label, arrow) in enumerate(edge_items):
                    y_pos = edge_y - 0.08 - (i * 0.07)
                    ax2.plot([0.1, 0.18], [y_pos, y_pos], color=color, linewidth=3,
                            transform=ax2.transAxes)
                    ax2.text(0.2, y_pos, label, fontsize=10, va='center',
                            transform=ax2.transAxes)
                
                # Panel 3: Graph Statistics
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.axis('off')
                
                # Count node types
                n_obs = len([n for n in G.nodes() if G.nodes[n].get('node_type') == 'observation'])
                n_pos = len([n for n in G.nodes() if G.nodes[n].get('polarity') == 'pos'])
                n_neg = len([n for n in G.nodes() if G.nodes[n].get('polarity') == 'neg'])
                n_regions = len([n for n in G.nodes() if n.startswith('R:')])
                n_entities = len([n for n in G.nodes() if n.startswith('E:')])
                
                stats_text = f"""STATISTICS
━━━━━━━━━━━━━━

Nodes: {len(G.nodes())}
Edges: {len(G.edges())}

OBSERVATIONS
  Total: {n_obs}
  Positive: {n_pos}
  Negative: {n_neg}
  
CONNECTED
  Regions: {n_regions}
  Entities: {n_entities}

RELATIONS
  located_in: {sum(1 for _,_,d in G.edges(data=True) if d.get('relation')=='located_in')}
  finding: {sum(1 for _,_,d in G.edges(data=True) if d.get('relation')=='finding')}
  other: {sum(1 for _,_,d in G.edges(data=True) if d.get('relation') not in ['located_in', 'finding'])}"""
                
                ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                                 edgecolor='#dee2e6', alpha=0.95))
                
                # Panel 4: Observations list with clinical details
                ax4 = fig.add_subplot(gs[1, 1:])
                ax4.axis('off')
                
                # Build observation summary table
                obs_text = "OBSERVATIONS DETAIL\n" + "━" * 70 + "\n\n"
                
                for i, (obs_id, obs) in enumerate(obs_list[:8]):
                    polarity = obs.get('positiveness', '?')
                    marker = "[+]" if polarity == 'pos' else "[-]" if polarity == 'neg' else "[?]"
                    name = obs.get('name', 'unknown')[:35]
                    
                    # Get regions (using CORRECT field)
                    regions_list = obs.get('regions', [])
                    region_strs = []
                    for r in regions_list[:2]:
                        if isinstance(r, dict):
                            region_strs.append(r.get('region', '?')[:15])
                        else:
                            region_strs.append(str(r)[:15])
                    regions_str = ', '.join(region_strs) if region_strs else 'N/A'
                    
                    # Get entities
                    entities = obs.get('obs_entities', [])[:2]
                    entities_str = ', '.join(e[:12] for e in entities) if entities else 'N/A'
                    
                    obs_text += f"{marker} {name}\n"
                    obs_text += f"    Region: {regions_str}  |  Entity: {entities_str}\n\n"
                
                ax4.text(0.02, 0.95, obs_text, transform=ax4.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                 edgecolor='#f0ad4e', alpha=0.95))
                
                plt.tight_layout()
                out_path = viz_dir / f"network_{successful+1:02d}_{patient_id}_{study_id}.png"
                plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                logger.info(f"  [OK] Pipeline-style graph saved: {out_path.name}")
                successful += 1
                
            except Exception as e:
                logger.debug(f"Scene graph visualization failed for {sg_file}: {e}")
                traceback.print_exc()
                continue
        
        if successful > 0:
            logger.info(f"\n[NETWORK] {successful} pipeline-style network graphs saved to: {viz_dir}")
        
        # Also create the old-style summary panels
        self._visualize_scene_graph_panels(num_samples)
    
    def _visualize_scene_graph_panels(self, num_samples: int = 5):
        """
        Create summary panel visualizations for scene graphs.
        Shows categories, polarity, and clinical context in a clean layout.
        """
        scene_data_dir = self.mimic_qa_path / 'scene_data'
        if not scene_data_dir.exists():
            return
        
        sg_files = list(scene_data_dir.rglob('*.scene_graph.json'))
        if not sg_files:
            return
        
        sample_files = random.sample(sg_files, min(num_samples * 2, len(sg_files)))
        viz_dir = self.output_dir / 'scene_graph_panels'
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
                
                # Create clean 2x2 panel
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.patch.set_facecolor('white')
                
                # Panel 1: Observation Categories
                ax1 = axes[0, 0]
                categories = Counter()
                for obs in observations.values():
                    for cat in obs.get('obs_categories', []):
                        categories[cat] += 1
                
                if categories:
                    cats = list(categories.keys())[:8]
                    counts = [categories[c] for c in cats]
                    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cats)))
                    bars = ax1.barh(cats, counts, color=colors, edgecolor='white', linewidth=0.5)
                    ax1.set_xlabel('Count', fontsize=10)
                    ax1.set_title('Observation Categories', fontsize=12, fontweight='bold')
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    
                    # Add count labels
                    for bar, count in zip(bars, counts):
                        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                                str(count), va='center', fontsize=9)
                else:
                    ax1.text(0.5, 0.5, "No categories", ha='center', va='center', fontsize=12)
                    ax1.set_title('Observation Categories', fontsize=12, fontweight='bold')
                
                # Panel 2: Finding Polarity (Donut Chart)
                ax2 = axes[0, 1]
                polarity = Counter()
                for obs in observations.values():
                    polarity[obs.get('positiveness', 'unknown')] += 1
                
                if polarity:
                    labels = list(polarity.keys())
                    sizes = list(polarity.values())
                    color_map = {'pos': '#e74c3c', 'neg': '#27ae60', 'unknown': '#95a5a6'}
                    colors = [color_map.get(l, '#3498db') for l in labels]
                    
                    # Donut chart
                    wedges, texts, autotexts = ax2.pie(
                        sizes, labels=labels, autopct='%1.0f%%',
                        colors=colors, explode=[0.02] * len(labels),
                        wedgeprops=dict(width=0.6, edgecolor='white'),
                        textprops={'fontsize': 10}
                    )
                    ax2.set_title('Finding Polarity Distribution', fontsize=12, fontweight='bold')
                    
                    # Center text
                    total = sum(sizes)
                    ax2.text(0, 0, f'{total}\nfindings', ha='center', va='center', 
                            fontsize=14, fontweight='bold')
                
                # Panel 3: Region Coverage
                ax3 = axes[1, 0]
                region_counts = Counter()
                for obs in observations.values():
                    for region in obs.get('obs_regions', []):
                        region_counts[region] += 1
                
                if region_counts:
                    top_regions = region_counts.most_common(10)
                    regions_list = [r[0] for r in top_regions]
                    counts = [r[1] for r in top_regions]
                    
                    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(regions_list)))
                    bars = ax3.barh(regions_list[::-1], counts[::-1], color=colors[::-1], 
                                   edgecolor='white', linewidth=0.5)
                    ax3.set_xlabel('Mention Count', fontsize=10)
                    ax3.set_title('Top Anatomical Regions', fontsize=12, fontweight='bold')
                    ax3.spines['top'].set_visible(False)
                    ax3.spines['right'].set_visible(False)
                else:
                    ax3.text(0.5, 0.5, "No regions", ha='center', va='center', fontsize=12)
                    ax3.set_title('Top Anatomical Regions', fontsize=12, fontweight='bold')
                
                # Panel 4: Clinical Summary Text
                ax4 = axes[1, 1]
                ax4.axis('off')
                
                indication = sg.get('indication', {})
                
                summary_text = f"CLINICAL SUMMARY\n{'─' * 40}\n\n"
                summary_text += f"Patient ID: {patient_id}\n"
                summary_text += f"Study ID: {study_id}\n\n"
                summary_text += f"Observations: {len(observations)}\n"
                summary_text += f"Regions: {len(regions)}\n"
                summary_text += f"Report Sentences: {len(sentences)}\n\n"
                
                if indication:
                    if indication.get('indication_summary'):
                        summary_text += f"Indication:\n  {indication['indication_summary'][:100]}...\n\n"
                    if indication.get('patient_info'):
                        summary_text += f"Patient Info:\n  {indication['patient_info'][:80]}\n"
                
                ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                                 edgecolor='#dee2e6', alpha=0.9))
                
                fig.suptitle(f'Scene Graph Analysis: {patient_id} / {study_id}', 
                            fontsize=14, fontweight='bold', y=1.02)
                
                plt.tight_layout()
                out_path = viz_dir / f"panel_{successful+1:02d}_{patient_id}_{study_id}.png"
                plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                logger.info(f"  [OK] Panel saved: {out_path.name}")
                successful += 1
                
            except Exception as e:
                logger.debug(f"Panel visualization failed for {sg_file}: {e}")
                continue


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

