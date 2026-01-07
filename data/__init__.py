"""
MIMIC-CXR VQA Data Module

Provides dataset classes and data loading utilities for:
- MIMIC-CXR-JPG images
- MIMIC-Ext-CXR-QBA scene graphs and QA pairs
- CheXpert labels
- External datasets (VQA-RAD, SLAKE) for cross-dataset evaluation
"""

from .mimic_cxr_dataset import (
    MIMICCXRVQADataset,
    CheXpertLabelLoader,
    SceneGraphProcessor,
    create_dataloader,
    collate_fn,
    CHEXPERT_CATEGORIES,
    QUESTION_TYPE_MAP,
)

# External datasets are optional (for cross-dataset evaluation only)
try:
    from .external_datasets import (
        VQARADDataset,
        SLAKEDataset,
    )
    EXTERNAL_DATASETS_AVAILABLE = True
except ImportError as e:
    VQARADDataset = None
    SLAKEDataset = None
    EXTERNAL_DATASETS_AVAILABLE = False

__all__ = [
    # MIMIC-CXR (Primary)
    'MIMICCXRVQADataset',
    'CheXpertLabelLoader',
    'SceneGraphProcessor',
    'create_dataloader',
    'collate_fn',
    'CHEXPERT_CATEGORIES',
    'QUESTION_TYPE_MAP',
    # External Datasets (Optional - for cross-dataset evaluation)
    'VQARADDataset',
    'SLAKEDataset',
    'EXTERNAL_DATASETS_AVAILABLE',
]
