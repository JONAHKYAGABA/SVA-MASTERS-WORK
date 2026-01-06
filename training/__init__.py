"""
MIMIC-CXR VQA Training Module

Provides loss functions and metrics for training.
"""

from .loss import (
    MultiTaskLoss,
    FocalLoss,
    MultiLabelFocalLoss,
)

from .metrics import (
    VQAMetrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
)

__all__ = [
    'MultiTaskLoss',
    'FocalLoss',
    'MultiLabelFocalLoss',
    'VQAMetrics',
    'compute_confusion_matrix',
    'compute_per_class_metrics',
]
