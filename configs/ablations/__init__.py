"""
Ablation Study Configurations

Implements ablation conditions from methodology Section 16.1:

1. FULL MODEL (SG-Enhanced) - Baseline with all components
2. NO-SG BASELINE - Scene graph removed
3. VISION+QUESTION BASELINE - Minimal late fusion
4. RESNET18 BACKBONE - Original SSG-VQA configuration  
5. GENERIC TOKENIZER - Standard BERT instead of Bio+ClinicalBERT
6. SINGLE-HEAD ANSWER - Unified classification head

All ablation conditions are trained from scratch under identical settings
on MIMIC-Ext-CXR-QA only (no external data).
"""

from .ablation_configs import (
    get_ablation_config,
    get_all_ablation_configs,
    ABLATION_CONDITIONS,
    AblationCondition,
)

__all__ = [
    'get_ablation_config',
    'get_all_ablation_configs',
    'ABLATION_CONDITIONS',
    'AblationCondition',
]

