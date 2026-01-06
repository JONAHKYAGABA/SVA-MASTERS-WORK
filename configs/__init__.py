"""
MIMIC-CXR VQA Configuration Module

Provides configuration classes for:
- Model architecture
- Training hyperparameters
- Data loading
- Ablation study conditions (methodology Section 16.1)
"""

from .mimic_cxr_config import (
    MIMICCXRVQAConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    WandbConfig,
    DeepSpeedConfig,
    get_default_config,
    get_debug_config,
    get_pretrain_config,
    get_finetune_config,
    load_config_from_file,
)

from .ablations import (
    get_ablation_config,
    get_all_ablation_configs,
    ABLATION_CONDITIONS,
    AblationCondition,
)

__all__ = [
    # Main configs
    'MIMICCXRVQAConfig',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'WandbConfig',
    'DeepSpeedConfig',
    'get_default_config',
    'get_debug_config',
    'get_pretrain_config',
    'get_finetune_config',
    'load_config_from_file',
    # Ablation configs (methodology Section 16.1)
    'get_ablation_config',
    'get_all_ablation_configs',
    'ABLATION_CONDITIONS',
    'AblationCondition',
]
