"""
Ablation Study Configuration Definitions

From methodology Section 16.1 - All conditions trained from scratch
under identical settings on MIMIC-Ext-CXR-QA only.

Ablation Conditions:
====================

CONDITION 1: FULL PROPOSED MODEL (SG-Enhanced) [BASELINE]
- ConvNeXt-Base visual backbone
- ROI Align using scene graph bboxes
- Scene Graph: Full 134-dim embeddings
- Bio+ClinicalBERT text encoder
- Scene-embedded Interaction Module (SIM) - 2 layers
- Multi-head answers (binary + category + region + severity)
- Total Parameters: ~150M

CONDITION 2: NO-SG BASELINE (Scene Graph Removed)
- ❌ Scene Graph Encoder removed
- ❌ Scene-embedded Interaction Module removed
- ❌ ROI Align removed
- ✓ ConvNeXt-Base (global features only)
- ✓ Bio+ClinicalBERT
- ✓ Multi-head answer architecture
- Purpose: Quantify scene graph contribution

CONDITION 3: VISION+QUESTION BASELINE (Minimal)
- ConvNeXt-Base (global pooled features only)
- Bio+ClinicalBERT (question encoding)
- Simple late fusion (concatenate + MLP)
- ❌ All region-level features removed
- ❌ All scene graph components removed
- ❌ Cross-modal attention mechanisms removed
- Purpose: Lower bound - vision+language alone

CONDITION 4: ResNet18 BACKBONE (Original SSG-VQA)
- Replace ConvNeXt-Base with ResNet18
- Purpose: Quantify ConvNeXt upgrade benefit (+4-7% expected)

CONDITION 5: GENERIC TOKENIZER (Original SSG-VQA)
- Replace Bio+ClinicalBERT with standard BERT
- Purpose: Quantify medical tokenizer benefit (+2.8-4.5% expected)

CONDITION 6: SINGLE-HEAD ANSWER (Unified Classification)
- Single classification head (500 unified classes)
- Instead of multi-head architecture
- Purpose: Quantify multi-head architecture benefit
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class AblationCondition(str, Enum):
    """Enumeration of ablation conditions."""
    FULL_MODEL = "full_model"
    NO_SCENE_GRAPH = "no_sg"
    VISION_QUESTION = "vision_question"
    RESNET18_BACKBONE = "resnet18"
    GENERIC_TOKENIZER = "generic_tokenizer"
    SINGLE_HEAD = "single_head"


@dataclass
class AblationConfig:
    """Configuration for a single ablation condition."""
    
    # Condition identifier
    name: str
    condition: AblationCondition
    description: str
    
    # Model architecture
    visual_backbone: str = "convnext_base"
    text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT"
    
    # Scene graph settings
    use_scene_graph: bool = True
    scene_graph_dim: int = 134
    use_roi_features: bool = True
    use_sim_module: bool = True  # Scene-embedded Interaction Module
    
    # Answer architecture
    use_multi_head: bool = True
    num_unified_classes: int = 500  # For single-head ablation
    
    # Components to disable
    disable_components: List[str] = field(default_factory=list)
    
    # Training overrides
    training_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Expected performance delta (for reference)
    expected_delta: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'condition': self.condition.value,
            'description': self.description,
            'visual_backbone': self.visual_backbone,
            'text_encoder': self.text_encoder,
            'use_scene_graph': self.use_scene_graph,
            'scene_graph_dim': self.scene_graph_dim,
            'use_roi_features': self.use_roi_features,
            'use_sim_module': self.use_sim_module,
            'use_multi_head': self.use_multi_head,
            'num_unified_classes': self.num_unified_classes,
            'disable_components': self.disable_components,
            'training_overrides': self.training_overrides,
            'expected_delta': self.expected_delta,
        }


# =============================================================================
# Ablation Condition Definitions
# =============================================================================

ABLATION_CONDITIONS: Dict[AblationCondition, AblationConfig] = {
    
    # CONDITION 1: Full Proposed Model (Baseline)
    AblationCondition.FULL_MODEL: AblationConfig(
        name="Full Model (SG-Enhanced)",
        condition=AblationCondition.FULL_MODEL,
        description=(
            "Complete architecture with ConvNeXt-Base, YOLOv8-derived region features, "
            "Bio+ClinicalBERT, and full Scene-embedded Interaction Module. "
            "This is the BASELINE for all comparisons."
        ),
        visual_backbone="convnext_base",
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        use_scene_graph=True,
        scene_graph_dim=134,
        use_roi_features=True,
        use_sim_module=True,
        use_multi_head=True,
        expected_delta={
            "overall_accuracy": "Baseline",
            "spatial_accuracy": "Baseline",
            "relational_accuracy": "Baseline",
        }
    ),
    
    # CONDITION 2: No Scene Graph
    AblationCondition.NO_SCENE_GRAPH: AblationConfig(
        name="No-SG Baseline",
        condition=AblationCondition.NO_SCENE_GRAPH,
        description=(
            "Scene graph encoder and all graph-related interaction modules removed. "
            "Tests model without any graph supervision - strong vision-language baseline."
        ),
        visual_backbone="convnext_base",
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        use_scene_graph=False,
        scene_graph_dim=0,
        use_roi_features=False,  # No bboxes without scene graph
        use_sim_module=False,    # SIM requires scene graph
        use_multi_head=True,
        disable_components=[
            "scene_graph_encoder",
            "scene_proj",
            "sim",  # Scene-embedded Interaction Module
        ],
        expected_delta={
            "overall_accuracy": "-8 to -12%",
            "spatial_accuracy": "-15 to -25%",
            "relational_accuracy": "-20 to -30%",
            "training_loss": "+10-15%",
        }
    ),
    
    # CONDITION 3: Vision+Question Baseline
    AblationCondition.VISION_QUESTION: AblationConfig(
        name="Vision+Question Baseline",
        condition=AblationCondition.VISION_QUESTION,
        description=(
            "Minimal ablation with only global visual features and text encoding. "
            "No region features, no scene graph, no cross-modal attention. "
            "Lower bound showing what vision+language alone achieves."
        ),
        visual_backbone="convnext_base",
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        use_scene_graph=False,
        scene_graph_dim=0,
        use_roi_features=False,
        use_sim_module=False,
        use_multi_head=True,
        disable_components=[
            "scene_graph_encoder",
            "scene_proj",
            "sim",
            "roi_align",
            "cross_attention",  # Use simple late fusion instead
        ],
        training_overrides={
            "fusion_type": "late_fusion",  # Simple concatenate + MLP
        },
        expected_delta={
            "overall_accuracy": "-12 to -18%",
            "spatial_accuracy": "-25 to -35%",
            "relational_accuracy": "-30 to -40%",
            "training_loss": "+15-20%",
        }
    ),
    
    # CONDITION 4: ResNet18 Backbone
    AblationCondition.RESNET18_BACKBONE: AblationConfig(
        name="ResNet18 Backbone",
        condition=AblationCondition.RESNET18_BACKBONE,
        description=(
            "Original SSG-VQA configuration with ResNet18 instead of ConvNeXt-Base. "
            "Quantifies the visual backbone upgrade benefit."
        ),
        visual_backbone="resnet18",  # Changed from convnext_base
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        use_scene_graph=True,
        scene_graph_dim=134,
        use_roi_features=True,
        use_sim_module=True,
        use_multi_head=True,
        training_overrides={
            "visual_feature_dim": 512,  # ResNet18 output dim
        },
        expected_delta={
            "overall_accuracy": "-4 to -7%",
            "spatial_accuracy": "-6 to -10%",
            "relational_accuracy": "-5 to -8%",
            "training_loss": "+5-8%",
        }
    ),
    
    # CONDITION 5: Generic Tokenizer
    AblationCondition.GENERIC_TOKENIZER: AblationConfig(
        name="Generic Tokenizer",
        condition=AblationCondition.GENERIC_TOKENIZER,
        description=(
            "Standard BERT tokenizer instead of Bio+ClinicalBERT. "
            "Quantifies the medical domain tokenizer benefit."
        ),
        visual_backbone="convnext_base",
        text_encoder="bert-base-uncased",  # Changed from Bio+ClinicalBERT
        use_scene_graph=True,
        scene_graph_dim=134,
        use_roi_features=True,
        use_sim_module=True,
        use_multi_head=True,
        expected_delta={
            "overall_accuracy": "-2.8 to -4.5%",
            "spatial_accuracy": "-2 to -4%",
            "relational_accuracy": "-3 to -5%",
            "training_loss": "+3-5%",
        }
    ),
    
    # CONDITION 6: Single-Head Answer
    AblationCondition.SINGLE_HEAD: AblationConfig(
        name="Single-Head Answer",
        condition=AblationCondition.SINGLE_HEAD,
        description=(
            "Single unified classification head instead of multi-head architecture. "
            "Tests benefit of specialized heads per answer type."
        ),
        visual_backbone="convnext_base",
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        use_scene_graph=True,
        scene_graph_dim=134,
        use_roi_features=True,
        use_sim_module=True,
        use_multi_head=False,  # Changed
        num_unified_classes=500,  # Single head with many classes
        expected_delta={
            "overall_accuracy": "-3 to -6%",
            "spatial_accuracy": "-2 to -4%",
            "relational_accuracy": "-2 to -4%",
            "training_loss": "+2-4%",
        }
    ),
}


# =============================================================================
# Factory Functions
# =============================================================================

def get_ablation_config(condition: AblationCondition) -> AblationConfig:
    """
    Get configuration for a specific ablation condition.
    
    Args:
        condition: AblationCondition enum value
    
    Returns:
        AblationConfig for the condition
    """
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(f"Unknown ablation condition: {condition}")
    
    return ABLATION_CONDITIONS[condition]


def get_all_ablation_configs() -> Dict[AblationCondition, AblationConfig]:
    """
    Get all ablation configurations.
    
    Returns:
        Dict mapping condition to config
    """
    return ABLATION_CONDITIONS.copy()


def create_ablation_experiment_matrix() -> List[Dict[str, Any]]:
    """
    Create experiment matrix for all ablation comparisons.
    
    Returns:
        List of experiment configurations
    """
    experiments = []
    baseline = get_ablation_config(AblationCondition.FULL_MODEL)
    
    for condition, config in ABLATION_CONDITIONS.items():
        if condition == AblationCondition.FULL_MODEL:
            continue  # Skip baseline vs baseline
        
        experiments.append({
            'baseline': baseline.to_dict(),
            'ablation': config.to_dict(),
            'comparison_name': f"Full Model vs {config.name}",
            'hypothesis': f"Scene graph contribution from {config.description}",
        })
    
    return experiments


def print_ablation_summary():
    """Print summary of all ablation conditions."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY CONDITIONS (from methodology Section 16.1)")
    print("=" * 80)
    
    for i, (condition, config) in enumerate(ABLATION_CONDITIONS.items(), 1):
        is_baseline = condition == AblationCondition.FULL_MODEL
        marker = "[BASELINE]" if is_baseline else ""
        
        print(f"\nCONDITION {i}: {config.name} {marker}")
        print("-" * 60)
        print(f"  Visual Backbone:    {config.visual_backbone}")
        print(f"  Text Encoder:       {config.text_encoder}")
        print(f"  Scene Graph:        {'✓' if config.use_scene_graph else '❌'}")
        print(f"  ROI Features:       {'✓' if config.use_roi_features else '❌'}")
        print(f"  SIM Module:         {'✓' if config.use_sim_module else '❌'}")
        print(f"  Multi-Head:         {'✓' if config.use_multi_head else '❌'}")
        
        if config.expected_delta:
            print(f"\n  Expected Delta (vs Baseline):")
            for metric, delta in config.expected_delta.items():
                print(f"    {metric}: {delta}")
    
    print("\n" + "=" * 80)


# =============================================================================
# YAML Export for Experiment Tracking
# =============================================================================

def export_ablation_configs_yaml(output_dir: str = "configs/ablations"):
    """
    Export all ablation configs as individual YAML files.
    
    Args:
        output_dir: Directory to save YAML files
    """
    import yaml
    
    os.makedirs(output_dir, exist_ok=True)
    
    for condition, config in ABLATION_CONDITIONS.items():
        filename = f"{condition.value}_config.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        print(f"Exported: {filepath}")


if __name__ == "__main__":
    # Print summary when run directly
    print_ablation_summary()
    
    # Export YAML configs
    print("\nExporting YAML configs...")
    export_ablation_configs_yaml()

