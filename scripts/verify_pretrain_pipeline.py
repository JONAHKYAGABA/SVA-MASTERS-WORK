#!/usr/bin/env python3
"""
Pretraining Pipeline Verification Script

Tests the complete data → model → loss → metrics pipeline
BEFORE running full training to catch any errors early.

Usage:
    python scripts/verify_pretrain_pipeline.py --config configs/pretrain_config.yaml
    
This script will:
1. Load a small sample of the dataset
2. Initialize the model
3. Run forward pass
4. Compute loss
5. Compute backward pass (gradient check)
6. Report any issues found

Run this BEFORE starting full pretraining!
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path - handle both direct run and module run
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Change to project root for relative imports

# Debug path info
print(f"Project root: {project_root}")
print(f"Python path includes: {project_root}")

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_imports():
    """Verify all required imports work."""
    logger.info("=" * 60)
    logger.info("STEP 1: Verifying imports...")
    logger.info("=" * 60)
    
    issues = []
    
    # Core imports
    try:
        from configs.mimic_cxr_config import (
            MIMICCXRVQAConfig, 
            get_pretrain_config,
            load_config_from_file
        )
        logger.info("  [OK] configs.mimic_cxr_config")
    except Exception as e:
        issues.append(f"configs.mimic_cxr_config: {e}")
        logger.error(f"  [FAIL] configs.mimic_cxr_config: {e}")
    
    try:
        from data.mimic_cxr_dataset import (
            MIMICCXRVQADataset,
            create_dataloader,
            collate_fn,
            QUESTION_TYPE_MAP,
            CHEXPERT_CATEGORIES
        )
        logger.info("  [OK] data.mimic_cxr_dataset")
        logger.info(f"       Question types defined: {len(QUESTION_TYPE_MAP)}")
        logger.info(f"       CheXpert categories: {len(CHEXPERT_CATEGORIES)}")
    except Exception as e:
        issues.append(f"data.mimic_cxr_dataset: {e}")
        logger.error(f"  [FAIL] data.mimic_cxr_dataset: {e}")
    
    try:
        from models.mimic_vqa_model import MIMICCXRVQAModel, MIMICVQAOutput
        logger.info("  [OK] models.mimic_vqa_model")
    except Exception as e:
        issues.append(f"models.mimic_vqa_model: {e}")
        logger.error(f"  [FAIL] models.mimic_vqa_model: {e}")
    
    try:
        from training.loss import MultiTaskLoss
        logger.info("  [OK] training.loss")
    except Exception as e:
        issues.append(f"training.loss: {e}")
        logger.error(f"  [FAIL] training.loss: {e}")
    
    try:
        from training.metrics import VQAMetrics
        logger.info("  [OK] training.metrics")
    except Exception as e:
        issues.append(f"training.metrics: {e}")
        logger.error(f"  [FAIL] training.metrics: {e}")
    
    return issues


def verify_config(config_path: str = None):
    """Verify configuration loading."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Verifying configuration...")
    logger.info("=" * 60)
    
    from configs.mimic_cxr_config import (
        get_pretrain_config, 
        load_config_from_file,
        MIMICCXRVQAConfig
    )
    
    if config_path and Path(config_path).exists():
        config = load_config_from_file(config_path)
        logger.info(f"  [OK] Loaded config from: {config_path}")
    else:
        config = get_pretrain_config()
        logger.info("  [OK] Using default pretrain config")
    
    # Print key settings
    logger.info(f"\n  Configuration Summary:")
    logger.info(f"  ─────────────────────────────────────")
    logger.info(f"  Model:")
    logger.info(f"    - Visual backbone: {config.model.visual_backbone}")
    logger.info(f"    - Text encoder: {config.model.text_encoder}")
    logger.info(f"    - Hidden size: {config.model.hidden_size}")
    logger.info(f"    - Scene graph dim: {config.model.scene_graph_dim}")
    logger.info(f"  Data:")
    logger.info(f"    - Quality grade: {config.data.quality_grade}")
    logger.info(f"    - View filter: {config.data.view_filter}")
    logger.info(f"    - MIMIC-CXR path: {config.data.mimic_cxr_jpg_path}")
    logger.info(f"    - MIMIC-QA path: {config.data.mimic_ext_cxr_qba_path}")
    logger.info(f"  Training:")
    logger.info(f"    - Batch size: {config.training.batch_size_per_gpu}")
    logger.info(f"    - Grad accum: {config.training.gradient_accumulation_steps}")
    logger.info(f"    - Learning rate: {config.training.learning_rate}")
    logger.info(f"    - FP16: {config.training.fp16}")
    
    # Verify paths exist
    issues = []
    
    cxr_path = Path(config.data.mimic_cxr_jpg_path)
    if not cxr_path.exists():
        issues.append(f"MIMIC-CXR path not found: {cxr_path}")
        logger.warning(f"  [WARN] MIMIC-CXR path not found: {cxr_path}")
    else:
        logger.info(f"  [OK] MIMIC-CXR path exists")
    
    qa_path = Path(config.data.mimic_ext_cxr_qba_path)
    if not qa_path.exists():
        issues.append(f"MIMIC-QA path not found: {qa_path}")
        logger.warning(f"  [WARN] MIMIC-QA path not found: {qa_path}")
    else:
        # Check for extracted data
        qa_dir = qa_path / 'qa'
        scene_dir = qa_path / 'scene_data'
        
        if not qa_dir.exists():
            issues.append(f"QA directory not found: {qa_dir} (need to extract qa.zip)")
            logger.warning(f"  [WARN] qa/ directory not found - extract qa.zip!")
        else:
            logger.info(f"  [OK] qa/ directory exists")
            
        if not scene_dir.exists():
            issues.append(f"Scene data not found: {scene_dir} (need to extract scene_data.zip)")
            logger.warning(f"  [WARN] scene_data/ directory not found - extract scene_data.zip!")
        else:
            logger.info(f"  [OK] scene_data/ directory exists")
    
    return config, issues


def verify_dataset(config, max_samples: int = 10):
    """Verify dataset loading."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Verifying dataset loading...")
    logger.info("=" * 60)
    
    from data.mimic_cxr_dataset import (
        MIMICCXRVQADataset,
        create_dataloader,
        QUESTION_TYPE_MAP
    )
    
    issues = []
    
    try:
        dataset = MIMICCXRVQADataset(
            mimic_cxr_path=config.data.mimic_cxr_jpg_path,
            mimic_qa_path=config.data.mimic_ext_cxr_qba_path,
            split='train',
            tokenizer_name=config.model.text_encoder,
            max_question_length=config.model.max_question_length,
            quality_grade=config.data.quality_grade,
            view_filter=config.data.view_filter,
            max_samples=max_samples,
        )
        
        logger.info(f"  [OK] Dataset loaded with {len(dataset)} samples")
        
        # Test single sample
        sample = dataset[0]
        
        logger.info(f"\n  Sample structure:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"    - {key}: Tensor shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                logger.info(f"    - {key}: Dict with {len(value)} keys: {list(value.keys())[:5]}...")
            elif isinstance(value, list):
                logger.info(f"    - {key}: List len={len(value)}")
            else:
                logger.info(f"    - {key}: {type(value).__name__} = {str(value)[:50]}")
        
        # Verify question type is in mapping
        q_type = sample['question_types']
        head = QUESTION_TYPE_MAP.get(q_type)
        if head:
            logger.info(f"  [OK] Question type '{q_type}' -> head '{head}'")
        else:
            issues.append(f"Unknown question type: {q_type}")
            logger.warning(f"  [WARN] Unknown question type: {q_type}")
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=min(4, max_samples),
            shuffle=False,
            num_workers=0  # Use 0 for testing
        )
        
        batch = next(iter(dataloader))
        
        logger.info(f"\n  Batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"    - {key}: Tensor shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    logger.info(f"    - {key}: List[Dict] len={len(value)}, keys={list(value[0].keys())[:5]}")
                else:
                    logger.info(f"    - {key}: List len={len(value)}, sample={str(value[0])[:30] if value else 'empty'}")
        
        return dataset, dataloader, issues
        
    except Exception as e:
        import traceback
        issues.append(f"Dataset loading failed: {e}")
        logger.error(f"  [FAIL] Dataset loading failed: {e}")
        traceback.print_exc()
        return None, None, issues


def verify_model(config):
    """Verify model initialization."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Verifying model initialization...")
    logger.info("=" * 60)
    
    from models.mimic_vqa_model import MIMICCXRVQAModel
    
    issues = []
    
    try:
        model = MIMICCXRVQAModel(
            visual_backbone=config.model.visual_backbone,
            text_encoder=config.model.text_encoder,
            visual_feature_dim=config.model.visual_feature_dim,
            scene_graph_dim=config.model.scene_graph_dim,
            num_regions=config.model.num_regions,
            num_entities=config.model.num_entities,
            hidden_size=config.model.hidden_size,
            num_hidden_layers=config.model.num_hidden_layers,
            num_attention_heads=config.model.num_attention_heads,
            sim_layers=config.model.sim_layers,
            num_binary_classes=config.model.num_binary_classes,
            num_category_classes=config.model.num_category_classes,
            num_region_classes=config.model.num_region_classes,
            num_severity_classes=config.model.num_severity_classes,
            dropout=config.model.hidden_dropout_prob,
            use_chexpert_head=True,
            gradient_checkpointing=config.training.gradient_checkpointing
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"  [OK] Model initialized")
        logger.info(f"       Total parameters: {total_params:,}")
        logger.info(f"       Trainable parameters: {trainable_params:,}")
        logger.info(f"       Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        
        # Print model structure
        logger.info(f"\n  Model components:")
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            logger.info(f"    - {name}: {type(module).__name__} ({num_params:,} params)")
        
        return model, issues
        
    except Exception as e:
        import traceback
        issues.append(f"Model initialization failed: {e}")
        logger.error(f"  [FAIL] Model initialization failed: {e}")
        traceback.print_exc()
        return None, issues


def verify_forward_pass(model, batch, config, device):
    """Verify forward pass."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Verifying forward pass...")
    logger.info("=" * 60)
    
    issues = []
    
    try:
        model = model.to(device)
        model.eval()
        
        # Move batch to device
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
        scene_graphs = batch['scene_graphs']
        question_types = batch['question_types']
        image_widths = batch.get('image_widths', torch.full((images.shape[0],), 224, dtype=torch.long)).to(device)
        image_heights = batch.get('image_heights', torch.full((images.shape[0],), 224, dtype=torch.long)).to(device)
        
        logger.info(f"  Input shapes:")
        logger.info(f"    - images: {images.shape}")
        logger.info(f"    - input_ids: {input_ids.shape}")
        logger.info(f"    - scene_graphs: {len(scene_graphs)} items")
        logger.info(f"    - question_types: {question_types}")
        
        with torch.no_grad():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                scene_graphs=scene_graphs,
                question_types=question_types,
                image_widths=image_widths,
                image_heights=image_heights
            )
        
        logger.info(f"\n  [OK] Forward pass successful!")
        logger.info(f"  Output structure:")
        logger.info(f"    - vqa_logits:")
        for head_name, logits in outputs.vqa_logits.items():
            logger.info(f"        {head_name}: {logits.shape}")
        
        if outputs.chexpert_logits is not None:
            logger.info(f"    - chexpert_logits: {outputs.chexpert_logits.shape}")
        logger.info(f"    - pooled_output: {outputs.pooled_output.shape}")
        
        return outputs, issues
        
    except Exception as e:
        import traceback
        issues.append(f"Forward pass failed: {e}")
        logger.error(f"  [FAIL] Forward pass failed: {e}")
        traceback.print_exc()
        return None, issues


def verify_loss_computation(outputs, batch, config, device):
    """Verify loss computation."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Verifying loss computation...")
    logger.info("=" * 60)
    
    from training.loss import MultiTaskLoss
    
    issues = []
    
    try:
        criterion = MultiTaskLoss(
            vqa_weight=config.training.vqa_loss_weight,
            chexpert_weight=config.training.chexpert_loss_weight,
            binary_weight=config.training.binary_head_weight,
            category_weight=config.training.category_head_weight,
            region_weight=config.training.region_head_weight,
            severity_weight=config.training.severity_head_weight,
        )
        
        answer_idx = batch['answer_idx'].to(device)
        chexpert_labels = batch['chexpert_labels'].to(device)
        chexpert_mask = batch['chexpert_mask'].to(device)
        question_types = batch['question_types']
        
        vqa_targets = {
            'binary': answer_idx,
            'category': answer_idx,
            'region': answer_idx,
            'severity': answer_idx,
        }
        
        loss, loss_dict = criterion(
            outputs,
            vqa_targets,
            chexpert_labels,
            chexpert_mask,
            question_types
        )
        
        logger.info(f"  [OK] Loss computation successful!")
        logger.info(f"  Loss values:")
        logger.info(f"    - Total loss: {loss.item():.4f}")
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"    - {name}: {value.item():.4f}")
            else:
                logger.info(f"    - {name}: {value:.4f}")
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            issues.append("Loss is NaN or Inf!")
            logger.error("  [FAIL] Loss is NaN or Inf!")
        
        return loss, loss_dict, issues
        
    except Exception as e:
        import traceback
        issues.append(f"Loss computation failed: {e}")
        logger.error(f"  [FAIL] Loss computation failed: {e}")
        traceback.print_exc()
        return None, None, issues


def verify_backward_pass(model, batch, config, device):
    """Verify backward pass and gradients."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Verifying backward pass...")
    logger.info("=" * 60)
    
    from training.loss import MultiTaskLoss
    
    issues = []
    
    try:
        # Set model to train mode and enable gradients
        model.train()
        model.zero_grad()
        
        # Re-run forward pass WITH gradients
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
        scene_graphs = batch['scene_graphs']
        question_types = batch['question_types']
        image_widths = batch.get('image_widths', torch.full((images.shape[0],), 224, dtype=torch.long)).to(device)
        image_heights = batch.get('image_heights', torch.full((images.shape[0],), 224, dtype=torch.long)).to(device)
        
        # Forward pass WITH gradients
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            scene_graphs=scene_graphs,
            question_types=question_types,
            image_widths=image_widths,
            image_heights=image_heights
        )
        
        # Compute loss
        criterion = MultiTaskLoss(
            vqa_weight=config.training.vqa_loss_weight,
            chexpert_weight=config.training.chexpert_loss_weight,
            binary_weight=config.training.binary_head_weight,
            category_weight=config.training.category_head_weight,
            region_weight=config.training.region_head_weight,
            severity_weight=config.training.severity_head_weight,
        )
        
        answer_idx = batch['answer_idx'].to(device)
        chexpert_labels = batch['chexpert_labels'].to(device)
        chexpert_mask = batch['chexpert_mask'].to(device)
        
        vqa_targets = {
            'binary': answer_idx,
            'category': answer_idx,
            'region': answer_idx,
            'severity': answer_idx,
        }
        
        loss, _ = criterion(
            outputs,
            vqa_targets,
            chexpert_labels,
            chexpert_mask,
            question_types
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = {}
        has_nan_grad = False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                
                if np.isnan(grad_norm) or np.isinf(grad_norm):
                    has_nan_grad = True
                    issues.append(f"NaN/Inf gradient in {name}")
        
        if has_nan_grad:
            logger.error("  [FAIL] Found NaN/Inf gradients!")
        else:
            logger.info("  [OK] Backward pass successful!")
            
        # Summary statistics
        grad_values = list(grad_norms.values())
        logger.info(f"\n  Gradient statistics:")
        logger.info(f"    - Parameters with gradients: {len(grad_values)}")
        if grad_values:
            logger.info(f"    - Min gradient norm: {min(grad_values):.6f}")
            logger.info(f"    - Max gradient norm: {max(grad_values):.6f}")
            logger.info(f"    - Mean gradient norm: {np.mean(grad_values):.6f}")
        
        return issues
        
    except Exception as e:
        import traceback
        issues.append(f"Backward pass failed: {e}")
        logger.error(f"  [FAIL] Backward pass failed: {e}")
        traceback.print_exc()
        return issues


def main():
    parser = argparse.ArgumentParser(description='Verify pretraining pipeline')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum samples to test')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  PRETRAINING PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    all_issues = []
    
    # Step 1: Verify imports
    issues = verify_imports()
    all_issues.extend(issues)
    if issues:
        logger.error("\nImport errors found. Cannot continue.")
        return 1
    
    # Step 2: Verify config
    config, issues = verify_config(args.config)
    all_issues.extend(issues)
    
    # Step 3: Verify dataset
    dataset, dataloader, issues = verify_dataset(config, args.max_samples)
    all_issues.extend(issues)
    if dataset is None:
        logger.error("\nDataset loading failed. Cannot continue.")
        return 1
    
    # Step 4: Verify model
    model, issues = verify_model(config)
    all_issues.extend(issues)
    if model is None:
        logger.error("\nModel initialization failed. Cannot continue.")
        return 1
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Step 5: Verify forward pass
    outputs, issues = verify_forward_pass(model, batch, config, device)
    all_issues.extend(issues)
    if outputs is None:
        logger.error("\nForward pass failed. Cannot continue.")
        return 1
    
    # Step 6: Verify loss
    loss, loss_dict, issues = verify_loss_computation(outputs, batch, config, device)
    all_issues.extend(issues)
    if loss is None:
        logger.error("\nLoss computation failed. Cannot continue.")
        return 1
    
    # Step 7: Verify backward (re-runs forward with gradients enabled)
    issues = verify_backward_pass(model, batch, config, device)
    all_issues.extend(issues)
    
    # Final summary
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    
    if all_issues:
        print(f"\n⚠️  Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("\n❌ PRETRAINING PIPELINE HAS ISSUES - Fix before training!")
        return 1
    else:
        print("\n✓ All checks passed!")
        print("\n✅ PRETRAINING PIPELINE READY")
        print("\nYou can now start training with:")
        print(f"  deepspeed --num_gpus=4 train_mimic_cxr.py --config {args.config or 'configs/pretrain_config.yaml'}")
        return 0


if __name__ == '__main__':
    sys.exit(main())

