#!/usr/bin/env python3
"""
MIMIC-CXR VQA Training Script

Complete training pipeline with:
- Weights & Biases logging
- Hugging Face Hub checkpointing
- DeepSpeed distributed training support
- Mixed precision training
- Gradient checkpointing

Based on methodology.md specifications.

IMPORTANT: Run analyze_data.py FIRST to verify data is ready before training!

Usage:
    # Step 1: Analyze data (required)
    python analyze_data.py --mimic_cxr_path /path/to/MIMIC-CXR-JPG --mimic_qa_path /path/to/QA
    
    # Step 2: Train model (only if analysis passes)
    python train_mimic_cxr.py --config configs/default_config.yaml
    
    # Step 3: Evaluate model
    python evaluate.py --model_path ./checkpoints/best_model --config configs/default_config.yaml
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("huggingface_hub not available. Install with: pip install huggingface-hub")

# Local imports
from configs.mimic_cxr_config import (
    MIMICCXRVQAConfig,
    get_default_config,
    load_config_from_file
)
from data.mimic_cxr_dataset import MIMICCXRVQADataset, create_dataloader
from models.mimic_vqa_model import MIMICCXRVQAModel, MIMICVQAOutput
from training.loss import MultiTaskLoss
from training.metrics import VQAMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_wandb(config: MIMICCXRVQAConfig) -> Optional[Any]:
    """Initialize Weights & Biases tracking."""
    if not WANDB_AVAILABLE or not config.wandb.enabled:
        return None
    
    run_name = config.wandb.name or f"ssg-vqa-{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity or None,
        name=run_name,
        group=config.wandb.group,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config=config.to_dict(),
        resume="allow",
        save_code=True,
    )
    
    # Define custom metrics
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("train/vqa_loss", summary="min")
    wandb.define_metric("train/chexpert_loss", summary="min")
    wandb.define_metric("val/accuracy", summary="max")
    wandb.define_metric("val/binary_accuracy", summary="max")
    wandb.define_metric("val/category_f1", summary="max")
    
    return run


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    config: MIMICCXRVQAConfig,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint_dir = Path(config.training.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config.to_dict(),
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}"
    checkpoint_path.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_path / "pytorch_model.bin")
    
    # Save config
    with open(checkpoint_path / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Save training metadata
    metadata = {
        'epoch': epoch,
        'global_step': global_step,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    with open(checkpoint_path / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model"
        best_path.mkdir(exist_ok=True)
        torch.save(checkpoint, best_path / "pytorch_model.bin")
        with open(best_path / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved best model to {best_path}")
    
    # Clean up old checkpoints
    _cleanup_old_checkpoints(checkpoint_dir, config.training.save_total_limit)


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int = 5):
    """Remove old checkpoints, keeping only the most recent ones."""
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )
    
    # Remove old checkpoints
    for checkpoint in checkpoints[:-keep_last]:
        import shutil
        shutil.rmtree(checkpoint)
        logger.info(f"Removed old checkpoint: {checkpoint}")


def push_to_hub(
    model: nn.Module,
    config: MIMICCXRVQAConfig,
    metrics: Dict[str, float],
    commit_message: str = "Training checkpoint"
):
    """Push model to Hugging Face Hub."""
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub not available. Skipping push to hub.")
        return
    
    if not config.training.hub_model_id:
        logger.warning("hub_model_id not set. Skipping push to hub.")
        return
    
    try:
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            create_repo(
                config.training.hub_model_id,
                private=config.training.hub_private_repo,
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Could not create repo: {e}")
        
        # Save model locally first
        save_dir = Path(config.training.output_dir) / "hub_upload"
        save_dir.mkdir(exist_ok=True)
        
        torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
        
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Generate model card
        model_card = _generate_model_card(config, metrics)
        with open(save_dir / "README.md", 'w') as f:
            f.write(model_card)
        
        # Upload
        upload_folder(
            folder_path=str(save_dir),
            repo_id=config.training.hub_model_id,
            commit_message=commit_message,
        )
        
        logger.info(f"Pushed model to {config.training.hub_model_id}")
        
    except Exception as e:
        logger.error(f"Failed to push to hub: {e}")


def _generate_model_card(config: MIMICCXRVQAConfig, metrics: Dict[str, float]) -> str:
    """Generate Hugging Face model card."""
    card = f"""---
language: en
license: mit
library_name: pytorch
tags:
  - medical-vqa
  - chest-x-ray
  - scene-graph
  - visual-question-answering
  - mimic-cxr
datasets:
  - mimic-cxr-jpg
  - mimic-ext-cxr-qba
---

# SSG-VQA-Net for MIMIC-CXR Visual Question Answering

## Model Description

This model adapts the SSG-VQA-Net architecture for chest X-ray visual question answering
using the MIMIC-CXR-JPG images and MIMIC-Ext-CXR-QBA question-answer pairs.

### Architecture

- **Visual Backbone**: ConvNeXt-Base (pre-trained on ImageNet-22k)
- **Text Encoder**: Bio+ClinicalBERT (medical domain)
- **Scene Graph**: Expanded 134-dim embeddings ({config.model.num_regions} regions, {config.model.num_entities} entities)
- **Fusion**: Scene-Embedded Interaction Module (SIM)
- **Answer Heads**: Multi-head (Binary, Category, Region, Severity)

## Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | {metrics.get('accuracy', 'N/A'):.3f if isinstance(metrics.get('accuracy'), float) else 'N/A'} |
| Binary Accuracy | {metrics.get('binary_accuracy', 'N/A'):.3f if isinstance(metrics.get('binary_accuracy'), float) else 'N/A'} |
| Category F1 | {metrics.get('category_f1', 'N/A'):.3f if isinstance(metrics.get('category_f1'), float) else 'N/A'} |
| CheXpert AUROC | {metrics.get('chexpert_auroc', 'N/A'):.3f if isinstance(metrics.get('chexpert_auroc'), float) else 'N/A'} |

## Training Details

- **Batch Size**: {config.training.batch_size_per_gpu} per GPU
- **Learning Rate**: {config.training.learning_rate}
- **Epochs**: {config.training.num_epochs}
- **Mixed Precision**: {'Enabled' if config.training.fp16 else 'Disabled'}

## Usage

```python
from models.mimic_vqa_model import MIMICCXRVQAModel

model = MIMICCXRVQAModel.from_pretrained("{config.training.hub_model_id}")
```

## Citation

```bibtex
@article{{ssg-vqa-mimic,
  title={{Scene Graph-Enhanced VQA for Chest X-Ray Analysis}},
  year={{2026}}
}}
```
"""
    return card


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int,
    config: MIMICCXRVQAConfig,
    scaler: Optional[GradScaler] = None,
    global_step: int = 0
) -> tuple[float, int]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        leave=False
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['images'].to(device) if 'images' in batch else None
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
        scene_graphs = batch['scene_graphs']
        question_types = batch['question_types']
        answer_idx = batch['answer_idx'].to(device)
        chexpert_labels = batch['chexpert_labels'].to(device)
        chexpert_mask = batch['chexpert_mask'].to(device)
        
        # Prepare VQA targets
        vqa_targets = {
            'binary': answer_idx,  # Use same target for all heads for now
            'category': answer_idx,
            'region': answer_idx,
            'severity': answer_idx,
        }
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        if scaler is not None and config.training.fp16:
            with autocast():
                if images is not None:
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        scene_graphs=scene_graphs,
                        token_type_ids=token_type_ids,
                        question_types=question_types
                    )
                else:
                    continue  # Skip if no images
                
                loss, loss_dict = criterion(
                    outputs,
                    vqa_targets,
                    chexpert_labels,
                    chexpert_mask,
                    question_types
                )
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.training.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.max_grad_norm
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            if images is not None:
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    scene_graphs=scene_graphs,
                    token_type_ids=token_type_ids,
                    question_types=question_types
                )
            else:
                continue
            
            loss, loss_dict = criterion(
                outputs,
                vqa_targets,
                chexpert_labels,
                chexpert_mask,
                question_types
            )
            
            loss.backward()
            
            if config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.max_grad_norm
                )
            
            optimizer.step()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'vqa': f'{loss_dict.get("vqa_loss", 0):.4f}',
            'chex': f'{loss_dict.get("chexpert_loss", 0):.4f}'
        })
        
        # Log to wandb
        if WANDB_AVAILABLE and config.wandb.enabled and global_step % config.training.logging_steps == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/vqa_loss': loss_dict.get('vqa_loss', 0).item() if torch.is_tensor(loss_dict.get('vqa_loss', 0)) else loss_dict.get('vqa_loss', 0),
                'train/chexpert_loss': loss_dict.get('chexpert_loss', 0).item() if torch.is_tensor(loss_dict.get('chexpert_loss', 0)) else loss_dict.get('chexpert_loss', 0),
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'global_step': global_step,
            })
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    config: MIMICCXRVQAConfig
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    metrics_calculator = VQAMetrics()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    
    for batch in progress_bar:
        # Move data to device
        images = batch['images'].to(device) if 'images' in batch else None
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
        scene_graphs = batch['scene_graphs']
        question_types = batch['question_types']
        answer_idx = batch['answer_idx'].to(device)
        chexpert_labels = batch['chexpert_labels'].to(device)
        chexpert_mask = batch['chexpert_mask'].to(device)
        
        vqa_targets = {
            'binary': answer_idx,
            'category': answer_idx,
            'region': answer_idx,
            'severity': answer_idx,
        }
        
        if images is None:
            continue
        
        # Forward pass
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            scene_graphs=scene_graphs,
            token_type_ids=token_type_ids,
            question_types=question_types
        )
        
        loss, _ = criterion(
            outputs,
            vqa_targets,
            chexpert_labels,
            chexpert_mask,
            question_types
        )
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update metrics
        metrics_calculator.update(
            outputs,
            vqa_targets,
            chexpert_labels,
            chexpert_mask,
            question_types
        )
    
    # Compute final metrics
    metrics = metrics_calculator.compute()
    metrics['loss'] = total_loss / max(num_batches, 1)
    
    return metrics


def check_data_readiness(config) -> bool:
    """
    Check if data analysis has been run and data is ready.
    
    Returns:
        True if data is ready, False otherwise
    """
    analysis_report_path = Path('./analysis_output/analysis_report.json')
    
    if not analysis_report_path.exists():
        logger.error("=" * 60)
        logger.error("DATA ANALYSIS NOT FOUND!")
        logger.error("=" * 60)
        logger.error("\nPlease run data analysis first:")
        logger.error(f"  python analyze_data.py \\")
        logger.error(f"    --mimic_cxr_path {config.data.mimic_cxr_jpg_path} \\")
        logger.error(f"    --mimic_qa_path {config.data.mimic_ext_cxr_qba_path}")
        logger.error("\nThen run training again.")
        logger.error("=" * 60)
        return False
    
    try:
        with open(analysis_report_path) as f:
            report = json.load(f)
        
        is_ready = report.get('summary', {}).get('is_ready', False)
        
        if not is_ready:
            logger.error("=" * 60)
            logger.error("DATA NOT READY FOR TRAINING!")
            logger.error("=" * 60)
            
            issues = report.get('issues', [])
            if issues:
                logger.error("\nCritical issues found:")
                for issue in issues:
                    logger.error(f"  • {issue}")
            
            warnings = report.get('warnings', [])
            if warnings:
                logger.warning("\nWarnings:")
                for warning in warnings:
                    logger.warning(f"  • {warning}")
            
            logger.error("\nPlease resolve issues and re-run analyze_data.py")
            logger.error("=" * 60)
            return False
        
        # Data is ready - show summary
        summary = report.get('summary', {})
        logger.info("=" * 60)
        logger.info("DATA READINESS CHECK: PASSED ✓")
        logger.info("=" * 60)
        logger.info(f"  Images:       {summary.get('total_images', 0):,}")
        logger.info(f"  QA Pairs:     {summary.get('total_qa_pairs', 0):,}")
        logger.info(f"  Scene Graphs: {summary.get('total_scene_graphs', 0):,}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error reading analysis report: {e}")
        logger.error("Please re-run analyze_data.py")
        return False


def main(args):
    """Main training function."""
    # Load config
    if args.config and os.path.exists(args.config):
        config = load_config_from_file(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line args
    if args.mimic_cxr_path:
        config.data.mimic_cxr_jpg_path = args.mimic_cxr_path
    if args.mimic_qa_path:
        config.data.mimic_ext_cxr_qba_path = args.mimic_qa_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.batch_size:
        config.training.batch_size_per_gpu = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.hub_model_id:
        config.training.hub_model_id = args.hub_model_id
    if args.wandb_project:
        config.wandb.project = args.wandb_project
    if args.disable_wandb:
        config.wandb.enabled = False
    
    # Check data readiness (unless skipped)
    if not args.skip_data_check:
        if not check_data_readiness(config):
            logger.info("\nTo skip this check (not recommended), use --skip_data_check")
            sys.exit(1)
    else:
        logger.warning("Skipping data readiness check (--skip_data_check)")
    
    # Setup
    seed_everything(config.training.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check for multi-GPU
    n_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {n_gpus}")
    
    # Initialize wandb
    wandb_run = init_wandb(config)
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(config.training.output_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Load datasets
    logger.info("Loading datasets...")
    
    train_dataset = MIMICCXRVQADataset(
        mimic_cxr_path=config.data.mimic_cxr_jpg_path,
        mimic_qa_path=config.data.mimic_ext_cxr_qba_path,
        split='train',
        tokenizer_name=config.model.text_encoder,
        max_question_length=config.model.max_question_length,
        quality_grade=config.data.quality_grade,
        view_filter=config.data.view_filter,
        question_types=config.data.question_types if config.data.question_types else None,
        chexpert_labels_path=config.data.chexpert_labels_path if config.data.chexpert_labels_path else None,
        max_samples=args.max_samples
    )
    
    val_dataset = MIMICCXRVQADataset(
        mimic_cxr_path=config.data.mimic_cxr_jpg_path,
        mimic_qa_path=config.data.mimic_ext_cxr_qba_path,
        split='validate',
        tokenizer_name=config.model.text_encoder,
        max_question_length=config.model.max_question_length,
        quality_grade=config.data.quality_grade,
        view_filter=config.data.view_filter,
        question_types=config.data.question_types if config.data.question_types else None,
        chexpert_labels_path=config.data.chexpert_labels_path if config.data.chexpert_labels_path else None,
        max_samples=args.max_samples // 10 if args.max_samples else None
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=True,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=config.training.dataloader_pin_memory
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size_per_gpu * 2,
        shuffle=False,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=config.training.dataloader_pin_memory
    )
    
    # Initialize model
    logger.info("Initializing model...")
    
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
    
    # Enable gradient checkpointing if configured
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Multi-GPU
    if n_gpus > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Log model params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    if WANDB_AVAILABLE and config.wandb.enabled and config.wandb.watch_model:
        wandb.watch(model, log="all", log_freq=config.wandb.watch_log_freq)
    
    # Loss function
    criterion = MultiTaskLoss(
        vqa_weight=config.training.vqa_loss_weight,
        chexpert_weight=config.training.chexpert_loss_weight,
        binary_weight=config.training.binary_head_weight,
        category_weight=config.training.category_head_weight,
        region_weight=config.training.region_head_weight,
        severity_weight=config.training.severity_head_weight,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_dataloader) * config.training.num_epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.learning_rate,
        total_steps=total_steps,
        pct_start=config.training.warmup_ratio,
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.training.fp16 else None
    
    # Training loop
    best_metric = 0.0
    global_step = 0
    epochs_without_improvement = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, config.training.num_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{config.training.num_epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss, global_step = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            global_step=global_step
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            config=config
        )
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Accuracy: {val_metrics.get('accuracy', 0):.4f}")
        logger.info(f"Val Binary Acc: {val_metrics.get('binary_accuracy', 0):.4f}")
        
        # Log to wandb
        if WANDB_AVAILABLE and config.wandb.enabled:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics.get('accuracy', 0),
                'val/binary_accuracy': val_metrics.get('binary_accuracy', 0),
                'val/category_f1': val_metrics.get('category_f1', 0),
                'val/chexpert_auroc': val_metrics.get('chexpert_auroc', 0),
            })
        
        # Check for best model
        current_metric = val_metrics.get(config.training.metric_for_best_model, 0)
        is_best = current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
            epochs_without_improvement = 0
            logger.info(f"New best {config.training.metric_for_best_model}: {best_metric:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint
        if global_step % config.training.save_steps == 0 or is_best:
            save_checkpoint(
                model=model.module if hasattr(model, 'module') else model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                metrics=val_metrics,
                config=config,
                is_best=is_best
            )
        
        # Push to hub
        if is_best and config.training.hub_model_id:
            push_to_hub(
                model=model.module if hasattr(model, 'module') else model,
                config=config,
                metrics=val_metrics,
                commit_message=f"Epoch {epoch} - Acc: {val_metrics.get('accuracy', 0):.4f}"
            )
        
        # Early stopping
        if epochs_without_improvement >= config.training.early_stopping_patience:
            logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
            break
    
    # Final save
    logger.info("\nTraining complete!")
    logger.info(f"Best {config.training.metric_for_best_model}: {best_metric:.4f}")
    
    save_checkpoint(
        model=model.module if hasattr(model, 'module') else model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        metrics=val_metrics,
        config=config,
        is_best=False
    )
    
    # Final push to hub
    if config.training.hub_model_id:
        push_to_hub(
            model=model.module if hasattr(model, 'module') else model,
            config=config,
            metrics={'best_accuracy': best_metric, **val_metrics},
            commit_message=f"Final model - Best Acc: {best_metric:.4f}"
        )
    
    # Close wandb
    if WANDB_AVAILABLE and config.wandb.enabled:
        wandb.finish()
    
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIMIC-CXR VQA Model")
    
    # Config
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    
    # Data paths
    parser.add_argument('--mimic_cxr_path', type=str, help='Path to MIMIC-CXR-JPG dataset')
    parser.add_argument('--mimic_qa_path', type=str, help='Path to MIMIC-Ext-CXR-QBA dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/mimic-cxr-vqa', help='Output directory')
    
    # Training params
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples (for debugging)')
    
    # Hub
    parser.add_argument('--hub_model_id', type=str, help='Hugging Face Hub model ID')
    
    # Wandb
    parser.add_argument('--wandb_project', type=str, default='mimic-cxr-vqa', help='W&B project name')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable W&B logging')
    
    # Data check
    parser.add_argument('--skip_data_check', action='store_true', 
                       help='Skip data readiness check (not recommended)')
    
    args = parser.parse_args()
    
    main(args)

