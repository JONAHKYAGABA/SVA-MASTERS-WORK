"""
Utility Functions for MIMIC-CXR VQA Training

Provides helper classes and functions for:
- Training state management
- Checkpointing
- Logging setup
- Learning rate utilities
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking training metrics.
    """

    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def seed_everything(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = 'checkpoint.pt',
    is_best: bool = False
) -> str:
    """
    Save training checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(state, best_path)
    
    return str(filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map checkpoint to
        
    Returns:
        Dictionary with loaded checkpoint info (epoch, metrics, etc.)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
    }


def setup_logging(
    log_dir: str,
    log_file: str = 'training.log',
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging to file and console.
    
    Args:
        log_dir: Directory for log file
        log_file: Log filename
        level: Logging level
        
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_file
    
    # Create logger
    logger = logging.getLogger('mimic_vqa')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')
