# Multi-GPU Training Guide

This guide explains how to run distributed training on multiple GPUs with automatic hardware optimization.

## Quick Start

### Option 1: Auto-Optimized Training (Recommended)

Simply run the launch script - it automatically detects your hardware and configures optimal settings:

```bash
./launch_distributed_training.sh --config configs/default_config.yaml
```

The script will:
1. **Auto-detect GPUs**: Count, memory, compute capability
2. **Auto-detect CPU/RAM**: Cores, available memory
3. **Calculate optimal settings**: Batch size, workers, accumulation steps
4. **Select best distributed method**: DeepSpeed ZeRO-2 or DDP
5. **Set environment variables**: OMP threads, CUDA optimizations

### Option 2: Direct Python Launch

```bash
# Auto-optimized (default)
python train_mimic_cxr.py --config configs/default_config.yaml

# With DeepSpeed explicitly
deepspeed --num_gpus=4 train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --use_deepspeed

# With DDP
torchrun --nproc_per_node=4 train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --use_ddp
```

## Hardware Auto-Optimization

### What Gets Detected

| Component | Detection Method | Used For |
|-----------|------------------|----------|
| **GPUs** | `nvidia-smi`, `torch.cuda` | Batch size, distributed method |
| **GPU Memory** | Per-GPU VRAM | Batch size per GPU |
| **CPU Cores** | `nproc` | DataLoader workers |
| **RAM** | `/proc/meminfo` | Prefetch factor |

### Automatic Settings by GPU Memory

| GPU Memory | Batch Size | Gradient Accum | Effective Batch |
|------------|------------|----------------|-----------------|
| ≥20 GB (L4, A10) | 16 | 4 | 256 (4 GPUs) |
| 14-20 GB (T4) | 12 | 5 | 240 (4 GPUs) |
| 10-14 GB | 8 | 8 | 256 (4 GPUs) |
| 6-10 GB | 4 | 16 | 256 (4 GPUs) |
| <6 GB | 2 | 32 | 256 (4 GPUs) |

### Automatic Distributed Training Selection

```
┌─────────────────────────────────────────────────────────┐
│                  GPU Count Decision                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1 GPU  ──────►  Single GPU Training                    │
│                  (DataParallel disabled)                │
│                                                         │
│  2+ GPUs ──────► DeepSpeed Available?                   │
│                        │                                │
│                   Yes  │  No                            │
│                   ▼    │  ▼                             │
│             DeepSpeed  │  DDP (torchrun)                │
│             ZeRO-2     │                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## DeepSpeed ZeRO-2 Optimizations

When DeepSpeed is enabled (auto-enabled for 2+ GPUs), you get:

- **Optimizer State Partitioning**: 30-40% memory savings
- **CPU Offloading**: Move optimizer states to CPU RAM
- **Gradient Accumulation**: Built-in support
- **Mixed Precision**: FP16 with automatic loss scaling
- **Activation Checkpointing**: Trade compute for memory

### DeepSpeed Config (`configs/deepspeed_config.json`)

The config is automatically optimized based on detected hardware:

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "fp16": {
    "enabled": true
  },
  "activation_checkpointing": {
    "partition_activations": true
  }
}
```

## Memory Optimization Strategies

### For Limited GPU Memory (<16 GB)

1. **Reduce batch size**: Auto-handled by hardware detection
2. **Enable gradient checkpointing**: `gradient_checkpointing: true`
3. **Use DeepSpeed ZeRO-3**: Partitions model weights too
4. **CPU offloading**: Enabled by default in DeepSpeed config

### For 4x NVIDIA L4 (96 GB total)

Optimal configuration (auto-detected):
```yaml
training:
  batch_size_per_gpu: 16
  gradient_accumulation_steps: 4
  # Effective batch: 16 * 4 * 4 = 256
  fp16: true
  gradient_checkpointing: true
  dataloader_num_workers: 12  # 3 per GPU

deepspeed:
  enabled: true
  stage: 2
```

## Monitoring Training

### Weights & Biases Integration

Training metrics are automatically logged to W&B:
- Loss curves (VQA + CheXpert)
- Learning rate schedule
- GPU memory usage
- Throughput (samples/sec)

Set your API key:
```bash
export WANDB_API_KEY=your_key_here
```

### GPU Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check per-process memory
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `batch_size_per_gpu`
2. Increase `gradient_accumulation_steps` to maintain effective batch
3. Enable gradient checkpointing
4. Use DeepSpeed ZeRO-3

### Slow DataLoader

1. Increase `dataloader_num_workers` (up to 3 per GPU)
2. Enable `pin_memory: true`
3. Increase `prefetch_factor` (if RAM available)

### DeepSpeed Errors

```bash
# Check DeepSpeed installation
ds_report

# Reinstall if needed
pip install deepspeed --upgrade
```

### NCCL Communication Issues

```bash
# Disable InfiniBand (if not available)
export NCCL_IB_DISABLE=1

# Debug NCCL
export NCCL_DEBUG=INFO
```

## Performance Expectations

### Training Time (4x L4, 96 GB)

| Phase | Samples | Epochs | Time |
|-------|---------|--------|------|
| Pre-training (B-grade) | 31.2M | 3-5 | 2.5-3 days |
| Fine-tuning (A-grade) | 7.5M | 10-20 | 2-2.5 days |
| **Total** | - | - | **~5-6 days** |

### Scaling Efficiency

| GPUs | Expected Speedup | Efficiency |
|------|------------------|------------|
| 1 | 1.0x (baseline) | 100% |
| 2 | 1.9x | 95% |
| 4 | 3.5x | 87% |
| 8 | 6.5x | 81% |

## Command Reference

```bash
# Full options
./launch_distributed_training.sh \
    --config configs/default_config.yaml \
    --num_gpus 4 \                    # Override auto-detect
    --deepspeed \                      # Force DeepSpeed
    --debug                            # Debug mode (small batch)

# Python direct launch
python train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --auto_optimize \                  # Enable hardware optimization
    --use_deepspeed \                  # Use DeepSpeed
    --batch_size 16 \                  # Override batch size
    --epochs 20                        # Override epochs

# Disable auto-optimization
python train_mimic_cxr.py --no_auto_optimize
```

## Hardware Detection Utility

Run standalone to check your hardware:

```bash
python -m utils.hardware_utils
```

Output example:
```
======================================================================
HARDWARE DETECTION RESULTS
======================================================================

                     GPU CONFIGURATION                     
----------------------------------------------------------------------
  Number of GPUs:        4
  Total GPU Memory:      96.0 GB
  Min GPU Memory:        24.0 GB

  GPU 0: NVIDIA L4
       Memory: 24.0 GB (Free: 23.5 GB)
       Compute: 8.9

                    OPTIMAL TRAINING SETTINGS                     
----------------------------------------------------------------------
  Batch size per GPU:    16
  Gradient accumulation: 4
  Effective batch size:  256
  DataLoader workers:    12
  Mixed precision (FP16): Enabled
  Gradient checkpointing: Enabled
  DeepSpeed ZeRO stage:  2
======================================================================
```

