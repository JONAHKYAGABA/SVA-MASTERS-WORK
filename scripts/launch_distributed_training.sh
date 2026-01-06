#!/bin/bash
# ============================================================================
# MIMIC-CXR VQA Distributed Training Launcher
# ============================================================================
# This script launches multi-GPU training with DeepSpeed ZeRO-2 optimization
# as specified in the methodology (Section 11: Training Optimizations)
#
# Hardware auto-detection: Automatically detects GPUs, memory, and CPUs
# to maximize hardware utilization.
#
# Usage:
#   ./launch_distributed_training.sh [OPTIONS]
#
# Options:
#   --config PATH       Path to config file (default: configs/default_config.yaml)
#   --num_gpus N        Number of GPUs (default: auto-detect)
#   --deepspeed         Force DeepSpeed ZeRO-2
#   --torchrun          Force torchrun DDP instead of DeepSpeed
#   --no_auto           Disable hardware auto-optimization
#   --debug             Enable debug mode with small batch
# ============================================================================

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Default values
CONFIG_PATH="configs/default_config.yaml"
NUM_GPUS=""
USE_DEEPSPEED=""  # Empty = auto-detect
USE_TORCHRUN=false
DEBUG_MODE=false
AUTO_OPTIMIZE=true
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --deepspeed)
            USE_DEEPSPEED=true
            USE_TORCHRUN=false
            shift
            ;;
        --torchrun)
            USE_TORCHRUN=true
            USE_DEEPSPEED=false
            shift
            ;;
        --no_auto)
            AUTO_OPTIMIZE=false
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ============================================
# HARDWARE AUTO-DETECTION
# ============================================

# Auto-detect GPUs if not specified
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "⚠️  No GPUs detected. Running on CPU (not recommended)."
    fi
fi

# Get GPU memory info
if [ "$NUM_GPUS" -gt 0 ]; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    TOTAL_GPU_MEM=$((GPU_MEM * NUM_GPUS))
else
    GPU_MEM=0
    GPU_NAME="None"
    TOTAL_GPU_MEM=0
fi

# Get CPU count
NUM_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")

# Get RAM info (in GB)
if [ -f /proc/meminfo ]; then
    TOTAL_RAM=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    FREE_RAM=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
else
    TOTAL_RAM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}' || echo "16")
    FREE_RAM=$TOTAL_RAM
fi

# ============================================
# CALCULATE OPTIMAL SETTINGS
# ============================================

# Determine optimal batch size based on GPU memory
if [ "$GPU_MEM" -ge 20000 ]; then       # 20GB+ (L4, A10, RTX 3090)
    BATCH_PER_GPU=16
elif [ "$GPU_MEM" -ge 14000 ]; then     # 14-20GB (T4, RTX 4080)
    BATCH_PER_GPU=12
elif [ "$GPU_MEM" -ge 10000 ]; then     # 10-14GB (RTX 3060)
    BATCH_PER_GPU=8
elif [ "$GPU_MEM" -ge 6000 ]; then      # 6-10GB
    BATCH_PER_GPU=4
else                                      # <6GB
    BATCH_PER_GPU=2
fi

# Calculate gradient accumulation to target effective batch ~256
TARGET_EFFECTIVE=256
if [ "$NUM_GPUS" -gt 0 ]; then
    CURRENT_EFFECTIVE=$((BATCH_PER_GPU * NUM_GPUS))
    GRAD_ACCUM=$((TARGET_EFFECTIVE / CURRENT_EFFECTIVE))
    [ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1
    [ "$GRAD_ACCUM" -gt 16 ] && GRAD_ACCUM=16
else
    GRAD_ACCUM=8
fi

EFFECTIVE_BATCH=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))
[ "$EFFECTIVE_BATCH" -eq 0 ] && EFFECTIVE_BATCH=$((BATCH_PER_GPU * GRAD_ACCUM))

# Calculate optimal workers (3 per GPU, capped at CPUs - 2)
WORKERS_PER_GPU=3
OPTIMAL_WORKERS=$((WORKERS_PER_GPU * NUM_GPUS))
MAX_WORKERS=$((NUM_CPUS - 2))
[ "$MAX_WORKERS" -lt 1 ] && MAX_WORKERS=1
[ "$OPTIMAL_WORKERS" -gt "$MAX_WORKERS" ] && OPTIMAL_WORKERS=$MAX_WORKERS

# Auto-select distributed training method
if [ -z "$USE_DEEPSPEED" ]; then
    if [ "$NUM_GPUS" -gt 1 ]; then
        # Check if DeepSpeed is available
        if python -c "import deepspeed" 2>/dev/null; then
            USE_DEEPSPEED=true
        else
            USE_TORCHRUN=true
            echo "⚠️  DeepSpeed not found. Using torchrun instead."
        fi
    else
        USE_DEEPSPEED=false
    fi
fi

# ============================================
# PRINT HARDWARE SUMMARY
# ============================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           MIMIC-CXR VQA DISTRIBUTED TRAINING LAUNCHER                ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║                                                                      ║"
echo "║  DETECTED HARDWARE:                                                  ║"
echo "║  ──────────────────────────────────────────────────────────────────  ║"
printf "║  GPUs:             %-4s x %-40s ║\n" "$NUM_GPUS" "$GPU_NAME"
printf "║  GPU Memory:       %-4s GB per GPU (%-4s GB total)                  ║\n" "$((GPU_MEM/1000))" "$((TOTAL_GPU_MEM/1000))"
printf "║  CPU Cores:        %-50s ║\n" "$NUM_CPUS"
printf "║  RAM:              %-4s GB total (%-4s GB free)                     ║\n" "$TOTAL_RAM" "$FREE_RAM"
echo "║                                                                      ║"
echo "║  OPTIMIZED SETTINGS:                                                 ║"
echo "║  ──────────────────────────────────────────────────────────────────  ║"
printf "║  Batch per GPU:    %-50s ║\n" "$BATCH_PER_GPU"
printf "║  Grad accumulation:%-50s ║\n" "$GRAD_ACCUM"
printf "║  Effective batch:  %-50s ║\n" "$EFFECTIVE_BATCH"
printf "║  DataLoader workers:%-49s ║\n" "$OPTIMAL_WORKERS"
printf "║  DeepSpeed:        %-50s ║\n" "$USE_DEEPSPEED"
printf "║  Config:           %-50s ║\n" "$CONFIG_PATH"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Optimal OMP threads (prevent oversubscription)
if [ "$NUM_GPUS" -gt 0 ]; then
    OMP_THREADS=$((NUM_CPUS / (NUM_GPUS * 4)))
    [ "$OMP_THREADS" -lt 1 ] && OMP_THREADS=1
else
    OMP_THREADS=4
fi

export OMP_NUM_THREADS=$OMP_THREADS
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# NCCL optimizations for multi-GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
fi

# Debug mode settings
if [ "$DEBUG_MODE" = true ]; then
    echo "🔧 [DEBUG MODE] Using small batch for testing..."
    DEBUG_ARGS="--max_samples 100 --batch_size 4"
else
    DEBUG_ARGS=""
fi

# Auto-optimize flag
if [ "$AUTO_OPTIMIZE" = true ]; then
    AUTO_ARGS="--auto_optimize"
else
    AUTO_ARGS="--no_auto_optimize"
fi

# ============================================
# LAUNCH TRAINING
# ============================================

if [ "$USE_DEEPSPEED" = true ] && [ "$NUM_GPUS" -gt 0 ]; then
    echo "🚀 Launching with DeepSpeed ZeRO-2 on $NUM_GPUS GPUs..."
    echo ""
    
    deepspeed --num_gpus=$NUM_GPUS \
        train_mimic_cxr.py \
        --config "$CONFIG_PATH" \
        --use_deepspeed \
        --deepspeed_config configs/deepspeed_config.json \
        $AUTO_ARGS \
        $DEBUG_ARGS \
        $EXTRA_ARGS

elif [ "$USE_TORCHRUN" = true ] && [ "$NUM_GPUS" -gt 0 ]; then
    echo "🚀 Launching with torchrun (DDP) on $NUM_GPUS GPUs..."
    echo ""
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_mimic_cxr.py \
        --config "$CONFIG_PATH" \
        --use_ddp \
        $AUTO_ARGS \
        $DEBUG_ARGS \
        $EXTRA_ARGS

elif [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 Launching with DataParallel on $NUM_GPUS GPUs..."
    echo "💡 Tip: Install DeepSpeed for better multi-GPU performance"
    echo ""
    
    python train_mimic_cxr.py \
        --config "$CONFIG_PATH" \
        $AUTO_ARGS \
        $DEBUG_ARGS \
        $EXTRA_ARGS

else
    echo "🚀 Launching single GPU/CPU training..."
    echo ""
    
    python train_mimic_cxr.py \
        --config "$CONFIG_PATH" \
        $AUTO_ARGS \
        $DEBUG_ARGS \
        $EXTRA_ARGS
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                      TRAINING COMPLETE!                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

