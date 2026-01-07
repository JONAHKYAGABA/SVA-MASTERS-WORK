#!/bin/bash
# ============================================================================
# Quick Start Script for GCP Server (4x NVIDIA L4 GPUs)
# ============================================================================
# 
# This script sets up and runs MIMIC-CXR VQA training on your GCP instance.
# 
# Prerequisites:
#   - MIMIC-CXR-JPG at: ~/dataset/
#   - MIMIC-Ext-CXR-QBA at: ~/scenegraphdata/physionet.org/files/mimic-ext-cxr-qba/1.0.0/
#   - Conda environment "mimic-vqa" created (see setup_gcp.sh)
#
# Usage:
#   cd ~/SSG-VQA-main  # or wherever you cloned the repo
#   ./scripts/quick_start_gcp.sh
#
# ============================================================================

set -e

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Configuration ---
MIMIC_CXR_PATH="$HOME/dataset"
MIMIC_QA_PATH="$HOME/scenegraphdata/physionet.org/files/mimic-ext-cxr-qba/1.0.0"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="configs/gcp_server_config.yaml"

echo -e "\n${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║         MIMIC-CXR VQA - GCP Quick Start                       ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# --- Step 1: Check dataset paths ---
log_info "Step 1: Checking dataset paths..."

if [ ! -d "$MIMIC_CXR_PATH" ]; then
    log_error "MIMIC-CXR-JPG not found at: $MIMIC_CXR_PATH"
    exit 1
fi
log_success "MIMIC-CXR-JPG found at: $MIMIC_CXR_PATH"

if [ ! -d "$MIMIC_QA_PATH" ]; then
    log_error "MIMIC-Ext-CXR-QBA not found at: $MIMIC_QA_PATH"
    exit 1
fi
log_success "MIMIC-Ext-CXR-QBA found at: $MIMIC_QA_PATH"

# --- Step 2: Check if ZIPs need extraction ---
log_info "Step 2: Checking if ZIP extraction is needed..."

QA_ZIP="$MIMIC_QA_PATH/qa.zip"
SCENE_ZIP="$MIMIC_QA_PATH/scene_data.zip"
QA_DIR="$MIMIC_QA_PATH/qa"
SCENE_DIR="$MIMIC_QA_PATH/scene_data"

NEEDS_EXTRACTION=false

if [ -f "$QA_ZIP" ] && [ ! -d "$QA_DIR" ]; then
    log_warning "qa.zip exists but not extracted"
    NEEDS_EXTRACTION=true
fi

if [ -f "$SCENE_ZIP" ] && [ ! -d "$SCENE_DIR" ]; then
    log_warning "scene_data.zip exists but not extracted"
    NEEDS_EXTRACTION=true
fi

if [ "$NEEDS_EXTRACTION" = true ]; then
    log_info "Running setup_data.py to extract ZIPs..."
    cd "$PROJECT_DIR"
    python setup_data.py \
        --mimic_cxr_path "$MIMIC_CXR_PATH" \
        --mimic_qa_path "$MIMIC_QA_PATH"
    
    if [ $? -ne 0 ]; then
        log_error "ZIP extraction failed!"
        exit 1
    fi
    log_success "ZIP extraction complete"
else
    log_success "ZIP files already extracted or not needed"
fi

# --- Step 3: Check extracted directories ---
log_info "Step 3: Verifying extracted directories..."

if [ -d "$QA_DIR" ]; then
    QA_COUNT=$(find "$QA_DIR" -name "*.json" 2>/dev/null | head -100 | wc -l)
    log_success "qa/ directory exists with $QA_COUNT+ JSON files"
else
    log_error "qa/ directory not found at: $QA_DIR"
    exit 1
fi

if [ -d "$SCENE_DIR" ]; then
    SG_COUNT=$(find "$SCENE_DIR" -name "*.json" 2>/dev/null | head -100 | wc -l)
    log_success "scene_data/ directory exists with $SG_COUNT+ JSON files"
else
    log_error "scene_data/ directory not found at: $SCENE_DIR"
    exit 1
fi

# --- Step 4: Load environment ---
log_info "Step 4: Loading environment..."

# Load secrets if available
if [ -f "$HOME/.env" ]; then
    export $(cat "$HOME/.env" | grep -v '^#' | xargs)
    log_success "Loaded secrets from ~/.env"
else
    log_warning "~/.env not found - wandb/huggingface logging may not work"
    log_info "Run: ./scripts/setup_gcp.sh --configure-secrets"
fi

# --- Step 5: Run data analysis ---
log_info "Step 5: Running data analysis..."

cd "$PROJECT_DIR"
python analyze_data.py \
    --mimic_cxr_path "$MIMIC_CXR_PATH" \
    --mimic_qa_path "$MIMIC_QA_PATH" \
    --output_dir ./analysis_output

if [ $? -ne 0 ]; then
    log_error "Data analysis failed!"
    exit 1
fi
log_success "Data analysis complete"

# --- Step 6: Check GPU availability ---
log_info "Step 6: Checking GPU availability..."

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    log_success "Detected $NUM_GPUS GPU(s) with $GPU_MEMORY each"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -4 | while read gpu; do
        log_info "  - $gpu"
    done
else
    log_warning "nvidia-smi not found - GPU detection failed"
    NUM_GPUS=1
fi

# --- Step 7: Start training ---
echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log_info "Step 7: Starting training..."
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

log_info "Configuration: $CONFIG_FILE"
log_info "Number of GPUs: $NUM_GPUS"

# Launch training
if [ "$NUM_GPUS" -gt 1 ]; then
    log_info "Using DeepSpeed for multi-GPU training..."
    deepspeed --num_gpus=$NUM_GPUS train_mimic_cxr.py \
        --config "$CONFIG_FILE" \
        --deepspeed_config configs/deepspeed_config.json
else
    log_info "Using single GPU training..."
    python train_mimic_cxr.py --config "$CONFIG_FILE"
fi

echo -e "\n${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║              ✓ Training Complete!                             ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

