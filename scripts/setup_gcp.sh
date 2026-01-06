#!/bin/bash
# =============================================================================
# MIMIC-CXR VQA - Complete GCP Setup Script
# =============================================================================
# 
# For: GCP g2-standard-48 with 4x NVIDIA L4 GPUs, Debian 12
#
# Your data locations:
#   - MIMIC-CXR-JPG:      ~/dataset
#   - MIMIC-Ext-CXR-QBA:  ~/scenegraphdata
#
# Usage:
#   chmod +x setup_gcp.sh
#   ./setup_gcp.sh
#
# After setup, add your secrets:
#   1. Edit ~/.env with your HuggingFace and Wandb tokens
#   2. Or run: ./setup_gcp.sh --configure-secrets
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - Modify these if needed
# =============================================================================
ENV_NAME="mimic-vqa"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"  # L4 GPUs work best with CUDA 12.x

# Data paths (your GCP instance)
MIMIC_CXR_PATH="$HOME/dataset"
MIMIC_QA_PATH="$HOME/scenegraphdata"

# Project directory
PROJECT_DIR="$HOME/SVA-MASTERS-WORK"

# =============================================================================
# Colors for output
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Helper functions
# =============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# =============================================================================
# Parse arguments
# =============================================================================
CONFIGURE_SECRETS=false
SKIP_CONDA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --configure-secrets)
            CONFIGURE_SECRETS=true
            shift
            ;;
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./setup_gcp.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --configure-secrets   Only configure HuggingFace and Wandb secrets"
            echo "  --skip-conda          Skip conda installation (if already installed)"
            echo "  --help, -h            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Configure secrets only
# =============================================================================
if [ "$CONFIGURE_SECRETS" = true ]; then
    log_step "Configuring Secrets"
    
    echo -e "${CYAN}Enter your HuggingFace token (from https://huggingface.co/settings/tokens):${NC}"
    read -s HF_TOKEN
    echo ""
    
    echo -e "${CYAN}Enter your Wandb API key (from https://wandb.ai/authorize):${NC}"
    read -s WANDB_API_KEY
    echo ""
    
    # Save to .env file
    cat > "$HOME/.env" << EOF
# MIMIC-CXR VQA Secrets
# Generated on $(date)

# HuggingFace Token (for model upload)
HF_TOKEN=${HF_TOKEN}

# Weights & Biases API Key (for experiment tracking)
WANDB_API_KEY=${WANDB_API_KEY}
EOF
    
    chmod 600 "$HOME/.env"
    log_info "Secrets saved to ~/.env"
    
    # Also configure wandb directly
    if command -v wandb &> /dev/null; then
        echo "$WANDB_API_KEY" | wandb login --relogin 2>/dev/null || true
        log_info "Wandb configured"
    fi
    
    # Configure huggingface
    if command -v huggingface-cli &> /dev/null; then
        echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
        log_info "HuggingFace configured"
    fi
    
    log_info "Done! Your secrets are configured."
    exit 0
fi

# =============================================================================
# Main Setup
# =============================================================================
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║         MIMIC-CXR VQA - GCP Setup Script                      ║"
echo "║                                                               ║"
echo "║   Machine: g2-standard-48 (4x NVIDIA L4)                      ║"
echo "║   OS: Debian 12 (Bookworm)                                    ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# Step 1: System Dependencies
# =============================================================================
log_step "[1/8] Installing System Dependencies"

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    htop \
    tmux \
    vim \
    ca-certificates \
    gnupg \
    lsb-release

log_info "System dependencies installed"

# =============================================================================
# Step 2: Install Miniconda
# =============================================================================
log_step "[2/8] Installing Miniconda"

if [ "$SKIP_CONDA" = false ] && [ ! -d "$HOME/miniconda3" ]; then
    log_info "Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    log_info "Installing Miniconda..."
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    source $HOME/.bashrc
    
    log_info "Miniconda installed"
else
    log_info "Miniconda already installed or skipped"
fi

# Source conda
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh

# =============================================================================
# Step 3: Create Conda Environment
# =============================================================================
log_step "[3/8] Creating Conda Environment: ${ENV_NAME}"

# Remove existing environment if exists
conda env remove -n $ENV_NAME 2>/dev/null || true

# Create new environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
conda activate $ENV_NAME

log_info "Environment created and activated"

# =============================================================================
# Step 4: Install NVIDIA Drivers & CUDA (if needed)
# =============================================================================
log_step "[4/8] Checking NVIDIA/CUDA Setup"

if command -v nvidia-smi &> /dev/null; then
    log_info "NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    log_warn "NVIDIA driver not found. Installing..."
    
    # Install NVIDIA driver (for Debian 12 with L4 GPUs)
    sudo apt-get install -y nvidia-driver
    
    log_warn "NVIDIA driver installed. A reboot may be required."
fi

# =============================================================================
# Step 5: Install PyTorch with CUDA
# =============================================================================
log_step "[5/8] Installing PyTorch with CUDA ${CUDA_VERSION}"

# For L4 GPUs, CUDA 12.1 is recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

log_info "PyTorch installed"

# =============================================================================
# Step 6: Clone Repository & Install Requirements
# =============================================================================
log_step "[6/8] Setting Up Project"

cd $HOME

# Clone repository if not exists
if [ ! -d "$PROJECT_DIR" ]; then
    log_info "Cloning repository..."
    git clone https://github.com/JONAHKYAGABA/SVA-MASTERS-WORK.git
fi

cd $PROJECT_DIR

# Install Python requirements
log_info "Installing Python requirements..."
pip install -r requirements.txt

# Install additional packages for distributed training
pip install deepspeed flash-attn --no-build-isolation 2>/dev/null || log_warn "flash-attn installation failed (optional)"

# Download NLTK data
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
print('NLTK data downloaded')
"

log_info "Requirements installed"

# =============================================================================
# Step 7: Extract Data ZIP Files
# =============================================================================
log_step "[7/8] Extracting Data ZIP Files"

# Check and extract qa.zip
QA_ZIP="$MIMIC_QA_PATH/qa.zip"
QA_DIR="$MIMIC_QA_PATH/qa"

if [ -f "$QA_ZIP" ] && [ ! -d "$QA_DIR" ]; then
    log_info "Extracting qa.zip (7.5 GB)... This will take several minutes..."
    cd "$MIMIC_QA_PATH"
    unzip -o -q qa.zip
    log_info "qa.zip extracted"
elif [ -d "$QA_DIR" ]; then
    log_info "qa/ directory already exists"
else
    log_warn "qa.zip not found at $QA_ZIP"
fi

# Check and extract scene_data.zip
SCENE_ZIP="$MIMIC_QA_PATH/scene_data.zip"
SCENE_DIR="$MIMIC_QA_PATH/scene_data"

if [ -f "$SCENE_ZIP" ] && [ ! -d "$SCENE_DIR" ]; then
    log_info "Extracting scene_data.zip (1.3 GB)..."
    cd "$MIMIC_QA_PATH"
    unzip -o -q scene_data.zip
    log_info "scene_data.zip extracted"
elif [ -d "$SCENE_DIR" ]; then
    log_info "scene_data/ directory already exists"
else
    log_warn "scene_data.zip not found at $SCENE_ZIP"
fi

cd $PROJECT_DIR

# =============================================================================
# Step 8: Create Configuration Files
# =============================================================================
log_step "[8/8] Creating Configuration Files"

# Create paths configuration
cat > configs/paths.yaml << EOF
# Auto-generated paths configuration for GCP instance
# Generated on $(date)

data:
  mimic_cxr_jpg_path: "${MIMIC_CXR_PATH}"
  mimic_ext_cxr_qba_path: "${MIMIC_QA_PATH}"
  chexpert_labels_path: "${MIMIC_CXR_PATH}/mimic-cxr-2.0.0-chexpert.csv.gz"
EOF

log_info "Created configs/paths.yaml"

# Create secrets template
cat > "$HOME/.env.template" << 'EOF'
# =============================================================================
# MIMIC-CXR VQA Secrets Configuration
# =============================================================================
# Copy this file to ~/.env and fill in your tokens:
#   cp ~/.env.template ~/.env
#   nano ~/.env
#
# Then run: source ~/.env
# =============================================================================

# HuggingFace Token
# Get from: https://huggingface.co/settings/tokens
# Required for: Uploading trained models to HuggingFace Hub
HF_TOKEN=hf_your_token_here

# Weights & Biases API Key  
# Get from: https://wandb.ai/authorize
# Required for: Experiment tracking and logging
WANDB_API_KEY=your_wandb_key_here

# Optional: Wandb entity (your username or team name)
WANDB_ENTITY=your_username

# Optional: Wandb project name
WANDB_PROJECT=mimic-cxr-vqa
EOF

log_info "Created ~/.env.template"

# Create run script
cat > run_training.sh << 'EOF'
#!/bin/bash
# =============================================================================
# Training Launch Script
# =============================================================================

# Load environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mimic-vqa

# Load secrets
if [ -f "$HOME/.env" ]; then
    export $(cat $HOME/.env | grep -v '^#' | xargs)
    echo "Secrets loaded from ~/.env"
else
    echo "Warning: ~/.env not found. Run: ./setup_gcp.sh --configure-secrets"
fi

# Set wandb mode
export WANDB_MODE=online

# Launch distributed training with hardware auto-detection
# This automatically detects GPUs, memory, and optimizes settings
./scripts/launch_distributed_training.sh --config configs/default_config.yaml "\$@"
EOF
chmod +x run_training.sh

log_info "Created run_training.sh (with hardware auto-optimization)"

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║              ✓ SETUP COMPLETE!                                ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}Environment:${NC} $ENV_NAME"
echo -e "${GREEN}Project:${NC} $PROJECT_DIR"
echo -e "${GREEN}Data:${NC}"
echo -e "  - MIMIC-CXR-JPG:      $MIMIC_CXR_PATH"
echo -e "  - MIMIC-Ext-CXR-QBA:  $MIMIC_QA_PATH"

echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "\n${CYAN}1. Configure your secrets:${NC}"
echo -e "   ${GREEN}./scripts/setup_gcp.sh --configure-secrets${NC}"
echo -e "   Or manually edit ~/.env"

echo -e "\n${CYAN}2. Activate the environment:${NC}"
echo -e "   ${GREEN}conda activate mimic-vqa${NC}"

echo -e "\n${CYAN}3. Verify data setup:${NC}"
echo -e "   ${GREEN}python analyze_data.py --mimic_cxr_path $MIMIC_CXR_PATH --mimic_qa_path $MIMIC_QA_PATH${NC}"

echo -e "\n${CYAN}4. Check hardware optimization:${NC}"
echo -e "   ${GREEN}python -m utils.hardware_utils${NC}"
echo -e "   (Shows detected GPUs, memory, and optimal settings)"

echo -e "\n${CYAN}5. Start training (auto-optimized):${NC}"
echo -e "   ${GREEN}./run_training.sh${NC}"
echo -e "   Or with distributed launcher directly:"
echo -e "   ${GREEN}./scripts/launch_distributed_training.sh --config configs/default_config.yaml${NC}"
echo -e "   "
echo -e "   Hardware auto-detection will:"
echo -e "   - Detect your 4x NVIDIA L4 GPUs (96GB total)"
echo -e "   - Set batch_size=16/GPU, grad_accum=4 (effective 256)"
echo -e "   - Enable DeepSpeed ZeRO-2 for optimal memory"
echo -e "   - Configure 12 DataLoader workers"

echo -e "\n${CYAN}6. Monitor training:${NC}"
echo -e "   - Wandb: https://wandb.ai/your-username/mimic-cxr-vqa"
echo -e "   - GPU usage: ${GREEN}watch -n 1 nvidia-smi${NC}"

echo -e "\n${CYAN}6. Use tmux for long training (recommended):${NC}"
echo -e "   ${GREEN}tmux new -s training${NC}"
echo -e "   ${GREEN}./run_training.sh${NC}"
echo -e "   (Detach with Ctrl+B, then D)"

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

