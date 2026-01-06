#!/bin/bash
# =============================================================================
# MIMIC-CXR VQA Setup Script for Linux Server
# =============================================================================
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh --mimic_cxr /path/to/mimic-cxr-jpg --mimic_qa /path/to/mimic-ext-cxr-qba
#
# Or with conda environment:
#   ./setup.sh --create_env --mimic_cxr /path/to/data --mimic_qa /path/to/qa
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CREATE_ENV=false
ENV_NAME="mimic-vqa"
PYTHON_VERSION="3.10"
MIMIC_CXR_PATH=""
MIMIC_QA_PATH=""
CUDA_VERSION="11.8"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --create_env)
            CREATE_ENV=true
            shift
            ;;
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        --mimic_cxr)
            MIMIC_CXR_PATH="$2"
            shift 2
            ;;
        --mimic_qa)
            MIMIC_QA_PATH="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --create_env        Create a new conda environment"
            echo "  --env_name NAME     Name for conda environment (default: mimic-vqa)"
            echo "  --mimic_cxr PATH    Path to MIMIC-CXR-JPG dataset"
            echo "  --mimic_qa PATH     Path to MIMIC-Ext-CXR-QBA dataset"
            echo "  --cuda VERSION      CUDA version (default: 11.8)"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  MIMIC-CXR VQA Setup Script${NC}"
echo -e "${BLUE}=============================================${NC}"

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root is not recommended${NC}"
fi

# Step 1: Check system dependencies
echo -e "\n${GREEN}[1/6] Checking system dependencies...${NC}"

# Check for unzip
if ! command -v unzip &> /dev/null; then
    echo -e "${YELLOW}Installing unzip...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y unzip
    elif command -v yum &> /dev/null; then
        sudo yum install -y unzip
    else
        echo -e "${RED}Please install unzip manually${NC}"
    fi
fi
echo -e "  ✓ unzip available"

# Check for git
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git not found. Please install git.${NC}"
    exit 1
fi
echo -e "  ✓ git available"

# Step 2: Create conda environment if requested
if [ "$CREATE_ENV" = true ]; then
    echo -e "\n${GREEN}[2/6] Creating conda environment: ${ENV_NAME}...${NC}"
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Conda not found. Please install Miniconda or Anaconda.${NC}"
        echo -e "  Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Create environment
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    echo -e "  ✓ Environment created and activated"
else
    echo -e "\n${GREEN}[2/6] Skipping conda environment creation${NC}"
fi

# Step 3: Install PyTorch with CUDA
echo -e "\n${GREEN}[3/6] Installing PyTorch with CUDA ${CUDA_VERSION}...${NC}"

if [ "$CUDA_VERSION" = "11.8" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$CUDA_VERSION" = "12.1" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo -e "${YELLOW}Unknown CUDA version, using default PyTorch install${NC}"
    pip install torch torchvision torchaudio
fi

echo -e "  ✓ PyTorch installed"

# Step 4: Install requirements
echo -e "\n${GREEN}[4/6] Installing Python requirements...${NC}"

pip install -r requirements.txt

echo -e "  ✓ Requirements installed"

# Step 5: Download NLTK data
echo -e "\n${GREEN}[5/6] Downloading NLTK data...${NC}"

python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
print('  ✓ NLTK data downloaded')
"

# Step 6: Setup data
echo -e "\n${GREEN}[6/6] Setting up data...${NC}"

if [ -n "$MIMIC_CXR_PATH" ] && [ -n "$MIMIC_QA_PATH" ]; then
    echo -e "  Running automatic data setup..."
    python setup_data.py \
        --mimic_cxr_path "$MIMIC_CXR_PATH" \
        --mimic_qa_path "$MIMIC_QA_PATH"
    
    # Update config file
    echo -e "\n  Updating config file..."
    cat > configs/paths.yaml << EOF
# Auto-generated paths configuration
data:
  mimic_cxr_jpg_path: "${MIMIC_CXR_PATH}"
  mimic_ext_cxr_qba_path: "${MIMIC_QA_PATH}"
EOF
    echo -e "  ✓ Config saved to configs/paths.yaml"
else
    echo -e "${YELLOW}  Skipping data setup (no paths provided)${NC}"
    echo -e "  Run later: python setup_data.py --mimic_cxr /path/to/data --mimic_qa /path/to/qa"
fi

# Print summary
echo -e "\n${BLUE}=============================================${NC}"
echo -e "${GREEN}  Setup Complete! ✓${NC}"
echo -e "${BLUE}=============================================${NC}"

if [ "$CREATE_ENV" = true ]; then
    echo -e "\nActivate environment:"
    echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
fi

echo -e "\nNext steps:"
echo -e "  1. ${YELLOW}python analyze_data.py --mimic_cxr_path /path/to/mimic-cxr --mimic_qa_path /path/to/qa${NC}"
echo -e "  2. ${YELLOW}python train_mimic_cxr.py --config configs/default_config.yaml${NC}"
echo -e "  3. ${YELLOW}python evaluate.py --model_path ./checkpoints/best_model${NC}"

echo -e "\n${GREEN}Done!${NC}"

