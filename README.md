# SSG-VQA for MIMIC-CXR: Medical Visual Question Answering

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Scene-Graph enhanced Visual Question Answering** adapted for Chest X-Ray analysis using MIMIC-CXR-JPG and MIMIC-Ext-CXR-QBA datasets.

<p align="center">
  <img src="asset/model.png" width="800" alt="SSG-VQA Architecture"/>
</p>

## 🔬 Key Features

- **ConvNeXt-Base Visual Backbone**: Pre-trained vision model for chest X-ray feature extraction
- **Bio+ClinicalBERT Text Encoder**: Domain-specific language model for medical questions  
- **Scene Graph Integration**: 134-dimensional scene graph embeddings from MIMIC-Ext-CXR-QBA
- **Multi-Head Answer Module**: Specialized heads for Binary, Category, Region, and Severity answers
- **Multi-Task Learning**: Joint VQA + CheXpert classification training
- **Hardware Auto-Optimization**: Automatic detection and optimization for any GPU configuration

## 📁 Repository Structure

```
SSG-VQA-main/
├── 📂 configs/                    # Configuration files
│   ├── default_config.yaml        # Default training configuration
│   ├── deepspeed_config.json      # DeepSpeed ZeRO-2 settings
│   └── mimic_cxr_config.py        # Python config dataclasses
│
├── 📂 data/                       # Data loading & processing
│   ├── __init__.py
│   └── mimic_cxr_dataset.py       # MIMIC-CXR VQA dataset class
│
├── 📂 models/                     # Model architectures
│   ├── __init__.py
│   └── mimic_vqa_model.py         # Complete SSG-VQA model
│
├── 📂 training/                   # Training utilities
│   ├── __init__.py
│   ├── loss.py                    # Multi-task loss functions
│   └── metrics.py                 # Evaluation metrics
│
├── 📂 utils/                      # Utility functions
│   ├── __init__.py
│   ├── utils.py                   # General utilities
│   └── hardware_utils.py          # Hardware auto-detection
│
├── 📂 scripts/                    # Shell scripts
│   ├── setup_gcp.sh               # GCP environment setup
│   ├── setup.sh                   # General setup script
│   └── launch_distributed_training.sh  # Multi-GPU training launcher
│
├── 📂 docs/                       # Documentation
│   ├── MIMIC_CXR_VQA_ANALYSIS.md  # Detailed methodology analysis
│   ├── MULTI_GPU_TRAINING.md      # Multi-GPU training guide
│   ├── TRAINING_GUIDE.md          # Step-by-step training guide
│   ├── SETUP_DATA.md              # Data setup instructions
│   ├── architecture_diagram.md    # Architecture details
│   ├── methodology.md             # Research methodology
│   ├── mimic-cxr-jpg.md           # MIMIC-CXR-JPG documentation
│   └── mimic-ext-cxr-qba.md       # MIMIC-Ext-CXR-QBA documentation
│
├── 📂 tests/                      # Unit tests
├── 📂 examples/                   # Example notebooks/scripts
├── 📂 asset/                      # Images and assets
│
├── 📜 train_mimic_cxr.py          # Main training script
├── 📜 evaluate.py                 # Model evaluation script
├── 📜 analyze_data.py             # Data analysis & validation
├── 📜 setup_data.py               # Data extraction & setup
├── 📜 requirements.txt            # Python dependencies
├── 📜 environment.yml             # Conda environment
└── 📜 LICENSE                     # MIT License
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/SSG-VQA-main.git
cd SSG-VQA-main

# Create conda environment
conda create -n mimic-vqa python=3.10 -y
conda activate mimic-vqa

# Install PyTorch with CUDA 12.1 (for L4/A10 GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

```bash
# Setup data paths (modify for your environment)
export MIMIC_CXR_PATH=/path/to/mimic-cxr-jpg
export MIMIC_QA_PATH=/path/to/mimic-ext-cxr-qba

# Extract scene graph data (if zipped)
python setup_data.py --extract_all --mimic_qa_path $MIMIC_QA_PATH

# Analyze and validate data
python analyze_data.py --mimic_cxr_path $MIMIC_CXR_PATH --mimic_qa_path $MIMIC_QA_PATH
```

### 3. Training

```bash
# Option 1: Auto-optimized training (recommended)
# Automatically detects hardware and sets optimal parameters
./scripts/launch_distributed_training.sh --config configs/default_config.yaml

# Option 2: Direct Python launch
python train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --mimic_cxr_path $MIMIC_CXR_PATH \
    --mimic_qa_path $MIMIC_QA_PATH

# Option 3: GCP setup (4x L4 GPUs)
./scripts/setup_gcp.sh
```

### 4. Evaluation

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model \
    --config configs/default_config.yaml
```

## ⚡ Hardware Auto-Optimization

The training pipeline automatically detects your hardware and optimizes settings:

```bash 
# Check detected hardware and optimal settings
python -m utils.hardware_utils
```

**Example output for 4x NVIDIA L4:**
```
╔══════════════════════════════════════════════════════════════════════╗
║           HARDWARE DETECTION RESULTS                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  GPUs:             4    x NVIDIA L4                                   ║
║  GPU Memory:       24   GB per GPU (96 GB total)                      ║
║  Optimal Settings:                                                    ║
║    Batch per GPU:    16                                               ║
║    Grad accumulation: 4                                               ║
║    Effective batch:  256                                              ║
║    DeepSpeed:        ZeRO-2                                           ║
╚══════════════════════════════════════════════════════════════════════╝
```

## 📊 Model Architecture

| Component | Details |
|-----------|---------|
| **Visual Backbone** | ConvNeXt-Base (pretrained) → 512-dim features |
| **Text Encoder** | Bio+ClinicalBERT → 768-dim embeddings |
| **Scene Graph** | 134-dim (6 bbox + 64 region + 64 entity) |
| **Fusion** | Scene-embedded Interaction Module (SIM) |
| **Answer Heads** | Binary (2), Category (14), Region (26), Severity (4) |
| **Auxiliary** | CheXpert 14-class classification |

## 🔧 Configuration

Edit `configs/default_config.yaml`:

```yaml
model:
  visual_backbone: "convnext_base"
  text_encoder: "emilyalsentzer/Bio_ClinicalBERT"
  hidden_dim: 512
  
training:
  batch_size_per_gpu: 16           # Auto-optimized by hardware
  gradient_accumulation_steps: 4   # Effective batch = 256
  learning_rate: 2.0e-5
  num_epochs: 20
  fp16: true
  gradient_checkpointing: true
  
deepspeed:
  enabled: true                    # Auto-enabled for multi-GPU
  stage: 2                         # ZeRO optimization level
```

## 📈 Experiment Tracking

**Weights & Biases** integration for real-time monitoring:

```bash
# Set API key (or add to ~/.env)
export WANDB_API_KEY=your_key_here

# Training will automatically log to W&B
python train_mimic_cxr.py --config configs/default_config.yaml
```

**Hugging Face Hub** for model checkpointing:

```bash
# Set token (or add to ~/.env)
export HF_TOKEN=your_token_here

# Configure in config.yaml
training:
  hub_model_id: "your-username/mimic-cxr-vqa"
```

## 📚 Documentation

- **[Training Guide](docs/TRAINING_GUIDE.md)**: Complete training walkthrough
- **[Multi-GPU Training](docs/MULTI_GPU_TRAINING.md)**: Distributed training setup
- **[Data Setup](docs/SETUP_DATA.md)**: Dataset preparation
- **[Methodology](docs/methodology.md)**: Research methodology
- **[Architecture](docs/architecture_diagram.md)**: Detailed model architecture

## 🔬 Datasets

This project uses two MIMIC datasets:

| Dataset | Description | Access |
|---------|-------------|--------|
| **MIMIC-CXR-JPG** | 377,110 chest X-ray images | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/) |
| **MIMIC-Ext-CXR-QBA** | 38.7M QA pairs with scene graphs | [PhysioNet](https://physionet.org/content/mimic-ext-cxr-qba/) |

⚠️ **Access Requirements**: Both datasets require credentialed PhysioNet access and CITI training.

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{seenivasan2023ssgqa,
  title={Surgical-VQA: Visual Question Answering in Surgical Scenes using Transformer},
  author={Seenivasan, Lalithkumar and Islam, Mobarakol and others},
  journal={MICCAI},
  year={2022}
}

@article{johnson2019mimic,
  title={MIMIC-CXR-JPG: A large publicly available database of labeled chest radiographs},
  author={Johnson, Alistair EW and others},
  journal={arXiv preprint},
  year={2019}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
