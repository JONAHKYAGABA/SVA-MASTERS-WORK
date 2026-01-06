# MIMIC-CXR VQA Training Guide

Complete training pipeline for SSG-VQA-Net adapted for chest X-ray Visual Question Answering.

## Pipeline Overview

The training pipeline consists of **three distinct phases**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIMIC-CXR VQA Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Phase 1: DATA ANALYSIS          (analyze_data.py)             │
│   ├── Validate dataset paths                                    │
│   ├── Analyze distributions (polarity, regions, question types) │
│   ├── Assess scene graph quality                                │
│   ├── Detect biases                                             │
│   └── Generate readiness report                                 │
│                                                                 │
│            ↓ (Only proceed if data is ready)                    │
│                                                                 │
│   Phase 2: TRAINING               (train_mimic_cxr.py)          │
│   ├── Load and preprocess data                                  │
│   ├── Initialize model (ConvNeXt + Bio+ClinicalBERT + SIM)      │
│   ├── Multi-task training (VQA + CheXpert auxiliary)            │
│   ├── Log to Weights & Biases                                   │
│   └── Save checkpoints to Hugging Face Hub                      │
│                                                                 │
│            ↓                                                    │
│                                                                 │
│   Phase 3: EVALUATION             (evaluate.py)                 │
│   ├── Answer Accuracy (EM, F1, BLEU-4, ROUGE-L, BERTScore)      │
│   ├── Spatial Reasoning (IoU, mAP, Pointing Accuracy)           │
│   ├── Clinical Relevance (Sensitivity, Specificity, AUROC)      │
│   ├── Explainability (Attention Plausibility)                   │
│   └── Statistical Significance Testing                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
SSG-VQA-main/
├── analyze_data.py              # Phase 1: Data analysis
├── train_mimic_cxr.py           # Phase 2: Training
├── evaluate.py                  # Phase 3: Evaluation
├── requirements.txt             # Dependencies
├── TRAINING_GUIDE.md            # This file
│
├── configs/
│   ├── mimic_cxr_config.py      # Configuration classes
│   ├── default_config.yaml      # Default training config
│   └── deepspeed_config.json    # DeepSpeed settings
│
├── data/
│   └── mimic_cxr_dataset.py     # Dataset and dataloader
│
├── models/
│   └── mimic_vqa_model.py       # Complete model architecture
│
├── training/
│   ├── loss.py                  # Multi-task loss function
│   └── metrics.py               # Evaluation metrics
│
└── utils/
    └── utils.py                 # Utility functions
```

## Installation

```bash
# Create conda environment
conda create -n mimic-vqa python=3.10
conda activate mimic-vqa

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Optional: Login to services
wandb login
huggingface-cli login
```

## Required Datasets

### 1. MIMIC-CXR-JPG
Download from PhysioNet: https://physionet.org/content/mimic-cxr-jpg/

Expected structure:
```
MIMIC-CXR-JPG/
├── files/
│   ├── p10/
│   │   ├── p10000032/
│   │   │   ├── s50000001/
│   │   │   │   └── *.jpg
│   ...
└── mimic-cxr-2.0.0-split.csv.gz
```

### 2. MIMIC-Ext-CXR-QBA
Scene graphs and QA pairs dataset.

Expected structure:
```
MIMIC-Ext-CXR-QBA/
├── qa/
│   ├── p10/
│   │   ├── p10000032/
│   │   │   └── s50000001.qa.json
│   ...
├── scene_graphs/
│   ├── p10/
│   ...
└── dataset_info.json
```

---

## Phase 1: Data Analysis

**ALWAYS run this phase first!** Training will not proceed without a passing analysis.

### Basic Analysis

```bash
python analyze_data.py \
    --mimic_cxr_path /path/to/MIMIC-CXR-JPG \
    --mimic_qa_path /path/to/MIMIC-Ext-CXR-QBA
```

### Using Config File

```bash
python analyze_data.py --config configs/default_config.yaml
```

### Output

The analysis produces:
- `analysis_output/analysis_report.json` - JSON report with all statistics
- `analysis_output/distribution_plots.png` - Visualization of distributions

### What's Analyzed

1. **Dataset Statistics**
   - Total images, QA pairs, scene graphs
   - Train/val/test split distribution

2. **Distribution Analysis**
   - Polarity (positive/negative findings)
   - Anatomical regions (cardiac, pulmonary, pleural, mediastinal, osseous)
   - Question types (binary, category, region, severity)

3. **Scene Graph Quality**
   - Average observations per graph
   - Bounding box coverage
   - Region distribution

4. **Bias Detection**
   - Polarity imbalance warnings
   - Region under-representation
   - Question type skew

### Passing Criteria

✅ Data is ready when:
- All required paths exist
- Images are found
- QA pairs are available
- No critical issues

❌ Training blocked when:
- Missing dataset paths
- No images found
- No QA pairs found
- Critical structural issues

---

## Phase 2: Training

### Prerequisites

✅ Run `analyze_data.py` first and ensure it passes

### Basic Training

```bash
python train_mimic_cxr.py --config configs/default_config.yaml
```

### With Wandb Logging

```bash
python train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --wandb_project mimic-cxr-vqa
```

### With Hugging Face Hub

```bash
python train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --hub_model_id your-username/mimic-cxr-vqa
```

### Full Command Example

```bash
python train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --mimic_cxr_path /data/MIMIC-CXR-JPG \
    --mimic_qa_path /data/MIMIC-Ext-CXR-QBA \
    --output_dir ./checkpoints/experiment-1 \
    --batch_size 16 \
    --epochs 20 \
    --learning_rate 5e-5 \
    --wandb_project mimic-cxr-vqa \
    --hub_model_id your-username/mimic-cxr-vqa
```

### Debug Mode (Quick Test)

```bash
python train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --max_samples 100 \
    --epochs 2 \
    --disable_wandb \
    --skip_data_check
```

### Multi-GPU Training

```bash
# Automatic with DataParallel
python train_mimic_cxr.py --config configs/default_config.yaml

# With DeepSpeed
deepspeed train_mimic_cxr.py \
    --config configs/default_config.yaml \
    --deepspeed configs/deepspeed_config.json
```

### Training Output

```
checkpoints/mimic-cxr-vqa/
├── config.json
├── checkpoint-5000/
│   ├── pytorch_model.bin
│   └── training_metadata.json
├── checkpoint-10000/
├── best_model/
│   └── pytorch_model.bin
└── hub_upload/
```

---

## Phase 3: Evaluation

### Run Evaluation

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model \
    --config configs/default_config.yaml
```

### With Attention Analysis

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model \
    --config configs/default_config.yaml \
    --compute_attention
```

### Evaluation Output

```
evaluation_results/
└── evaluation_report.json
```

### Metrics Computed

#### Answer Accuracy Metrics
| Metric | Description |
|--------|-------------|
| Exact Match | % of exact answer matches |
| F1 Score | Token-level F1 |
| BLEU-4 | 4-gram precision for fluency |
| ROUGE-L | Longest common subsequence recall |
| BERTScore | Semantic similarity via BERT embeddings |

#### Spatial Reasoning Metrics
| Metric | Description |
|--------|-------------|
| Mean IoU | Average Intersection over Union |
| Pointing Accuracy | % with IoU > 0.5 |
| mAP@0.5 | Mean Average Precision at IoU=0.5 |
| mAP@0.75 | Mean Average Precision at IoU=0.75 |

#### Clinical Relevance Metrics
| Metric | Description |
|--------|-------------|
| Sensitivity | True Positive Rate |
| Specificity | True Negative Rate |
| PPV | Positive Predictive Value |
| NPV | Negative Predictive Value |
| MCC | Matthews Correlation Coefficient |
| AUROC | Area Under ROC Curve |

#### Per-Head Accuracy
| Head | Description |
|------|-------------|
| Binary | Yes/No questions |
| Category | Finding type classification |
| Region | Anatomical region identification |
| Severity | Severity level classification |

#### Explainability Metrics
| Metric | Description |
|--------|-------------|
| Attention Plausibility | IoU between attention and radiologist ROIs |
| Attention Entropy | Focus vs. diffusion of attention |

---

## Configuration Reference

### Model Configuration

```yaml
model:
  visual_backbone: "convnext_base"      # ConvNeXt-Base
  text_encoder: "emilyalsentzer/Bio_ClinicalBERT"
  visual_feature_dim: 512
  scene_graph_dim: 134                  # 6 bbox + 64 region + 64 entity
  hidden_size: 768
  num_hidden_layers: 6
  num_attention_heads: 12
  sim_layers: 2                         # Scene-Embedded Interaction layers
```

### Training Configuration

```yaml
training:
  batch_size_per_gpu: 16
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  num_epochs: 20
  fp16: true                            # Mixed precision
  gradient_checkpointing: true
  
  # Loss weights
  vqa_loss_weight: 1.0
  chexpert_loss_weight: 0.3             # Auxiliary supervision weight
```

### Data Configuration

```yaml
data:
  mimic_cxr_jpg_path: "/path/to/MIMIC-CXR-JPG"
  mimic_ext_cxr_qba_path: "/path/to/MIMIC-Ext-CXR-QBA"
  quality_grade: "A"                    # A for fine-tuning, B for pre-training
  view_filter: "frontal_only"
```

---

## Troubleshooting

### "DATA ANALYSIS NOT FOUND" Error

```bash
# Run the analysis first
python analyze_data.py \
    --mimic_cxr_path /path/to/MIMIC-CXR-JPG \
    --mimic_qa_path /path/to/MIMIC-Ext-CXR-QBA
```

### Out of Memory

```yaml
# In config, reduce batch size and enable memory optimization
training:
  batch_size_per_gpu: 8
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  fp16: true
```

### Slow Training

```yaml
# Increase workers and enable pin memory
training:
  dataloader_num_workers: 12
  dataloader_pin_memory: true
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Architecture

Based on SSG-VQA-Net adapted for chest X-ray VQA:

```
                    ┌──────────────────┐
                    │   Input Image    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  ConvNeXt-Base   │ ← Visual Backbone
                    │   (512 dims)     │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼─────────┐  ┌───────▼────────┐
│ Scene Graph    │  │  Question Text   │  │   ROI Features │
│ Encoder        │  │ Bio+ClinicalBERT │  │                │
│ (134 dims)     │  │   (768 dims)     │  │   (512 dims)   │
└───────┬────────┘  └────────┬─────────┘  └───────┬────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │ Scene-Embedded   │ ← Cross-modal Attention
                    │ Interaction (SIM)│
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
   ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
   │ Binary Head   │ │ Category Head │ │ CheXpert Head │
   │  (Yes/No)     │ │  (Findings)   │ │ (Auxiliary)   │
   └───────────────┘ └───────────────┘ └───────────────┘
```

---

## Citation

```bibtex
@article{ssg-vqa-mimic,
  title={Scene Graph-Enhanced VQA for Chest X-Ray Analysis},
  year={2026}
}
```
