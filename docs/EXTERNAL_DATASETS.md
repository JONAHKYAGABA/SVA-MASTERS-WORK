# External Datasets for Cross-Dataset Evaluation

This document describes the external datasets required for cross-dataset evaluation as specified in **methodology Section 16.3: Cross-Dataset Evaluation**.

## Overview

Zero-shot cross-dataset evaluation tests the model's generalization ability on medical VQA datasets it was **never trained on**. This is critical for assessing real-world clinical utility.

## Required Datasets

### 1. VQA-RAD (Radiology VQA)

| Property | Value |
|----------|-------|
| **Images** | 315 radiology images |
| **QA Pairs** | 3,515 question-answer pairs |
| **Question Types** | Modality, Plane, Organ, Abnormality, Presence, etc. |
| **Answer Types** | CLOSED (Yes/No) and OPEN (free-form) |
| **Modalities** | CT, MRI, X-ray |

**Download**: https://osf.io/89kps/

**Reference**:
> Lau et al., "A Dataset of Clinically Generated Visual Questions and Answers about Radiology Images" (2018)

**Directory Structure**:
```
external_datasets/
└── vqa_rad/
    ├── images/
    │   ├── synpic100132.jpg
    │   ├── synpic100176.jpg
    │   └── ...
    ├── trainset.json
    ├── valset.json
    └── testset.json
```

---

### 2. SLAKE (Semantically-Labeled Knowledge-Enhanced)

| Property | Value |
|----------|-------|
| **Images** | 701 radiology images |
| **QA Pairs** | ~14,000 question-answer pairs |
| **Languages** | English and Chinese |
| **Features** | Semantic labels, knowledge-enhanced |
| **Modalities** | CT, MRI, X-ray |

**Download**: https://www.med-vqa.com/slake/

**Reference**:
> Liu et al., "SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering" (2021)

**Directory Structure**:
```
external_datasets/
└── slake/
    ├── imgs/
    │   ├── xmlab0/
    │   │   └── source.jpg
    │   ├── xmlab1/
    │   └── ...
    ├── train.json
    ├── validate.json
    └── test.json
```

---

## Setup Instructions

### Step 1: Create Directory Structure

```bash
mkdir -p external_datasets/vqa_rad
mkdir -p external_datasets/slake
```

### Step 2: Download VQA-RAD

1. Go to https://osf.io/89kps/
2. Download all files
3. Extract images to `external_datasets/vqa_rad/images/`
4. Place JSON files in `external_datasets/vqa_rad/`

### Step 3: Download SLAKE

1. Go to https://www.med-vqa.com/slake/
2. Request access (academic use)
3. Download and extract to `external_datasets/slake/`

### Step 4: Verify Setup

```bash
python -c "from data.external_datasets import check_external_datasets; check_external_datasets()"
```

Expected output:
```
============================================================
EXTERNAL DATASETS STATUS
============================================================
  VQA_RAD: ✅ AVAILABLE
  SLAKE: ✅ AVAILABLE
============================================================
```

---

## Running Cross-Dataset Evaluation

### Using evaluate.py

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model \
    --cross_dataset \
    --vqa_rad_path ./external_datasets/vqa_rad \
    --slake_path ./external_datasets/slake
```

### Using Python API

```python
from data.external_datasets import get_external_dataset, create_external_dataloader

# Load VQA-RAD
vqa_rad = get_external_dataset('vqa_rad', './external_datasets/vqa_rad', split='test')
vqa_rad_loader = create_external_dataloader(vqa_rad, batch_size=32)

# Load SLAKE
slake = get_external_dataset('slake', './external_datasets/slake', split='test')
slake_loader = create_external_dataloader(slake, batch_size=32)
```

---

## Expected Performance

Based on methodology Section 16.3, expected cross-dataset generalization:

| Dataset | Metric | Expected Range |
|---------|--------|----------------|
| VQA-RAD | Binary Accuracy | 65-75% |
| VQA-RAD | Exact Match | 55-65% |
| SLAKE | Binary Accuracy | 60-70% |
| SLAKE | Exact Match | 50-60% |

**Note**: These are zero-shot results (no fine-tuning on external datasets).

---

## Troubleshooting

### VQA-RAD: "Image not found"

VQA-RAD images may be in different directories depending on the download:
- `images/`
- `VQA_RAD Image Folder/`
- `Images/`

The loader checks all these paths automatically.

### SLAKE: "File not found"

Ensure the JSON files are named correctly:
- `train.json` (not `Train.json`)
- `validate.json` (not `validation.json`)
- `test.json`

### Permission Issues

Some datasets require academic access. Ensure you have:
1. Agreed to data use agreements
2. Downloaded using authorized credentials

---

## License & Citation

### VQA-RAD
```bibtex
@article{lau2018dataset,
  title={A dataset of clinically generated visual questions and answers about radiology images},
  author={Lau, Jason J and Gayen, Soumya and Ben Abacha, Asma and Demner-Fushman, Dina},
  journal={Scientific data},
  volume={5},
  number={1},
  pages={1--10},
  year={2018}
}
```

### SLAKE
```bibtex
@inproceedings{liu2021slake,
  title={Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering},
  author={Liu, Bo and Zhan, Li-Ming and Xu, Li and Ma, Lin and Yang, Yan and Wu, Xiao-Ming},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  pages={1650--1654},
  year={2021},
  organization={IEEE}
}
```

