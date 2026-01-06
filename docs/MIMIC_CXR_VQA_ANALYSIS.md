# Detailed Analysis: Adapting SSG-VQA-Net for Chest X-Ray Visual Question Answering

## Executive Summary

This document provides a comprehensive analysis of adapting the SSG-VQA-Net architecture (originally designed for surgical video VQA) to work with chest X-ray images using the MIMIC-CXR-JPG and MIMIC-Ext-CXR-QBA datasets.

---

## 1. Current State Assessment

### 1.1 What We Have

| Component | Source | Description |
|-----------|--------|-------------|
| **Model Architecture** | SSG-VQA-Net | VisualBERT-based model with Scene-embedded Interaction Module (SIM) |
| **Image Dataset** | MIMIC-CXR-JPG | 377,110 chest X-ray images (570 GB) for 65,317 patients |
| **VQA Dataset** | MIMIC-Ext-CXR-QBA | 42 million QA pairs with scene graphs, bounding boxes (26 GB) |
| **Scene Graphs** | MIMIC-Ext-CXR-QBA | Anatomical regions, observations, relationships, bounding boxes |
| **Radiologist Annotations** | MIMIC-CXR-JPG | CheXpert/NegBio labels (14 categories) + Gold test set labels |
| **Structured Labels** | mimic-cxr-2.0.0-chexpert.csv | 227,827 studies with automated CheXpert labels |
| **Gold Standard Labels** | mimic-cxr-2.1.0-test-set-labeled.csv | Radiologist-annotated test set (ground truth) |

### 1.2 Key Architectural Differences: Video → Image

| Aspect | SSG-VQA (Surgical) | MIMIC-CXR (Radiology) |
|--------|-------------------|----------------------|
| **Input Type** | Video frames (temporal sequence) | Static chest X-ray images |
| **Domain** | Laparoscopic surgery | Chest radiology |
| **Objects** | Surgical instruments + anatomies | Anatomical structures + pathologies |
| **Scene Graph Source** | YOLO detection + spatial relations | LLM extraction + anatomical localization |
| **Answer Classes** | 51 classes (instruments, actions, anatomies) | 237 finding classes, 310 region classes |
| **Scale** | ~45 videos, thousands of frames | 377,110 images, 227,239 studies |
| **Patients** | Not patient-centric | 65,317 patients with multiple studies |

---

## 2. Dataset Correlation Analysis

### 2.1 Patient Record Linking Strategy

Both datasets share the **same patient and study identifiers** from MIMIC-CXR:

```
MIMIC-CXR-JPG (Images)          MIMIC-Ext-CXR-QBA (QA + Scene Graphs)
        │                                    │
        │ subject_id (patient_id)            │ patient_id
        │ study_id                           │ study_id
        │ dicom_id (image_id)                │ image_id
        │                                    │
        └──────────── LINKED BY ─────────────┘
                   subject_id + study_id
```

### 2.2 Data Hierarchy Structure

```
PATIENT (subject_id / patient_id)
    │
    ├── STUDY 1 (study_id)
    │       ├── Image A (dicom_id / image_id)  ← MIMIC-CXR-JPG
    │       ├── Image B (dicom_id / image_id)  ← MIMIC-CXR-JPG
    │       ├── Scene Graph                     ← MIMIC-Ext-CXR-QBA
    │       └── QA Pairs (multiple)             ← MIMIC-Ext-CXR-QBA
    │
    ├── STUDY 2 (study_id)
    │       ├── Image C
    │       ├── Scene Graph
    │       └── QA Pairs
    │
    └── ... more studies
```

### 2.3 Quantitative Breakdown

| Level | MIMIC-CXR-JPG | MIMIC-Ext-CXR-QBA | Notes |
|-------|---------------|-------------------|-------|
| **Patients** | 65,379 | 65,317 | ~99.9% overlap |
| **Studies** | 227,827 | 227,239 | ~99.7% overlap |
| **Images** | 377,110 | 376,175 (with bboxes) | Near complete |
| **QA Pairs** | N/A | 42,172,827 | Per study, not per image |
| **Scene Graphs** | N/A | 227,239 | One per study |

### 2.4 Images Per Patient Distribution

Based on the dataset statistics:
- **Average studies per patient**: 227,239 ÷ 65,317 ≈ **3.5 studies/patient**
- **Average images per study**: 377,110 ÷ 227,239 ≈ **1.66 images/study**
- **Average images per patient**: 377,110 ÷ 65,317 ≈ **5.8 images/patient**

**Image View Positions** (critical for bounding box quality):
- Frontal views (AP/PA): Primary focus for training (better bbox quality)
- Lateral views: Secondary (poorer localization, excluded in fine-tuning grade)

### 2.5 Scene Graph Quality Statistics (Known Noise Characteristics)

The MIMIC-Ext-CXR-QBA scene graphs are automatically generated via LLM-based extraction and atlas-based bounding box detection from the Chest ImaGenome dataset. While quality filtering based on model confidence is applied, these scene graphs contain inherent noise that models must handle robustly.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SCENE GRAPH NOISE CHARACTERISTICS                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  BOUNDING BOX QUALITY ISSUES:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ • 5.6% of bounding boxes require correction                             │    │
│  │ • 0.8% of bounding boxes are missing entirely                           │    │
│  │ • 99.996% of boxes exhibit overlap with other boxes                     │    │
│  │ • Average maximum IoU between overlapping boxes: 37.1%                  │    │
│  │   (causes anatomical location ambiguity)                                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  IMPLICATIONS FOR TRAINING:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ • Models must be robust to noisy spatial annotations                    │    │
│  │ • Quality filtering (A/A+/A++ grades) reduces but doesn't eliminate     │    │
│  │ • Scene graph augmentation (dropout, perturbation) improves robustness  │    │
│  │ • Cross-dataset evaluation should disable scene graph input             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

| Issue Type | Prevalence | Impact | Mitigation Strategy |
|------------|------------|--------|---------------------|
| **Boxes needing correction** | 5.6% | Incorrect localization | Quality filtering (A+ grade) |
| **Missing boxes** | 0.8% | Entity not localized | Default to full image region |
| **Overlapping boxes** | 99.996% | Ambiguous boundaries | Learned disambiguation via embeddings |
| **High IoU overlap** | 37.1% avg | Anatomical confusion | Multi-region aggregation |

---

## 2.6 Data Exploration and Bias Mitigation

### 2.6.1 Pre-Training Distribution Analysis

Prior to training, comprehensive exploratory analysis characterizes three critical dimensions:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DATA DISTRIBUTION ANALYSIS DIMENSIONS                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DIMENSION 1: OBSERVATION POLARITY                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Positive findings (abnormality present)    vs    Negative findings     │    │
│  │  • Cardiomegaly: present                         • No cardiomegaly      │    │
│  │  • Consolidation: observed                       • Lungs clear          │    │
│  │  • Pleural effusion: bilateral                   • No effusion          │    │
│  │                                                                          │    │
│  │  GOAL: Achieve 50/50 balance to prevent model bias toward negatives     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  DIMENSION 2: ANATOMICAL REGION REPRESENTATION                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Region Category    │ Expected Coverage │ Risk if Underrepresented      │    │
│  │  ───────────────────┼───────────────────┼─────────────────────────────  │    │
│  │  Cardiac            │ 20-25%            │ Miss cardiomegaly patterns    │    │
│  │  Pulmonary          │ 35-40%            │ Miss lung pathologies         │    │
│  │  Pleural            │ 15-20%            │ Miss effusions, thickening    │    │
│  │  Mediastinal        │ 10-15%            │ Miss masses, lymphadenopathy  │    │
│  │  Osseous            │ 5-10%             │ Miss fractures, deformities   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  DIMENSION 3: QUESTION COMPLEXITY STRATIFICATION                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Complexity Level   │ Example                      │ Reasoning Required  │    │
│  │  ───────────────────┼──────────────────────────────┼──────────────────── │    │
│  │  Single-entity      │ "Is there cardiomegaly?"     │ Direct lookup       │    │
│  │  Multi-entity       │ "What findings are present?" │ Enumeration         │    │
│  │  Relational         │ "Is the mass near the hilum?"│ Spatial reasoning   │    │
│  │  Multi-hop          │ "What caused the effusion?"  │ Causal inference    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.6.2 Multi-Level Augmentation Strategies

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    AUGMENTATION STRATEGY PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LEVEL 1: IMAGE-LEVEL AUGMENTATION                                               │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Transformation         │ Parameters      │ Notes                       │    │
│  │  ───────────────────────┼─────────────────┼──────────────────────────── │    │
│  │  Random Rotation        │ ±5° maximum     │ Limited - anatomy oriented  │    │
│  │  Horizontal Flip        │ With label adj. │ ★ CRITICAL: Adjust L/R labels│   │
│  │  Brightness/Contrast    │ ±10%            │ Simulate exposure variation │    │
│  │  Gaussian Noise         │ σ = 0.01        │ Simulate image artifacts    │    │
│  │                                                                          │    │
│  │  ⚠️  NO VERTICAL FLIP - Anatomically invalid                            │    │
│  │  ⚠️  NO LARGE ROTATIONS - Destroys anatomical orientation               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  LEVEL 2: SCENE GRAPH AUGMENTATION                                               │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Node Dropout:     10% of observation nodes randomly dropped            │    │
│  │                    • Simulates missing/incomplete scene graphs          │    │
│  │                    • Prevents over-reliance on specific nodes           │    │
│  │                                                                          │    │
│  │  Edge Perturbation: 5% of region-region relations randomly modified     │    │
│  │                    • "left_of" → "adjacent_to" (spatial softening)      │    │
│  │                    • Improves robustness to annotation noise            │    │
│  │                                                                          │    │
│  │  Bbox Jittering:   ±5% coordinate perturbation                          │    │
│  │                    • Simulates localization uncertainty                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  LEVEL 3: QUESTION-LEVEL AUGMENTATION                                            │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Synonym Replacement:                                                   │    │
│  │    "Is there cardiomegaly?" → "Is the heart enlarged?"                  │    │
│  │    "pneumothorax" → "collapsed lung"                                    │    │
│  │    Using UMLS medical thesaurus for valid synonyms                      │    │
│  │                                                                          │    │
│  │  Back-Translation:                                                       │    │
│  │    English → German → English (paraphrase generation)                   │    │
│  │    Enhances linguistic diversity while preserving meaning               │    │
│  │                                                                          │    │
│  │  Template Variation:                                                     │    │
│  │    "Is X present?" → "Can you identify X?" → "Do you see X?"            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.6.3 Stratified Sampling Configuration

| Dimension | Target Balance | Sampling Strategy |
|-----------|----------------|-------------------|
| **Polarity** | 50% positive / 50% negative | Weighted random sampling per batch |
| **Anatomical Region** | Uniform across 5 categories | Region-stratified mini-batches |
| **Question Complexity** | Even distribution | Complexity-aware curriculum |
| **Question Type** | Proportional to test distribution | Type-balanced sampling |

---

## 3. Radiologist Annotations & Structured Labels Utilization

### 3.1 Available Annotation Sources

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MIMIC-CXR ANNOTATION HIERARCHY                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LEVEL 1: Automated Labels (Training & Validation)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ mimic-cxr-2.0.0-chexpert.csv.gz  (227,827 studies)                      │    │
│  │ mimic-cxr-2.0.0-negbio.csv.gz    (227,827 studies)                      │    │
│  │                                                                          │    │
│  │ 14 CheXpert Categories:                                                  │    │
│  │   • Atelectasis        • Lung Opacity        • Pneumothorax             │    │
│  │   • Cardiomegaly       • Pleural Effusion    • Pleural Other            │    │
│  │   • Consolidation      • Pneumonia           • Support Devices          │    │
│  │   • Edema              • Enlarged Cardio-    • No Finding               │    │
│  │   • Fracture             mediastinum                                    │    │
│  │   • Lung Lesion                                                          │    │
│  │                                                                          │    │
│  │ Label Values:                                                            │    │
│  │   1.0  = Positive (finding present)                                     │    │
│  │   0.0  = Negative (finding absent)                                      │    │
│  │  -1.0  = Uncertain (may or may not be present)                          │    │
│  │  blank = Not mentioned in report                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  LEVEL 2: Radiologist Annotations (Test Set Gold Standard)                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ mimic-cxr-2.1.0-test-set-labeled.csv                                    │    │
│  │                                                                          │    │
│  │ • Same 14 categories as CheXpert                                        │    │
│  │ • Manually annotated by radiologists                                    │    │
│  │ • Ground truth for final evaluation                                     │    │
│  │ • Used to evaluate NegBio and CheXpert classifiers                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 How Annotations Are Utilized in the Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ANNOTATION UTILIZATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║  USE CASE 1: MULTI-TASK AUXILIARY SUPERVISION                             ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  CheXpert Labels (14 categories) serve as auxiliary supervision signal:   ║  │
│  ║                                                                           ║  │
│  ║  Main Task Loss:     L_vqa = CrossEntropy(predicted_answer, true_answer)  ║  │
│  ║  Auxiliary Loss:     L_aux = BCE(predicted_labels, chexpert_labels)       ║  │
│  ║  Combined Loss:      L_total = L_vqa + λ * L_aux   (λ = 0.3 recommended)  ║  │
│  ║                                                                           ║  │
│  ║  Benefits:                                                                ║  │
│  ║    • Regularizes visual encoder to detect clinically relevant features   ║  │
│  ║    • Improves performance on "has_finding" question types                ║  │
│  ║    • Provides structured supervision beyond free-text QA pairs           ║  │
│  ║                                                                           ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║  USE CASE 2: ANSWER VALIDATION & CONSISTENCY CHECK                        ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  Scene Graph QA Answer    ──────►  Compare  ◄──────  CheXpert Label       ║  │
│  ║  "Is there cardiomegaly?"           │              Cardiomegaly: 1.0      ║  │
│  ║  Answer: "Yes"                      ▼                                     ║  │
│  ║                              Consistency Score                            ║  │
│  ║                                                                           ║  │
│  ║  Use cases:                                                               ║  │
│  ║    • Filter noisy QA pairs where answer contradicts CheXpert label       ║  │
│  ║    • Weight training samples by consistency (higher weight = consistent)  ║  │
│  ║    • Quality-aware training using label agreement                         ║  │
│  ║                                                                           ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║  USE CASE 3: STRATIFIED EVALUATION BY PATHOLOGY                           ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  Test Set with Radiologist Labels enables per-pathology metrics:          ║  │
│  ║                                                                           ║  │
│  ║  Pathology          │ Accuracy │ Sensitivity │ Specificity │ F1          ║  │
│  ║  ───────────────────┼──────────┼─────────────┼─────────────┼─────────    ║  │
│  ║  Cardiomegaly       │   XX%    │    XX%      │     XX%     │  XX%        ║  │
│  ║  Pneumothorax       │   XX%    │    XX%      │     XX%     │  XX%        ║  │
│  ║  Pleural Effusion   │   XX%    │    XX%      │     XX%     │  XX%        ║  │
│  ║  Consolidation      │   XX%    │    XX%      │     XX%     │  XX%        ║  │
│  ║  ...                │   ...    │    ...      │     ...     │  ...        ║  │
│  ║                                                                           ║  │
│  ║  Gold standard labels ensure fair, unbiased evaluation                    ║  │
│  ║                                                                           ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║  USE CASE 4: CLASS WEIGHTING FOR IMBALANCED LABELS                        ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  CheXpert label distribution informs loss weighting:                      ║  │
│  ║                                                                           ║  │
│  ║  Common findings:     "Support Devices", "No Finding"  → lower weight     ║  │
│  ║  Rare findings:       "Pneumothorax", "Fracture"       → higher weight    ║  │
│  ║                                                                           ║  │
│  ║  Weight calculation:   w_i = (N_total / N_class_i) ^ 0.5                  ║  │
│  ║                                                                           ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║  USE CASE 5: UNCERTAINTY HANDLING (-1.0 LABELS)                           ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  Strategy for uncertain labels (-1.0):                                    ║  │
│  ║                                                                           ║  │
│  ║  Option A: Exclude - Don't use uncertain samples for that finding        ║  │
│  ║  Option B: Soft Label - Treat as probability 0.5 (BCE with soft targets) ║  │
│  ║  Option C: U-Ignore - Use special "ignore" mask in loss computation      ║  │
│  ║                                                                           ║  │
│  ║  RECOMMENDED: Option C (U-Ignore) - maximizes data usage                  ║  │
│  ║                                                                           ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Multi-Task Learning Architecture with CheXpert Labels

```python
class MIMICCXRVQAWithLabels(nn.Module):
    """
    Extended VQA model with auxiliary CheXpert label prediction.
    """
    def __init__(self, vqa_model, num_chexpert_classes=14):
        super().__init__()
        
        # Main VQA model (SSG-VQA-Net adapted)
        self.vqa_model = vqa_model
        
        # Auxiliary CheXpert classifier (multi-label)
        self.chexpert_head = nn.Sequential(
            nn.Linear(1024, 512),  # From visual encoder hidden dim
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_chexpert_classes)  # 14 CheXpert categories
        )
        
    def forward(self, images, questions, scene_graphs):
        # Main VQA forward pass
        vqa_output, visual_features = self.vqa_model(images, questions, scene_graphs)
        
        # Auxiliary CheXpert prediction from visual features
        # Use global pooled visual features before fusion
        visual_pooled = visual_features.mean(dim=1)  # (B, 1024)
        chexpert_logits = self.chexpert_head(visual_pooled)  # (B, 14)
        
        return {
            'vqa_output': vqa_output,
            'chexpert_logits': chexpert_logits
        }


class MultiTaskLoss(nn.Module):
    """
    Combined loss for VQA + CheXpert auxiliary task.
    """
    def __init__(self, vqa_weight=1.0, chexpert_weight=0.3):
        super().__init__()
        self.vqa_weight = vqa_weight
        self.chexpert_weight = chexpert_weight
        self.vqa_loss = nn.CrossEntropyLoss()
        self.chexpert_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, outputs, vqa_targets, chexpert_targets, chexpert_mask):
        """
        Args:
            outputs: Dict with 'vqa_output' and 'chexpert_logits'
            vqa_targets: (B,) answer indices
            chexpert_targets: (B, 14) CheXpert labels (1.0, 0.0, or 0.5 for uncertain)
            chexpert_mask: (B, 14) mask where 1 = use this label, 0 = ignore (uncertain)
        """
        # VQA loss
        l_vqa = self.vqa_loss(outputs['vqa_output'], vqa_targets)
        
        # CheXpert loss with uncertainty masking
        l_chexpert_raw = self.chexpert_loss(outputs['chexpert_logits'], chexpert_targets)
        l_chexpert = (l_chexpert_raw * chexpert_mask).sum() / (chexpert_mask.sum() + 1e-8)
        
        # Combined loss
        total_loss = self.vqa_weight * l_vqa + self.chexpert_weight * l_chexpert
        
        return total_loss, {'vqa_loss': l_vqa, 'chexpert_loss': l_chexpert}
```

### 3.4 CheXpert Label Loading Pipeline

```python
import pandas as pd

class CheXpertLabelLoader:
    """
    Loads and preprocesses CheXpert labels for training and evaluation.
    """
    
    CHEXPERT_CATEGORIES = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
        'Lung Opacity', 'Pleural Effusion', 'Pneumonia',
        'Pneumothorax', 'Pleural Other', 'Support Devices', 'No Finding'
    ]
    
    def __init__(self, chexpert_csv_path, test_labels_csv_path=None):
        """
        Args:
            chexpert_csv_path: Path to mimic-cxr-2.0.0-chexpert.csv.gz
            test_labels_csv_path: Path to mimic-cxr-2.1.0-test-set-labeled.csv (gold standard)
        """
        # Load automated labels for train/val
        self.chexpert_df = pd.read_csv(chexpert_csv_path, compression='gzip')
        self.chexpert_df.set_index(['subject_id', 'study_id'], inplace=True)
        
        # Load radiologist labels for test (if provided)
        if test_labels_csv_path:
            self.test_labels_df = pd.read_csv(test_labels_csv_path)
            self.test_labels_df.set_index('study_id', inplace=True)
        else:
            self.test_labels_df = None
            
    def get_labels(self, subject_id, study_id, is_test=False):
        """
        Get CheXpert labels for a study.
        
        Returns:
            labels: np.array of shape (14,) with values in {0, 1, 0.5}
            mask: np.array of shape (14,) with 1 for valid labels, 0 for uncertain
        """
        import numpy as np
        
        labels = np.zeros(14)
        mask = np.ones(14)  # 1 = use this label
        
        try:
            if is_test and self.test_labels_df is not None:
                row = self.test_labels_df.loc[study_id]
            else:
                row = self.chexpert_df.loc[(subject_id, study_id)]
                
            for i, cat in enumerate(self.CHEXPERT_CATEGORIES):
                val = row.get(cat, np.nan)
                
                if pd.isna(val):
                    # Missing = not mentioned, treat as negative
                    labels[i] = 0.0
                    mask[i] = 0.5  # Lower confidence for missing
                elif val == 1.0:
                    labels[i] = 1.0
                elif val == 0.0:
                    labels[i] = 0.0
                elif val == -1.0:
                    # Uncertain - use soft label and reduced mask
                    labels[i] = 0.5  # Soft target
                    mask[i] = 0.0    # Ignore in loss (or use 0.5 for partial)
                    
        except KeyError:
            # Study not found, return zeros with low mask
            mask = np.zeros(14)
            
        return labels, mask
```

---

## 4. YOLOv8 Object Detection Module (Methodology Upgrade)

### 4.1 Role of YOLOv8 in the Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OBJECT DETECTION IN SSG-VQA PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ORIGINAL SSG-VQA (Surgical):                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Video Frame → YOLOv5 → Detected Objects → Scene Graph → VQA Model      │    │
│  │                  │                                                       │    │
│  │                  └── Real-time detection of instruments & anatomies     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  MIMIC-CXR ADAPTATION (Two Modes):                                               │
│                                                                                  │
│  MODE A: Training on MIMIC-CXR (Scene Graphs Pre-computed)                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  X-ray Image ──────┐                                                    │    │
│  │                    ├──► ConvNeXt Feature Extraction ──► VQA Model       │    │
│  │  Scene Graph ──────┘    (using pre-computed bboxes)                     │    │
│  │  (from MIMIC-Ext-CXR-QBA)                                               │    │
│  │                                                                          │    │
│  │  ★ YOLOv8 NOT needed here - bboxes already provided in dataset          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  MODE B: Inference on New Images / Cross-Dataset (VQA-RAD, SLAKE)                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  New X-ray Image → YOLOv8 Detection → Generate Scene Graph → VQA Model  │    │
│  │                        │                                                 │    │
│  │                        └── ★ YOLOv8 REQUIRED for new image detection    │    │
│  │                                                                          │    │
│  │  Use cases:                                                              │    │
│  │    • VQA-RAD dataset (no scene graphs provided)                         │    │
│  │    • SLAKE-EN dataset (no scene graphs provided)                        │    │
│  │    • Real-world deployment on new chest X-rays                          │    │
│  │    • Refining noisy MIMIC-Ext-CXR-QBA bboxes (optional)                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  MODE C: Ablation Study (No Scene Graph Baseline)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  X-ray Image → ConvNeXt (global features only) → VQA Model              │    │
│  │                                                                          │    │
│  │  ★ YOLOv8 disabled - tests model without any detection/scene graph      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 YOLOv8 vs YOLOv5 Upgrade Justification

From **methodology.md** section on Object Detection Adjustment:

| Aspect | YOLOv5 (Original) | YOLOv8 (Upgraded) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Detection Paradigm** | Anchor-based | Anchor-free | Simpler, more flexible |
| **Head Design** | Coupled head | Decoupled heads | Independent optimization |
| **Feature Fusion** | PANet | C2f module | Enhanced multi-scale |
| **Medical Precision** | ~85% | Up to 99.17% | +14.17% |
| **Medical Sensitivity** | ~80% | Up to 97.5% | +17.5% |
| **Lung Cancer Detection** | ~75% precision | 90.32% precision | +15.32% |

**Expected improvements for chest X-ray scene graphs:**
- **mAP for anatomical structures**: +3.5-5.8%
- **mAP for pathologies**: +6.2-9.4%

### 4.3 YOLOv8 Detection Classes for Chest X-ray

```python
# YOLOv8 trained on chest X-ray anatomical structures and findings
YOLOV8_CXR_CLASSES = {
    # Anatomical Structures (26 regions)
    'anatomical': [
        'left_lung', 'right_lung', 'heart', 'aorta', 'trachea',
        'mediastinum', 'left_clavicle', 'right_clavicle', 'spine',
        'left_diaphragm', 'right_diaphragm', 'left_costophrenic_angle',
        'right_costophrenic_angle', 'left_hilum', 'right_hilum',
        'carina', 'cardiac_silhouette', 'aortic_arch',
        'left_upper_lobe', 'left_lower_lobe', 'right_upper_lobe',
        'right_middle_lobe', 'right_lower_lobe', 'pleura',
        'abdomen_visible', 'neck_visible'
    ],
    
    # Pathological Findings (14 CheXpert categories + extras)
    'findings': [
        'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
        'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
        'lung_opacity', 'pleural_effusion', 'pneumonia',
        'pneumothorax', 'pleural_thickening', 'mass', 'nodule',
        'infiltrate', 'fibrosis', 'emphysema'
    ],
    
    # Medical Devices
    'devices': [
        'endotracheal_tube', 'central_venous_catheter', 'chest_tube',
        'pacemaker', 'picc_line', 'nasogastric_tube', 'tracheostomy',
        'surgical_clips', 'port_catheter'
    ]
}
```

### 4.4 YOLOv8 Integration Code

```python
from ultralytics import YOLO
import torch
import numpy as np

class YOLOv8ChestXrayDetector:
    """
    YOLOv8 detector for chest X-ray anatomical structures and findings.
    Used for:
      1. Inference on new images (cross-dataset evaluation)
      2. Optional refinement of MIMIC-Ext-CXR-QBA bounding boxes
      3. Scene graph generation for deployment
    """
    
    def __init__(self, model_path='yolov8_cxr_finetuned.pt', conf_threshold=0.25):
        """
        Args:
            model_path: Path to fine-tuned YOLOv8 weights for chest X-ray
            conf_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Class mapping (aligned with MIMIC-Ext-CXR-QBA regions)
        self.class_names = self._build_class_mapping()
        
    def _build_class_mapping(self):
        """Build mapping from YOLOv8 class IDs to region/entity names."""
        return {
            0: 'left_lung', 1: 'right_lung', 2: 'heart', 3: 'aorta',
            4: 'trachea', 5: 'mediastinum', 6: 'cardiomegaly',
            7: 'consolidation', 8: 'pleural_effusion', 9: 'pneumothorax',
            # ... (full mapping for 50+ classes)
        }
        
    def detect(self, image):
        """
        Run YOLOv8 detection on chest X-ray image.
        
        Args:
            image: numpy array (H, W, 3) or PIL Image
            
        Returns:
            detections: List of dicts with keys:
                - 'class': str (region/finding name)
                - 'bbox': [x1, y1, x2, y2] normalized
                - 'confidence': float
                - 'category': 'anatomical' | 'finding' | 'device'
        """
        results = self.model(image, conf=self.conf_threshold)[0]
        
        detections = []
        h, w = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            detections.append({
                'class': self.class_names.get(cls_id, f'class_{cls_id}'),
                'bbox': [x1/w, y1/h, x2/w, y2/h],  # Normalized
                'bbox_abs': [x1, y1, x2, y2],      # Absolute
                'confidence': conf,
                'category': self._get_category(cls_id)
            })
            
        return detections
    
    def _get_category(self, cls_id):
        """Determine if detection is anatomical, finding, or device."""
        if cls_id < 26:
            return 'anatomical'
        elif cls_id < 43:
            return 'finding'
        else:
            return 'device'
            
    def generate_scene_graph(self, image):
        """
        Generate a scene graph from YOLOv8 detections.
        Compatible with MIMIC-Ext-CXR-QBA format.
        
        Returns:
            scene_graph: Dict in MIMIC-Ext-CXR-QBA format
        """
        detections = self.detect(image)
        
        scene_graph = {
            'observations': {},
            'regions': {},
            'region_region_relations': []
        }
        
        for i, det in enumerate(detections):
            obs_id = f'O{i:02d}'
            
            if det['category'] == 'finding':
                # Finding observation
                scene_graph['observations'][obs_id] = {
                    'name': det['class'],
                    'obs_entities': [det['class']],
                    'probability': 'positive' if det['confidence'] > 0.5 else 'uncertain',
                    'localization': {
                        'detected': {
                            'bboxes': [det['bbox_abs']],
                            'confidence': det['confidence']
                        }
                    }
                }
            elif det['category'] == 'anatomical':
                # Anatomical region
                scene_graph['regions'][det['class']] = {
                    'localization': {
                        'bboxes': [det['bbox_abs']],
                        'confidence': det['confidence']
                    }
                }
                
        # Generate spatial relations between detected regions
        scene_graph['region_region_relations'] = self._compute_spatial_relations(
            scene_graph['regions']
        )
        
        return scene_graph
        
    def _compute_spatial_relations(self, regions):
        """Compute spatial relations (above, below, left_of, right_of) between regions."""
        relations = []
        region_names = list(regions.keys())
        
        for i, r1 in enumerate(region_names):
            for j, r2 in enumerate(region_names):
                if i >= j:
                    continue
                    
                bbox1 = regions[r1]['localization']['bboxes'][0]
                bbox2 = regions[r2]['localization']['bboxes'][0]
                
                # Compute centroids
                cx1, cy1 = (bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2
                cx2, cy2 = (bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2
                
                # Determine relation
                if cy1 < cy2 - 50:
                    relations.append((r1, 'above', r2))
                elif cy1 > cy2 + 50:
                    relations.append((r1, 'below', r2))
                if cx1 < cx2 - 50:
                    relations.append((r1, 'left_of', r2))
                elif cx1 > cx2 + 50:
                    relations.append((r1, 'right_of', r2))
                    
        return relations
```

### 4.5 YOLOv8 Fine-Tuning Pipeline

```python
# YOLOv8 fine-tuning configuration for chest X-ray
yolov8_config = {
    # Base model
    'model': 'yolov8m.pt',  # Medium model (good balance)
    
    # Training data (from MIMIC-Ext-CXR-QBA bboxes)
    'data': 'cxr_detection.yaml',  # Points to converted YOLO format data
    
    # Training parameters
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'workers': 8,
    
    # Augmentation (medical-specific)
    'augment': True,
    'degrees': 5.0,        # Limited rotation (anatomy is oriented)
    'translate': 0.1,
    'scale': 0.2,
    'fliplr': 0.0,         # NO horizontal flip (left/right matters!)
    'flipud': 0.0,         # NO vertical flip
    'mosaic': 0.0,         # Disable mosaic (single images)
    
    # Optimization
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'weight_decay': 0.0005,
}

# Data conversion: MIMIC-Ext-CXR-QBA → YOLO format
"""
Convert scene graph bboxes to YOLO format:

MIMIC-Ext-CXR-QBA format:
{
  "observations": {
    "O01": {
      "obs_entities": ["consolidation"],
      "localization": {
        "image_id": {"bboxes": [[x1, y1, x2, y2]]}
      }
    }
  }
}

YOLO format (txt file per image):
class_id x_center y_center width height  (all normalized 0-1)
7 0.45 0.52 0.15 0.20
"""
```

### 4.6 When to Use YOLOv8 vs Pre-computed Scene Graphs

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    DECISION MATRIX: YOLOv8 USAGE                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Scenario                          │ Use YOLOv8? │ Use Pre-computed SG?    │
│  ──────────────────────────────────┼─────────────┼────────────────────     │
│  Training on MIMIC-CXR-JPG         │     ❌      │        ✅               │
│  Validation on MIMIC-CXR-JPG       │     ❌      │        ✅               │
│  Testing on MIMIC-CXR-JPG          │     ❌      │        ✅               │
│  Cross-dataset: VQA-RAD            │     ✅      │        ❌ (not avail)   │
│  Cross-dataset: SLAKE-EN           │     ✅      │        ❌ (not avail)   │
│  Real-world deployment             │     ✅      │        ❌ (not avail)   │
│  Ablation: No-SG baseline          │     ❌      │        ❌               │
│  Ablation: YOLOv8-refined bboxes   │     ✅      │        ❌               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Loading Pipeline Design

### 3.1 File Organization Comparison

**MIMIC-CXR-JPG Structure:**
```
files/
  p10/                          # First 3 chars of patient_id
    p10000032/                  # Full patient_id
      s50414267/                # study_id
        02aa804e-bde0afdd-....jpg   # dicom_id (image)
        174413ec-4ec4c1f7-....jpg   # dicom_id (image)
```

**MIMIC-Ext-CXR-QBA Structure:**
```
scene_data/
  p10/
    p10000032/
      s50414267.scene_graph.json
      s50414267.metadata.json

qa/
  p10/
    p10000032/
      s50414267.qa.json
```

### 3.2 Loading Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA LOADING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1: INDEX BUILDING                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 1. Load metadata CSVs (patient, study, question, image)          │   │
│  │ 2. Create patient → study → image mappings                       │   │
│  │ 3. Create study → QA pairs mappings                              │   │
│  │ 4. Filter by quality grade (A/A+/A++ for fine-tuning)           │   │
│  │ 5. Filter by view position (frontal only recommended)            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  ↓                                       │
│  PHASE 2: DATA CORRELATION                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ For each QA pair:                                                │   │
│  │   ├── Get study_id, patient_id                                   │   │
│  │   ├── Load scene_graph.json (scene features + bboxes)            │   │
│  │   ├── Get image paths from MIMIC-CXR-JPG                         │   │
│  │   └── Load qa.json (questions + answers)                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  ↓                                       │
│  PHASE 3: FEATURE EXTRACTION                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Visual Features (per image):                                     │   │
│  │   ├── Load JPG image                                             │   │
│  │   ├── Extract ResNet/ConvNeXt features (global)                  │   │
│  │   ├── Extract ROI features using bboxes from scene graph         │   │
│  │   └── Cache to HDF5 for efficiency                               │   │
│  │                                                                  │   │
│  │ Scene Graph Features (per study):                                │   │
│  │   ├── Parse observations (findings, regions, positiveness)       │   │
│  │   ├── Encode entity/region names (embeddings)                    │   │
│  │   ├── Extract bounding box coordinates                           │   │
│  │   └── Build adjacency matrix for graph structure                 │   │
│  │                                                                  │   │
│  │ Text Features (per QA pair):                                     │   │
│  │   ├── Tokenize question with Bio+ClinicalBERT                    │   │
│  │   └── Encode answer labels                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Dataset Split Alignment

**Official MIMIC-CXR-JPG Splits:**
```
mimic-cxr-2.0.0-split.csv:
  - train: ~368,960 images
  - validate: ~2,991 images  
  - test: ~5,159 images
```

**MIMIC-Ext-CXR-QBA Splits:**
```
  - Train: 222,180 studies, 41.2M QA pairs
  - Validation: 1,805 studies, 600K QA pairs
  - Test: 3,254 studies, 333K QA pairs
```

**Critical**: Both datasets use the **same patient-level splits** - patients in train are not in val/test.

---

## 4. Feature Format Transformation

### 4.1 Original SSG-VQA Visual Features

```python
# Current SSG-VQA format (530 dimensions per ROI)
visual_features = [
    scene_graph_features[:18],  # 18 dims: bboxes + spatial relations
    pixel_features[18:]         # 512 dims: ResNet features
]
# Shape: (num_objects, 530)
```

### 4.2 SELECTED: Option B - Expanded Feature Format (Leverage Rich Scene Graphs)

We will use the **expanded feature format** to fully leverage the rich scene graph information available in MIMIC-Ext-CXR-QBA (310 regions, 237 entities):

```python
# SELECTED: Expanded format for chest X-ray (646 dims)
visual_features = [
    # Scene Graph Component (134 dims) - EXPANDED to leverage MIMIC-Ext-CXR-QBA richness
    bbox_coords[:4],            # 4 dims: normalized (x1, y1, x2, y2)
    bbox_area[:1],              # 1 dim: normalized bbox area
    aspect_ratio[:1],           # 1 dim: bbox width/height ratio
    region_embedding[:64],      # 64 dims: learned embedding for 310 regions
    entity_embedding[:64],      # 64 dims: learned embedding for 237 finding entities
    # ─────────────────────────── 134 dims total (scene graph)
    
    # Visual Component (512 dims) - ConvNeXt-Base backbone
    pixel_features[:512]        # 512 dims: ConvNeXt-Base ROI features
    # ─────────────────────────── 512 dims total (visual)
]
# Total: 646 dims (requires VisualBertConfig.visual_embedding_dim = 646)
```

### 4.2.1 Alignment Analysis: SSG-VQA vs MIMIC-Ext-CXR-QBA

```
┌────────────────────────────────────────────────────────────────────────────────┐
│           FEATURE FORMAT ALIGNMENT: Original → Expanded                         │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  SSG-VQA ORIGINAL (530 dims)          MIMIC-CXR EXPANDED (646 dims)            │
│  ════════════════════════             ═══════════════════════════════          │
│                                                                                 │
│  SCENE GRAPH FEATURES (18 dims)       SCENE GRAPH FEATURES (134 dims)          │
│  ├─ bbox_coords (4)      ─────────►   ├─ bbox_coords (4)        ✓ same         │
│  ├─ spatial_relations (8) ────────►   ├─ bbox_area (1)          ✓ enhanced    │
│  │   (left, right, above, below...)   ├─ aspect_ratio (1)       ✓ enhanced    │
│  │                                    │                                        │
│  ├─ object_class (6)     ─────────►   ├─ region_embedding (64)  ★ EXPANDED    │
│  │   (6 surgical objects)             │   (310 anatomical regions)             │
│  │                                    │   lungs, heart, mediastinum,           │
│  │                                    │   pleura, ribs, spine, etc.            │
│  │                                    │                                        │
│  │                       ─────────►   ├─ entity_embedding (64)  ★ EXPANDED    │
│  │                                    │   (237 finding entities)               │
│  │                                    │   consolidation, cardiomegaly,         │
│  │                                    │   pneumothorax, edema, etc.            │
│  │                                    │                                        │
│  └─ (no positiveness)    ─────────►   └─ [encoded in entity_embedding]        │
│                                                                                 │
│  VISUAL FEATURES (512 dims)           VISUAL FEATURES (512 dims)               │
│  ├─ ResNet18 features    ─────────►   ├─ ConvNeXt-Base features  ★ UPGRADED   │
│  └─ ROI pooled                        └─ ROI Aligned                           │
│                                                                                 │
│  TOTAL: 530 dims                      TOTAL: 646 dims (+116 dims)              │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2.2 Why Expanded Format is Better for MIMIC-Ext-CXR-QBA

| Aspect | Original 530 | Expanded 646 | Advantage |
|--------|--------------|--------------|-----------|
| **Region Coverage** | 6 (one-hot) | 310 (learned 64-dim) | 50× more regions, semantic similarity |
| **Entity Coverage** | 4 categories | 237 entities (learned 64-dim) | 60× more specific findings |
| **Spatial Info** | 8 dims (discrete) | 6 dims (continuous) | Better bbox precision |
| **Semantic Learning** | Fixed one-hot | Learnable embeddings | Captures relationships |
| **Model Change** | None | `visual_embedding_dim=646` | Single config change |

### 4.3 ConvNeXt-Base Visual Feature Extraction

```python
# Visual backbone: ConvNeXt-Base (SELECTED over ResNet18)
# 
# Justification from methodology.md:
# - 91.5% accuracy in melanoma classification vs 87.2% for ResNet50
# - 94.3% sensitivity in lung nodule detection vs 89.7% for ResNet
# - Better for detecting subtle chest abnormalities

import timm
import torch.nn as nn

class ConvNeXtFeatureExtractor(nn.Module):
    """
    ConvNeXt-Base feature extractor for chest X-ray images.
    Outputs 512-dim features to match SSG-VQA format.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load ConvNeXt-Base (1024 output dims)
        self.backbone = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k',
            pretrained=pretrained,
            features_only=True,
            out_indices=[3]  # Last feature map
        )
        
        # Project to 512 dims for SSG-VQA compatibility
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # ROI Align for region-specific features
        self.roi_align = torchvision.ops.RoIAlign(
            output_size=(7, 7),
            spatial_scale=1/32,  # ConvNeXt downsamples by 32
            sampling_ratio=2
        )
        
    def forward(self, images, bboxes=None):
        """
        Args:
            images: (B, 3, H, W) chest X-ray images
            bboxes: (B, N, 4) bounding boxes per image, or None for global
        Returns:
            features: (B, N, 512) or (B, 512) if no bboxes
        """
        # Extract feature maps
        feature_maps = self.backbone(images)[0]  # (B, 1024, H/32, W/32)
        
        if bboxes is None:
            # Global features only
            return self.projection(feature_maps)
        
        # ROI features for each bounding box
        B, N, _ = bboxes.shape
        roi_features = []
        
        for b in range(B):
            # Format bboxes for RoI Align: (batch_idx, x1, y1, x2, y2)
            batch_bboxes = torch.cat([
                torch.full((N, 1), b, device=bboxes.device),
                bboxes[b]
            ], dim=1)
            
            roi_feats = self.roi_align(feature_maps[b:b+1], batch_bboxes)
            roi_feats = self.projection(roi_feats)  # (N, 512)
            roi_features.append(roi_feats)
            
        return torch.stack(roi_features)  # (B, N, 512)
```

### 4.4 Scene Graph Feature Encoding (134 dims - EXPANDED)

```python
class SceneGraphEncoderExpanded(nn.Module):
    """
    Encodes MIMIC-Ext-CXR-QBA scene graph into 134-dim features
    using learned embeddings to leverage the full richness of the dataset.
    
    Feature breakdown:
      - bbox_coords: 4 dims (normalized x1, y1, x2, y2)
      - bbox_area: 1 dim
      - aspect_ratio: 1 dim
      - region_embedding: 64 dims (learned from 310 regions)
      - entity_embedding: 64 dims (learned from 237 entities)
      Total: 134 dims
    """
    def __init__(self, 
                 num_regions=310,      # MIMIC-Ext-CXR-QBA has 310 regions
                 num_entities=237,     # MIMIC-Ext-CXR-QBA has 237 finding entities
                 embedding_dim=64):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Learnable region embeddings (310 anatomical regions → 64 dims)
        # Includes: lungs, left_lung, right_lung, heart, left_ventricle,
        #           mediastinum, aorta, trachea, spine, ribs, clavicle, etc.
        self.region_embedding = nn.Embedding(num_regions + 1, embedding_dim)  # +1 for unknown
        
        # Learnable entity embeddings (237 finding entities → 64 dims)
        # Includes: consolidation, cardiomegaly, pneumothorax, edema, 
        #           pleural_effusion, atelectasis, pneumonia, mass, nodule, etc.
        self.entity_embedding = nn.Embedding(num_entities + 1, embedding_dim)  # +1 for unknown
        
        # Region name to index mapping (loaded from dataset_info.json)
        self.region_to_idx = {}  # Will be populated from dataset
        self.entity_to_idx = {}  # Will be populated from dataset
        
        # Projection layers for multi-region/entity aggregation
        self.region_aggregator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        self.entity_aggregator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
    def load_vocab(self, dataset_info_path):
        """Load region and entity vocabularies from dataset_info.json"""
        import json
        with open(dataset_info_path) as f:
            info = json.load(f)
        
        # Build region vocabulary
        for idx, region in enumerate(info.get('regions', [])):
            self.region_to_idx[region.lower()] = idx
            
        # Build entity vocabulary  
        for idx, entity in enumerate(info.get('finding_entities', [])):
            self.entity_to_idx[entity.lower()] = idx
            
    def encode_observation(self, obs, image_width, image_height):
        """
        Encode single observation node into 134-dim vector.
        
        Args:
            obs: Observation dict from scene graph
            image_width, image_height: Image dimensions for normalization
            
        Returns:
            torch.Tensor of shape (134,)
        """
        # ═══════════════════════════════════════════════════════════════
        # PART 1: Bounding Box Features (6 dims)
        # ═══════════════════════════════════════════════════════════════
        
        if 'localization' in obs and obs['localization']:
            loc = obs['localization']
            # Handle nested structure: localization -> image_id -> bboxes
            if isinstance(loc, dict):
                # Get first image's bboxes
                for img_id, img_loc in loc.items():
                    if isinstance(img_loc, dict) and 'bboxes' in img_loc:
                        bbox = img_loc['bboxes'][0] if img_loc['bboxes'] else [0,0,image_width,image_height]
                        break
                else:
                    bbox = [0, 0, image_width, image_height]
            else:
                bbox = [0, 0, image_width, image_height]
        else:
            bbox = [0, 0, image_width, image_height]
            
        # Normalized bbox coordinates (4 dims)
        x1_norm = bbox[0] / image_width
        y1_norm = bbox[1] / image_height
        x2_norm = bbox[2] / image_width
        y2_norm = bbox[3] / image_height
        
        # Bbox area (1 dim)
        w = x2_norm - x1_norm
        h = y2_norm - y1_norm
        area = w * h
        
        # Aspect ratio (1 dim)
        aspect = w / max(h, 1e-6)
        
        bbox_features = torch.tensor([x1_norm, y1_norm, x2_norm, y2_norm, area, aspect])
        
        # ═══════════════════════════════════════════════════════════════
        # PART 2: Region Embedding (64 dims)
        # ═══════════════════════════════════════════════════════════════
        
        regions = obs.get('regions', [])
        if not regions:
            regions = obs.get('default_regions', ['unknown'])
            
        region_indices = []
        for r in regions:
            region_name = r.get('region', r) if isinstance(r, dict) else r
            region_name = region_name.lower()
            idx = self.region_to_idx.get(region_name, len(self.region_to_idx))  # unknown
            region_indices.append(idx)
            
        # Get embeddings and aggregate (mean pool + projection)
        region_indices = torch.tensor(region_indices)
        region_embs = self.region_embedding(region_indices)  # (num_regions, 64)
        region_emb = region_embs.mean(dim=0)  # (64,)
        region_emb = self.region_aggregator(region_emb)
        
        # ═══════════════════════════════════════════════════════════════
        # PART 3: Entity Embedding (64 dims)
        # ═══════════════════════════════════════════════════════════════
        
        entities = obs.get('obs_entities', [])
        if not entities:
            entities = ['unknown']
            
        entity_indices = []
        for e in entities:
            entity_name = e.lower()
            idx = self.entity_to_idx.get(entity_name, len(self.entity_to_idx))  # unknown
            entity_indices.append(idx)
            
        # Get embeddings and aggregate
        entity_indices = torch.tensor(entity_indices)
        entity_embs = self.entity_embedding(entity_indices)  # (num_entities, 64)
        entity_emb = entity_embs.mean(dim=0)  # (64,)
        entity_emb = self.entity_aggregator(entity_emb)
        
        # ═══════════════════════════════════════════════════════════════
        # COMBINE: Total 134 dims
        # ═══════════════════════════════════════════════════════════════
        
        combined = torch.cat([
            bbox_features,      # 6 dims
            region_emb,         # 64 dims
            entity_emb          # 64 dims
        ])  # Total: 134 dims
        
        return combined
        
    def encode_scene_graph(self, scene_graph, image_width, image_height):
        """
        Encode all observations in a scene graph.
        
        Returns:
            torch.Tensor of shape (num_observations, 134)
        """
        observations = scene_graph.get('observations', {})
        
        if not observations:
            # Return dummy observation for empty graphs
            return torch.zeros(1, 134)
            
        encoded = []
        for obs_id, obs in observations.items():
            enc = self.encode_observation(obs, image_width, image_height)
            encoded.append(enc)
            
        return torch.stack(encoded)  # (N, 134)
```

### 4.4.1 Embedding Coverage Comparison

```
MIMIC-Ext-CXR-QBA COVERAGE vs SSG-VQA COVERAGE

REGIONS (310 in MIMIC vs 6 in original):
┌─────────────────────────────────────────────────────────────────────┐
│ MIMIC-Ext-CXR-QBA Regions (310 total):                              │
│                                                                     │
│ LUNGS:        lungs, left_lung, right_lung, lung_apex,             │
│               left_lung_upper_lobe, left_lung_lower_lobe,           │
│               right_lung_upper_lobe, right_lung_middle_lobe,        │
│               right_lung_lower_lobe, lung_bases, ...                │
│                                                                     │
│ CARDIAC:      heart, left_ventricle, right_ventricle,              │
│               left_atrium, right_atrium, cardiac_silhouette,        │
│               pericardium, ...                                      │
│                                                                     │
│ MEDIASTINUM:  mediastinum, aorta, aortic_arch, descending_aorta,   │
│               trachea, carina, hilar_region, ...                    │
│                                                                     │
│ PLEURA:       pleura, left_pleura, right_pleura,                   │
│               costophrenic_angle, ...                               │
│                                                                     │
│ BONES:        ribs, spine, thoracic_spine, clavicle,               │
│               scapula, humerus, ...                                 │
│                                                                     │
│ OTHER:        diaphragm, abdomen, soft_tissue, ...                 │
└─────────────────────────────────────────────────────────────────────┘

ENTITIES (237 in MIMIC vs ~20 categories in original):
┌─────────────────────────────────────────────────────────────────────┐
│ MIMIC-Ext-CXR-QBA Finding Entities (237 total):                     │
│                                                                     │
│ PULMONARY:    consolidation, infiltrate, opacity, nodule, mass,    │
│               atelectasis, pneumonia, fibrosis, scarring,           │
│               emphysema, bronchiectasis, ...                        │
│                                                                     │
│ CARDIAC:      cardiomegaly, enlarged_heart, heart_failure,         │
│               pericardial_effusion, ...                             │
│                                                                     │
│ PLEURAL:      pleural_effusion, pneumothorax, pleural_thickening,  │
│               hemothorax, ...                                       │
│                                                                     │
│ VASCULAR:     pulmonary_edema, vascular_congestion,                │
│               pulmonary_hypertension, ...                           │
│                                                                     │
│ DEVICES:      endotracheal_tube, central_line, pacemaker,          │
│               chest_tube, nasogastric_tube, picc_line, ...          │
│                                                                     │
│ BONES:        fracture, degenerative_changes, scoliosis, ...       │
│                                                                     │
│ OTHER:        no_finding, normal, stable, ...                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5 Combined Feature Assembly (646 dims)

```python
def assemble_visual_features(image, scene_graph, convnext_extractor, sg_encoder):
    """
    Assemble 646-dim visual features for each observation in scene graph.
    
    Feature breakdown:
      - Scene graph features: 134 dims (6 bbox + 64 region_emb + 64 entity_emb)
      - Visual features: 512 dims (ConvNeXt-Base ROI features)
      - Total: 646 dims
    
    Returns:
        visual_features: (num_observations, 646)
    """
    observations = scene_graph.get('observations', {})
    image_h, image_w = image.shape[-2:]
    
    if not observations:
        # Return dummy features for empty scene graphs
        return torch.zeros(1, 646)
    
    # Collect bboxes for batch ROI extraction
    bboxes = []
    
    for obs_id, obs in observations.items():
        # Bounding box for ROI extraction
        if 'localization' in obs and obs['localization']:
            loc = obs['localization']
            # Handle nested structure
            for img_id, img_loc in loc.items():
                if isinstance(img_loc, dict) and 'bboxes' in img_loc:
                    bbox = img_loc['bboxes'][0] if img_loc['bboxes'] else [0,0,image_w,image_h]
                    break
            else:
                bbox = [0, 0, image_w, image_h]
        else:
            bbox = [0, 0, image_w, image_h]
        bboxes.append(bbox)
    
    # Extract scene graph features (134 dims each) - uses learned embeddings
    sg_features = sg_encoder.encode_scene_graph(scene_graph, image_w, image_h)  # (N, 134)
    
    # Extract ConvNeXt ROI features (512 dims each)
    bboxes_tensor = torch.tensor(bboxes).unsqueeze(0)  # (1, N, 4)
    visual_features = convnext_extractor(image.unsqueeze(0), bboxes_tensor)  # (1, N, 512)
    visual_features = visual_features.squeeze(0)  # (N, 512)
    
    # Concatenate: scene (134) + visual (512) = 646 dims
    combined = torch.cat([sg_features, visual_features], dim=1)  # (N, 646)
    
    return combined


class MIMICCXRVisualFeatures(nn.Module):
    """
    Complete visual feature module for MIMIC-CXR VQA.
    Produces 646-dimensional features per observation.
    """
    def __init__(self, 
                 num_regions=310,
                 num_entities=237,
                 embedding_dim=64,
                 visual_dim=512):
        super().__init__()
        
        self.convnext = ConvNeXtFeatureExtractor(pretrained=True)
        self.sg_encoder = SceneGraphEncoderExpanded(
            num_regions=num_regions,
            num_entities=num_entities,
            embedding_dim=embedding_dim
        )
        
        # Total output: 6 (bbox) + 64 (region) + 64 (entity) + 512 (visual) = 646
        self.output_dim = 6 + 2 * embedding_dim + visual_dim
        
    def forward(self, images, scene_graphs):
        """
        Args:
            images: (B, 3, H, W) batch of chest X-ray images
            scene_graphs: List of B scene graph dicts
            
        Returns:
            features: List of B tensors, each (N_i, 646) where N_i is num observations
        """
        batch_features = []
        
        for i, (img, sg) in enumerate(zip(images, scene_graphs)):
            features = assemble_visual_features(
                img, sg, self.convnext, self.sg_encoder
            )
            batch_features.append(features)
            
        return batch_features
```

### 4.6 Model Configuration Update

```python
# Required change to VisualBertConfig for expanded features
from models.VisualBert_ssgqa import VisualBertConfig

# Original SSG-VQA config
original_config = VisualBertConfig(
    vocab_size=vocab_size,
    visual_embedding_dim=530,  # Original: 18 scene + 512 visual
    num_hidden_layers=6,
    num_attention_heads=8,
    hidden_size=1024,
)

# NEW: Expanded config for MIMIC-CXR
expanded_config = VisualBertConfig(
    vocab_size=vocab_size,
    visual_embedding_dim=646,  # EXPANDED: 134 scene + 512 visual
    num_hidden_layers=6,
    num_attention_heads=8,
    hidden_size=1024,
)
```

### 4.3 Scene Graph Structure Mapping

| SSG-VQA (Surgical) | MIMIC-Ext-CXR-QBA (Radiology) | Notes |
|--------------------|-------------------------------|-------|
| `objects` | `observations` + `regions` | Findings map to observations |
| `bounding_boxes` | `localization.bboxes` | Per-image bbox lists |
| `relationships` (left, right, etc.) | `region_region_relations` | Spatial relations |
| `actions` (grasp, cut, etc.) | `probability`, `positiveness` | Disease presence |
| N/A | `obs_entities` | Finding type (237 classes) |
| N/A | `obs_categories` | Category (DISEASE, DEVICE, etc.) |

---

## 5. Answer Space Design

### 5.1 Original SSG-VQA Answer Classes (51)

```
Numbers: 0-10 (11 classes)
Boolean: True, False (2 classes)
Anatomies: gallbladder, liver, cystic_duct... (16 classes)
Instruments: grasper, hook, scissors... (9 classes)
Actions: grasp, retract, dissect... (8 classes)
Colors: red, blue, yellow... (6 classes)
```

`### 5.2 MIMIC-CXR Answer Space: Multi-Head Architecture (SELECTED)

Since the MIMIC-Ext-CXR-QBA dataset contains **ALL THREE answer formats**, we will implement a **unified multi-head architecture** that handles all formats simultaneously:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ANSWER ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  VisualBERT Encoder + SIM Module                                         │
│              │                                                           │
│              ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Pooled Output (1024 dims)                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│              │                                                           │
│              ├──────────────┬──────────────┬──────────────┐             │
│              ▼              ▼              ▼              ▼             │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌──────────┐ │
│  │  BINARY HEAD   │ │ CATEGORY HEAD  │ │  REGION HEAD   │ │TEXT HEAD │ │
│  │  (Yes/No)      │ │ (14 CheXpert)  │ │ (26 regions)   │ │(Generate)│ │
│  │  2 classes     │ │ 14 classes     │ │ 26 classes     │ │ Decoder  │ │
│  └────────────────┘ └────────────────┘ └────────────────┘ └──────────┘ │
│         │                  │                  │                │        │
│         ▼                  ▼                  ▼                ▼        │
│  is_abnormal        consolidation        left_lung      "There is..."  │
│  has_finding        cardiomegaly         right_lung                    │
│  is_normal          pneumothorax         cardiac                       │
│                     edema                mediastinum                    │
│                     ...                  ...                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Question Type → Head Routing

| Question Type | Primary Head | Secondary Head | Example Question |
|---------------|--------------|----------------|------------------|
| `is_abnormal` | Binary | - | "Is there any abnormality?" |
| `is_normal` | Binary | - | "Is the heart normal?" |
| `has_finding` | Binary | Category | "Is there pneumothorax?" |
| `describe_all` | Text | Category | "Describe the study" |
| `describe_region` | Text | Region | "Describe the lungs" |
| `describe_finding` | Text | Category | "Describe the cardiomegaly" |
| `where_is_finding` | Region | Category | "Where is the consolidation?" |
| `how_severe` | Category | - | "How severe is the edema?" |

### 5.4 Answer Head Specifications

```python
class MultiHeadAnswerModule(nn.Module):
    """
    Multi-head answer module supporting all MIMIC-Ext-CXR-QBA answer formats.
    """
    def __init__(self, hidden_size=1024):
        super().__init__()
        
        # Head 1: Binary Classification (Yes/No questions)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Yes, No
        )
        
        # Head 2: Category Classification (Finding types)
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 14)  # 14 CheXpert classes
        )
        
        # Head 3: Region Classification (Anatomical regions)
        self.region_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 26)  # 26 major anatomical regions
        )
        
        # Head 4: Severity Classification
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # none, mild, moderate, severe
        )
        
        # Head 5: Text Generation (for free-text answers) - OPTIONAL
        # Can use frozen/fine-tuned decoder or template filling
        
    def forward(self, pooled_output, question_type):
        outputs = {}
        
        # Route to appropriate head(s) based on question type
        if question_type in ['is_abnormal', 'is_normal', 'has_finding', 'has_device']:
            outputs['binary'] = self.binary_head(pooled_output)
            
        if question_type in ['describe_finding', 'has_finding', 'how_severe']:
            outputs['category'] = self.category_head(pooled_output)
            
        if question_type in ['where_is_finding', 'describe_region', 'is_abnormal_region']:
            outputs['region'] = self.region_head(pooled_output)
            
        if question_type in ['how_severe']:
            outputs['severity'] = self.severity_head(pooled_output)
            
        return outputs
```

### 5.5 Training Strategy for Multi-Head

```
Phase 1: Binary Head Only (Weeks 1-2)
  ├── Train only binary_head
  ├── Use ~15M binary questions (is_abnormal, is_normal, has_*)
  ├── Loss: CrossEntropyLoss
  └── Freeze other heads

Phase 2: Add Category + Region Heads (Weeks 2-3)
  ├── Unfreeze category_head and region_head
  ├── Add categorical questions to training
  ├── Multi-task loss: L = L_binary + λ₁*L_category + λ₂*L_region
  └── λ₁ = λ₂ = 0.5 initially

Phase 3: Full Multi-Head Training (Weeks 3-4)
  ├── All heads active
  ├── Question-type routing active
  ├── Dynamic loss weighting based on question type distribution
  └── Evaluate per-head metrics separately
```

### 5.6 Loss Function Design

```python
class MultiHeadLoss(nn.Module):
    def __init__(self, weights={'binary': 1.0, 'category': 0.5, 'region': 0.5, 'severity': 0.3}):
        super().__init__()
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)  # -1 for N/A
        
    def forward(self, outputs, targets, question_types):
        total_loss = 0.0
        
        for head_name, head_output in outputs.items():
            if head_name in targets and targets[head_name] is not None:
                head_loss = self.ce_loss(head_output, targets[head_name])
                total_loss += self.weights[head_name] * head_loss
                
        return total_loss
```

---

## 6. Training Pipeline Architecture

### 6.1 Data Flow Comparison

**Original SSG-VQA Pipeline:**
```
Video Frame → YOLOv5 Detection → ROI Features → Scene Graph (18 dims)
                                      ↓
                              Visual Features (512 dims)
                                      ↓
Question → BERT Tokenizer → Text Embeddings
                                      ↓
                    [Concatenate: Scene + Visual + Text]
                                      ↓
                    VisualBERT Encoder + SIM Module
                                      ↓
                    Classification Head (51 classes)
```

**Proposed MIMIC-CXR Pipeline:**
```
X-ray Image → ConvNeXt Backbone → Global Visual Features
                    ↓
Scene Graph JSON → Region/Entity Embeddings → Scene Features
                    ↓
                ROI Align (using scene graph bboxes)
                    ↓
Question → Bio+ClinicalBERT → Text Embeddings
                    ↓
            [Concatenate: Scene + Visual + Text]
                    ↓
            VisualBERT Encoder + SIM Module (preserved)
                    ↓
            Classification/Generation Head
```

### 6.2 Batch Construction Strategy

**Challenge**: One study may have multiple images and thousands of QA pairs.

**Solution: Hierarchical Batching**
```
Batch Level 1: Studies
  - Sample N studies per batch
  
Batch Level 2: Questions per Study  
  - Sample M questions per study
  - Balance across question types
  
Batch Level 3: Images per Study
  - Use primary frontal image
  - Or aggregate features across views
  
Effective Batch: N × M samples
  - Recommended: N=8 studies × M=8 questions = 64 samples
```

### 6.3 Memory Management

**Dataset Scale Considerations:**
- 377K images × ~1MB average = ~377 GB raw images
- 42M QA pairs × ~1KB = ~42 GB QA data
- Pre-extracted features recommended

**Pre-extraction Strategy:**
```
1. Extract ConvNeXt features for all images → HDF5 files
   - ~2048 dims × 377K images × 4 bytes = ~3 GB
   
2. Extract ROI features per scene graph bbox → HDF5 files
   - ~512 dims × 10 ROIs × 227K studies × 4 bytes = ~4.6 GB
   
3. Pre-compute scene graph embeddings → HDF5 files
   - ~64 dims × 20 nodes × 227K studies × 4 bytes = ~1.2 GB

Total pre-extracted: ~10 GB (vs 600+ GB raw)
```

---

## 7. Multi-Image Per Study Handling

### 7.1 The Challenge

Unlike surgical videos where each frame is processed independently, chest X-ray studies often have **multiple views**:

```
Study s50414267:
  ├── Image 1: PA view (frontal, posterior-anterior)
  ├── Image 2: Lateral view
  └── QA pairs reference the STUDY, not individual images
```

**Scene graph bounding boxes are provided per-image**, meaning different localizations for different views.

### 7.2 Proposed Solutions

**Approach A: Primary Image Selection (Simplest)**
```
For each study:
  1. Select the primary frontal image (PA preferred over AP)
  2. Use only that image's bounding boxes
  3. Discard lateral views
  
Pros: Simple, consistent with fine-tuning grade recommendations
Cons: Loses information from lateral views
```

**Approach B: Multi-View Fusion**
```
For each study:
  1. Extract features from all images
  2. Fuse features using attention or averaging
  3. Union of bounding boxes across views
  
Pros: Richer representation
Cons: Complex, variable-length inputs
```

**Approach C: View-Specific Training (Advanced)**
```
For each study:
  1. Create separate samples per view
  2. Add view_position as conditioning signal
  3. Train model to handle view-specific questions
  
Pros: Handles view-specific findings
Cons: Increases dataset size, complexity
```

### 7.3 Recommendation

**Start with Approach A** (primary frontal only):
- Matches MIMIC-Ext-CXR-QBA's fine-tuning grade (frontal only)
- Simplifies pipeline
- Can extend to multi-view later

---

## 8. Quality Filtering Strategy

### 8.1 Available Quality Grades

```
MIMIC-Ext-CXR-QBA Quality Levels:
  A++ : 1,368,001 pairs (highest quality)
  A+  : 1,115,789 pairs
  A   : 5,347,580 pairs
  ─────────────────────── Fine-tuning threshold
  B   : 24,812,681 pairs  
  ─────────────────────── Pre-training threshold
  C   : 700,747 pairs
  D   : 547,763 pairs
  Unrated: 8,280,266 pairs
```

### 8.2 Recommended Training Stages

**Stage 1: Pre-training**
```
Dataset: B_frontal (31.2M pairs)
Quality: B or better
Images: Frontal only
Purpose: Learn general visual-language alignment
Epochs: 3-5
```

**Stage 2: Fine-tuning**
```
Dataset: A_frontal (7.5M pairs)
Quality: A or better
Images: Frontal only
Purpose: Refine for high-quality QA
Epochs: 10-20
```

**Stage 3: High-precision tuning**
```
Dataset: App subset (1.3M pairs)
Quality: A++ only
Purpose: Maximum answer quality
Epochs: 5-10
```

---

## 9. Implementation Roadmap

### Phase 1: Data Preparation (Week 1-2)

```
□ Download MIMIC-CXR-JPG dataset (~570 GB)
□ Download MIMIC-Ext-CXR-QBA dataset (~26 GB)
□ Verify data integrity (SHA256 checksums)
□ Create unified patient-study-image index
□ Implement metadata loaders (CSV/Parquet)
□ Create train/val/test splits aligned with official splits
```

### Phase 2: Feature Extraction (Week 2-3)

```
□ Implement ConvNeXt visual feature extractor
□ Implement ROI feature extraction using scene graph bboxes
□ Create scene graph parser for MIMIC-Ext-CXR-QBA format
□ Implement scene graph embedding layer
□ Pre-extract and cache all features to HDF5
□ Validate feature dimensions match model expectations
```

### Phase 3: DataLoader Development (Week 3-4)

```
□ Create MIMIC_CXR_VQA_Dataset class
□ Implement study-level sampling
□ Implement question-level batching
□ Add quality filtering
□ Add view position filtering
□ Create collate function for variable-length inputs
□ Unit test with small subset
```

### Phase 4: Model Adaptation (Week 4-5)

```
□ Modify VisualBertConfig for new visual_embedding_dim
□ Update VisualBertEmbeddings for new scene graph format
□ Preserve Scene-embedded Interaction Module (SIM)
□ Replace tokenizer with Bio+ClinicalBERT
□ Create new classification head for answer space
□ Implement multi-level answer heads if needed
```

### Phase 5: Training Pipeline (Week 5-6)

```
□ Implement training loop with new dataloader
□ Add gradient accumulation for large batches
□ Implement learning rate scheduling
□ Add validation metrics (F1, accuracy per question type)
□ Implement checkpointing
□ Add tensorboard/wandb logging
```

### Phase 6: Evaluation (Week 6+)

```
□ Evaluate on MIMIC-Ext-CXR-QBA test set
□ Compute metrics per question type
□ Cross-dataset evaluation on VQA-RAD (zero-shot)
□ Cross-dataset evaluation on SLAKE-EN (zero-shot)
□ Ablation studies (with/without scene graphs)
□ Error analysis
```

### 9.1 Detailed Timeline and Milestones (6 Months)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PROJECT TIMELINE: 6 MONTHS                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MONTH 1: DATA PREPARATION & VALIDATION                                          │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  Week 1-2: Dataset Download & Setup                                              │
│  ├── □ Download MIMIC-CXR-JPG (~570 GB)                                         │
│  ├── □ Download MIMIC-Ext-CXR-QBA (~26 GB)                                      │
│  ├── □ Verify data integrity (SHA256 checksums)                                 │
│  └── □ Set up data infrastructure (HDF5, Parquet)                               │
│                                                                                  │
│  Week 3-4: Scene Graph Validation & Quality Assessment                           │
│  ├── □ Analyze scene graph quality statistics                                   │
│  ├── □ Identify bbox error patterns (5.6% needing correction)                   │
│  ├── □ Implement quality filtering pipelines                                    │
│  ├── □ Create data exploration notebooks                                        │
│  └── □ Document bias analysis findings                                          │
│                                                                                  │
│  DELIVERABLE: Validated dataset ready for training                              │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  MONTH 2: SSG-VQA-NET IMPLEMENTATION & ADAPTATION                                │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  Week 5-6: Core Model Implementation                                             │
│  ├── □ Implement ConvNeXt-Base visual backbone                                  │
│  ├── □ Implement expanded scene graph encoder (134 dims)                        │
│  ├── □ Integrate Bio+ClinicalBERT text encoder                                  │
│  └── □ Adapt Scene-embedded Interaction Module (SIM)                            │
│                                                                                  │
│  Week 7-8: Training Infrastructure                                               │
│  ├── □ Implement multi-head answer architecture                                 │
│  ├── □ Set up DeepSpeed ZeRO-2 distributed training                             │
│  ├── □ Implement mixed precision training                                       │
│  ├── □ Configure Weights & Biases logging                                       │
│  └── □ Set up HuggingFace Hub checkpointing                                     │
│                                                                                  │
│  DELIVERABLE: Training-ready model on single GPU                                │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  MONTH 3: TRAINING PHASE 1 (PRE-TRAINING)                                        │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  Week 9-12: Pre-training on B-grade data (31.2M pairs)                           │
│  ├── □ 3-5 epochs on full pre-training split                                    │
│  ├── □ Monitor loss curves and early stopping                                   │
│  ├── □ Track per-head convergence                                               │
│  ├── □ Validate on held-out subset                                              │
│  └── □ Save best pre-training checkpoint                                        │
│                                                                                  │
│  ESTIMATED TIME: ~2.5-3 days training                                           │
│  DELIVERABLE: Pre-trained model checkpoint                                      │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  MONTH 4: TRAINING PHASE 2 (FINE-TUNING) + CROSS-DATASET PREP                   │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  Week 13-14: Fine-tuning on A-grade data (7.5M pairs)                            │
│  ├── □ 10-20 epochs on fine-tuning split                                        │
│  ├── □ Learning rate decay and early stopping                                   │
│  └── □ Save best fine-tuning checkpoint                                         │
│                                                                                  │
│  Week 15-16: Cross-Dataset Preparation                                           │
│  ├── □ Prepare VQA-RAD evaluation pipeline                                      │
│  ├── □ Prepare SLAKE-EN evaluation pipeline                                     │
│  ├── □ Implement zero-shot inference mode (no scene graphs)                     │
│  └── □ Set up ablation experiment configurations                                │
│                                                                                  │
│  ESTIMATED TIME: ~2-2.5 days training                                           │
│  DELIVERABLE: Fine-tuned model + cross-dataset pipelines                        │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  MONTH 5: COMPREHENSIVE EVALUATION & ABLATIONS                                   │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  Week 17-18: In-Domain Evaluation                                                │
│  ├── □ Full MIMIC-Ext-CXR-QBA test set evaluation                               │
│  ├── □ Per-question-type stratified metrics                                     │
│  ├── □ Per-pathology (CheXpert) metrics                                         │
│  ├── □ Attention plausibility analysis                                          │
│  └── □ Generate attention visualization samples                                 │
│                                                                                  │
│  Week 19-20: Cross-Dataset & Ablation Studies                                    │
│  ├── □ Zero-shot evaluation on VQA-RAD                                          │
│  ├── □ Zero-shot evaluation on SLAKE-EN                                         │
│  ├── □ Run all 6 ablation conditions                                            │
│  ├── □ Statistical significance testing                                         │
│  └── □ Error analysis across failure categories                                 │
│                                                                                  │
│  DELIVERABLE: Complete evaluation results + ablation data                       │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  MONTH 6: ANALYSIS, DOCUMENTATION & FINALIZATION                                 │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  Week 21-22: Statistical Analysis & Visualization                                │
│  ├── □ Compile all metrics with confidence intervals                            │
│  ├── □ Generate publication-ready figures                                       │
│  ├── □ Compute effect sizes (Cohen's d)                                         │
│  └── □ Apply Bonferroni corrections                                             │
│                                                                                  │
│  Week 23-24: Documentation & Model Release                                       │
│  ├── □ Write comprehensive model card                                           │
│  ├── □ Document reproducibility instructions                                    │
│  ├── □ Prepare HuggingFace Hub model release                                    │
│  └── □ Final code cleanup and repository organization                           │
│                                                                                  │
│  DELIVERABLE: Publication-ready results + released model                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Milestone Summary

| Month | Milestone | Key Deliverable | Success Criteria |
|-------|-----------|-----------------|------------------|
| **1** | Data Ready | Validated dataset | Quality metrics documented |
| **2** | Model Ready | Training infrastructure | Single-GPU training works |
| **3** | Pre-trained | Pre-training complete | Val loss < baseline |
| **4** | Fine-tuned | Fine-tuning complete | Test accuracy > 60% |
| **5** | Evaluated | All experiments done | Statistical tests passed |
| **6** | Published | Model released | Documentation complete |

---

## 10. Critical Decision Points

### 10.1 Final Design Decisions

| Decision | Options | **SELECTED** | Rationale |
|----------|---------|--------------|-----------|
| **Visual backbone** | ResNet18 vs ConvNeXt-Base | ✅ **ConvNeXt-Base** | 4-7% accuracy improvement for medical imaging |
| **Feature dimensions** | 530 (original) vs 646 (expanded) | ✅ **646 dims (EXPANDED)** | Leverage full richness of 310 regions, 237 entities |
| **Scene graph dims** | 18 (original) vs 134 (expanded) | ✅ **134 dims (EXPANDED)** | Learned embeddings for regions/entities |
| **Answer format** | Single-head vs Multi-head | ✅ **Multi-head** | Dataset has binary, categorical, and text answers |
| **Multi-image handling** | Primary only vs fusion | ✅ **Primary frontal only** | Matches fine-tuning grade, simpler pipeline |
| **Quality threshold** | A/A+/A++ | ✅ **B (pretrain), A (finetune)** | 31M→7.5M sample progression |
| **Tokenizer** | Original vs Bio+ClinicalBERT | ✅ **Bio+ClinicalBERT** | 2.8-4.5% accuracy gain on medical text |
| **Training precision** | FP32 vs Mixed (FP16/BF16) | ✅ **Mixed Precision (FP16)** | 1.5-2× speedup |
| **Optimizer** | Standard vs DeepSpeed | ✅ **DeepSpeed ZeRO-2** | 30-40% memory savings |

### 10.2 Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Memory overflow** | Training crash | Pre-extract features, gradient accumulation |
| **Class imbalance** | Poor minority class performance | Stratified sampling, weighted loss |
| **Scene graph noise** | Incorrect supervision | Quality filtering (A+ grade) |
| **Domain gap** | Poor transfer | Bio+ClinicalBERT, domain pretraining |
| **Compute time** | Slow iteration | Feature caching, distributed training |

---

## 11. Training Optimizations (4× NVIDIA L4 Configuration)

### 11.1 Speed Optimizations

| Optimization | Speedup | Implementation |
|--------------|---------|----------------|
| **Mixed Precision (FP16/BF16)** | 1.5-2× | `torch.cuda.amp` / `accelerate` |
| **Gradient Checkpointing** | Enables 2× batch | `model.gradient_checkpointing_enable()` |
| **DeepSpeed ZeRO Stage 2** | 30-40% memory savings | Optimizer state partitioning |
| **Pre-extracted Features** | 10× data loading | HDF5 caching (already planned ✓) |
| **Multi-GPU DataParallel** | ~3.5× scaling | `DistributedDataParallel` on 4 L4s |

### 11.2 Recommended Training Configuration

```python
# Optimal settings for 4× NVIDIA L4 GPUs (96GB total VRAM)
training_config = {
    # Batch Configuration
    "batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 16 * 4 * 4,  # = 256
    
    # Mixed Precision
    "fp16": True,  # or bf16=True for better numerical stability
    "fp16_opt_level": "O2",  # Aggressive mixed precision
    
    # Learning Rate
    "learning_rate": 5e-5,
    "lr_scheduler": "cosine_with_warmup",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    
    # Data Loading (utilize 48 vCPUs)
    "dataloader_num_workers": 12,  # 3 per GPU
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 4,
    
    # Gradient Checkpointing
    "gradient_checkpointing": True,
    
    # DeepSpeed Config
    "deepspeed": {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
        },
        "fp16": {"enabled": True},
    },
    
    # Checkpointing
    "save_steps": 5000,
    "save_total_limit": 3,
    "logging_steps": 100,
}
```

### 11.3 Memory Budget Analysis (4× L4, 96GB total)

```
Per-GPU Memory Breakdown (24GB each):
├── Model Parameters (VisualBERT + SIM):     ~1.5 GB
├── Gradients:                                ~1.5 GB
├── Optimizer States (AdamW):                 ~3.0 GB
├── Activations (with checkpointing):         ~4.0 GB
├── Batch Data (16 samples × 530 dims):       ~0.5 GB
├── Safety Margin:                            ~3.5 GB
└── Available for larger batches:            ~10.0 GB

With DeepSpeed ZeRO-2:
├── Optimizer states partitioned across GPUs
├── Effective per-GPU memory freed:          ~2.0 GB
└── Can increase batch_size_per_gpu to: 24-32
```

### 11.4 Expected Training Times (with optimizations)

| Phase | Without Optimization | With Optimization | Speedup |
|-------|---------------------|-------------------|---------|
| **Pre-training (3 epochs)** | 4.5 days | 2.5-3 days | 1.5-1.8× |
| **Fine-tuning (10 epochs)** | 3.5 days | 2-2.5 days | 1.4-1.75× |
| **Total** | ~10 days | ~5-6 days | **~1.7×** |

---

## 12. Logging, Checkpointing & Experiment Tracking

### 12.1 Overview: Hugging Face + Weights & Biases Integration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT TRACKING INFRASTRUCTURE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║                         WEIGHTS & BIASES (wandb)                          ║  │
│  ║  ─────────────────────────────────────────────────────────────────────    ║  │
│  ║  • Real-time training metrics dashboard                                   ║  │
│  ║  • Loss curves (VQA + CheXpert auxiliary)                                ║  │
│  ║  • Per-head accuracy tracking                                            ║  │
│  ║  • Learning rate schedules                                               ║  │
│  ║  • GPU utilization & memory                                              ║  │
│  ║  • Gradient norms & histograms                                           ║  │
│  ║  • Hyperparameter logging                                                ║  │
│  ║  • Model artifact versioning                                             ║  │
│  ║  • Attention visualization samples                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║                         HUGGING FACE HUB                                  ║  │
│  ║  ─────────────────────────────────────────────────────────────────────    ║  │
│  ║  • Model checkpoint storage & versioning                                  ║  │
│  ║  • Automatic push on save_steps                                          ║  │
│  ║  • Tokenizer & config preservation                                        ║  │
│  ║  • Model cards with training metadata                                    ║  │
│  ║  • Easy loading for inference                                            ║  │
│  ║  • Collaboration & sharing                                               ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Weights & Biases Configuration

```python
import wandb
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# WANDB INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_wandb_tracking(config):
    """
    Initialize Weights & Biases experiment tracking.
    """
    run = wandb.init(
        # Project & Run Identification
        project="mimic-cxr-vqa",
        name=f"ssg-vqa-{config.get('experiment_name', 'default')}-{datetime.now().strftime('%Y%m%d_%H%M')}",
        group=config.get('experiment_group', 'main'),  # e.g., "ablation", "hyperparameter-sweep"
        tags=[
            "SSG-VQA-Net",
            "MIMIC-CXR",
            f"backbone-{config.get('visual_backbone', 'convnext')}",
            f"text-encoder-{config.get('text_encoder', 'bioclinicalbert')}",
            f"feature-dims-{config.get('feature_dim', 646)}",
        ],
        
        # Configuration Tracking
        config={
            # Model Architecture (from methodology.md)
            "model": {
                "name": "SSG-VQA-Net-MIMIC",
                "visual_backbone": config.get("visual_backbone", "ConvNeXt-Base"),
                "text_encoder": config.get("text_encoder", "Bio+ClinicalBERT"),
                "visual_embedding_dim": config.get("feature_dim", 646),
                "scene_graph_dim": 134,  # Expanded: 6 bbox + 64 region + 64 entity
                "visual_feature_dim": 512,
                "hidden_size": 1024,
                "num_hidden_layers": 6,
                "num_attention_heads": 8,
                "sim_layers": 2,  # Scene-Embedded Interaction Module layers
                "answer_heads": ["binary", "category", "region", "severity"],
            },
            
            # Training Configuration
            "training": {
                "batch_size_per_gpu": config.get("batch_size", 16),
                "gradient_accumulation_steps": config.get("grad_accum", 4),
                "effective_batch_size": config.get("batch_size", 16) * 4 * config.get("grad_accum", 4),
                "learning_rate": config.get("lr", 5e-5),
                "lr_scheduler": config.get("scheduler", "cosine_with_warmup"),
                "warmup_ratio": config.get("warmup", 0.1),
                "weight_decay": config.get("weight_decay", 0.01),
                "max_epochs": config.get("epochs", 20),
                "fp16": config.get("fp16", True),
                "gradient_checkpointing": config.get("grad_checkpoint", True),
            },
            
            # DeepSpeed Configuration
            "deepspeed": {
                "enabled": True,
                "zero_stage": 2,
                "offload_optimizer": True,
            },
            
            # Dataset Configuration
            "dataset": {
                "name": "MIMIC-Ext-CXR-QBA",
                "train_pairs": "31.2M (B grade) → 7.5M (A grade)",
                "quality_filter": config.get("quality_grade", "A"),
                "view_filter": "frontal_only",
                "chexpert_auxiliary": True,
            },
            
            # Hardware
            "hardware": {
                "gpus": 4,
                "gpu_type": "NVIDIA L4",
                "total_vram_gb": 96,
                "num_workers": 12,
            },
        },
        
        # Sync & Resume Settings
        resume="allow",  # Resume from previous run if crashed
        save_code=True,  # Save a copy of the training code
    )
    
    # Define custom metrics for better dashboard organization
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("train/vqa_loss", summary="min")
    wandb.define_metric("train/chexpert_loss", summary="min")
    wandb.define_metric("val/accuracy", summary="max")
    wandb.define_metric("val/binary_accuracy", summary="max")
    wandb.define_metric("val/category_f1", summary="max")
    wandb.define_metric("val/region_accuracy", summary="max")
    wandb.define_metric("val/severity_accuracy", summary="max")
    
    return run


# ═══════════════════════════════════════════════════════════════════════════════
# WANDB LOGGING CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

class WandbLoggingCallback:
    """
    Comprehensive logging callback for training metrics.
    """
    
    def __init__(self, log_interval=100, eval_interval=1000):
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.step = 0
        
    def on_train_step(self, metrics):
        """Log training step metrics."""
        self.step += 1
        
        if self.step % self.log_interval == 0:
            wandb.log({
                # Loss metrics
                "train/loss": metrics.get("loss"),
                "train/vqa_loss": metrics.get("vqa_loss"),
                "train/chexpert_loss": metrics.get("chexpert_loss"),
                
                # Per-head losses
                "train/binary_loss": metrics.get("binary_loss"),
                "train/category_loss": metrics.get("category_loss"),
                "train/region_loss": metrics.get("region_loss"),
                "train/severity_loss": metrics.get("severity_loss"),
                
                # Learning rate
                "train/learning_rate": metrics.get("lr"),
                
                # Gradient norms
                "train/grad_norm": metrics.get("grad_norm"),
                
                # Throughput
                "train/samples_per_second": metrics.get("throughput"),
                "train/gpu_memory_mb": metrics.get("gpu_memory"),
                
                # Step counter
                "global_step": self.step,
            }, step=self.step)
            
    def on_validation(self, metrics, epoch):
        """Log validation metrics."""
        wandb.log({
            # Overall metrics
            "val/accuracy": metrics.get("accuracy"),
            "val/loss": metrics.get("loss"),
            
            # Per-head metrics
            "val/binary_accuracy": metrics.get("binary_accuracy"),
            "val/binary_f1": metrics.get("binary_f1"),
            "val/category_accuracy": metrics.get("category_accuracy"),
            "val/category_f1": metrics.get("category_f1"),
            "val/region_accuracy": metrics.get("region_accuracy"),
            "val/region_iou": metrics.get("region_iou"),
            "val/severity_accuracy": metrics.get("severity_accuracy"),
            
            # CheXpert auxiliary metrics
            "val/chexpert_auroc": metrics.get("chexpert_auroc"),
            "val/chexpert_f1": metrics.get("chexpert_f1"),
            
            # Per-question-type metrics
            "val/is_abnormal_acc": metrics.get("is_abnormal_accuracy"),
            "val/has_finding_acc": metrics.get("has_finding_accuracy"),
            "val/where_is_acc": metrics.get("where_is_accuracy"),
            "val/how_severe_acc": metrics.get("how_severe_accuracy"),
            
            "epoch": epoch,
        }, step=self.step)
        
    def on_sample_predictions(self, samples, epoch):
        """Log sample predictions for qualitative analysis."""
        # Create a wandb Table for sample predictions
        columns = ["image", "question", "ground_truth", "prediction", "correct", "confidence"]
        table = wandb.Table(columns=columns)
        
        for sample in samples[:20]:  # Log 20 samples
            table.add_data(
                wandb.Image(sample["image"]),
                sample["question"],
                sample["ground_truth"],
                sample["prediction"],
                sample["correct"],
                sample["confidence"],
            )
            
        wandb.log({"val/sample_predictions": table}, step=self.step)
        
    def on_attention_maps(self, attention_data, epoch):
        """Log attention visualizations."""
        images = []
        for data in attention_data[:5]:  # Log 5 attention maps
            # Create attention heatmap overlay
            fig = create_attention_heatmap(
                image=data["image"],
                attention_weights=data["attention"],
                question=data["question"],
            )
            images.append(wandb.Image(fig, caption=data["question"][:50]))
            
        wandb.log({"val/attention_maps": images}, step=self.step)


# ═══════════════════════════════════════════════════════════════════════════════
# WANDB METRICS DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

WANDB_METRICS_CONFIG = {
    # Training metrics to track
    "train_metrics": [
        "loss",                    # Total loss
        "vqa_loss",               # VQA task loss
        "chexpert_loss",          # Auxiliary CheXpert loss
        "binary_loss",            # Binary head loss
        "category_loss",          # Category head loss
        "region_loss",            # Region head loss
        "severity_loss",          # Severity head loss
        "learning_rate",          # Current LR
        "grad_norm",              # Gradient norm
        "samples_per_second",     # Throughput
        "gpu_memory_mb",          # Memory usage
    ],
    
    # Validation metrics to track
    "val_metrics": [
        "accuracy",               # Overall accuracy
        "loss",                   # Validation loss
        
        # Per-head metrics
        "binary_accuracy",
        "binary_f1",
        "binary_precision",
        "binary_recall",
        
        "category_accuracy",
        "category_f1",
        "category_precision",
        "category_recall",
        
        "region_accuracy",
        "region_iou",
        "region_pointing_accuracy",
        
        "severity_accuracy",
        "severity_mae",
        
        # CheXpert auxiliary
        "chexpert_auroc",
        "chexpert_f1",
        "chexpert_per_class_auroc",
        
        # Per question type (from methodology)
        "is_abnormal_accuracy",
        "is_normal_accuracy",
        "has_finding_accuracy",
        "describe_finding_accuracy",
        "where_is_finding_accuracy",
        "how_severe_accuracy",
        "describe_region_accuracy",
    ],
    
    # Metrics for wandb alerts
    "alert_thresholds": {
        "loss_spike": 2.0,        # Alert if loss > 2× baseline
        "grad_norm_spike": 10.0,  # Alert if grad_norm > 10
        "val_accuracy_drop": 0.05, # Alert if accuracy drops > 5%
    },
}
```

### 12.3 Hugging Face Model Checkpointing

```python
from transformers import TrainingArguments, Trainer
from huggingface_hub import HfApi, create_repo
import os

# ═══════════════════════════════════════════════════════════════════════════════
# HUGGING FACE CHECKPOINTING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_training_arguments(config):
    """
    Create Hugging Face TrainingArguments with checkpoint configuration.
    """
    return TrainingArguments(
        # Output & Checkpointing
        output_dir=config.get("output_dir", "./checkpoints/mimic-cxr-vqa"),
        overwrite_output_dir=False,
        
        # Checkpoint Strategy
        save_strategy="steps",
        save_steps=5000,                  # Save every 5000 steps
        save_total_limit=5,               # Keep only last 5 checkpoints
        save_safetensors=True,            # Use safetensors format (faster, safer)
        
        # Best Model Selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        
        # Hugging Face Hub Integration
        push_to_hub=True,
        hub_model_id=config.get("hub_model_id", "your-username/mimic-cxr-vqa-ssg"),
        hub_strategy="checkpoint",        # Push checkpoints to hub
        hub_token=os.environ.get("HF_TOKEN"),
        hub_private_repo=True,            # Keep private (MIMIC requires DUA)
        
        # Training Configuration
        num_train_epochs=config.get("epochs", 20),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("eval_batch_size", 32),
        gradient_accumulation_steps=config.get("grad_accum", 4),
        
        # Optimization
        learning_rate=config.get("lr", 5e-5),
        lr_scheduler_type=config.get("scheduler", "cosine"),
        warmup_ratio=config.get("warmup", 0.1),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=1.0,
        
        # Mixed Precision
        fp16=config.get("fp16", True),
        fp16_full_eval=True,
        
        # Gradient Checkpointing
        gradient_checkpointing=config.get("grad_checkpoint", True),
        
        # Logging
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=2500,
        eval_accumulation_steps=10,
        
        # Wandb Integration
        report_to=["wandb"],
        run_name=config.get("experiment_name", "ssg-vqa-mimic"),
        
        # DataLoader
        dataloader_num_workers=config.get("num_workers", 12),
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        
        # Reproducibility
        seed=42,
        data_seed=42,
        
        # DeepSpeed
        deepspeed=config.get("deepspeed_config", "configs/deepspeed_config.json"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Checkpoint directory structure:

checkpoints/mimic-cxr-vqa/
├── checkpoint-5000/
│   ├── config.json                    # Model configuration
│   ├── model.safetensors              # Model weights (safetensors format)
│   ├── optimizer.pt                   # Optimizer state
│   ├── scheduler.pt                   # LR scheduler state
│   ├── trainer_state.json             # Training state (step, epoch, best_metric)
│   ├── training_args.bin              # Training arguments
│   ├── rng_state.pth                  # Random state for reproducibility
│   └── special_tokens_map.json        # Tokenizer special tokens
│
├── checkpoint-10000/
│   └── ... (same structure)
│
├── checkpoint-15000/
│   └── ... (same structure)
│
├── runs/                              # TensorBoard logs
│   └── events.out.tfevents.*
│
└── best_model/                        # Best checkpoint (copied)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── vocab.txt
    └── README.md                      # Auto-generated model card
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CHECKPOINT CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

from transformers import TrainerCallback
import json
import shutil

class CustomCheckpointCallback(TrainerCallback):
    """
    Custom callback for enhanced checkpointing with metadata.
    """
    
    def __init__(self, save_best_only=True, monitor_metric="eval_accuracy"):
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.best_metric = None
        
    def on_save(self, args, state, control, **kwargs):
        """Enhanced checkpoint saving with metadata."""
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        
        # Save additional metadata
        metadata = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            
            # Training progress
            "total_steps": state.max_steps,
            "progress_percent": (state.global_step / state.max_steps) * 100 if state.max_steps else 0,
            
            # Model info
            "model_name": "SSG-VQA-Net-MIMIC",
            "architecture": {
                "visual_backbone": "ConvNeXt-Base",
                "text_encoder": "Bio+ClinicalBERT",
                "feature_dim": 646,
                "sim_layers": 2,
            },
            
            # Dataset info
            "dataset": "MIMIC-Ext-CXR-QBA",
            "training_samples_seen": state.global_step * args.train_batch_size * args.gradient_accumulation_steps,
            
            # Current metrics
            "metrics": {
                "train_loss": state.log_history[-1].get("loss") if state.log_history else None,
                "eval_accuracy": state.log_history[-1].get("eval_accuracy") if state.log_history else None,
            },
        }
        
        with open(f"{checkpoint_dir}/training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Log to wandb
        if wandb.run:
            wandb.log({
                "checkpoint/step": state.global_step,
                "checkpoint/epoch": state.epoch,
                "checkpoint/best_metric": state.best_metric,
            })
            
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Track best model and log to wandb."""
        current_metric = metrics.get(self.monitor_metric)
        
        if current_metric is not None:
            if self.best_metric is None or current_metric > self.best_metric:
                self.best_metric = current_metric
                
                if wandb.run:
                    wandb.run.summary["best_accuracy"] = self.best_metric
                    wandb.run.summary["best_step"] = state.global_step


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CARD GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_model_card(config, metrics):
    """
    Generate a Hugging Face model card with training details.
    """
    model_card = f"""
---
language: en
license: mit
library_name: transformers
tags:
  - medical-vqa
  - chest-x-ray
  - scene-graph
  - visual-question-answering
  - mimic-cxr
datasets:
  - mimic-cxr-jpg
  - mimic-ext-cxr-qba
metrics:
  - accuracy
  - f1
  - auroc
---

# SSG-VQA-Net for MIMIC-CXR Visual Question Answering

## Model Description

This model adapts the SSG-VQA-Net architecture for chest X-ray visual question answering
using the MIMIC-CXR-JPG images and MIMIC-Ext-CXR-QBA question-answer pairs.

### Architecture

- **Visual Backbone**: ConvNeXt-Base (pre-trained on ImageNet-22k)
- **Text Encoder**: Bio+ClinicalBERT (medical domain)
- **Scene Graph**: Expanded 134-dim embeddings (310 regions, 237 entities)
- **Fusion**: Scene-Embedded Interaction Module (SIM)
- **Answer Heads**: Multi-head (Binary, Category, Region, Severity)

### Key Innovations

1. **ConvNeXt-Base** visual encoder (+4-7% accuracy over ResNet18)
2. **Bio+ClinicalBERT** text encoder (+2.8-4.5% on medical text)
3. **Expanded Scene Graph** with learned embeddings
4. **Multi-task learning** with CheXpert auxiliary supervision

## Training Details

- **Dataset**: MIMIC-Ext-CXR-QBA ({config.get('train_samples', '31.2M')} pairs)
- **Hardware**: 4× NVIDIA L4 GPUs (96GB total)
- **Training Time**: ~{config.get('training_days', '5-6')} days
- **Mixed Precision**: FP16
- **Optimizer**: AdamW with DeepSpeed ZeRO-2

## Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | {metrics.get('accuracy', 'N/A'):.1%} |
| Binary Accuracy | {metrics.get('binary_accuracy', 'N/A'):.1%} |
| Category F1 | {metrics.get('category_f1', 'N/A'):.1%} |
| Region IoU | {metrics.get('region_iou', 'N/A'):.1%} |
| CheXpert AUROC | {metrics.get('chexpert_auroc', 'N/A'):.3f} |

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("your-username/mimic-cxr-vqa-ssg")
tokenizer = AutoTokenizer.from_pretrained("your-username/mimic-cxr-vqa-ssg")

# Note: Requires MIMIC-CXR data access (PhysioNet credentialed)
```

## Citation

```bibtex
@article{{your-paper,
  title={{Scene Graph-Enhanced VQA for Chest X-Ray Analysis}},
  author={{Your Name}},
  year={{2026}}
}}
```

## Acknowledgments

- SSG-VQA-Net original paper
- MIMIC-CXR database (PhysioNet)
- MIMIC-Ext-CXR-QBA dataset
"""
    return model_card
```

### 12.4 Complete Training Script with Logging

```python
#!/usr/bin/env python3
"""
Complete training script with Hugging Face + Wandb integration.
"""

import os
import torch
import wandb
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback

# Local imports
from models.mimic_vqa_model import MIMICCXRVQAModel
from data.dataloader import MIMICCXRVQADataset
from training.loss import MultiHeadLoss
from training.metrics import compute_metrics

def main():
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    
    config = {
        # Experiment
        "experiment_name": "ssg-vqa-mimic-v1",
        "experiment_group": "main",
        
        # Model
        "visual_backbone": "ConvNeXt-Base",
        "text_encoder": "Bio+ClinicalBERT",
        "feature_dim": 646,
        
        # Training
        "batch_size": 16,
        "grad_accum": 4,
        "lr": 5e-5,
        "epochs": 20,
        "warmup": 0.1,
        "weight_decay": 0.01,
        
        # Optimization
        "fp16": True,
        "grad_checkpoint": True,
        
        # Data
        "quality_grade": "A",
        "num_workers": 12,
        
        # Checkpointing
        "output_dir": "./checkpoints/mimic-cxr-vqa",
        "hub_model_id": "your-username/mimic-cxr-vqa-ssg",
        
        # DeepSpeed
        "deepspeed_config": "configs/deepspeed_config.json",
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # INITIALIZE WANDB
    # ═══════════════════════════════════════════════════════════════════════
    
    wandb_run = init_wandb_tracking(config)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ═══════════════════════════════════════════════════════════════════════
    
    print("Loading datasets...")
    train_dataset = MIMICCXRVQADataset(
        split="train",
        quality_grade=config["quality_grade"],
        use_chexpert_labels=True,
    )
    eval_dataset = MIMICCXRVQADataset(
        split="validation",
        quality_grade=config["quality_grade"],
        use_chexpert_labels=True,
    )
    
    # Log dataset info to wandb
    wandb.config.update({
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    })
    
    # ═══════════════════════════════════════════════════════════════════════
    # INITIALIZE MODEL
    # ═══════════════════════════════════════════════════════════════════════
    
    print("Initializing model...")
    model = MIMICCXRVQAModel(
        visual_backbone=config["visual_backbone"],
        text_encoder=config["text_encoder"],
        feature_dim=config["feature_dim"],
        use_chexpert_head=True,
    )
    
    # Enable gradient checkpointing
    if config["grad_checkpoint"]:
        model.gradient_checkpointing_enable()
    
    # Log model architecture to wandb
    wandb.watch(model, log="all", log_freq=1000)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING ARGUMENTS
    # ═══════════════════════════════════════════════════════════════════════
    
    training_args = get_training_arguments(config)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════
    
    callbacks = [
        CustomCheckpointCallback(save_best_only=True, monitor_metric="eval_accuracy"),
        WandbCallback(),  # Built-in HF wandb callback
    ]
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINER
    # ═══════════════════════════════════════════════════════════════════════
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════════
    
    print("Starting training...")
    train_result = trainer.train()
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE FINAL MODEL
    # ═══════════════════════════════════════════════════════════════════════
    
    print("Saving final model...")
    trainer.save_model(f"{config['output_dir']}/final")
    
    # Push to Hugging Face Hub
    if config.get("hub_model_id"):
        trainer.push_to_hub(
            commit_message=f"Training complete - Accuracy: {train_result.metrics.get('eval_accuracy', 'N/A')}"
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # LOG FINAL METRICS
    # ═══════════════════════════════════════════════════════════════════════
    
    # Final evaluation
    eval_results = trainer.evaluate()
    
    # Log final metrics to wandb
    wandb.log({
        "final/train_loss": train_result.training_loss,
        "final/eval_accuracy": eval_results.get("eval_accuracy"),
        "final/eval_loss": eval_results.get("eval_loss"),
        "final/runtime_hours": train_result.metrics.get("train_runtime", 0) / 3600,
    })
    
    # Update wandb summary
    wandb.run.summary.update({
        "final_accuracy": eval_results.get("eval_accuracy"),
        "final_loss": eval_results.get("eval_loss"),
        "total_steps": train_result.global_step,
        "training_hours": train_result.metrics.get("train_runtime", 0) / 3600,
    })
    
    # Generate and save model card
    model_card = generate_model_card(config, eval_results)
    with open(f"{config['output_dir']}/final/README.md", "w") as f:
        f.write(model_card)
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINISH
    # ═══════════════════════════════════════════════════════════════════════
    
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
```

### 12.5 DeepSpeed Configuration File

```json
// configs/deepspeed_config.json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": "auto"
    }
  },
  
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
  
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true,
    "cpu_checkpointing": false
  },
  
  "wall_clock_breakdown": true,
  "tensorboard": {
    "enabled": true,
    "output_path": "./logs/tensorboard",
    "job_name": "mimic_cxr_vqa"
  },
  
  "wandb": {
    "enabled": true,
    "project": "mimic-cxr-vqa",
    "group": "deepspeed"
  }
}
```

### 12.6 Logging Summary Table

| What | Where | Format | Frequency |
|------|-------|--------|-----------|
| **Training Loss** | wandb | Scalar | Every 100 steps |
| **Per-Head Losses** | wandb | Scalars | Every 100 steps |
| **Learning Rate** | wandb | Scalar | Every 100 steps |
| **Gradient Norms** | wandb | Scalar | Every 100 steps |
| **GPU Memory** | wandb | Scalar | Every 100 steps |
| **Validation Metrics** | wandb + HF | Dict | Every 2500 steps |
| **Sample Predictions** | wandb | Table | Every epoch |
| **Attention Maps** | wandb | Images | Every epoch |
| **Model Checkpoints** | HF Hub | Safetensors | Every 5000 steps |
| **Training Metadata** | Local JSON | JSON | Every checkpoint |
| **Model Card** | HF Hub | Markdown | Final save |
| **Config** | wandb + HF | YAML/JSON | Run start |


---

## 13. Comprehensive Evaluation Framework

### 13.1 Answer Accuracy Metrics

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ANSWER ACCURACY METRICS SUITE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  METRIC 1: EXACT MATCH (EM)                                                      │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Percentage of answers matching ground truth exactly                          │
│  • Best for: Binary (Yes/No) and categorical questions                          │
│  • Formula: EM = (# exact matches) / (# total predictions) × 100%               │
│                                                                                  │
│  METRIC 2: F1 SCORE (Token-Level)                                                │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Token-level overlap between predicted and ground-truth answers               │
│  • Balances precision and recall at lexical level                               │
│  • Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)                │
│                                                                                  │
│  METRIC 3: BLEU-4                                                                │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • 4-gram precision for fluency assessment in free-text answers                 │
│  • Includes brevity penalty for overly short responses                          │
│  • Best for: Descriptive answers (describe_finding, describe_region)            │
│                                                                                  │
│  METRIC 4: ROUGE-L                                                               │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Longest common subsequence-based recall                                      │
│  • Captures sentence-level structure beyond n-gram matching                     │
│  • Formula: ROUGE-L = LCS(prediction, reference) / length(reference)            │
│                                                                                  │
│  METRIC 5: BERTScore                                                             │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Semantic similarity using contextual embeddings                              │
│  • Robust to paraphrasing and lexical variation                                 │
│  • Uses Bio+ClinicalBERT embeddings for medical domain                          │
│                                                                                  │
│  METRIC 6: CLINICAL TERM F1                                                      │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • F1 computed specifically over medical terminology                            │
│  • Extracted using MetaMap/QuickUMLS for medical concept identification         │
│  • Measures clinical accuracy independent of general language fluency           │
│                                                                                  │
│  METRIC 7: SEMANTIC ANSWER TYPE ACCURACY                                         │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Whether predicted answer belongs to correct semantic category                │
│  • Categories: anatomical location, disease name, measurement, device           │
│  • Captures conceptual correctness beyond surface-level matching                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

| Metric | Question Types | Target | Priority |
|--------|----------------|--------|----------|
| **Exact Match** | Binary, categorical | 60-70% | Primary |
| **F1 Score** | All types | 55-65% | Primary |
| **BLEU-4** | Free-text descriptions | 0.35+ | Secondary |
| **ROUGE-L** | Free-text descriptions | 0.45+ | Secondary |
| **BERTScore** | All text answers | 0.75+ | Primary |
| **Clinical Term F1** | Finding-related | 60-70% | Primary |
| **Semantic Type Acc** | All types | 75-85% | Secondary |

### 13.2 Spatial Reasoning Metrics

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SPATIAL REASONING METRICS SUITE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  METRIC 1: INTERSECTION OVER UNION (IoU)                                         │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Primary metric for bounding box predictions                                  │
│  • Formula: IoU = Area(Pred ∩ GT) / Area(Pred ∪ GT)                             │
│  • Mean IoU computed across all localization questions                          │
│                                                                                  │
│  METRIC 2: POINTING ACCURACY                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Binary success criterion: IoU > 0.5 threshold                                │
│  • Percentage of predictions meeting this threshold                             │
│  • Aligned with standard object detection evaluation protocols                  │
│                                                                                  │
│  METRIC 3: mAP@[0.5, 0.75]                                                       │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Detection-style evaluation at multiple IoU thresholds                        │
│  • AP@0.5: Loose localization (coarse region identification)                    │
│  • AP@0.75: Strict localization (precise boundary detection)                    │
│  • Formula: mAP = (AP@0.5 + AP@0.75) / 2                                        │
│                                                                                  │
│  METRIC 4: SPATIAL RELATION ACCURACY                                             │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • For questions involving explicit spatial relationships                       │
│  • Examples: "Is the nodule superior to the hilum?"                             │
│              "Is the effusion adjacent to the diaphragm?"                       │
│  • Measures understanding of anatomical topology                                │
│                                                                                  │
│  METRIC 5: MULTI-REGION LOCALIZATION F1                                          │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • For questions requiring identification of multiple regions                   │
│  • Example: "Locate all areas of consolidation"                                 │
│  • F1 computed over set of predicted vs. ground-truth regions                   │
│                                                                                  │
│  METRIC 6: LOCALIZATION CONFIDENCE CALIBRATION (ECE)                             │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Expected Calibration Error                                                   │
│  • Correlation between predicted confidence and actual accuracy                 │
│  • Formula: ECE = Σ (|Bm|/n) × |acc(Bm) - conf(Bm)|                             │
│  • Lower ECE indicates better calibrated predictions                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

| Metric | Target | Applicability |
|--------|--------|---------------|
| **Mean IoU** | 40-50% | where_is_finding questions |
| **Pointing Accuracy** | 55-65% | All localization questions |
| **mAP@[0.5, 0.75]** | 45-55% | Detection-style evaluation |
| **Spatial Relation Acc** | 60-70% | Relational questions |
| **Multi-region F1** | 50-60% | Multi-entity localization |
| **ECE** | < 0.15 | Calibration assessment |

### 13.3 Clinical Relevance Metrics

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CLINICAL RELEVANCE METRICS SUITE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DIAGNOSTIC PERFORMANCE METRICS:                                                 │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  Sensitivity (True Positive Rate):                                       │    │
│  │  • Proportion of actual pathologies correctly identified                │    │
│  │  • Formula: Sensitivity = TP / (TP + FN)                                │    │
│  │  • Critical for: Screening applications (don't miss pathology)          │    │
│  │                                                                          │    │
│  │  Specificity (True Negative Rate):                                       │    │
│  │  • Proportion of normal cases correctly identified as negative          │    │
│  │  • Formula: Specificity = TN / (TN + FP)                                │    │
│  │  • Critical for: Reducing unnecessary follow-ups                        │    │
│  │                                                                          │    │
│  │  Positive Predictive Value (PPV):                                        │    │
│  │  • Precision for positive findings                                      │    │
│  │  • Formula: PPV = TP / (TP + FP)                                        │    │
│  │                                                                          │    │
│  │  Negative Predictive Value (NPV):                                        │    │
│  │  • Precision for negative findings                                      │    │
│  │  • Formula: NPV = TN / (TN + FN)                                        │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ADVANCED CLINICAL METRICS:                                                      │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  Matthews Correlation Coefficient (MCC):                                 │    │
│  │  • Balanced measure accounting for class imbalance                      │    │
│  │  • Formula: MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))    │    │
│  │  • Range: -1 to +1 (0 = random, 1 = perfect)                            │    │
│  │                                                                          │    │
│  │  Area Under ROC Curve (AUROC):                                          │    │
│  │  • Discrimination ability across all decision thresholds                │    │
│  │  • Computed per pathology class and averaged                            │    │
│  │  • Target: > 0.85 for primary pathologies                               │    │
│  │                                                                          │    │
│  │  Cohen's Kappa (Diagnostic Agreement):                                   │    │
│  │  • Concordance between model and radiologist ground truth               │    │
│  │  • Formula: κ = (po - pe) / (1 - pe)                                    │    │
│  │  • Interpretation: > 0.6 = substantial, > 0.8 = almost perfect          │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  PER-PATHOLOGY EVALUATION (14 CheXpert Classes):                                │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Pathology              │ Sensitivity │ Specificity │ AUROC │ Priority         │
│  ───────────────────────┼─────────────┼─────────────┼───────┼─────────────     │
│  Pneumothorax           │   > 90%     │   > 85%     │ > 0.90│ Critical         │
│  Pleural Effusion       │   > 85%     │   > 80%     │ > 0.85│ High             │
│  Cardiomegaly           │   > 80%     │   > 80%     │ > 0.85│ High             │
│  Consolidation          │   > 80%     │   > 80%     │ > 0.85│ High             │
│  Pneumonia              │   > 75%     │   > 80%     │ > 0.80│ High             │
│  Edema                  │   > 75%     │   > 80%     │ > 0.80│ Medium           │
│  Atelectasis            │   > 70%     │   > 75%     │ > 0.75│ Medium           │
│  ...                    │   ...       │   ...       │ ...   │ ...              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 13.4 Relational Reasoning Metrics

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RELATIONAL REASONING METRICS                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  These metrics assess utilization of scene graph structure for reasoning:        │
│                                                                                  │
│  METRIC 1: GRAPH ENTITY RECALL                                                   │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Percentage of ground-truth scene graph entities appearing in answers         │
│  • Formula: GER = |Entities_answer ∩ Entities_graph| / |Entities_graph|         │
│  • Measures whether models leverage available structured knowledge              │
│  • Target: > 60% for open-ended questions                                       │
│                                                                                  │
│  METRIC 2: RELATIONSHIP ACCURACY                                                 │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • For questions requiring explicit relational reasoning                        │
│  • Examples:                                                                    │
│    - "What is adjacent to the consolidation?"                                   │
│    - "Which structure is inferior to the carina?"                               │
│    - "Is the nodule in the left or right lung?"                                 │
│  • Percentage of correctly identified graph relationships                       │
│  • Target: > 55% accuracy                                                       │
│                                                                                  │
│  METRIC 3: RELATIONAL REASONING ACCURACY (Multi-Hop)                             │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Questions requiring traversal of multiple graph edges                        │
│  • Identified via dependency parsing using spaCy                                │
│  • Examples:                                                                    │
│    - "What pathology is in the region adjacent to the heart?"                   │
│    - "Describe findings in both lungs compared to the cardiac region"           │
│  • Measures complex reasoning chains over scene graph structure                 │
│  • Target: > 45% accuracy (most challenging category)                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

| Metric | Question Examples | Target | Scene Graph Dependence |
|--------|-------------------|--------|------------------------|
| **Graph Entity Recall** | Open-ended descriptions | > 60% | High |
| **Relationship Accuracy** | "What is adjacent to X?" | > 55% | Critical |
| **Multi-Hop Reasoning** | Complex spatial chains | > 45% | Critical |

### 13.5 Evaluation Stratification

Performance metrics are systematically stratified across two critical dimensions:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION STRATIFICATION MATRIX                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STRATIFICATION 1: BY QUESTION TYPE                                              │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Question Type      │ Primary Metrics          │ Secondary Metrics              │
│  ───────────────────┼──────────────────────────┼─────────────────────────────   │
│  Where (spatial)    │ IoU, Pointing Acc, mAP   │ Spatial Relation Acc           │
│  What (entity)      │ F1, Clinical Term F1     │ BERTScore                       │
│  How (severity)     │ Exact Match, Ordinal Acc │ MCC                             │
│  Yes/No (binary)    │ Exact Match, Sensitivity │ Specificity, AUROC              │
│  Describe (open)    │ BERTScore, ROUGE-L       │ BLEU-4, Graph Entity Recall     │
│                                                                                  │
│  STRATIFICATION 2: BY ANATOMICAL REGION                                          │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Region             │ Key Pathologies          │ Evaluation Focus                │
│  ───────────────────┼──────────────────────────┼─────────────────────────────   │
│  Cardiac            │ Cardiomegaly, effusion   │ Size estimation, borders        │
│  Pulmonary          │ Consolidation, nodules   │ Texture, density changes        │
│  Pleural            │ Effusion, pneumothorax   │ Boundary detection              │
│  Mediastinal        │ Masses, lymphadenopathy  │ Width measurements              │
│  Osseous            │ Fractures, deformities   │ Structural abnormalities        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Explainability and Transparency Assessment

### 14.1 Attention Analysis Framework

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EXPLAINABILITY ASSESSMENT FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Given the critical clinical context where decisions directly impact patient    │
│  care, model transparency and interpretability are paramount requirements.       │
│                                                                                  │
│  ATTENTION HEATMAP ANALYSIS:                                                     │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  Extraction Method:                                                      │    │
│  │  • Extract attention weights from cross-attention layers                │    │
│  │  • Specifically from VisualBERT's visual-text attention heads           │    │
│  │  • Also from Scene-embedded Interaction Module (SIM) attention          │    │
│  │                                                                          │    │
│  │  Visualization:                                                          │    │
│  │  • Overlay normalized attention heatmaps on original X-ray images       │    │
│  │  • Highlight regions receiving highest attention for each question      │    │
│  │  • Compare against radiologist-annotated regions of interest            │    │
│  │                                                                          │    │
│  │  Comparison Standard:                                                    │    │
│  │  • Radiologist ROI annotations from MIMIC-Ext-CXR-QBA bounding boxes   │    │
│  │  • Gold-standard attention targets from scene graph localizations       │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 14.2 Plausibility Metric

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION PLAUSIBILITY METRIC                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DEFINITION:                                                                     │
│  Plausibility = IoU(Attention_model, ROI_radiologist)                           │
│                                                                                  │
│  COMPUTATION:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  1. Extract attention heatmap from model for given question             │    │
│  │  2. Normalize attention weights to [0, 1] range                         │    │
│  │  3. Threshold at 0.5 to create binary attention mask                    │    │
│  │  4. Compare with radiologist-annotated ROI bounding box                 │    │
│  │  5. Compute IoU between attention mask and ROI                          │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  INTERPRETATION:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  Plausibility Score │ Interpretation              │ Clinical Suitability │    │
│  │  ────────────────────┼────────────────────────────┼─────────────────────  │    │
│  │  ≥ 0.65             │ Clinically acceptable       │ ✅ Deployable        │    │
│  │  0.50 - 0.65        │ Partially aligned           │ ⚠️ Needs review      │    │
│  │  0.35 - 0.50        │ Weak alignment              │ ❌ Not recommended   │    │
│  │  < 0.35             │ Misaligned attention        │ ❌ Unreliable        │    │
│  │                                                                          │    │
│  │  ★ MINIMUM THRESHOLD FOR CLINICAL USE: Plausibility ≥ 0.65              │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 14.3 Attention Entropy Analysis

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION ENTROPY METRIC                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FORMULA:                                                                        │
│  H(Attention) = -Σᵢ αᵢ × log(αᵢ)                                                │
│                                                                                  │
│  Where αᵢ represents attention weights over spatial positions                   │
│                                                                                  │
│  INTERPRETATION:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  Entropy Level │ Attention Pattern       │ Clinical Meaning              │    │
│  │  ──────────────┼─────────────────────────┼──────────────────────────     │    │
│  │  Low (< 2.0)   │ Focused on specific     │ Model confident about         │    │
│  │                │ anatomical regions      │ relevant regions              │    │
│  │                │                         │                               │    │
│  │  Medium (2-4)  │ Moderate spread         │ Multiple regions relevant     │    │
│  │                │ across regions          │ or uncertainty present        │    │
│  │                │                         │                               │    │
│  │  High (> 4.0)  │ Diffuse attention       │ Model uncertain, looking      │    │
│  │                │ across entire image     │ everywhere (potentially       │    │
│  │                │                         │ unreliable prediction)        │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  EXPECTED PATTERNS BY QUESTION TYPE:                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  Question Type         │ Expected Entropy │ Rationale                    │    │
│  │  ──────────────────────┼──────────────────┼─────────────────────────     │    │
│  │  where_is_finding      │ Low (< 2.0)      │ Should focus on location     │    │
│  │  is_abnormal_region    │ Low-Medium       │ Focus on specific region     │    │
│  │  has_finding           │ Medium           │ May need whole-image scan    │    │
│  │  describe_all          │ High             │ Requires global attention    │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Statistical Significance Testing Framework

### 15.1 Hypothesis Testing Methods

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STATISTICAL SIGNIFICANCE TESTING                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  All pairwise model comparisons undergo rigorous statistical evaluation         │
│  to establish whether observed performance differences reflect genuine          │
│  model capabilities rather than random variation.                               │
│                                                                                  │
│  TEST 1: PAIRED t-TEST (Continuous Metrics)                                      │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Formula: t = d̄ / (sd / √n)                                                     │
│                                                                                  │
│  Where:                                                                          │
│  • d̄ = mean pairwise difference between models                                 │
│  • sd = standard deviation of differences                                       │
│  • n = sample size                                                              │
│                                                                                  │
│  Applied to: IoU, BERTScore, BLEU, ROUGE-L, F1 scores                           │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  TEST 2: McNEMAR'S TEST (Binary Metrics)                                         │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Formula: χ² = (n₁₀ - n₀₁)² / (n₁₀ + n₀₁)                                       │
│                                                                                  │
│  Where:                                                                          │
│  • n₁₀ = cases where Model A correct, Model B incorrect                         │
│  • n₀₁ = cases where Model B correct, Model A incorrect                         │
│                                                                                  │
│  Applied to: Exact Match, Pointing Accuracy, Binary classification              │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  TEST 3: BOOTSTRAP CONFIDENCE INTERVALS                                          │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Method: Non-parametric bootstrap with 10,000 resamples                         │
│  CI Formula: [Percentile₂.₅(θ*), Percentile₉₇.₅(θ*)]                            │
│                                                                                  │
│  Where θ* represents metric computed on each bootstrap sample                   │
│                                                                                  │
│  Provides: 95% confidence intervals for all reported metrics                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 15.2 Effect Size and Multiple Comparisons

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EFFECT SIZE & CORRECTION METHODS                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  EFFECT SIZE: COHEN'S d                                                          │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Formula: d = (x̄₁ - x̄₂) / √((s₁² + s₂²) / 2)                                   │
│                                                                                  │
│  Interpretation:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  |d| Value │ Effect Size  │ Practical Significance                      │    │
│  │  ──────────┼──────────────┼──────────────────────────────────────────   │    │
│  │  < 0.2    │ Negligible   │ Not practically meaningful                  │    │
│  │  0.2-0.5  │ Small        │ Detectable but minor improvement            │    │
│  │  0.5-0.8  │ Medium       │ Meaningful improvement worth pursuing       │    │
│  │  > 0.8    │ Large        │ Substantial improvement, highly impactful   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  MULTIPLE COMPARISONS: BONFERRONI CORRECTION                                     │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Formula: α_corrected = α / k                                                   │
│                                                                                  │
│  Where:                                                                          │
│  • α = 0.05 (standard significance threshold)                                   │
│  • k = number of simultaneous comparisons                                       │
│                                                                                  │
│  Example for ablation study with 6 comparisons:                                 │
│  • α_corrected = 0.05 / 6 = 0.0083                                             │
│  • Only differences with p < 0.0083 considered significant                      │
│                                                                                  │
│  Purpose: Controls family-wise error rate, ensuring probability of any          │
│           false positive across all tests remains below 0.05                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 15.3 Reporting Standards

| Comparison | Test Used | Report Format | Significance Threshold |
|------------|-----------|---------------|------------------------|
| Model A vs B (continuous) | Paired t-test | t(df) = X.XX, p = 0.XXX, d = X.XX | p < 0.05 (corrected) |
| Model A vs B (binary) | McNemar's | χ²(1) = X.XX, p = 0.XXX | p < 0.05 (corrected) |
| All metrics | Bootstrap | Mean [95% CI: lower, upper] | Non-overlapping CIs |
| Cross-dataset | Paired t-test | With domain effect noted | p < 0.05 (corrected) |

---

## 16. Detailed Ablation Study Design

### 16.1 Ablation Conditions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ABLATION STUDY CONDITIONS                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  All ablation conditions trained from scratch under identical settings          │
│  on MIMIC-Ext-CXR-QBA only (no external data):                                  │
│                                                                                  │
│  CONDITION 1: FULL PROPOSED MODEL (SG-Enhanced) [BASELINE]                       │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  │                                                                              │
│  │  Components:                                                                 │
│  │  ├── Visual Backbone: ConvNeXt-Base (pretrained ImageNet-22k)               │
│  │  ├── Region Features: ROI Align using scene graph bboxes                    │
│  │  ├── Scene Graph: Full 134-dim embeddings (region + entity)                 │
│  │  ├── Text Encoder: Bio+ClinicalBERT                                         │
│  │  ├── Fusion: Scene-embedded Interaction Module (SIM) - 2 layers             │
│  │  └── Answer Heads: Multi-head (binary + category + region + severity)       │
│  │                                                                              │
│  │  Total Parameters: ~150M                                                     │
│  │                                                                              │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONDITION 2: NO-SG BASELINE (Scene Graph Removed)                               │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  │                                                                              │
│  │  Components REMOVED:                                                         │
│  │  ├── ❌ Scene Graph Encoder (region/entity embeddings)                       │
│  │  ├── ❌ Scene-embedded Interaction Module (SIM)                              │
│  │  ├── ❌ ROI Align using scene graph bboxes                                   │
│  │  └── ❌ All graph-related attention mechanisms                               │
│  │                                                                              │
│  │  Components RETAINED:                                                        │
│  │  ├── ✓ ConvNeXt-Base (global features only)                                 │
│  │  ├── ✓ Bio+ClinicalBERT                                                     │
│  │  ├── ✓ Multi-head answer architecture                                       │
│  │  └── ✓ Standard vision-language attention                                   │
│  │                                                                              │
│  │  Purpose: Quantify scene graph contribution                                  │
│  │                                                                              │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONDITION 3: VISION+QUESTION BASELINE (Minimal)                                 │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  │                                                                              │
│  │  Components:                                                                 │
│  │  ├── ConvNeXt-Base (global pooled features only)                            │
│  │  ├── Bio+ClinicalBERT (question encoding)                                   │
│  │  └── Simple late fusion (concatenate + MLP)                                 │
│  │                                                                              │
│  │  Removed:                                                                    │
│  │  ├── ❌ All region-level features                                           │
│  │  ├── ❌ All scene graph components                                          │
│  │  └── ❌ Cross-modal attention mechanisms                                    │
│  │                                                                              │
│  │  Purpose: Lower bound - what vision+language alone achieves                 │
│  │                                                                              │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONDITION 4: ResNet18 BACKBONE (Original SSG-VQA Configuration)                 │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  │                                                                              │
│  │  Changes from Baseline:                                                      │
│  │  └── Visual backbone: ResNet18 instead of ConvNeXt-Base                     │
│  │                                                                              │
│  │  Purpose: Quantify ConvNeXt upgrade benefit (+4-7% expected)                │
│  │                                                                              │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONDITION 5: GENERIC TOKENIZER (Original SSG-VQA Configuration)                 │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  │                                                                              │
│  │  Changes from Baseline:                                                      │
│  │  └── Text encoder: Standard BERT instead of Bio+ClinicalBERT                │
│  │                                                                              │
│  │  Purpose: Quantify medical tokenizer benefit (+2.8-4.5% expected)           │
│  │                                                                              │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONDITION 6: SINGLE-HEAD ANSWER (Unified Classification)                        │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  │                                                                              │
│  │  Changes from Baseline:                                                      │
│  │  └── Single classification head (500 unified answer classes)                │
│  │      instead of multi-head architecture                                     │
│  │                                                                              │
│  │  Purpose: Quantify multi-head architecture benefit                          │
│  │                                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 16.2 Expected Results Matrix

| Ablation | Overall Acc | Spatial Acc | Relational Acc | Training Loss @Ep10 |
|----------|-------------|-------------|----------------|---------------------|
| **Full Model** | Baseline | Baseline | Baseline | Baseline |
| **No-SG** | -8 to -12% | -15 to -25% | -20 to -30% | +10-15% |
| **Vision+Question** | -12 to -18% | -25 to -35% | -30 to -40% | +15-20% |
| **ResNet18 Backbone** | -4 to -7% | -6 to -10% | -5 to -8% | +5-8% |
| **Generic Tokenizer** | -2.8 to -4.5% | -2 to -4% | -3 to -5% | +3-5% |
| **Single-Head** | -3 to -6% | -2 to -4% | -2 to -4% | +2-4% |

### 16.3 Cross-Dataset Zero-Shot Protocol

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ZERO-SHOT CROSS-DATASET EVALUATION                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CRITICAL: No training or fine-tuning on target datasets                        │
│  Models applied directly after MIMIC-Ext-CXR-QA training only                   │
│                                                                                  │
│  TARGET DATASETS:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  VQA-RAD:                                                                │    │
│  │  • 315 images, 3,515 QA pairs                                           │    │
│  │  • Radiology-specific questions                                         │    │
│  │  • NO scene graphs provided                                             │    │
│  │                                                                          │    │
│  │  SLAKE-EN:                                                               │    │
│  │  • 701 images, ~14,000 QA pairs                                         │    │
│  │  • Semantic labels for knowledge-enhanced assessment                    │    │
│  │  • NO scene graphs provided                                             │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  INFERENCE CONFIGURATION:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  • Scene graph input branch: DISABLED                                   │    │
│  │  • Visual features: Global ConvNeXt features only                       │    │
│  │  • No bbox-based ROI features (not available)                           │    │
│  │                                                                          │    │
│  │  PURPOSE: Test whether scene graph pre-training benefits transfer       │    │
│  │           even when graphs unavailable at inference time                │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  HYPOTHESIS:                                                                     │
│  Scene-graph-aware pre-training induces superior visual-semantic                │
│  representations that transfer to datasets without explicit graphs.              │
│                                                                                  │
│  Expected gains on spatial/relational subsets: +10-25% vs No-SG baseline        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 17. Error Analysis Framework

### 17.1 Failure Mode Categories

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ERROR ANALYSIS FRAMEWORK                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CATEGORY 1: SCENE GRAPH ERRORS                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Error Type           │ Prevalence │ Impact                            │    │
│  │  ──────────────────────┼────────────┼──────────────────────────────     │    │
│  │  Missing entities     │ 0.8%       │ Cannot reference certain findings │    │
│  │  Incorrect relations  │ ~3%        │ Wrong spatial reasoning           │    │
│  │  Entity misclassify   │ ~2%        │ Wrong finding type identified     │    │
│  │  Duplicate entities   │ ~1.5%      │ Redundant information             │    │
│  │                                                                          │    │
│  │  Analysis Approach:                                                      │    │
│  │  • Compare model errors vs scene graph errors                           │    │
│  │  • Identify propagation rate (SG error → model error)                   │    │
│  │  • Measure model robustness to SG noise                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  CATEGORY 2: OVERLAPPING BOUNDING BOX CONFUSION                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  Problem: 99.996% of boxes overlap with avg max IoU of 37.1%            │    │
│  │                                                                          │    │
│  │  Manifestations:                                                         │    │
│  │  • "Where is the consolidation?" → Model points to wrong lung           │    │
│  │  • Adjacent structures confused (hilum vs heart border)                 │    │
│  │  • Overlapping pathologies misattributed                                │    │
│  │                                                                          │    │
│  │  Analysis Approach:                                                      │    │
│  │  • Stratify errors by bbox IoU with neighbors                           │    │
│  │  • High-overlap regions: expect higher error rates                      │    │
│  │  • Measure disambiguation capability via learned embeddings             │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  CATEGORY 3: QUESTION TYPES WHERE GRAPHS DON'T HELP                              │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │  Questions where scene graphs provide minimal/no benefit:               │    │
│  │                                                                          │    │
│  │  • Simple binary queries: "Is there cardiomegaly?"                      │    │
│  │    → Global image features sufficient                                   │    │
│  │                                                                          │    │
│  │  • Whole-image assessments: "Is this a normal chest X-ray?"             │    │
│  │    → Holistic judgment, not region-specific                             │    │
│  │                                                                          │    │
│  │  • Counting questions: "How many nodules are present?"                  │    │
│  │    → May actually hurt if SG has wrong count                            │    │
│  │                                                                          │    │
│  │  Analysis Approach:                                                      │    │
│  │  • Compare Full Model vs No-SG per question type                        │    │
│  │  • Identify question types where Δ accuracy ≈ 0 or negative             │    │
│  │  • Inform future architectural improvements                             │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 17.2 Error Analysis Metrics

| Category | Measurement | Target Analysis |
|----------|-------------|-----------------|
| **SG Propagation** | % of SG errors → model errors | < 50% (robust model) |
| **Bbox Confusion** | Error rate vs neighbor IoU | Correlation analysis |
| **SG-Independent** | Question types where SG hurts | Identify & document |
| **Attention Misalignment** | Plausibility < 0.35 cases | Root cause analysis |

---

## 18. Expected Outcomes and Research Contributions

### 18.1 Research Contributions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EXPECTED RESEARCH CONTRIBUTIONS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CONTRIBUTION 1: FIRST SYSTEMATIC EVALUATION OF SCENE GRAPH IMPACT              │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  • First large-scale study on scene graph contribution to Med-VQA              │
│  • Quantified impact on spatial reasoning questions                            │
│  • Analysis across 42 million QA pairs with controlled ablations               │
│  • Novel insights into when scene graphs help vs. hurt performance             │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONTRIBUTION 2: RIGOROUS ABLATION METHODOLOGY FOR MED-VQA                       │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  • Standardized ablation framework with 6 controlled conditions                │
│  • Statistical significance testing with Bonferroni correction                 │
│  • Effect size reporting (Cohen's d) for practical significance               │
│  • Bootstrap confidence intervals for robust uncertainty estimation            │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONTRIBUTION 3: TRANSFER LEARNING INSIGHTS FOR GRAPH-ENHANCED MODELS           │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  • Zero-shot cross-dataset evaluation protocol                                 │
│  • Evidence that scene graph pre-training benefits transfer                    │
│    even when graphs unavailable at inference                                   │
│  • Practical guidance for deploying graph-enhanced models                      │
│    in real-world settings without graph annotations                            │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONTRIBUTION 4: PROGRESSION ANALYSIS & TRAINING DYNAMICS                        │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  • Detailed training curves comparing SG vs. non-SG variants                   │
│  • Convergence analysis (expected 10-15% faster with scene graphs)             │
│  • Per-question-type learning dynamics                                         │
│  • Insights into optimal training strategies for graph-enhanced VQA            │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CONTRIBUTION 5: ARCHITECTURAL ADAPTATIONS FOR MEDICAL IMAGING                   │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  • ConvNeXt-Base adaptation for chest X-ray feature extraction                 │
│  • Expanded scene graph embeddings (310 regions, 237 entities)                 │
│  • Multi-head answer architecture for diverse question types                   │
│  • Bio+ClinicalBERT integration for medical terminology                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 18.2 Performance Targets (Based on Methodology)

| Metric | Target | Question Type | Head Used |
|--------|--------|---------------|-----------|
| **Exact Match** | 60-70% | Binary (Yes/No) | `binary_head` |
| **F1 Score** | 55-65% | Finding classification | `category_head` |
| **IoU** | 40-50% | Localization | `region_head` |
| **Accuracy** | 50-60% | Severity | `severity_head` |
| **BERTScore** | 0.75+ | Free-text (future) | `text_head` |

### 18.3 Per-Question-Type Expected Performance

| Question Strategy | Question Type | Expected Accuracy | Training Priority |
|-------------------|---------------|-------------------|-------------------|
| **Abnormal** | is_abnormal | 70-75% | Phase 1 |
| **Abnormal** | is_normal | 68-73% | Phase 1 |
| **Finding** | has_finding | 65-70% | Phase 1 |
| **Finding** | describe_finding | 55-60% | Phase 2 |
| **Finding** | where_is_finding | 50-55% | Phase 2 |
| **Finding** | how_severe_is_finding | 55-60% | Phase 2 |
| **Region** | describe_region | 50-55% | Phase 2 |
| **Region** | is_abnormal_region | 62-67% | Phase 2 |
| **Indication** | indication | 45-50% | Phase 3 |

### 18.4 Original Ablation Study Design (Brief Summary)

```
Experiment 1: Full model (scene graph + ConvNeXt visual + text)     [BASELINE]
Experiment 2: No scene graph (ConvNeXt visual + text only)          [SG impact]
Experiment 3: No visual features (scene graph + text only)          [Visual impact]
Experiment 4: ResNet18 backbone (scene graph + ResNet + text)       [Backbone comparison]
Experiment 5: Single-head vs Multi-head answer architecture         [Head design]
Experiment 6: With/without mixed precision training                 [Optimization impact]
```

*Note: See Section 16 for detailed ablation study design with specific conditions.*

---

## 19. Appendix: Key Data Structures

### A. Scene Graph JSON Example

```json
{
  "patient_id": "p10000032",
  "study_id": "s50414267",
  "observations": {
    "O01": {
      "name": "no focal consolidation",
      "summary_sentence": "There is no focal consolidation.",
      "regions": [{"region": "lungs"}],
      "obs_entities": ["consolidation"],
      "probability": "negative",
      "localization": {
        "image_id_here": {
          "bboxes": [[888.0, 370.0, 1610.0, 1642.0]]
        }
      }
    }
  },
  "regions": {
    "left lung": {
      "localization": {"bboxes": [[...]]}
    }
  }
}
```

### B. QA JSON Example

```json
{
  "patient_id": "p10000032",
  "study_id": "s50414267",
  "questions": [
    {
      "question_id": "xxx",
      "question_type": "is_abnormal_region",
      "question": "Are there any abnormalities in the lungs?",
      "answers": [
        {
          "text": "No, there are no focal consolidations in the lungs.",
          "positiveness": "neg",
          "regions": ["lungs"],
          "localization": {"bboxes": [[...]]}
        }
      ]
    }
  ]
}
```

### C. Metadata CSV Relationships

```
patient_metadata.csv
  └── patient_id (unique)
       └── study_metadata.csv
            └── study_id (unique per patient)
                 ├── image_metadata.csv
                 │    └── image_id (dicom_id)
                 └── question_metadata.csv
                      └── question_id (unique per study)
                           └── answer_metadata.csv
                                └── answer_id
```

### D. Multi-Head Output Format

```python
# Example model output structure
model_output = {
    'binary': torch.tensor([[0.2, 0.8]]),      # [No, Yes] probabilities
    'category': torch.tensor([[0.1, 0.05, ..., 0.3]]),  # 14 CheXpert classes
    'region': torch.tensor([[0.4, 0.1, ..., 0.2]]),     # 26 region classes
    'severity': torch.tensor([[0.6, 0.2, 0.15, 0.05]]), # [none, mild, moderate, severe]
}

# Training target format
targets = {
    'binary': torch.tensor([1]),       # Yes
    'category': torch.tensor([3]),     # e.g., 'Edema'
    'region': torch.tensor([0]),       # e.g., 'left_lung'
    'severity': torch.tensor([2]),     # e.g., 'moderate'
}
```

---

## 20. Summary of Selected Configuration

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MIMIC-CXR VQA FINAL CONFIGURATION                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  VISUAL BACKBONE:     ConvNeXt-Base (convnext_base.fb_in22k_ft_in1k)        ║
║  FEATURE DIMENSIONS:  646 dims (EXPANDED: 134 scene + 512 visual)           ║
║  SCENE GRAPH DIMS:    134 dims (6 bbox + 64 region_emb + 64 entity_emb)     ║
║  TEXT ENCODER:        Bio+ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)    ║
║  ANSWER ARCHITECTURE: Multi-head (binary + category + region + severity)     ║
║                                                                              ║
║  SCENE GRAPH COVERAGE:                                                       ║
║    ├── Regions:            310 anatomical regions (learned 64-dim embed)    ║
║    ├── Entities:           237 finding entities (learned 64-dim embed)      ║
║    └── VisualBert Config:  visual_embedding_dim = 646                       ║
║                                                                              ║
║  TRAINING:                                                                   ║
║    ├── Mixed Precision:    FP16 enabled                                     ║
║    ├── DeepSpeed:          ZeRO Stage 2                                     ║
║    ├── Batch Size:         256 effective (16 × 4 GPUs × 4 accum)            ║
║    ├── Pre-training:       31.2M pairs, 3-5 epochs (~2.5-3 days)            ║
║    └── Fine-tuning:        7.5M pairs, 10-20 epochs (~2-2.5 days)           ║
║                                                                              ║
║  HARDWARE:                                                                   ║
║    ├── GPUs:               4× NVIDIA L4 (96GB total)                        ║
║    ├── CPU:                48 vCPUs (Intel Cascade Lake)                    ║
║    ├── RAM:                192 GB                                           ║
║    └── Storage:            1000 GB SSD                                      ║
║                                                                              ║
║  ESTIMATED TOTAL TIME:  5-6 days (with optimizations)                        ║
║                                                                              ║
║  KEY INNOVATIONS FROM METHODOLOGY:                                           ║
║    ✓ ConvNeXt-Base (vs ResNet18): +4-7% accuracy                            ║
║    ✓ Bio+ClinicalBERT: +2.8-4.5% on medical text                            ║
║    ✓ Expanded Scene Graph: 310 regions, 237 entities (vs 6/4 original)      ║
║    ✓ Multi-Head Answers: Binary + Category + Region + Severity              ║
║    ✓ Scene-Embedded Interaction Module (SIM): Preserved from SSG-VQA        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**See `architecture_diagram.md` for detailed Mermaid diagrams of the complete architecture.**

---

## Document Information

- **Created**: January 5, 2026
- **Updated**: January 5, 2026
- **Purpose**: Analysis document for adapting SSG-VQA-Net to MIMIC-CXR
- **Status**: Pre-implementation analysis - DECISIONS FINALIZED
- **Next Steps**: Proceed to Phase 1 implementation (data download & setup)

### Document Sections Summary

| Section | Title | Content |
|---------|-------|---------|
| 1 | Current State Assessment | What we have, architectural differences |
| 2 | Dataset Correlation Analysis | Data linking, hierarchy, quality statistics, bias mitigation |
| 3 | Radiologist Annotations | CheXpert labels, multi-task learning |
| 4 | YOLOv8 Object Detection | Detection module for scene graph construction |
| 5 | Data Loading Pipeline | File organization, loading architecture |
| 6 | Feature Format Transformation | Visual features, scene graph encoding |
| 7 | Multi-Image Handling | View-specific processing |
| 8 | Quality Filtering Strategy | Training stages by quality grade |
| 9 | Implementation Roadmap | Phases, timeline, milestones |
| 10 | Critical Decision Points | Final design decisions, risks |
| 11 | Training Optimizations | Speed, memory, configuration |
| 12 | Logging & Checkpointing | wandb, HuggingFace Hub |
| 13 | Comprehensive Evaluation | Answer, spatial, clinical, relational metrics |
| 14 | Explainability Assessment | Attention analysis, plausibility, entropy |
| 15 | Statistical Testing | t-tests, McNemar's, bootstrap, effect sizes |
| 16 | Ablation Study Design | 6 conditions, cross-dataset protocol |
| 17 | Error Analysis Framework | Failure modes, scene graph errors |
| 18 | Expected Outcomes | Research contributions, performance targets |
| 19 | Appendix | Data structure examples |
| 20 | Summary | Final configuration |

