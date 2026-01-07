# Metadata Integration Guide: MIMIC-CXR VQA Training

## Overview

This document maps all available metadata from your analysis to the model training pipeline, showing:
- What metadata is available
- Current integration status
- How each piece should flow through the model

---

## 1. Complete Metadata Inventory

### 1.1 MIMIC-CXR-JPG Metadata

| Source File | Metadata | Current Status | Used For |
|-------------|----------|----------------|----------|
| `mimic-cxr-2.0.0-chexpert.csv.gz` | 14 CheXpert labels per study | ✅ **INTEGRATED** | Auxiliary loss supervision |
| `mimic-cxr-2.0.0-metadata.csv.gz` | ViewPosition, Rows, Columns, StudyDate | ✅ **INTEGRATED** | View filtering, image dimensions |
| `mimic-cxr-2.0.0-split.csv.gz` | train/validate/test splits | ✅ **INTEGRATED** | Dataset splitting |
| `mimic-cxr-2.1.0-test-set-labeled.csv` | Gold standard radiologist labels | ⚠️ **PARTIAL** | Evaluation only |

### 1.2 MIMIC-Ext-CXR-QBA Metadata (from `metadata/` folder)

| Source File | Metadata Fields | Current Status | Should Be Used For |
|-------------|-----------------|----------------|-------------------|
| **patient_metadata.parquet** | split, patient.total_studies, patient.total_study_timespan | ⚠️ **NOT USED** | Patient-level sampling weights |
| **study_metadata.parquet** | study.procedure, study.num_images, study.num_observations, study.quality.* | ⚠️ **PARTIAL** | Quality filtering, study weighting |
| **image_metadata.parquet** | img.view_position, img.size, img.bbox_* | ⚠️ **PARTIAL** | Bbox normalization |
| **question_metadata.parquet** | question_type, question_strategy, question_quality | ⚠️ **NOT USED** | Question-type balanced sampling |
| **answer_metadata.parquet** | answer_type, positiveness, regions, entities, modifiers | ⚠️ **PARTIAL** | Answer parsing, class weights |

### 1.3 Scene Graph Data (from `scene_data/`)

| Component | Fields | Current Status | Used For |
|-----------|--------|----------------|----------|
| **observations** | name, regions, obs_entities, positiveness, localization | ✅ **INTEGRATED** | Scene graph features |
| **regions** | region names, bboxes | ✅ **INTEGRATED** | Region embeddings |
| **located_at_relations** | observation → region mappings | ⚠️ **NOT USED** | Could enhance spatial reasoning |
| **obs_relations** | parent-child observations | ⚠️ **NOT USED** | Could enhance relational reasoning |
| **sentences** | natural language descriptions | ⚠️ **NOT USED** | Could be used for text augmentation |

### 1.4 QA Data (from `qa/`)

| Field | Current Status | Used For |
|-------|----------------|----------|
| question_id | ✅ Used | Tracking |
| question_type | ✅ Used | Head routing |
| question_strategy | ⚠️ Not used | Could improve balanced sampling |
| question | ✅ Used | Model input |
| question_quality | ⚠️ Partial | Quality filtering |
| answers[].text | ✅ Used | Answer parsing |
| answers[].positiveness | ✅ Used | Binary answer labels |
| answers[].regions | ⚠️ Partial | Region head target |
| answers[].obs_entities | ⚠️ Partial | Entity head target |
| answers[].modifiers | ⚠️ Partial | Severity parsing |
| obs_ids | ⚠️ Not used | Link to scene graph observations |

---

## 2. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       COMPLETE METADATA INTEGRATION MAP                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: DATA LOADING (MIMICCXRVQADataset.__init__)                    │    │
│  │                                                                          │    │
│  │  MIMIC-CXR-JPG:                                                         │    │
│  │  ├── splits.csv.gz         ─► Filter by train/validate/test            │    │
│  │  ├── metadata.csv.gz       ─► Load ViewPosition for view filtering     │    │
│  │  └── chexpert.csv.gz       ─► Initialize CheXpertLabelLoader           │    │
│  │                                                                          │    │
│  │  MIMIC-Ext-CXR-QBA:                                                     │    │
│  │  ├── metadata/*.parquet    ─► Load for quality filtering               │    │
│  │  └── dataset_info.json     ─► Load region/entity vocabularies          │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: SAMPLE LOADING (MIMICCXRVQADataset.__getitem__)               │    │
│  │                                                                          │    │
│  │  For each QA pair:                                                       │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ 1. IMAGE LOADING                                                 │    │    │
│  │  │    qa/p{XX}/p{subject_id}/s{study_id}.qa.json                   │    │    │
│  │  │              ↓                                                   │    │    │
│  │  │    ├── subject_id, study_id ─► Locate image in MIMIC-CXR-JPG    │    │    │
│  │  │    ├── ViewPosition ─► Select PA/AP frontal image               │    │    │
│  │  │    └── Image ─► transforms ─► image_tensor (3, 224, 224)        │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ 2. SCENE GRAPH LOADING                                          │    │    │
│  │  │    scene_data/p{XX}/p{subject_id}/s{study_id}.scene_graph.json  │    │    │
│  │  │              ↓                                                   │    │    │
│  │  │    ├── observations ─► Extract per observation:                 │    │    │
│  │  │    │   ├── localization.bboxes ─► normalized bboxes (N, 4)      │    │    │
│  │  │    │   ├── regions[].region ─► region_ids (N,)                  │    │    │
│  │  │    │   ├── obs_entities ─► entity_ids (N,)                      │    │    │
│  │  │    │   └── positiveness ─► polarity (N,)                        │    │    │
│  │  │    │                                                             │    │    │
│  │  │    └── Output: scene_graph_features dict                        │    │    │
│  │  │        {bboxes, region_ids, entity_ids, positiveness}           │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ 3. QUESTION PROCESSING                                          │    │    │
│  │  │    From qa.json:                                                │    │    │
│  │  │    ├── question ─► Bio+ClinicalBERT tokenizer                   │    │    │
│  │  │    │   └── input_ids, attention_mask (128,)                     │    │    │
│  │  │    │                                                             │    │    │
│  │  │    ├── question_type ─► Head routing (string)                   │    │    │
│  │  │    │                                                             │    │    │
│  │  │    └── question_quality ─► Quality filtering                    │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ 4. ANSWER PROCESSING                                            │    │    │
│  │  │    From qa.json answers[]:                                      │    │    │
│  │  │    ├── positiveness ─► Binary answer (0/1)                      │    │    │
│  │  │    ├── modifiers[severity] ─► Severity class (0-3)              │    │    │
│  │  │    ├── regions ─► Region target class                           │    │    │
│  │  │    ├── obs_entities ─► Category target class                    │    │    │
│  │  │    │                                                             │    │    │
│  │  │    └── Output: answer_idx (target for loss)                     │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ 5. CHEXPERT LABELS                                              │    │    │
│  │  │    From chexpert.csv.gz:                                        │    │    │
│  │  │    ├── 14 disease labels per (subject_id, study_id)             │    │    │
│  │  │    ├── Handle -1 (uncertain) → mask or soft label               │    │    │
│  │  │    │                                                             │    │    │
│  │  │    └── Output:                                                  │    │    │
│  │  │        chexpert_labels (14,) ─► Auxiliary supervision           │    │    │
│  │  │        chexpert_mask (14,)   ─► Ignore uncertain labels         │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 3: BATCH COLLATION (collate_fn)                                  │    │
│  │                                                                          │    │
│  │  Output Dict:                                                            │    │
│  │  {                                                                       │    │
│  │    'images':           (B, 3, 224, 224)  ─► ConvNeXt Backbone           │    │
│  │    'input_ids':        (B, 128)          ─► Bio+ClinicalBERT            │    │
│  │    'attention_mask':   (B, 128)          ─► Bio+ClinicalBERT            │    │
│  │    'token_type_ids':   (B, 128)          ─► Bio+ClinicalBERT            │    │
│  │    'scene_graphs':     List[Dict] len=B  ─► SceneGraphEncoder           │    │
│  │    'question_types':   List[str] len=B   ─► Head routing                │    │
│  │    'answer_idx':       (B,)              ─► VQA loss target             │    │
│  │    'chexpert_labels':  (B, 14)           ─► Auxiliary loss target       │    │
│  │    'chexpert_mask':    (B, 14)           ─► Mask uncertain labels       │    │
│  │  }                                                                       │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 4: MODEL FORWARD (MIMICCXRVQAModel.forward)                      │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ VISUAL BRANCH                                                   │    │    │
│  │  │ images ─► ConvNeXt-Base ─► feature_maps                         │    │    │
│  │  │                   ↓                                              │    │    │
│  │  │ scene_graphs.bboxes ─► ROI Align ─► roi_features (B, N, 512)    │    │    │
│  │  │                                             ↓                    │    │    │
│  │  │                                     visual_features              │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ SCENE GRAPH BRANCH                                              │    │    │
│  │  │ scene_graphs:                                                   │    │    │
│  │  │   ├── bboxes (N, 4)     ─► normalized coords (6 dims)           │    │    │
│  │  │   ├── region_ids (N,)   ─► nn.Embedding(310, 64) ─► 64 dims     │    │    │
│  │  │   ├── entity_ids (N,)   ─► nn.Embedding(237, 64) ─► 64 dims     │    │    │
│  │  │   └── positiveness      ─► encoded in entity embedding          │    │    │
│  │  │                    Total: 6 + 64 + 64 = 134 dims                │    │    │
│  │  │                                             ↓                    │    │    │
│  │  │                                     scene_features               │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ TEXT BRANCH                                                     │    │    │
│  │  │ input_ids + attention_mask ─► Bio+ClinicalBERT ─► text_features │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ FUSION (Scene-Embedded Interaction Module)                      │    │    │
│  │  │                                                                  │    │    │
│  │  │ visual_features ◄──► text_features ◄──► scene_features           │    │    │
│  │  │         │               │                    │                   │    │    │
│  │  │         └────────Cross Attention─────────────┘                   │    │    │
│  │  │                         ↓                                        │    │    │
│  │  │                  fused_output (B, 768)                           │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ OUTPUT HEADS                                                    │    │    │
│  │  │                                                                  │    │    │
│  │  │ fused_output ─► MultiHeadAnswerModule                           │    │    │
│  │  │    ├── binary_head:   (B, 2)    ◄── question_type routing       │    │    │
│  │  │    ├── category_head: (B, 14)   ◄── question_type routing       │    │    │
│  │  │    ├── region_head:   (B, 26)   ◄── question_type routing       │    │    │
│  │  │    └── severity_head: (B, 4)    ◄── question_type routing       │    │    │
│  │  │                                                                  │    │    │
│  │  │ visual_features ─► CheXpertHead ─► (B, 14)                      │    │    │
│  │  │                                                                  │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 5: LOSS COMPUTATION (MultiTaskLoss)                              │    │
│  │                                                                          │    │
│  │  question_types ─► Select correct head per sample                       │    │
│  │                                                                          │    │
│  │  L_vqa = Σ CrossEntropy(head_logits[i], answer_idx[i])                  │    │
│  │          for each sample i, using head from question_type               │    │
│  │                                                                          │    │
│  │  L_chexpert = BCE(chexpert_logits, chexpert_labels) × chexpert_mask     │    │
│  │                                                                          │    │
│  │  L_total = L_vqa + λ × L_chexpert                                       │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Metadata That Needs Enhancement

Based on your analysis, here are the enhancements needed:

### 3.1 Study Quality Metadata (Currently Partial)

From `study_metadata.parquet`:
```
study.quality.extraction_rating:
  - 3_B: 203,244 (89.44%)  ← Pre-training data
  - 1_D: 15,042
  - 2_C: 6,408
  - 4_A: 2,211             ← Fine-tuning data
  - 6_A++: 309             ← Fine-tuning data
  - 5_A+: 25               ← Fine-tuning data
```

**Enhancement**: Add quality-weighted sampling for training.

### 3.2 View Position (Currently Used for Filtering Only)

From `image_metadata.parquet`:
```
img.view_position:
  - frontal: 243,334 (64.7%)
  - lateral: 132,841 (35.3%)
```

**Enhancement**: Could add view_position as a conditioning signal to the model.

### 3.3 Patient Temporal Information (Not Used)

From `patient_metadata.parquet`:
```
patient.total_studies: range=[1, 158], mean=3.5
patient.total_study_timespan: up to 1331 days
```

**Enhancement**: Could be used for temporal modeling or patient-level consistency.

---

## 4. Implementation: Enhanced Dataset

Here's the enhanced dataset that incorporates all metadata:

---

## 5. Current Integration Status

Based on code analysis, here's what's **CURRENTLY INTEGRATED** vs **WHAT NEEDS ENHANCEMENT**:

### ✅ Fully Integrated (Working)

| Metadata | File/Component | How It's Used |
|----------|----------------|---------------|
| **CheXpert labels (14 categories)** | `CheXpertLabelLoader` | Auxiliary loss supervision |
| **Split (train/val/test)** | `_load_samples()` | Dataset split filtering |
| **ViewPosition** | `metadata_df`, `_is_valid_view()` | Frontal view prioritization |
| **Image dimensions** | `original_size` | Bbox normalization |
| **Scene graph observations** | `SceneGraphProcessor.process()` | Feature extraction |
| **Observation regions** | `region_ids` | 64-dim learned embeddings |
| **Observation entities** | `entity_ids` | 64-dim learned embeddings |
| **Observation positiveness** | `positiveness` array | Binary encoding (0/1) |
| **Observation bboxes** | `bboxes` array | Normalized (N, 4) |
| **Question text** | tokenizer | `input_ids`, `attention_mask` |
| **Question type** | `QUESTION_TYPE_MAP` | Head routing |
| **Answer positiveness** | `_get_answer_idx()` | Binary answer target |
| **Answer modifiers (severity)** | `_get_answer_idx()` | Severity class (0-3) |

### ⚠️ Partially Used (Can Be Enhanced)

| Metadata | Current Use | Potential Enhancement |
|----------|-------------|----------------------|
| **study.quality.extraction_rating** | Used for filtering | Could use for sample weighting |
| **Answer regions** | Not used directly | Could be region head target |
| **Answer entities** | Not used directly | Could be category head target |
| **study.num_observations** | Not used | Could normalize attention |
| **study.num_images** | Not used | Multi-view fusion (future) |

### ❌ Not Yet Integrated (Should Add)

| Metadata | Source | Recommended Use |
|----------|--------|-----------------|
| **question_strategy** | qa.json | Balanced sampling |
| **question_quality** | qa.json | Sample weighting |
| **obs_ids** | qa.json | Link answers to scene graph nodes |
| **located_at_relations** | scene_graph.json | Graph attention edges |
| **obs_relations** | scene_graph.json | Relational reasoning |
| **patient.total_studies** | patient_metadata.parquet | Patient-level aggregation |
| **study.procedure** | study_metadata.parquet | Could be conditioning signal |

---

## 6. Model Input/Output Summary

### 6.1 Current Model Inputs (from `collate_fn`)

```python
batch = {
    # VISUAL INPUTS
    'images': (B, 3, 224, 224),           # ConvNeXt-Base input
    
    # TEXT INPUTS
    'input_ids': (B, 128),                 # Bio+ClinicalBERT tokens
    'attention_mask': (B, 128),            # Attention mask
    'token_type_ids': (B, 128),            # Token types
    
    # SCENE GRAPH INPUTS (per sample, variable length)
    'scene_graphs': List[Dict] len=B,      # Each dict contains:
        # {
        #   'bboxes': (N, 4),              # Normalized bboxes
        #   'region_ids': (N,),            # Region embedding indices
        #   'entity_ids': (N,),            # Entity embedding indices
        #   'positiveness': (N,),          # 0=neg, 1=pos findings
        #   'num_objects': int             # Number of observations
        # }
    
    # ROUTING
    'question_types': List[str] len=B,     # For head selection
    
    # TARGETS
    'answer_idx': (B,),                    # VQA answer class
    'chexpert_labels': (B, 14),            # Multi-label targets
    'chexpert_mask': (B, 14),              # Ignore uncertain labels
}
```

### 6.2 Model Outputs

```python
outputs = {
    # VQA HEAD OUTPUTS (only active head per question_type)
    'vqa_outputs': {
        'binary': (B, 2),     # For is_*, has_* questions
        'category': (B, 14),  # For describe_finding
        'region': (B, 26),    # For where_is_*, describe_region
        'severity': (B, 4),   # For how_severe
    },
    
    # AUXILIARY OUTPUT
    'chexpert_logits': (B, 14),  # Multi-label classification
    
    # OPTIONAL (for explainability)
    'attention_weights': (...),  # For visualization
}
```

---

## 7. Answer Head Mappings

### 7.1 Question Type → Head Routing

From your analysis data, here's how question types map to answer heads:

| Question Type | Primary Head | Answer Source | Count |
|--------------|--------------|---------------|-------|
| `C03_is_abnormal_region` | **binary** | `positiveness` | 124,434 |
| `C04_is_normal_region` | **binary** | `positiveness` (inverted) | 124,478 |
| `D02_has_finding` | **binary** | `positiveness` | 112,049 |
| `C08_has_region_device` | **binary** | `positiveness` | 206,215 |
| `D03_where_is_finding` | **region** | `regions[0]` | 88,990 |
| `C01_describe_region` | **region** | `regions[0]` | 124,312 |
| `D04_how_severe_is_finding` | **severity** | `modifiers[severity]` | 58,490 |
| `D01_describe_finding` | **category** | `obs_entities[0]` | 112,193 |
| `A_indication` | **category** | special handling | 9,645 |

### 7.2 Updated Question Type Map

```python
# Enhanced question type mapping with all MIMIC-Ext-CXR-QBA types
QUESTION_TYPE_MAP = {
    # Binary Head (Yes/No)
    'C03_is_abnormal_region': 'binary',
    'C04_is_normal_region': 'binary',
    'D02_has_finding': 'binary',
    'D06_has_device': 'binary',
    'C08_has_region_device': 'binary',
    'B10_is_abnormal_subcat': 'binary',
    'B11_is_normal_subcat': 'binary',
    'B13_has_devices': 'binary',
    
    # Region Head (Anatomical Location)
    'D03_where_is_finding': 'region',
    'D07_where_is_device': 'region',
    'C01_describe_region': 'region',
    'C02_describe_abnormal_region': 'region',
    
    # Severity Head
    'D04_how_severe_is_finding': 'severity',
    
    # Category Head (Finding Type / Entity)
    'D01_describe_finding': 'category',
    'D05_describe_device': 'category',
    'C07_describe_region_device': 'category',
    'B08_describe_subcat': 'category',
    'B09_describe_abnormal_subcat': 'category',
    'B12_describe_device': 'category',
    'A_indication': 'category',
    
    # Legacy mappings (backward compatibility)
    'is_abnormal': 'binary',
    'is_normal': 'binary',
    'has_finding': 'binary',
    'has_device': 'binary',
    'where_is_finding': 'region',
    'describe_region': 'region',
    'how_severe': 'severity',
    'describe_finding': 'category',
}
```

---

## 8. Data Flow Diagram: Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE MODEL DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ╔═══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                              INPUT LAYER                                          ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║  images (B,3,224,224)     input_ids (B,128)      scene_graphs [B dicts]          ║  │
│  ║         │                       │                       │                        ║  │
│  ║         ▼                       ▼                       ▼                        ║  │
│  ║  ┌─────────────┐         ┌─────────────┐         ┌─────────────────┐             ║  │
│  ║  │ ConvNeXt-   │         │ Bio+Clinical│         │ Scene Graph     │             ║  │
│  ║  │ Base        │         │ BERT        │         │ Encoder         │             ║  │
│  ║  │ (pretrained)│         │ (pretrained)│         │ (from scratch)  │             ║  │
│  ║  └──────┬──────┘         └──────┬──────┘         └────────┬────────┘             ║  │
│  ║         │                       │                         │                      ║  │
│  ║         ▼                       ▼                         ▼                      ║  │
│  ║  feature_map            text_features              scene_features                ║  │
│  ║  (B,1024,7,7)           (B,128,768)                (B,N,134)                     ║  │
│  ║                                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                          │
│  ╔═══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                           FEATURE PROCESSING                                      ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║  feature_map + bboxes ──► ROI Align ──► roi_features (B,N,512)                   ║  │
│  ║                                              │                                    ║  │
│  ║  scene_features + roi_features ──► concat ──► visual_scene (B,N,646)             ║  │
│  ║                                              │                                    ║  │
│  ║                         ┌────────────────────┴───────────────────┐               ║  │
│  ║                         │        Scene-Embedded Interaction       │               ║  │
│  ║                         │              Module (SIM)               │               ║  │
│  ║                         │                                         │               ║  │
│  ║       text_features ────►  Cross-Attention (2 layers)   ◄─── visual_scene        ║  │
│  ║                         │                                         │               ║  │
│  ║                         └────────────────────┬───────────────────┘               ║  │
│  ║                                              │                                    ║  │
│  ║                                              ▼                                    ║  │
│  ║                                      fused_output (B,768)                        ║  │
│  ║                                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                          │
│  ╔═══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                            OUTPUT HEADS                                           ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║                              fused_output (B,768)                                ║  │
│  ║                                      │                                            ║  │
│  ║         ┌──────────┬────────────────┼────────────────┬───────────┐               ║  │
│  ║         │          │                │                │           │               ║  │
│  ║         ▼          ▼                ▼                ▼           ▼               ║  │
│  ║  ┌─────────┐ ┌──────────┐   ┌─────────────┐  ┌──────────┐ ┌───────────┐         ║  │
│  ║  │ Binary  │ │ Category │   │   Region    │  │ Severity │ │ CheXpert  │         ║  │
│  ║  │  Head   │ │   Head   │   │    Head     │  │   Head   │ │   Head    │         ║  │
│  ║  └────┬────┘ └────┬─────┘   └──────┬──────┘  └────┬─────┘ └─────┬─────┘         ║  │
│  ║       │           │                │              │             │               ║  │
│  ║       ▼           ▼                ▼              ▼             ▼               ║  │
│  ║   (B, 2)      (B, 14)          (B, 26)        (B, 4)        (B, 14)             ║  │
│  ║   Yes/No      Entities         Regions      none/mild/    Multi-label          ║  │
│  ║                                              mod/severe                          ║  │
│  ║                                                                                   ║  │
│  ║  Head Selection: question_types[i] ──► QUESTION_TYPE_MAP ──► active head        ║  │
│  ║                                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                          │
│  ╔═══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                              LOSS COMPUTATION                                     ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║  VQA Loss (per sample, routed by question_type):                                 ║  │
│  ║  ─────────────────────────────────────────────────                               ║  │
│  ║  L_vqa = Σᵢ CrossEntropy(head_output[i], answer_idx[i])                          ║  │
│  ║          where head = QUESTION_TYPE_MAP[question_types[i]]                       ║  │
│  ║                                                                                   ║  │
│  ║  CheXpert Auxiliary Loss:                                                        ║  │
│  ║  ─────────────────────────                                                       ║  │
│  ║  L_chex = BCE(chexpert_logits, chexpert_labels) × chexpert_mask                  ║  │
│  ║                                                                                   ║  │
│  ║  Total Loss:                                                                      ║  │
│  ║  ───────────                                                                      ║  │
│  ║  L_total = L_vqa + 0.3 × L_chex                                                  ║  │
│  ║                                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Recommendations

### 9.1 High Priority Enhancements

1. **Add Answer Region Target**: Currently region head target is parsed from text. Should use `answers[].regions` field directly.

2. **Add Answer Entity Target**: For category head, use `answers[].obs_entities` directly for cleaner label mapping.

3. **Quality-Weighted Sampling**: Use `study.quality.extraction_rating` to weight samples during training.

### 9.2 Medium Priority Enhancements

4. **Link obs_ids to Scene Graph**: Use `qa.obs_ids` to connect answers to specific scene graph observations for better grounding.

5. **Question Strategy Balancing**: Use `question_strategy` for balanced sampling across question types.

### 9.3 Low Priority (Future)

6. **Graph Attention from Relations**: Use `located_at_relations` and `obs_relations` for graph neural network edges.

7. **Temporal Patient Modeling**: Use patient-level temporal data for longitudinal studies.

---

## 10. Summary: Metadata Coverage

| Category | Fields Available | Fields Used | Coverage |
|----------|-----------------|-------------|----------|
| **Images** | 7 | 4 | 57% |
| **CheXpert** | 16 | 16 | 100% |
| **Scene Graphs** | 12+ | 8 | 67% |
| **Questions** | 8 | 4 | 50% |
| **Answers** | 10+ | 4 | 40% |
| **Study Quality** | 8 | 2 | 25% |

**Overall**: ~60% of available metadata is currently being utilized.

The remaining 40% can be incorporated for:
- Better sampling strategies
- Enhanced answer label accuracy
- Graph-based reasoning improvements
- Quality-aware training

