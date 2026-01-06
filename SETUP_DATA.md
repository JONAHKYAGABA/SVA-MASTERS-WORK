# Dataset Setup Guide

## Required Datasets

You need two datasets from PhysioNet (requires credentialed access):

### 1. MIMIC-CXR-JPG 2.1.0
**Source:** https://physionet.org/content/mimic-cxr-jpg/2.1.0/

Download via Google Cloud or wget:
```bash
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

### 2. MIMIC-Ext-CXR-QBA 1.0.0
**Source:** https://physionet.org/content/mimic-ext-cxr-qba/1.0.0/

Download via wget:
```bash
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimic-ext-cxr-qba/1.0.0/
```

---

## Expected File Structure After Download

```
📁 MIMIC-CXR-JPG/
├── 📁 files/
│   ├── 📁 p10/
│   │   ├── 📁 p10000032/
│   │   │   ├── 📁 s50414267/
│   │   │   │   ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
│   │   │   │   └── 174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg
│   │   │   ...
│   ...
├── mimic-cxr-2.0.0-chexpert.csv.gz
├── mimic-cxr-2.0.0-metadata.csv.gz
├── mimic-cxr-2.0.0-split.csv.gz
└── mimic-cxr-2.1.0-test-set-labeled.csv

📁 MIMIC-Ext-CXR-QBA/
├── 📁 exports/
├── 📁 metadata/
├── 📁 stats/
├── qa.zip                  ← NEEDS EXTRACTION! (6.9 GB)
├── scene_data.zip          ← NEEDS EXTRACTION! (1.1 GB)
├── quality_mappings.csv
├── LICENSE.txt
└── README.md
```

---

## ⚠️ IMPORTANT: Extract ZIP Files!

The MIMIC-Ext-CXR-QBA dataset has two ZIP files that **MUST** be extracted:

### Windows (PowerShell)

```powershell
cd "C:\path\to\MIMIC-Ext-CXR-QBA"

# Extract QA pairs (6.9 GB → ~25 GB extracted)
Expand-Archive -Path "qa.zip" -DestinationPath "."

# Extract Scene graphs (1.1 GB → ~4 GB extracted)  
Expand-Archive -Path "scene_data.zip" -DestinationPath "."
```

### Linux/Mac

```bash
cd /path/to/MIMIC-Ext-CXR-QBA

# Extract QA pairs
unzip qa.zip

# Extract Scene graphs
unzip scene_data.zip
```

---

## File Structure After Extraction

```
📁 MIMIC-Ext-CXR-QBA/
├── 📁 qa/                     ← EXTRACTED from qa.zip
│   ├── 📁 p10/
│   │   ├── 📁 p10000032/
│   │   │   ├── s50414267.qa.json
│   │   │   └── ...
│   ...
├── 📁 scene_data/             ← EXTRACTED from scene_data.zip
│   ├── 📁 p10/
│   │   ├── 📁 p10000032/
│   │   │   ├── s50414267.scene_graph.json
│   │   │   └── ...
│   ...
├── 📁 exports/
├── 📁 metadata/
├── 📁 stats/
├── qa.zip
├── scene_data.zip
├── quality_mappings.csv
└── ...
```

---

## Update Configuration

Edit `configs/default_config.yaml`:

```yaml
data:
  # Update these paths to YOUR local paths!
  mimic_cxr_jpg_path: "C:/path/to/MIMIC-CXR-JPG"      # Windows
  mimic_ext_cxr_qba_path: "C:/path/to/MIMIC-Ext-CXR-QBA"
  
  # Or for Linux/Mac:
  # mimic_cxr_jpg_path: "/data/MIMIC-CXR-JPG"
  # mimic_ext_cxr_qba_path: "/data/MIMIC-Ext-CXR-QBA"
```

---

## Verify Setup

Run the data analysis script to verify everything is ready:

```bash
python analyze_data.py \
    --mimic_cxr_path "C:/path/to/MIMIC-CXR-JPG" \
    --mimic_qa_path "C:/path/to/MIMIC-Ext-CXR-QBA"
```

### Expected Output (if successful):

```
[1/7] Validating dataset paths...
✓ MIMIC-CXR path found
✓ MIMIC-Ext-CXR-QBA path found
✓ Images directory found
✓ QA directory found
✓ Scene data directory found

...

============================================================
✅ DATA IS READY FOR TRAINING
   Run: python train_mimic_cxr.py --config configs/default_config.yaml
============================================================
```

### If you see this error:

```
✗ qa.zip found but NOT EXTRACTED!
  Please extract qa.zip first
```

➡️ You need to extract the ZIP files (see instructions above)

---

## Quick Checklist

- [ ] Downloaded MIMIC-CXR-JPG from PhysioNet
- [ ] Downloaded MIMIC-Ext-CXR-QBA from PhysioNet  
- [ ] Extracted `qa.zip` → `qa/` folder
- [ ] Extracted `scene_data.zip` → `scene_data/` folder
- [ ] Updated paths in `configs/default_config.yaml`
- [ ] Ran `python analyze_data.py` successfully

---

## Storage Requirements

| Dataset | Download | Extracted |
|---------|----------|-----------|
| MIMIC-CXR-JPG | ~500 GB | ~500 GB |
| MIMIC-Ext-CXR-QBA qa.zip | 6.9 GB | ~25 GB |
| MIMIC-Ext-CXR-QBA scene_data.zip | 1.1 GB | ~4 GB |
| **Total** | **~510 GB** | **~530 GB** |

---

## Next Steps

Once data is ready:

```bash
# 1. Verify data
python analyze_data.py --mimic_cxr_path /path/to/images --mimic_qa_path /path/to/qa

# 2. Train model  
python train_mimic_cxr.py --config configs/default_config.yaml

# 3. Evaluate
python evaluate.py --model_path ./checkpoints/best_model
```

