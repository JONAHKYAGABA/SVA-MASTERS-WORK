# Examples

This directory contains example scripts and notebooks for using the MIMIC-CXR VQA model.

## Available Examples

### 1. Inference Example (`inference_example.py`)

Demonstrates how to load a trained model and run inference on a chest X-ray image.

```bash
# Basic usage
python examples/inference_example.py \
    --model_path ./checkpoints/best_model \
    --image_path /path/to/chest_xray.jpg \
    --question "Is there any abnormality visible?"

# With specific device
python examples/inference_example.py \
    --model_path ./checkpoints/best_model \
    --image_path /path/to/chest_xray.jpg \
    --question "What findings are present?" \
    --device cuda
```

**Example Output:**
```
Using device: cuda
Loading model from: ./checkpoints/best_model
Processing image: chest_xray.jpg
Question: Is there any abnormality visible?

============================================================
PREDICTION RESULTS
============================================================
Question: Is there any abnormality visible?
Question Type: is_abnormal
Answer: Yes
Confidence: 87.3%

CheXpert Findings (probability):
  Lung Opacity: 72.1%
  Atelectasis: 45.3%
  Pleural Effusion: 23.8%
  Cardiomegaly: 12.4%
  Consolidation: 8.2%
============================================================
```

## Question Types

The model supports different types of questions:

| Question Type | Description | Example Questions |
|---------------|-------------|-------------------|
| `is_abnormal` | Binary yes/no | "Is there any abnormality?", "Is the image normal?" |
| `describe_finding` | Category | "What finding is present?", "What is the diagnosis?" |
| `where_is_finding` | Region | "Where is the opacity located?", "Which region shows abnormality?" |
| `how_severe` | Severity | "How severe is the finding?", "What is the severity level?" |

## Creating Your Own Examples

1. **Load the model:**
```python
from models.mimic_vqa_model import MIMICCXRVQAModel
from configs import load_config_from_file

config = load_config_from_file('checkpoints/best_model/config.json')
model = MIMICCXRVQAModel(config)
model.load_state_dict(torch.load('checkpoints/best_model/pytorch_model.bin'))
model.eval()
```

2. **Preprocess image:**
```python
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('chest_xray.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)
```

3. **Run inference:**
```python
outputs = model(
    images=image_tensor,
    input_ids=tokenized_question.input_ids,
    attention_mask=tokenized_question.attention_mask,
    # ... other inputs
)
```

## Notebooks (Coming Soon)

- `training_visualization.ipynb` - Visualize training metrics
- `attention_analysis.ipynb` - Analyze model attention patterns
- `error_analysis.ipynb` - Detailed error analysis

