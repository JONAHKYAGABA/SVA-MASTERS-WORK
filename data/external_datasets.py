"""
External Dataset Loaders for Cross-Dataset Evaluation

Implements loaders for:
- VQA-RAD: 315 images, 3,515 QA pairs (radiology-specific)
- SLAKE-EN: 701 images, ~14,000 QA pairs (knowledge-enhanced)

These datasets are used for zero-shot cross-dataset evaluation
as specified in methodology Section 16.3.

IMPORTANT: These datasets must be downloaded separately:
- VQA-RAD: https://osf.io/89kps/
- SLAKE: https://www.med-vqa.com/slake/

Directory structure expected:
    external_datasets/
    ├── vqa_rad/
    │   ├── images/
    │   │   ├── synpic100132.jpg
    │   │   └── ...
    │   ├── trainset.json
    │   ├── valset.json
    │   └── testset.json
    │
    └── slake/
        ├── imgs/
        │   ├── xmlab0/
        │   └── ...
        ├── train.json
        ├── validate.json
        └── test.json
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# VQA-RAD Dataset
# =============================================================================

class VQARADDataset(Dataset):
    """
    VQA-RAD Dataset Loader.
    
    VQA-RAD contains 315 radiology images with 3,515 QA pairs.
    Questions are categorized into:
    - Modality, Plane, Organ System (closed-ended)
    - Abnormality, Presence, Color, Size, etc. (open/closed)
    
    Reference: Lau et al., "A Dataset of Clinically Generated Visual Questions
               and Answers about Radiology Images" (2018)
    
    Download from: https://osf.io/89kps/
    """
    
    # Answer type mapping for VQA-RAD
    ANSWER_TYPES = {
        'CLOSED': 'binary',      # Yes/No questions
        'OPEN': 'category',      # Open-ended questions
    }
    
    # Question categories in VQA-RAD
    QUESTION_CATEGORIES = [
        'Modality', 'Plane', 'Organ', 'Abnormality', 'Object/Condition Presence',
        'Positional Reasoning', 'Color', 'Size', 'Attribute', 'Counting', 'Other'
    ]
    
    def __init__(
        self,
        data_path: str,
        split: str = 'test',
        tokenizer_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
        max_question_length: int = 128,
        image_size: int = 224,
        transform: Optional[Any] = None
    ):
        """
        Args:
            data_path: Path to VQA-RAD dataset directory
            split: 'train', 'val', or 'test'
            tokenizer_name: HuggingFace tokenizer for question encoding
            max_question_length: Maximum question token length
            image_size: Target image size for transforms
            transform: Optional custom transforms
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_question_length = max_question_length
        self.image_size = image_size
        
        # Initialize tokenizer
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            raise ImportError("transformers required for VQARADDataset")
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Load QA data
        self.samples = self._load_samples()
        logger.info(f"VQA-RAD {split}: Loaded {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load QA pairs from JSON files."""
        samples = []
        
        # Map split names to file names
        split_map = {
            'train': 'trainset.json',
            'val': 'valset.json',
            'validation': 'valset.json',
            'test': 'testset.json'
        }
        
        json_file = self.data_path / split_map.get(self.split, f'{self.split}set.json')
        
        if not json_file.exists():
            logger.error(f"VQA-RAD data file not found: {json_file}")
            logger.error("=" * 60)
            logger.error("MISSING DATASET: VQA-RAD")
            logger.error("=" * 60)
            logger.error("Download from: https://osf.io/89kps/")
            logger.error(f"Extract to: {self.data_path}")
            logger.error("Expected structure:")
            logger.error("  vqa_rad/")
            logger.error("  ├── images/")
            logger.error("  │   ├── synpic100132.jpg")
            logger.error("  │   └── ...")
            logger.error("  ├── trainset.json")
            logger.error("  ├── valset.json")
            logger.error("  └── testset.json")
            logger.error("=" * 60)
            return []
        
        with open(json_file) as f:
            data = json.load(f)
        
        # VQA-RAD format: list of QA entries
        for entry in data:
            image_name = entry.get('image_name', entry.get('image', ''))
            image_path = self.data_path / 'images' / image_name
            
            if not image_path.exists():
                # Try alternative paths
                for alt_dir in ['VQA_RAD Image Folder', 'Images', 'imgs']:
                    alt_path = self.data_path / alt_dir / image_name
                    if alt_path.exists():
                        image_path = alt_path
                        break
            
            samples.append({
                'image_path': str(image_path),
                'question': entry.get('question', ''),
                'answer': entry.get('answer', ''),
                'answer_type': entry.get('answer_type', 'OPEN'),
                'question_type': entry.get('question_type', 'Other'),
                'phrase_type': entry.get('phrase_type', ''),
                'image_organ': entry.get('image_organ', ''),
            })
        
        return samples
    
    def _get_answer_label(self, answer: str, answer_type: str) -> int:
        """Convert answer to classification label."""
        answer_lower = answer.lower().strip()
        
        if answer_type == 'CLOSED':
            # Binary: Yes/No
            if answer_lower in ['yes', 'true', '1']:
                return 1
            else:
                return 0
        else:
            # For open-ended, we'd need a vocabulary
            # For now, return 0 (evaluation uses text matching)
            return 0
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            original_size = image.size
        except Exception as e:
            logger.warning(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
            original_size = (self.image_size, self.image_size)
        
        image_tensor = self.transform(image)
        
        # Tokenize question
        question_inputs = self.tokenizer(
            sample['question'],
            padding='max_length',
            truncation=True,
            max_length=self.max_question_length,
            return_tensors='pt'
        )
        
        # Get answer label
        answer_idx = self._get_answer_label(sample['answer'], sample['answer_type'])
        
        # Map answer type to head type
        head_type = self.ANSWER_TYPES.get(sample['answer_type'], 'category')
        
        return {
            'images': image_tensor,
            'input_ids': question_inputs['input_ids'].squeeze(0),
            'attention_mask': question_inputs['attention_mask'].squeeze(0),
            'token_type_ids': question_inputs.get('token_type_ids', torch.zeros_like(question_inputs['input_ids'])).squeeze(0),
            'question': sample['question'],
            'answer': sample['answer'],
            'answer_idx': torch.tensor(answer_idx, dtype=torch.long),
            'answer_type': sample['answer_type'],
            'question_type': sample['question_type'],
            'head_type': head_type,
            'image_width': torch.tensor(original_size[0], dtype=torch.float),
            'image_height': torch.tensor(original_size[1], dtype=torch.float),
            # No scene graphs for VQA-RAD (zero-shot evaluation)
            'scene_graphs': {
                'bboxes': np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                'region_ids': np.array([0], dtype=np.int64),
                'entity_ids': np.array([0], dtype=np.int64),
                'positiveness': np.array([0], dtype=np.int64),
                'num_objects': 1
            }
        }


# =============================================================================
# SLAKE Dataset
# =============================================================================

class SLAKEDataset(Dataset):
    """
    SLAKE (Semantically-Labeled Knowledge-Enhanced) Dataset Loader.
    
    SLAKE contains 701 radiology images with ~14,000 QA pairs.
    Provides semantic labels for knowledge-enhanced assessment.
    
    Reference: Liu et al., "SLAKE: A Semantically-Labeled Knowledge-Enhanced
               Dataset for Medical Visual Question Answering" (2021)
    
    Download from: https://www.med-vqa.com/slake/
    """
    
    # Question types in SLAKE
    QUESTION_TYPES = [
        'Plane', 'Modality', 'Organ', 'KG',  # Knowledge Graph questions
        'Position', 'Abnormality', 'Color', 'Shape', 'Size', 'Quantity'
    ]
    
    def __init__(
        self,
        data_path: str,
        split: str = 'test',
        tokenizer_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
        max_question_length: int = 128,
        image_size: int = 224,
        language: str = 'en',  # 'en' or 'zh'
        transform: Optional[Any] = None
    ):
        """
        Args:
            data_path: Path to SLAKE dataset directory
            split: 'train', 'validate', or 'test'
            tokenizer_name: HuggingFace tokenizer for question encoding
            max_question_length: Maximum question token length
            image_size: Target image size for transforms
            language: 'en' for English, 'zh' for Chinese
            transform: Optional custom transforms
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_question_length = max_question_length
        self.image_size = image_size
        self.language = language
        
        # Initialize tokenizer
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            raise ImportError("transformers required for SLAKEDataset")
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Load QA data
        self.samples = self._load_samples()
        logger.info(f"SLAKE {split} ({language}): Loaded {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load QA pairs from JSON files."""
        samples = []
        
        # Map split names
        split_map = {
            'train': 'train.json',
            'val': 'validate.json',
            'validation': 'validate.json',
            'test': 'test.json'
        }
        
        json_file = self.data_path / split_map.get(self.split, f'{self.split}.json')
        
        if not json_file.exists():
            logger.error(f"SLAKE data file not found: {json_file}")
            logger.error("=" * 60)
            logger.error("MISSING DATASET: SLAKE")
            logger.error("=" * 60)
            logger.error("Download from: https://www.med-vqa.com/slake/")
            logger.error(f"Extract to: {self.data_path}")
            logger.error("Expected structure:")
            logger.error("  slake/")
            logger.error("  ├── imgs/")
            logger.error("  │   ├── xmlab0/")
            logger.error("  │   │   └── source.jpg")
            logger.error("  │   └── ...")
            logger.error("  ├── train.json")
            logger.error("  ├── validate.json")
            logger.error("  └── test.json")
            logger.error("=" * 60)
            return []
        
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        
        # SLAKE format: list of QA entries with image paths
        for entry in data:
            # Filter by language
            if self.language == 'en' and entry.get('q_lang', 'en') != 'en':
                continue
            if self.language == 'zh' and entry.get('q_lang', 'en') != 'zh':
                continue
            
            img_name = entry.get('img_name', entry.get('image', ''))
            image_path = self.data_path / 'imgs' / img_name
            
            if not image_path.exists():
                # Try with img_id
                img_id = entry.get('img_id', '')
                if img_id:
                    image_path = self.data_path / 'imgs' / img_id / 'source.jpg'
            
            samples.append({
                'image_path': str(image_path),
                'question': entry.get('question', ''),
                'answer': str(entry.get('answer', '')),
                'answer_type': entry.get('answer_type', 'OPEN'),
                'question_type': entry.get('content_type', entry.get('q_type', 'Other')),
                'img_id': entry.get('img_id', ''),
                'modality': entry.get('modality', ''),
                'organ': entry.get('location', ''),  # anatomical location
            })
        
        return samples
    
    def _get_answer_label(self, answer: str, answer_type: str) -> int:
        """Convert answer to classification label."""
        answer_lower = answer.lower().strip()
        
        if answer_type == 'CLOSED':
            if answer_lower in ['yes', 'true', '1']:
                return 1
            else:
                return 0
        return 0
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            original_size = image.size
        except Exception as e:
            logger.warning(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
            original_size = (self.image_size, self.image_size)
        
        image_tensor = self.transform(image)
        
        # Tokenize question
        question_inputs = self.tokenizer(
            sample['question'],
            padding='max_length',
            truncation=True,
            max_length=self.max_question_length,
            return_tensors='pt'
        )
        
        # Get answer label
        answer_idx = self._get_answer_label(sample['answer'], sample['answer_type'])
        
        # Map to head type
        head_type = 'binary' if sample['answer_type'] == 'CLOSED' else 'category'
        
        return {
            'images': image_tensor,
            'input_ids': question_inputs['input_ids'].squeeze(0),
            'attention_mask': question_inputs['attention_mask'].squeeze(0),
            'token_type_ids': question_inputs.get('token_type_ids', torch.zeros_like(question_inputs['input_ids'])).squeeze(0),
            'question': sample['question'],
            'answer': sample['answer'],
            'answer_idx': torch.tensor(answer_idx, dtype=torch.long),
            'answer_type': sample['answer_type'],
            'question_type': sample['question_type'],
            'head_type': head_type,
            'image_width': torch.tensor(original_size[0], dtype=torch.float),
            'image_height': torch.tensor(original_size[1], dtype=torch.float),
            'modality': sample.get('modality', ''),
            'organ': sample.get('organ', ''),
            # No scene graphs for SLAKE (zero-shot evaluation)
            'scene_graphs': {
                'bboxes': np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                'region_ids': np.array([0], dtype=np.int64),
                'entity_ids': np.array([0], dtype=np.int64),
                'positiveness': np.array([0], dtype=np.int64),
                'num_objects': 1
            }
        }


# =============================================================================
# Unified External Dataset Loader
# =============================================================================

def get_external_dataset(
    dataset_name: str,
    data_path: str,
    split: str = 'test',
    **kwargs
) -> Dataset:
    """
    Factory function to get external dataset by name.
    
    Args:
        dataset_name: 'vqa_rad' or 'slake'
        data_path: Path to dataset directory
        split: Dataset split
        **kwargs: Additional arguments for dataset class
    
    Returns:
        Dataset instance
    """
    dataset_map = {
        'vqa_rad': VQARADDataset,
        'vqarad': VQARADDataset,
        'slake': SLAKEDataset,
        'slake_en': SLAKEDataset,
    }
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")
    
    return dataset_map[dataset_name.lower()](data_path, split, **kwargs)


def external_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for external datasets."""
    
    # Stack tensors
    images = torch.stack([item['images'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    answer_idx = torch.stack([item['answer_idx'] for item in batch])
    image_widths = torch.stack([item['image_width'] for item in batch])
    image_heights = torch.stack([item['image_height'] for item in batch])
    
    # Collect scene graphs and metadata
    scene_graphs = [item['scene_graphs'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    answer_types = [item['answer_type'] for item in batch]
    question_types = [item['question_type'] for item in batch]
    head_types = [item['head_type'] for item in batch]
    
    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'scene_graphs': scene_graphs,
        'answer_idx': answer_idx,
        'questions': questions,
        'answers': answers,
        'answer_types': answer_types,
        'question_types': question_types,
        'head_types': head_types,
        'image_widths': image_widths,
        'image_heights': image_heights,
    }


def create_external_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = False
) -> DataLoader:
    """Create DataLoader for external datasets."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=external_collate_fn
    )


# =============================================================================
# Dataset Availability Check
# =============================================================================

def check_external_datasets(base_path: str = './external_datasets') -> Dict[str, bool]:
    """
    Check which external datasets are available.
    
    Returns:
        Dict mapping dataset name to availability status
    """
    base_path = Path(base_path)
    
    availability = {
        'vqa_rad': False,
        'slake': False,
    }
    
    # Check VQA-RAD
    vqa_rad_path = base_path / 'vqa_rad'
    if vqa_rad_path.exists():
        required_files = ['testset.json']
        required_dirs = ['images']
        
        files_exist = all((vqa_rad_path / f).exists() for f in required_files)
        dirs_exist = any((vqa_rad_path / d).exists() for d in required_dirs + ['VQA_RAD Image Folder', 'Images'])
        
        availability['vqa_rad'] = files_exist and dirs_exist
    
    # Check SLAKE
    slake_path = base_path / 'slake'
    if slake_path.exists():
        required_files = ['test.json']
        required_dirs = ['imgs']
        
        files_exist = all((slake_path / f).exists() for f in required_files)
        dirs_exist = all((slake_path / d).exists() for d in required_dirs)
        
        availability['slake'] = files_exist and dirs_exist
    
    # Print status
    print("\n" + "=" * 60)
    print("EXTERNAL DATASETS STATUS")
    print("=" * 60)
    
    for name, available in availability.items():
        status = "✅ AVAILABLE" if available else "❌ MISSING"
        print(f"  {name.upper()}: {status}")
    
    if not all(availability.values()):
        print("\n" + "-" * 60)
        print("DOWNLOAD INSTRUCTIONS:")
        print("-" * 60)
        
        if not availability['vqa_rad']:
            print("\n📥 VQA-RAD:")
            print("   URL: https://osf.io/89kps/")
            print(f"   Extract to: {base_path / 'vqa_rad'}")
        
        if not availability['slake']:
            print("\n📥 SLAKE:")
            print("   URL: https://www.med-vqa.com/slake/")
            print(f"   Extract to: {base_path / 'slake'}")
    
    print("=" * 60 + "\n")
    
    return availability


if __name__ == '__main__':
    # Check dataset availability
    check_external_datasets()

