"""
Unit tests for MIMIC-CXR VQA Model components.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelComponents:
    """Test individual model components."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.batch_size = 2
        self.image_size = 224
        self.hidden_dim = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_convnext_feature_extractor(self):
        """Test visual backbone initialization and forward pass."""
        from models.mimic_vqa_model import ConvNeXtFeatureExtractor
        
        model = ConvNeXtFeatureExtractor(
            model_name='convnext_base',
            pretrained=False,  # Don't download for testing
            output_dim=self.hidden_dim
        ).to(self.device)
        
        # Create dummy input
        images = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            features = model(images)
        
        # Check output shape
        assert features.shape == (self.batch_size, self.hidden_dim), \
            f"Expected shape ({self.batch_size}, {self.hidden_dim}), got {features.shape}"
    
    def test_scene_graph_encoder(self):
        """Test scene graph encoder."""
        from models.mimic_vqa_model import SceneGraphEncoderExpanded
        
        model = SceneGraphEncoderExpanded(
            num_regions=310,
            num_entities=237,
            region_embed_dim=64,
            entity_embed_dim=64,
            output_dim=self.hidden_dim
        ).to(self.device)
        
        # Create dummy scene graph features
        scene_graphs = [{
            'bboxes': torch.randn(5, 4).numpy(),
            'region_ids': torch.randint(0, 310, (5,)).numpy(),
            'entity_ids': torch.randint(0, 237, (5,)).numpy(),
            'positiveness': torch.randint(0, 2, (5,)).numpy(),
            'num_objects': 5
        } for _ in range(self.batch_size)]
        
        # Forward pass
        with torch.no_grad():
            features = model(scene_graphs, self.device)
        
        # Check output shape
        assert features.shape == (self.batch_size, self.hidden_dim), \
            f"Expected shape ({self.batch_size}, {self.hidden_dim}), got {features.shape}"
    
    def test_multi_head_answer_module(self):
        """Test multi-head answer module."""
        from models.mimic_vqa_model import MultiHeadAnswerModule
        
        model = MultiHeadAnswerModule(
            hidden_dim=self.hidden_dim,
            num_binary_classes=2,
            num_category_classes=14,
            num_region_classes=26,
            num_severity_classes=4
        ).to(self.device)
        
        # Create dummy features
        features = torch.randn(self.batch_size, self.hidden_dim).to(self.device)
        question_types = ['is_abnormal', 'describe_finding']
        
        # Forward pass
        with torch.no_grad():
            outputs = model(features, question_types)
        
        # Check outputs
        assert 'binary' in outputs
        assert 'category' in outputs
        assert outputs['binary'].shape == (self.batch_size, 2)
        assert outputs['category'].shape == (self.batch_size, 14)


class TestDataset:
    """Test dataset components."""
    
    def test_question_type_mapping(self):
        """Test question type to answer head mapping."""
        from data.mimic_cxr_dataset import QUESTION_TYPE_MAP
        
        # Check expected mappings
        assert QUESTION_TYPE_MAP['is_abnormal'] == 'binary'
        assert QUESTION_TYPE_MAP['describe_finding'] == 'category'
        assert QUESTION_TYPE_MAP['where_is_finding'] == 'region'
        assert QUESTION_TYPE_MAP['how_severe'] == 'severity'


class TestLoss:
    """Test loss functions."""
    
    def test_multi_task_loss(self):
        """Test multi-task loss computation."""
        from training.loss import MultiTaskLoss
        
        loss_fn = MultiTaskLoss(
            vqa_weights={'binary': 1.0, 'category': 1.0, 'region': 1.0, 'severity': 1.0},
            chexpert_weight=0.5
        )
        
        batch_size = 4
        
        # Create dummy model outputs
        model_outputs = {
            'vqa_outputs': {
                'binary': torch.randn(batch_size, 2),
                'category': torch.randn(batch_size, 14),
            },
            'chexpert_logits': torch.randn(batch_size, 14)
        }
        
        # Create dummy targets
        vqa_targets = {
            'binary': torch.randint(0, 2, (batch_size,)),
            'category': torch.full((batch_size,), -1, dtype=torch.long),  # Ignored
        }
        chexpert_targets = torch.randint(0, 2, (batch_size, 14)).float()
        chexpert_mask = torch.ones(batch_size, 14)
        
        # Compute loss
        loss, loss_details = loss_fn(
            model_outputs=model_outputs,
            vqa_targets=vqa_targets,
            chexpert_targets=chexpert_targets,
            chexpert_mask=chexpert_mask
        )
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert 'vqa_loss' in loss_details
        assert 'chexpert_loss' in loss_details


class TestHardwareUtils:
    """Test hardware detection utilities."""
    
    def test_hardware_detection(self):
        """Test hardware info detection."""
        from utils.hardware_utils import detect_hardware, HardwareInfo
        
        info = detect_hardware()
        
        assert isinstance(info, HardwareInfo)
        assert info.num_cpus >= 1
        assert info.optimal_batch_size >= 1
        assert info.optimal_grad_accum >= 1
    
    def test_optimal_settings_calculation(self):
        """Test optimal settings are reasonable."""
        from utils.hardware_utils import detect_hardware
        
        info = detect_hardware()
        
        # Effective batch should be reasonable
        effective_batch = info.optimal_batch_size * max(1, info.num_gpus) * info.optimal_grad_accum
        assert 16 <= effective_batch <= 512, f"Effective batch {effective_batch} out of expected range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

