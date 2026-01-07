"""
Multi-Task Loss Functions for MIMIC-CXR VQA

Combines:
- VQA multi-head loss (binary, category, region, severity)
- CheXpert auxiliary classification loss

Based on MIMIC_CXR_VQA_ANALYSIS.md specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any


class MultiTaskLoss(nn.Module):
    """
    Combined loss for VQA + CheXpert auxiliary task.
    
    Supports:
    - Binary head: Yes/No questions
    - Category head: Finding classification
    - Region head: Anatomical region prediction
    - Severity head: None/Mild/Moderate/Severe
    - CheXpert auxiliary: Multi-label classification for 14 conditions
    """
    
    def __init__(
        self,
        vqa_weight: float = 1.0,
        chexpert_weight: float = 0.3,
        binary_weight: float = 1.0,
        category_weight: float = 0.5,
        region_weight: float = 0.5,
        severity_weight: float = 0.3,
        ignore_index: int = -1,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.vqa_weight = vqa_weight
        self.chexpert_weight = chexpert_weight
        
        self.head_weights = {
            'binary': binary_weight,
            'category': category_weight,
            'region': region_weight,
            'severity': severity_weight,
        }
        
        self.ignore_index = ignore_index
        
        # Cross-entropy for classification heads
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
        # BCE for multi-label CheXpert
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self,
        outputs: Any,
        vqa_targets: Dict[str, torch.Tensor],
        chexpert_labels: torch.Tensor,
        chexpert_mask: torch.Tensor,
        question_types: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs with vqa_logits and chexpert_logits
            vqa_targets: Dict mapping head names to target tensors
            chexpert_labels: (B, 14) CheXpert ground truth
            chexpert_mask: (B, 14) mask for valid labels
            question_types: List of question types for routing
            
        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary with individual loss components
        """
        device = chexpert_labels.device
        loss_dict = {}
        
        # Get outputs
        vqa_logits = outputs.vqa_logits if hasattr(outputs, 'vqa_logits') else outputs.get('vqa_logits', {})
        chexpert_logits = outputs.chexpert_logits if hasattr(outputs, 'chexpert_logits') else outputs.get('chexpert_logits', None)
        
        # VQA Loss
        total_vqa_loss = torch.tensor(0.0, device=device)
        vqa_head_losses = {}
        
        # Determine which heads to use based on question types
        head_to_indices = self._get_head_indices(question_types)
        
        for head_name, indices in head_to_indices.items():
            if not indices or head_name not in vqa_logits:
                continue
            
            indices_tensor = torch.tensor(indices, device=device)
            
            # Get logits and targets for this head
            head_logits = vqa_logits[head_name]
            
            if head_name in vqa_targets:
                head_targets = vqa_targets[head_name]
                
                # Select only relevant samples
                if len(indices) < head_logits.shape[0]:
                    head_logits = head_logits[indices_tensor]
                    head_targets = head_targets[indices_tensor]
                
                # Compute loss
                head_loss = self.ce_loss(head_logits, head_targets)
                
                if not torch.isnan(head_loss):
                    vqa_head_losses[head_name] = head_loss
                    total_vqa_loss = total_vqa_loss + self.head_weights.get(head_name, 1.0) * head_loss
        
        loss_dict['vqa_loss'] = total_vqa_loss
        loss_dict.update({f'vqa_{k}_loss': v for k, v in vqa_head_losses.items()})
        
        # CheXpert Loss
        chexpert_loss = torch.tensor(0.0, device=device)
        
        if chexpert_logits is not None and chexpert_labels is not None:
            # Compute BCE loss
            raw_loss = self.bce_loss(chexpert_logits, chexpert_labels)
            
            # Apply mask (ignore uncertain labels)
            masked_loss = raw_loss * chexpert_mask
            
            # Mean over valid labels
            valid_count = chexpert_mask.sum()
            if valid_count > 0:
                chexpert_loss = masked_loss.sum() / valid_count
            
        loss_dict['chexpert_loss'] = chexpert_loss
        
        # Total loss
        total_loss = self.vqa_weight * total_vqa_loss + self.chexpert_weight * chexpert_loss
        
        return total_loss, loss_dict
    
    def _get_head_indices(self, question_types: List[str]) -> Dict[str, List[int]]:
        """
        Map question types to answer heads.
        
        Uses the same QUESTION_TYPE_MAP as the dataset to ensure consistency.
        Returns dict mapping head names to sample indices.
        """
        # Import the authoritative mapping from the dataset module
        try:
            from data.mimic_cxr_dataset import QUESTION_TYPE_MAP
        except ImportError:
            # Fallback mapping if import fails
            QUESTION_TYPE_MAP = self._get_fallback_question_type_map()
        
        head_indices = {
            'binary': [],
            'category': [],
            'region': [],
            'severity': [],
        }
        
        for idx, q_type in enumerate(question_types):
            # Look up head from the mapping
            head = QUESTION_TYPE_MAP.get(q_type)
            
            # If not found, try to infer from question type name
            if head is None:
                q_lower = q_type.lower()
                if any(x in q_lower for x in ['is_abnormal', 'is_normal', 'has_']):
                    head = 'binary'
                elif any(x in q_lower for x in ['where_is', 'describe_region']):
                    head = 'region'
                elif 'severe' in q_lower:
                    head = 'severity'
                elif any(x in q_lower for x in ['describe_', 'indication']):
                    head = 'category'
                else:
                    head = 'binary'  # Default
            
            if head in head_indices:
                head_indices[head].append(idx)
        
        return head_indices
    
    def _get_fallback_question_type_map(self) -> Dict[str, str]:
        """Fallback question type mapping if import fails."""
        return {
            # MIMIC-Ext-CXR-QBA Question Types
            'C03_is_abnormal_region': 'binary',
            'C04_is_normal_region': 'binary',
            'C08_has_region_device': 'binary',
            'D02_has_finding': 'binary',
            'D06_has_device': 'binary',
            'B10_is_abnormal_subcat': 'binary',
            'B11_is_normal_subcat': 'binary',
            'B13_has_devices': 'binary',
            'C01_describe_region': 'region',
            'C02_describe_abnormal_region': 'region',
            'D03_where_is_finding': 'region',
            'D07_where_is_device': 'region',
            'D04_how_severe_is_finding': 'severity',
            'D01_describe_finding': 'category',
            'D05_describe_device': 'category',
            'C07_describe_region_device': 'category',
            'B08_describe_subcat': 'category',
            'B09_describe_abnormal_subcat': 'category',
            'B12_describe_device': 'category',
            'A_indication': 'category',
            # Legacy mappings
            'is_abnormal': 'binary',
            'is_normal': 'binary',
            'has_finding': 'binary',
            'has_device': 'binary',
            'describe_finding': 'category',
            'where_is_finding': 'region',
            'describe_region': 'region',
            'how_severe': 'severity',
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification (CheXpert).
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        
        if mask is not None:
            focal = focal * mask
            if self.reduction == 'mean':
                return focal.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return focal.sum()
            else:
                return focal
        else:
            if self.reduction == 'mean':
                return focal.mean()
            elif self.reduction == 'sum':
                return focal.sum()
            else:
                return focal
