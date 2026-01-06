"""
Evaluation Metrics for MIMIC-CXR VQA

Computes metrics for:
- Multi-head VQA (accuracy, F1 per head)
- CheXpert classification (AUROC, F1)

Based on MIMIC_CXR_VQA_ANALYSIS.md specifications.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)


class VQAMetrics:
    """
    Aggregates and computes metrics for VQA evaluation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self.predictions = defaultdict(list)
        self.targets = defaultdict(list)
        self.chexpert_preds = []
        self.chexpert_labels = []
        self.chexpert_masks = []
        self.question_types = []
    
    def update(
        self,
        outputs: Any,
        vqa_targets: Dict[str, torch.Tensor],
        chexpert_labels: torch.Tensor,
        chexpert_mask: torch.Tensor,
        question_types: List[str]
    ):
        """
        Update metrics with a batch of predictions.
        """
        # Get VQA logits
        vqa_logits = outputs.vqa_logits if hasattr(outputs, 'vqa_logits') else outputs.get('vqa_logits', {})
        chexpert_logits = outputs.chexpert_logits if hasattr(outputs, 'chexpert_logits') else outputs.get('chexpert_logits', None)
        
        # Store question types
        self.question_types.extend(question_types)
        
        # Map question types to heads
        head_indices = self._get_head_indices(question_types)
        
        # Store VQA predictions per head
        for head_name, indices in head_indices.items():
            if not indices or head_name not in vqa_logits:
                continue
            
            head_logits = vqa_logits[head_name]
            
            # Get predictions
            if len(indices) < head_logits.shape[0]:
                indices_tensor = torch.tensor(indices, device=head_logits.device)
                head_logits = head_logits[indices_tensor]
            
            preds = head_logits.argmax(dim=-1).cpu().numpy()
            self.predictions[head_name].extend(preds.tolist())
            
            # Get targets
            if head_name in vqa_targets:
                head_targets = vqa_targets[head_name]
                if len(indices) < head_targets.shape[0]:
                    indices_tensor = torch.tensor(indices, device=head_targets.device)
                    head_targets = head_targets[indices_tensor]
                targets = head_targets.cpu().numpy()
                self.targets[head_name].extend(targets.tolist())
        
        # Store CheXpert predictions
        if chexpert_logits is not None:
            probs = torch.sigmoid(chexpert_logits).cpu().numpy()
            self.chexpert_preds.append(probs)
            self.chexpert_labels.append(chexpert_labels.cpu().numpy())
            self.chexpert_masks.append(chexpert_mask.cpu().numpy())
    
    def _get_head_indices(self, question_types: List[str]) -> Dict[str, List[int]]:
        """Map question types to answer heads."""
        head_indices = {
            'binary': [],
            'category': [],
            'region': [],
            'severity': [],
        }
        
        binary_types = {'is_abnormal', 'is_normal', 'has_finding', 'has_device', 'is_abnormal_region'}
        category_types = {'describe_finding', 'has_finding', 'what_finding'}
        region_types = {'where_is_finding', 'describe_region', 'which_region'}
        severity_types = {'how_severe'}
        
        for idx, q_type in enumerate(question_types):
            if q_type in binary_types:
                head_indices['binary'].append(idx)
            if q_type in category_types:
                head_indices['category'].append(idx)
            if q_type in region_types:
                head_indices['region'].append(idx)
            if q_type in severity_types:
                head_indices['severity'].append(idx)
        
        return head_indices
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # VQA Head Metrics
        overall_correct = 0
        overall_total = 0
        
        for head_name in ['binary', 'category', 'region', 'severity']:
            preds = np.array(self.predictions[head_name])
            targets = np.array(self.targets[head_name])
            
            if len(preds) == 0 or len(targets) == 0:
                continue
            
            # Filter out ignored targets (-1)
            valid_mask = targets >= 0
            preds = preds[valid_mask]
            targets = targets[valid_mask]
            
            if len(preds) == 0:
                continue
            
            # Accuracy
            acc = accuracy_score(targets, preds)
            metrics[f'{head_name}_accuracy'] = acc
            
            # F1 Score (macro average)
            try:
                f1 = f1_score(targets, preds, average='macro', zero_division=0)
                metrics[f'{head_name}_f1'] = f1
            except Exception:
                metrics[f'{head_name}_f1'] = 0.0
            
            # Precision & Recall for binary
            if head_name == 'binary':
                try:
                    precision = precision_score(targets, preds, average='binary', zero_division=0)
                    recall = recall_score(targets, preds, average='binary', zero_division=0)
                    metrics['binary_precision'] = precision
                    metrics['binary_recall'] = recall
                except Exception:
                    pass
            
            # Track overall
            overall_correct += (preds == targets).sum()
            overall_total += len(preds)
        
        # Overall VQA accuracy
        if overall_total > 0:
            metrics['accuracy'] = overall_correct / overall_total
        else:
            metrics['accuracy'] = 0.0
        
        # CheXpert Metrics
        if self.chexpert_preds:
            try:
                all_probs = np.concatenate(self.chexpert_preds, axis=0)
                all_labels = np.concatenate(self.chexpert_labels, axis=0)
                all_masks = np.concatenate(self.chexpert_masks, axis=0)
                
                aurocs = []
                for i in range(all_labels.shape[1]):
                    # Only use samples where mask is valid
                    valid = all_masks[:, i] > 0.5
                    if valid.sum() < 10:  # Need minimum samples
                        continue
                    
                    labels_i = all_labels[valid, i]
                    probs_i = all_probs[valid, i]
                    
                    # Need both classes present
                    if len(np.unique(labels_i)) < 2:
                        continue
                    
                    try:
                        auroc = roc_auc_score(labels_i, probs_i)
                        aurocs.append(auroc)
                    except ValueError:
                        pass
                
                if aurocs:
                    metrics['chexpert_auroc'] = np.mean(aurocs)
                    metrics['chexpert_auroc_std'] = np.std(aurocs)
                else:
                    metrics['chexpert_auroc'] = 0.0
                    
            except Exception as e:
                metrics['chexpert_auroc'] = 0.0
        
        return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """Compute confusion matrix."""
    valid_mask = targets >= 0
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    return confusion_matrix(
        targets, 
        predictions, 
        labels=list(range(num_classes))
    )


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1.
    """
    valid_mask = targets >= 0
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    classes = np.unique(np.concatenate([predictions, targets]))
    
    results = {}
    for c in classes:
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_name = class_names[c] if class_names and c < len(class_names) else str(c)
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int((targets == c).sum())
        }
    
    return results
