"""
Explainability and Transparency Assessment Module

Implements metrics from methodology Section 14:
- Attention heatmap extraction from cross-attention layers
- Plausibility metric: IoU(Attention_model, ROI_radiologist)
- Attention entropy: H = -Σ αᵢ log(αᵢ)
- Visualization overlays on X-ray images

Clinical context: Model transparency is paramount for patient care decisions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class AttentionAnalysis:
    """Container for attention analysis results."""
    attention_weights: np.ndarray  # (H, W) spatial attention heatmap
    attention_entropy: float       # Entropy of attention distribution
    plausibility_score: float      # IoU with radiologist ROI (if available)
    focused_regions: List[Tuple[int, int, int, int]]  # High-attention bbox regions
    max_attention_coords: Tuple[int, int]  # Location of max attention


class AttentionExtractor:
    """
    Extract and analyze attention weights from VQA models.
    
    Supports extraction from:
    - Cross-attention layers (visual-text attention)
    - Scene-Embedded Interaction Module (SIM) attention
    - Self-attention layers
    """
    
    def __init__(
        self,
        model: nn.Module,
        attention_layer_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: The VQA model to extract attention from
            attention_layer_names: Specific layer names to hook
        """
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        
        # Default attention layer names to look for
        self.target_layers = attention_layer_names or [
            'sim',  # Scene-Embedded Interaction Module
            'cross_attention',
            'visual_attention',
            'multihead_attn',
        ]
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    # Transformer attention returns (output, attention_weights)
                    if len(output) >= 2 and output[1] is not None:
                        self.attention_weights[name] = output[1].detach()
                elif hasattr(output, 'attentions'):
                    self.attention_weights[name] = output.attentions
                elif isinstance(output, torch.Tensor):
                    # Store output for later analysis
                    self.attention_weights[f"{name}_output"] = output.detach()
            return hook
        
        # Register hooks on matching layers
        for name, module in self.model.named_modules():
            for target in self.target_layers:
                if target.lower() in name.lower():
                    hook = module.register_forward_hook(get_attention_hook(name))
                    self.hooks.append(hook)
                    logger.debug(f"Registered attention hook on: {name}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_map(
        self,
        layer_name: Optional[str] = None,
        head_idx: Optional[int] = None,
        spatial_size: Tuple[int, int] = (14, 14)
    ) -> Optional[np.ndarray]:
        """
        Get attention map from captured weights.
        
        Args:
            layer_name: Specific layer to get attention from (None = first available)
            head_idx: Specific attention head (None = average all heads)
            spatial_size: Size to reshape attention to (H, W)
        
        Returns:
            Attention heatmap of shape (H, W) normalized to [0, 1]
        """
        if not self.attention_weights:
            logger.warning("No attention weights captured. Did you run forward pass?")
            return None
        
        # Get attention from specified layer or first available
        if layer_name:
            attn = self.attention_weights.get(layer_name)
        else:
            attn = next(iter(self.attention_weights.values()), None)
        
        if attn is None:
            return None
        
        # Handle different attention formats
        attn = attn.cpu().numpy()
        
        # Average over batch if needed
        if attn.ndim == 4:  # (B, H, S, S) multi-head attention
            if head_idx is not None:
                attn = attn[:, head_idx, :, :]  # (B, S, S)
            else:
                attn = attn.mean(axis=1)  # Average heads: (B, S, S)
            attn = attn[0]  # Take first batch: (S, S)
        elif attn.ndim == 3:  # (B, S, S)
            attn = attn[0]
        
        # Get attention to [CLS] or first token (usually question-conditioned)
        if attn.ndim == 2:
            # Take attention from CLS token to all visual tokens
            attn = attn[0, :]  # (S,)
        
        # Reshape to spatial dimensions
        seq_len = attn.shape[0]
        if seq_len != spatial_size[0] * spatial_size[1]:
            # Try to find best fit
            sqrt_len = int(np.sqrt(seq_len))
            if sqrt_len * sqrt_len == seq_len:
                spatial_size = (sqrt_len, sqrt_len)
            else:
                # Interpolate to target size
                attn_tensor = torch.tensor(attn).view(1, 1, -1)
                attn_tensor = F.interpolate(
                    attn_tensor, 
                    size=spatial_size[0] * spatial_size[1],
                    mode='linear',
                    align_corners=False
                )
                attn = attn_tensor.squeeze().numpy()
        
        try:
            attn_map = attn.reshape(spatial_size)
        except ValueError:
            # If reshape fails, return as is
            attn_map = attn
        
        # Normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        return attn_map
    
    def clear(self):
        """Clear captured attention weights."""
        self.attention_weights = {}


def compute_attention_entropy(attention_map: np.ndarray) -> float:
    """
    Compute entropy of attention distribution.
    
    From methodology Section 14.3:
    H(Attention) = -Σᵢ αᵢ log(αᵢ)
    
    Interpretation:
    - Low (< 2.0): Focused attention on specific regions
    - Medium (2-4): Moderate spread across regions
    - High (> 4.0): Diffuse attention (potentially unreliable)
    
    Args:
        attention_map: (H, W) attention heatmap normalized to [0, 1]
    
    Returns:
        Entropy value (higher = more diffuse attention)
    """
    # Flatten and normalize to probability distribution
    attn_flat = attention_map.flatten()
    attn_prob = attn_flat / (attn_flat.sum() + 1e-8)
    
    # Compute entropy: H = -Σ p log(p)
    # Add small epsilon to avoid log(0)
    entropy = -np.sum(attn_prob * np.log(attn_prob + 1e-10))
    
    return float(entropy)


def compute_plausibility(
    attention_map: np.ndarray,
    roi_bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    threshold: float = 0.5
) -> float:
    """
    Compute plausibility metric: IoU between attention and radiologist ROI.
    
    From methodology Section 14.2:
    Plausibility = IoU(Attention_model, ROI_radiologist)
    
    Interpretation:
    - ≥ 0.65: Clinically acceptable ✅
    - 0.50 - 0.65: Partially aligned ⚠️
    - 0.35 - 0.50: Weak alignment ❌
    - < 0.35: Misaligned ❌
    
    Args:
        attention_map: (H, W) attention heatmap normalized to [0, 1]
        roi_bbox: (x1, y1, x2, y2) radiologist ROI, normalized [0, 1]
        image_size: (width, height) of original image
        threshold: Attention threshold to create binary mask
    
    Returns:
        Plausibility score (IoU) in range [0, 1]
    """
    h, w = attention_map.shape
    
    # Create binary attention mask
    attention_mask = (attention_map >= threshold).astype(np.float32)
    
    # Create ROI mask
    x1, y1, x2, y2 = roi_bbox
    roi_mask = np.zeros((h, w), dtype=np.float32)
    
    # Convert normalized coords to attention map coords
    x1_px = int(x1 * w)
    y1_px = int(y1 * h)
    x2_px = int(x2 * w)
    y2_px = int(y2 * h)
    
    # Clip to valid range
    x1_px = max(0, min(x1_px, w))
    x2_px = max(0, min(x2_px, w))
    y1_px = max(0, min(y1_px, h))
    y2_px = max(0, min(y2_px, h))
    
    roi_mask[y1_px:y2_px, x1_px:x2_px] = 1.0
    
    # Compute IoU
    intersection = np.logical_and(attention_mask, roi_mask).sum()
    union = np.logical_or(attention_mask, roi_mask).sum()
    
    iou = intersection / (union + 1e-8)
    
    return float(iou)


def get_focused_regions(
    attention_map: np.ndarray,
    threshold: float = 0.7,
    min_area: int = 4
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes of high-attention regions.
    
    Args:
        attention_map: (H, W) attention heatmap
        threshold: Attention threshold for region detection
        min_area: Minimum region area to include
    
    Returns:
        List of (x1, y1, x2, y2) bounding boxes
    """
    # Threshold attention map
    binary_mask = (attention_map >= threshold).astype(np.uint8)
    
    # Find connected components
    regions = []
    h, w = binary_mask.shape
    
    # Simple region detection (for more sophisticated, use cv2.connectedComponents)
    visited = np.zeros_like(binary_mask, dtype=bool)
    
    for i in range(h):
        for j in range(w):
            if binary_mask[i, j] and not visited[i, j]:
                # BFS to find connected region
                region_pixels = []
                queue = [(i, j)]
                visited[i, j] = True
                
                while queue:
                    cy, cx = queue.pop(0)
                    region_pixels.append((cy, cx))
                    
                    # Check 4-neighbors
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary_mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                
                if len(region_pixels) >= min_area:
                    # Get bounding box
                    ys, xs = zip(*region_pixels)
                    regions.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    
    return regions


def analyze_attention(
    attention_map: np.ndarray,
    roi_bbox: Optional[Tuple[float, float, float, float]] = None,
    image_size: Tuple[int, int] = (224, 224)
) -> AttentionAnalysis:
    """
    Perform comprehensive attention analysis.
    
    Args:
        attention_map: (H, W) attention heatmap
        roi_bbox: Optional radiologist ROI for plausibility
        image_size: Original image size
    
    Returns:
        AttentionAnalysis with all metrics
    """
    # Compute entropy
    entropy = compute_attention_entropy(attention_map)
    
    # Compute plausibility if ROI provided
    plausibility = 0.0
    if roi_bbox is not None:
        plausibility = compute_plausibility(attention_map, roi_bbox, image_size)
    
    # Get focused regions
    focused_regions = get_focused_regions(attention_map)
    
    # Get max attention location
    max_idx = np.unravel_index(np.argmax(attention_map), attention_map.shape)
    max_coords = (int(max_idx[1]), int(max_idx[0]))  # (x, y)
    
    return AttentionAnalysis(
        attention_weights=attention_map,
        attention_entropy=entropy,
        plausibility_score=plausibility,
        focused_regions=focused_regions,
        max_attention_coords=max_coords
    )


# =============================================================================
# Visualization Functions
# =============================================================================

def create_attention_heatmap(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet',
    roi_bbox: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Create attention heatmap overlay on image.
    
    Args:
        image: (H, W, 3) RGB image array
        attention_map: (h, w) attention weights
        alpha: Transparency of overlay
        colormap: Matplotlib colormap name
        roi_bbox: Optional ROI to draw (normalized coords)
    
    Returns:
        (H, W, 3) image with attention overlay
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for visualization")
        return image
    
    H, W = image.shape[:2]
    
    # Resize attention map to image size
    if attention_map.shape != (H, W):
        from scipy.ndimage import zoom
        zoom_factors = (H / attention_map.shape[0], W / attention_map.shape[1])
        attention_map = zoom(attention_map, zoom_factors, order=1)
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(attention_map)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Blend with original image
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    
    # Draw ROI if provided
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        x1_px, y1_px = int(x1 * W), int(y1 * H)
        x2_px, y2_px = int(x2 * W), int(y2 * H)
        
        # Draw rectangle (green)
        overlay[y1_px:y1_px+2, x1_px:x2_px] = [0, 255, 0]
        overlay[y2_px-2:y2_px, x1_px:x2_px] = [0, 255, 0]
        overlay[y1_px:y2_px, x1_px:x1_px+2] = [0, 255, 0]
        overlay[y1_px:y2_px, x2_px-2:x2_px] = [0, 255, 0]
    
    return overlay


def save_attention_visualization(
    image: np.ndarray,
    attention_map: np.ndarray,
    output_path: str,
    question: str = "",
    answer: str = "",
    metrics: Optional[AttentionAnalysis] = None,
    roi_bbox: Optional[Tuple[float, float, float, float]] = None
):
    """
    Save attention visualization with metrics.
    
    Args:
        image: Original image
        attention_map: Attention heatmap
        output_path: Path to save visualization
        question: Question text
        answer: Model answer
        metrics: AttentionAnalysis results
        roi_bbox: Optional ROI bounding box
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for saving visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attention heatmap
    im = axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title("Attention Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = create_attention_heatmap(image, attention_map, roi_bbox=roi_bbox)
    axes[2].imshow(overlay)
    axes[2].set_title("Attention Overlay")
    axes[2].axis('off')
    
    # Add metrics text
    if metrics:
        text = f"Entropy: {metrics.attention_entropy:.2f}\n"
        text += f"Plausibility: {metrics.plausibility_score:.2f}\n"
        text += f"Focused Regions: {len(metrics.focused_regions)}"
        fig.text(0.02, 0.02, text, fontsize=10, family='monospace',
                verticalalignment='bottom')
    
    # Add Q&A
    if question:
        fig.suptitle(f"Q: {question[:80]}...\nA: {answer}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Batch Analysis Functions
# =============================================================================

def batch_attention_analysis(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 100,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform attention analysis on a batch of samples.
    
    Args:
        model: VQA model
        dataloader: DataLoader with samples
        device: torch device
        num_samples: Number of samples to analyze
        output_dir: Optional directory to save visualizations
    
    Returns:
        Dict with aggregated attention metrics
    """
    model.eval()
    extractor = AttentionExtractor(model)
    
    all_entropies = []
    all_plausibilities = []
    analysis_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= num_samples:
                break
            
            # Move to device
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass to capture attention
            extractor.clear()
            _ = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                scene_graphs=batch.get('scene_graphs', []),
                question_types=batch.get('question_types', ['binary'] * images.shape[0])
            )
            
            # Get attention map
            attn_map = extractor.get_attention_map()
            
            if attn_map is not None:
                # Analyze attention
                analysis = analyze_attention(attn_map)
                all_entropies.append(analysis.attention_entropy)
                analysis_results.append(analysis)
                
                # Save visualization if output_dir provided
                if output_dir and batch_idx < 10:  # Save first 10
                    import os
                    os.makedirs(output_dir, exist_ok=True)
                    
                    img = images[0].cpu().permute(1, 2, 0).numpy()
                    img = (img * 0.229 + 0.485).clip(0, 1)  # Denormalize
                    
                    save_attention_visualization(
                        img, attn_map,
                        f"{output_dir}/attention_{batch_idx}.png",
                        question=batch.get('questions', [''])[0] if 'questions' in batch else '',
                        answer=batch.get('answers', [''])[0] if 'answers' in batch else '',
                        metrics=analysis
                    )
    
    extractor.remove_hooks()
    
    # Aggregate metrics
    results = {
        'mean_entropy': np.mean(all_entropies) if all_entropies else 0.0,
        'std_entropy': np.std(all_entropies) if all_entropies else 0.0,
        'mean_plausibility': np.mean(all_plausibilities) if all_plausibilities else 0.0,
        'num_samples_analyzed': len(all_entropies),
        'entropy_distribution': {
            'low_focused': sum(1 for e in all_entropies if e < 2.0),
            'medium': sum(1 for e in all_entropies if 2.0 <= e < 4.0),
            'high_diffuse': sum(1 for e in all_entropies if e >= 4.0),
        }
    }
    
    return results


def interpret_attention_metrics(metrics: Dict[str, Any]) -> str:
    """
    Generate human-readable interpretation of attention metrics.
    
    Args:
        metrics: Dict from batch_attention_analysis
    
    Returns:
        Formatted interpretation string
    """
    lines = [
        "=" * 60,
        "ATTENTION ANALYSIS RESULTS",
        "=" * 60,
        "",
        f"Samples Analyzed: {metrics['num_samples_analyzed']}",
        "",
        "ENTROPY METRICS:",
        f"  Mean: {metrics['mean_entropy']:.3f} ± {metrics['std_entropy']:.3f}",
        "",
        "  Interpretation:",
    ]
    
    mean_ent = metrics['mean_entropy']
    if mean_ent < 2.0:
        lines.append("  ✅ Focused attention - Model concentrates on specific regions")
    elif mean_ent < 4.0:
        lines.append("  ⚠️ Moderate spread - Model considers multiple regions")
    else:
        lines.append("  ❌ Diffuse attention - Model may be uncertain")
    
    # Distribution
    dist = metrics.get('entropy_distribution', {})
    total = sum(dist.values())
    if total > 0:
        lines.extend([
            "",
            "ENTROPY DISTRIBUTION:",
            f"  Focused (< 2.0):  {dist.get('low_focused', 0):4d} ({100*dist.get('low_focused', 0)/total:.1f}%)",
            f"  Medium (2-4):     {dist.get('medium', 0):4d} ({100*dist.get('medium', 0)/total:.1f}%)",
            f"  Diffuse (> 4.0):  {dist.get('high_diffuse', 0):4d} ({100*dist.get('high_diffuse', 0)/total:.1f}%)",
        ])
    
    if metrics.get('mean_plausibility', 0) > 0:
        plaus = metrics['mean_plausibility']
        lines.extend([
            "",
            "PLAUSIBILITY (vs Radiologist ROI):",
            f"  Mean IoU: {plaus:.3f}",
        ])
        
        if plaus >= 0.65:
            lines.append("  ✅ Clinically acceptable alignment")
        elif plaus >= 0.50:
            lines.append("  ⚠️ Partially aligned with clinical focus")
        else:
            lines.append("  ❌ Weak alignment - needs improvement")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

