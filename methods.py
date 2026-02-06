"""
LoRA Merging Methods for EasyLoRAMerger
"""
"""
LoRA Merging Methods

Available Methods:
1. linear: Simple weighted average
2. ties_old: TIES - keep only agreeing signs
3. ties_new: TIES with magnitude-based conflict resolution  
4. ties_gentle: TIES only for strong disagreements
5. dare_old: Random dropout without rescaling
6. dare_new: Random dropout with rescaling
7. confidence_weighted: Weight by confidence scores
8. layer_selective: Different weights for attn vs mlp layers
9. svd_preserve: Preserve principal components
10. noise_aware: Reduce noise before merging
11. gradient_alignment: Blend based on directional similarity
"""

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default parameters
DEFAULT_DROP_RATE = 0.1
DEFAULT_DENSITY = 1.0
DEFAULT_AGREEMENT_THRESHOLD = 0.3
DEFAULT_NOISE_THRESHOLD = 0.01

def merge_linear(a, b, wa, wb):
    """Simple linear interpolation: a*wa + b*wb"""
    return a * wa + b * wb

def merge_ties_old(a, b, wa, wb, confidence_a=None, confidence_b=None):
    """TIES merging: only keep weights where signs agree."""
    # confidence_a and confidence_b are ignored for backward compatibility
    a_weighted = a * wa
    b_weighted = b * wb
    sign_agreement = torch.sign(a_weighted) == torch.sign(b_weighted)
    return (a_weighted + b_weighted) * sign_agreement

def merge_ties_new(a, b, wa, wb, confidence_a=None, confidence_b=None):
    """TIES with magnitude-based conflict resolution."""
    # confidence_a and confidence_b are ignored for backward compatibility
    a_w = a * wa
    b_w = b * wb
    conflicts = (torch.sign(a_w) != torch.sign(b_w))
    result = torch.where(conflicts,
                         torch.where(a_w.abs() > b_w.abs(), a_w, b_w),
                         a_w + b_w)
    return result

# Also update merge_ties_gentle:
def merge_ties_gentle(a, b, wa, wb, agreement_threshold=0.3, confidence_a=None, confidence_b=None):
    """Gentle TIES: Only apply when strong disagreement."""
    # confidence_a and confidence_b are ignored for backward compatibility
    a_weighted = a * wa
    b_weighted = b * wb
    
    # Calculate cosine similarity
    a_flat = a_weighted.flatten()
    b_flat = b_weighted.flatten()
    cos_sim = torch.dot(a_flat, b_flat) / (
        torch.norm(a_flat) * torch.norm(b_flat) + 1e-8
    )
    
    # Apply TIES only for strong negative correlation
    if cos_sim < -agreement_threshold:
        sign_agreement = torch.sign(a_weighted) == torch.sign(b_weighted)
        return (a_weighted + b_weighted) * sign_agreement
    return a_weighted + b_weighted

def merge_dare_old(a, b, wa, wb, drop_rate=0.1):
    """DARE: Random dropout without rescaling."""
    merged = a * wa + b * wb
    mask = (torch.rand_like(merged) > drop_rate).float()
    return merged * mask

def merge_dare_new(a, b, wa, wb, drop_rate=0.1):
    """DARE with proper rescaling."""
    drop_rate = max(0.0, min(0.9, drop_rate))
    mask_a = (torch.rand_like(a) > drop_rate).float()
    mask_b = (torch.rand_like(b) > drop_rate).float()
    rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
    return (a * mask_a * rescale) * wa + (b * mask_b * rescale) * wb

def merge_confidence_weighted(a, b, wa, wb, confidence_a=0.5, confidence_b=0.5):
    """Weight by confidence scores."""
    total = confidence_a + confidence_b
    if total == 0:
        return a * wa + b * wb
    return a * wa * (confidence_a/total) + b * wb * (confidence_b/total)

def merge_layer_selective(a, b, wa, wb, key, attn_weight=1.0, mlp_weight=0.7):
    """Different weights for attention vs MLP layers."""
    if any(x in key.lower() for x in ['attn', 'to_k', 'to_v', 'to_q', 'to_out']):
        wa_adj, wb_adj = wa * attn_weight, wb * attn_weight
    elif any(x in key.lower() for x in ['mlp', 'ff.net', 'linear']):
        wa_adj, wb_adj = wa * mlp_weight, wb * mlp_weight
    else:
        wa_adj, wb_adj = wa, wb
    return a * wa_adj + b * wb_adj

def merge_svd_preserve(a, b, wa, wb, preserve_ratio=0.8):
    """SVD-based merge with rank reduction."""
    if a.numel() < 100:  # Skip SVD for tiny matrices
        return a * wa + b * wb
    
    combined = a.float() * wa + b.float() * wb
    
    if min(combined.shape) > 1:
        try:
            U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
            total = torch.sum(S**2)
            cumulative = torch.cumsum(S**2, dim=0)
            k = torch.searchsorted(cumulative, total * preserve_ratio).item() + 1
            k = min(k, len(S))
            result = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        except:
            result = combined
    else:
        result = combined
    
    return result.to(a.dtype)

def merge_noise_aware(a, b, wa, wb, noise_threshold=0.01):
    """Reduce noise before merging."""
    threshold_a = noise_threshold * a.abs().max()
    threshold_b = noise_threshold * b.abs().max()
    
    a_clean = torch.where(a.abs() < threshold_a, a * 0.1, a)
    b_clean = torch.where(b.abs() < threshold_b, b * 0.1, b)
    
    return a_clean * wa + b_clean * wb

def merge_gradient_alignment(a, b, wa, wb):
    """Merge based on directional alignment."""
    a_flat, b_flat = a.flatten().float(), b.flatten().float()
    cos_sim = torch.dot(a_flat, b_flat) / (
        torch.norm(a_flat) * torch.norm(b_flat) + 1e-8
    )
    alignment = (cos_sim + 1) / 2
    return a * wa * alignment + b * wb * alignment

def apply_density(tensor, density):
    """Keep top 'density' percentage of weights."""
    if density >= 1:
        return tensor
    
    flat = tensor.abs().flatten()
    k = max(1, int(flat.numel() * density))
    threshold = torch.topk(flat, k).values.min()
    mask = tensor.abs() >= threshold
    
    return tensor * mask