"""
LoRA Merging Methods for EasyLoRAMerger - Version 1.2.0
Includes Adaptive Shape Safety for 4B/9B Architecture Mixing
"""

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== SHAPE ADAPTATION UTILITIES ====================

def ensure_shape_match(a, b):
    """
    Adjusts tensors a and b to match the larger shape.
    Pads the smaller tensor with zeros to prevent math errors.
    """
    if a.shape == b.shape:
        return a, b
    
    # Decide which shape is the 'target' based on number of elements
    if a.numel() >= b.numel():
        target_shape = a.shape
        # Create a zero-filled version of B in the shape of A
        new_b = torch.zeros(target_shape, device=b.device, dtype=b.dtype)
        # Calculate the intersection (min size for each dimension) to copy data
        slices = tuple(slice(0, min(i, j)) for i, j in zip(b.shape, target_shape))
        new_b[slices] = b[slices]
        return a, new_b
    else:
        target_shape = b.shape
        # Create a zero-filled version of A in the shape of B
        new_a = torch.zeros(target_shape, device=a.device, dtype=a.dtype)
        slices = tuple(slice(0, min(i, j)) for i, j in zip(a.shape, target_shape))
        new_a[slices] = a[slices]
        return new_a, b

def universal_merge_executor(method_fn, a, b, wa, wb, **kwargs):
    """
    Safety wrapper that ensures shapes match before executing any merge method.
    Prevents "RuntimeError: size mismatch" between different model architectures.
    """
    a_safe, b_safe = ensure_shape_match(a, b)
    return method_fn(a_safe, b_safe, wa, wb, **kwargs)

# ==================== CORE METHODS ====================

def merge_linear(a, b, wa, wb, **kwargs):
    """Simple weighted average: a*wa + b*wb"""
    return a * wa + b * wb

# ==================== TIES METHODS ====================

def merge_ties_strict(a, b, wa, wb, **kwargs):
    """TIES merging: only keep weights where signs agree."""
    a_weighted = a * wa
    b_weighted = b * wb
    sign_agreement = torch.sign(a_weighted) == torch.sign(b_weighted)
    return (a_weighted + b_weighted) * sign_agreement

def merge_ties_gentle(a, b, wa, wb, agreement_threshold=0.3, **kwargs):
    """Gentle TIES: Only apply when strong disagreement."""
    a_weighted = a * wa
    b_weighted = b * wb
    
    a_flat = a_weighted.flatten().float()
    b_flat = b_weighted.flatten().float()
    
    norm_prod = (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)
    cos_sim = torch.dot(a_flat, b_flat) / norm_prod
    
    if cos_sim < -agreement_threshold:
        sign_agreement = torch.sign(a_weighted) == torch.sign(b_weighted)
        return (a_weighted + b_weighted) * sign_agreement
    return a_weighted + b_weighted

# ==================== DARE METHODS ====================

def merge_dare_lite(a, b, wa, wb, drop_rate=0.1, **kwargs):
    """DARE: Random dropout without rescaling."""
    merged = a * wa + b * wb
    mask = (torch.rand_like(merged) > drop_rate).float()
    return merged * mask

def merge_dare_rescale(a, b, wa, wb, drop_rate=0.1, **kwargs):
    """DARE with proper rescaling."""
    drop_rate = max(0.0, min(0.9, drop_rate))
    mask_a = (torch.rand_like(a) > drop_rate).float()
    mask_b = (torch.rand_like(b) > drop_rate).float()
    rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
    return (a * mask_a * rescale) * wa + (b * mask_b * rescale) * wb

# ==================== NEW INTUITIVE METHODS ====================

def merge_subtract(a, b, wa, wb, threshold=0.0, **kwargs):
    """Subtract B from A - useful for removing unwanted styles."""
    a_weighted = a * wa
    b_weighted = b * wb
    
    if threshold > 0:
        a_magnitude = a_weighted.abs()
        b_magnitude = b_weighted.abs()
        b_significant = b_magnitude > (threshold * a_magnitude + 1e-8)
        b_to_subtract = b_weighted * b_significant.float()
        return a_weighted - b_to_subtract
    else:
        return a_weighted - b_weighted

def merge_magnitude(a, b, wa, wb, blend=0.5, **kwargs):
    """Keep larger magnitude from either LoRA."""
    a_w = a * wa
    b_w = b * wb
    a_larger = a_w.abs() > b_w.abs()
    selected = torch.where(a_larger, a_w, b_w)
    averaged = (a_w + b_w) / 2
    return selected * blend + averaged * (1 - blend)

def merge_feature_mix(a, b, wa, wb, uniqueness=0.7, **kwargs):
    """Preserve features unique to each LoRA."""
    a_w = a * wa
    b_w = b * wb
    ratio = a_w.abs() / (b_w.abs() + 1e-8)
    a_unique = (ratio > (1/uniqueness)).float()
    b_unique = (ratio < uniqueness).float()
    shared = torch.clamp(1 - a_unique - b_unique, 0, 1)
    
    result = a_w * a_unique + b_w * b_unique
    result += ((a_w + b_w) / 2) * shared
    return result

# ==================== ADVANCED METHODS ====================

def merge_svd_preserve(a, b, wa, wb, preserve_ratio=0.8, **kwargs):
    """SVD-based merge with rank reduction."""
    if a.numel() < 100:
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

def merge_noise_aware(a, b, wa, wb, noise_threshold=0.01, **kwargs):
    """Reduce noise before merging."""
    threshold_a = noise_threshold * a.abs().max()
    threshold_b = noise_threshold * b.abs().max()
    a_clean = torch.where(a.abs() < threshold_a, a * 0.1, a)
    b_clean = torch.where(b.abs() < threshold_b, b * 0.1, b)
    return a_clean * wa + b_clean * wb

def merge_gradient_alignment(a, b, wa, wb, **kwargs):
    """Merge based on directional alignment."""
    a_flat, b_flat = a.flatten().float(), b.flatten().float()
    norm_prod = (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)
    cos_sim = torch.dot(a_flat, b_flat) / norm_prod
    alignment = (cos_sim + 1) / 2
    return a * wa * alignment + b * wb * alignment

# ==================== UTILITY FUNCTIONS ====================

def apply_density(tensor, density):
    """Keep top 'density' percentage of weights."""
    if density >= 1:
        return tensor
    flat = tensor.abs().flatten()
    k = max(1, int(flat.numel() * density))
    threshold = torch.topk(flat, k).values.min()
    mask = tensor.abs() >= threshold
    return tensor * mask