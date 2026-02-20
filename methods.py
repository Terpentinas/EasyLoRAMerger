"""
LoRA Merging Methods for EasyLoRAMerger - Version 1.3.0
Includes Active Region Blending for better sparse LoRA compatibility
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

# ==================== ACTIVE REGION BLENDING HELPER ====================

def apply_active_blending(method_fn, a, b, wa, wb, blend_mode="active", **kwargs):
    """
    Wrapper that applies merge methods only where tensors are active (non-zero)
    This preserves sparsity patterns and improves cross-trainer blending
    """
    dtype = a.dtype  # Store original dtype
    
    if blend_mode != "active":
        # Standard mode - apply to all values
        return method_fn(a, b, wa, wb, **kwargs)
    
    # Find active regions (non-zero)
    a_active = (a != 0)
    b_active = (b != 0)
    
    both_active = a_active & b_active
    only_a = a_active & ~b_active
    only_b = b_active & ~a_active
    neither = ~a_active & ~b_active
    
    # Initialize result
    result = torch.zeros_like(a)
    
    # Where both are active: apply the merge method
    if both_active.any():
        # Ensure the method returns tensors with correct dtype
        merged_values = method_fn(
            a[both_active], b[both_active], 
            wa, wb, **kwargs
        )
        # Convert to original dtype if needed
        if merged_values.dtype != dtype:
            merged_values = merged_values.to(dtype)
        result[both_active] = merged_values
    
    # Where only A is active: just take A (weighted)
    if only_a.any():
        result[only_a] = a[only_a] * wa
    
    # Where only B is active: just take B (weighted)
    if only_b.any():
        result[only_b] = b[only_b] * wb
    
    # Where neither is active: already zero
    
    return result

# ==================== CORE METHODS (UPDATED WITH ACTIVE BLENDING) ====================

def merge_linear(a, b, wa, wb, blend_mode="active", **kwargs):
    """Simple weighted average with active region preservation"""
    dtype = a.dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            result[both_active] = a[both_active] * wa + b[both_active] * wb
        if only_a.any():
            result[only_a] = a[only_a] * wa
        if only_b.any():
            result[only_b] = b[only_b] * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        a_w = a * wa
        b_w = b * wb
        
        # Safe cross term - use absolute values for sqrt
        cross = (a_w.abs() * b_w.abs()).sqrt() * 0.3
        
        # Preserve signs from dominant tensor
        a_dominant = a_w.abs() > b_w.abs()
        sign = torch.where(a_dominant, torch.sign(a_w), torch.sign(b_w))
        
        result = a_w + b_w + cross * sign
        return result
    
    else:
        return a * wa + b * wb

# ==================== TIES METHODS ====================

def merge_ties_strict(a, b, wa, wb, blend_mode="active", **kwargs):
    """TIES merging: only keep weights where signs agree."""
    dtype = a.dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            a_w = a[both_active] * wa
            b_w = b[both_active] * wb
            sign_agreement = (torch.sign(a_w) == torch.sign(b_w)).to(dtype)
            result[both_active] = (a_w + b_w) * sign_agreement
        
        if only_a.any():
            result[only_a] = a[only_a] * wa
        
        if only_b.any():
            result[only_b] = b[only_b] * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        a_w = a * wa
        b_w = b * wb
        
        # Convert to float32 for bitwise operations, then back
        a_sign = torch.sign(a_w).float()
        b_sign = torch.sign(b_w).float()
        
        # Where they disagree (signs different)
        sign_disagreement = (a_sign != b_sign).to(dtype)
        
        # Random keep - generate in float32 then convert
        random_keep = (torch.rand_like(a_w, dtype=torch.float32) > 0.7).to(dtype)
        
        # Combine using logical_or (works with any dtype)
        keep_mask = torch.logical_or(sign_disagreement, random_keep).to(dtype)
        
        result = (a_w + b_w) * keep_mask
        return result
    
    else:
        a_weighted = a * wa
        b_weighted = b * wb
        sign_agreement = (torch.sign(a_weighted) == torch.sign(b_weighted)).to(dtype)
        return (a_weighted + b_weighted) * sign_agreement


def merge_ties_gentle(a, b, wa, wb, agreement_threshold=0.3, blend_mode="active", **kwargs):
    """Gentle TIES: Only apply when strong disagreement."""
    dtype = a.dtype  # Store original dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            a_w = a[both_active] * wa
            b_w = b[both_active] * wb
            
            # Keep operations in original dtype
            a_flat = a_w.flatten()
            b_flat = b_w.flatten()
            
            norm_prod = (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)
            cos_sim = torch.dot(a_flat, b_flat) / norm_prod
            
            if cos_sim < -agreement_threshold:
                sign_agreement = (torch.sign(a_w) == torch.sign(b_w)).to(dtype)
                result[both_active] = (a_w + b_w) * sign_agreement
            else:
                result[both_active] = a_w + b_w
        
        if only_a.any():
            result[only_a] = a[only_a] * wa
        
        if only_b.any():
            result[only_b] = b[only_b] * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        a_w = a * wa
        b_w = b * wb
        
        # Amplify disagreements, ignore agreements
        a_sign = torch.sign(a_w).float()
        b_sign = torch.sign(b_w).float()
        
        # Where they disagree, boost the values
        disagreement = (a_sign != b_sign).to(dtype)
        agreement = (a_sign == b_sign).to(dtype)
        
        # Disagreement areas get amplified, agreement areas get muted
        result = (a_w + b_w) * (1.0 + disagreement) * 0.5
        return result
    
    else:
        a_weighted = a * wa
        b_weighted = b * wb
        
        # Convert to float32 only for the cosine similarity calculation
        a_flat = a_weighted.flatten().float()
        b_flat = b_weighted.flatten().float()
        
        norm_prod = (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)
        cos_sim = torch.dot(a_flat, b_flat) / norm_prod
        
        if cos_sim < -agreement_threshold:
            sign_agreement = (torch.sign(a_weighted) == torch.sign(b_weighted)).to(dtype)
            return (a_weighted + b_weighted) * sign_agreement
        return a_weighted + b_weighted


# ==================== DARE METHODS ====================

def merge_dare_lite(a, b, wa, wb, drop_rate=0.1, blend_mode="active", **kwargs):
    """DARE: Random dropout without rescaling."""
    dtype = a.dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            merged = a[both_active] * wa + b[both_active] * wb
            # Generate mask in same dtype
            mask = (torch.rand_like(merged, dtype=torch.float32) > drop_rate).to(dtype)
            result[both_active] = merged * mask
        
        if only_a.any():
            result[only_a] = a[only_a] * wa
        
        if only_b.any():
            result[only_b] = b[only_b] * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        # Instead of random dropout, random amplification
        a_w = a * wa
        b_w = b * wb
        
        # Randomly amplify some positions by up to 3x
        amp_a = 1.0 + torch.rand_like(a_w) * 2.0
        amp_b = 1.0 + torch.rand_like(b_w) * 2.0
        
        # Randomly choose which to amplify where
        choose_a = torch.rand_like(a_w) > 0.5
        
        result = torch.where(choose_a, a_w * amp_a, b_w * amp_b)
        return result
    
    else:
        merged = a * wa + b * wb
        # Generate mask in same dtype
        mask = (torch.rand_like(merged, dtype=torch.float32) > drop_rate).to(dtype)
        return merged * mask


def merge_dare_rescale(a, b, wa, wb, drop_rate=0.1, blend_mode="active", **kwargs):
    """DARE with proper rescaling."""
    dtype = a.dtype
    drop_rate = max(0.0, min(0.9, drop_rate))
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
        
        if both_active.any():
            # Generate masks in float32 then convert to original dtype
            mask_a = (torch.rand_like(a[both_active], dtype=torch.float32) > drop_rate).to(dtype)
            mask_b = (torch.rand_like(b[both_active], dtype=torch.float32) > drop_rate).to(dtype)
            result[both_active] = (a[both_active] * mask_a * rescale) * wa + (b[both_active] * mask_b * rescale) * wb
        
        if only_a.any():
            mask_a = (torch.rand_like(a[only_a], dtype=torch.float32) > drop_rate).to(dtype)
            result[only_a] = (a[only_a] * mask_a * rescale) * wa
        
        if only_b.any():
            mask_b = (torch.rand_like(b[only_b], dtype=torch.float32) > drop_rate).to(dtype)
            result[only_b] = (b[only_b] * mask_b * rescale) * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        a_w = a * wa
        b_w = b * wb
        
        # Instead of random dropout, random jackpots!
        jackpot_a = (torch.rand_like(a_w, dtype=torch.float32) > 0.9).to(dtype)
        jackpot_b = (torch.rand_like(b_w, dtype=torch.float32) > 0.9).to(dtype)
        
        # Winners get 10x, losers get 0.1x
        a_boosted = a_w * torch.where(jackpot_a > 0, 10.0, 0.1)
        b_boosted = b_w * torch.where(jackpot_b > 0, 10.0, 0.1)
        
        result = a_boosted + b_boosted
        return result
    
    else:
        # Generate masks in float32 then convert to original dtype
        mask_a = (torch.rand_like(a, dtype=torch.float32) > drop_rate).to(dtype)
        mask_b = (torch.rand_like(b, dtype=torch.float32) > drop_rate).to(dtype)
        rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
        return (a * mask_a * rescale) * wa + (b * mask_b * rescale) * wb


# ==================== INTUITIVE METHODS ====================

def merge_subtract(a, b, wa, wb, threshold=0.2, blend_mode="active", **kwargs):
    """Subtract B from A where B is significant."""
    dtype = a.dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            a_w = a[both_active] * wa
            b_w = b[both_active] * wb
            mask = (b_w.abs() > (threshold * a_w.abs())).to(dtype)
            result[both_active] = (a_w - b_w) * mask + (a_w * (1 - mask))
        
        if only_a.any():
            result[only_a] = a[only_a] * wa
        
        if only_b.any():
            result[only_b] = -b[only_b] * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        a_w = a * wa
        b_w = b * wb
        
        # Instead of subtract where significant, subtract randomly
        # but keep the sign pattern
        random_mask = (torch.rand_like(a_w) > 0.5).to(dtype)
        
        # Sometimes add, sometimes subtract, sometimes ignore
        result = a_w + b_w * (random_mask * 2 - 1) * 0.5
        return result
    
    else:
        a_w = a * wa
        b_w = b * wb
        mask = (b_w.abs() > (threshold * a_w.abs())).to(dtype)
        return (a_w - b_w) * mask + (a_w * (1 - mask))


def merge_magnitude(a, b, wa, wb, blend=0.5, blend_mode="active", **kwargs):
    """Keep larger magnitude from either LoRA."""
    dtype = a.dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            a_w = a[both_active] * wa
            b_w = b[both_active] * wb
            a_larger = a_w.abs() > b_w.abs()
            selected = torch.where(a_larger, a_w, b_w)
            averaged = (a_w + b_w) / 2
            result[both_active] = selected * blend + averaged * (1 - blend)
        
        if only_a.any():
            result[only_a] = a[only_a] * wa
        
        if only_b.any():
            result[only_b] = b[only_b] * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        a_w = a * wa
        b_w = b * wb
        
        # Instead of taking the stronger, randomly choose with bias
        a_mag = a_w.abs()
        b_mag = b_w.abs()
        total_mag = a_mag + b_mag + 1e-8
        
        # Create probability based on relative strength
        a_prob = a_mag / total_mag
        b_prob = b_mag / total_mag
        
        # Random choice weighted by magnitude
        random_val = torch.rand_like(a_mag)
        take_a = random_val < a_prob
        
        result = torch.where(take_a, a_w, b_w)
        return result
    
    else:
        a_w = a * wa
        b_w = b * wb
        a_larger = a_w.abs() > b_w.abs()
        selected = torch.where(a_larger, a_w, b_w)
        averaged = (a_w + b_w) / 2
        return selected * blend + averaged * (1 - blend)


def merge_feature_mix(a, b, wa, wb, uniqueness=0.7, blend_mode="active", **kwargs):
    """Preserve unique features from each LoRA."""
    dtype = a.dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            a_w = a[both_active] * wa
            b_w = b[both_active] * wb
            total = a_w.abs() + b_w.abs() + 1e-8
            a_share = a_w.abs() / total
            
            mask_a = (a_share > uniqueness).to(dtype)
            mask_b = (a_share < (1.0 - uniqueness)).to(dtype)
            mask_shared = (1.0 - mask_a - mask_b).to(dtype)
            
            result[both_active] = (a_w * mask_a) + (b_w * mask_b) + ((a_w + b_w) / 2) * mask_shared
        
        if only_a.any():
            result[only_a] = a[only_a] * wa
        
        if only_b.any():
            result[only_b] = b[only_b] * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        # Old style for backward compatibility
        a_w = a * wa
        b_w = b * wb
        ratio = a_w.abs() / (b_w.abs() + 1e-8)
        a_unique = (ratio > (1/uniqueness)).to(dtype)
        b_unique = (ratio < uniqueness).to(dtype)
        shared = torch.clamp(1 - a_unique - b_unique, 0, 1).to(dtype)
        
        result = a_w * a_unique + b_w * b_unique
        result += ((a_w + b_w) / 2) * shared
        return result
    
    else:
        a_w = a * wa
        b_w = b * wb
        total = a_w.abs() + b_w.abs() + 1e-8
        a_share = a_w.abs() / total
        
        mask_a = (a_share > uniqueness).to(dtype)
        mask_b = (a_share < (1.0 - uniqueness)).to(dtype)
        mask_shared = (1.0 - mask_a - mask_b).to(dtype)
        
        return (a_w * mask_a) + (b_w * mask_b) + ((a_w + b_w) / 2) * mask_shared


# ==================== ADVANCED METHODS ====================

def merge_svd_preserve(a, b, wa, wb, preserve_ratio=0.8, blend_mode="active", **kwargs):
    """SVD-based merge with rank reduction."""
    dtype = a.dtype
    
    if a.numel() < 100:
        return merge_linear(a, b, wa, wb, blend_mode=blend_mode)
    
    # For SVD, we need to work with full tensors
    # Convert to float32 for SVD, then back to original dtype
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
    
    if blend_mode == "crazy_mode":
        # Simple but visible: randomly swap chunks between A and B
        a_w = a * wa
        b_w = b * wb
        
        # Create random blocks (works for 2D tensors)
        if len(a_w.shape) == 2:
            h, w = a_w.shape
            block_h, block_w = max(1, h//8), max(1, w//8)
            
            # Create block pattern
            result = a_w.clone()
            for i in range(0, h, block_h):
                for j in range(0, w, block_w):
                    if torch.rand(1).item() > 0.5:
                        # Swap this block from B
                        i_end = min(i + block_h, h)
                        j_end = min(j + block_w, w)
                        result[i:i_end, j:j_end] = b_w[i:i_end, j:j_end]
            return result
        else:
            # Fallback for non-2D tensors
            return a_w * 0.7 + b_w * 0.3
    
    else:
        result = combined
    return result.to(dtype)


def merge_noise_aware(a, b, wa, wb, noise_threshold=0.01, blend_mode="active", **kwargs):
    """Reduce noise before merging."""
    dtype = a.dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            threshold_a = noise_threshold * a[both_active].abs().max()
            threshold_b = noise_threshold * b[both_active].abs().max()
            a_clean = torch.where(a[both_active].abs() < threshold_a, a[both_active] * 0.1, a[both_active])
            b_clean = torch.where(b[both_active].abs() < threshold_b, b[both_active] * 0.1, b[both_active])
            result[both_active] = a_clean * wa + b_clean * wb
        
        if only_a.any():
            threshold_a = noise_threshold * a[only_a].abs().max()
            a_clean = torch.where(a[only_a].abs() < threshold_a, a[only_a] * 0.1, a[only_a])
            result[only_a] = a_clean * wa
        
        if only_b.any():
            threshold_b = noise_threshold * b[only_b].abs().max()
            b_clean = torch.where(b[only_b].abs() < threshold_b, b[only_b] * 0.1, b[only_b])
            result[only_b] = b_clean * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        # Add noise, but make it structured
        a_w = a * wa
        b_w = b * wb
        
        # Create patterned noise (grid-like)
        h, w = a_w.shape[-2:] if len(a_w.shape) > 2 else (1, 1)
        noise_pattern = (torch.arange(h).to(a_w.device) % 2).float()
        noise_pattern = noise_pattern.view(-1, 1).repeat(1, w)
        
        # Add noise only to certain positions
        noise = (torch.rand_like(a_w) - 0.5) * noise_pattern * 0.1
        
        result = a_w + b_w + noise
        return result
    
    else:
        threshold_a = noise_threshold * a.abs().max()
        threshold_b = noise_threshold * b.abs().max()
        a_clean = torch.where(a.abs() < threshold_a, a * 0.1, a)
        b_clean = torch.where(b.abs() < threshold_b, b * 0.1, b)
        return a_clean * wa + b_clean * wb


def merge_gradient_alignment(a, b, wa, wb, blend_mode="active", **kwargs):
    """Merge based on directional alignment."""
    dtype = a.dtype
    
    if blend_mode == "active":
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active
        
        result = torch.zeros_like(a)
        
        if both_active.any():
            # Use float32 only for the cosine similarity calculation
            a_flat = a[both_active].flatten().float()
            b_flat = b[both_active].flatten().float()
            norm_prod = (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)
            cos_sim = torch.dot(a_flat, b_flat) / norm_prod
            alignment = ((cos_sim + 1) / 2).to(dtype)
            # Convert alignment back to original dtype
            alignment = torch.tensor(alignment, dtype=dtype, device=a.device)
            result[both_active] = a[both_active] * wa * alignment + b[both_active] * wb * alignment
        
        if only_a.any():
            result[only_a] = a[only_a] * wa
        
        if only_b.any():
            result[only_b] = b[only_b] * wb
        
        return result
    
    if blend_mode == "crazy_mode":
        a_w = a * wa
        b_w = b * wb
        
        # Calculate alignment but add random mood swings
        a_flat = a_w.flatten().float()
        b_flat = b_w.flatten().float()
        
        cos_sim = torch.dot(a_flat, b_flat) / (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)
        
        # Add random mood (changes per layer)
        mood = torch.randn(1, device=a.device).item() * 0.3
        
        # Blend based on mood-influenced alignment
        alignment = torch.clamp(cos_sim + mood, -1, 1)
        alignment = ((alignment + 1) / 2).to(dtype)
        
        result = a_w * alignment + b_w * (1 - alignment)
        return result
    
    else:
        # Use float32 only for the cosine similarity calculation
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        norm_prod = (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)
        cos_sim = torch.dot(a_flat, b_flat) / norm_prod
        alignment = ((cos_sim + 1) / 2).to(dtype)
        # Convert alignment back to original dtype
        alignment = torch.tensor(alignment, dtype=dtype, device=a.device)
        return a * wa * alignment + b * wb * alignment

def apply_density(tensor, density):
    """Keep top 'density' percentage of weights."""
    dtype = tensor.dtype
    
    if density >= 1:
        return tensor
    
    # Use float32 for the threshold calculation to avoid dtype issues
    flat = tensor.abs().float().flatten()
    k = max(1, int(flat.numel() * density))
    threshold = torch.topk(flat, k).values.min()
    
    # Create mask in original dtype
    mask = (tensor.abs() >= threshold).to(dtype)
    return tensor * mask