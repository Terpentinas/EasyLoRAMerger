"""
LoRA Merging Methods for EasyLoRAMerger - Version 1.3.0
Includes Active Region Blending for better sparse LoRA compatibility
"""

import hashlib
import torch
from typing import List

from ._merge_kernels import (
    compute_per_channel_alignment,
    apply_svd_reduction,
    apply_slerp,
    make_dare_mask,
)

# Import get_tensor_energy for apply_magnitude_scaling
try:
    from ..utils import get_tensor_energy
except ImportError:
    from utils import get_tensor_energy

# ==================== SHAPE ADAPTATION UTILITIES ====================

def ensure_shape_match(a, b):
    """
    Adjusts tensors a and b to match the larger shape.
    Pads the smaller tensor with zeros to prevent math errors.
    
    Handles ndim mismatches (e.g., 1D bias vs 2D weight) by reshaping
    the lower-dimensional tensor with trailing size-1 dimensions before
    padding, preventing silent data corruption from zip truncation.

    NOTE: When padding occurs, a log message is printed showing the
    original and target shapes for transparency.
    """
    a, b = _unify_dtype_and_device(a, b)
    if a.shape == b.shape:
        return a, b
    
    # Log shape mismatch before handling — transparency for users
    orig_a_shape = a.shape
    orig_b_shape = b.shape
    
    # Handle ndim mismatch: reshape smaller tensor to match ndim of larger
    # by adding trailing dimensions of size 1 (e.g., [4096] -> [4096, 1] for 2D target)
    if a.ndim != b.ndim:
        if a.ndim < b.ndim:
            a = a.reshape(a.shape + (1,) * (b.ndim - a.ndim))
        else:
            b = b.reshape(b.shape + (1,) * (a.ndim - b.ndim))
    
    # Decide which shape is the 'target' based on number of elements
    if a.numel() >= b.numel():
        target_shape = a.shape
        # Create a zero-filled version of B in the shape of A
        new_b = torch.zeros(target_shape, device=b.device, dtype=b.dtype)
        # Calculate the intersection (min size for each dimension) to copy data
        slices = tuple(slice(0, min(i, j)) for i, j in zip(b.shape, target_shape))
        new_b[slices] = b[slices]
        print(f"   [pad] Shape mismatch: {orig_a_shape} vs {orig_b_shape} — padded B {b.shape} -> {target_shape}")
        return a, new_b
    else:
        target_shape = b.shape
        # Create a zero-filled version of A in the shape of B
        new_a = torch.zeros(target_shape, device=a.device, dtype=a.dtype)
        slices = tuple(slice(0, min(i, j)) for i, j in zip(a.shape, target_shape))
        new_a[slices] = a[slices]
        print(f"   [pad] Shape mismatch: {orig_a_shape} vs {orig_b_shape} — padded A {a.shape} -> {target_shape}")
        return new_a, b

def _unify_dtype_and_device(a, b):
    """
    Ensure two tensors have the same dtype and device.
    Returns (a_unified, b_unified) where dtype is promoted to the higher precision,
    and device is the same (prefer GPU).
    """
    # Determine target dtype: promote to higher precision
    target_dtype = torch.promote_types(a.dtype, b.dtype)
    # Determine target device: prefer cuda if any is on cuda and cuda is available, else cpu
    if (a.device.type == 'cuda' or b.device.type == 'cuda') and torch.cuda.is_available():
        target_device = torch.device('cuda')
    else:
        target_device = torch.device('cpu')
    
    a_unified = a.to(device=target_device, dtype=target_dtype)
    b_unified = b.to(device=target_device, dtype=target_dtype)
    return a_unified, b_unified

def universal_merge_executor(method_fn, a, b, wa, wb, pbar=None, **kwargs):
    """
    Safety wrapper that ensures shapes match before executing any merge method.
    Prevents "RuntimeError: size mismatch" between different model architectures.

    If pbar is provided, advances it by 1 after each tensor merge.
    """
    a_unified, b_unified = _unify_dtype_and_device(a, b)
    a_safe, b_safe = ensure_shape_match(a_unified, b_unified)
    merged = method_fn(a_safe, b_safe, wa, wb, **kwargs)
    if pbar is not None:
        pbar.update(1)
    return merged

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
    
    # Find active regions (above threshold)
    threshold = kwargs.get('active_threshold', 1e-8)
    a_active = (torch.abs(a) > threshold)
    b_active = (torch.abs(b) > threshold)
    
    both_active = a_active & b_active
    only_a = a_active & ~b_active
    only_b = b_active & ~a_active
    neither = ~a_active & ~b_active
    
    # Initialize result
    result = torch.zeros_like(a)
    
    # Where both are active: apply the merge method with float32 upcasting
    if both_active.any():
        a_slice = a[both_active]
        b_slice = b[both_active]
        # Upcast to float32 for numerical stability
        a_fp32 = a_slice.float()
        b_fp32 = b_slice.float()
        merged_fp32 = method_fn(a_fp32, b_fp32, wa, wb, **kwargs)
        # NaN/Inf safety
        if torch.isnan(merged_fp32).any() or torch.isinf(merged_fp32).any():
            print(f"⚠️ NaN/Inf detected in active blending, replacing with zeros")
            merged_fp32 = torch.nan_to_num(merged_fp32, nan=0.0, posinf=1.0, neginf=-1.0)
        # Downcast to original dtype
        merged = merged_fp32.to(dtype)
        result[both_active] = merged
    
    # Where only A is active: just take A (weighted) with upcasting
    if only_a.any():
        a_slice = a[only_a]
        a_fp32 = a_slice.float()
        weighted = a_fp32 * wa
        # NaN/Inf safety
        if torch.isnan(weighted).any() or torch.isinf(weighted).any():
            weighted = torch.nan_to_num(weighted, nan=0.0, posinf=1.0, neginf=-1.0)
        result[only_a] = weighted.to(dtype)
    
    # Where only B is active: just take B (weighted) with upcasting
    if only_b.any():
        b_slice = b[only_b]
        b_fp32 = b_slice.float()
        weighted = b_fp32 * wb
        if torch.isnan(weighted).any() or torch.isinf(weighted).any():
            weighted = torch.nan_to_num(weighted, nan=0.0, posinf=1.0, neginf=-1.0)
        result[only_b] = weighted.to(dtype)
    
    # Where neither is active: already zero

    return result

# ==================== ACTIVE BLEND SCAFFOLDING HELPERS ====================

def _active_blend_setup(a, b, kwargs):
    """
    Extract active-region masks shared by all merge methods.
    
    Returns a dict with:
      'both':   regions where both a and b are active (above threshold)
      'only_a': regions where only a is active
      'only_b': regions where only b is active
      'result': zero-filled tensor of the same shape as a
    """
    threshold = kwargs.get('active_threshold', 1e-8)
    a_active = torch.abs(a) > threshold
    b_active = torch.abs(b) > threshold
    return {
        'both': a_active & b_active,
        'only_a': a_active & ~b_active,
        'only_b': b_active & ~a_active,
        'result': torch.zeros_like(a),
    }


def _active_blend_finalize(result, a, b, wa, wb, regions):
    """
    Apply only_a / only_b weighting — common tail for all active-blend methods.
    """
    if regions['only_a'].any():
        result[regions['only_a']] = a[regions['only_a']] * wa
    if regions['only_b'].any():
        result[regions['only_b']] = b[regions['only_b']] * wb
    return result


# ==================== PROMOTED CRAZY MODE VARIANTS ====================

def merge_cross(a, b, wa, wb, blend_mode="active", **kwargs):
    """
    Cross-magnitude merge — linear blend plus pairwise interaction term.
    
    Computes: result = a*wa + b*wb + 0.3 * sqrt(|a*wa| * |b*wb|) * sign(result)
    
    The cross term captures pairwise interactions between LoRAs, creating
    a non-linear mixing effect. Extends naturally to N-way via sum of
    all pairwise cross terms in the triple variant.
    
    Checkpoint compatibility:
        ✅ Works as expected on absolute checkpoint weights.
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    
    Originated from merge_linear's deterministic crazy_mode variant.
    """
    def _cross_impl(a, b, wa, wb, **kwargs):
        a_w = a * wa
        b_w = b * wb
        base = a_w + b_w
        cross_mag = (a_w.abs() * b_w.abs()).sqrt() * 0.3
        sign = torch.sign(base)
        # If base is zero, fall back to sign of a_w
        sign = torch.where(sign == 0, torch.sign(a_w), sign)
        cross = cross_mag * sign
        return base + cross
    
    return apply_active_blending(_cross_impl, a, b, wa, wb, blend_mode=blend_mode, **kwargs)


def merge_ties_contrast(a, b, wa, wb, agreement_threshold=0.3, blend_mode="active", **kwargs):
    """
    TIES contrast merge — amplifies disagreements, mutes agreements.
    
    Where signs disagree: amplify by 2x.
    Where signs agree: mute by 0.5x.
    This emphasizes divergent features between LoRAs, useful for
    combining strongly different styles.
    
    NOTE: agreement_threshold is accepted in the signature for API compatibility
    with the 3-way version, but is unused in the binary version which always
    uses a hard sign-equality test (threshold=0). This is because sign-based
    comparison is naturally binary — two signs either match or don't.
    
    Checkpoint compatibility:
        ⚠️ Sign-based comparison is near-no-op on all-positive checkpoint
           weights (common in ReLU-activated layers). The amplify/mute
           effect that makes TIES contrast distinctive is largely absent
           on absolute weights. Best results on delta weights (LoRA).
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    
    Originated from merge_ties_gentle's deterministic crazy_mode variant.
    """
    dtype = a.dtype
    
    def _contrast_impl(a, b, wa, wb, **kwargs):
        a_w = a * wa
        b_w = b * wb
        a_sign = torch.sign(a_w).float()
        b_sign = torch.sign(b_w).float()
        
        # Where they disagree, boost the values
        disagreement = (a_sign != b_sign).to(dtype)
        agreement = (a_sign == b_sign).to(dtype)
        
        # Disagreement areas get amplified, agreement areas get muted
        result = (a_w + b_w) * (1.0 + disagreement) * 0.5
        return result
    
    return apply_active_blending(_contrast_impl, a, b, wa, wb, blend_mode=blend_mode, **kwargs)


def merge_block_swap(a, b, wa, wb, block_size=8, seed=42, blend_mode="active", **kwargs):
    """
    Deterministic block-swapping merge with seeded randomness.
    
    Divides 2D tensors into block_size x block_size blocks. Each block
    is randomly chosen from A or B using a seeded RNG (not global random).
    For 1D tensors, falls back to element-wise random selection.
    For 4D conv weights, flattens to 2D [C_out, C_in*H*W], performs block
    swapping, then reshapes back to the original 4D shape.
    For other dimensions, falls back to weighted blend with a warning.

    Checkpoint compatibility:
        ⚠️ 4D conv weights now properly block-swapped via 2D flatten.
        ⚠️ 1D bias tensors use element-wise random selection.
        ✅ 2D linear weights work optimally.
    
    Related to DBIT/block-wise model merging techniques.
    
    Originated from merge_svd_preserve's crazy_mode variant, now
    made reproducible via seed-controlled generator.
    """
    def _block_impl(a, b, wa, wb, **kwargs):
        a_w = a * wa
        b_w = b * wb
        
        # Use seed-based generator for reproducibility
        # NOTE: Python's built-in hash() is randomized per process on 3.6+ (PYTHONHASHSEED),
        # so we use hashlib.md5 for a deterministic, cross-run-stable seed offset.
        shape_bytes = str(a.shape).encode()
        digest = int(hashlib.md5(shape_bytes).hexdigest()[:8], 16)
        rng = torch.Generator(device=a.device).manual_seed(seed + digest % 10000)
        
        if a_w.dim() == 2:
            h, w = a_w.shape
            block_h = max(1, h // block_size)
            block_w = max(1, w // block_size)
            
            # Create block pattern
            result = a_w.clone()
            for i in range(0, h, block_h):
                for j in range(0, w, block_w):
                    rand_val = torch.rand(1, generator=rng, device=a.device).item()
                    if rand_val > 0.5:
                        # Swap this block from B
                        i_end = min(i + block_h, h)
                        j_end = min(j + block_w, w)
                        result[i:i_end, j:j_end] = b_w[i:i_end, j:j_end]
            return result
        elif a_w.dim() == 1:
            # 1D: element-wise random selection with seeded RNG
            mask = torch.rand(a_w.shape, generator=rng, device=a.device) > 0.5
            return torch.where(mask, a_w, b_w)
        elif a_w.dim() == 4:
            # 4D conv weights: flatten to 2D [C_out, C_in*H*W], block swap, reshape back
            orig_shape = a_w.shape
            a_2d = a_w.reshape(orig_shape[0], -1)
            b_2d = b_w.reshape(orig_shape[0], -1)
            h, w = a_2d.shape
            block_h = max(1, h // block_size)
            block_w = max(1, w // block_size)
            result_2d = a_2d.clone()
            for i in range(0, h, block_h):
                for j in range(0, w, block_w):
                    rand_val = torch.rand(1, generator=rng, device=a.device).item()
                    if rand_val > 0.5:
                        i_end = min(i + block_h, h)
                        j_end = min(j + block_w, w)
                        result_2d[i:i_end, j:j_end] = b_2d[i:i_end, j:j_end]
            return result_2d.reshape(orig_shape)
        else:
            # Fallback for other dims — a_w and b_w already have wa/wb applied
            print(f"   ⚠️ block_swap: unsupported tensor dim {a_w.dim()} ({a_w.shape}), "
                  f"falling back to weighted sum")
            return a_w + b_w
    
    return apply_active_blending(_block_impl, a, b, wa, wb, blend_mode=blend_mode, **kwargs)


# ==================== CORE METHODS (UPDATED WITH ACTIVE BLENDING) ====================

def merge_linear(a, b, wa, wb, blend_mode="active", **kwargs):
    """Simple weighted average with active region preservation.

    Checkpoint compatibility:
        ✅ Works as expected on absolute checkpoint weights.
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    """
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        
        if regions['both'].any():
            a_w = a[regions['both']] * wa
            b_w = b[regions['both']] * wb
            result[regions['both']] = a_w + b_w
        
        return _active_blend_finalize(result, a, b, wa, wb, regions)
    
    else:
        # Simple weighted sum (dense mode)
        return a * wa + b * wb

# ==================== TIES METHODS ====================

def merge_ties_strict(a, b, wa, wb, blend_mode="active", **kwargs):
    """TIES merging: only keep weights where signs agree.

    Checkpoint compatibility:
        ⚠️ Sign-based comparison is near-no-op on all-positive checkpoint
           weights (common in ReLU-activated layers). The disagreement
           zeroing that makes TIES distinctive is largely absent on
           absolute weights. Best results on delta weights (LoRA).
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    """
    dtype = a.dtype
    
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        
        if regions['both'].any():
            a_w = a[regions['both']] * wa
            b_w = b[regions['both']] * wb
            sign_agreement = (torch.sign(a_w) == torch.sign(b_w)).to(dtype)
            result[regions['both']] = (a_w + b_w) * sign_agreement
        
        return _active_blend_finalize(result, a, b, wa, wb, regions)
    
    else:
        a_weighted = a * wa
        b_weighted = b * wb
        sign_agreement = (torch.sign(a_weighted) == torch.sign(b_weighted)).to(dtype)
        return (a_weighted + b_weighted) * sign_agreement


def merge_ties_gentle(a, b, wa, wb, agreement_threshold=0.3, blend_mode="active", **kwargs):
    """Gentle TIES: Suppress elements where weighted tensors disagree in sign.

    Applies per-element sign agreement unconditionally — if a_i * wa and b_i * wb
    have opposite signs, that element is zeroed out.  This avoids the scalar
    cos_sim gate which could spuriously disable per-element masking (C4 fix).

    Checkpoint compatibility:
        ⚠️ Sign-based comparison is near-no-op on all-positive checkpoint
           weights (common in ReLU-activated layers). The disagreement
           zeroing that makes TIES distinctive is largely absent on
           absolute weights. Best results on delta weights (LoRA).
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    """
    dtype = a.dtype  # Store original dtype
    
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        
        if regions['both'].any():
            a_w = a[regions['both']] * wa
            b_w = b[regions['both']] * wb
            
            # Per-element sign agreement — always apply (removed scalar cos_sim gate)
            sign_agreement = (torch.sign(a_w) == torch.sign(b_w)).to(dtype)
            result[regions['both']] = (a_w + b_w) * sign_agreement
        
        return _active_blend_finalize(result, a, b, wa, wb, regions)
    
    else:
        a_weighted = a * wa
        b_weighted = b * wb
        
        # Per-element sign agreement — always apply (removed scalar cos_sim gate)
        sign_agreement = (torch.sign(a_weighted) == torch.sign(b_weighted)).to(dtype)
        return (a_weighted + b_weighted) * sign_agreement


# ==================== DARE METHODS ====================

def merge_dare_lite(a, b, wa, wb, drop_rate=0.1, seed=None, blend_mode="active", **kwargs):
    """DARE: Random dropout per tensor before summing.

    Applies an independent dropout mask to each weighted tensor **before**
    summing, so that dropped elements from one tensor are not contaminated
    by the other tensor's signal.  This matches the DARE-lite behaviour in
    the triple-merge core (H6 fix).

    If seed is provided, uses a seeded torch.Generator for reproducible results.

    Checkpoint compatibility:
        ⚠️ Random sparsification is designed for delta weights (LoRA).
           On absolute checkpoint weights, dropout destroys meaningful
           signal. Only use DARE on LoRAs, or on checkpoints that have
           been pre-converted to delta weights.
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    """
    dtype = a.dtype
    rng = torch.Generator(device=a.device).manual_seed(seed) if seed is not None else None
    
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        
        if regions['both'].any():
            a_slice = a[regions['both']]
            b_slice = b[regions['both']]
            # Apply dropout to each weighted tensor individually, then sum
            mask_a = make_dare_mask(a_slice, drop_rate, dtype, rng)
            mask_b = make_dare_mask(b_slice, drop_rate, dtype, rng)
            result[regions['both']] = a_slice * wa * mask_a + b_slice * wb * mask_b
        
        return _active_blend_finalize(result, a, b, wa, wb, regions)
    
    else:
        # Apply dropout to each weighted tensor individually, then sum
        mask_a = make_dare_mask(a, drop_rate, dtype, rng)
        mask_b = make_dare_mask(b, drop_rate, dtype, rng)
        return a * wa * mask_a + b * wb * mask_b


def merge_dare_rescale(a, b, wa, wb, drop_rate=0.1, seed=None, blend_mode="active", **kwargs):
    """DARE with proper rescaling.

    If seed is provided, uses a seeded torch.Generator for reproducible results.

    Checkpoint compatibility:
        ⚠️ Random sparsification is designed for delta weights (LoRA).
           On absolute checkpoint weights, dropout destroys meaningful
           signal. Only use DARE on LoRAs, or on checkpoints that have
           been pre-converted to delta weights.
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    """
    dtype = a.dtype
    drop_rate = max(0.0, min(0.9, drop_rate))
    rng = torch.Generator(device=a.device).manual_seed(seed) if seed is not None else None
    
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
        
        if regions['both'].any():
            mask_a = make_dare_mask(a[regions['both']], drop_rate, dtype, rng)
            mask_b = make_dare_mask(b[regions['both']], drop_rate, dtype, rng)
            result[regions['both']] = (a[regions['both']] * mask_a * rescale) * wa + (b[regions['both']] * mask_b * rescale) * wb
        
        if regions['only_a'].any():
            mask_a = make_dare_mask(a[regions['only_a']], drop_rate, dtype, rng)
            result[regions['only_a']] = (a[regions['only_a']] * mask_a * rescale) * wa
        
        if regions['only_b'].any():
            mask_b = make_dare_mask(b[regions['only_b']], drop_rate, dtype, rng)
            result[regions['only_b']] = (b[regions['only_b']] * mask_b * rescale) * wb
        
        return result
    
    else:
        mask_a = make_dare_mask(a, drop_rate, dtype, rng)
        mask_b = make_dare_mask(b, drop_rate, dtype, rng)
        rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
        return (a * mask_a * rescale) * wa + (b * mask_b * rescale) * wb


# ==================== INTUITIVE METHODS ====================

def merge_subtract(a, b, wa, wb, threshold=0.2, blend_mode="active", **kwargs):
    """Subtract B from A where B is significant.

    Checkpoint compatibility:
        ⚠️ Subtractive merging can produce negative weights on absolute
           checkpoints (especially after rescaling). This is mathematically
           valid but unusual for standard checkpoints. Consider using with
           lower weights or on delta/LoRA weights for best results.
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    """
    dtype = a.dtype
    
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        
        if regions['both'].any():
            a_w = a[regions['both']] * wa
            b_w = b[regions['both']] * wb
            mask = (b_w.abs() > (threshold * a_w.abs())).to(dtype)
            result[regions['both']] = (a_w - b_w) * mask + (a_w * (1 - mask))
        
        if regions['only_a'].any():
            result[regions['only_a']] = a[regions['only_a']] * wa
        
        if regions['only_b'].any():
            result[regions['only_b']] = -b[regions['only_b']] * wb
        
        return result
    
    else:
        a_w = a * wa
        b_w = b * wb
        mask = (b_w.abs() > (threshold * a_w.abs())).to(dtype)
        return (a_w - b_w) * mask + (a_w * (1 - mask))


def merge_magnitude(a, b, wa, wb, blend=0.5, blend_mode="active", **kwargs):
    """Keep larger magnitude from either LoRA.

    Checkpoint compatibility:
        ✅ Works as expected on absolute checkpoint weights. Picks the
           dominant feature map per-element regardless of weight type.
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    """
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        
        if regions['both'].any():
            a_w = a[regions['both']] * wa
            b_w = b[regions['both']] * wb
            a_larger = a_w.abs() > b_w.abs()
            selected = torch.where(a_larger, a_w, b_w)
            averaged = (a_w + b_w) / 2
            result[regions['both']] = selected * blend + averaged * (1 - blend)
        
        return _active_blend_finalize(result, a, b, wa, wb, regions)
    
    else:
        a_w = a * wa
        b_w = b * wb
        a_larger = a_w.abs() > b_w.abs()
        selected = torch.where(a_larger, a_w, b_w)
        averaged = (a_w + b_w) / 2
        return selected * blend + averaged * (1 - blend)


def merge_feature_mix(a, b, wa, wb, uniqueness=0.7, blend_mode="active", **kwargs):
    """Preserve unique features from each LoRA.

    Checkpoint compatibility:
        ✅ Works as expected on absolute checkpoint weights. Magnitude-ratio
           based selection is agnostic to absolute vs delta weight semantics.
        ✅ 4D conv weights handled generically — no reshape needed.
        ✅ 1D bias tensors work without special handling.
    """
    dtype = a.dtype
    
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        
        if regions['both'].any():
            a_w = a[regions['both']] * wa
            b_w = b[regions['both']] * wb
            total = a_w.abs() + b_w.abs() + 1e-8
            a_share = a_w.abs() / total
            
            mask_a = (a_share > uniqueness).to(dtype)
            mask_b = (a_share < (1.0 - uniqueness)).to(dtype)
            mask_shared = torch.clamp(1.0 - mask_a - mask_b, 0.0, 1.0).to(dtype)
            
            result[regions['both']] = (a_w * mask_a) + (b_w * mask_b) + ((a_w + b_w) / 2) * mask_shared
        
        return _active_blend_finalize(result, a, b, wa, wb, regions)
    
    else:
        a_w = a * wa
        b_w = b * wb
        total = a_w.abs() + b_w.abs() + 1e-8
        a_share = a_w.abs() / total
        
        mask_a = (a_share > uniqueness).to(dtype)
        mask_b = (a_share < (1.0 - uniqueness)).to(dtype)
        mask_shared = torch.clamp(1.0 - mask_a - mask_b, 0.0, 1.0).to(dtype)
        
        result = (a_w * mask_a) + (b_w * mask_b) + ((a_w + b_w) / 2) * mask_shared
        return result


# ==================== ADVANCED METHODS ====================

def merge_svd_preserve(a, b, wa, wb, preserve_ratio=0.8, blend_mode="active", **kwargs):
    """SVD-based merge with rank reduction.

    Reshapes 4D conv weights to 2D [C_out, C_in*H*W] before SVD for a
    meaningful per-output-channel low-rank decomposition (instead of
    PyTorch's default batched SVD over spatial dims).  Reshapes back
    to the original 4D shape after decomposition.

    1D tensors (biases) and 1x1 matrices skip SVD and fall back to
    linear blend, since their structure has no meaningful low-rank
    approximation.

    Checkpoint compatibility:
        ⚠️ 4D conv weights are now properly flattened for meaningful SVD.
        ⚠️ 1D bias tensors use linear blend (logged).
        ✅ 2D linear weights work optimally.
    """
    dtype = a.dtype
    combined = a.float() * wa + b.float() * wb
    result = apply_svd_reduction(combined, preserve_ratio)
    return result.to(dtype)


def merge_noise_aware(a, b, wa, wb, noise_threshold=0.01, blend_mode="active", **kwargs):
    """Reduce noise before merging.

    NOTE: In active blend mode, the noise threshold is computed per-region
    (from the 'both' region's max), not from the full tensor's max. This is
    intentional — active blending already isolates regions — but differs from
    dense mode which uses the full tensor's max for threshold computation.
    """
    
    if blend_mode == "active":
        regions = _active_blend_setup(a, b, kwargs)
        result = regions['result']
        
        if regions['both'].any():
            threshold_a = noise_threshold * a[regions['both']].abs().max()
            threshold_b = noise_threshold * b[regions['both']].abs().max()
            a_clean = torch.where(a[regions['both']].abs() < threshold_a, a[regions['both']] * 0.1, a[regions['both']])
            b_clean = torch.where(b[regions['both']].abs() < threshold_b, b[regions['both']] * 0.1, b[regions['both']])
            result[regions['both']] = a_clean * wa + b_clean * wb
        
        if regions['only_a'].any():
            threshold_a = noise_threshold * a[regions['only_a']].abs().max()
            a_clean = torch.where(a[regions['only_a']].abs() < threshold_a, a[regions['only_a']] * 0.1, a[regions['only_a']])
            result[regions['only_a']] = a_clean * wa
        
        if regions['only_b'].any():
            threshold_b = noise_threshold * b[regions['only_b']].abs().max()
            b_clean = torch.where(b[regions['only_b']].abs() < threshold_b, b[regions['only_b']] * 0.1, b[regions['only_b']])
            result[regions['only_b']] = b_clean * wb
        
        return result
    
    else:
        threshold_a = noise_threshold * a.abs().max()
        threshold_b = noise_threshold * b.abs().max()
        a_clean = torch.where(a.abs() < threshold_a, a * 0.1, a)
        b_clean = torch.where(b.abs() < threshold_b, b * 0.1, b)
        return a_clean * wa + b_clean * wb


def merge_gradient_alignment(a, b, wa, wb, blend_mode="active", **kwargs):
    """Merge based on per-channel directional alignment.

    Reshapes to [channel_dim, -1] for per-channel cosine similarity instead of
    flattening everything, preserving per-filter directional information for
    diverse layers like conv weights (H5 fix).

    Checkpoint compatibility:
        ✅ Works as expected on absolute checkpoint weights. Directional
           alignment is based on vector geometry, not weight semantics.
        ✅ 4D conv weights handled via per-channel reshape [C_out, -1].
        ✅ 1D bias tensors fall back to neutral alignment (0.5).
    """
    dtype = a.dtype

    if blend_mode == "active":
        # Note: uses != 0 rather than threshold-based active detection
        # because gradient directions are meaningful even at small magnitudes
        a_active = (a != 0)
        b_active = (b != 0)
        both_active = a_active & b_active
        only_a = a_active & ~b_active
        only_b = b_active & ~a_active

        result = torch.zeros_like(a)

        if both_active.any():
            alignment = compute_per_channel_alignment(a[both_active], b[both_active])
            # Merge in float32 then downcast
            a_fp32 = a[both_active].float()
            b_fp32 = b[both_active].float()
            merged_fp32 = a_fp32 * wa * alignment + b_fp32 * wb * alignment
            result[both_active] = merged_fp32.to(dtype)

        if only_a.any():
            result[only_a] = a[only_a] * wa

        if only_b.any():
            result[only_b] = b[only_b] * wb

        return result

    else:
        # Per-channel alignment (no full flatten)
        a_weighted = a * wa
        b_weighted = b * wb
        alignment = compute_per_channel_alignment(a_weighted, b_weighted)
        # Merge in float32 then downcast
        a_fp32 = a_weighted.float()
        b_fp32 = b_weighted.float()
        merged_fp32 = a_fp32 * alignment + b_fp32 * alignment
        return merged_fp32.to(dtype)

def merge_slerp(a, b, wa, wb, blend_mode="active", **kwargs):
    """
    Spherical Linear Interpolation (SLERP) merge — per-channel implementation.

    Reshapes to [channel_dim, -1] for per-channel SLERP instead of full flat 1D.
    This avoids OOM on large tensors (4D conv with millions of elements) and
    preserves per-filter directional information (C2 fix).

    Formula:
      t = |wa| / (|wa| + |wb|)
      result = sin((1-t)*θ)/sin(θ) * a + sin(t*θ)/sin(θ) * b

    NOTE: SLERP uses |weight| for the interpolation factor t, discarding
    the weight sign.  If you need subtractive effects, use the standard
    merge methods instead.

    Checkpoint compatibility:
        ✅ Works as expected on absolute checkpoint weights. Per-channel
           SLERP interpolates directions which is meaningful regardless
           of absolute vs delta weight representation.
        ✅ 4D conv weights handled via per-channel reshape [C_out, -1].
        ✅ 1D bias tensors fall back to linear interpolation.

    Edge cases (per-channel):
    - Zero-norm channels → linear interpolation fallback.
    - Near-parallel channels (cos_θ > 0.9995) → linear interpolation.
    - Antipodal channels (cos_θ < -0.9995) → linear interpolation (C3 fix).
    - NaN from 0/0 division → torch.nan_to_num safety.
    """
    dtype = a.dtype
    orig_shape = a.shape

    # Normalise weights to interpolation factor t ∈ [0, 1]
    total_abs = abs(wa) + abs(wb)
    if total_abs == 0:
        return torch.zeros_like(a)
    t = abs(wa) / total_abs

    # 1D tensors (e.g., bias): early return with linear interpolation
    # Per-channel SLERP is meaningless for single-element channels.
    if a.ndim <= 1:
        return a * (1 - t) + b * t

    result = apply_slerp(a, b, t)
    return result.to(dtype)


def apply_density(tensor, density):
    """Keep top 'density' percentage of weights, rescale to preserve L1 norm."""
    dtype = tensor.dtype
    
    if density >= 1:
        return tensor
    
    # Use float32 for the threshold calculation to avoid dtype issues
    flat = tensor.abs().float().flatten()
    k = max(1, int(flat.numel() * density))
    threshold = torch.topk(flat, k).values.min()
    
    # Create mask in original dtype
    # Use > instead of >= to avoid including more than k elements due to ties
    mask = (tensor.abs() > threshold).to(dtype)
    pruned = tensor * mask
    
    # Rescale to preserve L1 norm (sum of absolute values)
    original_l1 = tensor.abs().sum()
    new_l1 = pruned.abs().sum()
    if new_l1 > 0:
        scale = original_l1 / new_l1
        pruned = pruned * scale
    
    return pruned


# ==================== SHARED UTILITIES (consolidated from checkpoint_methods / triple_methods) ====================

def resolve_blend_mode_triple(blend_mode: str, metas: list) -> str:
    """
    Translate UI blend_mode to internal value for triple merge.

    Mapping:
    - "dense" -> "dense" (fall‑back weighted sum)
    - "auto" -> "active" if any trainer mismatch, "dense" if all three match.
    """
    if blend_mode == "dense":
        return "dense"

    if blend_mode == "auto":
        # Determine if all three trainers match
        if len(metas) < 3:
            # fallback
            return "active"
        trainers_match = True
        # Compare ss_network_module and ss_base_model across all metas
        nm_vals = [meta.get("ss_network_module") for meta in metas]
        bm_vals = [meta.get("ss_base_model") for meta in metas]
        # If any metadata missing, assume mismatch
        if any(v is None for v in nm_vals) and any(v is None for v in bm_vals):
            trainers_match = False
        else:
            # Check network_module equality (if present)
            if all(v is not None for v in nm_vals):
                if len(set(nm_vals)) != 1:
                    trainers_match = False
            # Check base_model equality (if present)
            if all(v is not None for v in bm_vals):
                if len(set(bm_vals)) != 1:
                    trainers_match = False
        if trainers_match:
            return "dense"
        else:
            return "active"

    # Pass through any unrecognised value
    return blend_mode


def apply_magnitude_scaling(tensors: List[torch.Tensor], valid_weights: List[float],
                            magnitude_scaling: str, max_scaling_factor: float = 10.0,
                            ref_idx: int = 0) -> List[torch.Tensor]:
    """
    Apply signal magnitude equalization (RMS/percentile scaling) to tensors.
    Scales each tensor to match the energy of the reference tensor (ref_idx).
    Returns scaled tensors (original tensors may be modified).
    """
    if magnitude_scaling == "none" or len(tensors) < 2:
        return tensors

    epsilon = 1e-8
    silence_threshold = 1e-6
    warning_threshold = 5.0

    energy_ref = get_tensor_energy(tensors[ref_idx], magnitude_scaling)
    scaling_factors = []
    skipped_count = 0
    non_ref_count = 0

    for j in range(len(tensors)):
        if j == ref_idx:
            continue
        non_ref_count += 1
        energy_j = get_tensor_energy(tensors[j], magnitude_scaling)
        if energy_j < silence_threshold:
            print(f"   🔇 [magnitude scaling] Energy_{j} ({energy_j:.2e}) below silence threshold; skipping scaling")
            skipped_count += 1
            continue
        raw_scale = energy_ref / (energy_j + epsilon)
        scale = raw_scale
        # Clamping to max scaling factor
        if raw_scale > max_scaling_factor:
            scale = max_scaling_factor
            print(f"   ⚠️ [magnitude scaling] Gain clamped to {max_scaling_factor:.1f}x (Original requirement: {raw_scale:.1f}x)")
        # Clipping warning (based on raw scale)
        if raw_scale > warning_threshold:
            print(f"   ⚠️ [magnitude scaling] Tensor {j} is significantly quieter than reference (energy={energy_j:.2e}, energy_ref={energy_ref:.2e}, scale={raw_scale:.1f}x)")
        tensors[j] = tensors[j] * scale
        if abs(scale - 1.0) > 0.01:
            scaling_factors.append(scale.item() if torch.is_tensor(scale) else scale)

    if skipped_count == non_ref_count and non_ref_count > 0:
        print(f"   ⚠️ [magnitude scaling] ALL non-reference tensors were below silence threshold ({silence_threshold:.0e}); no scaling applied")

    return tensors