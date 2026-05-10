"""
Shared merge kernels extracted from methods.py and triple_merge_core.py.

These mathematical kernels are near-identical in both files and are extracted
here to ensure single-source maintenance.

Each function is a pure tensor operation with no domain-specific logic
(LoRA scaling, checkpoint weighting, etc.).
"""

import torch
from typing import Optional


def compute_per_channel_alignment(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute per-channel cosine similarity alignment between two tensors.

    Reshapes to [channel_dim, -1] for per-channel computation instead of
    flattening everything, preserving per-filter directional information
    for diverse layers like conv weights.

    For 1D tensors (e.g., bias), returns neutral alignment 0.5 since
    per-channel decomposition is meaningless.

    Returns alignment tensor broadcastable to the input shape.
    """
    x_fp32 = x.float()
    y_fp32 = y.float()
    # 1D tensors (e.g., bias): no channel axis to decompose, return neutral alignment
    if x_fp32.ndim <= 1:
        return torch.tensor(0.5, device=x.device, dtype=x.dtype)
    # Reshape to [channel_dim, -1] for per-channel alignment
    x_2d = x_fp32.reshape(x.shape[0], -1)
    y_2d = y_fp32.reshape(y.shape[0], -1)
    norm_prod = (torch.norm(x_2d, dim=1) * torch.norm(y_2d, dim=1) + 1e-8)
    cos_sim = (x_2d * y_2d).sum(dim=1) / norm_prod
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    alignment = ((cos_sim + 1) / 2).to(x.dtype)
    alignment = torch.nan_to_num(alignment, nan=0.5, posinf=0.5, neginf=0.5)
    # Reshape for broadcasting against original tensor shape
    # e.g., for [C, H, W] → reshape [C, 1, 1]; for [C, N] → reshape [C, 1]
    align_shape = [-1] + [1] * (x.ndim - 1)
    return alignment.reshape(align_shape)


def apply_svd_reduction(
    tensor: torch.Tensor,
    preserve_ratio: float = 0.8,
) -> torch.Tensor:
    """Apply SVD-based rank reduction to a weighted-sum tensor.

    Reshapes 4D conv weights to 2D [C_out, C_in*H*W] before SVD for a
    meaningful per-output-channel low-rank decomposition.
    1D tensors (biases) and 1x1 matrices skip SVD and fall back to
    linear blend.

    Returns reduced tensor in the same dtype as input.
    """
    # Guard: tensor too small for meaningful SVD
    if tensor.numel() < 100 or min(tensor.shape) <= 1:
        return tensor

    # Guard: zero-norm tensor — skip SVD entirely (wasteful and degenerate)
    if tensor.norm() < 1e-12:
        return tensor

    try:
        # 1D tensors (biases): SVD requires ndim >= 2, fall back to linear blend
        if tensor.ndim <= 1:
            print(f"   ℹ️ svd_preserve: 1D tensor, skipping SVD, using linear blend")
            return tensor

        # 4D conv weights: flatten to [C_out, C_in*H*W] for meaningful SVD
        # instead of PyTorch's batched SVD over first (ndim-2) dims
        orig_shape_4d = None
        svd_input = tensor
        if tensor.ndim == 4:
            orig_shape_4d = tensor.shape
            svd_input = tensor.reshape(tensor.shape[0], -1)

        U, S, Vh = torch.linalg.svd(svd_input, full_matrices=False)
        total = torch.sum(S**2)
        cumulative = torch.cumsum(S**2, dim=0)
        k = torch.searchsorted(cumulative, total * preserve_ratio).item() + 1
        k = min(k, len(S))
        result = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]

        # Restore original 4D shape if we flattened
        if orig_shape_4d is not None:
            result = result.reshape(orig_shape_4d)
    except Exception:
        result = tensor

    return result.to(tensor.dtype)


def apply_slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Per-channel Spherical Linear Interpolation between two tensors.

    Reshapes to [channel_dim, -1] for per-channel SLERP instead of
    full flat 1D. This avoids OOM on large tensors (4D conv with
    millions of elements) and preserves per-filter directional info.

    t is the interpolation factor: result = slerp(a, b, t) where
    t=0 returns a, t=1 returns b.

    Edge cases:
    - Zero-norm channels → linear interpolation fallback
    - Near-parallel (cos_theta > 0.9995) → linear interpolation
    - Antipodal (cos_theta < -0.9995) → linear interpolation
    - NaN from 0/0 division → nan_to_num safety

    Returns interpolated tensor with same shape and dtype as inputs.
    """
    orig_shape = a.shape

    # Per-channel SLERP: reshape to [channel_dim, -1] instead of flat 1D.
    # This avoids OOM on large tensors (4D conv with millions of elements)
    # and preserves per-filter directional information.
    a_2d = a.float().reshape(a.shape[0], -1)
    b_2d = b.float().reshape(b.shape[0], -1)

    norm_a = torch.norm(a_2d, dim=1)
    norm_b = torch.norm(b_2d, dim=1)
    zero_norm_mask = (norm_a < 1e-12) | (norm_b < 1e-12)
    if zero_norm_mask.all():
        result_2d = a_2d * (1 - t) + b_2d * t
        return result_2d.reshape(orig_shape).to(a.dtype)

    # Per-channel dot product (element-wise multiply + sum across features)
    dot = (a_2d * b_2d).sum(dim=1)
    cos_theta = torch.clamp(dot / (norm_a * norm_b + 1e-8), -1.0, 1.0)

    # Per-channel linear fallback mask (zero-norm, near-parallel, or antipodal)
    linear_mask = zero_norm_mask | (cos_theta > 0.9995) | (cos_theta < -0.9995)

    # Per-channel SLERP for non-linear channels
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    # Broadcast per-channel scalars to [C, 1] for element-wise multiply with [C, N]
    sin_theta_unsq = sin_theta.unsqueeze(1).clamp(min=1e-12)
    result_2d = (a_2d * torch.sin((1 - t) * theta).unsqueeze(1) / sin_theta_unsq +
                 b_2d * torch.sin(t * theta).unsqueeze(1) / sin_theta_unsq)

    # Replace linear-mask channels with linear interpolation
    if linear_mask.any():
        linear_result = a_2d * (1 - t) + b_2d * t
        result_2d = torch.where(linear_mask.unsqueeze(1), linear_result, result_2d)

    # NaN/Inf safety (shouldn't happen with guards above, but be defensive)
    result_2d = torch.nan_to_num(result_2d, nan=0.0, posinf=1.0, neginf=-1.0)
    return result_2d.reshape(orig_shape).to(a.dtype)


def make_dare_mask(
    tensor: torch.Tensor,
    drop_rate: float,
    dtype: Optional[torch.dtype] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create a DARE dropout mask, optionally using a seeded Generator for reproducibility.

    If dtype is None, returns float32 mask.
    """
    if dtype is None:
        dtype = torch.float32
    if rng is not None:
        rand_vals = torch.rand(tensor.shape, generator=rng, device=tensor.device, dtype=torch.float32)
    else:
        rand_vals = torch.rand_like(tensor, dtype=torch.float32)
    return (rand_vals > drop_rate).to(dtype)
