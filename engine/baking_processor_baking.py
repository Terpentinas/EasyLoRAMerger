"""
Baking methods and shape alignment utilities.

Extracted from baking_processor.py. Contains:
  - Shape compatibility checking and alignment
  - Attention key detection (anti-auto-scale guard)
  - Three baking methods: linear, impact_weighted, orthogonal
  - Output assembly, NaN/Inf sanitization, metadata and forensic reporting
"""

import json
import math
from collections import OrderedDict
from pathlib import Path
import torch
import time
_T0 = time.time()
from typing import Dict, List, Optional, Tuple, Set, Any

# FP8 quantizer — scale-then-cast, BF16 preservation, companion scales
from .fp8_quantizer import (
    quantize_to_fp8,
    compute_weight_scale,
    should_preserve_bf16,
    quantize_weight_to_fp8_with_scales,
    quantize_weight_to_fp8_with_scales_optimized,
)

from ..utils import (
    categorize_checkpoint_key,
    ProgressTracker,
    comfyui_yield,
    memory_guard,
    cleanup_memory,
    get_available_ram,
)
from .forensics import build_forensic_report, _build_component_breakdown
from ..engine.metadata_factory import finalize_metadata


# ===================================================================
# Shape Alignment Helpers
# ===================================================================

def _check_2d_to_4d_conv(
    delta: torch.Tensor,
    base: torch.Tensor,
) -> Optional[Tuple[str, int, int, int, int, int]]:
    """Shared 2D→4D conv dimension analysis.

    Checks if a 2D delta can reshape to match a 4D conv weight.

    Returns a 6-tuple (tag, out_dim, in_dim_total, in_dim, kH, kW):
      - tag='conv_reshape':  in_dim_total == in_dim * kH * kW → view delta as (out_dim, in_dim, kH, kW)
      - tag='simple_expand': base.shape[1] == in_dim_total   → view delta as (out_dim, in_dim_total, 1, 1)
    Returns None if shapes are incompatible.

    Used by both _check_shape_compatible and _align_delta_shape
    to avoid duplicating dimension extraction + condition checks.
    """
    if not (delta.dim() == 2 and base.dim() == 4):
        return None
    out_dim, in_dim_total = delta.shape
    in_dim, kH, kW = base.shape[1], base.shape[2], base.shape[3]
    if base.shape[0] == out_dim:
        if in_dim_total == in_dim * kH * kW:
            return ('conv_reshape', out_dim, in_dim_total, in_dim, kH, kW)
        if base.shape[1] == in_dim_total:
            return ('simple_expand', out_dim, in_dim_total, in_dim, kH, kW)
    return None


def _check_shape_compatible(
    delta: torch.Tensor,
    base: torch.Tensor,
) -> bool:
    """
    Check whether a delta tensor can be shape-aligned to a base weight tensor.

    Non-raising predicate mirror of _align_delta_shape's logic. Used in
    _find_matching_keys() to verify shape compatibility BEFORE registering
    a match, preventing fallback strategies (energy_redirect, shape_block,
    etc.) from routing deltas to checkpoint keys with incompatible dims.

    Args:
        delta: LoRA delta tensor (typically 2D from SVD reconstruction)
        base: Checkpoint weight tensor (2D linear, 4D conv, or 1D bias)

    Returns:
        True if shapes can be aligned (exact match, conv reshape, or 4D→2D),
        False if fundamentally incompatible.
    """
    # Case 1: Exact match → always compatible
    if delta.shape == base.shape:
        return True

    # Case 2: 2D delta → 4D conv weight (uses shared dim analysis)
    conv_info = _check_2d_to_4d_conv(delta, base)
    if conv_info is not None:
        return True

    # Case 3: 4D delta → 2D base
    if delta.dim() == 4 and base.dim() == 2:
        return delta.numel() == base.numel()

    # Case 4: 1D (bias) — only exact match (caught by Case 1)
    return False


def _align_delta_shape(
    delta: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    """
    Align delta shape to match base tensor shape.

    Handles:
    - 2D delta → 4D conv weight (reshape delta to conv format)
    - 4D delta → 2D linear weight (reverse reshape)
    - Exact match passthrough

    Raises:
        ValueError: If shapes cannot be aligned (prevents silent OOM from
                   broadcasting incompatible shapes, e.g., 2D [320, 320]
                   delta applied to 4D [1280, 1280, 1, 1] base).
    """
    if delta.shape == base.shape:
        return delta

    # --- 2D delta → 4D conv weight (uses shared dim analysis) ---
    conv_info = _check_2d_to_4d_conv(delta, base)
    if conv_info is not None:
        tag, out_dim, in_dim_total, in_dim, kH, kW = conv_info
        if tag == 'conv_reshape':
            return delta.view(out_dim, in_dim, kH, kW)
        return delta.view(out_dim, in_dim_total, 1, 1).expand_as(base)

    if delta.dim() == 4 and base.dim() == 2:
        # Reverse: delta 4D → base 2D (shouldn't happen but handle gracefully)
        return delta.view(base.shape)

    # If shapes are incompatible, raise an error instead of returning
    # the mismatched delta (which would cause silent broadcasting explosions
    # or OOM crashes in the bake methods).
    raise ValueError(
        f"Cannot align delta shape {tuple(delta.shape)} to "
        f"base shape {tuple(base.shape)} — incompatible dimensions. "
        f"This key will be skipped."
    )


def _align_delta_safe(
    delta: torch.Tensor,
    base: torch.Tensor,
    ckpt_key: Optional[str] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Wrapper around _align_delta_shape for safe shape alignment.

    All keys (attention and non-attention) are treated equally — if the delta
    shape cannot be aligned to the base shape, a ValueError is raised and the
    key is skipped. This matches ComfyUI's behavior in calculate_weight()
    which warns and skips on any shape mismatch.

    Raises:
        ValueError: If shape alignment fails.
    """
    original_shape = delta.shape
    if delta.shape != base.shape and verbose:
        print(f"      [!] Shape mismatch for '{ckpt_key}': "
              f"delta={tuple(delta.shape)} vs base={tuple(base.shape)} "
              f"delta_dim={delta.dim()} base_dim={base.dim()}")
    try:
        result = _align_delta_shape(delta, base)
    except ValueError as e:
        raise ValueError(
            f"Cannot align delta shape {tuple(delta.shape)} to "
            f"base shape {tuple(base.shape)} for key '{ckpt_key}'. "
            f"delta_dim={delta.dim()} base_dim={base.dim()}. "
            f"This key will be skipped during baking."
        ) from e

    if result.shape != original_shape and verbose:
        print(f"      [i] Reshaped '{ckpt_key}': {tuple(original_shape)} -> {tuple(result.shape)} "
              f"(numel unchanged, reshape only)")
    return result


# ===================================================================
# Orthogonal Projection
# ===================================================================

def _orthogonal_projection_2d(
    base: torch.Tensor,
    delta: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """
    Compute orthogonal component: delta_ortho = delta - proj_base(delta)

    proj_base(delta) = (base · delta) / (base · base) × base

    For each row (output neuron), project the delta row onto the base row.
    """
    # Per-row projection: each row is independent
    base_norm_sq = (base * base).sum(dim=1, keepdim=True)
    projection_factor = (base * delta).sum(dim=1, keepdim=True) / (base_norm_sq + epsilon)
    delta_parallel = projection_factor * base
    delta_ortho = delta - delta_parallel
    return delta_ortho


def _orthogonal_projection(
    base: torch.Tensor,
    delta: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """
    Compute orthogonal component of delta relative to base.

    For 4D conv weights: flatten to 2D, project, reshape back.
    """
    if base.dim() == 4:
        # Conv weight: flatten to 2D
        base_flat = base.view(base.shape[0], -1)
        delta_flat = delta.view(base.shape[0], -1)
        delta_ortho = _orthogonal_projection_2d(base_flat, delta_flat, epsilon)
        return delta_ortho.view(base.shape)

    if base.dim() == 2:
        return _orthogonal_projection_2d(base, delta, epsilon)

    if base.dim() == 1:
        # Bias: scalar projection
        base_norm_sq = (base * base).sum()
        if base_norm_sq > epsilon:
            projection = (base * delta).sum() / base_norm_sq
            return delta - projection * base
        return delta

    # Fallback for other dimensions
    base_norm_sq = (base * base).sum()
    if base_norm_sq > epsilon:
        projection = (base * delta).sum() / base_norm_sq
        return delta - projection * base
    return delta


# ===================================================================
# Per-Component Weight Resolution
# ===================================================================

def _get_component_weight(
    key: str,
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
) -> float:
    """
    Determine per-component weight for a checkpoint key.

    Uses categorize_checkpoint_key to classify the key,
    then returns the appropriate weight parameter.
    Falls back to weight_unet for unrecognized components.
    """
    component = categorize_checkpoint_key(key)
    if component == 'te':
        return weight_te
    elif component == 'clip_vision':
        return weight_clip
    elif component == 'vae':
        return weight_vae
    elif component == 'model_diffusion':
        return weight_unet
    else:
        # 'other' — fall back to global UNet weight
        return weight_unet


# ===================================================================
# Shared Bake Helper
# ===================================================================

def _safe_bake_add(
    base: torch.Tensor,
    delta: torch.Tensor,
    scale: float,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute base + delta × scale in the specified compute dtype with NaN/Inf sanitization.

    Performs the arithmetic in the requested compute_dtype (default float32)
    for overflow safety, then clamps to float16 range before downcasting to
    the base tensor's original dtype.

    NOTE: FP8 base tensors are dequantized to bf16 upstream by the lazy
    mapping (see baker_node.py:286-288), so base.dtype is always bf16 or
    another non-FP8 dtype when this function is called.  The old FP8 base
    dequant branch has been removed as dead code.

    Args:
        base: Checkpoint weight tensor (any dtype, never fp8 at this point).
        delta: Aligned delta tensor (any dtype).
        scale: Combined strength × component_weight factor.
        compute_dtype: Torch dtype to use for computation. Defaults to float32.
                       When fp8 is selected, computation falls back to float16
                       or float32 as supported by the hardware.

    Returns:
        Result in base's dtype.  Always NaN/Inf-sanitized and clamped.
    """
    # Cast to compute dtype; if fp8, use float16 as compute (most GPUs don't
    # support native fp8 arithmetic on tensors)
    if compute_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        compute_dtype = torch.float16

    result = base.to(dtype=compute_dtype) + delta.to(dtype=compute_dtype) * scale
    result = torch.nan_to_num(result, nan=0.0, posinf=65504.0, neginf=-65504.0)
    result = torch.clamp(result, -65504.0, 65504.0)
    return result.to(dtype=base.dtype)


# ===================================================================
# Bake Key Preparation Helper
# ===================================================================

def _prepare_bake_key(
    self,
    ckpt_key: str,
    delta: torch.Tensor,
    ckpt_sd: Dict[str, torch.Tensor],
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[float], Optional[str]]:
    """Prepare a single bake key: validate, resolve weight, align shape.

    Returns (base, delta_adjusted, comp_weight, skip_reason):
      - If the key should be skipped: (None, None, None, reason_string)
      - If valid: (base_tensor, aligned_delta_tensor, component_weight, None)

    Encapsulates the preamble logic shared by all three bake methods:
      1. Key existence check
      2. Per-component weight resolution
      3. Device transfer
      4. Shape alignment with safe error handling

    NOTE: Raises are intentionally NOT caught here — the caller is expected
    to handle ValueError from _align_delta_safe when verbose logging of the
    error message is needed (e.g., bake_orthogonal prints the exception).
    """
    if ckpt_key not in ckpt_sd:
        return (None, None, None, "ckpt_key not in ckpt_sd")
    comp_weight = _get_component_weight(
        ckpt_key, weight_unet, weight_te, weight_clip, weight_vae
    )
    if comp_weight == 0.0:
        return (None, None, None, "component_weight_zero")
    # Move delta to GPU on-demand (no longer done upfront for all 112 at once — Fix #1)
    delta = delta.to(device=self.device)
    base = ckpt_sd[ckpt_key].to(device=self.device)
    # The original delta.to(device=self.device) was a no-op in practice since PyTorch's
    # .to() handles the index normalization. We keep the comment as documentation but
    # remove the assertion — comparing device types between str and torch.device with
    # different index semantics is fragile.
    try:
        delta_adjusted = _align_delta_safe(
            delta, base,
            ckpt_key=ckpt_key,
            verbose=getattr(self, '_verbose', False),
        )
    except ValueError:
        return (None, None, None,
                f"shape_mismatch: delta={tuple(delta.shape)} base={tuple(base.shape)}")
    return (base, delta_adjusted, comp_weight, None)


# ===================================================================
# Delta Statistics Accumulation Helper
# ===================================================================

def _accumulate_delta_stats(
    base: torch.Tensor,
    delta_adjusted: torch.Tensor,
    delta_norms: List[float],
    base_norms: List[float],
    delta_ratios: List[float],
) -> None:
    """Accumulate delta/base norms and ratio during bake loop.

    Call right after computing result, while base and delta_adjusted are
    still in GPU registers. Accumulates into caller-provided lists (mutated
    in-place).  Eliminates the need for a post-bake stats loop that re-reads
    all base tensors from disk (Fix #2).

    NOTE: Only accumulates finite values — NaN/Inf are silently skipped.
    """
    dn = torch.norm(delta_adjusted.float()).item()
    bn = torch.norm(base.float()).item()
    if math.isfinite(dn) and math.isfinite(bn):
        delta_norms.append(dn)
        base_norms.append(bn)
        delta_ratios.append(dn / max(bn, 1e-12))
    
    


# ===================================================================
# Three Baking Methods
# ===================================================================

def bake_linear(
    self,
    ckpt_sd: Dict[str, torch.Tensor],
    matched_deltas: Dict[str, torch.Tensor],
    output_dict: Dict[str, torch.Tensor],
    strength: float = 1.0,
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
    batch_size: int = 64,
) -> Tuple[List[float], List[float]]:
    """
    Standard linear baking: ckpt[key] += delta[key] × strength × component_weight.

    Each matched key is scaled per-component (weight_unet, weight_te, etc.).
    Use weight_te=0.0 to skip Text Encoder keys.
    Computes in float32 with float16 range clamp to prevent overflow.

    Writes directly to `output_dict` (Fix #3) and accumulates delta statistics
    during the bake loop (Fix #2), eliminating the post-bake stats loop.

    Returns:
        (delta_norms, delta_ratios) — lists of per-key norm values for forensic reporting.
    """
    print(f"   🧪 Baking method: Linear (strength={strength})")
    print(f"   🧱 Weight Block Map — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}")
    if not hasattr(self, '_bake_dropped'):
        self._bake_dropped = {}

    keys = list(matched_deltas.keys())  # strings only — no tensor refs (Fix #7)
    total = len(keys)
    delta_norms: List[float] = []
    delta_ratios: List[float] = []
    base_norms: List[float] = []
    num_batches = (total + batch_size - 1) // batch_size
    with ProgressTracker(total=total, desc="Baking (linear)") as bake_progress:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_keys = keys[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1

            for ckpt_key in batch_keys:
                delta = matched_deltas.pop(ckpt_key)  # pop — refcount drops to zero after del
                base, delta_adjusted, comp_weight, skip_reason = _prepare_bake_key(
                    self, ckpt_key, delta, ckpt_sd,
                    weight_unet, weight_te, weight_clip, weight_vae,
                )
                if skip_reason is not None:
                    self._bake_dropped[ckpt_key] = skip_reason
                    if getattr(self, '_verbose', False) and skip_reason.startswith("shape_mismatch"):
                        print(f"      ⚠️ Skipping {ckpt_key}: {skip_reason}")
                    del delta
                    bake_progress += 1
                    continue

                result = _safe_bake_add(base, delta_adjusted, strength * comp_weight,
                                         compute_dtype=self.dtype)
                result_cpu = result.cpu()
                output_dict[ckpt_key] = result_cpu  # Write directly to output dict (Fix #3)
                # Accumulate delta statistics during bake (Fix #2)
                _accumulate_delta_stats(base, delta_adjusted, delta_norms, base_norms, delta_ratios)
                # Explicitly free CUDA intermediates and consumed delta immediately
                del base, delta_adjusted, result, delta
                bake_progress += 1
            # ── Per-batch RAM profile (peak usage, before cleanup — matches weaver pattern) ──
            _avail = get_available_ram()
            if _avail is not None:
                print(f"      📊 RAM: {_avail / (1024**3):.2f} GB available  [Batch {batch_num}/{num_batches}]")
            # Batch-level cleanup: release GPU tensors before next batch
            cleanup_memory()
            comfyui_yield()
    return (delta_norms, delta_ratios)


def bake_impact_weighted(
    self,
    ckpt_sd: Dict[str, torch.Tensor],
    matched_deltas: Dict[str, torch.Tensor],
    output_dict: Dict[str, torch.Tensor],
    strength: float = 1.0,
    energy_concentration: float = 0.80,
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
    batch_size: int = 64,
) -> Tuple[List[float], List[float]]:
    """
    Impact-Weighted baking: only bake into layers carrying most LoRA energy.

    Uses per-component scaling. For each matched key, compute tensor
    energy = mean(delta²). Sort by energy descending. Find top N capturing
    `energy_concentration` (default 80%) of cumulative energy. Bake only
    into those primary driver layers. Non-primary layers get reduced
    strength × 0.25 or are skipped.
    Computes in float32 with float16 range clamp to prevent overflow.

    Writes directly to `output_dict` (Fix #3) and accumulates delta statistics
    during the bake loop (Fix #2), eliminating the post-bake stats loop.

    Returns:
        (delta_norms, delta_ratios) — lists of per-key norm values for forensic reporting.
    """
    print(f"   🧪 Baking method: Impact-Weighted (energy_concentration={energy_concentration})")
    print(f"   🧱 Weight Block Map — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}")

    # --- Single-pass: filter, align, compute energy (cache results) ---
    # Cache (ckpt_key, base, aligned_delta, energy) to avoid re-running
    # shape alignment in the bake loop — saves ~50% compute.
    key_data: List[Tuple[str, torch.Tensor, torch.Tensor, float]] = []
    key_list = list(matched_deltas.items())
    total_keys = len(key_list)
    with ProgressTracker(total=total_keys, desc="Analyzing impact energy") as energy_progress:
        for batch_start in range(0, total_keys, batch_size):
            batch_end = min(batch_start + batch_size, total_keys)
            batch_items = key_list[batch_start:batch_end]

            for ckpt_key, delta in batch_items:
                base, delta_adjusted, comp_weight, skip_reason = _prepare_bake_key(
                    self, ckpt_key, delta, ckpt_sd,
                    weight_unet, weight_te, weight_clip, weight_vae,
                )
                if skip_reason is not None:
                    if skip_reason.startswith("shape_mismatch"):
                        print(f"      ⚠️ Skipping {ckpt_key} in energy analysis: {skip_reason}")
                    energy_progress += 1
                    continue

                energy = delta_adjusted.pow(2).mean().item()
                key_data.append((ckpt_key, base, delta_adjusted, energy))
                energy_progress += 1
            # Batch-level cleanup: release GPU tensors before next batch
            del batch_items
            cleanup_memory()
            comfyui_yield()

    if not key_data:
        print("   ⚠️  No valid keys for impact-weighted baking")
        return ([], [])

    # Sort by energy descending
    key_data.sort(key=lambda x: x[3], reverse=True)
    total_energy = sum(e for _, _, _, e in key_data)
    if total_energy <= 0:
        print("   ⚠️  Zero total energy — falling back to linear bake")
        return bake_linear(
            self, ckpt_sd, matched_deltas, output_dict, strength,
            weight_unet, weight_te, weight_clip, weight_vae,
            batch_size=batch_size,
        )

    # ── Free original delta tensors — no longer needed (Fix #7) ──
    # key_data already caches aligned (base, delta_adjusted) tensors, so the
    # original deltas in matched_deltas and their key_list wrappers are redundant.
    del key_list
    matched_deltas.clear()
    cleanup_memory()

    # Find primary driver set
    cumulative = 0.0
    primary_keys: Set[str] = set()
    for ckpt_key, _, _, energy in key_data:
        primary_keys.add(ckpt_key)
        cumulative += energy
        if cumulative / total_energy >= energy_concentration:
            break

    reduced_strength = strength * 0.25
    print(f"   📊 Primary drivers: {len(primary_keys)}/{len(key_data)} keys ({cumulative/total_energy:.1%} energy)")

    # 🔥 Move cached base+delta tensors to CPU to free VRAM before bake loop.
    # During energy analysis, _prepare_bake_key loaded them to CUDA and they've
    # been held in key_data ever since — potentially gigabytes of VRAM.
    # They'll be moved back to device on-demand in the bake loop below.
    key_data = [(k, b.cpu(), d.cpu(), e) for k, b, d, e in key_data]
    cleanup_memory()
    comfyui_yield()

    # --- Bake loop (reuses cached aligned deltas) ---
    delta_norms: List[float] = []
    delta_ratios: List[float] = []
    base_norms: List[float] = []
    total_bake = len(key_data)
    num_batches = (total_bake + batch_size - 1) // batch_size
    with ProgressTracker(total=total_bake, desc="Baking (impact-weighted)") as bake_progress:
        for batch_start in range(0, total_bake, batch_size):
            batch_end = min(batch_start + batch_size, total_bake)
            batch_items = key_data[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1

            for ckpt_key, base, delta_adjusted, _ in batch_items:
                # Move cached CPU tensors back to device for computation
                base = base.to(device=self.device)
                delta_adjusted = delta_adjusted.to(device=self.device)
                comp_weight = _get_component_weight(
                    ckpt_key, weight_unet, weight_te, weight_clip, weight_vae
                )
                effective_strength = strength if ckpt_key in primary_keys else reduced_strength
                result = _safe_bake_add(base, delta_adjusted, effective_strength * comp_weight,
                                         compute_dtype=self.dtype)
                result_cpu = result.cpu()
                output_dict[ckpt_key] = result_cpu  # Write directly to output dict (Fix #3)
                # Accumulate delta statistics during bake (Fix #2)
                _accumulate_delta_stats(base, delta_adjusted, delta_norms, base_norms, delta_ratios)
                # Explicitly free CUDA intermediates immediately
                del base, delta_adjusted, result
                bake_progress += 1
            # ── Per-batch RAM profile (peak usage, before cleanup — matches weaver pattern) ──
            _avail = get_available_ram()
            if _avail is not None:
                print(f"      📊 RAM: {_avail / (1024**3):.2f} GB available  [Batch {batch_num}/{num_batches}]")
            # Batch-level cleanup: release GPU tensors before next batch
            del batch_items
            cleanup_memory()
            comfyui_yield()
    return (delta_norms, delta_ratios)


def bake_orthogonal(
    self,
    ckpt_sd: Dict[str, torch.Tensor],
    matched_deltas: Dict[str, torch.Tensor],
    output_dict: Dict[str, torch.Tensor],
    strength: float = 1.0,
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
    batch_size: int = 64,
) -> Tuple[List[float], List[float]]:
    """
    Orthogonal baking: project delta onto orthogonal complement of base weights.

    delta_orthogonal = delta - proj_base(delta)
    where proj_base(delta) = (sum(base * delta) / sum(base²)) × base

    Applies per-component scaling after orthogonal projection.
    For 4D conv weights: flatten to 2D, project, reshape back.
    Use weight_te=0.0 to skip Text Encoder keys.
    Computes in float32 with float16 range clamp to prevent overflow.

    Writes directly to `output_dict` (Fix #3) and accumulates delta statistics
    during the bake loop (Fix #2), eliminating the post-bake stats loop.

    Returns:
        (delta_norms, delta_ratios) — lists of per-key norm values for forensic reporting.
    """
    print(f"   🧪 Baking method: Orthogonal (strength={strength})")
    print(f"   🧱 Weight Block Map — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}")

    keys = list(matched_deltas.keys())  # strings only — no tensor refs (Fix #7)
    total = len(keys)
    delta_norms: List[float] = []
    delta_ratios: List[float] = []
    base_norms: List[float] = []
    num_batches = (total + batch_size - 1) // batch_size
    with ProgressTracker(total=total, desc="Baking (orthogonal)") as bake_progress:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_keys = keys[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1

            for ckpt_key in batch_keys:
                delta = matched_deltas.pop(ckpt_key)  # pop — refcount drops to zero after del
                base, delta_adjusted, comp_weight, skip_reason = _prepare_bake_key(
                    self, ckpt_key, delta, ckpt_sd,
                    weight_unet, weight_te, weight_clip, weight_vae,
                )
                if skip_reason is not None:
                    if skip_reason.startswith("shape_mismatch"):
                        print(f"      ⚠️ Skipping {ckpt_key}: {skip_reason}")
                    del delta
                    bake_progress += 1
                    continue

                # Orthogonal projection
                delta_ortho = _orthogonal_projection(base, delta_adjusted)

                result = _safe_bake_add(base, delta_ortho, strength * comp_weight,
                                         compute_dtype=self.dtype)
                result_cpu = result.cpu()
                output_dict[ckpt_key] = result_cpu  # Write directly to output dict (Fix #3)
                # Accumulate delta statistics during bake (Fix #2)
                _accumulate_delta_stats(base, delta_adjusted, delta_norms, base_norms, delta_ratios)
                # Explicitly free CUDA intermediates and consumed delta immediately
                del base, delta_adjusted, delta_ortho, result, delta
                bake_progress += 1
            # ── Per-batch RAM profile (peak usage, before cleanup — matches weaver pattern) ──
            _avail = get_available_ram()
            if _avail is not None:
                print(f"      📊 RAM: {_avail / (1024**3):.2f} GB available  [Batch {batch_num}/{num_batches}]")
            # Batch-level cleanup: release GPU tensors before next batch
            cleanup_memory()
            comfyui_yield()
    return (delta_norms, delta_ratios)


# ===================================================================
# Output Assembly and Safety
# ===================================================================

def _assemble_output(
    original_ckpt_sd: Dict[str, torch.Tensor],
    baked_keys: Dict[str, torch.Tensor],
    fp8_dequantized: bool = False,
    target_dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    Assemble final state dict with VAE pass-through.

    CRITICAL: Start with a full copy of the original checkpoint state dict.
    Only overwrite keys that were actually baked.
    All unrecognized keys (VAE, CLIP, embeddings, etc.) pass through unchanged.

    Args:
        fp8_dequantized: When True, the checkpoint had FP8 native weights that
            were dequantized to bfloat16 for baking.  In this case, cast to the
            user's target_dtype rather than the original checkpoint dtype (which
            is FP8 and would re-quantize the delta to zero).
        target_dtype: User's chosen output precision (e.g., bfloat16, float16).
            Used only when fp8_dequantized=True.  If None, keep the baked tensor's
            dtype (which is already bfloat16 from Fix 1 in _safe_bake_add).
    """
    output_sd = dict(original_ckpt_sd)
    # 🔥 FP8: Strip stale artifact keys when checkpoint was dequantized
    if fp8_dequantized:
        stale_keys = [k for k in output_sd
                      if k.endswith(('.weight_scale', '.input_scale', '.comfy_quant'))]
        for k in stale_keys:
            del output_sd[k]
        if stale_keys:
            print(f"   🧹 Removed {len(stale_keys)} stale FP8 artifact keys from output")
    for key, tensor in baked_keys.items():
        if key in output_sd:
            if fp8_dequantized:
                # FP8 checkpoint was dequantized — cast to user's target dtype
                # or keep as-is (bfloat16 from Fix 1) if no target specified.
                if target_dtype is not None:
                    output_sd[key] = tensor.to(
                        device=output_sd[key].device,
                        dtype=target_dtype,
                    )
                else:
                    output_sd[key] = tensor.to(device=output_sd[key].device)
            else:
                # Normal path: match the original checkpoint's dtype
                output_sd[key] = tensor.to(
                    device=output_sd[key].device,
                    dtype=output_sd[key].dtype,
                )
    preserved = len(output_sd) - len(baked_keys)
    print(f"   📦 Assembled output: {len(baked_keys)} baked + {preserved} preserved (VAE/etc.)")
    return output_sd


def _assemble_output_lazy(
    ckpt_path: Path,
    ckpt_header: Dict[str, Any],
    baked_keys: Dict[str, torch.Tensor],
    detected_fp8: bool = False,
    user_fp8_dtype: Optional[torch.dtype] = None,
) -> Any:  # Returns _LazyCheckpointMapping (dict-like)
    """
    Assemble output as a lazy mapping with baked key overlay.

    Instead of copying all non-baked tensors into a dict (17 GiB for Klein 9B),
    create a lazy mapping that loads non-baked tensors on demand from the
    original checkpoint file via mmap.

    Baked tensors are stored in the write cache (already in RAM).
    Compatible with all downstream usage (save_safetensors_stream,
    load_state_dict_as_model_objects).

    When *user_fp8_dtype* is set (torch.float8_e4m3fn or torch.float8_e5m2),
    the output is built in FP8 mode:
      - FP8 source: preserved keys stay FP8 from file (no FP8→BF16→FP8 round trip),
        companion scales pass through. Only stale .comfy_quant keys are stripped.
      - BF16/FP16 source: preserved keys stay at original file dtype. The
        post-assembly FP8 conversion block (in baking_processor.py) converts them.
      - Baked keys are quantized via quantize_weight_to_fp8_with_scales() with
        proper companion scale injection (matching Creator-quality algorithm).

    Args:
        ckpt_path: Path to the original checkpoint .safetensors file.
        ckpt_header: Full safetensors header dict (tensor names -> shape/dtype/offsets).
        baked_keys: Dict of {checkpoint_key: baked_tensor} to overlay.
        detected_fp8: When True, the checkpoint has FP8 native weights.
        user_fp8_dtype: When set, produce FP8 output using creator-quality
            scale-then-cast quantization.

    Returns:
        _LazyCheckpointMapping with baked keys in write cache and all other
        keys lazy-loaded from the original file.
    """
    from .musubi_checkpoint_studio import MusubiCheckpointStudio

    metadata = ckpt_header.get('__metadata__', {})
    want_fp8_output = user_fp8_dtype is not None

    if want_fp8_output:
        # ═══════════════════════════════════════════════════════════════
        # FP8 OUTPUT MODE — handles BOTH FP8 source and BF16/FP16 source
        # ═══════════════════════════════════════════════════════════════

        # ── Step 1: Create lazy mapping ──
        if detected_fp8:
            # FP8 source: preserved keys stay FP8 from file (no dequant).
            # Companion scales from file header pass through to output
            # (items() includes them naturally).
            mapping = MusubiCheckpointStudio._LazyCheckpointMapping(
                ckpt_path, metadata,
                target_dtype=None,  # No dequant — preserve FP8 as-is
            )
        else:
            # BF16/FP16 source: preserved keys stay original from file.
            # Post-assembly block will convert them to FP8 sequentially.
            mapping = MusubiCheckpointStudio._LazyCheckpointMapping(
                ckpt_path, metadata,
                target_dtype=None,
            )

        # ── Step 2: Quantize baked keys to FP8 ──
        fp8_converted = 0
        bf16_preserved = 0
        for key, tensor in baked_keys.items():
            if key.endswith('.weight') and not should_preserve_bf16(key):
                q, wscale, iscale = quantize_weight_to_fp8_with_scales_optimized(
                    tensor, user_fp8_dtype
                )
                mapping[key] = q
                base_key = key[:-len('.weight')]
                mapping[base_key + '.weight_scale'] = wscale
                mapping[base_key + '.input_scale'] = iscale
                fp8_converted += 1
            elif should_preserve_bf16(key):
                mapping[key] = tensor.to(dtype=torch.bfloat16)
                bf16_preserved += 1
            else:
                mapping[key] = tensor.to(dtype=user_fp8_dtype)

        # ── Step 3: Stale FP8 key cleanup ──
        # Only strip .comfy_quant — weight_scale/input_scale are needed for
        # FP8 passthrough in the saved output file.
        if detected_fp8:
            mapping._ensure_open()
            stale = [k for k in mapping._header
                     if k.endswith(('.comfy_quant', '.weight_scale', '.input_scale'))]
            for k in stale:
                mapping.pop(k, None)
            if stale:
                print(f"   🧹 Removed {len(stale)} stale FP8 artifact keys")

        # Flag so post-assembly code (baking_processor.py) knows FP8 was handled
        mapping._user_fp8_mode = True
        # Store the fp8 dtype string for _quantization_metadata building
        fp8_dtype_str = 'F8_E4M3' if user_fp8_dtype == torch.float8_e4m3fn else 'F8_E5M2'
        mapping._user_fp8_dtype_str = fp8_dtype_str

        print(f"   🎯 FP8 lazy assembly: {fp8_converted} FP8 quantized + "
              f"{bf16_preserved} BF16 preserved baked keys")

    else:
        # ═══════════════════════════════════════════════════════════════
        # ORIGINAL BEHAVIOR — no FP8 output requested
        # ═══════════════════════════════════════════════════════════════
        mapping = MusubiCheckpointStudio._LazyCheckpointMapping(
            ckpt_path, metadata,
            target_dtype=torch.bfloat16 if detected_fp8 else None,
        )
        for key, tensor in baked_keys.items():
            mapping[key] = tensor

        # Strip stale FP8 artifact keys (weight_scale, input_scale, comfy_quant)
        # meaningless after dequantization to bfloat16.
        if detected_fp8:
            mapping._ensure_open()
            stale_keys = [k for k in mapping._header
                          if k.endswith(('.weight_scale', '.input_scale', '.comfy_quant'))]
            for k in stale_keys:
                mapping.pop(k, None)
            if stale_keys:
                print(f"   🧹 Removed {len(stale_keys)} stale FP8 artifact keys from output")

    preserved = len(mapping) - len(baked_keys)
    print(f"   📦 Assembled output (lazy): {len(baked_keys)} baked + {preserved} preserved via mmap")
    return mapping


def _sanitize_or_revert(
    tensor: torch.Tensor,
    key: str,
    original_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Check and sanitize NaN/Inf values.

    🔥 REVERT-TO-ORIGINAL POLICY: If original_tensor is provided and the baked
    tensor contains NaN/Inf, REVERT to the original weight instead of replacing
    NaN with zero. Zero replacements cause black/dark patches in generated
    images because the layer's contribution becomes zero.

    NOTE: FP8 dtypes (float8_e4m3fn, float8_e5m2) don't support torch.isinf()
    or torch.isnan() — cast to float32 for detection when needed.
    """
    # FP8 dtypes lack isinf/isnan kernels — cast to float32 for detection
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        check_tensor = tensor.float()
    else:
        check_tensor = tensor

    if torch.isnan(check_tensor).any() or torch.isinf(check_tensor).any():
        if original_tensor is not None:
            print(f"      ⚠️ NaN/Inf detected in {key} — REVERTING TO ORIGINAL")
            return original_tensor.to(dtype=tensor.dtype, device=tensor.device)
        else:
            print(f"      ⚠️ NaN/Inf detected in {key} — sanitizing with nan_to_num")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    return tensor


def _sanitize_baked(
    baked: Dict[str, torch.Tensor],
    original_ckpt_sd: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sanitize all baked tensors for NaN/Inf.

    If original_ckpt_sd is provided, any key with NaN/Inf will be REVERTED
    to its original checkpoint weight rather than replaced with zeros.
    """
    for key in list(baked.keys()):
        orig = None
        if original_ckpt_sd is not None and key in original_ckpt_sd:
            orig = original_ckpt_sd[key]
        baked[key] = _sanitize_or_revert(baked[key], key, original_tensor=orig)
    return baked


# ===================================================================
# Metadata and Forensic Report
# ===================================================================

def _build_metadata(
    baking_method: str,
    strength: float,
    lora_source: str,
    checkpoint_name: str,
    matched_count: int,
    preserved_count: int,
    original_metadata: Optional[Dict[str, str]],
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
    metadata_mode: str = "preserve_a",
) -> Dict[str, str]:
    """Build baking metadata using the unified metadata_factory.

    Args:
        metadata_mode: Controls original metadata merging behavior.
            "none"       — Discard original metadata; baking signature only.
            "preserve_a" — Keep original, original has priority over signature.
            "preserve_b" — Same as preserve_a (symmetrical at single-source level).
            "merge_basic" — Baking signature has priority over original.
    """
    # 🔥 FP8: Strip quantization metadata from original checkpoint before
    # passing to finalize_metadata.  FP8 checkpoints have metadata like
    # "quantization metadata version 1" that tells ComfyUI to use the slow
    # mixed precision path.  Since we dequantize to bfloat16, this metadata
    # is stale and must be removed (safety net — primary stripping happens
    # in baker_node.py:252 Fix 2).
    if original_metadata:
        cleaned_metadata = {
            k: v for k, v in original_metadata.items()
            if not any(q in k.lower() for q in ['quantization', 'scale'])
        }
    else:
        cleaned_metadata = None

    return finalize_metadata(
        metadata=cleaned_metadata,
        mode=metadata_mode,
        component="baker",
        extra_fields={
            "baking_method": baking_method,
            "baking_strength": str(strength),
            "baked_lora_source": lora_source,
            "source_checkpoint": checkpoint_name,
            "baked_key_count": str(matched_count),
            "preserved_key_count": str(preserved_count),
            "weight_unet": str(weight_unet),
            "weight_te": str(weight_te),
            "weight_clip": str(weight_clip),
            "weight_vae": str(weight_vae),
            "metadata_mode": metadata_mode,
        }
    )


def _build_forensic_report(
    baking_method: str,
    strength: float,
    lora_source: str,
    checkpoint_name: str,
    matched_count: int,
    preserved_count: int,
    total_ckpt_keys: int,
    impact_profile: Optional[Dict[str, Any]] = None,
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
    component_breakdown: Optional[Dict[str, Any]] = None,
    delta_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a human-readable forensic report using the shared builder."""
    title_data = OrderedDict([
        ("📦 BAKED LORA", lora_source),
        ("🎯 SOURCE CHECKPOINT", checkpoint_name),
        ("🔧 BAKING METHOD", f"{baking_method} (strength={strength})"),
        ("📋 SUMMARY", f"{baking_method} | {matched_count} modified, {preserved_count} preserved, {total_ckpt_keys} total | UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}"),
    ])

    sections: List[Tuple[Optional[str], List[str]]] = []

    # Raw header continuation (not passed through title_data to avoid extra colon)
    sections.append((None, [
        f'🧱 WEIGHT BLOCK MAP — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}',
        f'📅 BAKING DATE: {time.strftime("%Y-%m-%d %H:%M:%S")}',
    ]))

    if impact_profile:
        ip = impact_profile
        impact_lines: List[str] = []
        impact_lines.append(f'   Method: {ip.get("baking_method", baking_method)}')
        impact_lines.append(f'   Matched: {ip.get("matched_keys", matched_count)} keys')
        impact_lines.append(f'   Baked: {ip.get("baked_keys", matched_count)} keys')
        impact_lines.append(f'   Preserved: {ip.get("preserved_keys", preserved_count)} keys')
        impact_lines.append(f'   Total: {ip.get("total_keys", total_ckpt_keys)} keys')
        impact_lines.append(f'   Strength: {ip.get("strength", strength)}')
        wu = ip.get('weight_unet', weight_unet)
        wt = ip.get('weight_te', weight_te)
        wc = ip.get('weight_clip', weight_clip)
        wv = ip.get('weight_vae', weight_vae)
        impact_lines.append(f'   🧱 Weights — UNet:{wu} TE:{wt} CLIP:{wc} VAE:{wv}')
        sections.append(("IMPACT PROFILE", impact_lines))

    if component_breakdown:
        comp_lines = _build_component_breakdown(
            component_breakdown,
            icons={'unet': '🔷', 'te': '📝', 'clip': '📷', 'vae': '🎬'},
            comp_order=['unet', 'te', 'clip', 'vae'],
        )
        sections.append(("COMPONENT BREAKDOWN", comp_lines))

    if delta_analysis:
        da = delta_analysis
        delta_lines: List[str] = []
        delta_lines.append(f'   Layers analyzed: {da.get("layers_analyzed", 0)}')
        delta_lines.append(f'   Mean ratio: {da.get("mean_ratio_percent", 0.0)}%')
        delta_lines.append(f'   Median ratio: {da.get("median_ratio_percent", 0.0)}%')
        delta_lines.append(f'   Max ratio: {da.get("max_ratio_percent", 0.0)}%')
        delta_lines.append(f'   Std ratio: {da.get("std_ratio_percent", 0.0)}%')
        delta_lines.append(f'   Mean delta norm: {da.get("mean_delta_norm", 0.0)}')
        delta_lines.append(f'   Max delta norm: {da.get("max_delta_norm", 0.0)}')
        effect_visual = da.get('effect_visual', '')
        if effect_visual:
            delta_lines.append(f'   Effect visual: {effect_visual}')
        sections.append(("DELTA ANALYSIS", delta_lines))

    return build_forensic_report(
        report_type="EASY LoRA BAKER",
        title_data=title_data,
        sections=sections,
        footer_width=50,
    )


