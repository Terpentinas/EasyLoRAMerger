"""
Baking methods and shape alignment utilities.

Extracted from baking_processor.py. Contains:
  - Shape compatibility checking and alignment
  - Attention key detection (anti-auto-scale guard)
  - Three baking methods: linear, impact_weighted, orthogonal
  - Output assembly, NaN/Inf sanitization, metadata and forensic reporting
"""

from collections import OrderedDict
import torch
import time
from typing import Dict, List, Optional, Tuple, Set, Any

from ..utils import (
    categorize_checkpoint_key,
    ProgressTracker,
    comfyui_yield,
    memory_guard,
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

    Args:
        base: Checkpoint weight tensor (any dtype).
        delta: Aligned delta tensor (any dtype).
        scale: Combined strength × component_weight factor.
        compute_dtype: Torch dtype to use for computation. Defaults to float32.
                       When fp8 is selected, computation falls back to float16
                       or float32 as supported by the hardware.

    Returns:
        Result tensor in base's original dtype, NaN/Inf-sanitized and clamped.
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
    base = ckpt_sd[ckpt_key].to(device=self.device)
    # delta is already on device — moved at baking_processor.py:344-346
    # NOTE: delta.device may be 'cuda:0' while self.device is 'cuda' (string without index).
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
# Three Baking Methods
# ===================================================================

def bake_linear(
    self,
    ckpt_sd: Dict[str, torch.Tensor],
    matched_deltas: Dict[str, torch.Tensor],
    strength: float = 1.0,
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Standard linear baking: ckpt[key] += delta[key] × strength × component_weight.

    Each matched key is scaled per-component (weight_unet, weight_te, etc.).
    Use weight_te=0.0 to skip Text Encoder keys.
    Computes in float32 with float16 range clamp to prevent overflow.

    NOTE: This function receives `self` for access to self.device, self._verbose,
    self._bake_dropped.
    """
    print(f"   🧪 Baking method: Linear (strength={strength})")
    print(f"   🧱 Weight Block Map — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}")
    baked: Dict[str, torch.Tensor] = {}
    if not hasattr(self, '_bake_dropped'):
        self._bake_dropped = {}

    key_list = list(matched_deltas.items())
    total = len(key_list)
    with ProgressTracker(total=total, desc="Baking (linear)") as bake_progress:
        for batch_start in range(0, total, batch_size):
            memory_guard()
            batch_end = min(batch_start + batch_size, total)
            batch_items = key_list[batch_start:batch_end]

            # ⚡ Pre-fetch all base tensors for this batch — fewer CUDA round-trips
            for ckpt_key, _ in batch_items:
                _ = ckpt_sd[ckpt_key].to(device=self.device)

            for ckpt_key, delta in batch_items:
                base, delta_adjusted, comp_weight, skip_reason = _prepare_bake_key(
                    self, ckpt_key, delta, ckpt_sd,
                    weight_unet, weight_te, weight_clip, weight_vae,
                )
                if skip_reason is not None:
                    self._bake_dropped[ckpt_key] = skip_reason
                    if getattr(self, '_verbose', False) and skip_reason.startswith("shape_mismatch"):
                        print(f"      ⚠️ Skipping {ckpt_key}: {skip_reason}")
                    bake_progress += 1
                    continue

                baked[ckpt_key] = _safe_bake_add(base, delta_adjusted, strength * comp_weight,
                                                  compute_dtype=self.dtype)
                bake_progress += 1
            comfyui_yield()
    return baked


def bake_impact_weighted(
    self,
    ckpt_sd: Dict[str, torch.Tensor],
    matched_deltas: Dict[str, torch.Tensor],
    strength: float = 1.0,
    energy_concentration: float = 0.80,
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Impact-Weighted baking: only bake into layers carrying most LoRA energy.

    Uses per-component scaling. For each matched key, compute tensor
    energy = mean(delta²). Sort by energy descending. Find top N capturing
    `energy_concentration` (default 80%) of cumulative energy. Bake only
    into those primary driver layers. Non-primary layers get reduced
    strength × 0.25 or are skipped.
    Computes in float32 with float16 range clamp to prevent overflow.

    NOTE: This function receives `self` for access to self.device, self._verbose.
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
            memory_guard()
            batch_end = min(batch_start + batch_size, total_keys)
            batch_items = key_list[batch_start:batch_end]

            # ⚡ Pre-fetch all base tensors for this batch
            for ckpt_key, _ in batch_items:
                _ = ckpt_sd[ckpt_key].to(device=self.device)

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
            comfyui_yield()

    if not key_data:
        print("   ⚠️  No valid keys for impact-weighted baking")
        return {}

    # Sort by energy descending
    key_data.sort(key=lambda x: x[3], reverse=True)
    total_energy = sum(e for _, _, _, e in key_data)
    if total_energy <= 0:
        print("   ⚠️  Zero total energy — falling back to linear bake")
        return bake_linear(
            self, ckpt_sd, matched_deltas, strength,
            weight_unet, weight_te, weight_clip, weight_vae,
        )

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

    # --- Bake loop (reuses cached aligned deltas) ---
    baked: Dict[str, torch.Tensor] = {}
    total_bake = len(key_data)
    with ProgressTracker(total=total_bake, desc="Baking (impact-weighted)") as bake_progress:
        for batch_start in range(0, total_bake, batch_size):
            memory_guard()
            batch_end = min(batch_start + batch_size, total_bake)
            batch_items = key_data[batch_start:batch_end]

            # ⚡ Pre-fetch all base tensors for this batch
            for ckpt_key, _, _, _ in batch_items:
                _ = ckpt_sd[ckpt_key].to(device=self.device)

            for ckpt_key, base, delta_adjusted, _ in batch_items:
                comp_weight = _get_component_weight(
                    ckpt_key, weight_unet, weight_te, weight_clip, weight_vae
                )
                effective_strength = strength if ckpt_key in primary_keys else reduced_strength
                baked[ckpt_key] = _safe_bake_add(base, delta_adjusted, effective_strength * comp_weight,
                                                  compute_dtype=self.dtype)
                bake_progress += 1
            comfyui_yield()
    return baked


def bake_orthogonal(
    self,
    ckpt_sd: Dict[str, torch.Tensor],
    matched_deltas: Dict[str, torch.Tensor],
    strength: float = 1.0,
    weight_unet: float = 1.0,
    weight_te: float = 1.0,
    weight_clip: float = 1.0,
    weight_vae: float = 1.0,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Orthogonal baking: project delta onto orthogonal complement of base weights.

    delta_orthogonal = delta - proj_base(delta)
    where proj_base(delta) = (sum(base * delta) / sum(base²)) × base

    Applies per-component scaling after orthogonal projection.
    For 4D conv weights: flatten to 2D, project, reshape back.
    Use weight_te=0.0 to skip Text Encoder keys.
    Computes in float32 with float16 range clamp to prevent overflow.
    """
    print(f"   🧪 Baking method: Orthogonal (strength={strength})")
    print(f"   🧱 Weight Block Map — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}")

    baked: Dict[str, torch.Tensor] = {}
    key_list = list(matched_deltas.items())
    total = len(key_list)

    with ProgressTracker(total=total, desc="Baking (orthogonal)") as bake_progress:
        for batch_start in range(0, total, batch_size):
            memory_guard()
            batch_end = min(batch_start + batch_size, total)
            batch_items = key_list[batch_start:batch_end]

            # ⚡ Pre-fetch all base tensors for this batch — fewer CUDA round-trips
            for ckpt_key, _ in batch_items:
                _ = ckpt_sd[ckpt_key].to(device=self.device)

            for ckpt_key, delta in batch_items:
                base, delta_adjusted, comp_weight, skip_reason = _prepare_bake_key(
                    self, ckpt_key, delta, ckpt_sd,
                    weight_unet, weight_te, weight_clip, weight_vae,
                )
                if skip_reason is not None:
                    if skip_reason.startswith("shape_mismatch"):
                        print(f"      ⚠️ Skipping {ckpt_key}: {skip_reason}")
                    bake_progress += 1
                    continue

                # Orthogonal projection
                delta_ortho = _orthogonal_projection(base, delta_adjusted)

                baked[ckpt_key] = _safe_bake_add(base, delta_ortho, strength * comp_weight,
                                                  compute_dtype=self.dtype)
                bake_progress += 1
            comfyui_yield()
    return baked


# ===================================================================
# Output Assembly and Safety
# ===================================================================

def _assemble_output(
    original_ckpt_sd: Dict[str, torch.Tensor],
    baked_keys: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Assemble final state dict with VAE pass-through.

    CRITICAL: Start with a full copy of the original checkpoint state dict.
    Only overwrite keys that were actually baked.
    All unrecognized keys (VAE, CLIP, embeddings, etc.) pass through unchanged.
    """
    output_sd = dict(original_ckpt_sd)
    for key, tensor in baked_keys.items():
        if key in output_sd:
            output_sd[key] = tensor.to(
                device=output_sd[key].device,
                dtype=output_sd[key].dtype,
            )
    preserved = len(output_sd) - len(baked_keys)
    print(f"   📦 Assembled output: {len(baked_keys)} baked + {preserved} preserved (VAE/etc.)")
    return output_sd


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
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
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
    return finalize_metadata(
        metadata=original_metadata,
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


