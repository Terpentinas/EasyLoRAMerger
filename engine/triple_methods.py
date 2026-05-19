"""
Triple LoRA Merging Methods for EasyLoRAMerger - Version 1.3.0
Specialized algorithms for merging three LoRAs simultaneously.

This module is now a thin wrapper around engine.triple_merge_core,
preserving the same public API (merge_triple_method returning 4 values)
while eliminating ~700 lines of duplicated merge function bodies.
"""

import math
from typing import List, Dict, Optional, Tuple

import torch

from ..utils import (
    safe_get_rank,
    silent_pad_or_truncate,
    categorize_key,
    compute_component_energy_ratios,
    ProgressTracker,
    DeviceManager,
)
from .methods import apply_magnitude_scaling
from .scale_utils import find_alpha_value, apply_alpha_correction

# ── Import shared merge functions and helpers from unified core ──────────
from .triple_merge_core import (
    merge_linear,
    merge_cross,
    merge_ties_strict,
    merge_ties_gentle,
    merge_ties_contrast,
    merge_slerp,
    merge_feature_mix,
    merge_magnitude,
    merge_subtract,
    merge_dare_rescale,
    merge_dare_lite,
    merge_svd_preserve,
    merge_block_swap,
    merge_noise_aware,
    merge_gradient_alignment,
    ensure_energy_preservation_triple,
    MERGE_METHOD_REGISTRY as _CORE_MERGE_REGISTRY,
)

# ── Backward-compatible aliases (so internal references still work) ─────
# These allow `triple_merge_svd_preserve` to call `triple_merge_linear`,
# and `merge_triple_method` to use `triple_merge_linear` as fallback.
triple_merge_linear = merge_linear
triple_merge_cross = merge_cross
triple_merge_ties_strict = merge_ties_strict
triple_merge_ties_gentle = merge_ties_gentle
triple_merge_ties_contrast = merge_ties_contrast
triple_merge_slerp = merge_slerp
triple_merge_feature_mix = merge_feature_mix
triple_merge_magnitude = merge_magnitude
triple_merge_subtract = merge_subtract
triple_merge_dare_rescale = merge_dare_rescale
triple_merge_dare_lite = merge_dare_lite
triple_merge_svd_preserve = merge_svd_preserve
triple_merge_block_swap = merge_block_swap
triple_merge_noise_aware = merge_noise_aware
triple_merge_gradient_alignment = merge_gradient_alignment


# ==================== HELPER FUNCTIONS ====================


def adjust_ranks(tensors: List[torch.Tensor], key: str) -> List[torch.Tensor]:
    """
    Pad/truncate all tensors to the same rank (max rank among them).
    Returns adjusted tensors.
    """
    ranks = [safe_get_rank(t, key) for t in tensors]
    target_rank = max(ranks)

    adjusted_tensors = []
    for i, t in enumerate(tensors):
        if ranks[i] != target_rank:
            t = silent_pad_or_truncate(t, target_rank, f"{key}_{i}")
        adjusted_tensors.append(t)

    return adjusted_tensors


def apply_lora_scaling_triple(tensor: torch.Tensor, original_sd: Dict[str, torch.Tensor],
                              key: str, mapping: Optional[Dict[str, str]] = None,
                              is_converted: bool = False) -> torch.Tensor:
    """
    Apply alpha/rank scaling correction to LoRA tensors (up‑weight only).
    Delegates to :func:`scale_utils.find_alpha_value` and
    :func:`scale_utils.apply_alpha_correction`.

    Preserves exact behavior:
      - Down weights (lora_A / lora_down) are never scaled
      - Converted LoRAs skip scaling
      - No alpha found → no scaling (assume already scaled or alpha == rank)
    """
    # Down weights should not be scaled; scaling is applied only to up weights
    if any(suffix in key for suffix in ['.lora_A.weight', '.lora_down.weight']):
        return tensor

    # Determine rank
    rank = safe_get_rank(tensor, key)
    rank = max(1, rank)

    # Find alpha value (uses mapping internally for candidate generation)
    alpha_value = find_alpha_value(original_sd, key, mapping=mapping, rank=rank)

    # Skip if converted or no alpha found
    if is_converted or alpha_value is None:
        return tensor

    # Apply linear (alpha / rank) scaling to up-weight
    return apply_alpha_correction(tensor, alpha_value, rank, mode="linear")


# ==================== REGISTRY ====================

# Re-export the core registry unchanged — all 15 method names map to the same
# unified merge functions that both wrappers now share.
TRIPLE_METHOD_REGISTRY = dict(_CORE_MERGE_REGISTRY)


# ── Triple-capability warnings ────────────────────────────────────────


METHOD_3WAY_WARNINGS: Dict[str, str] = {
    "slerp": (
        "slerp only works with exactly 2 active LoRAs. "
        "When merging 3+ LoRAs, only the first 2 are used."
    ),
}


def _warn_triple_capability(method: str, num_tensors: int) -> None:
    """Emit warnings for methods with limited 3-way support."""
    if num_tensors > 2 and method in METHOD_3WAY_WARNINGS:
        print(f"⚠️ WARNING [{method}]: {METHOD_3WAY_WARNINGS[method]}")


# ==================== MAIN TRIPLE MERGE FUNCTION ====================


def merge_triple_method(sds: List[Dict[str, torch.Tensor]],
                        weights: List[float],
                        method: str = "linear",
                        density: float = 1.0,
                        uniqueness: float = 0.7,
                        threshold: float = 0.0,
                        blend: float = 0.5,
                        blend_mode: str = "auto",
                        device: str = "auto",
                        magnitude_scaling: str = "none",
                        max_scaling_factor: float = 10.0,
                        batch_size: int = 32,
                        streaming: bool = True,
                        energy_preservation: bool = True,
                        balancing_mode: str = "disabled",
                        mappings: Optional[List[Dict[str, str]]] = None,
                        original_sds: Optional[List[Dict[str, torch.Tensor]]] = None,
                        metas: Optional[List[Dict[str, str]]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], Optional[Dict], Optional[List[float]]]:
    """
    Core triple merge logic with rank adjustment and magnitude scaling.

    Parameters:
        sds: List of state dictionaries (normalized).
        weights: List of weights for each LoRA.
        method: Merge method name.
        density: Density parameter (for DARE and sparsity).
        uniqueness: Uniqueness parameter for feature_mix.
        threshold: Threshold parameter for subtract.
        blend: Blend parameter for magnitude.
        blend_mode: Blend mode ("active" or "dense").
                     Note: "slerp" is now a first-class method in the method dropdown.
        magnitude_scaling: Magnitude scaling mode ("none", "rms", "top_5%", etc.).
        max_scaling_factor: Maximum scaling factor for magnitude equalization.
        batch_size: Number of keys to process per batch (streaming).
        streaming: Whether to process keys in batches to save VRAM.
        energy_preservation: Whether to apply energy preservation safety check.
        balancing_mode: Weight balancing mode ("disabled", "safe", "creative").
        mappings: Optional list of key mappings from normalized to original keys.
        original_sds: Optional list of original state dicts (for alpha scaling).
        metas: Optional list of metadata dicts.

    Returns:
        Tuple of (merged state dictionary, master_key_map, energy_by_component, adjusted_weights).
        Energy data is None if balancing is disabled or only 2 LoRAs.
    """
    # Forensic data collectors (populated inside the balancing block if active)
    _forensic_energy_data: Optional[Dict] = None
    _forensic_adjusted_weights: Optional[List[float]] = None

    # Emit triple-capability warning (e.g., slerp only works with 2 LoRAs)
    _warn_triple_capability(method, len(sds))

    all_keys = set()
    for sd in sds:
        all_keys.update(sd.keys())

    # Auto-weight balancing based on energy ratios
    if balancing_mode != "disabled" and len(sds) == 3:
        # Keys present in all three LoRAs
        common_keys = [key for key in all_keys if all(key in sd for sd in sds)]
        if common_keys:
            print(f"   ⚖️ Auto-weight balancing ({balancing_mode}) analyzing {len(common_keys)} common keys...")
            print(f"      Energy computed via per-element mean (rank-independent) on shared layers only")
            # Use shared utility for rank-independent mean energy computation
            energy_by_component = compute_component_energy_ratios(
                norm_sds=sds,
                common_keys=common_keys,
                original_sds=original_sds,
                mappings=mappings,
                converted_flags=[False, False, False],  # triple uses original SDs directly
                key_categorizer=categorize_key,
            )
            # Accumulate global energy across all components
            energy_per_lora = [0.0, 0.0, 0.0]
            for component, energies in energy_by_component.items():
                for i in range(3):
                    energy_per_lora[i] += energies[i]

            epsilon = 1e-12
            # Compute average energy
            avg_energy = sum(energy_per_lora) / 3.0
            # Compute scaling factors per LoRA based on balancing_mode
            factors = [1.0, 1.0, 1.0]
            if balancing_mode == "safe":
                # Safe mode: reduce weight of louder LoRAs to match average
                for i in range(3):
                    if energy_per_lora[i] > avg_energy and energy_per_lora[i] > epsilon:
                        ratio = math.sqrt(avg_energy / energy_per_lora[i])
                        factors[i] = max(0.1, min(10.0, ratio))
                        print(f"      LoRA {i} energy {energy_per_lora[i]:.2e} is {energy_per_lora[i]/avg_energy:.2f}x average, weight scaled by {factors[i]:.2f}")
            elif balancing_mode == "creative":
                # Creative mode: compromise scaling for all LoRAs
                for i in range(3):
                    if energy_per_lora[i] > epsilon:
                        ratio = math.sqrt(avg_energy / energy_per_lora[i])
                        creative_factor = (1.0 + ratio) / 2.0  # 50% compromise
                        factors[i] = max(0.1, min(10.0, creative_factor))
                        print(f"      LoRA {i} energy {energy_per_lora[i]:.2e} → compromise factor {factors[i]:.2f}")
            # Apply factors to weights
            adjusted_weights = [weights[i] * factors[i] for i in range(3)]
            print(f"   ⚖️ Original weights: {weights}")
            print(f"   ⚖️ Adjusted weights: {adjusted_weights}")
            if method == "slerp":
                print(f"   ℹ️ SLERP uses adjusted weight ratios for the interpolation factor t = |w| / Σ|w|")
            # Replace weights for the rest of the merge
            weights = adjusted_weights
            # Capture forensic data for upstream report
            _forensic_energy_data = energy_by_component
            _forensic_adjusted_weights = adjusted_weights

    # Use DeviceManager instead of hardcoded inline resolution
    device = DeviceManager.get_device(device)
    merged = {}

    # Convert to list for deterministic ordering
    key_list = list(all_keys)
    total_keys = len(key_list)

    # Memory guard helper — checks VRAM pressure, empties cache if >80%
    def memory_guard():
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            ratio = allocated / total if total > 0 else 0
            if ratio > 0.8:
                print(f"⚠️ High GPU memory usage: {allocated / (1024**3):.2f} GB / {total / (1024**3):.2f} GB ({ratio:.1%}). Clearing cache.")
                torch.cuda.empty_cache()

    # Process keys in batches
    with ProgressTracker(total=total_keys, desc="Merging triple LoRA keys") as merge_progress:
        for batch_start in range(0, total_keys, batch_size):
            memory_guard()
            batch_keys = key_list[batch_start:batch_start + batch_size]
            for key in batch_keys:
                tensors = []
                valid_weights = []

                # Collect tensors that exist in each SD, along with original state dicts and mappings
                original_sd_list = []
                mapping_list = []
                for i, sd in enumerate(sds):
                    if key in sd:
                        # Async H2D transfer to overlap with computation
                        t = sd[key]
                        if t.device.type == 'cpu' and device.type == 'cuda':
                            t = t.to(device, non_blocking=True)
                        t = t.to(torch.bfloat16)
                        tensors.append(t)
                        valid_weights.append(weights[i])
                        # Store corresponding original_sd and mapping if available
                        if original_sds is not None and i < len(original_sds):
                            original_sd_list.append(original_sds[i])
                        else:
                            original_sd_list.append(None)
                        if mappings is not None and i < len(mappings):
                            mapping_list.append(mappings[i])
                        else:
                            mapping_list.append(None)

                # Apply alpha/rank scaling (up‑weight only) if original_sds provided
                if original_sds is not None:
                    for idx, (tensor, orig_sd, mapping) in enumerate(zip(tensors, original_sd_list, mapping_list)):
                        if orig_sd is not None:
                            tensors[idx] = apply_lora_scaling_triple(tensor, orig_sd, key, mapping=mapping, is_converted=False)

                if len(tensors) < 2:
                    if tensors:
                        # Unique key (present in only one LoRA) – still apply scaling already done
                        result = tensors[0] * valid_weights[0]
                        merged[key] = result
                    continue

                # Signal magnitude equalization (RMS/percentile scaling) if enabled
                if magnitude_scaling != "none" and len(tensors) >= 2:
                    # Determine reference tensor (LoRA A) - index 0 in original sds order
                    orig_indices = [i for i, sd in enumerate(sds) if key in sd]
                    if 0 in orig_indices:
                        ref_idx = orig_indices.index(0)
                        tensors = apply_magnitude_scaling(tensors, valid_weights,
                                                          magnitude_scaling, max_scaling_factor,
                                                          ref_idx=ref_idx)

                # Rank adjustment (pad/truncate to same rank)
                tensors = adjust_ranks(tensors, key)

                # Apply method with adjusted tensors
                merge_fn = TRIPLE_METHOD_REGISTRY.get(method)
                if merge_fn is None:
                    # Fallback to linear
                    merge_fn = triple_merge_linear
                # Prepare kwargs for the method
                METHOD_KWARGS = {
                    "feature_mix": {"uniqueness": uniqueness},
                    "magnitude": {"blend": blend},
                    "subtract": {"threshold": threshold},
                    "dare_rescale": {"density": density},
                    "dare_lite": {"density": density},
                }
                kwargs = METHOD_KWARGS.get(method, {})


                result = merge_fn(tensors, valid_weights, blend_mode=blend_mode, **kwargs)

                # Energy safety check
                if energy_preservation:
                    result = ensure_energy_preservation_triple(result, tensors, valid_weights, threshold=0.8, gain_min=1.0, target='avg')

                # Apply density if needed
                # Note: dare_rescale and dare_lite handle density internally
                if density < 1.0 and method not in ("dare_rescale", "dare_lite"):
                    flat = result.abs().flatten()
                    k = max(1, int(flat.numel() * density))
                    threshold_val = torch.topk(flat, k).values.min()
                    mask = result.abs() >= threshold_val
                    result = result * mask

                # Keep result on GPU within batch; batch transfer to CPU below
                merged[key] = result

                # Cleanup
                del tensors

                merge_progress += 1

        # Transfer entire batch from GPU → CPU in one shot, then clear cache
        if device.type == 'cuda':
            for key in batch_keys:
                if key in merged and merged[key].device.type == 'cuda':
                    merged[key] = merged[key].cpu()
            if streaming:
                torch.cuda.empty_cache()

    # Build master_map from input mappings (identity map for key restoration)
    master_map: Dict[str, str] = {}
    mapped_count = 0
    if mappings:
        for key in merged:
            found = False
            # Priority order: LoRA A > B > C
            for i in range(len(mappings)):
                if i < len(mappings) and mappings[i] is not None and key in mappings[i]:
                    master_map[key] = mappings[i][key]
                    found = True
                    break
            if not found:
                # Key not in any mapping (e.g., generated alpha key) — identity fallback
                master_map[key] = key
            else:
                mapped_count += 1
    else:
        # No mappings provided — identity fallback for all keys
        master_map = {key: key for key in merged}

    if mappings:
        print(f"   🗺️ Built master_map: {mapped_count}/{len(merged)} keys mapped, {len(merged) - mapped_count} identity fallback")

    return merged, master_map, _forensic_energy_data, _forensic_adjusted_weights
