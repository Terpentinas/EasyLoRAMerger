"""
Easy Component Merger — ComfyUI Node Definition

Merges 2-3 component state dicts (e.g., CLIP/text-encoder, UNet, or VAE data)
from the EasyComponentExtractor into a single merged checkpoint dict using
the full suite of merge methods.  Pure orchestration of existing utilities.

Design follows ``plans/working_principles.md``:
    P1 (Delete First)  — zero new infrastructure, pure orchestration of existing utils
    P2 (No Temp Files)  — no disk I/O; merge in RAM
    P3 (One Pattern)    — uses existing universal_merge_executor, method dispatch
    P4 (Resist Guards)  — clean linear flow: collect keys → merge → return
    P5 (Simple Dispatch) — 2 branches (≤2 inputs vs 3 inputs), each predictable
"""

import time
from typing import Dict, List, Optional, Callable

import torch

from ..utils import (
    cleanup_memory,
    NodeCache,
)
from ..config import PRECISION_EXTENDED, DEVICE_OPTIONS, DevicePrecisionConfig
from .fp8_quantizer import dequant_fp8_tensor
from .methods import (
    universal_merge_executor,
    merge_linear,
    merge_ties_strict,
    merge_ties_gentle,
    merge_dare_lite,
    merge_dare_rescale,
    merge_slerp,
    merge_subtract,
    merge_magnitude,
    merge_feature_mix,
    merge_svd_preserve,
    merge_noise_aware,
    merge_gradient_alignment,
    merge_cross,
    merge_ties_contrast,
    merge_block_swap,
)


# ── Precision conversion (same pattern as component_extractor.py/combiner.py) ──
def _convert_precision(
    state_dict: Dict[str, torch.Tensor],
    target_dtype: torch.dtype,
    *,
    component_label: str = "",
) -> Dict[str, torch.Tensor]:
    """Convert all tensors in *state_dict* to *target_dtype*.

    Skips conversion if *target_dtype* is ``None`` (keep native).
    """
    if target_dtype is None:
        return state_dict

    converted = {}
    for key, tensor in state_dict.items():
        if tensor.dtype != target_dtype:
            converted[key] = tensor.to(target_dtype)
        else:
            converted[key] = tensor

    if component_label:
        print(f"   🔧 {component_label}: converted {len(converted)} tensors → {target_dtype}")
    return converted


# ── Method dispatch table ──────────────────────────────────────────
# Maps method name → binary merge function for per-key merging.
_BINARY_METHODS: Dict[str, Callable] = {
    "linear": merge_linear,
    "ties_strict": merge_ties_strict,
    "ties_gentle": merge_ties_gentle,
    "dare_lite": merge_dare_lite,
    "dare_rescale": merge_dare_rescale,
    "slerp": merge_slerp,
    "subtract": merge_subtract,
    "magnitude": merge_magnitude,
    "feature_mix": merge_feature_mix,
    "svd_preserve": merge_svd_preserve,
    "noise_aware": merge_noise_aware,
    "gradient_alignment": merge_gradient_alignment,
    "cross": merge_cross,
    "ties_contrast": merge_ties_contrast,
    "block_swap": merge_block_swap,
}


def _resolve_single_triple(
    tensors: List[torch.Tensor],
    weights: List[float],
    method_fn: Callable,
    density: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Merge 3 tensors into 1 using pair-wise reduction.

    For linear-method 3-way merges this produces an exact weighted sum.
    For other methods it merges (a,b) first, then merges result with c.

    Args:
        tensors: [a, b, c] — 3 tensors to merge.
        weights: [wa, wb, wc] — corresponding weights.
        method_fn: The binary merge function from _BINARY_METHODS.
        density: Density parameter (for DARE methods).
        **kwargs: Additional args passed to method_fn (blend_mode, device, etc.).

    Returns:
        Merged tensor.
    """
    # Merge a+b → ab
    merged_ab = universal_merge_executor(
        method_fn, tensors[0], tensors[1],
        weights[0], weights[1],
        pbar=None, density=density, **kwargs,
    )
    # Merge ab+c → abc
    merged = universal_merge_executor(
        method_fn, merged_ab, tensors[2],
        1.0, weights[2],  # First weight = 1.0 (already blended into ab)
        pbar=None, density=density, **kwargs,
    )
    return merged


def merge_component_dicts(
    sds: List[Dict[str, torch.Tensor]],
    weights: List[float],
    method: str = "linear",
    density: float = 1.0,
    blend_mode: str = "auto",
    device: str = "auto",
    uniqueness: float = 0.7,
    threshold: float = 0.2,
    blend: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Merge 2-3 component state dicts (e.g. CLIP/TE data) into one.

    Iterates over all unique keys across all inputs.  For each key:
    - Present in 1 source → pass through (weighted)
    - Present in 2 sources → binary merge via ``universal_merge_executor()``
    - Present in 3 sources → triple merge via ``_resolve_single_triple()``

    Args:
        sds: List of 2-3 state dicts containing the same component keys.
        weights: Corresponding weights for each input.
        method: Merge method name (see ``_BINARY_METHODS`` keys).
        density: Density parameter for DARE methods.
        blend_mode: ``"auto"``, ``"active"``, or ``"dense"``.
        device: Target device string (``"auto"``, ``"cpu"``, ``"cuda"``).
        uniqueness: For ``feature_mix`` — higher = preserve more unique features.
        threshold: For ``subtract`` — minimum magnitude to subtract.
        blend: For ``magnitude`` — 0=strict max, 1=blended.

    Returns:
        A single merged state dict.
    """
    if not sds:
        return {}

    if len(sds) == 1:
        # Single source: pass through (weighted if needed)
        if abs(weights[0] - 1.0) > 1e-6:
            return {k: v * weights[0] for k, v in sds[0].items()}
        return dict(sds[0])

    method_fn = _BINARY_METHODS.get(method)
    if method_fn is None:
        print(f"   ⚠️ Unknown method '{method}' — falling back to linear")
        method_fn = merge_linear

    # Collect all unique keys across all inputs
    all_keys: set = set()
    for sd in sds:
        all_keys.update(sd.keys())

    merged: Dict[str, torch.Tensor] = {}
    skipped_count = 0
    key_count = len(all_keys)

    for idx, key in enumerate(sorted(all_keys)):  # deterministic order
        # Which inputs have this key?
        present_indices = [i for i, sd in enumerate(sds) if key in sd]
        present_tensors = [sds[i][key] for i in present_indices]
        present_weights = [weights[i] for i in present_indices]

        if len(present_tensors) == 0:
            continue

        elif len(present_tensors) == 1:
            # Key only in one source — pass through (apply weight)
            w = present_weights[0]
            if abs(w - 1.0) > 1e-6:
                merged[key] = present_tensors[0] * w
            else:
                merged[key] = present_tensors[0]

        elif len(present_tensors) == 2:
            # Binary merge via universal_merge_executor
            try:
                merged[key] = universal_merge_executor(
                    method_fn,
                    present_tensors[0],
                    present_tensors[1],
                    present_weights[0],
                    present_weights[1],
                    pbar=None,
                    density=density,
                    blend_mode=blend_mode,
                    uniqueness=uniqueness,
                    threshold=threshold,
                    blend=blend,
                )
            except Exception as e:
                print(f"   ⚠️ Failed to merge key '{key}': {e} — pass-through from first source")
                w = present_weights[0]
                merged[key] = present_tensors[0] * w if abs(w - 1.0) > 1e-6 else present_tensors[0]
                skipped_count += 1

        elif len(present_tensors) == 3:
            # Triple merge via pair-wise reduction
            try:
                merged[key] = _resolve_single_triple(
                    present_tensors,
                    present_weights,
                    method_fn,
                    density=density,
                    blend_mode=blend_mode,
                    uniqueness=uniqueness,
                    threshold=threshold,
                    blend=blend,
                )
            except Exception as e:
                print(f"   ⚠️ Failed to triple-merge key '{key}': {e} — fallback to binary")
                try:
                    # Fallback: merge first two only
                    merged[key] = universal_merge_executor(
                        method_fn,
                        present_tensors[0],
                        present_tensors[1],
                        present_weights[0],
                        present_weights[1],
                        pbar=None,
                        density=density,
                        blend_mode=blend_mode,
                    )
                except Exception:
                    w = present_weights[0]
                    merged[key] = present_tensors[0] * w if abs(w - 1.0) > 1e-6 else present_tensors[0]
                    skipped_count += 1

        # Log progress every 200 keys
        if (idx + 1) % 200 == 0:
            print(f"   ⏳ Merged {idx + 1}/{key_count} keys...")

    if skipped_count:
        print(f"   ⚠️ {skipped_count} keys had merge errors — passed through from primary source")

    print(f"   📊 Merged: {len(merged)}/{key_count} keys | "
          f"Method: {method} | "
          f"Sources: {len(sds)} | "
          f"Density: {density}")

    return merged


# ===================================================================
# Node class
# ===================================================================
class EasyComponentMerger:
    """
    Merge 2-3 component state dicts (CLIP/TE, UNet, or VAE data) from
    :class:`EasyComponentExtractor` into a single merged ``CHECKPOINT``
    dict.

    Chains directly into :class:`EasyComponentCombiner` for reassembly
    with other components.

    Use cases:
      - Merge CLIP/text-encoders from two fine-tunes → combine with
        a different UNet (e.g., anime prompt understanding + realistic
        rendering).
      - Blend UNet weights while preserving CLIP+VAE from a single
        source.
      - Three-way CLIP merge with TIES or DARE for novel prompt
        interpretation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── SECTION 1: SOURCES ───────────────────────────────────
                "component_data_a": ("CHECKPOINT", {
                    "tooltip": "Primary component state dict (chained from Component Extractor)",
                }),
                "component_data_b": ("CHECKPOINT", {
                    "tooltip": "Secondary component state dict to merge with primary",
                }),

                # ── SECTION 2: METHOD ────────────────────────────────────
                "method": (["linear", "ties_strict", "ties_gentle", "dare_lite",
                           "dare_rescale", "slerp", "subtract", "magnitude",
                           "feature_mix", "svd_preserve", "noise_aware",
                           "gradient_alignment", "cross", "ties_contrast", "block_swap"], {
                    "default": "linear",
                    "tooltip": "How tensors are combined per key",
                }),
                "density": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01,
                    "tooltip": "Density for DARE methods (keep top % of weights). "
                               "WARNING: Values < 1.0 sparsify weights, may degrade quality.",
                }),
            },
            "optional": {
                # ── SECTION 1 (continued) ───────────────────────────────
                "component_data_c": ("CHECKPOINT", {
                    "tooltip": "Tertiary component state dict (optional, for 3-way merge)",
                }),

                # ── SECTION 3: WEIGHTS ───────────────────────────────────
                "weight_a": ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Strength of first component",
                }),
                "weight_b": ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Strength of second component",
                }),
                "weight_c": ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Strength of third component (only used if component_data_c provided)",
                }),

                # ── SECTION 4: METHOD-SPECIFIC PARAMS ────────────────────
                "uniqueness": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01,
                    "tooltip": "For feature_mix: higher = preserve more unique features from each source",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "For subtract: minimum magnitude threshold for subtraction",
                }),
                "blend": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "For magnitude: 0=strict max-magnitude selector, 1=fully blended",
                }),

                # ── SECTION 5: HARDWARE ─────────────────────────────────
                "blend_mode": (["auto", "active", "dense"], {
                    "default": "auto",
                    "tooltip": "auto: Smart choice | active: Only merge non-zero regions | dense: Traditional weighted sum",
                }),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "precision": (PRECISION_EXTENDED, {"default": "auto"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache implementation for ComfyUI change detection."""
        # If any data input is connected, always re-execute
        if (kwargs.get("component_data_a") is not None
                or kwargs.get("component_data_b") is not None
                or kwargs.get("component_data_c") is not None):
            return float("nan")

        return NodeCache.is_changed(
            cls.__name__,
            method=kwargs.get("method", "linear"),
            density=kwargs.get("density", 1.0),
            weight_a=kwargs.get("weight_a", 1.0),
            weight_b=kwargs.get("weight_b", 1.0),
            weight_c=kwargs.get("weight_c", 1.0),
            uniqueness=kwargs.get("uniqueness", 0.7),
            threshold=kwargs.get("threshold", 0.2),
            blend=kwargs.get("blend", 0.5),
            blend_mode=kwargs.get("blend_mode", "auto"),
            device=kwargs.get("device", "auto"),
            precision=kwargs.get("precision", "auto"),
        )

    RETURN_TYPES = ("CHECKPOINT",)
    RETURN_NAMES = ("merged_component_data",)
    FUNCTION = "merge"
    CATEGORY = "Checkpoint/Utils"
    OUTPUT_NODE = True

    def merge(
        self,
        component_data_a=None,
        component_data_b=None,
        method: str = "linear",
        density: float = 1.0,
        component_data_c=None,
        weight_a: float = 1.0,
        weight_b: float = 1.0,
        weight_c: float = 1.0,
        uniqueness: float = 0.7,
        threshold: float = 0.2,
        blend: float = 0.5,
        blend_mode: str = "auto",
        device: str = "auto",
        precision: str = "auto",
        **kwargs,
    ):
        """
        Merge 2-3 component state dicts using the specified method.

        Returns ``(merged_component_data,)`` — a single ``CHECKPOINT``
        dict ready to chain into :class:`EasyComponentCombiner` or
        another merge.
        """
        # ══ Runtime precision guard — protect against stale workflow JSONs ══
        if precision not in PRECISION_EXTENDED:
            print(f"   ⚠️ Precision '{precision}' is no longer available, falling back to 'auto'")
            precision = "auto"

        print("\n" + "=" * 60)
        print("🧩 Easy Component Merger")
        print("=" * 60)
        start_time = time.time()

        # ── Step 1: Collect inputs ──────────────────────────────────────
        # Unwrap tuple-wrapped state dicts (ComfyUI CHECKPOINT type convention)
        def _unwrap(data):
            if isinstance(data, tuple) and len(data) >= 1:
                return data[0]
            return data

        raw_sds: List[Dict] = []
        raw_weights: List[float] = []
        labels: List[str] = []

        for label, data, weight in [
            ("A", component_data_a, weight_a),
            ("B", component_data_b, weight_b),
            ("C", component_data_c, weight_c),
        ]:
            sd = _unwrap(data)
            if sd is not None and isinstance(sd, dict) and len(sd) > 0:
                raw_sds.append(sd)
                raw_weights.append(weight)
                labels.append(label)
                print(f"   📦 Source {label}: {len(sd)} tensors, weight={weight:+g}")

        if len(raw_sds) < 2:
            print("   ⚠️ Need at least 2 component state dicts to merge")
            if raw_sds:
                print("   ➡️ Pass-through: returning single source unchanged")
                result_sd = raw_sds[0]
            else:
                print("   ➡️ No valid inputs — returning None")
                result_sd = None
            elapsed = time.time() - start_time
            print(f"\n⏱️  Merge complete: {elapsed:.2f}s (no-op)")
            print("=" * 60)
            cleanup_memory(skip_gc=True)
            return (result_sd,)

        print(f"   ⚖️ Weights: {dict(zip(labels, raw_weights))}")
        print(f"   🔧 Method: {method}, Density: {density}, Blend mode: {blend_mode}")
        if method == "feature_mix":
            print(f"   🎯 uniqueness={uniqueness}")
        elif method == "subtract":
            print(f"   🎯 threshold={threshold}")
        elif method == "magnitude":
            print(f"   🎯 blend={blend}")

        # ── Step 2: Resolve hardware ────────────────────────────────────
        dpc = DevicePrecisionConfig(device_type=device, precision=precision)
        print(f"   💻 Device: {dpc.device} | 🎯 Precision: {dpc.dtype}")

        # ── Step 2a: Dequantize any FP8 input tensors ───────────────────
        # _unify_dtype_and_device (called by universal_merge_executor via
        # merge_component_dicts) runs torch.promote_types() which crashes on
        # fp8.  Dequant to bf16 upfront to prevent the crash.
        for i, sd in enumerate(raw_sds):
            needs_dequant = any(
                t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                for t in sd.values()
            )
            if needs_dequant:
                for key in list(sd.keys()):
                    t = sd[key]
                    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        sd[key] = dequant_fp8_tensor(t, key, torch.bfloat16, sd)
                print(f"   🚀 Dequantized FP8 tensors in source {labels[i]}")

        # ── Step 3: Merge ───────────────────────────────────────────────
        merged_sd = merge_component_dicts(
            sds=raw_sds,
            weights=raw_weights,
            method=method,
            density=density,
            blend_mode=blend_mode,
            device=device,
            uniqueness=uniqueness,
            threshold=threshold,
            blend=blend,
        )

        # ── Step 4: Optional precision conversion ───────────────────────
        if merged_sd and dpc.dtype is not None:
            merged_sd = _convert_precision(
                merged_sd, dpc.dtype, component_label="Merged",
            )

        elapsed = time.time() - start_time
        print(f"\n⏱️  Merge complete: {elapsed:.2f}s")
        if merged_sd:
            print(f"   📦 Output: {len(merged_sd)} tensors")
        else:
            print("   📦 Output: None")
        print("=" * 60)

        # Cleanup
        cleanup_memory(skip_gc=True)

        return (merged_sd,)
