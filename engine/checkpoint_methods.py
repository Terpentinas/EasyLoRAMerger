"""
Checkpoint Triple Merging Methods for Easy Checkpoint Merger
Adapted from triple_methods.py for full model checkpoints (no LoRA-specific scaling).

This module is now a thin wrapper around engine.triple_merge_core,
preserving the same public API (merge_triple_method returning 2 values)
while eliminating ~700 lines of duplicated merge function bodies.
"""

import gc
import math
import time
from typing import List, Dict, Optional, Tuple, Callable

import torch

# Import cleanup_memory for GPU memory management before returning to weaver
try:
    from ..utils import cleanup_memory
except ImportError:
    from utils import cleanup_memory

# Maximum reasonable single tensor size: 4 GB
# Tensors exceeding this are likely corrupted (absurd shape metadata).
# Prevents CUDA OOM from corrupt safetensors headers.
MAX_TENSOR_BYTES = 4 * 1024**3

# ── Phase 3: Density sparsification guard ──────────────────────────
# Maximum tensor size (bytes) for density sparsification via torch.topk.
# Beyond this, topk on full-size checkpoint weights becomes prohibitively
# expensive (O(n log k) on tensors with millions of elements).
MAX_SPARSIFY_BYTES = 50 * 1024 * 1024  # 50 MB

# ── Phase 3: Shape-based batching ──────────────────────────────────
# Minimum number of same-shaped keys required to attempt batched merge
# (torch.stack → single merge call → unstack).
MIN_BATCH_SHAPE_GROUP_SIZE = 2

# ── Phase 3: Methods incompatible with shape batching ──────────────
# These methods have dimension-specific logic (SVD decomposition, block
# dimensions) that produces incorrect results when stacking produces 3D
# [batch, rows, cols] tensors.  They fall back to the safe sequential path.
#   - svd_preserve: batched SVD misinterprets batch dim (Bug 1)
#   - block_swap:  dimension-specific logic assumes 2D/4D only (Bug 2)
BATCH_INCOMPATIBLE_METHODS = {"svd_preserve", "block_swap"}

from ..config import MergeMethodRegistry
from ..utils import (
    categorize_checkpoint_key as categorize_key,
    apply_component_scaling,
    compute_component_energy_ratios,
    DeviceManager,
    comfyui_yield,
)
from .methods import (
    apply_magnitude_scaling,
    ensure_shape_match,
)

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
    For checkpoints, rank adjustment is not needed (no LoRA rank).
    Return tensors unchanged.
    """
    return tensors


def _check_shape_consistency(
    tensors: List[torch.Tensor], key: str
) -> List[torch.Tensor]:
    """
    Ensure all tensors for a key have matching shapes, padding if needed.

    When shapes differ across checkpoints (e.g. different attention head
    dimensions), pads the smaller tensor with zeros to match the larger.
    Returns a new list of tensors all sharing the same shape.
    """
    if len(tensors) < 2:
        return tensors
    result = [tensors[0]]
    for t in tensors[1:]:
        if t.shape != result[0].shape:
            print(f"   ⚠️ Shape mismatch for '{key}': {result[0].shape} vs {t.shape} — padding")
            _, padded = ensure_shape_match(result[0].clone(), t.clone())
            result.append(padded)
        else:
            result.append(t)
    return result


def _ensure_compute_dtype(tensor: torch.Tensor,
                          resolved_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert FP8 tensors to a compute-capable dtype before merge operations.

    CUDA does not implement mul/add kernels for Float8_e4m3fn or Float8_e5m2.
    If the tensor is FP8, convert to bfloat16 (or float16 fallback) regardless
    of resolved_dtype. Non-FP8 tensors are left unchanged if already in a
    compute-capable dtype — resolved_dtype is NEVER used as a target when it
    is an FP8 dtype, to avoid converting bf16/float32 tensors back to FP8.

    Args:
        tensor: Input tensor, possibly FP8.
        resolved_dtype: Target dtype from checkpoint_weaver (may also be FP8).

    Returns:
        Tensor in a compute-capable dtype.
    """
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 is not computable on CUDA — always convert to bfloat16/float16
        fallback = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   🔄 FP8 tensor converted to {fallback} for merge computation")
        return tensor.to(dtype=fallback)

    # Non-FP8: determine safe compute dtype
    # Never convert TO an FP8 dtype — resolved_dtype may be float8_e4m3fn
    # when the user selected fp8 mode.  In that case the weaver has already
    # pre-converted tensors to bf16, so we must NOT re-quantize them.
    safe_dtype = resolved_dtype
    if safe_dtype is None or safe_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        safe_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if tensor.dtype != safe_dtype:
        return tensor.to(dtype=safe_dtype)
    return tensor


# ==================== REGISTRY ====================

# Re-export the core registry unchanged.
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
                        precision: str = "auto",
                        magnitude_scaling: str = "none",
                        max_scaling_factor: float = 10.0,
                        batch_size: int = 64,
                        streaming: bool = True,
                        energy_preservation: bool = True,
                        balancing_mode: str = "disabled",
                        weight_unet: float = 1.0,
                        weight_clip: float = 1.0,
                        weight_vae: float = 1.0,
                        weight_te: float = 1.0,
                        mappings: Optional[List[Dict[str, str]]] = None,
                        original_sds: Optional[List[Dict[str, torch.Tensor]]] = None,
                        metas: Optional[List[Dict[str, str]]] = None,
                        on_substep: Optional[Callable[[int], None]] = None,
                        resolved_device: Optional[torch.device] = None,
                        resolved_dtype: Optional[torch.dtype] = None,
                        sequential_only: bool = False) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Core triple merge logic for checkpoints with component scaling (Weight Block Map).

    Parameters:
        sds: List of state dictionaries (normalized).
        weights: List of weights for each checkpoint.
        method: Merge method name.
        density: Density parameter (for DARE and sparsity).
        uniqueness: Uniqueness parameter for feature_mix.
        threshold: Threshold parameter for subtract.
        blend: Blend parameter for magnitude.
        blend_mode: Blend mode ("active" or "dense").
        magnitude_scaling: Magnitude scaling mode ("none", "rms", "top_5%", etc.).
        max_scaling_factor: Maximum scaling factor for magnitude equalization.
        batch_size: Number of keys to process per batch (streaming).
        streaming: Whether to process keys in batches to save VRAM.
        energy_preservation: Whether to apply energy preservation safety check.
        balancing_mode: Weight balancing mode ("disabled", "safe", "creative").
        weight_unet: Global scaling factor for UNET components.
        weight_clip: Global scaling factor for CLIP visual components.
        weight_vae: Global scaling factor for VAE components.
        weight_te: Global scaling factor for Text Encoder components.
        mappings: Optional list of key mappings per source.
        original_sds: Optional list of original state dicts per source.
        metas: Optional list of metadata dicts.
        on_substep: Optional callback invoked as on_substep(n) where n is the
                    number of keys processed. Enables fine-grained progress
                    reporting without coupling to ProgressTracker.
        resolved_device: Pre-resolved target device (bypasses DeviceManager.get_device).
                         Provided by checkpoint_weaver for device consistency.
        resolved_dtype: Pre-resolved target dtype (bypasses hardcoded torch.bfloat16).
                        Provided by checkpoint_weaver for dtype consistency.

    Returns:
        Merged state dictionary.
    """
    # Emit triple-capability warning (e.g., slerp only works with 2 LoRAs)
    _warn_triple_capability(method, len(sds))

    all_keys = set()
    for sd in sds:
        all_keys.update(sd.keys())

    # Auto-weight balancing based on energy ratios
    if balancing_mode != "disabled" and len(sds) == 3:
        # Keys present in all three checkpoints
        common_keys = [key for key in all_keys if all(key in sd for sd in sds)]
        if common_keys:
            print(f"   ⚖️ Auto-weight balancing ({balancing_mode}) analyzing {len(common_keys)} common keys...")
            print(f"      Energy computed via per-element mean (rank-independent) on shared layers only")
            # Use shared utility for rank-independent mean energy computation
            # Checkpoints have no LoRA alpha/rank scaling, so original_sds/mappings are omitted
            energy_by_component = compute_component_energy_ratios(
                norm_sds=sds,
                common_keys=common_keys,
                original_sds=None,
                mappings=None,
                converted_flags=None,
                key_categorizer=categorize_key,
            )
            # Accumulate global energy across all components
            energy_per_checkpoint = [0.0, 0.0, 0.0]
            for component, energies in energy_by_component.items():
                for i in range(3):
                    energy_per_checkpoint[i] += energies[i]

            epsilon = 1e-12
            # Compute average energy
            avg_energy = sum(energy_per_checkpoint) / 3.0
            # Compute scaling factors per checkpoint based on balancing_mode
            factors = [1.0, 1.0, 1.0]
            if balancing_mode == "safe":
                # Safe mode: reduce weight of louder checkpoints to match average
                for i in range(3):
                    if energy_per_checkpoint[i] > avg_energy and energy_per_checkpoint[i] > epsilon:
                        ratio = math.sqrt(avg_energy / energy_per_checkpoint[i])
                        factors[i] = max(0.1, min(10.0, ratio))
                        print(f"      Checkpoint {i} energy {energy_per_checkpoint[i]:.2e} is {energy_per_checkpoint[i]/avg_energy:.2f}x average, weight scaled by {factors[i]:.2f}")
            elif balancing_mode == "creative":
                # Creative mode: compromise scaling for all checkpoints
                for i in range(3):
                    if energy_per_checkpoint[i] > epsilon:
                        ratio = math.sqrt(avg_energy / energy_per_checkpoint[i])
                        creative_factor = (1.0 + ratio) / 2.0  # 50% compromise
                        factors[i] = max(0.1, min(10.0, creative_factor))
                        print(f"      Checkpoint {i} energy {energy_per_checkpoint[i]:.2e} → compromise factor {factors[i]:.2f}")
            # Apply factors to weights
            adjusted_weights = [weights[i] * factors[i] for i in range(3)]
            print(f"   ⚖️ Original weights: {weights}")
            print(f"   ⚖️ Adjusted weights: {adjusted_weights}")
            # Replace weights for the rest of the merge
            weights = adjusted_weights

    # Use DeviceManager instead of hardcoded inline resolution.
    # If resolved_device is provided (from checkpoint_weaver), use it directly
    # to ensure device consistency across the merge pipeline (L2).
    if resolved_device is not None:
        device = resolved_device
    else:
        device = DeviceManager.get_device(device)
    merged = {}
    _mt_t0 = time.time()
    print(f"   🕐 [t={0.0:.1f}s] merge_triple_method: starting with {len(sds)} sources, method={method}")

    # Convert to list for deterministic ordering
    key_list = list(all_keys)
    total_keys = len(key_list)

    # Memory guard helper — checks VRAM pressure, forces GC + empty_cache if >75%
    def memory_guard():
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            ratio = allocated / total if total > 0 else 0
            if ratio > 0.75:  # Lowered from 0.8 for earlier intervention
                print(f"⚠️ High GPU memory: {allocated/1e9:.2f} GB / {total/1e9:.2f} GB ({ratio:.1%}). Forcing cleanup.")
                gc.collect()
                torch.cuda.empty_cache()
                # Re-check after cleanup
                allocated = torch.cuda.memory_allocated()
                ratio = allocated / total if total > 0 else 0
                if ratio > 0.85:
                    print(f"⚠️ GPU memory still high ({ratio:.1%}) after cleanup — risk of OOM")

    # ── Phase 3: Shape-based batching pre-scan ──────────────────────────
    # Group keys by tensor shape for potential batched merge.
    # Same-shaped keys from the same architecture (e.g., all attention
    # projection weights of the same layer) can be stacked into a 3D
    # tensor and processed with a single merge call, reducing per-key
    # Python overhead and improving GPU utilization.
    shape_to_keys: Dict[Tuple[int, ...], List[str]] = {}
    for key in key_list:
        for sd in sds:
            if key in sd:
                shape = sd[key].shape
                if shape not in shape_to_keys:
                    shape_to_keys[shape] = []
                shape_to_keys[shape].append(key)
                break

    # Separate batchable groups (size ≥ MIN_BATCH_SHAPE_GROUP_SIZE)
    # from sequential keys (small groups + singletons).
    batchable_shapes: Dict[Tuple[int, ...], List[str]] = {}
    sequential_key_list: List[str] = []
    for shape, keys in shape_to_keys.items():
        if len(keys) >= MIN_BATCH_SHAPE_GROUP_SIZE:
            batchable_shapes[shape] = keys
        else:
            sequential_key_list.extend(keys)
    # Free intermediate structure — no longer needed after batching setup
    del shape_to_keys
    print(f"   🕐 [t={time.time()-_mt_t0:.1f}s] merge_triple_method: shape-batch distribution complete ({len(batchable_shapes)} groups, {len(sequential_key_list)} sequential)")

    # ── VRAM check: only batch if the largest group fits ───────────────
    batching_enabled = False
    if not sequential_only and batchable_shapes and device.type == 'cuda' and method not in BATCH_INCOMPATIBLE_METHODS:
        largest_shape, largest_keys = max(
            batchable_shapes.items(),
            key=lambda x: x[0].numel() * len(x[1])
        )
        # Method-specific safety factors for stacked tensor intermediates.
        # Different merge methods create vastly different intermediate tensor sizes:
        # - default 2: element-wise ops (linear, subtract, dare_*, ties_*) create
        #   only ~1 temporary copy of input — peak ~2× stacked input
        # - feature_mix 6: creates [3, batch, ...] intermediates (magnitudes,
        #   shares) plus int64 dominant_idx (4× per element) — peak ~16× stacked
        # - svd_preserve 6: float32 copies (4× bf16) + batched SVD U/S/Vh
        #   outputs — peak similarly large
        # - magnitude 4: [3, batch, ...] stacked + int64 argmax, no shares copy
        # - gradient_alignment 4: 3× float32 copies for per-channel alignment
        _METHOD_SAFETY_FACTORS = {
            "feature_mix": 6,
            "svd_preserve": 6,
            "magnitude": 4,
            "gradient_alignment": 4,
        }
        base_factor = _METHOD_SAFETY_FACTORS.get(method, 2)
        # Active blend mode creates up to 10 boolean masks via _compute_active_masks()
        # (same shape as stacked input) plus _weight_tensors() creates 3 weighted copies.
        # These are NOT accounted for by the base safety factor (which only covers
        # merge-method-specific intermediates), so we add a flat +2 penalty so the VRAM
        # estimate correctly reflects the actual peak memory usage of active-blend merges.
        blend_penalty = 2 if blend_mode == 'active' else 0
        safety_factor = base_factor + blend_penalty

        # Stacked tensors + method-specific intermediates estimate.
        # Source tensors are on CPU (loaded via safe_open), not GPU prior to this.
        # The stacked source tensors ARE the GPU footprint — they replace per-key
        # individual tensors that would otherwise be loaded one-at-a-time in
        # sequential mode.  Merge intermediates add safety_factor × stacked size.
        # Use the actual number of active sources (not hardcoded 3) so two-way
        # merges get a realistic estimate.
        num_sources = len(sds)
        stacked_bytes = num_sources * len(largest_keys) * largest_shape.numel() * 2 * safety_factor
        est_bytes = stacked_bytes

        # Compute actual available VRAM as total - current_allocated.
        # get_free_vram returns total - reserved which overestimates free
        # memory after empty_cache (cached pool != actually allocatable space).
        total_memory = torch.cuda.get_device_properties(device).total_memory
        current_allocated = torch.cuda.memory_allocated(device)
        free_vram = total_memory - current_allocated
        # Safety margin: only batch if stacked+intermediates fit within free VRAM.
        # We compare est_bytes directly against free_vram (NOT current_allocated +
        # est_bytes) because the stacked source tensors REPLACE per-key GPU tensors
        # that sequential mode would load anyway — there's no additive cost for the
        # source data itself; only the merge intermediates are extra.
        # Using free_vram × 50% gives headroom for merge intermediates that
        # briefly spike during the merge_fn call (temp1 + temp2 + result can
        # temporarily reach 3-4× the stacked source size before Python's
        # reference counting deallocates the temporaries).
        # Previously 80% — caused CUDA OOM on large shape groups because
        # cp_buffers (now freed) and merge intermediate spikes were unaccounted.
        if free_vram > 0 and est_bytes < free_vram * 0.5:
            batching_enabled = True
            shape_str = '×'.join(str(d) for d in largest_shape)
            print(f"   🚀 Shape batching enabled: {len(batchable_shapes)} groups, "
                  f"largest = {shape_str} × {len(largest_keys)} keys "
                  f"(~{est_bytes / 1e9:.2f} GB est, "
                  f"{current_allocated / 1e9:.2f} GB allocated, "
                  f"{free_vram / 1e9:.2f} GB free, "
                  f"within 50% threshold)")
        else:
            shape_str = '×'.join(str(d) for d in largest_shape)
            pct = 100.0 * est_bytes / free_vram if free_vram > 0 else float('inf')
            print(f"   ℹ️ Shape batching skipped: "
                  f"~{est_bytes / 1e9:.2f} GB est would use {pct:.0f}% of "
                  f"{free_vram / 1e9:.2f} GB free VRAM (50% threshold) "
                  f"— falling back to sequential")

    # ── Shared merge helpers ───────────────────────────────────────────

    # Resolve merge function once (same for all keys)
    merge_fn = TRIPLE_METHOD_REGISTRY.get(method, triple_merge_linear)

    def _build_merge_kwargs(method_name: str) -> dict:
        """Build method-specific kwargs for the merge function."""
        kw = {}
        if method_name == "feature_mix":
            kw["uniqueness"] = uniqueness
        elif method_name == "magnitude":
            kw["blend"] = blend
        elif method_name == "subtract":
            kw["threshold"] = threshold
        elif method_name in ("dare_rescale", "dare_lite"):
            kw["density"] = density
        return kw

    merge_kwargs = _build_merge_kwargs(method)

    def _process_single_key(
        key: str,
        cp_tensors: List[torch.Tensor],
        cp_weights: List[float],
    ) -> torch.Tensor:
        """Process a single key: scaling → magnitude → merge → energy."""
        # Component scaling
        for idx in range(len(cp_tensors)):
            cp_tensors[idx] = apply_component_scaling(
                cp_tensors[idx], key,
                weight_unet, weight_clip, weight_vae, weight_te)

        # Magnitude equalization
        if magnitude_scaling != "none" and len(cp_tensors) >= 2:
            orig_indices = [i for i, sd in enumerate(sds) if key in sd]
            if 0 in orig_indices:
                ref_idx = orig_indices.index(0)
                cp_tensors = apply_magnitude_scaling(
                    cp_tensors, cp_weights,
                    magnitude_scaling, max_scaling_factor,
                    ref_idx=ref_idx)

        # Rank adjustment (no-op for checkpoints)
        cp_tensors = adjust_ranks(cp_tensors, key)

        # Shape consistency guard — pad mismatched tensors across checkpoints
        cp_tensors = _check_shape_consistency(cp_tensors, key)

        # Merge with CUDA OOM fallback (L1)
        try:
            result = merge_fn(cp_tensors, cp_weights,
                              blend_mode=blend_mode, **merge_kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "CUDA out of memory" in str(e):
                print(f"   ⚠️ CUDA OOM on key '{key}' — falling back to CPU")
                cp_tensors_cpu = [t.cpu() for t in cp_tensors]
                result = merge_fn(cp_tensors_cpu, cp_weights,
                                  blend_mode=blend_mode, **merge_kwargs)
                result = result.to(device)
            else:
                raise

        # Energy safety
        if energy_preservation:
            result = ensure_energy_preservation_triple(
                result, cp_tensors, cp_weights,
                threshold=0.8, gain_min=1.0, target='avg')
        return result

    def _apply_density(result: torch.Tensor, key: str) -> torch.Tensor:
        """Apply density sparsification with size guard (Phase 3.2)."""
        if density >= 1.0 or method in ("dare_rescale", "dare_lite"):
            return result

        # Phase 3.2: Skip sparsification for oversized tensors.
        # torch.topk on full-size checkpoint weights is O(n log k) with
        # millions of elements — prohibitively expensive.
        result_bytes = result.numel() * result.element_size()
        if result_bytes > MAX_SPARSIFY_BYTES:
            param_count = result.numel()
            print(f"   ⚠️ WARNING: density={density} sparsification skipped for "
                  f"'{key}' — tensor is {result_bytes / (1024**2):.1f} MB "
                  f"({param_count / 1e6:.1f}M params), exceeds "
                  f"{MAX_SPARSIFY_BYTES / (1024**2):.0f} MB limit")
            return result

        # Phase 3.2: Enhanced warning for very large tensors (>100M params)
        if result.numel() > 100_000_000:
            print(f"   ⚠️ WARNING: density={density} on "
                  f"{result.numel() / 1e6:.0f}M-param tensor '{key}'. "
                  f"topk on full-size checkpoint weights is computationally expensive.")

        flat = result.abs().flatten()
        k = max(1, int(flat.numel() * density))
        threshold_val = torch.topk(flat, k).values.min()
        mask = result.abs() >= threshold_val
        return result * mask

    # ── Merge loop ─────────────────────────────────────────────────────
    corrupted_keys = []

    # ═══════════════════════════════════════════════════════════════════
    # PHASE A: Batched processing (shape-group stacking)
    # ═══════════════════════════════════════════════════════════════════
    if batching_enabled:
        total_batched = sum(len(keys) for keys in batchable_shapes.values())
        total_sequential = len(sequential_key_list)
        # No internal tracker — on_substep callback handles progress reporting.

        # ── Process each batchable shape group ────────────────────
        for shape, keys in batchable_shapes.items():
            memory_guard()

            # Per-checkpoint collections for this shape group
            # Dynamically sized to match the actual number of sources (may be 2
            # or 3 for two-way / three-way merge).  Previously hardcoded to 3
            # slots, which was safe only because unused slots remained empty.
            num_sources = len(sds)
            cp_buffers: List[List[torch.Tensor]] = [[] for _ in range(num_sources)]
            cp_weight_vals: List[List[float]] = [[] for _ in range(num_sources)]
            batch_keys: List[str] = []

            # ── Load per-key tensors to GPU ───────────────────────
            # try/except catches CUDA OOM during .to(device) for large shape
            # groups (e.g. 24576×4096 × 4 keys in Flux).  Previous OOMs at this
            # stage propagated to the weave-level handler, forcing ALL subsequent
            # batches to CPU sequential.  Now we catch per-group and fall back to
            # per-key sequential on CPU for just this shape group.
            try:
                for key in keys:
                    collected_t: List[torch.Tensor] = []
                    collected_w: List[float] = []
                    skip_key = False
                    for i, sd in enumerate(sds):
                        if key in sd:
                            t = sd[key]
                            t_bytes = t.numel() * t.element_size()
                            if t_bytes > MAX_TENSOR_BYTES:
                                print(f"   ⚠️ Corrupt tensor '{key}' in source {i}: "
                                      f"{t_bytes / (1024**3):.2f} GB exceeds "
                                      f"{MAX_TENSOR_BYTES / (1024**3):.0f} GB limit – skipping")
                                if key not in corrupted_keys:
                                    corrupted_keys.append(key)
                                skip_key = True
                                break
                            # Guard redundant .to() calls — no-op on correct device/dtype
                            if t.device != device:
                                t = t.to(device)
                            t = _ensure_compute_dtype(t, resolved_dtype)
                            collected_t.append(t)
                            collected_w.append(weights[i])

                    if skip_key:
                        if on_substep:
                            on_substep(1)
                        continue

                    if len(collected_t) < 2:
                        # Unique key — process individually
                        if collected_t:
                            scaled = apply_component_scaling(
                                collected_t[0], key,
                                weight_unet, weight_clip, weight_vae, weight_te)
                            merged[key] = (scaled * collected_w[0]).cpu()
                        if on_substep:
                            on_substep(1)
                        continue

                    batch_keys.append(key)
                    for idx, (t, w) in enumerate(zip(collected_t, collected_w)):
                        cp_buffers[idx].append(t)
                        cp_weight_vals[idx].append(w)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(e):
                    print(f"   ⚠️ CUDA OOM loading keys for shape {shape} — per-key CPU fallback")
                    # Clean up partially-loaded GPU tensors in cp_buffers
                    del cp_buffers
                    del cp_weight_vals
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Process all keys in this shape group individually.
                    # Tensors in sds are still on CPU (the failed .to(device)
                    # left them there), so we pass CPU tensors to
                    # _process_single_key which handles GPU transfer per-key.
                    for key in keys:
                        t_list = []
                        w_list = []
                        corrupt = False
                        for i, sd in enumerate(sds):
                            if key in sd:
                                t = sd[key]
                                t_bytes = t.numel() * t.element_size()
                                if t_bytes > MAX_TENSOR_BYTES:
                                    if key not in corrupted_keys:
                                        corrupted_keys.append(key)
                                    corrupt = True
                                    break
                                t = _ensure_compute_dtype(t, resolved_dtype)
                                t_list.append(t)
                                w_list.append(weights[i])
                        if corrupt:
                            if on_substep:
                                on_substep(1)
                            continue
                        if len(t_list) >= 2:
                            result = _process_single_key(key, t_list, w_list)
                            result = _apply_density(result, key)
                            merged[key] = result.cpu()
                        elif len(t_list) == 1:
                            scaled = apply_component_scaling(
                                t_list[0], key,
                                weight_unet, weight_clip, weight_vae, weight_te)
                            merged[key] = (scaled * w_list[0]).cpu()
                        # Free source tensor references after processing
                        for sd in sds:
                            sd.pop(key, None)
                        if on_substep:
                            on_substep(1)
                    cleanup_memory()
                    continue
                else:
                    raise

            if len(batch_keys) < MIN_BATCH_SHAPE_GROUP_SIZE:
                # Not enough for efficient batching — process individually
                for ki, key in enumerate(batch_keys):
                    t_list = [cb[ki] for cb in cp_buffers if ki < len(cb)]
                    w_list = [cw[ki] for cw in cp_weight_vals if ki < len(cw)]
                    if len(t_list) >= 2:
                        result = _process_single_key(key, t_list, w_list)
                        result = _apply_density(result, key)
                        merged[key] = result.cpu()
                    if on_substep:
                        on_substep(1)
                continue

            # ── Shape consistency: verify all keys have matching shapes across ──
            # ── checkpoints. If any key has shape mismatches, fall back to     ──
            # ── sequential processing (each key goes through _process_single_key ──
            # ── which pads mismatched shapes via _check_shape_consistency).     ──
            shape_diverged = False
            for ki, key in enumerate(batch_keys):
                t_list = [cb[ki] for cb in cp_buffers if ki < len(cb)]
                if len(t_list) >= 2:
                    shapes = set(t.shape for t in t_list)
                    if len(shapes) > 1:
                        shape_diverged = True
                        break

            if shape_diverged:
                for ki, key in enumerate(batch_keys):
                    t_list = [cb[ki] for cb in cp_buffers if ki < len(cb)]
                    w_list = [cw[ki] for cw in cp_weight_vals if ki < len(cw)]
                    if len(t_list) >= 2:
                        result = _process_single_key(key, t_list, w_list)
                        result = _apply_density(result, key)
                        merged[key] = result.cpu()
                    elif len(t_list) == 1:
                        scaled = apply_component_scaling(
                            t_list[0], key,
                            weight_unet, weight_clip, weight_vae, weight_te)
                        merged[key] = (scaled * w_list[0]).cpu()
                    if on_substep:
                        on_substep(1)
                continue

            # ── Stack tensors per-checkpoint: [batch, rows, cols] ──
            # try/except covers stacking AND merge to handle OOM during
            # torch.stack() (previously outside the merge_fn try/except).
            # If stacking itself OOMs, cp_buffers still exists and we extract
            # from cp_buffers directly. If merge_fn OOMs, cp_buffers was freed
            # and we extract from stacked_cp instead.
            try:
                stacked_cp: List[torch.Tensor] = []
                stacked_cp_weights: List[float] = []
                for cp_idx in range(num_sources):
                    if cp_buffers[cp_idx]:
                        stacked_cp.append(
                            torch.stack(cp_buffers[cp_idx], dim=0))
                        # All keys share the same checkpoint weight
                        stacked_cp_weights.append(cp_weight_vals[cp_idx][0])

                # Free individual GPU tensors — only stacked version needed.
                # Without this, cp_buffers (~4 GB for largest Flux groups) coexists
                # with stacked_cp (~4 GB), doubling source-data GPU memory and
                # causing CUDA OOM when merge intermediates are added on top.
                del cp_buffers
                del cp_weight_vals

                if len(stacked_cp) < 2:
                    # Not enough sources for batched merge — process individually
                    # to avoid dropping unique keys (e.g. position embeddings
                    # present in only 1 checkpoint).
                    # Note: cp_buffers was freed above — extract from stacked_cp instead.
                    for ki, key in enumerate(batch_keys):
                        t_list = [sc[ki] for sc in stacked_cp if ki < sc.size(0)]
                        w_list = [stacked_cp_weights[si] for si, sc in enumerate(stacked_cp) if ki < sc.size(0)]
                        if len(t_list) >= 2:
                            result = _process_single_key(key, t_list, w_list)
                            result = _apply_density(result, key)
                            merged[key] = result.cpu()
                        elif len(t_list) == 1:
                            # Single-source key: component-scale and weight
                            scaled = apply_component_scaling(
                                t_list[0], key,
                                weight_unet, weight_clip, weight_vae, weight_te)
                            merged[key] = (scaled * w_list[0]).cpu()
                        if on_substep:
                            on_substep(1)
                    continue

                # Component scaling — verify all keys belong to same component
                rep_key = batch_keys[0]
                rep_category = categorize_key(rep_key)
                all_same_comp = all(categorize_key(k) == rep_category for k in batch_keys)
                if not all_same_comp:
                    # Mixed-component group — fall back to per-key processing
                    # Note: cp_buffers was freed above — extract from stacked_cp instead.
                    for ki, key in enumerate(batch_keys):
                        t_list = [sc[ki] for sc in stacked_cp if ki < sc.size(0)]
                        w_list = [stacked_cp_weights[si] for si, sc in enumerate(stacked_cp) if ki < sc.size(0)]
                        if len(t_list) >= 2:
                            result = _process_single_key(key, t_list, w_list)
                            result = _apply_density(result, key)
                            merged[key] = result.cpu()
                        elif len(t_list) == 1:
                            # Single-source key: component-scale and weight
                            scaled = apply_component_scaling(
                                t_list[0], key,
                                weight_unet, weight_clip, weight_vae, weight_te)
                            merged[key] = (scaled * w_list[0]).cpu()
                        if on_substep:
                            on_substep(1)
                    continue
                # All same component — proceed with batched scaling
                for idx in range(len(stacked_cp)):
                    stacked_cp[idx] = apply_component_scaling(
                        stacked_cp[idx], rep_key,
                        weight_unet, weight_clip, weight_vae, weight_te)

                # Magnitude equalization
                if magnitude_scaling != "none" and len(stacked_cp) >= 2:
                    orig_indices = [i for i, sd in enumerate(sds)
                                    if rep_key in sd]
                    if 0 in orig_indices:
                        ref_idx = orig_indices.index(0)
                        stacked_cp = apply_magnitude_scaling(
                            stacked_cp, stacked_cp_weights,
                            magnitude_scaling, max_scaling_factor,
                            ref_idx=ref_idx)

                # Rank adjustment (no-op)
                stacked_cp = adjust_ranks(stacked_cp, rep_key)

                # ── Single merge call on stacked 3D tensors ────────────
                memory_guard()  # Check right before the big allocation
                batch_result = merge_fn(
                    stacked_cp, stacked_cp_weights,
                    blend_mode=blend_mode, **merge_kwargs)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(e):
                    print(f"   ⚠️ CUDA OOM on shape group {shape} — falling back to per-key CPU processing")
                    # Determine whether cp_buffers was already freed (OOM at merge_fn
                    # or during scaling) or still exists (OOM at torch.stack above).
                    try:
                        _ = cp_buffers
                        cp_freed = False
                    except NameError:
                        cp_freed = True

                    if cp_freed:
                        # cp_buffers was deleted — extract from stacked_cp
                        for ki, key in enumerate(batch_keys):
                            t_list = [sc[ki].cpu() for sc in stacked_cp if ki < sc.size(0)]
                            w_list = [stacked_cp_weights[si] for si, sc in enumerate(stacked_cp) if ki < sc.size(0)]
                            if len(t_list) >= 2:
                                result = _process_single_key(key, t_list, w_list)
                                result = _apply_density(result, key)
                                merged[key] = result.cpu()
                            elif len(t_list) == 1:
                                scaled = apply_component_scaling(
                                    t_list[0], key,
                                    weight_unet, weight_clip, weight_vae, weight_te)
                                merged[key] = (scaled * w_list[0]).cpu()
                            if on_substep:
                                on_substep(1)
                    else:
                        # cp_buffers still exists — extract directly (OOM at torch.stack)
                        for ki, key in enumerate(batch_keys):
                            t_list = [cb[ki].cpu() for cb in cp_buffers if ki < len(cb)]
                            w_list = [cw[ki] for cw in cp_weight_vals if ki < len(cw)]
                            if len(t_list) >= 2:
                                result = _process_single_key(key, t_list, w_list)
                                result = _apply_density(result, key)
                                merged[key] = result.cpu()
                            elif len(t_list) == 1:
                                scaled = apply_component_scaling(
                                    t_list[0], key,
                                    weight_unet, weight_clip, weight_vae, weight_te)
                                merged[key] = (scaled * w_list[0]).cpu()
                            if on_substep:
                                on_substep(1)
                    # Free source tensor references before continuing
                    # (the normal cleanup after the try/except is skipped by this continue)
                    for key in keys:
                        for sd in sds:
                            sd.pop(key, None)
                    continue
                else:
                    raise

            # ── Unstack results and assign per key ─────────────────
            for ki, key in enumerate(batch_keys):
                key_result = batch_result[ki]
                # Per-key energy preservation (not per-batch), to avoid
                # over/under-correcting individual keys with different
                # energy profiles within the same shape batch.
                if energy_preservation:
                    # Extract per-key tensors from stacked_cp (cp_buffers was freed
                    # after stacking to eliminate double GPU allocation).
                    per_key_tensors = [sc[ki] for sc in stacked_cp if ki < sc.size(0)]
                    key_result = ensure_energy_preservation_triple(
                        key_result, per_key_tensors, stacked_cp_weights,
                        threshold=0.8, gain_min=1.0, target='avg')
                key_result = _apply_density(key_result, key)
                merged[key] = key_result.cpu()

            if on_substep:
                on_substep(len(batch_keys))
            print(f"   🕐 [t={time.time()-_mt_t0:.1f}s] merge_triple_method: shape-batch group done ({len(batch_keys)} keys, shape={shape})")
            comfyui_yield()  # Prevent UI stalling between shape groups

            # ── Free GPU memory before next shape group ──
            # try/except safety net: torch.cuda.empty_cache() triggers a CUDA
            # sync which can surface ASYNCHRONOUS CUDA OOM errors from the
            # merge_fn that just completed.  Without this guard, such delayed
            # OOMs propagate past both Fix 7 and Fix 5 to the weave-level
            # handler, causing unnecessary CPU fallback for ALL remaining batches.
            try:
                if device.type == 'cuda':
                    # Free stacked/intermediate merge tensors
                    del stacked_cp
                    del batch_result
                    # Free processed source tensors from GPU (results are in merged dict on CPU)
                    for key in keys:
                        for sd in sds:
                            sd.pop(key, None)
                    torch.cuda.empty_cache()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(e):
                    # Delayed OOM from a previous operation surfacing at sync point.
                    # The shape group already completed successfully (merged dict populated).
                    # Just log and continue — no data loss.
                    print(f"   ⚠️ Delayed CUDA OOM during cleanup after shape group {shape} — continuing (no data loss)")
                else:
                    raise

        # ═══════════════════════════════════════════════════════════
        # PHASE B: Sequential processing (remaining keys)
        # ═══════════════════════════════════════════════════════════
        if sequential_key_list:
            effective_bs = batch_size if streaming else len(sequential_key_list)
            for batch_start in range(0, len(sequential_key_list), effective_bs):
                memory_guard()
                seq_keys = sequential_key_list[
                    batch_start:batch_start + effective_bs]
                for key in seq_keys:
                    tensors = []
                    valid_weights = []

                    for i, sd in enumerate(sds):
                        if key in sd:
                            t = sd[key]
                            t_bytes = t.numel() * t.element_size()
                            if t_bytes > MAX_TENSOR_BYTES:
                                print(f"   ⚠️ Corrupt tensor '{key}' in source {i}: "
                                      f"{t_bytes / (1024**3):.2f} GB exceeds "
                                      f"{MAX_TENSOR_BYTES / (1024**3):.0f} GB limit – skipping")
                                if key not in corrupted_keys:
                                    corrupted_keys.append(key)
                                continue
                            # Guard redundant .to() calls — no-op on correct device/dtype
                            if t.device != device:
                                t = t.to(device)
                            t = _ensure_compute_dtype(t, resolved_dtype)
                            tensors.append(t)
                            valid_weights.append(weights[i])

                    if len(tensors) < 2:
                        if tensors:
                            scaled = apply_component_scaling(
                                tensors[0], key,
                                weight_unet, weight_clip, weight_vae, weight_te)
                            merged[key] = (scaled * valid_weights[0]).cpu()
                        # Free GPU tensor reference even for single-source keys
                        for sd in sds:
                            sd.pop(key, None)
                        continue

                    result = _process_single_key(key, tensors, valid_weights)
                    result = _apply_density(result, key)
                    merged[key] = result.cpu()

                    # Free GPU tensor references immediately — matches batched path behavior
                    for sd in sds:
                        sd.pop(key, None)
                    del tensors
                    if on_substep:
                        on_substep(1)

                comfyui_yield()  # Prevent UI stalling between sequential batches

                if streaming and device.type == 'cuda':
                    torch.cuda.empty_cache()

    # No internal tracker cleanup — on_substep callback handles all advancement.

    # ═══════════════════════════════════════════════════════════════════
    # Fallback: no batching — pure sequential (existing behavior)
    # ═══════════════════════════════════════════════════════════════════
    else:
        print(f"   🕐 [t={time.time()-_mt_t0:.1f}s] merge_triple_method: shape batching not viable — using pure sequential")

        effective_bs = batch_size if streaming else total_keys
        for batch_start in range(0, total_keys, effective_bs):
            memory_guard()
            batch_keys = key_list[batch_start:batch_start + effective_bs]
            for key in batch_keys:
                tensors = []
                valid_weights = []

                for i, sd in enumerate(sds):
                    if key in sd:
                        t = sd[key]
                        t_bytes = t.numel() * t.element_size()
                        if t_bytes > MAX_TENSOR_BYTES:
                            print(f"   ⚠️ Corrupt tensor '{key}' in source {i}: "
                                  f"{t_bytes / (1024**3):.2f} GB exceeds "
                                  f"{MAX_TENSOR_BYTES / (1024**3):.0f} GB limit – skipping")
                            if key not in corrupted_keys:
                                corrupted_keys.append(key)
                            # Free the corrupt tensor reference from sds
                            for sd in sds:
                                sd.pop(key, None)
                            continue
                        # Guard redundant .to() calls — no-op on correct device/dtype
                        if t.device != device:
                            t = t.to(device)
                        t = _ensure_compute_dtype(t, resolved_dtype)
                        tensors.append(t)
                        valid_weights.append(weights[i])

                if len(tensors) < 2:
                    if tensors:
                        scaled = apply_component_scaling(
                            tensors[0], key,
                            weight_unet, weight_clip, weight_vae, weight_te)
                        merged[key] = (scaled * valid_weights[0]).cpu()
                    # Free GPU tensor reference even for single-source keys
                    for sd in sds:
                        sd.pop(key, None)
                    continue

                result = _process_single_key(key, tensors, valid_weights)
                result = _apply_density(result, key)
                merged[key] = result.cpu()

                # Free GPU tensor references immediately
                for sd in sds:
                    sd.pop(key, None)
                del tensors
                if on_substep:
                    on_substep(1)

            comfyui_yield()  # Prevent UI stalling between sequential batches

            if streaming and device.type == 'cuda':
                torch.cuda.empty_cache()

        # No internal tracker cleanup — on_substep callback handles all advancement.

    # Free intermediate structures — both paths (Phase B and Fallback) are done
    del batchable_shapes
    del sequential_key_list

    # Belt-and-suspenders: ensure any lingering GPU tensors are freed
    # before returning to the weaver
    cleanup_memory()

    if corrupted_keys:
        print(f"   ⚠️ {len(corrupted_keys)} corrupted/absurd tensors skipped during merge")
    print(f"   🕐 [t={time.time()-_mt_t0:.1f}s] merge_triple_method returning ({len(merged)} keys)")
    return merged, corrupted_keys
