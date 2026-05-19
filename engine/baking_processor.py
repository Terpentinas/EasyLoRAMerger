"""
Easy LoRA Baker — Core tensor-level LoRA baking engine (orchestration layer).

Rewritten as a thin orchestration core that imports specialized functions
from 5 sub-modules:
  - baking_processor_constants: Registry data (prefix mappings, patterns)
  - baking_processor_delta:     Delta reconstruction, alpha/rank scaling
  - baking_processor_baking:    Bake methods, shape alignment, assembly
  - baking_processor_matching:  Key matching, reverse map, strategy cascade
  - baking_processor_sd15:      SD1.5 auto-scale channel safety only
                                (old SD1.5 matching removed — bridge handles it)

The SmartBakingProcessor class delegates all method calls to module-level
functions assigned as class methods. External API (baker_node.py import)
is unchanged: SmartBakingProcessor still lives in engine/baking_processor.py.
"""

import torch
import time
_T0 = time.time()
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import Counter

from ..utils import (
    DeviceManager,
    ProgressTracker,
    categorize_checkpoint_key,
    comfyui_yield,
    memory_optimized_merge,
    get_available_ram,
    cleanup_memory,
    check_ram_guard,
)
from ..config import DevicePrecisionConfig

# =====================================================================
# Imports from sub-modules
# =====================================================================

# Delta reconstruction — lazy and eager
from .baking_processor_delta import (
    _reconstruct_lora_delta,
    _LazyDeltaDict,
    _MatchedDeltas,
    _group_lora_pairs,
    _ShapeInfo,
)

# FP8 quantizer — scale-then-cast, BF16 preservation, companion scales
from .fp8_quantizer import (
    quantize_to_fp8,
    compute_weight_scale,
    should_preserve_bf16,
    quantize_weight_to_fp8_with_scales,
)

# Baking methods and assembly
from .baking_processor_baking import (
    _check_shape_compatible,
    _align_delta_shape,
    _align_delta_safe,
    _orthogonal_projection_2d,
    _orthogonal_projection,
    _get_component_weight,
    bake_linear,
    bake_impact_weighted,
    bake_orthogonal,
    _assemble_output,
    _assemble_output_lazy,
    _sanitize_or_revert,
    _sanitize_baked,
    _build_metadata,
    _build_forensic_report,
)

# Key matching
from .baking_processor_matching import (
    _lora_key_to_checkpoint_key,
    _try_underscore_to_dot_conversion,
    _try_te_conditioner_prefix,
    _build_reverse_key_map,
    _is_spatial_depth_block,
    _extract_block_id,
    _build_shape_block_index,       # NEW: O(1) shape+block index for Strategy 4
    _build_shape_index,             # NEW: O(1) shape index for Strategy 5
    _global_search_by_shape_block,
    _categorize_lora_key,
    _create_effect_bar,
    _find_matching_keys,
)

# SD1.5 Diffusers Bridge
from .diffusers_bridge import (
    detect_native_sd15_checkpoint,
    detect_diffusers_sd15_lora,
    normalize_diffusers_preserving,
    build_diffusers_reverse_map,
    match_diffusers_deltas,
)


# =====================================================================
# SmartBakingProcessor — Orchestration Core
# =====================================================================

class SmartBakingProcessor:
    """
    Performs tensor-level LoRA baking into full model checkpoints.

    Workflow:
        1. Load checkpoint state dict (with metadata)
        2. Load LoRA state dict (from lora_data tuple or temp file)
        3. Apply alpha/rank scaling correction to LoRA tensors
        4. Reconstruct LoRA deltas from A/B pairs: delta = B @ A
        5. Match LoRA keys to checkpoint keys
        6. Apply selected baking method (linear/impact_weighted/orthogonal)
        7. Apply safety features (preservation mask, TE toggle)
        8. Assemble final state dict with VAE pass-through
        9. Embed baking metadata history
        10. Save as .safetensors file

    All method implementations are delegated to module-level functions
    from the baking_processor_* sub-modules.
    """

    # ------------------------------------------------------------------
    # Static method assignments (pure functions, no self)
    # ------------------------------------------------------------------
    _reconstruct_lora_delta = staticmethod(_reconstruct_lora_delta)

    _check_shape_compatible = staticmethod(_check_shape_compatible)
    _align_delta_shape = staticmethod(_align_delta_shape)
    _align_delta_safe = staticmethod(_align_delta_safe)
    _orthogonal_projection_2d = staticmethod(_orthogonal_projection_2d)
    _orthogonal_projection = staticmethod(_orthogonal_projection)
    _get_component_weight = staticmethod(_get_component_weight)

    _assemble_output = staticmethod(_assemble_output)
    _assemble_output_lazy = staticmethod(_assemble_output_lazy)
    _sanitize_or_revert = staticmethod(_sanitize_or_revert)
    _sanitize_baked = staticmethod(_sanitize_baked)
    _build_metadata = staticmethod(_build_metadata)
    _build_forensic_report = staticmethod(_build_forensic_report)

    _lora_key_to_checkpoint_key = staticmethod(_lora_key_to_checkpoint_key)
    _try_underscore_to_dot_conversion = staticmethod(_try_underscore_to_dot_conversion)
    _try_te_conditioner_prefix = staticmethod(_try_te_conditioner_prefix)
    _build_reverse_key_map = staticmethod(_build_reverse_key_map)
    _is_spatial_depth_block = staticmethod(_is_spatial_depth_block)
    _extract_block_id = staticmethod(_extract_block_id)
    _global_search_by_shape_block = staticmethod(_global_search_by_shape_block)
    _categorize_lora_key = staticmethod(_categorize_lora_key)
    _create_effect_bar = staticmethod(_create_effect_bar)

    # ------------------------------------------------------------------
    # Instance method assignments (functions that take self)
    # ------------------------------------------------------------------
    bake_linear = bake_linear
    bake_impact_weighted = bake_impact_weighted
    bake_orthogonal = bake_orthogonal
    _find_matching_keys = _find_matching_keys

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, device: str = "auto", precision: str = "auto",
                 verbose: bool = False,
                 device_precision: Optional['DevicePrecisionConfig'] = None):
        if device_precision is not None:
            self.device = device_precision.device
            self.dtype = device_precision.dtype
        else:
            self.device = DeviceManager.get_device(device)
            self.dtype = DeviceManager.get_dtype(precision, self.device)
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Early-abort helper: unifies the two nearly-identical error returns
    # ------------------------------------------------------------------

    def _abort_bake(
        self,
        error_msg: str,
        ckpt_sd: Dict[str, torch.Tensor],
        baking_method: str,
        strength: float,
        lora_source: str,
        checkpoint_name: str,
        original_metadata: Optional[Dict[str, str]],
        weight_unet: float,
        weight_te: float,
        weight_clip: float,
        weight_vae: float,
        metadata_mode: str,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], str, Dict[str, Any]]:
        """Early-abort return with error profile — preserves metadata/forensic shape."""
        print(f"❌ {error_msg}")
        # CRITICAL: When ckpt_sd is a _LazyCheckpointMapping, dict(ckpt_sd) would
        # trigger .items() which loads ALL tensors via mmap (17 GiB for Klein 9B).
        # Return the lazy mapping as-is (no tensor data loaded) or an empty dict.
        if hasattr(ckpt_sd, 'filepath'):
            output_for_abort: Dict[str, torch.Tensor] = {}  # type: ignore[assignment]
        else:
            output_for_abort = dict(ckpt_sd)
        return (
            output_for_abort,
            self._build_metadata(baking_method, strength,
                                 lora_source, checkpoint_name,
                                 0, len(ckpt_sd), original_metadata,
                                 weight_unet, weight_te, weight_clip, weight_vae,
                                 metadata_mode),
            self._build_forensic_report(baking_method, strength,
                                        lora_source, checkpoint_name,
                                        0, len(ckpt_sd), len(ckpt_sd),
                                        weight_unet=weight_unet, weight_te=weight_te,
                                        weight_clip=weight_clip, weight_vae=weight_vae,
                                        component_breakdown=getattr(self, '_component_breakdown', None),
                                        delta_analysis=getattr(self, '_delta_analysis', None)),
            {"error": error_msg},
        )

    # ------------------------------------------------------------------
    # Main Entry Point: bake
    # ------------------------------------------------------------------

    def bake(
        self,
        ckpt_sd: Dict[str, torch.Tensor],
        lora_sd: Dict[str, torch.Tensor],
        ckpt_header: Optional[Dict[str, Any]] = None,
        baking_method: str = "linear",
        strength: float = 1.0,
        energy_concentration: float = 0.80,
        lora_source: str = "unknown",
        checkpoint_name: str = "unknown",
        original_metadata: Optional[Dict[str, str]] = None,
        weight_unet: float = 1.0,
        weight_te: float = 1.0,
        weight_clip: float = 1.0,
        weight_vae: float = 1.0,
        metadata_mode: str = "preserve_a",
        batch_size: int = 64,
        detected_fp8: bool = False,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, str],
        str,
        Dict[str, Any],
    ]:
        """
        Execute the full baking pipeline with per-component scaling.

        Args:
            ckpt_sd: Source checkpoint state dict (can be _LazyCheckpointMapping).
            lora_sd: LoRA state dict to bake in.
            ckpt_header: Optional safetensors header dict for mmap-based
                         matching and lazy assembly. When provided, the
                         pipeline avoids loading tensor data for key matching
                         and assembles output as a lazy mapping.
            baking_method: "linear", "impact_weighted", or "orthogonal".
            strength: Overall baking strength multiplier.
            energy_concentration: Energy threshold for impact-weighted method.
            lora_source: Description of LoRA source for metadata.
            checkpoint_name: Source checkpoint filename for metadata.
            original_metadata: Original checkpoint .safetensors metadata dict.
            weight_unet: Per-component weight for U-Net (diffusion model) keys.
            weight_te: Per-component weight for Text Encoder keys.
            weight_clip: Per-component weight for CLIP Vision keys.
            weight_vae: Per-component weight for VAE keys.
            metadata_mode: "none" (baking only), "preserve_a" (original priority),
                           or "merge_basic" (baking priority).
            detected_fp8: When True, the checkpoint has FP8 native weights.
                          Computation is performed in bfloat16 (via Fix 1 in
                          _safe_bake_add) and stale FP8 scale keys are stripped
                          from the output.  Preserved (non-baked) keys are
                          auto-converted from FP8 to bfloat16 on access.

        Returns:
            output_sd:      Final assembled state dict with baked weights
            metadata:       Metadata dict for the .safetensors header
            forensic_report: Human-readable forensic report string (Lora Studio style)
            impact_profile:  Dict with impact analysis details
        """
        # Detect if ckpt_sd is a lazy mapping (has .filepath attribute)
        # _LazyCheckpointMapping from musubi_checkpoint_studio.py stores the
        # original file path, enabling lazy assembly without a separate path arg.
        self._lazy_mode = hasattr(ckpt_sd, 'filepath')
        self._ckpt_path = getattr(ckpt_sd, 'filepath', None)
        # 🔥 FP8: Store detection flag for assembly pass-through
        self._detected_fp8 = detected_fp8

        with memory_optimized_merge():
            # Step 2a: SD1.5 Diffusers Bridge detection
            use_diffusers_bridge = False
            if detect_native_sd15_checkpoint(ckpt_sd) and detect_diffusers_sd15_lora(lora_sd):
                print("   🔄 SD1.5 Diffusers Bridge ACTIVE")
                use_diffusers_bridge = True

            comfyui_yield()

            print("\n" + "=" * 50)
            print("🔥 Easy LoRA Baker — Baking Pipeline")
            precision_info = (f"(compute uses float16 fallback)"
                              if self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                              else "")
            print(f"   🎯 Precision: {self.dtype} {precision_info}")
            print("=" * 50)

            # =====================================================================
            # PIPELINE PROGRESS: Track outer bake phases
            # =====================================================================
            # The inner baking methods (linear/impact_weighted/orthogonal) and
            # key matching each have their own ProgressTracker.  This outer tracker
            # covers the phases BETWEEN those: normalization, delta reconstruction,
            # statistics, assembly, and metadata.
            PIPELINE_PHASES = 5
            pipeline_progress = ProgressTracker(total=PIPELINE_PHASES, desc="Bake pipeline")
            pipeline_progress.update(0)

            # Step 2b: Normalize LoRA keys to master format
            # (handles all formats: SDXL Kohya, ComfyUI native, Musubi, Flux Klein, Z-Image)
            if use_diffusers_bridge:
                # Preserve Diffusers block structure (no down_blocks→input_blocks conversion)
                lora_sd, lora_key_map = normalize_diffusers_preserving(lora_sd)
                print(f"   🔄 Normalized {len(lora_sd)} keys (Diffusers-preserving)")
            else:
                from .identity_normalizer import identity_normalize
                lora_sd, lora_key_map = identity_normalize(lora_sd)
                print(f"   🔄 Normalized {len(lora_sd)} keys to master format")

            pipeline_progress += 1  # Phase 1: Normalization done

            # Step 3: Group LoRA pairs for on-demand delta reconstruction
            # Instead of reconstructing all ≈112 deltas upfront (≈9.6 GB), store only
            # the A/B component tensors (≈14 MB) and reconstruct each delta on demand
            # during matching (shape inference) and baking (.pop() → one at a time).
            #
            # GPU acceleration: if CUDA is available, the A/B tensors are moved to GPU
            # during _reconstruct_one() inside _LazyDeltaDict.reconstruct(), keeping the
            # fast matmul path while avoiding VRAM accumulation since only one delta
            # is reconstructed at a time.
            device_for_reconstruction = self.device if 'cuda' in str(self.device) else None
            lora_pairs, lora_standalone = _group_lora_pairs(lora_sd)
            lora_deltas = _LazyDeltaDict(lora_pairs, lora_standalone, device=device_for_reconstruction)

            if not lora_deltas:
                return self._abort_bake(
                    "No LoRA deltas found — aborting bake",
                    ckpt_sd, baking_method, strength,
                    lora_source, checkpoint_name, original_metadata,
                    weight_unet, weight_te, weight_clip, weight_vae,
                    metadata_mode,
                )

            # Step 5: Log all LoRA delta keys before matching (debug aid)
            # Models ComfyUI's approach at comfy/sd.py:78-106 where convert_lora+load_lora
            # produce the patch_dict; we log pre-matching deltas to trace key coverage.
            print(f"\n   📋 LoRA delta keys ({len(lora_deltas)} total):")
            # 🔥 FIX: Use _categorize_lora_key() instead of naive 'lora_unet'/'lora_te' substring
            # checks. After Diffusers→ComfyUI conversion, keys use diffusion_model.* prefix
            # (not lora_unet_*), so the old substring check always returned 0 UNet keys.
            # _categorize_lora_key() recognizes all known normalized formats:
            #   unet: diffusion_model.*, input_blocks.*, output_blocks.*, middle_block.*
            #   te:   text_model.*, te.*, lora_te_*, transformer.text_model.*
            #   clip: clip*
            #   vae:  vae*, first_stage*
            comp_counts: Counter = Counter()
            for k in lora_deltas:
                comp_counts[self._categorize_lora_key(k)] += 1
            unet_count = comp_counts.get('unet', 0)
            te_count = comp_counts.get('te', 0)
            clip_count = comp_counts.get('clip', 0)
            vae_count = comp_counts.get('vae', 0)
            other_count = len(lora_deltas) - unet_count - te_count - clip_count - vae_count
            print(f"      UNet: {unet_count} keys, TE: {te_count} keys, "
                  f"CLIP: {clip_count} keys, VAE: {vae_count} keys, Other: {other_count} keys")
            if self._verbose:
                for i, (base, delta) in enumerate(sorted(lora_deltas.items())):
                    if isinstance(delta, _ShapeInfo):
                        norm_str = "N/A (lazy)"
                    else:
                        norm_str = f"{torch.norm(delta.float()).item():.4f}"
                    print(f"      [{i:>3d}] {str(base):>70s} shape={str(list(delta.shape)):>20s} norm={norm_str}")

            comfyui_yield()
            pipeline_progress += 1  # Phase 2: Delta reconstruction done

            # Step 5: Match keys
            if use_diffusers_bridge:
                # Build Diffusers-aware reverse map for 1:1 exact matching
                bridge_reverse_map = build_diffusers_reverse_map(ckpt_sd)
                matched_deltas = match_diffusers_deltas(ckpt_sd, lora_deltas, bridge_reverse_map)
                match_stats = {"diffusers_bridge_1:1": len(matched_deltas)}
                print(f"\n   📊 Match strategy distribution ({len(matched_deltas)} total):")
                print(f"      diffusers_bridge_1:1: {len(matched_deltas)} keys")
            else:
                matched_deltas, match_stats = self._find_matching_keys(
                    ckpt_sd, lora_deltas, return_stats=True,
                    ckpt_header=ckpt_header,
                )
                # Log match strategy distribution
                if match_stats:
                    print(f"\n   📊 Match strategy distribution ({sum(match_stats.values())} total):")
                    for strategy, count in sorted(match_stats.items(), key=lambda x: -x[1]):
                        print(f"      {strategy}: {count} keys")
            # NOTE: matched_deltas stay on CPU — moved to GPU on-demand per-key
            # inside _prepare_bake_key / each bake method's inner loop.
            if not matched_deltas:
                return self._abort_bake(
                    "No keys matched between LoRA and checkpoint — aborting bake",
                    ckpt_sd, baking_method, strength,
                    lora_source, checkpoint_name, original_metadata,
                    weight_unet, weight_te, weight_clip, weight_vae,
                    metadata_mode,
                )

            pipeline_progress += 1  # Phase 3: Key matching done

            # Step 6: Apply baking method with per-component scaling
            BAKE_DISPATCH = {
                "linear": self.bake_linear,
                "impact_weighted": self.bake_impact_weighted,
                "orthogonal": self.bake_orthogonal,
            }

            bake_fn = BAKE_DISPATCH.get(baking_method)
            if bake_fn is None:
                print(f"⚠️ Unknown baking method '{baking_method}' — falling back to linear")
                bake_fn = self.bake_linear
                baking_method = "linear"

            # ── RAM Guard: per-batch GPU memory guard ──
            if matched_deltas and ckpt_header is not None:
                sample_delta = next(iter(matched_deltas.values()))
                per_key_bytes = sample_delta.numel() * sample_delta.element_size() * 3
                batch_peak = per_key_bytes * batch_size
                _avail_ram = get_available_ram()
                if _avail_ram is not None:
                    _ram_gb = _avail_ram / (1024**3)
                    _peak_gb = batch_peak / (1024**3)
                    print(f"🧠 RAM Guard (baking): Estimated peak ~{_peak_gb:.2f} GB/batch, "
                          f"Available RAM ~{_ram_gb:.2f} GB, "
                          f"85% threshold ~{_ram_gb * 0.85:.2f} GB")
                    if batch_peak > _avail_ram * 0.85:
                        old_bs = batch_size
                        batch_size = 1
                        print(f"   ⚠️ Batch size {old_bs} exceeds RAM threshold — "
                              f"falling back to sequential (batch_size=1)")
                    else:
                        print(f"✅ RAM Guard (baking): Batch size {batch_size} within safe limits")
                else:
                    print(f"⚠️ RAM Guard (baking): Cannot detect available RAM — proceeding")

            # Initialize output dict (always dict mode — streaming path removed)
            output_dict: Dict[str, torch.Tensor] = {}

            if baking_method == "impact_weighted":
                delta_norms, delta_ratios = bake_fn(
                    ckpt_sd, matched_deltas, output_dict, strength, energy_concentration,
                    weight_unet, weight_te, weight_clip, weight_vae,
                    batch_size=batch_size,
                )
            else:
                delta_norms, delta_ratios = bake_fn(
                    ckpt_sd, matched_deltas, output_dict, strength,
                    weight_unet, weight_te, weight_clip, weight_vae,
                    batch_size=batch_size,
                )

            # Capture matched count before deletion
            _matched_count = len(matched_deltas) if matched_deltas is not None else 0

            # matched_deltas and lora_deltas are no longer needed after baking.
            # In lazy mode, _MatchedDeltas is a thin wrapper (few KB) and
            # _LazyDeltaDict holds only component tensors (≈14 MB for 112 pairs).
            # Delete eagerly to free residual memory (Fix #6).
            del matched_deltas, lora_deltas
            cleanup_memory()
            comfyui_yield()


            # Alias output_dict as baked for minimal downstream changes
            baked = output_dict

            # =====================================================================
            # COMPONENT BAKED COUNTING: Count how many keys per component were baked
            # =====================================================================
            if hasattr(self, '_component_breakdown') and self._component_breakdown:
                _comp_map = {
                    'model_diffusion': 'unet',
                    'te': 'te',
                    'clip_vision': 'clip',
                    'vae': 'vae',
                }
                for ckpt_key in baked:
                    ckpt_comp = categorize_checkpoint_key(ckpt_key)
                    mapped = _comp_map.get(ckpt_comp, 'unet')
                    if mapped in self._component_breakdown:
                        self._component_breakdown[mapped].setdefault("baked", 0)
                        self._component_breakdown[mapped]["baked"] += 1
                # Ensure all components have a "baked" field
                for comp, data in self._component_breakdown.items():
                    data.setdefault("baked", 0)

            # =====================================================================
            # BAKE DROPOUT REPORT: Keys matched but not baked
            # =====================================================================
            if hasattr(self, '_bake_dropped') and self._bake_dropped:
                drop_by_reason: Dict[str, int] = {}
                for key, reason in self._bake_dropped.items():
                    # Categorize by reason type (first word before colon)
                    reason_cat = reason.split(":")[0].strip() if ":" in reason else reason
                    drop_by_reason[reason_cat] = drop_by_reason.get(reason_cat, 0) + 1
                total_dropped = len(self._bake_dropped)
                print(f"\n   ⚠️ BAKE DROPOUT: {total_dropped} matched keys were NOT baked:")
                for reason, count in sorted(drop_by_reason.items(), key=lambda x: -x[1]):
                    print(f"      {reason}: {count} keys")
                if self._verbose:
                    for key, reason in sorted(self._bake_dropped.items()):
                        print(f"        - {key}: {reason}")

            # =====================================================================
            # DELTA STATISTICS: Use norms/ratios collected during bake loop
            # =====================================================================
            print("\n" + "-" * 50)
            print("📊 DELTA STATISTICS (aggregate across all baked keys):")

            if delta_norms:
                mean_dn = statistics.mean(delta_norms)
                max_dn = max(delta_norms)
                mean_ratio = statistics.mean(delta_ratios)
                max_ratio = max(delta_ratios)
                median_ratio = statistics.median(delta_ratios)
                # Manual stddev — Python 3.12's statistics.stdev has an internal
                # Fraction conversion issue that raises:
                #   AttributeError: 'float' object has no attribute 'numerator'
                if len(delta_ratios) > 1:
                    mean_r = statistics.mean(delta_ratios)
                    variance = sum((x - mean_r) ** 2 for x in delta_ratios) / (len(delta_ratios) - 1)
                    std_ratio = variance ** 0.5
                else:
                    std_ratio = 0.0
                mean_ratio_percent = round(mean_ratio * 100, 2)
                max_ratio_percent = round(max_ratio * 100, 2)
                self._delta_analysis = {
                    "layers_analyzed": len(delta_norms),
                    "mean_ratio_percent": mean_ratio_percent,
                    "median_ratio_percent": round(median_ratio * 100, 2),
                    "max_ratio_percent": max_ratio_percent,
                    "std_ratio_percent": round(std_ratio * 100, 2),
                    "mean_delta_norm": round(mean_dn, 4),
                    "max_delta_norm": round(max_dn, 4),
                    "effect_visual": self._create_effect_bar(mean_ratio_percent, max_ratio_percent),
                }
                print(f"   Layers analyzed:    {len(delta_norms)}")
                print(f"   Mean delta norm:    {mean_dn:.4e}")
                print(f"   Max delta norm:     {max_dn:.4e}")
                print(f"   Mean ratio (%):     {mean_ratio*100:.2f}%")
                print(f"   Median ratio (%):   {median_ratio*100:.2f}%")
                print(f"   Max ratio (%):      {max_ratio*100:.2f}%")
                print(f"   Std ratio (%):      {std_ratio*100:.2f}%")
            else:
                self._delta_analysis = {
                    "layers_analyzed": 0, "mean_ratio_percent": 0.0,
                    "median_ratio_percent": 0.0, "max_ratio_percent": 0.0,
                    "std_ratio_percent": 0.0, "mean_delta_norm": 0.0, "max_delta_norm": 0.0,
                    "effect_visual": self._create_effect_bar(0.0, 0.0),
                }
            print("-" * 50)
            pipeline_progress += 1  # Phase 4: Baking + statistics done

            # =====================================================================
            # 🔍 BAKE VERIFICATION: Sample-check baked weights differ from original
            # =====================================================================
            if baked and ckpt_sd:
                _verify_keys = list(baked.keys())[:3]
                _all_differ = True
                for _vk in _verify_keys:
                    if _vk in ckpt_sd:
                        _orig = ckpt_sd[_vk]
                        _baked_t = baked[_vk]
                        if isinstance(_orig, torch.Tensor) and isinstance(_baked_t, torch.Tensor):
                            if _orig.shape == _baked_t.shape:
                                _diff = (_baked_t.to(dtype=torch.float32) - _orig.to(dtype=torch.float32)).abs().max().item()
                                _dtype_match = "✅" if _baked_t.dtype == _orig.dtype else f"⚠️ dtype: {_baked_t.dtype}≠{_orig.dtype}"
                                print(f"   🔍 Bake verify '{_vk[:60]}...': |diff|_max={_diff:.6e} {_dtype_match}")
                                if _diff == 0.0:
                                    _all_differ = False
                            else:
                                print(f"   🔍 Bake verify '{_vk[:60]}...': shape mismatch {_baked_t.shape}≠{_orig.shape}")
                                _all_differ = False
                        else:
                            print(f"   🔍 Bake verify '{_vk[:60]}...': non-tensor type")
                            _all_differ = False
                if _all_differ and _verify_keys:
                    print(f"   ✅ Bake verification passed — sampled weights differ from originals")
                elif _verify_keys:
                    print(f"   ⚠️ Bake verification: some sampled weights match originals (delta=0)")
            else:
                print(f"   ⚠️ Bake verification: no baked keys to verify")

            # NaN/Inf sanitization
            # 🔥 REVERT-TO-ORIGINAL: Pass ckpt_sd so any key with persistent NaN/Inf
            # reverts to its original checkpoint weight instead of being replaced
            # with zeros (which cause black/dark patches in generated images).
            baked = self._sanitize_baked(baked, original_ckpt_sd=ckpt_sd)

            # Close mmap handles if ckpt_sd is a lazy mapping (Fix #4).
            # After sanitize, we no longer need to read from the original checkpoint.
            # The lazy assembly step creates a new mapping from self._ckpt_path instead.
            if hasattr(ckpt_sd, 'close'):
                ckpt_sd.close()

            # ── Measure pre-assembly RAM ──
            _pre_assembly_ram = get_available_ram()
            print(f"   🕐 [t={time.time()-_T0:.1f}s] Starting output assembly (_assemble_output_lazy)")

            # Step 8: Assemble output — always lazy when checkpoint is mmap'd.
            # _assemble_output_lazy creates a _LazyCheckpointMapping with baked keys
            # in the write cache. Non-baked keys stay mmap'd from the original file —
            # zero bytes copied. Compatible with both preview mode
            # (load_state_dict_as_model_objects) and save mode
            # (materialize via dict(items()) then save_safetensors_stream).
            if self._lazy_mode:
                if self._ckpt_path is not None and ckpt_header is not None:
                    _user_fp8 = (
                        self.dtype
                        if self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                        else None
                    )
                    output_sd = _assemble_output_lazy(
                        self._ckpt_path, ckpt_header, baked,
                        detected_fp8=self._detected_fp8,
                        user_fp8_dtype=_user_fp8,
                    )
                else:
                    raise RuntimeError(
                        "Lazy assembly requires both ckpt_path and ckpt_header. "
                        "Cannot fall back to eager assembly — would load 17+ GiB into RAM."
                    )
            else:
                output_sd = self._assemble_output(
                    ckpt_sd, baked,
                    fp8_dequantized=self._detected_fp8,
                    target_dtype=self.dtype,
                )

            # Free the baked dict — its CPU tensors are now held by output_sd
            # (or the lazy mapping's write cache). Reclaim the dict wrapper + any
            # stale references eagerly so cleanup_memory() can free residual GPU pages.
            # NOTE: Keep `baked` alive until all downstream uses (impact_profile,
            # metadata, forensic report, final print) are done.
            # Convert per-key count to local variables first for safe deletion later.
            _baked_count = len(baked)
            cleanup_memory()
            comfyui_yield()

            # ── Measure post-assembly RAM and print delta ──
            _post_assembly_ram = get_available_ram()
            if _pre_assembly_ram is not None and _post_assembly_ram is not None:
                _delta_gb = (_pre_assembly_ram - _post_assembly_ram) / (1024**3)
                print(f"📊 RAM: Pre-assembly {_pre_assembly_ram / (1024**3):.2f} GB → "
                      f"Post-assembly {_post_assembly_ram / (1024**3):.2f} GB "
                      f"(delta: {_delta_gb:+.2f} GB)")
            elif _post_assembly_ram is not None:
                print(f"📊 Post-assembly available RAM: {_post_assembly_ram / (1024**3):.2f} GB")
            cleanup_memory()
            print(f"   🕐 [t={time.time()-_T0:.1f}s] Output assembly complete — {len(output_sd)} keys in output_sd")

            # =====================================================================
            # 🔍 DIAGNOSTIC: Validate baked keys match checkpoint key format
            # =====================================================================
            # After the Section 4d/4e fix in _build_reverse_key_map(), all Flux Klein
            # LoRA keys match via reverse_map (Strategy 0) instead of shape_block
            # (Strategy 4). This diagnostic verifies:
            #   1. Baked keys use checkpoint-native prefix
            #      - Vanilla Flux: transformer.*
            #      - Klein Flux:   double_blocks.* / single_blocks.* (bare)
            #   2. The strategy distribution shows 0 shape_block keys for Flux
            #   3. ComfyUI's "left over keys" in Flux model loading is NORMAL —
            #      ComfyUI internally strips prefixes during loading
            if baked and ckpt_sd:
                # Check if this is a Flux checkpoint
                _ckpt_sample_keys = list(ckpt_sd.keys())[:5]
                _is_flux = any(
                    k.startswith(('transformer.', 'double_blocks.', 'single_blocks.'))
                    for k in _ckpt_sample_keys
                )
                if _is_flux:
                    # Detect checkpoint key prefix style
                    _has_transformer = any(k.startswith('transformer.') for k in _ckpt_sample_keys)
                    _has_bare_blocks = any(
                        k.startswith(('double_blocks.', 'single_blocks.'))
                        for k in _ckpt_sample_keys
                    )
                    _expected_prefix = 'transformer.' if _has_transformer else 'double_blocks./single_blocks.'
                    
                    # Verify baked keys use correct prefix
                    if _has_transformer:
                        _all_proper_prefix = all(k.startswith('transformer.') for k in baked.keys())
                    else:
                        _all_proper_prefix = all(
                            k.startswith(('double_blocks.', 'single_blocks.'))
                            for k in baked.keys()
                        )
                    
                    if _all_proper_prefix:
                        print(f"   ✅ Baked key format: all {len(baked)} keys use "
                              f"'{_expected_prefix}' prefix — correct for Flux model loader")
                        print(f"      (ComfyUI's Flux loader strips prefixes internally;")
                        print(f"       'left over keys' in logs is NORMAL behavior — weights ARE consumed)")
                    else:
                        _wrong_prefix = [k for k in baked.keys()
                                         if _has_transformer and not k.startswith('transformer.')
                                         or not _has_transformer and not k.startswith(('double_blocks.', 'single_blocks.'))]
                        print(f"   ⚠️  {len(_wrong_prefix)} baked keys with unexpected prefix:")
                        for _nk in _wrong_prefix[:3]:
                            print(f"      - {_nk}")
                        print(f"      These may NOT be consumed by ComfyUI's Flux model loader!")

            # Build impact profile (use captured lengths — baked still alive here)
            impact_profile = {
                "baking_method": baking_method,
                "matched_keys": _matched_count,
                "baked_keys": _baked_count,
                "preserved_keys": len(ckpt_sd) - _baked_count,
                "total_keys": len(ckpt_sd),
                "strength": strength,
                "weight_unet": weight_unet,
                "weight_te": weight_te,
                "weight_clip": weight_clip,
                "weight_vae": weight_vae,
            }

            # Step 9: Build metadata
            metadata = self._build_metadata(
                baking_method, strength,
                lora_source, checkpoint_name,
                _baked_count, len(ckpt_sd) - _baked_count,
                original_metadata,
                weight_unet, weight_te, weight_clip, weight_vae,
                metadata_mode,
            )

            # Build forensic report
            forensic_report = self._build_forensic_report(
                baking_method, strength,
                lora_source, checkpoint_name,
                _baked_count, len(ckpt_sd) - _baked_count,
                len(ckpt_sd), impact_profile,
                weight_unet=weight_unet, weight_te=weight_te,
                weight_clip=weight_clip, weight_vae=weight_vae,
                component_breakdown=getattr(self, '_component_breakdown', None),
                delta_analysis=getattr(self, '_delta_analysis', None),
            )

            print("=" * 50)
            print("✅ Baking pipeline complete!")
            print(f"   Baked {_baked_count} keys, preserved {len(ckpt_sd) - _baked_count} keys")
            print(f"   🧱 Weights — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}")
            print(f"   📋 Metadata mode: {metadata_mode}")
            print("=" * 50)
            pipeline_progress += 1  # Phase 5: Assembly + metadata done
            pipeline_progress.complete()

            # Free the baked dict now that all uses are done.
            del baked
            cleanup_memory()

            print(f"   🕐 [t={time.time()-_T0:.1f}s] SmartBakingProcessor.bake returning")
            return output_sd, metadata, forensic_report, impact_profile
