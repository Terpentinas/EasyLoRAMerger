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
import math
import statistics
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import Counter

from ..utils import (
    DeviceManager,
    ProgressTracker,
    categorize_checkpoint_key,
    comfyui_yield,
    memory_optimized_merge,
)
from ..config import DevicePrecisionConfig

# =====================================================================
# Imports from sub-modules
# =====================================================================

# Delta reconstruction
from .baking_processor_delta import (
    _reconstruct_lora_delta,
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
        return (
            dict(ckpt_sd),
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
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, str],
        str,
        Dict[str, Any],
    ]:
        """
        Execute the full baking pipeline with per-component scaling.

        Args:
            ckpt_sd: Source checkpoint state dict.
            lora_sd: LoRA state dict to bake in.
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

        Returns:
            output_sd:      Final assembled state dict with baked weights
            metadata:       Metadata dict for the .safetensors header
            forensic_report: Human-readable forensic report string (Lora Studio style)
            impact_profile:  Dict with impact analysis details
        """
        with memory_optimized_merge():
            # Step 2a: SD1.5 Diffusers Bridge detection
            use_diffusers_bridge = False
            if detect_native_sd15_checkpoint(ckpt_sd) and detect_diffusers_sd15_lora(lora_sd):
                print("   🔄 SD1.5 Diffusers Bridge ACTIVE")
                use_diffusers_bridge = True

            comfyui_yield()

            print("\n" + "=" * 50)
            print("🔥 Easy LoRA Baker — Baking Pipeline")
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

            # Step 3: Reconstruct deltas (alpha/rank scaling applied inside)
            # Use GPU for matmul if device is CUDA (respects user's auto/cuda/cpu choice)
            device_for_reconstruction = self.device if 'cuda' in str(self.device) else None
            lora_deltas = self._reconstruct_lora_delta(lora_sd, device=device_for_reconstruction)

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
                    print(f"      [{i:>3d}] {str(base):>70s} shape={str(list(delta.shape)):>20s} norm={torch.norm(delta.float()).item():.4f}")

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
                    ckpt_sd, lora_deltas, return_stats=True
                )
                # Log match strategy distribution
                if match_stats:
                    print(f"\n   📊 Match strategy distribution ({sum(match_stats.values())} total):")
                    for strategy, count in sorted(match_stats.items(), key=lambda x: -x[1]):
                        print(f"      {strategy}: {count} keys")
            # Move matched delta tensors to target device (only if not already there)
            if matched_deltas:
                sample = next(iter(matched_deltas.values()))
                if sample.device != self.device:
                    matched_deltas = {k: v.to(device=self.device)
                                      for k, v in matched_deltas.items()}
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

            if baking_method == "impact_weighted":
                baked = bake_fn(
                    ckpt_sd, matched_deltas, strength, energy_concentration,
                    weight_unet, weight_te, weight_clip, weight_vae,
                    batch_size=batch_size,
                )
            else:
                baked = bake_fn(
                    ckpt_sd, matched_deltas, strength,
                    weight_unet, weight_te, weight_clip, weight_vae,
                    batch_size=batch_size,
                )

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
            # DELTA STATISTICS: Collect per-layer norms and ratios for forensic analysis
            # =====================================================================
            print("\n" + "-" * 50)
            print("📊 DELTA STATISTICS (aggregate across all baked keys):")
            delta_norms: List[float] = []
            delta_ratios: List[float] = []
            base_norms: List[float] = []
            for ckpt_key in baked:
                if ckpt_key not in ckpt_sd:
                    continue
                delta = matched_deltas.get(ckpt_key)
                if delta is None:
                    continue
                base = ckpt_sd[ckpt_key]
                try:
                    delta_adjusted = self._align_delta_shape(delta, base)
                except ValueError as e:
                    print(f"      ⚠️ Skipping delta stats for {ckpt_key}: {e}")
                    continue
                comp_weight = self._get_component_weight(
                    ckpt_key, weight_unet, weight_te, weight_clip, weight_vae
                )
                if comp_weight == 0.0:
                    continue
                dn = torch.norm(delta_adjusted.float()).item()
                bn = torch.norm(base.float()).item()
                # Skip NaN/Inf values — they corrupt statistics computation
                if not math.isfinite(dn) or not math.isfinite(bn):
                    continue
                delta_norms.append(dn)
                base_norms.append(bn)
                delta_ratios.append(dn / max(bn, 1e-12))

            comfyui_yield()

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

            # NaN/Inf sanitization
            # 🔥 REVERT-TO-ORIGINAL: Pass ckpt_sd so any key with persistent NaN/Inf
            # reverts to its original checkpoint weight instead of being replaced
            # with zeros (which cause black/dark patches in generated images).
            baked = self._sanitize_baked(baked, original_ckpt_sd=ckpt_sd)

            # Step 8: Assemble with VAE pass-through
            output_sd = self._assemble_output(ckpt_sd, baked)

            # Build impact profile
            impact_profile = {
                "baking_method": baking_method,
                "matched_keys": len(matched_deltas),
                "baked_keys": len(baked),
                "preserved_keys": len(ckpt_sd) - len(baked),
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
                len(baked), len(ckpt_sd) - len(baked),
                original_metadata,
                weight_unet, weight_te, weight_clip, weight_vae,
                metadata_mode,
            )

            # Build forensic report
            forensic_report = self._build_forensic_report(
                baking_method, strength,
                lora_source, checkpoint_name,
                len(baked), len(ckpt_sd) - len(baked),
                len(ckpt_sd), impact_profile,
                weight_unet=weight_unet, weight_te=weight_te,
                weight_clip=weight_clip, weight_vae=weight_vae,
                component_breakdown=getattr(self, '_component_breakdown', None),
                delta_analysis=getattr(self, '_delta_analysis', None),
            )

            print("=" * 50)
            print("✅ Baking pipeline complete!")
            print(f"   Baked {len(baked)} keys, preserved {len(ckpt_sd) - len(baked)} keys")
            print(f"   🧱 Weights — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}")
            print(f"   📋 Metadata mode: {metadata_mode}")
            print("=" * 50)
            pipeline_progress += 1  # Phase 5: Assembly + metadata done
            pipeline_progress.complete()

            return output_sd, metadata, forensic_report, impact_profile
