"""
Identity Merge Engine for Ariadne Project.
Performs merging using key mapping to preserve original trainer names.
"""
from typing import Optional, Dict, Tuple
import torch

try:
    from ..config import (
        MergeConfig,
        MergeMethodRegistry,
    )
except ImportError:
    from config import (
        MergeConfig,
        MergeMethodRegistry,
    )

try:
    from ..utils import (
        silent_pad_or_truncate,
        DeviceManager,
        get_tensor_energy,
        categorize_key,
        compute_component_energy_ratios,
        compute_primary_driver_intensity_metric,
        ProgressTracker,
    )
except ImportError:
    from utils import (
        silent_pad_or_truncate,
        DeviceManager,
        get_tensor_energy,
        categorize_key,
        compute_component_energy_ratios,
        compute_primary_driver_intensity_metric,
        ProgressTracker,
    )

try:
    from .klein_normalizer import (
        safe_get_rank,
    )
except ImportError:
    from klein_normalizer import (
        safe_get_rank,
    )

try:
    from .methods import (
        universal_merge_executor
    )
except ImportError:
    from methods import (
        universal_merge_executor
    )

try:
    from .identity_normalizer import identity_normalize
except ImportError:
    from identity_normalizer import identity_normalize

try:
    from .scale_utils import find_alpha_value, resolve_scale
except ImportError:
    from scale_utils import find_alpha_value, resolve_scale

class IdentityMergeEngine:
    """Performs merging with identity mapping for perfect key restoration."""
    
    def __init__(self, config: MergeConfig):
        self.config = config
        # Use pre-resolved values from DevicePrecisionConfig (set in MergeConfig.__post_init__)
        self.device = config.device_precision.device
        self.dtype = config.device_precision.dtype
        
        # Optimize device settings
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if torch.cuda.get_device_capability(self.device)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"[GPU] Device: {self.device}, Precision: {self.dtype}")
        # Magnitude scaling statistics
        self._scaling_factors = []  # list of scale factors applied
        self._energy_ratios = []    # list of (key, ratio) for diagnostic
        self._warning_keys = set()  # keys that triggered warning (optional)
        self._disable_rms_scaling = False  # flag to skip RMS scaling for Z-Image LoRAs
        self._z_image_detected = False     # flag to indicate Z-Image LoRA for safety clamp
        self.target_alpha = None    # global target alpha (1 or None for rank)
        self.alpha_one_is_rank = False  # treat alpha=1 as rank for scaling
        self.both_alpha_one = False  # both LoRAs have alpha=1 (Alpha Mirroring)
        self.max_rank = None         # uniform rank for mixed‑conversion merges
        self.mixed_conversion = False # flag indicating mixing converted/unconverted LoRAs
        # Per-key energy ratio variance tracking (for high-variance diagnostic)
        self._per_key_stats: Dict[str, Dict[str, float]] = {}  # {component: {key: stats}}
        # Warning for low precision dtypes
        if config.device_precision.precision in ("fp8", "fp8_e4m3fn", "fp8_e5m2", "float16"):
            print("ℹ️ Low precision selected. Merge math will be upcast to float32 for stability.")
    
    def _detect_converted_lora(self, sd, original_sd, metadata=None):
        """Detect if a LoRA is already converted (alpha baked in)."""
        # Copy from merge_engine.py (simplified)
        if metadata is not None:
            if metadata.get("alpha_scaled") == "True" or "easy_lora_merger_version" in metadata:
                print(f"   📊 Detected merged LoRA via metadata - treating as converted")
                return True

        # --- Anima/Musubi format detection ---
        # Musubi-trained LoRAs (including Anima) have keys like:
        #   lora_unet_blocks_N_*, lora_unet_double_blocks_*, lora_unet_single_blocks_*
        # These are ALWAYS unconverted if they have alpha keys, regardless of key count.
        # The key count heuristic below fails for Anima because:
        #   - Partial Anima LoRA (e.g. q/k/v only) = ~840 keys → falls through to fallback (converted)
        #   - Full Anima LoRA (all modules) = ~1344 keys → hits key_count > 1000 (unconverted)
        # Both are actually the same format; we must detect them consistently.
        is_musubi = any(
            'lora_unet_blocks_' in k or 'lora_unet_double_blocks_' in k or 'lora_unet_single_blocks_' in k
            for k in original_sd.keys()
        )
        if is_musubi:
            has_alpha = any('.alpha' in k or '.lora_alpha' in k for k in original_sd.keys())
            if has_alpha:
                print(f"   📊 Detected unconverted Musubi/Anima LoRA ({len(sd)} keys, has alphas)")
                return False
            else:
                print(f"   📊 Detected converted Musubi/Anima LoRA ({len(sd)} keys, no alpha keys)")
                return True

        has_alpha = any('.alpha' in k for k in original_sd.keys())
        key_count = len(sd)
        if has_alpha and key_count > 200:
            if (220 <= key_count <= 260) or (320 <= key_count <= 350) or (600 <= key_count <= 650):
                print(f"   📊 Detected unconverted LoRA ({key_count} keys, has alphas)")
                return False
            if key_count > 1000:
                print(f"   📊 Detected unconverted LoRA with high key count ({key_count} keys, has alphas)")
                return False
        if not has_alpha:
            print(f"   📊 Detected converted LoRA (no alpha keys)")
            return True
        print(f"   ℹ️ LoRA status uncertain ({key_count} keys, has_alphas={has_alpha}) - assuming converted")
        return True
    
    def _has_alpha_one(self, sd):
        """Return True if any alpha key in state dict equals 1."""
        for k, v in sd.items():
            if '.alpha' in k and v.numel() == 1 and abs(v.item() - 1.0) < 1e-6:
                return True
        return False

    def _adjust_dtype_for_inputs(self, sd_a, sd_b):
        """
        Adjust self.dtype based on input LoRA dtypes (hybrid auto logic).
        Only called when config.precision == "auto".
        """
        # Hardware‑best dtype (fallback)
        hardware_dtype = DeviceManager.get_dtype("auto", self.device)
        
        # Determine input dtypes
        def get_uniform_dtype(sd):
            dtypes = {v.dtype for v in sd.values() if isinstance(v, torch.Tensor)}
            if len(dtypes) == 1:
                return next(iter(dtypes))
            return None
        
        dtype_a = get_uniform_dtype(sd_a)
        dtype_b = get_uniform_dtype(sd_b)
        
        # If both LoRAs have same uniform dtype, use that (if supported)
        if dtype_a is not None and dtype_b is not None and dtype_a == dtype_b:
            target_dtype = dtype_a
        else:
            # Pick higher precision among present dtypes
            all_dtypes = set()
            if dtype_a is not None:
                all_dtypes.add(dtype_a)
            if dtype_b is not None:
                all_dtypes.add(dtype_b)
            # If mixed dtypes within a LoRA, we cannot infer; fallback to hardware_dtype
            if not all_dtypes:
                target_dtype = hardware_dtype
            else:
                # Order by precision (float32 > bfloat16 > float16)
                if torch.float32 in all_dtypes:
                    target_dtype = torch.float32
                elif torch.bfloat16 in all_dtypes:
                    target_dtype = torch.bfloat16
                elif torch.float16 in all_dtypes:
                    target_dtype = torch.float16
                else:
                    target_dtype = hardware_dtype
        
        # Ensure target dtype is supported by hardware (if not, fallback)
        if target_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            target_dtype = torch.float16
        # If device is CPU, prefer float32 over bfloat16 for compatibility
        if target_dtype == torch.bfloat16 and self.device.type == 'cpu':
            target_dtype = torch.float32
        # If target dtype differs from current self.dtype, update it
        if target_dtype != self.dtype:
            print(f"   🔄 Auto‑precision adjusted: {self.dtype} → {target_dtype}")
            self.dtype = target_dtype
    
    def _normalize_merged_dtype(self, merged):
        """
        Ensure all tensors in merged dict have uniform dtype (self.dtype).
        Only converts tensors that differ from target dtype.
        Respects config.unify_dtype_to_engine flag (if False, skip).
        """
        if not merged:
            return
        if not self.config.unify_dtype_to_engine:
            return
        target_dtype = self.dtype
        # Check if any tensor has different dtype
        mismatched = [k for k, v in merged.items() if v.dtype != target_dtype]
        if mismatched:
            print(f"   🔧 Normalizing dtype of {len(mismatched)} tensors to {target_dtype}")
            for k in mismatched:
                merged[k] = merged[k].to(dtype=target_dtype)

    def _apply_lora_scaling(self, tensor, original_sd, key, mapping=None, is_converted=False):
        """Apply alpha/rank scaling correction to LoRA tensors using mapping."""
        import re
        
        # Down weights should not be scaled; scaling is applied only to up weights
        if any(suffix in key for suffix in ['.lora_A.weight', '.lora_down.weight']):
            return tensor
        
        rank = safe_get_rank(tensor, key)
        rank = max(1, rank)
        
        # Find alpha value with engine-state flags (alpha_one_is_rank, target_alpha)
        alpha_value = find_alpha_value(
            original_sd, key,
            mapping=mapping,
            alpha_one_is_rank=self.alpha_one_is_rank,
            rank=rank,
        )
        
        # Determine base scaling factor (alpha / rank) if not converted
        scale_factor = 1.0
        if not is_converted and alpha_value is not None:
            scale_factor = resolve_scale(alpha_value, rank, mode="linear")
        
        # Apply extra scaling for target alpha if needed
        extra_factor = 1.0
        if self.target_alpha is not None:
            if is_converted or alpha_value is None:
                # Converted LoRA: assume original alpha = rank
                extra_factor = self.target_alpha / rank
            else:
                # Unconverted LoRA with known alpha
                extra_factor = self.target_alpha / alpha_value
        
        total_scaling = scale_factor * extra_factor
        
        # Apply scaling
        if abs(total_scaling - 1.0) > 1e-6:
            return tensor * total_scaling
        return tensor

    def _apply_rms_scaling(self, tensor_a, tensor_b, key):
        """
        Apply magnitude scaling (RMS or percentile) to equalize signal magnitude.
        Only scales tensor_b to match tensor_a's energy.
        Includes safety valves: silence clamp, clipping warning, and max scaling clamp.
        """
        mode = self.config.magnitude_scaling
        if self._disable_rms_scaling:
            return tensor_a, tensor_b
        if mode == "none":
            # Diagnostic: compute energy ratio for imbalance detection
            energy_a = get_tensor_energy(tensor_a, "rms")
            energy_b = get_tensor_energy(tensor_b, "rms")
            epsilon = 1e-8
            ratio = energy_a / (energy_b + epsilon)
            self._energy_ratios.append((key, ratio))
            if ratio > 2.0 or ratio < 0.5:
                print(f"   ⚖️ [{key}] Magnitude imbalance (A/B): {ratio:.2f}x (A={energy_a:.2e}, B={energy_b:.2e})")
            return tensor_a, tensor_b

        epsilon = 1e-8
        silence_threshold = 1e-6
        warning_threshold = 5.0
        max_scaling_factor = self.config.max_scaling_factor
        if self._z_image_detected:
            max_scaling_factor = self.config.z_image_max_scaling_factor  # safety clamp for Z-Image merges

        # Compute energy for each tensor (original shape before rank adjustment)
        energy_a = get_tensor_energy(tensor_a, mode)
        energy_b = get_tensor_energy(tensor_b, mode)

        # Silence clamp: if energy_B is extremely low, skip scaling (avoid amplifying dust)
        if energy_b < silence_threshold:
            print(f"   🔇 [{key}] Energy_B ({energy_b:.2e}) below silence threshold; skipping scaling")
            return tensor_a, tensor_b

        # Compute scaling factor
        raw_scale = energy_a / (energy_b + epsilon)
        scale = raw_scale

        # Clamping to max scaling factor
        clamped = False
        if raw_scale > max_scaling_factor:
            scale = max_scaling_factor
            clamped = True
            print(f"   🔧 [{key}] Gain clamped to {max_scaling_factor:.1f}x (Original requirement: {raw_scale:.1f}x)")

        # Clipping warning: if scaling factor is very large, warn user (based on raw scale)
        if raw_scale > warning_threshold:
            self._warning_keys.add(key)
            print(f"   ⚠️ [{key}] LoRA B is significantly quieter than A (energy_B={energy_b:.2e}, energy_A={energy_a:.2e}, scale={raw_scale:.1f}x)")
            print(f"      Note: Strong gain may cause saturation; consider adjusting weights.")

        # Apply scaling only to tensor_b (reference matching to LoRA A)
        tensor_b = tensor_b * scale

        # Log scaling if factor > 1.01 or < 0.99 (use clamped scale for statistics)
        if abs(scale - 1.0) > 0.01:
            self._scaling_factors.append(scale.item() if torch.is_tensor(scale) else scale)

        return tensor_a, tensor_b

    def _compute_global_energy_ratio(self, norm_a, norm_b, common_keys,
                                     sd_a, sd_b, map_a, map_b,
                                     sd_a_converted, sd_b_converted):
        """
        Compute the per-component energy ratio between LoRA A and B across shared keys.

        Uses per-element mean energy (rank-independent) on common (shared) layers only.
        This prevents architectural mismatches (unique keys present in only one LoRA)
        from affecting the volume of the layers they both have.

        Returns a dict mapping component ('model', 'te') to magnitude ratio sqrt(e_a / e_b).
        """
        # Use the shared utility for rank-independent mean energy computation
        import math
        from collections import defaultdict
        energy_by_component = compute_component_energy_ratios(
            norm_sds=[norm_a, norm_b],
            common_keys=list(common_keys),
            original_sds=[sd_a, sd_b],
            mappings=[map_a, map_b],
            converted_flags=[sd_a_converted, sd_b_converted],
            key_categorizer=categorize_key,
        )

        # Compute per-key ratio distribution per component for diagnostic logging
        per_key_by_component = defaultdict(list)
        for key in common_keys:
            t_a = norm_a[key]
            t_b = norm_b[key]
            e_a = torch.mean(t_a ** 2).item()
            e_b = torch.mean(t_b ** 2).item()
            if e_b > 1e-12:
                ratio = math.sqrt(e_a / e_b)
                component = categorize_key(key)
                per_key_by_component[component].append(ratio)

        for component, ratios in per_key_by_component.items():
            if len(ratios) > 1:
                min_r = min(ratios)
                max_r = max(ratios)
                mean_r = sum(ratios) / len(ratios)
                variance = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)
                stddev = math.sqrt(variance)
                print(f"   [Auto‑Balance] [{component}] Per‑key ratio distribution: "
                      f"min={min_r:.2f}x, max={max_r:.2f}x, σ={stddev:.2f}")
                # Store per-key stats for high-variance diagnostic
                self._per_key_stats[component] = {
                    "min": min_r,
                    "max": max_r,
                    "mean": mean_r,
                    "stddev": stddev,
                    "count": len(ratios)
                }
            else:
                self._per_key_stats[component] = {
                    "stddev": 0.0,
                    "count": len(ratios)
                }
        epsilon = 1e-12
        ratios = {}
        for component, energies in energy_by_component.items():
            e_a = energies[0]
            e_b = energies[1]
            if e_b < epsilon:
                ratios[component] = 1.0
            else:
                ratios[component] = math.sqrt(e_a / e_b)

        return ratios

    def _get_component_weights(self, key):
        """
        Return weight_a, weight_b for a given key based on its component.
        """
        component = categorize_key(key)
        weight_a = self._component_weight_a.get(component, self._weight_a_adj)
        weight_b = self._component_weight_b.get(component, self._weight_b_adj)
        return weight_a, weight_b

    def _print_magnitude_scaling_summary(self):
        """Print aggregate statistics for magnitude scaling."""
        mode = self.config.magnitude_scaling
        
        # Print magnitude scaling summary if factors exist
        if self._scaling_factors:
            factors = self._scaling_factors
            count = len(factors)
            avg = sum(factors) / count
            minv = min(factors)
            maxv = max(factors)
            print(f"   📏 Magnitude Equalization ({mode}) applied to {count} layers. "
                  f"Average scale: {avg:.2f}x (Min: {minv:.1f}x, Max: {maxv:.1f}x).")
        
        # Print energy imbalance diagnostic when magnitude scaling is "none"
        if mode == "none" and self._energy_ratios:
            ratios = [r for _, r in self._energy_ratios]
            count = len(ratios)
            avg = sum(ratios) / count
            minv = min(ratios)
            maxv = max(ratios)
            # Count layers with significant imbalance
            severe = sum(1 for r in ratios if r > 2.0 or r < 0.5)
            if severe > 0:
                print(f"   ⚖️ Magnitude imbalance detected (no scaling). "
                      f"Average A/B ratio: {avg:.2f}x (Min: {minv:.1f}x, Max: {maxv:.1f}x). "
                      f"{severe} layers exceed 2:1 imbalance.")

    def _generate_unified_alpha_keys(self, merged, master_map):
        """
        Generate new alpha keys where alpha = target_alpha (1) if set, otherwise rank.
        Removes any existing .alpha keys from merged dict first.
        Maps normalized alpha key to original alpha key using master_map of weight keys.
        """
        
        # Remove any existing alpha keys
        alpha_keys = [k for k in merged.keys() if ".alpha" in k]
        for k in alpha_keys:
            del merged[k]
            # Also remove from master_map if present
            master_map.pop(k, None)
        
        alpha_count = 0
        processed_bases = set()
        for norm_key in list(merged.keys()):
            if ".lora_A.weight" in norm_key or ".lora_down.weight" in norm_key:
                # Extract base name (without .lora_A.weight or .lora_down.weight)
                base = norm_key.replace(".lora_A.weight", "").replace(".lora_down.weight", "")
                if base in processed_bases:
                    continue
                processed_bases.add(base)
                
                tensor = merged[norm_key]
                rank = safe_get_rank(tensor, norm_key)
                rank = max(1, rank)
                
                # Determine alpha value
                if self.both_alpha_one:
                    alpha_val = 1.0
                elif self.target_alpha is not None:
                    alpha_val = float(self.target_alpha)
                else:
                    alpha_val = float(rank)
                
                # Determine original weight key from master_map (fallback to norm_key)
                orig_weight_key = master_map.get(norm_key, norm_key)
                # Convert original weight key to original alpha key
                # Replace .lora_down.weight or .lora_up.weight with .alpha
                # Also handle .lora_A.weight and .lora_B.weight (though original keys use down/up)
                orig_alpha_key = orig_weight_key
                for suffix in [".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight"]:
                    if suffix in orig_alpha_key:
                        orig_alpha_key = orig_alpha_key.replace(suffix, ".alpha")
                        break
                # If no suffix replaced, append .alpha (should not happen)
                if orig_alpha_key == orig_weight_key:
                    orig_alpha_key = orig_alpha_key.rsplit(".", 1)[0] + ".alpha"
                
                # Create normalized alpha key (same as before)
                alpha_key = f"{base}.alpha"
                alpha_tensor = torch.tensor(alpha_val, dtype=self.dtype, device=self.device)
                merged[alpha_key] = alpha_tensor
                alpha_count += 1
                # Map normalized alpha key to original alpha key
                master_map[alpha_key] = orig_alpha_key
        if alpha_count > 0:
            if self.both_alpha_one:
                print(f"   🔧 Generated {alpha_count} unified alpha keys (alpha = 1)")
            elif self.target_alpha is not None:
                print(f"   🔧 Generated {alpha_count} unified alpha keys (alpha = {self.target_alpha})")
            else:
                print(f"   🔧 Generated {alpha_count} unified alpha keys (alpha = rank)")
        
    def _resolve_blend_mode(self, blend_mode: str, meta_a: Optional[Dict[str, str]], meta_b: Optional[Dict[str, str]]) -> str:
        """
        Translate UI blend_mode to the internal value used by merge methods.
        
        Mapping:
        - "dense" -> "dense" (fall‑back weighted sum)
        - "auto" -> "active" if trainers mismatch, "dense" if they match.
        
        Trainers are considered matching if ss_network_module or ss_base_model are present and equal.
        """
        if blend_mode == "dense":
            return "dense"
        
        if blend_mode == "auto":
            # Determine if trainers match
            trainers_match = False
            if meta_a and meta_b:
                # Check ss_network_module
                nm_a = meta_a.get("ss_network_module")
                nm_b = meta_b.get("ss_network_module")
                if nm_a is not None and nm_b is not None and nm_a == nm_b:
                    trainers_match = True
                # Check ss_base_model
                bm_a = meta_a.get("ss_base_model")
                bm_b = meta_b.get("ss_base_model")
                if not trainers_match and bm_a is not None and bm_b is not None and bm_a == bm_b:
                    trainers_match = True
            
            if trainers_match:
                return "dense"
            else:
                return "active"
        
        # Pass through any unrecognised value (should not happen with UI)
        return blend_mode

    @staticmethod
    def _sanitize_tensor(tensor: torch.Tensor, key: str, label: str = "EXPLOSION") -> torch.Tensor:
        """NaN/Inf interceptor: detect and sanitize exploding tensors."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"[*] {label} in layer: {key}")
            return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        return tensor

    def _merge_unique_keys(self, keys, norm_sd, sd, map_sd, converted_flag, weight_getter, progress_desc, merged, master_map):
        """Parameterized merge for unique keys from one LoRA (A or B)."""
        if not keys:
            return
        with ProgressTracker(total=len(keys), desc=progress_desc) as progress:
            for key in keys:
                weight = weight_getter(key)
                tensor = norm_sd[key].to(device=self.device)
                tensor = self._apply_lora_scaling(tensor, sd, key, mapping=map_sd, is_converted=converted_flag)
                tensor = tensor * weight
                tensor = self._sanitize_tensor(tensor, key, "NaN/Inf detected in unique layer")
                merged[key] = tensor
                if key in map_sd:
                    master_map[key] = map_sd[key]
                else:
                    master_map[key] = key
                progress += 1

    def merge_with_mapping(self, sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor],
                          meta_a: Optional[Dict[str, str]] = None,
                          meta_b: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
        """
        Merge two LoRA state dicts while preserving key mapping.
        
        Returns:
            merged_dict: State dict with normalized keys (mathematical keys).
            master_map: Mapping from normalized key -> original key (preferring LoRA A's naming).
        """
        print("🔍 Starting identity merge with mapping...")
        
        # Adjust dtype based on input LoRAs if precision is auto
        if self.config.device_precision.precision == "auto":
            self._adjust_dtype_for_inputs(sd_a, sd_b)
        
        # Detect Z-Image architecture and apply safety clamp to prevent saturation
        if meta_a and "modelspec.architecture" in meta_a and "Z-Image" in meta_a["modelspec.architecture"]:
            self._z_image_detected = True
        if meta_b and "modelspec.architecture" in meta_b and "Z-Image" in meta_b["modelspec.architecture"]:
            self._z_image_detected = True
        if self._z_image_detected:
            print(f"   🎯 Z-Image LoRA detected – applying safety clamp (max scaling factor {self.config.z_image_max_scaling_factor:.1f}x)")
        
        # Detect alpha=1 in either LoRA
        both_alpha_one = self._has_alpha_one(sd_a) and self._has_alpha_one(sd_b)
        if self._has_alpha_one(sd_a) or self._has_alpha_one(sd_b):
            self.alpha_one_is_rank = True
            self.target_alpha = None   # treat alpha=1 as rank, output alpha = rank
            print(f"   🎯 Global Alpha=1 detected, treating alpha as rank (output alpha = rank)")
            if both_alpha_one:
                self.both_alpha_one = True
                print(f"   🔄 Both LoRAs have Alpha=1 – Alpha Mirroring enabled (output alpha = 1)")
        else:
            self.target_alpha = None   # use rank per layer
            self.alpha_one_is_rank = False
        self.both_alpha_one = both_alpha_one
        
        # Step 1: Identity normalization with mapping
        # SD1.5 Diffusers LoRA detection: bridge preserves block structure (avoids ~175/396 key loss)
        from .diffusers_bridge import detect_diffusers_sd15_lora, normalize_diffusers_preserving
        
        print("🔄 Normalizing LoRA A with mapping...")
        if detect_diffusers_sd15_lora(sd_a):
            print("   🔄 SD1.5 Diffusers LoRA A detected – using Diffusers-preserving normalization")
            norm_a, map_a = normalize_diffusers_preserving(sd_a)
        else:
            norm_a, map_a = identity_normalize(sd_a, meta_a)

        print("🔄 Normalizing LoRA B with mapping...")
        if detect_diffusers_sd15_lora(sd_b):
            print("   🔄 SD1.5 Diffusers LoRA B detected – using Diffusers-preserving normalization")
            norm_b, map_b = normalize_diffusers_preserving(sd_b)
        else:
            norm_b, map_b = identity_normalize(sd_b, meta_b)
        
        # Step 2: Detect conversion status
        sd_a_converted = self._detect_converted_lora(norm_a, sd_a, meta_a)
        sd_b_converted = self._detect_converted_lora(norm_b, sd_b, meta_b)
        print(f"📊 LoRA A converted: {sd_a_converted}, LoRA B converted: {sd_b_converted}")
        
        # If mixing converted and unconverted, generate virtual alpha for consistency
        self.mixed_conversion = sd_a_converted != sd_b_converted
        if self.mixed_conversion:
            print(f"   ℹ️ Mixing converted/unconverted LoRAs - using virtual alpha (alpha = rank) for scaling consistency")
            # Compute max rank across all down weight tensors (for uniform alpha)
            all_down_keys = []
            for k in norm_a.keys():
                if '.lora_A.weight' in k or '.lora_down.weight' in k:
                    all_down_keys.append((k, norm_a))
            for k in norm_b.keys():
                if '.lora_A.weight' in k or '.lora_down.weight' in k:
                    all_down_keys.append((k, norm_b))
            max_rank = 0
            for k, sd in all_down_keys:
                tensor = sd[k]
                if len(tensor.shape) >= 2:
                    rank = min(tensor.shape)
                else:
                    rank = 1
                max_rank = max(max_rank, rank)
            self.max_rank = max_rank if max_rank > 0 else None
            print(f"   📊 Mixed conversion detected – uniform rank = {self.max_rank}")
        
        # Step 3: Find common keys
        keys_a = set(norm_a.keys())
        keys_b = set(norm_b.keys())
        common_keys = keys_a & keys_b
        unique_a = keys_a - keys_b
        unique_b = keys_b - keys_a
        
        # Filter out alpha keys (they will be replaced with unified alpha later)
        def is_alpha_key(k):
            return ".alpha" in k
        
        alpha_common = {k for k in common_keys if is_alpha_key(k)}
        common_keys = common_keys - alpha_common
        alpha_unique_a = {k for k in unique_a if is_alpha_key(k)}
        unique_a = unique_a - alpha_unique_a
        alpha_unique_b = {k for k in unique_b if is_alpha_key(k)}
        unique_b = unique_b - alpha_unique_b
        
        print(f"🧩 Found {len(common_keys)} common layers to merge (filtered out {len(alpha_common)} alpha keys)")
        print(f"📝 Unique to A: {len(unique_a)} (filtered out {len(alpha_unique_a)} alpha keys), Unique to B: {len(unique_b)} (filtered out {len(alpha_unique_b)} alpha keys)")
        
        # Auto-weight-balancing (per-component energy ratio)
        self._component_weight_a = {}
        self._component_weight_b = {}
        
        # Determine all components present across all keys (common + unique)
        all_keys = set(norm_a.keys()) | set(norm_b.keys())
        all_components = set()
        for key in all_keys:
            comp = categorize_key(key)
            all_components.add(comp)
        
        # Log component detection
        te_components = [c for c in all_components if c == 'te']
        if te_components:
            print(f"   🔍 Detected Text Encoder (TE) keys in component: {te_components}")
        else:
            print(f"   🔍 No Text Encoder (TE) keys detected; all keys belong to model component.")
        
        # Default weights (no balancing)
        for comp in all_components:
            self._component_weight_a[comp] = self.config.weight_a
            self._component_weight_b[comp] = self.config.weight_b
        
        # Apply balancing per component if there are common keys and balancing enabled
        if common_keys and self.config.balancing_mode != "disabled":
            if self.config.balancing_mode == "intensity":
                # ===== Intensity Mode v2: Adaptive Primary Driver Detection with Joint Set =====
                # Uses energy-concentration threshold (80%) per LoRA to find "Primary Driver" keys,
                # then compares both LoRAs on the JOINT union set for a fair apples-to-apples comparison.
                # This avoids the "apples to oranges" bias of the old top-20% approach which compared
                # different key sets and could detect the wrong LoRA as stronger.
                primary_metrics = compute_primary_driver_intensity_metric(
                    norm_sds=[norm_a, norm_b],
                    common_keys=list(common_keys),
                    energy_concentration=0.80,
                    original_sds=[sd_a, sd_b],
                    mappings=[map_a, map_b],
                    converted_flags=[sd_a_converted, sd_b_converted],
                    key_categorizer=categorize_key,
                )
                for comp, metrics in primary_metrics.items():
                    peak_a, peak_b = metrics['peaks'][0], metrics['peaks'][1]
                    joint_a, joint_b = metrics['joint_peaks'][0], metrics['joint_peaks'][1]
                    count_a, count_b = metrics['primary_driver_counts'][0], metrics['primary_driver_counts'][1]
                    joint_count = metrics['joint_count']
                    conc_a, conc_b = metrics['energy_concentration'][0], metrics['energy_concentration'][1]
                    overlap = metrics['overlap_count']

                    # Use JOINT ratio for scaling (fair apples-to-apples comparison)
                    intensity_ratio = metrics['ratio_joint']

                    # Diagnostic: sparsity profile
                    total_keys = len(common_keys)
                    sparsity_a = count_a / max(1, total_keys) * 100
                    sparsity_b = count_b / max(1, total_keys) * 100

                    print(f"   [Intensity‑Balance] [{comp}] Primary driver analysis:")
                    print(f"     A: {count_a}/{total_keys} keys ({sparsity_a:.1f}%) = {conc_a*100:.0f}% of energy")
                    print(f"     B: {count_b}/{total_keys} keys ({sparsity_b:.1f}%) = {conc_b*100:.0f}% of energy")
                    print(f"     Joint set: {joint_count} keys (overlap: {overlap}) — fair comparison")

                    # Apply global scaling using joint ratio
                    if intensity_ratio > 1.0:
                        self._component_weight_a[comp] = self.config.weight_a / intensity_ratio
                        self._component_weight_b[comp] = self.config.weight_b
                        print(f"   [Intensity‑Balance] [{comp}] LoRA A is {intensity_ratio:.2f}x stronger on joint primary drivers.")
                    else:
                        self._component_weight_a[comp] = self.config.weight_a
                        self._component_weight_b[comp] = self.config.weight_b * intensity_ratio
                        print(f"   [Intensity‑Balance] [{comp}] LoRA B is {1.0/intensity_ratio:.2f}x stronger on joint primary drivers.")

                    print(f"   [Intensity‑Balance] [{comp}] Joint peaks — A: {joint_a:.2e}, B: {joint_b:.2e}")
                    print(f"   [Intensity‑Balance] [{comp}] Global scaling: internal weights {self._component_weight_a[comp]:.2f} : {self._component_weight_b[comp]:.2f}")
            elif self.config.balancing_mode == "impact":
                # ===== Impact Mode v3: Intensity + Sparsity Correction =====
                # Extends Intensity v2 by detecting when one LoRA is significantly
                # more "concentrated" (fewer primary drivers for the same energy).
                # Applies a correction to prevent sparse character LoRAs from
                # visually dominating dense style LoRAs.
                #
                # Key formula: adjusted_ratio = intensity_ratio / density_ratio
                #   density_ratio = count_A / count_B
                #   (>1 means A is denser, B is more concentrated -> penalize B)
                #   (<1 means B is denser, A is more concentrated -> penalize A)
                primary_metrics = compute_primary_driver_intensity_metric(
                    norm_sds=[norm_a, norm_b],
                    common_keys=list(common_keys),
                    energy_concentration=0.80,
                    original_sds=[sd_a, sd_b],
                    mappings=[map_a, map_b],
                    converted_flags=[sd_a_converted, sd_b_converted],
                    key_categorizer=categorize_key,
                )
                for comp, metrics in primary_metrics.items():
                    peak_a, peak_b = metrics['peaks'][0], metrics['peaks'][1]
                    joint_a, joint_b = metrics['joint_peaks'][0], metrics['joint_peaks'][1]
                    count_a, count_b = metrics['primary_driver_counts'][0], metrics['primary_driver_counts'][1]
                    joint_count = metrics['joint_count']
                    conc_a, conc_b = metrics['energy_concentration'][0], metrics['energy_concentration'][1]
                    overlap = metrics['overlap_count']

                    # Base intensity ratio (same as Intensity mode)
                    intensity_ratio = metrics['ratio_joint']

                    # Diagnostic: sparsity profile
                    total_keys = len(common_keys)
                    sparsity_a = count_a / max(1, total_keys) * 100
                    sparsity_b = count_b / max(1, total_keys) * 100

                    # === Sparsity Correction ===
                    # density_ratio > 1: A has MORE primary drivers -> A is DENSER, B is more CONCENTRATED
                    # density_ratio < 1: B has MORE primary drivers -> B is DENSER, A is more CONCENTRATED
                    density_ratio = count_a / max(1, count_b)

                    # Divide intensity_ratio by density_ratio to penalize the concentrated LoRA
                    # This flips the correction toward reducing the sparse one
                    adjusted_ratio = intensity_ratio / density_ratio

                    print(f"   [Impact‑Balance] [{comp}] Primary driver analysis:")
                    print(f"     A: {count_a}/{total_keys} keys ({sparsity_a:.1f}%) = {conc_a*100:.0f}% of energy")
                    print(f"     B: {count_b}/{total_keys} keys ({sparsity_b:.1f}%) = {conc_b*100:.0f}% of energy")
                    print(f"     Joint set: {joint_count} keys (overlap: {overlap}) — fair comparison")
                    print(f"     Density ratio: {density_ratio:.2f}x (B is {max(density_ratio, 1/density_ratio):.2f}x {'more' if density_ratio > 1 else 'less'} concentrated)")
                    print(f"     Intensity ratio (raw): {intensity_ratio:.2f}x, Sparsity‑corrected: {adjusted_ratio:.2f}x")

                    # Apply global scaling using sparsity-corrected ratio
                    if adjusted_ratio > 1.0:
                        self._component_weight_a[comp] = self.config.weight_a / adjusted_ratio
                        self._component_weight_b[comp] = self.config.weight_b
                        print(f"   [Impact‑Balance] [{comp}] LoRA A is {adjusted_ratio:.2f}x stronger (sparsity‑corrected) — reducing A.")
                    else:
                        self._component_weight_a[comp] = self.config.weight_a
                        self._component_weight_b[comp] = self.config.weight_b * adjusted_ratio
                        print(f"   [Impact‑Balance] [{comp}] LoRA B is {1.0/adjusted_ratio:.2f}x stronger (sparsity‑corrected) — reducing B.")

                    print(f"   [Impact‑Balance] [{comp}] Joint peaks — A: {joint_a:.2e}, B: {joint_b:.2e}")
                    print(f"   [Impact‑Balance] [{comp}] Global scaling: internal weights {self._component_weight_a[comp]:.2f} : {self._component_weight_b[comp]:.2f}")
            else:
                # Existing safe/creative modes: use global mean energy ratio
                component_ratios = self._compute_global_energy_ratio(
                    norm_a, norm_b, common_keys,
                    sd_a, sd_b, map_a, map_b,
                    sd_a_converted, sd_b_converted
                )
                # component_ratios is dict component -> magnitude_ratio
                for comp, ratio in component_ratios.items():
                    if self.config.balancing_mode == "safe":
                        # Safe mode: adjust only the louder side to match energy
                        if ratio > 1.0:
                            self._component_weight_a[comp] = self.config.weight_a / ratio
                            self._component_weight_b[comp] = self.config.weight_b
                            print(f"   [Auto‑Balance] [{comp}] LoRA A energy is {ratio:.2f}x higher than B.")
                        else:
                            self._component_weight_a[comp] = self.config.weight_a
                            self._component_weight_b[comp] = self.config.weight_b * ratio
                            print(f"   [Auto‑Balance] [{comp}] LoRA B energy is {1.0/ratio:.2f}x higher than A.")
                        print(f"   [Auto‑Balance] [{comp}] Safe mode: adjusting internal weights to {self._component_weight_a[comp]:.2f} : {self._component_weight_b[comp]:.2f} to prevent noise.")
                    elif self.config.balancing_mode == "creative":
                        # Creative mode: uniform scaling preserving user ratio, 50% compromise
                        if ratio > 1.0:
                            safe_factor = 1.0 / ratio
                        else:
                            safe_factor = ratio
                        creative_factor = (1.0 + safe_factor) / 2.0  # 50% compromise
                        self._component_weight_a[comp] = self.config.weight_a * creative_factor
                        self._component_weight_b[comp] = self.config.weight_b * creative_factor
                        print(f"   [Auto‑Balance] [{comp}] LoRA A energy is {ratio:.2f}x {'higher' if ratio > 1.0 else 'lower'} than B.")
                        print(f"   [Auto‑Balance] [{comp}] Creative mode: uniform scaling factor {creative_factor:.2f}, adjusted weights {self._component_weight_a[comp]:.2f} : {self._component_weight_b[comp]:.2f}")
                    else:
                        # Should not happen (disabled already handled)
                        self._component_weight_a[comp] = self.config.weight_a
                        self._component_weight_b[comp] = self.config.weight_b
        
        # Enhanced diagnostic: check for high per-key variance (cross-concept merge)
        # (Only for safe/creative modes — intensity/impact modes use their own detection)
        if common_keys and self.config.balancing_mode not in ("disabled", "intensity", "impact"):
            high_variance_components = []
            for comp in self._per_key_stats:
                stats = self._per_key_stats[comp]
                stddev = stats.get("stddev", 0.0)
                if stddev > 0.20:
                    high_variance_components.append((comp, stddev, stats.get("min", 0), stats.get("max", 0)))
            if high_variance_components:
                print(f"   [Auto‑Balance] ⚠️ High per‑key ratio variance detected – likely a cross‑concept merge.")
                for comp, stddev, min_r, max_r in high_variance_components:
                    print(f"      [{comp}] σ={stddev:.2f}, per‑key range [{min_r:.2f}x – {max_r:.2f}x]")
                print(f"      ℹ️ The aggregate energy ratio may not reflect the per‑layer differences.")
                print(f"      💡 Consider using blend_mode='balanced' for per‑tensor interpolation")
                print(f"         if the result needs more equal concept representation.")
        
        # Keep global weight variables for compatibility (set to model component)
        model_weight_a = self._component_weight_a.get('model', self.config.weight_a)
        model_weight_b = self._component_weight_b.get('model', self.config.weight_b)
        self._weight_a_adj = model_weight_a
        self._weight_b_adj = model_weight_b

        # Disable per‑tensor magnitude scaling when auto‑weight‑balancing is active
        self._disable_rms_scaling = self.config.balancing_mode != "disabled"

        # Step 4: Prepare merged dict and master map
        merged = {}
        master_map = {}
        
        # Step 5: Merge common keys
        if common_keys:
            print("⚙️ Merging common layers...")
            # Choose merge method
            method = MergeMethodRegistry.get_method(self.config.method)
            # Resolve blend mode based on UI choice and metadata
            resolved_blend_mode = self._resolve_blend_mode(self.config.blend_mode, meta_a, meta_b)
            if resolved_blend_mode != self.config.blend_mode:
                print(f"   🔄 blend_mode '{self.config.blend_mode}' -> '{resolved_blend_mode}'")
            with ProgressTracker(total=len(common_keys), desc="Merging common keys") as merge_progress:
                for key in common_keys:
                    weight_a, weight_b = self._get_component_weights(key)
                    tensor_a = norm_a[key].to(device=self.device)
                    tensor_b = norm_b[key].to(device=self.device)
                    
                    # Apply scaling (alpha/rank)
                    tensor_a = self._apply_lora_scaling(tensor_a, sd_a, key, mapping=map_a, is_converted=sd_a_converted)
                    tensor_b = self._apply_lora_scaling(tensor_b, sd_b, key, mapping=map_b, is_converted=sd_b_converted)

                    # Rank adjustment (pad to max rank)
                    rank_a = safe_get_rank(tensor_a, key)
                    rank_b = safe_get_rank(tensor_b, key)
                    target_rank = max(rank_a, rank_b)
                    if rank_a != rank_b:
                        tensor_a = silent_pad_or_truncate(tensor_a, target_rank, key)
                        tensor_b = silent_pad_or_truncate(tensor_b, target_rank, key)

                    # Signal magnitude equalization (RMS scaling) if enabled
                    tensor_a, tensor_b = self._apply_rms_scaling(tensor_a, tensor_b, key)
                    
                    # Merge using universal_merge_executor
                    merged_tensor = universal_merge_executor(
                        method, tensor_a, tensor_b, weight_a, weight_b,
                        device=self.device,
                        blend_mode=resolved_blend_mode,
                        uniqueness=self.config.uniqueness,
                        threshold=self.config.threshold,
                        blend=self.config.blend,
                        active_threshold=self.config.active_threshold
                    )
                    merged_tensor = self._sanitize_tensor(merged_tensor, key, "EXPLOSION DETECTED in layer")
                    merged[key] = merged_tensor
                    merge_progress += 1
                    # Decide which original key to use (prefer A to preserve its format)
                    if key in map_a:
                        master_map[key] = map_a[key]
                    elif key in map_b:
                        master_map[key] = map_b[key]
                    else:
                        master_map[key] = key
        
        # Step 6: Add unique keys from A (weighted sum, skip gradient alignment)
        self._merge_unique_keys(
            unique_a, norm_a, sd_a, map_a, sd_a_converted,
            lambda k: self._get_component_weights(k)[0],
            "Adding unique A keys", merged, master_map
        )

        # Step 7: Add unique keys from B (weighted sum, skip gradient alignment)
        self._merge_unique_keys(
            unique_b, norm_b, sd_b, map_b, sd_b_converted,
            lambda k: self._get_component_weights(k)[1],
            "Adding unique B keys", merged, master_map
        )
        
        print(f"✅ Merge completed. Total merged keys: {len(merged)}")
        
        # Step 8: Generate unified alpha keys (alpha = rank)
        self._generate_unified_alpha_keys(merged, master_map)
        
        # Step 9: Ensure uniform dtype across all tensors
        self._normalize_merged_dtype(merged)
        
        self._print_magnitude_scaling_summary()
        
        return merged, master_map