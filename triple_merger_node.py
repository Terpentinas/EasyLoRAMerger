"""
Easy LoRA Merger Node (with Smart‑Pro Defaults)
Merges three LoRAs using advanced triple‑merge methods and auto‑formats output.
"""

import torch
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import folder_paths
import comfy.sd
import comfy.utils
from .config import MergeConfig, PRECISION_STANDARD, DEVICE_OPTIONS, DevicePrecisionConfig, ACTIVE_THRESHOLD_DEFAULT
from .utils import (
    load_lora_with_metadata,
    save_safetensors_file,
    get_user_output_path,
    get_experiment_temp_path,
    categorize_key,
    NodeCache,
)
from .engine.triple_forensics import build_triple_forensic_report
from .engine.klein_normalizer import (
    universal_normalize,
    detect_lora_format,
    infer_separator_style,
    infer_naming_style,
)
from .engine.identity_normalizer import identity_normalize
from .engine.serialization_factory import finalize_for_save
from .engine.triple_methods import merge_triple_method
from .engine.methods import resolve_blend_mode_triple
from .validation import MetadataMerger
from .engine.lora_studio_converter import MusubiLoraConverter
from .engine.metadata_factory import finalize_metadata
from .engine.merge_engine_v2 import IdentityMergeEngine


class EasyLoRATripleMerger:
    """Merge THREE LoRAs at once and apply to model (experimental)"""
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        default_folder = ""
        lora_folders = folder_paths.get_folder_paths("loras")
        if lora_folders:
            default_folder = str(lora_folders[0])
        
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "method": (["linear", "ties_strict", "ties_gentle", "dare_lite",
                           "dare_rescale", "subtract", "magnitude", "feature_mix",
                           "svd_preserve", "noise_aware", "gradient_alignment",
                           "slerp", "cross", "ties_contrast", "block_swap"], {
                    "default": "linear"
                }),
                "density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # ── SECTION 2: SOURCES ──────────────────────────────────
                "lora_a": (["None"] + loras,),
                "lora_b": (["None"] + loras,),
                "lora_c": (["None"] + loras,),
                "lora_data_a": ("LORA",),
                "lora_data_b": ("LORA",),
                "lora_data_c": ("LORA",),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                                       "tooltip": "Strength of first LoRA"}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                                       "tooltip": "Strength of second LoRA"}),
                "weight_c": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                                       "tooltip": "Strength of third LoRA"}),

                # ── SECTION 3: SETTINGS ─────────────────────────────────
                "blend_mode": (["auto", "active", "dense"], {
                    "default": "auto",
                    "tooltip": "auto: Smart choice based on trainer metadata (match → dense, mismatch → active) | dense: Traditional weighted sum"
                }),

                # ── SECTION 4: ADVANCED ─────────────────────────────────
                "energy_preservation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve energy distribution during merge (recommended). Disable for raw weighted sum."
                }),
                "balancing_mode": (["safe", "creative", "disabled", "intensity", "impact"], {
                    "default": "safe",
                    "tooltip": "Auto-weight-balancing: safe (hard-match energy on shared layers only, rank-independent), creative (preserve ratio with reduced magnitude), intensity (peak-energy detection for cross-concept merges), impact (intensity + sparsity correction for sparse-vs-dense merges like Anima), disabled (no adjustment)."
                }),
                "magnitude_scaling": (["none", "rms", "top_5%", "top_10%", "top_20%", "top_30%"], {
                    "default": "none",
                    "tooltip": "Signal magnitude scaling before merging – scales LoRA B and C to match LoRA A's energy using RMS or top‑X% percentile."
                }),
                "active_threshold": ("BOOLEAN", {"default": True,
                                               "tooltip": "Enable active region detection (threshold value from config.py). Disable for dense-style behavior."}),
                "uniqueness": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01,
                                        "tooltip": "For feature_mix: higher = preserve more unique features"}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "For subtract: minimum magnitude to subtract"}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "For magnitude: 0=strict, 1=blended"}),

                # ── SECTION 5: HARDWARE ─────────────────────────────────
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "precision": (PRECISION_STANDARD, {"default": "auto"}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 8,
                    "tooltip": "Number of keys to process per batch. "
                               "DeviceManager.suggest_batch_size() can auto-tune based on VRAM."}),
                "streaming": ("BOOLEAN", {"default": True,
                                          "tooltip": "Stream tensors to save VRAM"}),

                # ── SECTION 6: OUTPUT ───────────────────────────────────
                "save_trigger": ("BOOLEAN", {"default": False}),
                "filename": ("STRING", {"default": "triple_merged", "multiline": False}),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False}),
                "metadata_mode": (["none", "preserve_a", "preserve_b", "merge_basic"], {
                    "default": "merge_basic",
                    "tooltip": "How to handle metadata from source LoRAs"
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache implementation for ComfyUI change detection."""
        # Inline data guard: wired lora_data inputs can't be cheaply hashed
        for guard in ['lora_data_a', 'lora_data_b', 'lora_data_c']:
            if kwargs.get(guard) is not None:
                return float("nan")

        return NodeCache.is_changed(
            cls.__name__,
            lora_a=kwargs.get('lora_a', 'None'),
            lora_b=kwargs.get('lora_b', 'None'),
            lora_c=kwargs.get('lora_c', 'None'),
            weight_a=kwargs.get('weight_a', 1.0),
            weight_b=kwargs.get('weight_b', 1.0),
            weight_c=kwargs.get('weight_c', 1.0),
            method=kwargs.get('method', 'linear'),
            density=kwargs.get('density', 1.0),
            save_trigger=kwargs.get('save_trigger', False),
            uniqueness=kwargs.get('uniqueness', 0.7),
            threshold=kwargs.get('threshold', 0.0),
            blend=kwargs.get('blend', 0.5),
            active_threshold=kwargs.get('active_threshold', 1e-8),
            magnitude_scaling=kwargs.get('magnitude_scaling', 'none'),
            blend_mode=kwargs.get('blend_mode', 'auto'),
            precision=kwargs.get('precision', 'auto'),
            device=kwargs.get('device', 'auto'),
            batch_size=kwargs.get('batch_size', 32),
            streaming=kwargs.get('streaming', True),
            balancing_mode=kwargs.get('balancing_mode', 'safe'),
            energy_preservation=kwargs.get('energy_preservation', True),
            metadata_mode=kwargs.get('metadata_mode', 'merge_basic'),
        )
    
    
    RETURN_TYPES = ("LORA", "MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("lora", "model", "clip", "output_path", "forensic_report")
    FUNCTION = "merge_triple"
    CATEGORY = "LoRA/Experimental"
    
    def merge_triple(self, model, clip, method="linear", density=1.0,
                    lora_a="None", lora_b="None", lora_c="None",
                    lora_data_a=None, lora_data_b=None, lora_data_c=None,
                    weight_a=1.0, weight_b=1.0, weight_c=1.0,
                    uniqueness=0.7, threshold=0.0, blend=0.5, blend_mode="auto",
                    active_threshold=True,
                    magnitude_scaling="none",
                    save_trigger=False, save_folder="", filename="triple_merged",
                    device="auto", precision="auto", batch_size=32,
                    streaming=True, energy_preservation=True, balancing_mode="safe", metadata_mode="merge_basic"):

        print("\n" + "="*50)
        print("🎨 Easy LoRA Merger (with Smart‑Pro Defaults)")
        print("="*50)

        # Hidden studio knobs – smart‑pro defaults
        target_format = "auto"
        keep_alphas = False
        bake_custom_scale = 1.0

        # ══ Runtime precision guard — protect against stale workflow JSONs ══
        if precision not in PRECISION_STANDARD:
            print(f"   ⚠️ Precision '{precision}' is no longer available, falling back to 'auto'")
            precision = "auto"

        # Resolve active_threshold bool to actual float threshold value
        # True  → use the default threshold from config.py
        # False → use 0.0 (all non-zero values treated as active, effectively disabling)
        active_threshold_float = ACTIVE_THRESHOLD_DEFAULT if active_threshold else 0.0

        # Create config with only the variables we have
        config = MergeConfig(
            method=method,
            density=density,
            weight_a=weight_a,
            weight_b=weight_b,
            uniqueness=uniqueness,
            threshold=threshold,
            blend=blend,
            blend_mode=blend_mode,
            device_type=device,
            precision=precision,
            batch_size=batch_size,
            streaming=streaming,
            energy_preservation=energy_preservation,
            balancing_mode=balancing_mode,
            active_threshold=active_threshold_float,
            magnitude_scaling=magnitude_scaling,
            metadata_mode=metadata_mode
        )
        
        # Get paths for all three LoRAs
        paths = []
        loras_data = [lora_data_a, lora_data_b, lora_data_c]
        loras_dropdown = [lora_a, lora_b, lora_c]
        names = ["A", "B", "C"]
        
        two_way_mode = False
        missing_allowed_for_c = False
        sds = []
        metas = []
        for i, (data, dropdown, name) in enumerate(zip(loras_data, loras_dropdown, names)):
            if data is not None:
                # LORA data input: unpack state dict directly — no temp file needed
                lora_dict = data[0] if isinstance(data, (tuple, list)) else data
                sds.append(lora_dict)
                metas.append({})
                print(f"📄 {name}: LORA data (in‑memory, {len(lora_dict)} tensors)")
            elif dropdown != "None" and dropdown:
                path = folder_paths.get_full_path("loras", dropdown)
                print(f"📄 {name}: {dropdown}")
                sd, meta = load_lora_with_metadata(Path(path))
                sds.append(sd)
                metas.append(meta)
            else:
                # Missing input
                if name == "C":
                    # Allow missing C, treat as two‑way merge
                    print(f"⚠️ {name}: Missing input – falling back to two‑way merge (A + B)")
                    two_way_mode = True
                    missing_allowed_for_c = True
                    continue
                else:
                    # Missing A or B is fatal
                    print(f"❌ {name}: Missing input")
                    return (None, model, clip, "",
                            "❌ Aborted: Missing LoRA A or B — cannot proceed")
        
        # Output path logic
        if save_trigger:
            output_path = get_user_output_path(save_folder, filename)
        else:
            output_path = get_experiment_temp_path("triple")
        
        print("📥 Loaded LoRAs in memory")
        
        # Extract trigger words and warn about conflicts
        converter = MusubiLoraConverter()
        trigger_words = []
        for i, meta in enumerate(metas):
            info = converter.analyze_lora_structure(sds[i], meta)
            concept = info.get('concept_analysis', {})
            trigger = concept.get('trigger_word')
            trigger_words.append(trigger)
        # Compare pairs
        names = ['A', 'B', 'C']
        for i in range(len(trigger_words)):
            for j in range(i+1, len(trigger_words)):
                if trigger_words[i] and trigger_words[j] and trigger_words[i] == trigger_words[j]:
                    print(f"⚠️ Warning: Both LoRA {names[i]} and LoRA {names[j]} use the same trigger word '{trigger_words[i]}'. Results may be unpredictable.")
        
        # Check if they're all the same type
        print("🔍 Detecting formats...")
        formats = [detect_lora_format(sd) for sd in sds]
        print(f"   Formats: {formats}")
        
        # Normalize all to common format with identity mapping
        # SD1.5 Diffusers LoRA detection: bridge preserves block structure (avoids ~175/396 key loss)
        from .engine.diffusers_bridge import detect_diffusers_sd15_lora, normalize_diffusers_preserving
        
        print("🔄 Normalizing with identity mapping...")
        normalized_sds = []
        mappings = []
        for sd, meta in zip(sds, metas):
            if detect_diffusers_sd15_lora(sd):
                print("   🔄 SD1.5 Diffusers LoRA detected – using Diffusers-preserving normalization")
                norm, mapping = normalize_diffusers_preserving(sd)
            else:
                norm, mapping = identity_normalize(sd, meta)
            normalized_sds.append(norm)
            mappings.append(mapping)
        
        # Forensics data collectors
        energy_by_component = None
        adjusted_weights = None
        warnings_list: List[str] = []
        
        # Decide between triple and two‑way merge
        if len(sds) == 2:
            print("🔄 Two‑way merge (LoRA C missing) – using IdentityMergeEngine")
            warnings_list.append("Two-way fallback: LoRA C was missing")
            # Use IdentityMergeEngine with the same config
            engine = IdentityMergeEngine(config)
            merged_dict, master_map = engine.merge_with_mapping(sds[0], sds[1], metas[0], metas[1])
            print(f"✅ Two‑way merge completed ({len(merged_dict)} keys)")
            resolved_blend_mode = resolve_blend_mode_triple(blend_mode, metas)
        else:
            # Triple merge
            print(f"🎯 Method: {method}")
            if method == "slerp":
                print("   🔮 SLERP selected — uses |weight| for interpolation factor t; weight sign is discarded")
                print("   ℹ️ SLERP acts on the 2 strongest LoRAs (|weight| > 0); falls back to dense sum for 1 or 3 active")
            # Resolve UI blend_mode to internal value
            resolved_blend_mode = resolve_blend_mode_triple(blend_mode, metas)
            if resolved_blend_mode != blend_mode:
                print(f"   🔀 blend_mode '{blend_mode}' → '{resolved_blend_mode}'")
            
            merged_dict, master_map, energy_by_component, adjusted_weights = merge_triple_method(
                normalized_sds,
                [weight_a, weight_b, weight_c],
                method, density, uniqueness, threshold, blend, blend_mode=resolved_blend_mode,
                device=config.device_type,
                precision=config.precision,
                magnitude_scaling=config.magnitude_scaling,
                max_scaling_factor=config.max_scaling_factor,
                batch_size=config.batch_size,
                streaming=config.streaming,
                energy_preservation=config.energy_preservation,
                balancing_mode=config.balancing_mode,
                mappings=mappings,
                original_sds=sds,
                metas=metas
            )
        
        # ==================== SMART‑PRO DEFAULTS ====================
        # Apply up‑weight scaling (if requested)
        effective_scaling = abs(bake_custom_scale - 1.0) > 1e-6
        if effective_scaling:
            print(f"⚖️ Applying up‑weight scaling: {bake_custom_scale}x")
            scaled_keys = 0
            for key in list(merged_dict.keys()):
                if key.endswith(".weight") and ("lora_B" in key or "lora_up" in key):
                    merged_dict[key] = merged_dict[key] * bake_custom_scale
                    scaled_keys += 1
            print(f"   Scaled {scaled_keys} up‑weight tensors")
        
        # Auto‑detect target format if "auto"
        effective_target_format = target_format
        if target_format == "auto":
            # Infer separator and naming styles from normalized keys
            separator = infer_separator_style(list(merged_dict.keys()))
            naming = infer_naming_style(list(merged_dict.keys()))
            # Determine ecosystem (use metadata from first LoRA as proxy)
            target_ecosystem = metas[0].get('target_ecosystem', 'Unknown')
            primary_structure = metas[0].get('primary_structure', '')
            # Decision logic (same as Musubi converter)
            if target_ecosystem in ("Flux.1-Dev/S", "Z-Image"):
                # Flux/Z‑Image models benefit from forge_optimized (prefix mapping)
                if separator == "underscore" and naming == "lora_down_up":
                    effective_target_format = "forge_optimized"
                else:
                    # Fallback to comfy_native for dot separator
                    effective_target_format = "comfy_native"
            elif separator == "dot" and naming == "lora_a_b":
                effective_target_format = "comfy_native"
            else:
                # Default to standard_webui (underscore + lora_down_up)
                effective_target_format = "standard_webui"
            print(f"🔍 Auto‑selected target format: {effective_target_format} (separator={separator}, naming={naming}, ecosystem={target_ecosystem})")
        
        # Legacy‑aware alpha baking
        bake_alphas_flag = not keep_alphas
        if not keep_alphas:
            # User left checkbox unchecked → apply smart default based on model type
            target_ecosystem = metas[0].get('target_ecosystem', 'Unknown')
            if target_ecosystem in ("Flux.1-Dev/S", "Z-Image"):
                bake_alphas_flag = True   # bake alphas for Flux/Z‑Image
            else:
                bake_alphas_flag = False  # keep alphas for SDXL/SD1.5
            print(f"📝 Legacy‑aware alpha baking: alphas will be {'baked' if bake_alphas_flag else 'kept'} (detected {target_ecosystem})")
        
        # Handle alpha baking BEFORE finalize (separate from format conversion)
        if bake_alphas_flag:
            from .utils import bake_alphas as bake_alphas_func
            # infer_naming_style already imported at module level (line 28)
            baking_naming_style = infer_naming_style(list(merged_dict.keys()))
            merged_dict = bake_alphas_func(merged_dict, naming_style=baking_naming_style)
            # Remove baked alpha keys from master_map
            master_map = {k: v for k, v in master_map.items() if not k.endswith('.alpha')}
            print(f"🔥 Alpha keys baked into up‑weights ({len(merged_dict)} tensors)")

        # Use identity mapping to restore original keys (avoids fragile pattern matching)
        restored_dict, _ = finalize_for_save(
            merged_dict, master_map,
            target_format=effective_target_format,
            meta_a=metas[0] if metas else None,
            meta_b=metas[1] if metas and len(metas) > 1 else None
        )

        # Apply forge_optimized prefix mapping (Flux DiT blocks: diffusion_model. → transformer.)
        if effective_target_format == 'forge_optimized':
            forge_count = 0
            for key in list(restored_dict.keys()):
                new_key = key
                if ('double_blocks' in new_key or 'single_blocks' in new_key):
                    if new_key.startswith('diffusion_model.'):
                        new_key = 'transformer.' + new_key[len('diffusion_model.'):]
                        forge_count += 1
                if new_key != key:
                    restored_dict[new_key] = restored_dict.pop(key)
            if forge_count:
                print(f"   🔧 Applied forge prefix mapping to {forge_count} Flux keys")

        converted_dict = restored_dict
        print(f"🎯 Converted to {effective_target_format}: {len(converted_dict)} keys")
        
        # Merge metadata using unified metadata_factory
        merged_metadata = {}
        if metas and len(metas) >= 1:
            metas_nonnull = [m or {} for m in metas]
            
            if metadata_mode == "merge_basic":
                # Chain merge using MetadataMerger to combine multiple LoRA sources,
                # then finalize_metadata handles scrub + sign.
                merged = MetadataMerger.merge(metas_nonnull[0], metas_nonnull[1] if len(metas_nonnull) > 1 else {}, mode="merge_basic")
                if len(metas_nonnull) > 2:
                    merged = MetadataMerger.merge(merged, metas_nonnull[2], mode="merge_basic")
                print(f"📝 Metadata mode 'merge_basic' – merged metadata from {len(metas_nonnull)} LoRAs ({len(merged)} entries)")

                # Extract trigger words
                converter = MusubiLoraConverter()
                trigger_words = []
                for meta in metas_nonnull:
                    flat = converter._parse_metadata(meta)
                    triggers = converter._extract_trigger_words(flat)
                    if triggers:
                        trigger_words.append(triggers)
                if trigger_words:
                    all_triggers = set()
                    for tw in trigger_words:
                        if isinstance(tw, tuple):
                            words = tw[0]
                        elif isinstance(tw, list):
                            words = tw
                        else:
                            words = [p.strip() for p in tw.split(',') if p.strip()]
                        all_triggers.update(words)
                    if all_triggers:
                        merged["trigger_words"] = ', '.join(sorted(all_triggers))

                merged_metadata = finalize_metadata(
                    metadata=merged,
                    mode=metadata_mode,
                    component="merger",
                    extra_fields={"merged_key_count": str(len(merged_dict))}
                )
            else:
                # none / preserve_a / preserve_b — handled directly by finalize_metadata
                merged_metadata = finalize_metadata(
                    metadata=metas_nonnull[0],
                    mode=metadata_mode,
                    component="merger",
                    extra_fields={"merged_key_count": str(len(merged_dict))}
                )

        # Convert output LoRA tensors to user-chosen dtype (merge computation
        # stays in bfloat16 for stability, but the output honors the precision parameter)
        target_dtype = config.device_precision.dtype
        if target_dtype is not None:
            converted_count = 0
            for key in list(converted_dict.keys()):
                t = converted_dict[key]
                if t.dtype != target_dtype:
                    converted_dict[key] = t.to(target_dtype)
                    converted_count += 1
            if converted_count:
                print(f"🎯 Converted {converted_count} output tensors to {target_dtype}")

        # Finalize and save
        if save_trigger:
            output_path, save_path_str = save_safetensors_file(
                converted_dict, save_folder, filename, metadata=merged_metadata,
                folder_type="loras", default_name="triple_merged",
            )
            if output_path is None:
                print(f"   ⚠️ Save failed: {save_path_str}")
        else:
            print("🧠 Preview mode: LoRA kept in RAM, no file written")
        
        # Load into model (always from RAM - converted_dict is still in memory)
        try:
            lora_data = converted_dict
            print("🧠 Using merged LoRA from RAM")
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora_data, 1.0, 1.0
            )
            lora_tuple = (lora_data, 1.0, 1.0)
        except Exception as e:
            print(f"❌ Load failed: {e}")
            error_report = f"❌ Load failed after merge: {e}"
            return (None, model, clip, "", "❌ Load failed after merge: " + str(e))
        
        # Determine save path for return value
        if save_trigger and output_path is not None:
            final_save_path = str(output_path)
        else:
            final_save_path = ""
        
        # Collect density warning
        if density < 1.0:
            warnings_list.append(f"Density < 1.0 ({density}): sparse selection applied — individual key magnitudes may differ")
        
        # Build forensic report for success
        active_sds = [normalized_sds[i] for i in range(len(normalized_sds))]
        lora_labels = ["A", "B", "C"][:len(sds)]
        active_weights = [weight_a, weight_b, weight_c][:len(sds)]

        forensic_report = build_triple_forensic_report(
            method=method,
            density=density,
            blend_mode=blend_mode,
            resolved_blend_mode=resolved_blend_mode,
            balancing_mode=balancing_mode,
            weights=active_weights,
            lora_names=lora_labels,
            two_way_mode=(len(sds) == 2),
            active_lora_count=len(sds),
            merged_dict=merged_dict,
            original_sds=active_sds,
            energy_by_component=energy_by_component,
            adjusted_weights=adjusted_weights,
            warnings=warnings_list if warnings_list else None,
        )
        
        print("="*50)
        print("🎉 Triple Merge Complete!")
        print("="*50)
        
        return (lora_tuple, model_lora, clip_lora, final_save_path, forensic_report)