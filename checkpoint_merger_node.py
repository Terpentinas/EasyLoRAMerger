"""
Easy Checkpoint Merger Node
Merges two or three full model checkpoints (.safetensors) using triple‑merge methods with Weight Block Map.
Streaming engine ensures low RAM usage.
"""

import torch
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

import folder_paths
import comfy.utils

from .utils import (
    load_checkpoint_with_metadata,
    get_checkpoint_output_path,
    DeviceManager,
    memory_optimized_merge,
    cleanup_memory,
    NodeCache,
    load_state_dict_as_model_objects,
    check_ram_guard,
    estimate_disk_mode_peak,
    get_available_ram,
    get_free_disk_space,
    ProgressTracker,
    get_combined_model_list,
    resolve_model_path,
)
from .config import PRECISION_EXTENDED, DEVICE_OPTIONS
from .engine.methods import resolve_blend_mode_triple
from .engine.checkpoint_weaver import CheckpointTripleMerger
from .engine.fp8_quantizer import (
    build_fp8_quantization_metadata as _build_fp8_quantization_metadata,
    strip_input_scale_keys as _strip_input_scale_keys,
)


# ── Named constants ──────────────────────────────────────────────────
# Maximum scaling factor for magnitude equalization in checkpoint merges.
# Kept at 10.0 (vs MergeConfig.max_scaling_factor = 200.0) because full
# checkpoint weight matrices have much larger magnitudes than LoRA deltas,
# so aggressive scaling is unnecessary and potentially destabilising.
CHECKPOINT_MAX_SCALING_FACTOR: float = 10.0


class EasyCheckpointMerger:
    """
    Merge two or three checkpoints with component‑wise scaling (Weight Block Map).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_combined_model_list()
        default_folder = ""
        checkpoint_folders = folder_paths.get_folder_paths("checkpoints")
        if checkpoint_folders:
            default_folder = str(checkpoint_folders[0])
        
        # Method tooltips for better UX
        method_tooltips = {
            "linear": "Simple weighted average - good starting point",
            "ties_strict": "Keep only where signs agree - effective for conflicting styles",
            "ties_gentle": "Apply TIES only for strong disagreements",
            "dare_lite": "Random dropout without rescaling - stochastic sparsification",
            "dare_rescale": "Random dropout with rescaling - preserves magnitude distribution",
            "slerp": "Spherical linear interpolation - smooth directional blending between two vectors",
            "subtract": "Subtract B from A - remove unwanted features",
            "magnitude": "Keep larger magnitude from either checkpoint - blend controls strictness",
            "feature_mix": "Preserve unique features from each checkpoint - uniqueness controls preservation",
            "svd_preserve": "SVD-based rank reduction - preserves structure",
            "noise_aware": "Reduce small noise values before merging",
            "gradient_alignment": "Weight by directional similarity",
            "cross": "Cross-magnitude merge with pairwise interaction term",
            "ties_contrast": "Amplifies disagreements between checkpoints, mutes agreements",
            "block_swap": "Deterministic block-swapping via seeded RNG"
        }

        return {
            "required": {
                "method": (["linear", "ties_strict", "ties_gentle", "dare_lite",
                           "dare_rescale", "slerp", "subtract",
                           "magnitude", "feature_mix", "svd_preserve", "noise_aware",
                           "gradient_alignment", "cross", "ties_contrast", "block_swap"], {
                    "default": "linear",
                    "tooltip": "Choose merging method"
                }),
                "density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01,
                                      "tooltip": "Keep top % of weights after merging. WARNING: Values < 1.0 sparsify checkpoint weights, which may degrade quality. Only reduce if you understand the risk."}),
            },
            "optional": {
                # ── SECTION 2: SOURCES ──────────────────────────────────
                "checkpoint_a": (["None"] + checkpoints,),
                "checkpoint_b": (["None"] + checkpoints,),
                "checkpoint_c": (["None"] + checkpoints,),
                # Chain data inputs (raw state dicts from previous merger)
                "checkpoint_data_a": ("CHECKPOINT",),
                "checkpoint_data_b": ("CHECKPOINT",),
                "checkpoint_data_c": ("CHECKPOINT",),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                                       "tooltip": "Global strength of first checkpoint. For linear method: weights should sum close to 1.0 (e.g., 0.5+0.5) to avoid doubling magnitudes — 1.0+1.0 produces noise."}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                                       "tooltip": "Global strength of second checkpoint. For linear method: weights should sum close to 1.0 (e.g., 0.5+0.5) to avoid doubling magnitudes — 1.0+1.0 produces noise."}),
                "weight_c": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                                       "tooltip": "Global strength of third checkpoint. For linear method: weights should sum close to 1.0 (e.g., 0.5+0.5) to avoid doubling magnitudes — 1.0+1.0 produces noise."}),
                "weight_unet": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                          "tooltip": "Component scaling for UNET weights"}),
                "weight_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                          "tooltip": "Component scaling for CLIP visual encoder"}),
                "weight_vae": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                         "tooltip": "Component scaling for VAE"}),
                "weight_te": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                        "tooltip": "Component scaling for Text Encoder"}),

                # ── SECTION 3: SETTINGS ─────────────────────────────────
                "blend_mode": (["auto", "active", "dense"], {
                    "default": "auto",
                    "tooltip": "auto: Smart choice based on trainer metadata (match → dense, mismatch → active) | dense: Traditional weighted sum"
                }),

                # ── SECTION 4: ADVANCED ─────────────────────────────────
                # energy_preservation removed from UI — hardcoded True (safety net, never disabled)
                "balancing_mode": (["disabled", "safe", "creative"], {
                    "default": "disabled",
                    "tooltip": "disabled: Use weights as given (no equalization) | safe: Subtle equalization for cross-architecture merges | creative: Looser equalization (experimental)"
                }),
                "magnitude_scaling": (["none", "rms", "top_5%", "top_10%", "top_20%", "top_30%"], {
                    "default": "none",
                    "tooltip": "Signal magnitude scaling before merging – scales checkpoint B and C to match A's energy using RMS or top‑X% percentile."
                }),
                "uniqueness": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01,
                                         "tooltip": "For feature_mix: higher = preserve more unique features"}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "For subtract: minimum magnitude to subtract"}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "For magnitude: 0=strict, 1=blended"}),

                # ── SECTION 5: HARDWARE ─────────────────────────────────
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "precision": (PRECISION_EXTENDED, {"default": "auto"}),
                "batch_size": ("INT", {"default": 64, "min": 1, "max": 256, "step": 8,
                    "tooltip": "Number of keys to process per batch. "
                               "DeviceManager.suggest_batch_size() can auto-tune based on VRAM."}),
                # streaming removed from UI — hardcoded True (always optimal for checkpoints)

                # ── SECTION 6: OUTPUT ───────────────────────────────────
                "save_trigger": ("BOOLEAN", {"default": False}),
                "filename": ("STRING", {"default": "merged_checkpoint", "multiline": False}),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False}),
                "metadata_mode": (["none", "preserve_a", "preserve_b", "merge_basic"], {
                    "default": "merge_basic",
                    "tooltip": "How to handle metadata from source checkpoints"
                }),
            }
        }
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache implementation for ComfyUI change detection."""
        # If any data input is connected, always re-execute (can't cheaply hash state dicts)
        data_guards = ['checkpoint_data_a', 'checkpoint_data_b', 'checkpoint_data_c']
        if any(kwargs.get(p) is not None for p in data_guards):
            return float("nan")

        return NodeCache.is_changed(
            cls.__name__,
            checkpoint_a=kwargs.get('checkpoint_a', 'None'),
            checkpoint_b=kwargs.get('checkpoint_b', 'None'),
            checkpoint_c=kwargs.get('checkpoint_c', 'None'),
            weight_a=kwargs.get('weight_a', 1.0),
            weight_b=kwargs.get('weight_b', 1.0),
            weight_c=kwargs.get('weight_c', 1.0),
            weight_unet=kwargs.get('weight_unet', 1.0),
            weight_clip=kwargs.get('weight_clip', 1.0),
            weight_vae=kwargs.get('weight_vae', 1.0),
            weight_te=kwargs.get('weight_te', 1.0),
            method=kwargs.get('method', 'linear'),
            density=kwargs.get('density', 1.0),
            save_trigger=kwargs.get('save_trigger', False),
            uniqueness=kwargs.get('uniqueness', 0.7),
            threshold=kwargs.get('threshold', 0.0),
            blend=kwargs.get('blend', 0.5),
            magnitude_scaling=kwargs.get('magnitude_scaling', 'none'),
            blend_mode=kwargs.get('blend_mode', 'auto'),
            balancing_mode=kwargs.get('balancing_mode', 'disabled'),
            precision=kwargs.get('precision', 'auto'),
            device=kwargs.get('device', 'auto'),
            batch_size=kwargs.get('batch_size', 32),
            metadata_mode=kwargs.get('metadata_mode', 'merge_basic'),
        )
    
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CHECKPOINT", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "checkpoint_data", "output_path", "forensic_report")
    FUNCTION = "merge_triple"
    CATEGORY = "Checkpoint/Experimental"
    OUTPUT_NODE = True

    def merge_triple(self,
                     method="linear",
                     density=1.0,
                     checkpoint_a="None",
                     checkpoint_b="None",
                     checkpoint_c="None",
                     checkpoint_data_a=None,
                     checkpoint_data_b=None,
                     checkpoint_data_c=None,
                     weight_a=1.0,
                     weight_b=1.0,
                     weight_c=1.0,
                     weight_unet=1.0,
                     weight_clip=1.0,
                     weight_vae=1.0,
                     weight_te=1.0,
                     uniqueness=0.7,
                     threshold=0.0,
                     blend=0.5,
                     blend_mode="auto",
                     magnitude_scaling="none",
                     save_trigger=False,
                     save_folder="",
                     filename="merged_checkpoint",
                     device="auto",
                     precision="auto",
                     batch_size=64,
                     balancing_mode="disabled",
                     metadata_mode="merge_basic"):
        
        print("\n" + "="*60)
        print("🎨 Easy Checkpoint Merger")
        print("="*60)
        _T0 = time.time()
        print(f"   🕐 [t={0.0:.1f}s] merge_triple start")
        
        # ══ Runtime precision guard — protect against stale workflow JSONs ══
        if precision not in PRECISION_EXTENDED:
            print(f"   ⚠️ Precision '{precision}' is no longer available, falling back to 'auto'")
            precision = "auto"
        
        # ── Collect source paths and in-memory dicts ──
        # When a data input (checkpoint_data_*) is connected, pass the dict directly
        # to the writer via source_dicts, avoiding any temp file write.
        source_paths = []
        source_dicts = []
        weights = []
        missing_allowed_for_c = False
        for name, data, dropdown, weight in [
            ("A", checkpoint_data_a, checkpoint_a, weight_a),
            ("B", checkpoint_data_b, checkpoint_b, weight_b),
            ("C", checkpoint_data_c, checkpoint_c, weight_c),
        ]:
            if data is not None:
                # Chain data input: use in-memory dict directly — NO temp file needed
                print(f"📦 {name}: Chained checkpoint data (dict) — {len(data)} tensors")
                source_paths.append(Path(""))  # placeholder path (never opened for dict sources)
                source_dicts.append(data)
                weights.append(weight)
            elif dropdown != "None" and dropdown:
                full_path = resolve_model_path(dropdown)
                if full_path is None:
                    raise ValueError(f"Checkpoint {name} not found: {dropdown}")
                source_paths.append(Path(full_path))
                source_dicts.append(None)
                weights.append(weight)
            else:
                # Missing input
                if name == "C":
                    # Allow missing C, treat as two‑way merge
                    print(f"⚠️ {name}: Missing input – falling back to two‑way merge (A + B)")
                    missing_allowed_for_c = True
                    continue
                else:
                    # Missing A or B is fatal
                    raise ValueError(
                        f"Checkpoint {name}: neither file nor data provided. "
                        f"Provide at least checkpoint_{name.lower()} or checkpoint_data_{name.lower()}."
                    )
        
        if len(source_paths) < 2:
            raise ValueError("At least two checkpoints must be provided.")
        
        print(f"📁 Sources: {[p.name if p.name else '(in-memory dict)' for p in source_paths]}")
        print(f"⚖️ Global weights: {weights}")
        print(f"🧱 Component scaling – UNET: {weight_unet}, CLIP: {weight_clip}, VAE: {weight_vae}, TE: {weight_te}")
        
        # ── Determine output path ──
        _use_dict = True  # Default: merge to dict, load directly from dict (lowest RAM)
        if save_trigger and save_folder is not None and filename is not None:
            output_path = get_checkpoint_output_path(save_folder, filename)
        else:
            # Preview mode: no temp file needed unless RAM guard triggers fallback
            output_path = None
        
        # ── Merge configuration for streaming writer ──
        merge_config = {
            "method": method,
            "weights": weights,
            "density": density,
            "uniqueness": uniqueness,
            "threshold": threshold,
            "blend": blend,
            "blend_mode": blend_mode,
            "magnitude_scaling": magnitude_scaling,
            "max_scaling_factor": CHECKPOINT_MAX_SCALING_FACTOR,  # 10.0 — safer for full checkpoints vs MergeConfig default 200.0 (LoRA path)
            "batch_size": batch_size,
            "streaming": True,
            "energy_preservation": True,
            "balancing_mode": balancing_mode,
            "weight_unet": weight_unet,
            "weight_clip": weight_clip,
            "weight_vae": weight_vae,
            "weight_te": weight_te,
            "precision": precision,
            "device": device,
        }
        
        # ── Metadata options ──
        metadata_options = {
            "keep_metadata": metadata_mode != "none",
            "inject_sai_header": True,  # always inject
        }
        
        # ── Create streaming writer and initial survey (populate tensor_list) ──
        print("📋 Surveying checkpoint headers...")
        writer = CheckpointTripleMerger(
            source_paths=source_paths,
            output_path=output_path,
            merge_config=merge_config,
            metadata_options=metadata_options,
            use_dict=_use_dict,
            source_dicts=source_dicts if any(d is not None for d in source_dicts) else None,
        )
        writer.survey()  # Initial survey WITHOUT pbar (tensor_list needed for RAM guard)
        print(f"   🕐 [t={time.time()-_T0:.1f}s] Survey complete — {len(writer.tensor_list)} tensors")
        
        # ── RAM Safety Guard: check available memory before weave ──
        total_bytes = sum(t["size"] for t in writer.tensor_list)
        if not check_ram_guard(total_bytes, len(writer.tensor_list), batch_size, use_dict=_use_dict):
            # Allocate temp path NOW — needed for disk mode fallback
            temp_dir = Path(folder_paths.get_temp_directory())
            temp_dir.mkdir(parents=True, exist_ok=True)
            unique = uuid.uuid4().hex[:8]
            output_path = temp_dir / f"merged_checkpoint_{unique}.safetensors"
            print(f"   ⚠️ RAM fallback — writing merged checkpoint to temp file: {output_path}")
            print(f"   ℹ️  Auto-falling back to disk mode for low-RAM safety")
            print(f"   ⚠️  Re-surveying: reading all checkpoint headers again (I/O overhead)")
            _use_dict = False
            writer = CheckpointTripleMerger(
                source_paths=source_paths,
                output_path=output_path,
                merge_config=merge_config,
                metadata_options=metadata_options,
                use_dict=False,
                source_dicts=source_dicts if any(d is not None for d in source_dicts) else None,
            )
            writer.survey()  # Re-survey without pbar (fallback path)
        
        # ── Disk Space Guard: check free space before weave ──
        if output_path is not None:
            free_bytes = get_free_disk_space(output_path.parent)
            if free_bytes is not None:
                estimated_file_size = sum(t["size"] for t in writer.tensor_list) + 8 + 65536
                print(f"💾 Disk Space: Estimated {estimated_file_size / (1024**3):.2f} GB, "
                      f"Free on drive: {free_bytes / (1024**3):.2f} GB")
                if estimated_file_size > free_bytes * 0.95:
                    raise OSError(
                        f"🚫 Insufficient disk space.\n"
                        f"  Estimated file size: {estimated_file_size / (1024**3):.2f} GB\n"
                        f"  Free space: {free_bytes / (1024**3):.2f} GB"
                    )
            else:
                print("⚠️ Disk space check unavailable — proceeding without guard")
        
        # ── Unified ProgressTracker: single 0-100% across survey + weave ──
        total_survey = len(source_paths)
        total_weave = len(writer.tensor_list)
        # Use explicit enter/exit (not `with`) so pbar is in scope for
        # survey -> weave -> forensic report sequence.
        pbar = ProgressTracker(total=total_survey + total_weave, desc="Checkpoint merger")
        pbar.__enter__()
        try:
            # Re-survey with pbar to advance unified tracker (advances by total_survey steps)
            writer.survey(pbar=pbar)

            # ── Resolve blend_mode after survey (metas now available) ──
            # "auto" → "active" if trainers mismatch, "dense" if all match
            resolved_blend_mode = resolve_blend_mode_triple(blend_mode, writer.metadatas)
            if resolved_blend_mode != blend_mode:
                print(f"   🔀 blend_mode '{blend_mode}' → '{resolved_blend_mode}'")
            merge_config["blend_mode"] = resolved_blend_mode  # writer holds a ref, so this propagates

            # Weave with same pbar — merge_triple_method advances pbar via on_substep callback
            # (advances by total_weave steps, 1 per key)
            writer.weave(pbar=pbar)
        finally:
            pbar.__exit__(None, None, None)
        print(f"   🕐 [t={time.time()-_T0:.1f}s] Weave complete — merged {len(writer.tensor_list)} tensors")
        
        # ── Generate forensic report ──
        forensic_report = writer.generate_forensic_report()
        print(forensic_report)

        # ── Capture merged_metadata BEFORE deleting the writer ──
        merged_metadata = writer.merged_metadata
        
        # ── FP8 EXPORT: Detect if output is FP8 ──
        # The engine (checkpoint_weaver.py) emits companion scale slots during
        # survey() and quantizes to FP8 during weave(). The _quantization_metadata
        # was conditionally kept in survey() (not stripped when target is FP8),
        # but we still need to build fresh metadata from the output state dict
        # because the source metadata may describe INPUT FP8 quantization, not
        # the freshly-quantized OUTPUT.
        _is_fp8_output = writer.target_dtype_str in ("F8_E4M3", "F8_E5M2")
        if _is_fp8_output:
            print(f"   🎯 FP8 output detected: {writer.target_dtype_str}")
        
        # ── Measure pre-output available RAM for post-merge comparison ──
        _pre_avail_ram = get_available_ram()
        
        # ── Load merged model objects (both save_trigger True and False) ──
        if _use_dict:
            state_dict = writer.get_output_dict()
            
            # ── FP8 EXPORT: Strip .input_scale and inject _quantization_metadata ──
            # MixedPrecisionOps only needs .weight_scale — .input_scale keys
            # appear as "unexpected" when loading from dict. Build fresh metadata
            # by scanning the output state dict for FP8 tensor dtypes.
            if _is_fp8_output:
                _strip_input_scale_keys(state_dict)
                merged_metadata = _build_fp8_quantization_metadata(
                    state_dict, merged_metadata, label="checkpoint_merger"
                )
            
            if save_trigger:
                # ── Save to disk + load model (save + preview) ──
                print(f"💾 save_trigger=True — saving to disk + loading model")
                save_path_str = str(output_path)
                
                if hasattr(writer, '_omitted_components') and writer._omitted_components:
                    print(f"   ⚠️ Components {writer._omitted_components} omitted (weight=0)")
                
                del writer
                cleanup_memory()
                
                # Capture dict for chaining before freeing writer
                checkpoint_data = state_dict.copy() if state_dict else None
                
                # Serialize to disk — streaming, one tensor at a time
                if output_path is not None and state_dict is not None:
                    try:
                        from .utils import save_safetensors_stream
                        print(f"💾 Saving merged checkpoint to: {output_path} (streaming)")
                        save_safetensors_stream(state_dict, output_path, metadata=merged_metadata)
                        print(f"✅ Merged checkpoint saved to: {output_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to save merged checkpoint: {e}")
                        if output_path is not None and output_path.exists():
                            try:
                                output_path.unlink()
                                print(f"   🗑️ Removed partial file: {output_path}")
                            except Exception as unlink_err:
                                print(f"   ⚠️ Could not remove partial file: {unlink_err}")
                        save_path_str = ""
                
                # Always load model for downstream use (save + preview)
                print("🧠 Loading merged checkpoint from dict into ComfyUI...")
                try:
                    model, clip, vae = load_state_dict_as_model_objects(
                        state_dict,
                        metadata=merged_metadata,
                        output_vae=True,
                        output_clip=True
                    )
                    print(f"   🕐 [t={time.time()-_T0:.1f}s] Model loading complete (save path)")
                except Exception as e:
                    print(f"⚠️ Failed to load state dict as model objects: {e}")
                    model, clip, vae = None, None, None
                
                del state_dict
                cleanup_memory()
            
            else:
                # ── save_trigger=False: load model, no disk save ──
                print("🧠 Loading merged checkpoint from direct dict...")
                save_path_str = ""
                
                if hasattr(writer, '_omitted_components') and writer._omitted_components:
                    print(f"   ⚠️ Components {writer._omitted_components} omitted (weight=0)")
                
                del writer
                cleanup_memory()
                
                # No checkpoint_data retention when not saving
                checkpoint_data = None
                
                try:
                    model, clip, vae = load_state_dict_as_model_objects(
                        state_dict,
                        metadata=merged_metadata,
                        output_vae=True,
                        output_clip=True
                    )
                    print(f"   🕐 [t={time.time()-_T0:.1f}s] Model loading complete (preview path)")
                except Exception as e:
                    print(f"⚠️ Failed to load state dict as model objects: {e}")
                    model, clip, vae = None, None, None
                
                del state_dict
                cleanup_memory()
        
        else:
            # ── Disk mode (RAM Guard fallback) ──
            save_path_str = str(output_path)
            print(f"✅ Merged checkpoint saved to: {save_path_str}")
            
            del writer
            cleanup_memory()
            
            # Always load model via lazy mapping from saved file
            print(f"📥 Loading merged checkpoint from file into ComfyUI...")
            from .engine.musubi_checkpoint_studio import MusubiCheckpointStudio
            
            # ── FP8 EXPORT: Inject _quantization_metadata for disk fallback mode ──
            # The file was written by weave() with survey() metadata. For FP8 output,
            # build fresh quantization metadata by scanning the file header via
            # the lazy mapping (reads header JSON, not tensor data).
            if _is_fp8_output:
                # Create a temp lazy mapping just for header scan, then overwrite
                _temp_lazy = MusubiCheckpointStudio._LazyCheckpointMapping(
                    output_path, merged_metadata
                )
                merged_metadata = _build_fp8_quantization_metadata(
                    _temp_lazy, merged_metadata, label="disk_fallback"
                )
                _temp_lazy.permanent_close()
                del _temp_lazy
            
            lazy_mapping = MusubiCheckpointStudio._LazyCheckpointMapping(
                output_path, merged_metadata
            )
            try:
                model, clip, vae = load_state_dict_as_model_objects(
                    lazy_mapping,
                    metadata=merged_metadata,
                    output_vae=True,
                    output_clip=True
                )
                print(f"   🕐 [t={time.time()-_T0:.1f}s] Model loading complete (disk fallback path)")
            except Exception as e:
                print(f"⚠️ Failed to load state dict as model objects: {e}")
                model, clip, vae = None, None, None
        checkpoint_data = None
    
    # ── Measure post-merge available RAM and print delta ──
        _post_avail_ram = get_available_ram()
        if _pre_avail_ram is not None and _post_avail_ram is not None:
            _delta_gb = (_pre_avail_ram - _post_avail_ram) / (1024**3)
            print(f"📊 RAM: Pre-output {_pre_avail_ram / (1024**3):.2f} GB → "
                  f"Post-merge {_post_avail_ram / (1024**3):.2f} GB "
                  f"(delta: {_delta_gb:+.2f} GB)")
        elif _post_avail_ram is not None:
            print(f"📊 Post-merge available RAM: {_post_avail_ram / (1024**3):.2f} GB")
        
        print(f"   🕐 [t={time.time()-_T0:.1f}s] merge_triple returning")
        
        return (model, clip, vae, checkpoint_data, save_path_str, forensic_report)
