"""
Easy LoRA Baker — ComfyUI Node Definition

Bakes a LoRA (or merged LoRA) into a full model checkpoint at the tensor level,
producing ComfyUI MODEL+CLIP+VAE objects directly.  Lazy assembly via mmap
means near-zero RAM overhead for preserved keys — no temp files needed.

Features:
  - VAE passthrough & output
  - Per-component scaling (weight_unet, weight_te, weight_clip, weight_vae)
  - Lazy assembly: baked keys in write cache, preserved keys mmap'd from file
  - Metadata modes: none / preserve_a / merge_basic
  - Expanded precision: auto / float32 / bfloat16 / float16 / fp8_e4m3fn / fp8_e5m2
"""

import torch
import json
import time
_T0 = time.time()
from pathlib import Path
from collections import Counter

import folder_paths
import comfy.sd

from .utils import (
    load_checkpoint_with_metadata,
    load_lora_with_metadata,
    read_safetensors_header_only,
    cleanup_memory,
    NodeCache,
    save_safetensors_file,
    save_safetensors_stream,
    load_state_dict_as_model_objects,
    get_available_ram,
)
from .config import PRECISION_EXTENDED, DEVICE_OPTIONS, DevicePrecisionConfig
from .engine.baking_processor import SmartBakingProcessor


# ===================================================================
# FP8 metadata helpers — imported from shared fp8_quantizer module
# (One Pattern Per Concept — working_principles.md)
# ===================================================================
from .engine.fp8_quantizer import (
    build_fp8_quantization_metadata as _build_fp8_quantization_metadata,
    strip_input_scale_keys as _strip_input_scale_keys,
)


# ===================================================================
# Auto-precision helper: detect checkpoint's native weight dtype
# ===================================================================
def _detect_native_ckpt_dtype(ckpt_header: dict) -> "torch.dtype | None":
    """Detect the dominant weight dtype from the safetensors header.

    Scans all tensor entries in the header, skipping scale-factor keys
    (.weight_scale, .input_scale), and finds the most common floating-point
    dtype.  Only float types (float16, bfloat16, float32, float8_e4m3fn,
    float8_e5m2) are considered — int/bool tensors are excluded since they
    are not precision targets for weight computation.

    Returns:
        The most common torch.dtype among float weight tensors, or None
        if no float tensors are found.
    """
    from .engine.key_utils import SAFETENSORS_DTYPE_MAP as _SAFETENSORS_DTYPE_MAP

    _FLOAT_DTYPES = {
        torch.float16, torch.bfloat16, torch.float32,
        torch.float8_e4m3fn, torch.float8_e5m2,
    }

    dtype_counter: Counter = Counter()
    for key, info in ckpt_header.items():
        if not isinstance(info, dict):
            continue
        if key.endswith((".weight_scale", ".input_scale", ".comfy_quant")):
            continue
        dtype_str = info.get("dtype", "")
        if not dtype_str:
            continue
        dtype = _SAFETENSORS_DTYPE_MAP.get(dtype_str)
        if dtype is not None and dtype in _FLOAT_DTYPES:
            dtype_counter[dtype] += 1

    if dtype_counter:
        return dtype_counter.most_common(1)[0][0]
    return None


class SmartModelBaker:
    """
    ComfyUI node: Bakes a LoRA into a checkpoint at the tensor level.
    
    Inputs:
        - checkpoint: Pick from available checkpoints dropdown
        - lora_data: LORA type from Triple Merger or LoRA-Only Merger (optional)
        - lora_name: Alternate single LoRA from dropdown (optional)
        - baking_method: linear / impact_weighted / orthogonal
        - strength: LoRA strength 0.0–2.0
        - weight_unet: Per-component weight for U-Net keys (0.0–2.0)
        - weight_te: Per-component weight for Text Encoder keys (0.0–2.0)
        - weight_clip: Per-component weight for CLIP Vision keys (0.0–2.0)
        - weight_vae: Per-component weight for VAE keys (0.0–2.0)
        - metadata_mode: "none" / "preserve_a" / "merge_basic"
        - precision: auto / float32 / bfloat16 / float16 / fp8_e4m3fn / fp8_e5m2
        - save_trigger: Toggle to bake and save to disk permanently. When True, MODEL+CLIP+VAE are lazy-loaded from the saved file for low RAM usage. When False, baked result is kept in memory with automatic RAM Guard fallback.
        - save_folder: Custom output folder
        - filename: Output filename (.safetensors added automatically)
        - device: auto / cuda / cpu
    
    Outputs:
        - model: MODEL
        - clip: CLIP
        - vae: VAE
        - output: STRING path to saved baked checkpoint (empty in preview mode)
        - forensic_report: STRING Human-readable forensic report (Lora Studio style)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {},
            "optional": {
                # ── SECTION 1: INPUTS ───────────────────────────────────
                "checkpoint": (checkpoints,),
                # Wired LORA overrides dropdown — when lora_data is connected, lora_name is ignored
                "lora_data": ("LORA", {
                    "tooltip": "LORA output from Triple Merger or LoRA-Only Merger. When connected, overrides lora_name dropdown."
                }),
                "lora_name": (["None"] + loras, {
                    "tooltip": "Pick a single LoRA from dropdown. Used only when lora_data is not connected."
                }),

                # ── SECTION 3: SETTINGS ─────────────────────────────────
                "baking_method": (
                    ["linear", "impact_weighted", "orthogonal"],
                    {"default": "linear"},
                ),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA strength multiplier"
                }),
                "weight_unet": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Per-component weight for U-Net (diffusion model) keys"
                }),
                "weight_te": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Per-component weight for Text Encoder keys"
                }),
                "weight_clip": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Per-component weight for CLIP Vision keys"
                }),
                "weight_vae": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Per-component weight for VAE keys"
                }),

                # ── SECTION 5: HARDWARE ─────────────────────────────────
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "precision": (PRECISION_EXTENDED, {"default": "auto"}),
                "batch_size": ("INT", {
                    "default": 64, "min": 1, "max": 256, "step": 8,
                    "tooltip": "Number of keys to process per batch. Larger = faster but more VRAM. "
                               "memory_guard() runs between batches to prevent OOM. "
                               "DeviceManager.suggest_batch_size() can auto-tune based on VRAM."
                }),

                # ── SECTION 6: OUTPUT ───────────────────────────────────
                "save_trigger": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True: bake LoRA into checkpoint AND save as permanent .safetensors file (low RAM: saves first, then lazy-loads from file). When False: bake in-memory only (preview mode) with automatic RAM Guard fallback — connect MODEL+CLIP+VAE outputs downstream to test results before committing."
                }),
                "filename": ("STRING", {
                    "default": "baked_checkpoint",
                    "multiline": False,
                    "tooltip": "Output filename (.safetensors added automatically in disk mode)"
                }),
                "save_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom output folder (leave empty for default checkpoints dir)"
                }),
                "metadata_mode": (
                    ["preserve_a", "preserve_b", "none", "merge_basic"],
                    {"default": "preserve_a",
                     "tooltip": "'none'=baking only, 'preserve_a'=original priority, 'preserve_b'=second-source (fallback to A), 'merge_basic'=baking priority"}
                ),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "output_path", "forensic_report")
    FUNCTION = "bake"
    CATEGORY = "LoRA/Baking"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache implementation for ComfyUI change detection."""
        # Inline data guard: wired lora_data can't be cheaply hashed
        if kwargs.get('lora_data') is not None:
            return float("nan")

        return NodeCache.is_changed(
            cls.__name__,
            checkpoint=kwargs.get('checkpoint', ''),
            lora_name=kwargs.get('lora_name', 'None'),
            baking_method=kwargs.get('baking_method', 'linear'),
            strength=kwargs.get('strength', 1.0),
            weight_unet=kwargs.get('weight_unet', 1.0),
            weight_te=kwargs.get('weight_te', 1.0),
            weight_clip=kwargs.get('weight_clip', 1.0),
            weight_vae=kwargs.get('weight_vae', 1.0),
            metadata_mode=kwargs.get('metadata_mode', 'preserve_a'),
            save_trigger=kwargs.get('save_trigger', False),
            device=kwargs.get('device', 'auto'),
            precision=kwargs.get('precision', 'auto'),
            batch_size=kwargs.get('batch_size', 64),
        )
    
    def bake(
        self,
        checkpoint: str,
        baking_method: str = "linear",
        strength: float = 1.0,
        weight_unet: float = 1.0,
        weight_te: float = 1.0,
        weight_clip: float = 1.0,
        weight_vae: float = 1.0,
        batch_size: int = 64,
        metadata_mode: str = "preserve_a",
        save_trigger: bool = False,
        lora_data=None,
        lora_name: str = "None",
        save_folder: str = "",
        filename: str = "baked_checkpoint",
        device: str = "auto",
        precision: str = "auto",
        **kwargs,
    ):
        """
        Execute the baking pipeline with per-component scaling, RAM mode with
        automatic RAM Guard fallback, VAE output, and metadata control.
        
        Returns (model, clip, vae, output, forensic_report).
        """
        print("\n" + "=" * 60)
        print("🔥 Easy LoRA Baker — Baking LoRA into Checkpoint")
        print("=" * 60)
        print(f"   Checkpoint: {checkpoint}")
        print(f"   Method: {baking_method}, Strength: {strength}")
        print(f"   🧱 Weights — UNet:{weight_unet} TE:{weight_te} CLIP:{weight_clip} VAE:{weight_vae}")
        print(f"   💾 Preview: RAM mode with RAM Guard, Metadata: {metadata_mode}")
        
        # ══ Runtime precision guard — protect against stale workflow JSONs ══
        if precision not in PRECISION_EXTENDED:
            print(f"   ⚠️ Precision '{precision}' is no longer available, falling back to 'auto'")
            precision = "auto"
        
        # Default return values
        save_path = ""
        model_out = None
        clip_out = None
        vae_out = None
        forensic_report = ""
        # Validate inputs
        if not checkpoint:
            error_msg = "❌ No checkpoint selected — please choose a checkpoint from the dropdown"
            print(error_msg)
            forensic_report = json.dumps({"error": error_msg}, indent=2)
            return (model_out, clip_out, vae_out, save_path, forensic_report)

        if lora_data is None and (lora_name == "None" or not lora_name):
            error_msg = "❌ Either lora_data or lora_name must be provided"
            print(error_msg)
            forensic_report = json.dumps({"error": error_msg}, indent=2)
            return (model_out, clip_out, vae_out, save_path, forensic_report)
        
        # ------------------------------------------------------------------
        # Step 1: Load checkpoint
        # ------------------------------------------------------------------
        print("\n📥 Loading checkpoint...")
        try:
            ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
            if not ckpt_path:
                # Try direct path
                ckpt_path = checkpoint
            ckpt_path = Path(ckpt_path)
            
            # RAM Guard (informational): Log file size for diagnostics.
            # With mmap-based lazy loading, the checkpoint file is never fully
            # loaded into RAM. Only individual tensors are loaded on demand via
            # safe_open (mmap). A 17 GiB checkpoint uses ~50 KB for the header.
            ckpt_file_size = ckpt_path.stat().st_size
            print(f"   📦 Checkpoint file: {ckpt_file_size / (1024**3):.2f} GB on disk")
            
            # Use mmap-based lazy loading: read only the safetensors header JSON
            # (~50 KB), then wrap the file in a dict-like lazy mapping.
            # Tensors are loaded on demand when ckpt_sd[key] is first accessed.
            ckpt_header, ckpt_metadata = read_safetensors_header_only(ckpt_path)

            # 🔥 FP8 DETECTION: Check if any checkpoint tensor uses FP8 dtype.
            # FP8 e4m3fn has only 3 mantissa bits — step size ~0.125 at magnitude
            # 1.0, which is 64× coarser than a typical LoRA delta (~0.002).
            # We dequantize to bfloat16 for all baking computation, then
            # quantize once at final output (if the user selected fp8 precision).
            # Safetensors header uses short dtype strings: "F8_E4M3" for fp8_e4m3fn,
            # "F8_E5M2" for fp8_e5m2 (see _SAFETENSORS_DTYPE_MAP in musubi_checkpoint_studio.py).
            # Check for the "F8_" prefix which is unique to FP8 dtypes.
            detected_fp8 = any(
                info.get('dtype', '').startswith('F8_')
                for info in ckpt_header.values()
                if isinstance(info, dict)
            )
            if detected_fp8:
                print(f"   🚀 FP8 checkpoint detected — dequantizing to bfloat16 for baking precision")
                # Strip stale quantization metadata that tells ComfyUI to use
                # its slow mixed-precision path.  Since we dequantize to bf16,
                # this metadata is stale and would trigger incorrect behavior.
                # Secondary cleanup also happens in _build_metadata (Fix 6).
                fp8_meta_keys = [k for k in list(ckpt_metadata.keys())
                                 if 'quantization' in k.lower()]
                for k in fp8_meta_keys:
                    del ckpt_metadata[k]
                if fp8_meta_keys:
                    print(f"   🧹 Stripped {len(fp8_meta_keys)} quantization metadata keys")

            from .engine.musubi_checkpoint_studio import MusubiCheckpointStudio
            # 🔥 Pass target_dtype=bf16 for FP8 checkpoints so that __getitem__
            # applies per-channel scale factors during fp8→bf16 dequantization.
            # Without this, _safe_bake_add receives raw fp8 tensors and its naive
            # base.to(dtype=compute_dtype) cast ignores the per-channel scale
            # factors, producing completely wrong weight values (pure noise).
            ckpt_sd = MusubiCheckpointStudio._LazyCheckpointMapping(
                ckpt_path, ckpt_metadata,
                target_dtype=torch.bfloat16 if detected_fp8 else None,
            )
            print(f"   ✅ Lazily mapped checkpoint: {len(ckpt_header)} keys via mmap (0 bytes loaded)")
        except Exception as e:
            error_msg = f"❌ Failed to load checkpoint: {e}"
            print(error_msg)
            forensic_report = json.dumps({"error": error_msg}, indent=2)
            return (model_out, clip_out, vae_out, save_path, forensic_report)
        
        # ------------------------------------------------------------------
        # Step 2: Load LoRA
        # ------------------------------------------------------------------
        print("\n📥 Loading LoRA...")
        lora_sd = None
        lora_source = "unknown"
        
        try:
            if lora_data is not None:
                # LORA type from Triple Merger or LoRA-Only Merger
                # lora_data is a tuple: (state_dict, weight, weight)
                if isinstance(lora_data, tuple) and len(lora_data) >= 1:
                    lora_sd = lora_data[0]
                elif isinstance(lora_data, dict):
                    lora_sd = lora_data
                lora_source = "lora_data (from merger)"
                print(f"   ✅ Using lora_data from upstream node ({len(lora_sd)} tensors)")
                
                # Apply ComfyUI's convert_lora for non-standard LoRA formats
                # (BFL control, WanFun, USO — standard SD1.5 Kohya passes through)
                # Must match ComfyUI's ordering: convert_lora BEFORE key matching
                # See: comfy/sd.py:85 → comfy.lora_convert.convert_lora(lora)
                try:
                    from comfy.lora_convert import convert_lora
                    lora_sd = convert_lora(lora_sd)
                    print(f"   🔄 Applied convert_lora ({len(lora_sd)} tensors)")
                except ImportError:
                    pass  # Fallback: comfy not available in this context
            
            if lora_sd is None and lora_name != "None" and lora_name:
                # Load from dropdown pick
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path:
                    lora_sd, _ = load_lora_with_metadata(Path(lora_path))
                    lora_source = lora_name
                    print(f"   ✅ Loaded LoRA: {lora_name} ({len(lora_sd)} tensors)")
            
            if lora_sd is None:
                error_msg = "❌ Failed to load LoRA data"
                print(error_msg)
                forensic_report = json.dumps({"error": error_msg}, indent=2)
                return (model_out, clip_out, vae_out, save_path, forensic_report)
            
            # Filter out non-tensor metadata from LoRA state dict
            lora_sd = {k: v for k, v in lora_sd.items() if isinstance(v, torch.Tensor)}
            
        except Exception as e:
            error_msg = f"❌ Failed to load LoRA: {e}"
            print(error_msg)
            forensic_report = json.dumps({"error": error_msg}, indent=2)
            return (model_out, clip_out, vae_out, save_path, forensic_report)
        
        # ------------------------------------------------------------------
        # Step 3-9: Run baking pipeline
        # ------------------------------------------------------------------
        print("\n🔨 Running baking pipeline...")

        # 🎯 Auto-precision: detect checkpoint native dtype and use it
        if precision == "auto":
            native_dtype = _detect_native_ckpt_dtype(ckpt_header)
            if native_dtype is not None:
                _DTYPE_TO_PRECISION = {
                    torch.float8_e4m3fn: "fp8_e4m3fn",
                    torch.float8_e5m2:   "fp8_e5m2",
                    torch.bfloat16:      "bfloat16",
                    torch.float16:       "float16",
                    torch.float32:       "float32",
                }
                mapped = _DTYPE_TO_PRECISION.get(native_dtype)
                if mapped:
                    precision = mapped
                    print(f"   🎯 Auto precision: detected {native_dtype} → using {mapped}")

        try:
            dpc = DevicePrecisionConfig(device_type=device, precision=precision)
            processor = SmartBakingProcessor(device_precision=dpc)
            
            output_sd, metadata, forensic_report, _ = processor.bake(
                ckpt_sd=ckpt_sd,
                ckpt_header=ckpt_header,
                lora_sd=lora_sd,
                baking_method=baking_method,
                batch_size=batch_size,
                strength=strength,
                lora_source=lora_source,
                checkpoint_name=checkpoint,
                original_metadata=ckpt_metadata,
                weight_unet=weight_unet,
                weight_te=weight_te,
                weight_clip=weight_clip,
                weight_vae=weight_vae,
                metadata_mode=metadata_mode,
                detected_fp8=detected_fp8,
            )
            print(f"   🕐 [t={time.time()-_T0:.1f}s] Baking pipeline returned to baker_node — output_sd type: {type(output_sd).__name__}")
        except Exception as e:
            error_msg = f"❌ Baking pipeline failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            forensic_report = json.dumps({"error": error_msg}, indent=2)
            return (model_out, clip_out, vae_out, save_path, forensic_report)
        
        # ------------------------------------------------------------------
        # Step 10-11: Output as MODEL+CLIP+VAE with conditional save
        # ------------------------------------------------------------------
        print(f"\n💾 save_trigger={save_trigger}")

        # ── Measure pre-load RAM for post-load delta ──
        _pre_load_ram = get_available_ram()

        if save_trigger:
            # === Save Mode: permanent disk save + lazy load from file ===
            from .engine.musubi_checkpoint_studio import MusubiCheckpointStudio

            # 🎯 FP8 SAVE: Build _quantization_metadata for MixedPrecisionOps.
            # This runs before the if-else dispatch so ALL save paths benefit.
            metadata = _build_fp8_quantization_metadata(output_sd, metadata, label="save")

            if isinstance(output_sd, MusubiCheckpointStudio._LazyCheckpointMapping):
                # ── Lazy assembly mode (Fix 1): baked keys in write cache ──
                # output_sd.filepath points to the ORIGINAL checkpoint, NOT the
                # baked output.  Must materialize all keys to a new file.
                print("Saving baked checkpoint to disk (lazy assembly — materializing to file)...")
                from .utils import _get_output_path
                output_path = _get_output_path(
                    save_folder, filename, "baked_checkpoint", "checkpoints"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Materialize all keys (write cache + lazy reads from original file)
                all_tensors = dict(output_sd.items())
                # 🎯 FP8 SAVE: Strip .input_scale (MixedPrecisionOps only needs .weight_scale)
                _strip_input_scale_keys(all_tensors)
                save_safetensors_stream(all_tensors, output_path, metadata=metadata)
                del all_tensors
                save_path = str(output_path)
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ Saved: {output_path} ({file_size_mb:.1f} MB)")
                # Fix 2: Close the lazy mapping permanently — no more reads from original
                output_sd.permanent_close()
                cleanup_memory()

            else:
                # ── Dict mode: use existing save_safetensors_file path ──
                print("Saving baked checkpoint to disk permanently...")
                output_path, save_path = save_safetensors_file(
                    output_sd, save_folder, filename, metadata=metadata,
                    folder_type="checkpoints", default_name="baked_checkpoint",
                )
                if output_path is None:
                    # save_safetensors_file already printed the error
                    forensic_report = json.dumps({"error": save_path}, indent=2)
                    return (None, None, None, save_path, forensic_report)
                # Free in-memory state dict BEFORE loading model objects (low RAM)
                print("   🧹 Freeing in-memory state dict...")
                cleanup_memory(output_sd)

            # ── Common model loading for all save modes (Fix 5) ──
            print(f"\n📦 Loading baked checkpoint as MODEL+CLIP+VAE (lazy from file)...")
            lazy_mapping = None
            try:
                lazy_mapping = MusubiCheckpointStudio._LazyCheckpointMapping(
                    output_path, metadata,
                    target_dtype=None,  # 🎯 MixedPrecisionOps handles dequant via _quantization_metadata
                )
                model_out, clip_out, vae_out = load_state_dict_as_model_objects(
                    lazy_mapping, metadata=metadata,
                )
                print(f"   ✅ Loaded MODEL + CLIP + VAE from saved file (lazy, low RAM)")
            except Exception as e:
                print(f"   ⚠️ Failed to load baked checkpoint as model objects: {e}")
                print(f"   ℹ️  The checkpoint was saved successfully at: {save_path}")
            finally:
                # Fix 5: Close the mapping permanently after model objects are built.
                # This prevents __del__ from reopening the file handle during GC,
                # eliminating post-prompt file I/O on stale lazy mappings.
                if lazy_mapping is not None:
                    lazy_mapping.permanent_close()
        
        else:
            # === Preview Mode: load as MODEL+CLIP+VAE directly ===
            # output_sd is a lazy mapping (from _assemble_output_lazy) or a real dict
            # (from _assemble_output for non-lazy mode). Lazy mappings have near-zero
            # RAM overhead — no temp file fallback needed.
            print("Preview mode — loading as MODEL+CLIP+VAE...")
            try:
                # 🎯 FP8 PREVIEW: Use MixedPrecisionOps instead of eager dequant.
                # Unified detection: works for BOTH lazy mappings (has _user_fp8_mode flag)
                # and real dicts (check for FP8 tensor dtype).
                _has_fp8 = getattr(output_sd, '_user_fp8_mode', False) or any(
                    hasattr(v, 'dtype') and v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                    for v in output_sd.values()
                )
                if _has_fp8:
                    # Step 1: Pop .input_scale keys (MixedPrecisionOps
                    # only needs .weight_scale — .input_scale would appear as
                    # 'unet unexpected' keys)
                    _strip_input_scale_keys(output_sd)

                    # Step 2: Build _quantization_metadata from the mapping.
                    metadata = _build_fp8_quantization_metadata(output_sd, metadata, label="preview")

                print(f"   🕐 [t={time.time()-_T0:.1f}s] Starting model loading — load_state_dict_as_model_objects")
                model_out, clip_out, vae_out = load_state_dict_as_model_objects(
                    output_sd, metadata=metadata,
                )
                mode_label = "lazy (mmap)" if hasattr(output_sd, 'filepath') else "in-memory"
                print(f"   ✅ Loaded MODEL + CLIP + VAE from baked state dict ({mode_label})")
                print(f"   🕐 [t={time.time()-_T0:.1f}s] Model loading complete — MODEL+CLIP+VAE ready")
                # Free state dict after loading to recover memory
                cleanup_memory(output_sd)
                # Close lazy mapping to prevent stale file handles
                if hasattr(output_sd, 'filepath'):
                    output_sd.permanent_close()
            except Exception as e:
                error_msg = f"❌ In-memory model load failed: {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                forensic_report = json.dumps({"error": error_msg}, indent=2)
                return (None, None, None, "", forensic_report)

            save_path = ""  # No permanent save path in preview mode
        
        # SSD Fix: skip gc.collect() at end of bake — GC already ran naturally
        # during the pipeline.  Calling gc.collect() here triggers __del__ on
        # stale objects at unpredictable times, causing unexpected file I/O.
        cleanup_memory(skip_gc=True)

        # ── Measure post-load RAM and print delta ──
        _post_load_ram = get_available_ram()
        if _pre_load_ram is not None and _post_load_ram is not None:
            _delta_gb = (_pre_load_ram - _post_load_ram) / (1024**3)
            print(f"📊 RAM: Pre-load {_pre_load_ram / (1024**3):.2f} GB → "
                  f"Post-load {_post_load_ram / (1024**3):.2f} GB "
                  f"(delta: {_delta_gb:+.2f} GB)")
        elif _post_load_ram is not None:
            print(f"📊 Post-load available RAM: {_post_load_ram / (1024**3):.2f} GB")

        print("\n" + "=" * 60)
        print("✅ Easy LoRA Baker complete!")
        if save_trigger:
            print(f"   💾 Saved to: {save_path}")
            if model_out is not None:
                print(f"   📦 Loaded MODEL + CLIP + VAE from saved file (lazy)")
        else:
            print(f"   🧠 Preview mode — no disk save")
            if model_out is not None:
                print(f"   📦 Loaded baked MODEL + CLIP + VAE in memory")
            print(f"   ℹ️  Set save_trigger=True to save permanently to disk")
        print("=" * 60)
        
        print(f"   🕐 [t={time.time()-_T0:.1f}s] baker_node.bake returning")
        return (model_out, clip_out, vae_out, save_path, forensic_report)
