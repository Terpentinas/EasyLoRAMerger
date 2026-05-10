"""
Easy LoRA Baker — ComfyUI Node Definition

Bakes a LoRA (or merged LoRA) into a full model checkpoint at the tensor level,
producing ComfyUI MODEL+CLIP+VAE objects directly in memory with automatic
RAM Guard fallback to temp disk if system memory is insufficient.

Features:
  - VAE passthrough & output
  - Per-component scaling (weight_unet, weight_te, weight_clip, weight_vae)
  - Auto-fallback: RAM Guard falls back to temp disk mode if memory is insufficient
  - Metadata modes: none / preserve_a / merge_basic
  - Expanded precision: auto / float32 / bfloat16 / float16 / fp8_e4m3fn / fp8_e5m2
"""

import torch
import json
from pathlib import Path

import folder_paths
import comfy.sd

from .utils import (
    load_checkpoint_with_metadata,
    load_lora_with_metadata,
    cleanup_memory,
    NodeCache,
    save_checkpoint_data_to_temp,
    save_safetensors_file,
    load_state_dict_as_model_objects,
    check_ram_guard,
)
from .config import PRECISION_OPTIONS, DEVICE_OPTIONS, DevicePrecisionConfig
from .engine.baking_processor import SmartBakingProcessor


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
            "required": {
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
                "precision": (PRECISION_OPTIONS, {"default": "auto"}),
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
        
        # Default return values
        save_path = ""
        model_out = None
        clip_out = None
        vae_out = None
        forensic_report = ""
        temp_path = None  # For preview mode temp file cleanup
        
        # Validate inputs
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
            
            ckpt_sd, ckpt_metadata = load_checkpoint_with_metadata(ckpt_path)
            print(f"   ✅ Loaded checkpoint: {len(ckpt_sd)} keys")
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
        try:
            dpc = DevicePrecisionConfig(device_type=device, precision=precision)
            processor = SmartBakingProcessor(device_precision=dpc)
            
            output_sd, metadata, forensic_report, _ = processor.bake(
                ckpt_sd=ckpt_sd,
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
            )
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
        
        if save_trigger:
            # === Save Mode: permanent disk save + lazy load from file ===
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
            
            # Load MODEL+CLIP+VAE lazily from the saved file
            print(f"\n📦 Loading baked checkpoint as MODEL+CLIP+VAE (lazy from file)...")
            try:
                from .engine.musubi_checkpoint_studio import MusubiCheckpointStudio
                lazy_mapping = MusubiCheckpointStudio._LazyCheckpointMapping(
                    output_path, metadata
                )
                model_out, clip_out, vae_out = load_state_dict_as_model_objects(
                    lazy_mapping, metadata=metadata,
                )
                print(f"   ✅ Loaded MODEL + CLIP + VAE from saved file (lazy, low RAM)")
            except Exception as e:
                print(f"   ⚠️ Failed to load baked checkpoint as model objects: {e}")
                print(f"   ℹ️  The checkpoint was saved successfully at: {save_path}")
                # model_out, clip_out, vae_out remain None
        
        else:
            # === Preview Mode (RAM with RAM Guard): in-memory load, auto-fallback to temp disk ===
            print("Preview mode — checking memory before in-memory load...")
            total_data_bytes = sum(
                t.numel() * t.element_size() for t in output_sd.values()
            )
            use_temp_fallback = not check_ram_guard(
                total_bytes=total_data_bytes,
                num_tensors=len(output_sd),
                batch_size=batch_size,
                label="baking preview",
            )
            
            if use_temp_fallback:
                # Fallback: save to temp file + lazy load from temp
                temp_path = save_checkpoint_data_to_temp(output_sd, "baked_preview", metadata=metadata)
                if temp_path is not None:
                    try:
                        print(f"   📁 Writing temp file: {temp_path}")
                        # Free in-memory state dict
                        cleanup_memory(output_sd)
                        # Lazy load from temp file
                        from .engine.musubi_checkpoint_studio import MusubiCheckpointStudio
                        lazy_mapping = MusubiCheckpointStudio._LazyCheckpointMapping(
                            temp_path, metadata
                        )
                        model_out, clip_out, vae_out = load_state_dict_as_model_objects(
                            lazy_mapping, metadata=metadata,
                        )
                        print(f"   ✅ Loaded MODEL + CLIP + VAE from temp file (RAM-safe preview)")
                    except Exception as e2:
                        print(f"   ⚠️ Temp fallback failed: {e2}")
                        if 'output_sd' not in locals() or output_sd is None:
                            error_msg = f"❌ Preview mode failed (temp fallback + memory exhausted): {e2}"
                            print(error_msg)
                            forensic_report = json.dumps({"error": error_msg}, indent=2)
                            return (None, None, None, "", forensic_report)
                        print(f"   ℹ️  Falling through to in-memory load attempt")
                        try:
                            model_out, clip_out, vae_out = load_state_dict_as_model_objects(
                                output_sd, metadata=metadata,
                            )
                            print(f"   ✅ Loaded MODEL + CLIP + VAE from baked state dict (in-memory fallback)")
                        except Exception as e:
                            error_msg = f"❌ In-memory load fallback also failed: {e}"
                            print(error_msg)
                            import traceback
                            traceback.print_exc()
                            forensic_report = json.dumps({"error": error_msg}, indent=2)
                            return (None, None, None, "", forensic_report)
            else:
                # Direct in-memory load from state dict
                try:
                    model_out, clip_out, vae_out = load_state_dict_as_model_objects(
                        output_sd, metadata=metadata,
                    )
                    print(f"   ✅ Loaded MODEL + CLIP + VAE from baked state dict (in-memory)")
                    # Free state dict after loading to recover memory
                    cleanup_memory(output_sd)
                except Exception as e:
                    error_msg = f"❌ In-memory model load failed: {e}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    forensic_report = json.dumps({"error": error_msg}, indent=2)
                    return (None, None, None, "", forensic_report)
            
            save_path = ""  # No permanent save path in preview mode
        
        # Cleanup: free temp file if used (preview modes)
        if not save_trigger and temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
                print(f"   🧹 Removed temp file: {temp_path}")
            except Exception:
                pass
        
        cleanup_memory()
        
        print("\n" + "=" * 60)
        print("✅ Easy LoRA Baker complete!")
        if save_trigger:
            print(f"   💾 Saved to: {save_path}")
            if model_out is not None:
                print(f"   📦 Loaded MODEL + CLIP + VAE from saved file (lazy)")
        else:
            print(f"   🧠 Preview mode (RAM with RAM Guard) — no disk save")
            if model_out is not None:
                print(f"   📦 Loaded baked MODEL + CLIP + VAE in memory")
            print(f"   ℹ️  Set save_trigger=True to save permanently to disk")
        print("=" * 60)
        
        return (model_out, clip_out, vae_out, save_path, forensic_report)
