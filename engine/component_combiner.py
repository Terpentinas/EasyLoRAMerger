"""
Easy Component Combiner — ComfyUI Node Definition

Reassembles separate UNet, CLIP, and VAE state dicts into a single full
checkpoint.  Pure orchestration of existing utilities — zero new infrastructure.

Design follows ``plans/working_principles.md``:
    P1 (Delete First)  — no new infrastructure, pure orchestration of existing utils
    P2 (No Temp Files)  — no disk I/O unless explicit save; dict merge in RAM
    P3 (One Pattern)    — uses existing save_safetensors_stream,
                          load_state_dict_as_model_objects
    P4 (Resist Guards)  — clean linear flow: merge → precision → load → save
    P5 (Simple Dispatch) — 2 top-level branches (data provided vs not), ≤5 lines each
"""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import folder_paths

from ..utils import (
    load_state_dict_as_model_objects,
    save_safetensors_stream,
    cleanup_memory,
    NodeCache,
)
from ..config import PRECISION_EXTENDED, DEVICE_OPTIONS, DevicePrecisionConfig
from .fp8_quantizer import dequant_fp8_tensor


# ── Precision conversion (same pattern as component_extractor.py) ──────
def _convert_precision(
    state_dict: Dict[str, torch.Tensor],
    target_dtype: torch.dtype,
    *,
    component_label: str = "",
) -> Dict[str, torch.Tensor]:
    """Convert all tensors in *state_dict* to *target_dtype*.

    Skips conversion if *target_dtype* is ``None`` (keep native).
    Dequantizes FP8 tensors using companion scale factors before
    the ``.to()`` call — prevents ``RuntimeError`` on fp8 dtypes.
    """
    if target_dtype is None:
        return state_dict

    converted = {}
    for key, tensor in state_dict.items():
        if tensor.dtype != target_dtype:
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # Dequant FP8 → target_dtype using companion scales from state_dict
                tensor = dequant_fp8_tensor(tensor, key, target_dtype, state_dict)
            else:
                tensor = tensor.to(target_dtype)
            converted[key] = tensor
        else:
            converted[key] = tensor

    if component_label:
        print(f"   🔧 {component_label}: converted {len(converted)} tensors → {target_dtype}")
    return converted


# ===================================================================
# Node class
# ===================================================================
class EasyComponentCombiner:
    """
    Reassemble separate UNet, CLIP, and VAE state dicts into a single
    full checkpoint.  Accepts CHECKPOINT data chained from other nodes.

    The reverse operation of ``EasyComponentExtractor`` — enables workflows
    like GGUF→full checkpoint conversion, component swapping, and
    mixed-precision checkpoint assembly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        default_folder = ""
        checkpoint_folders = folder_paths.get_folder_paths("checkpoints")
        if checkpoint_folders:
            default_folder = str(checkpoint_folders[0])

        return {
            "required": {
                # ── SECTION 1: INPUTS ───────────────────────────────────
                "combine_unet": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include UNet state dict in combined output",
                }),
                "combine_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include CLIP state dict in combined output",
                }),
                "combine_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include VAE state dict in combined output",
                }),

                # ── SECTION 2: HARDWARE ─────────────────────────────────
                "precision": (PRECISION_EXTENDED, {
                    "default": "auto",
                    "tooltip": "Output precision for combined checkpoint. "
                               "'auto' preserves the source tensors' native dtype.",
                }),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),

                # ── SECTION 3: OUTPUT ───────────────────────────────────
                "save_trigger": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save combined checkpoint as .safetensors file",
                }),
            },
            "optional": {
                # ── SECTION 1 (continued) ───────────────────────────────
                "unet_data": ("CHECKPOINT", {
                    "tooltip": "UNet state dict (chained from Component Extractor or other node)",
                }),
                "clip_data": ("CHECKPOINT", {
                    "tooltip": "CLIP/text-encoder state dict",
                }),
                "vae_data": ("CHECKPOINT", {
                    "tooltip": "VAE state dict",
                }),

                # ── SECTION 3 (continued) ───────────────────────────────
                "save_folder": ("STRING", {
                    "default": default_folder,
                    "multiline": False,
                    "tooltip": "Output folder for saved checkpoint "
                               "(leave empty for default checkpoints dir)",
                }),
                "filename_prefix": ("STRING", {
                    "default": "combined",
                    "multiline": False,
                    "tooltip": "Prefix for saved filename "
                               "(e.g. 'my_model' → my_model.safetensors)",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache implementation for ComfyUI change detection."""
        # If any data input is connected, always re-execute
        if (kwargs.get("unet_data") is not None
                or kwargs.get("clip_data") is not None
                or kwargs.get("vae_data") is not None):
            return float("nan")

        return NodeCache.is_changed(
            cls.__name__,
            combine_unet=kwargs.get("combine_unet", True),
            combine_clip=kwargs.get("combine_clip", True),
            combine_vae=kwargs.get("combine_vae", True),
            precision=kwargs.get("precision", "auto"),
            device=kwargs.get("device", "auto"),
            save_trigger=kwargs.get("save_trigger", False),
            filename_prefix=kwargs.get("filename_prefix", "combined"),
        )

    RETURN_TYPES = ("CHECKPOINT", "MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("checkpoint", "model", "clip", "vae", "output_path")
    FUNCTION = "combine"
    CATEGORY = "Checkpoint/Utils"
    OUTPUT_NODE = True

    def combine(
        self,
        combine_unet: bool = True,
        combine_clip: bool = True,
        combine_vae: bool = True,
        precision: str = "auto",
        device: str = "auto",
        save_trigger: bool = False,
        unet_data: Optional[Dict] = None,
        clip_data: Optional[Dict] = None,
        vae_data: Optional[Dict] = None,
        save_folder: str = "",
        filename_prefix: str = "combined",
        **kwargs,
    ):
        """
        Combine separate component state dicts into a single full checkpoint.

        Merges UNet + CLIP + VAE dicts, applies precision, loads ComfyUI
        model objects, and optionally saves to disk.

        Returns ``(checkpoint, model, clip, vae, output_path)``.
        """
        print("\n" + "=" * 60)
        print("🧩 Easy Component Combiner")
        print("=" * 60)
        start_time = time.time()

        # ══ Runtime precision guard — protect against stale workflow JSONs ══
        if precision not in PRECISION_EXTENDED:
            print(f"   ⚠️ Precision '{precision}' is no longer available, falling back to 'auto'")
            precision = "auto"

        # Default returns
        model_obj = clip_obj = vae_obj = None
        output_path_str = ""

        # ── Step 1: Merge component dicts (P2: in-RAM, no temp file) ──
        print(f"   📦 Inputs: UNet={'✅' if unet_data is not None else '❌'}, "
              f"CLIP={'✅' if clip_data is not None else '❌'}, "
              f"VAE={'✅' if vae_data is not None else '❌'}")

        combined_sd: Dict[str, torch.Tensor] = {}
        source_count = 0

        if unet_data is not None and combine_unet:
            # Handle tuple wrapping (ComfyUI CHECKPOINT type convention)
            sd = unet_data[0] if isinstance(unet_data, tuple) and len(unet_data) >= 1 else unet_data
            combined_sd.update(sd)
            source_count += 1
            print(f"   🧠 UNet: {len(sd)} tensors merged")

        if clip_data is not None and combine_clip:
            sd = clip_data[0] if isinstance(clip_data, tuple) and len(clip_data) >= 1 else clip_data
            # Check for key overlap
            overlap = set(combined_sd.keys()) & set(sd.keys())
            if overlap:
                print(f"   ⚠️ Key overlap detected in CLIP data: {len(overlap)} keys will be overwritten")
            combined_sd.update(sd)
            source_count += 1
            print(f"   📝 CLIP: {len(sd)} tensors merged")

        if vae_data is not None and combine_vae:
            sd = vae_data[0] if isinstance(vae_data, tuple) and len(vae_data) >= 1 else vae_data
            overlap = set(combined_sd.keys()) & set(sd.keys())
            if overlap:
                print(f"   ⚠️ Key overlap detected in VAE data: {len(overlap)} keys will be overwritten")
            combined_sd.update(sd)
            source_count += 1
            print(f"   🎨 VAE: {len(sd)} tensors merged")

        if source_count == 0 or not combined_sd:
            print("   ⚠️ No component data provided — nothing to combine")
            elapsed = time.time() - start_time
            print(f"\n⏱️  Combine complete: {elapsed:.2f}s (no-op)")
            print("=" * 60)
            return (None, None, None, None, "")

        print(f"   📊 Combined: {len(combined_sd)} total tensors from {source_count} component(s)")

        # ── Step 2: Resolve precision ─────────────────────────────────
        dpc = DevicePrecisionConfig(device_type=device, precision=precision)
        target_dtype = dpc.dtype
        print(f"   🎯 Precision: {precision} → {target_dtype}")
        print(f"   💻 Device: {dpc.device}")

        if target_dtype is not None:
            combined_sd = _convert_precision(
                combined_sd, target_dtype, component_label="Combined",
            )

        # ── Step 3: Save to disk BEFORE loading model objects ─────────
        # P2: direct save, no temp files. Save must happen BEFORE
        # load_state_dict_as_model_objects because ComfyUI's
        # load_state_dict_guess_config pops keys from the input dict
        # (see utils.py:1407). Same order as EasyComponentExtractor.
        if save_trigger and save_folder:
            print(f"\n💾 Saving combined checkpoint...")
            save_dir = Path(save_folder)
            save_dir.mkdir(parents=True, exist_ok=True)

            output_path = save_dir / f"{filename_prefix}.safetensors"
            try:
                save_safetensors_stream(combined_sd, output_path)
                output_path_str = str(output_path)
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ Combined checkpoint saved: {output_path} ({file_size_mb:.1f} MB)")
            except Exception as e:
                print(f"   ❌ Failed to save combined checkpoint: {e}")

        # ── Step 4: Load model objects ────────────────────────────────
        # Note: combined_sd may be empty after this step (ComfyUI pops keys).
        # That's fine — save already happened above.
        try:
            print(f"\n📦 Loading model objects...")
            model_obj, clip_obj, vae_obj = load_state_dict_as_model_objects(
                combined_sd,
                output_vae=(vae_data is not None and combine_vae),
                output_clip=(clip_data is not None and combine_clip),
            )
            loaded_parts = []
            if model_obj is not None:
                loaded_parts.append("MODEL")
            if clip_obj is not None:
                loaded_parts.append("CLIP")
            if vae_obj is not None:
                loaded_parts.append("VAE")
            if loaded_parts:
                print(f"   ✅ Model objects loaded: {', '.join(loaded_parts)}")
            else:
                print("   ⚠️ No model objects could be loaded from combined checkpoint")
        except Exception as e:
            print(f"   ❌ Failed to load model objects from combined checkpoint: {e}")
            import traceback
            traceback.print_exc()
            model_obj = clip_obj = vae_obj = None

        # ── Cleanup ────────────────────────────────────────────────────
        cleanup_memory(skip_gc=True)

        elapsed = time.time() - start_time
        print(f"\n⏱️  Combine complete: {elapsed:.2f}s")
        print(f"   📦 MODEL:     {'✅ loaded' if model_obj is not None else '❌ none'}")
        print(f"   📦 CLIP:      {'✅ loaded' if clip_obj is not None else '❌ none'}")
        print(f"   📦 VAE:       {'✅ loaded' if vae_obj is not None else '❌ none'}")
        print(f"   📁 Saved to:  {output_path_str if output_path_str else '(not saved)'}")
        print("=" * 60)

        return (combined_sd, model_obj, clip_obj, vae_obj, output_path_str)
