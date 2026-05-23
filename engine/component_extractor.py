"""
Easy Component Extractor — ComfyUI Node Definition

Extracts CLIP (text encoder), VAE, and UNet/diffusion-model components from
any checkpoint with precision options.  Supports direct file loading and
chained CHECKPOINT data.

Design follows ``plans/working_principles.md``:
    P1 (Delete First)  — zero new infrastructure, pure orchestration of existing utils
    P2 (No Temp Files)  — no disk I/O unless explicit save; filtered lazy views otherwise
    P3 (One Pattern)    — uses existing _LazyCheckpointMapping, save_safetensors_stream,
                          load_state_dict_as_model_objects
    P4 (Resist Guards)  — clean linear flow: filter → precision → save → load full mapping
    P5 (Simple Dispatch) — 2 top-level branches, each ≤5 lines
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import folder_paths
import comfy.sd

from ..utils import (
    load_state_dict_as_model_objects,
    read_safetensors_header_only,
    save_safetensors_stream,
    cleanup_memory,
    NodeCache,
    get_combined_model_list,
    resolve_model_path,
)
from ..config import PRECISION_EXTENDED, DEVICE_OPTIONS, DevicePrecisionConfig
from .key_utils import categorize_key
from .fp8_quantizer import dequant_fp8_tensor

import comfy.utils  # for state_dict_prefix_replace, clip_text_transformers_convert


# ── Component filter ──────────────────────────────────────────────────
# Builds a materialised dict containing only keys that match a given
# component category.  Works for both lazy mappings and plain dicts.
def _filter_component(
    source,
    *,
    category: str,
    header: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """Extract a sub-dict of tensors whose ``categorize_key()`` returns *category*.

    Args:
        source: A dict-like (lazy mapping or plain dict) of tensor keys.
        category: One of ``'te'`` (CLIP), ``'vae'``, ``'unet'``, ``'clip'``.
        header: Optional safetensors header dict (used only for logging key counts).

    Returns:
        Filtered dict containing only the matching tensors (materialised).
    """
    filtered = {}
    for key in source.keys():
        comp = categorize_key(key)
        if comp == category:
            filtered[key] = source[key]

    # Log summary
    component_label = category.upper()
    hdr_note = ""
    if header is not None:
        total_in_header = sum(
            1 for k in header if isinstance(header.get(k), dict) and categorize_key(k) == category
        )
        hdr_note = f" ({total_in_header} in header)"
    print(f"   🔍 {component_label}: {len(filtered)} tensors loaded{hdr_note}")
    return filtered


# ── Precision conversion ──────────────────────────────────────────────
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


# ── CLIP key transformation for standalone save ────────────────────────
def _prepare_clip_for_save(
    clip_data: Dict[str, torch.Tensor],
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    """Transform CLIP state dict keys for standalone loading via CLIP Loader.

    For **SDXL dual CLIP** (keys with ``conditioner.embedders.0/1.*`` prefix),
    applies the same transformation as
    ``comfy.supported_models.SDXL.process_clip_state_dict()`` and then splits
    into separate CLIP-L / CLIP-G dicts with **unprefixed** keys suitable for
    ``load_text_encoder_state_dicts()``.

    For **SD1.5** (keys with ``cond_stage_model.`` prefix), strips the prefix
    and adds ``text_model.`` where needed so that ``SD1ClipModel.load_sd()``
    can find matching parameters.

    For other architectures the data is returned as-is (single dict).

    Returns:
        ``(clip_primary, clip_secondary)`` where *clip_secondary* is ``None``
        for single-encoder checkpoints.  Each dict has unprefixed keys like
        ``text_model.encoder.layers.0.self_attn.k_proj.weight``.
    """
    # ── SDXL dual CLIP ──────────────────────────────────────────────
    has_sdxl_clip_l = any(
        k.startswith("conditioner.embedders.0.transformer.text_model")
        for k in clip_data
    )
    has_sdxl_clip_g = any(
        k.startswith("conditioner.embedders.1.model") for k in clip_data
    )

    if has_sdxl_clip_l or has_sdxl_clip_g:
        print("   🔄 Detected SDXL dual CLIP — transforming keys for standalone save")

        # Same prefix replacements as SDXL.process_clip_state_dict
        replace_prefix = {
            "conditioner.embedders.0.transformer.text_model":
                "clip_l.transformer.text_model",
            "conditioner.embedders.1.model.":
                "clip_g.",
        }
        transformed = comfy.utils.state_dict_prefix_replace(
            dict(clip_data), replace_prefix, filter_keys=True,
        )
        # Convert CLIP-G from OpenAI resblocks format to transformers encoder format
        transformed = comfy.utils.clip_text_transformers_convert(
            transformed, "clip_g.", "clip_g.transformer.",
        )

        clip_l_data: Dict[str, torch.Tensor] = {}
        clip_g_data: Dict[str, torch.Tensor] = {}
        for key, tensor in transformed.items():
            if key.startswith("clip_l.transformer."):
                # clip_l.transformer.text_model.xxx → text_model.xxx
                new_key = key[len("clip_l.transformer."):]
                clip_l_data[new_key] = tensor
            elif key.startswith("clip_g.transformer."):
                # clip_g.transformer.xxx → xxx  (for text_model.* / text_projection.weight)
                new_key = key[len("clip_g.transformer."):]
                clip_g_data[new_key] = tensor
            else:
                # Keys like clip_g.logit_scale are silently dropped as they are
                # not needed for text encoding
                pass

        if not clip_l_data:
            clip_l_data = None
        if not clip_g_data:
            clip_g_data = None

        print(f"   📦 CLIP-L: {len(clip_l_data) if clip_l_data else 0} keys, "
              f"CLIP-G: {len(clip_g_data) if clip_g_data else 0} keys")
        return (clip_l_data, clip_g_data)

    # ── SD1.5 single CLIP ───────────────────────────────────────────
    has_sd15 = any(k.startswith("cond_stage_model.") for k in clip_data)
    if has_sd15:
        print("   🔄 Detected SD1.5 CLIP — transforming keys for standalone save")

        # Same as SD15.process_clip_state_dict: add text_model. then replace prefix
        out = dict(clip_data)
        for k in list(out.keys()):
            if k.startswith("cond_stage_model.transformer.") and \
               not k.startswith("cond_stage_model.transformer.text_model."):
                new_k = k.replace(
                    "cond_stage_model.transformer.",
                    "cond_stage_model.transformer.text_model.",
                )
                out[new_k] = out.pop(k)

        replace_prefix = {"cond_stage_model.": "clip_l."}
        out = comfy.utils.state_dict_prefix_replace(
            out, replace_prefix, filter_keys=True,
        )

        # Strip clip_l.transformer. → text_model.*
        result: Dict[str, torch.Tensor] = {}
        for key, tensor in out.items():
            if key.startswith("clip_l.transformer."):
                new_key = key[len("clip_l.transformer."):]
                result[new_key] = tensor
            elif key.startswith("clip_l."):
                # e.g. clip_l.transformer.text_model.* (already has text_model.)
                new_key = key[len("clip_l."):]
                result[new_key] = tensor
            else:
                print(f"   ⚠️ Unexpected key after SD1.5 CLIP transform: {key}")

        print(f"   📦 CLIP: {len(result)} keys")
        return (result if result else None, None)

    # ── Other architectures (Flux T5, etc.) — return as-is ──────────
    return (clip_data, None)


# ===================================================================
# Node class
# ===================================================================
class EasyComponentExtractor:
    """
    Extract CLIP (text encoder), VAE, and UNet/diffusion-model from any
    checkpoint as separate ComfyUI objects, with configurable precision
    and optional save-to-disk.

    Works for any architecture (SDXL, Flux, SD1.5, Anima, etc.) — key
    categorisation is handled by ``engine.key_utils.categorize_key()``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_combined_model_list()
        default_folder = ""
        checkpoint_folders = folder_paths.get_folder_paths("checkpoints")
        if checkpoint_folders:
            default_folder = str(checkpoint_folders[0])

        return {
            "required": {
                # ── SECTION 1: INPUTS ───────────────────────────────────
                "checkpoint": (["None"] + checkpoints,),

                # ── SECTION 2: EXTRACTION OPTIONS ───────────────────────
                "extract_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract CLIP text encoder (both CLIP-L and CLIP-G for SDXL)",
                }),
                "extract_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract VAE (autoencoder decoder + encoder)",
                }),
                "extract_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract UNet/diffusion model (the main denoising network)",
                }),

                # ── SECTION 3: HARDWARE ─────────────────────────────────
                "precision": (PRECISION_EXTENDED, {
                    "default": "auto",
                    "tooltip": "Output precision for extracted components. "
                               "'auto' preserves the checkpoint's native dtype.",
                }),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),

                # ── SECTION 4: OUTPUT ───────────────────────────────────
                "save_trigger": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save extracted components as standalone .safetensors files",
                }),
            },
            "optional": {
                # ── SECTION 1 (continued) ───────────────────────────────
                "checkpoint_data": ("CHECKPOINT", {
                    "tooltip": "Chained state dict from another node (overrides dropdown)",
                }),

                # ── SECTION 4 (continued) ───────────────────────────────
                "save_folder": ("STRING", {
                    "default": default_folder,
                    "multiline": False,
                    "tooltip": "Output folder for saved component files "
                               "(leave empty for default checkpoints dir)",
                }),
                "filename_prefix": ("STRING", {
                    "default": "extracted",
                    "multiline": False,
                    "tooltip": "Prefix for saved filenames "
                               "(e.g. 'my_model' → my_model_clip.safetensors)",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache implementation for ComfyUI change detection."""
        # If data input is connected, always re-execute (can't cheaply hash state dicts)
        if kwargs.get("checkpoint_data") is not None:
            return float("nan")

        return NodeCache.is_changed(
            cls.__name__,
            checkpoint=kwargs.get("checkpoint", "None"),
            extract_clip=kwargs.get("extract_clip", True),
            extract_vae=kwargs.get("extract_vae", True),
            extract_model=kwargs.get("extract_model", True),
            precision=kwargs.get("precision", "auto"),
            device=kwargs.get("device", "auto"),
            save_trigger=kwargs.get("save_trigger", False),
            filename_prefix=kwargs.get("filename_prefix", "extracted"),
        )

    RETURN_TYPES = ("CLIP", "VAE", "MODEL", "CHECKPOINT", "CHECKPOINT", "CHECKPOINT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("clip", "vae", "model", "clip_data", "vae_data", "unet_data", "clip_path", "vae_path", "unet_path")
    FUNCTION = "extract"
    CATEGORY = "Checkpoint/Utils"
    OUTPUT_NODE = True

    def extract(
        self,
        checkpoint: str = "None",
        extract_clip: bool = True,
        extract_vae: bool = True,
        extract_model: bool = True,
        precision: str = "auto",
        device: str = "auto",
        save_trigger: bool = False,
        checkpoint_data: Optional[Dict] = None,
        save_folder: str = "",
        filename_prefix: str = "extracted",
        **kwargs,
    ):
        """
        Extract CLIP + VAE + UNet components from a checkpoint.

        Returns ``(clip, vae, model, clip_data, vae_data, unet_data,
                    clip_path, vae_path, unet_path)``.
        """
        print("\n" + "=" * 60)
        print("🔧 Easy Component Extractor")
        print("=" * 60)
        start_time = time.time()

        # ══ Runtime precision guard — protect against stale workflow JSONs ══
        if precision not in PRECISION_EXTENDED:
            print(f"   ⚠️ Precision '{precision}' is no longer available, falling back to 'auto'")
            precision = "auto"

        # Default returns
        clip_obj = vae_obj = model_obj = None
        clip_data = vae_data = unet_data = None
        clip_path_str = vae_path_str = unet_path_str = ""

        # ── Resolve precision ──────────────────────────────────────────
        dpc = DevicePrecisionConfig(device_type=device, precision=precision)
        target_dtype = dpc.dtype  # torch.dtype or None (auto = keep native)
        print(f"   🎯 Precision: {precision} → {target_dtype}")
        print(f"   💻 Device: {dpc.device}")

        # ── Dispatch: load source ──────────────────────────────────────
        # P5: 2 top-level branches (chained data vs file path), ≤5 lines each
        source_sd = None
        source_metadata: Dict = {}
        source_header: Optional[Dict] = None

        if checkpoint_data is not None:
            # ── Branch 1: Chained data (P2: no temp file) ──
            print("   📦 Using chained checkpoint_data (in-memory)")
            if isinstance(checkpoint_data, dict):
                source_sd = checkpoint_data
            elif isinstance(checkpoint_data, tuple) and len(checkpoint_data) >= 1:
                source_sd = checkpoint_data[0]
            else:
                print("   ❌ checkpoint_data has unexpected type — expected dict")
                return (None, None, None, None, None, None, "", "", "")
            # No header available for chained data; metadata is empty
            source_metadata = {}

        elif checkpoint != "None" and checkpoint:
            # ── Branch 2: File path via _LazyCheckpointMapping (P2: mmap, 0 RAM) ──
            print(f"   📁 Loading checkpoint: {checkpoint}")
            ckpt_path = resolve_model_path(checkpoint)
            if ckpt_path is None:
                print(f"   ❌ Checkpoint not found: {checkpoint}")
                return (None, None, None, None, None, None, "", "", "")
            ckpt_path = Path(ckpt_path)

            # Read header only (P2: ~50 KB read, no tensor load)
            source_header, source_metadata = read_safetensors_header_only(ckpt_path)
            print(f"   📋 Header scanned: {len(source_header)} keys, "
                  f"{sum(1 for v in source_header.values() if isinstance(v, dict))} tensors")

            # Lazy mapping (P2: zero tensor load until extract)
            from .musubi_checkpoint_studio import MusubiCheckpointStudio
            source_sd = MusubiCheckpointStudio._LazyCheckpointMapping(
                ckpt_path, source_metadata,
                target_dtype=None,  # Keep native — precision applied after extraction
            )

        else:
            print("   ⚠️ No checkpoint selected and no checkpoint_data provided")
            return (None, None, None, None, None, None, "", "", "")

        if source_sd is None:
            print("   ❌ Failed to obtain source state dict")
            return (None, None, None, None, None, None, "", "", "")

        # ── Print component breakdown from header (if available) ───────
        if source_header is not None:
            comp_counts: Dict[str, int] = {}
            for k in source_header:
                if isinstance(source_header.get(k), dict):
                    comp = categorize_key(k)
                    comp_counts[comp] = comp_counts.get(comp, 0) + 1
            print(f"   📊 Component breakdown (header): "
                  f"UNet={comp_counts.get('unet', 0)}, "
                  f"TE={comp_counts.get('te', 0)}, "
                  f"VAE={comp_counts.get('vae', 0)}, "
                  f"CLIP={comp_counts.get('clip', 0)}, "
                  f"Other={comp_counts.get('other', 0)}")

        # ── Phase A: Extract CLIP (text encoder) ───────────────────────
        if extract_clip:
            print(f"\n📝 Extracting CLIP (text encoder)...")
            clip_data = _filter_component(
                source_sd, category="te", header=source_header,
            )
            if not clip_data:
                print("   ⚠️ No CLIP/text-encoder keys found in checkpoint")
                clip_data = None
            else:
                # Apply precision for saved file
                if target_dtype is not None:
                    clip_data = _convert_precision(
                        clip_data, target_dtype, component_label="CLIP",
                    )
        else:
            print("   ⏭️ CLIP extraction disabled")

        # ── Phase A: Extract UNet/diffusion model ─────────────────────
        if extract_model:
            print(f"\n🧠 Extracting UNet...")
            unet_data = _filter_component(
                source_sd, category="unet", header=source_header,
            )
            if not unet_data:
                print("   ⚠️ No UNet keys found in checkpoint")
                unet_data = None
            else:
                # Apply precision for saved file
                if target_dtype is not None:
                    unet_data = _convert_precision(
                        unet_data, target_dtype, component_label="UNet",
                    )
        else:
            print("   ⏭️ UNet extraction disabled")

        # ── Phase B: Extract VAE ───────────────────────────────────────
        if extract_vae:
            print(f"\n🎨 Extracting VAE...")
            vae_data = _filter_component(
                source_sd, category="vae", header=source_header,
            )
            if not vae_data:
                print("   ⚠️ No VAE keys found in checkpoint")
                vae_data = None
            else:
                # Apply precision for saved file
                if target_dtype is not None:
                    vae_data = _convert_precision(
                        vae_data, target_dtype, component_label="VAE",
                    )
        else:
            print("   ⏭️ VAE extraction disabled")

        # ── Optional save to disk ──────────────────────────────────────
        if save_trigger and save_folder:
            print(f"\n💾 Saving components...")
            save_dir = Path(save_folder)
            save_dir.mkdir(parents=True, exist_ok=True)

            if clip_data is not None and extract_clip:
                # ── Transform keys for standalone CLIP Loader compatibility ──
                clip_primary, clip_secondary = _prepare_clip_for_save(clip_data)

                # ── Save primary file ──────────────────────────────────
                if clip_primary is not None:
                    clip_path = save_dir / f"{filename_prefix}_clip.safetensors"
                    try:
                        save_safetensors_stream(
                            clip_primary, clip_path, metadata=source_metadata,
                        )
                        clip_path_str = str(clip_path)
                        file_size_mb = clip_path.stat().st_size / (1024 * 1024)
                        print(f"   ✅ CLIP saved: {clip_path} ({file_size_mb:.1f} MB)")
                    except Exception as e:
                        print(f"   ❌ Failed to save CLIP: {e}")
                else:
                    print("   ⚠️ No primary CLIP data to save (all keys were secondary)")

                # ── Save secondary file (e.g. CLIP-G for SDXL dual CLIP) ──
                if clip_secondary is not None:
                    clip_g_path = save_dir / f"{filename_prefix}_clip_g.safetensors"
                    try:
                        save_safetensors_stream(
                            clip_secondary, clip_g_path, metadata=source_metadata,
                        )
                        g_file_size_mb = clip_g_path.stat().st_size / (1024 * 1024)
                        print(f"   ✅ CLIP-G saved: {clip_g_path} ({g_file_size_mb:.1f} MB)")
                        print(f"   💡 Use Dual CLIP Loader with both files for full SDXL support")
                    except Exception as e:
                        print(f"   ❌ Failed to save CLIP-G: {e}")

            if vae_data is not None and extract_vae:
                vae_path = save_dir / f"{filename_prefix}_vae.safetensors"
                try:
                    save_safetensors_stream(
                        vae_data, vae_path, metadata=source_metadata,
                    )
                    vae_path_str = str(vae_path)
                    file_size_mb = vae_path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ VAE saved: {vae_path} ({file_size_mb:.1f} MB)")
                except Exception as e:
                    print(f"   ❌ Failed to save VAE: {e}")

            if unet_data is not None and extract_model:
                unet_path = save_dir / f"{filename_prefix}_unet.safetensors"
                try:
                    save_safetensors_stream(
                        unet_data, unet_path, metadata=source_metadata,
                    )
                    unet_path_str = str(unet_path)
                    file_size_mb = unet_path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ UNet saved: {unet_path} ({file_size_mb:.1f} MB)")
                except Exception as e:
                    print(f"   ❌ Failed to save UNet: {e}")

        # ── Phase D: Load live model objects from the FULL checkpoint ──
        # P1: Pass the full state dict (lazy mapping or plain dict) so that
        # ComfyUI's load_state_dict_guess_config can detect the architecture
        # (UNet prefix + TE + VAE) and correctly load all model objects.
        if extract_model or extract_clip or extract_vae:
            try:
                model_obj, clip_obj, vae_obj = load_state_dict_as_model_objects(
                    source_sd,
                    metadata=source_metadata,
                    output_vae=extract_vae,
                    output_clip=extract_clip,
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
                    print("   ⚠️ No model objects could be loaded from the checkpoint")
            except Exception as e:
                print(f"   ❌ Failed to load model objects from checkpoint: {e}")
                import traceback
                traceback.print_exc()
                model_obj = clip_obj = vae_obj = None

        # ── Free source (lazy mapping handles close via GC) ────────────
        if hasattr(source_sd, "filepath") and hasattr(source_sd, "permanent_close"):
            source_sd.permanent_close()
        cleanup_memory(skip_gc=True)

        elapsed = time.time() - start_time
        print(f"\n⏱️  Extraction complete: {elapsed:.2f}s")
        print(f"   📦 MODEL: {'✅ loaded' if model_obj is not None else '❌ none'}"
              f"{' / saved' if unet_path_str else ''}")
        print(f"   📦 CLIP:  {'✅ loaded' if clip_obj is not None else '❌ none'}"
              f"{' / saved' if clip_path_str else ''}")
        print(f"   📦 VAE:   {'✅ loaded' if vae_obj is not None else '❌ none'}"
              f"{' / saved' if vae_path_str else ''}")
        print("=" * 60)

        return (clip_obj, vae_obj, model_obj, clip_data, vae_data, unet_data,
                clip_path_str, vae_path_str, unet_path_str)
