#!/usr/bin/env python3
"""
Easy LoRA Studio – Modular node for converting LoRA formats with forensic analysis.
Extracted from easy_lora_merger.py as part of the modular refactor.
"""

import warnings
from pathlib import Path
import json
import hashlib
import re
from collections import Counter

import torch
from safetensors.torch import save_file

import folder_paths
import comfy.sd

# Local imports
from ..config import PRECISION_STANDARD, DEVICE_OPTIONS
from ..utils import (
    load_lora_with_metadata,
    categorize_key,
    DeviceManager,
    ProgressTracker,
    comfyui_yield,
    cleanup_memory,
)
from .klein_normalizer import (
    detect_lora_format,
    safe_get_rank,
    infer_separator_style,
    infer_naming_style,
)
from .identity_normalizer import identity_normalize
from .serialization_factory import finalize_for_save
from .metadata_factory import finalize_metadata

# Suppress warnings
warnings.filterwarnings("ignore", message="lora key not loaded")


class MusubiLoraConverter:
    OUTPUT_NODE = True
    """
    Universal LoRA converter: normalizes any trainer format (Musubi, Z‑Image, Klein, Flux)
    and outputs in the selected target format (Standard WebUI, Comfy Native, Forge‑Optimized).
    """

    @staticmethod
    def _first_of(d, *keys, default='Unknown'):
        """Return the first non‑None/non‑'Unknown' value found for any key, or *default*."""
        for k in keys:
            v = d.get(k)
            if v is not None and v != 'Unknown':
                return v
        return default

    @staticmethod
    def _detect_ecosystem(keys, training_metadata, all_metadata, primary_structure):
        """
        Detect target ecosystem and architecture notes from LoRA keys and metadata.

        Returns (target_ecosystem, architecture_notes).
        """
        target_ecosystem = 'Unknown'
        architecture_notes = []
        # Pattern mapping
        pattern_map = [
            ('double_blocks', 'Flux.1-Dev/S', 'Flux double-block architecture'),
            ('single_blocks', 'Flux.1-Dev/S', 'Flux single-block architecture'),
            ('joint_blocks', 'SD3', 'SD3 joint-block architecture'),
            ('lora_unet', 'SDXL', 'SDXL standard LoRA'),
            ('lora_unet_layers', 'Z-Image', 'Z-Image layered LoRA'),
            ('diffusion_model.layers', 'Z-Image', 'Transformer (DiT)'),
            ('adaln', 'Flux Klein', 'Flux Klein adaln attention'),
            ('adain', 'Flux Klein', 'Flux Klein adain normalization'),
            ('context_refiner', 'Z-Image', 'Z-Image context refiner'),
            ('context_encoder', 'Z-Image', 'Z-Image context encoder'),
        ]
        for pattern, ecosystem, note in pattern_map:
            if any(pattern in k for k in keys):
                if target_ecosystem == 'Unknown':
                    target_ecosystem = ecosystem
                architecture_notes.append(note)
        # Z‑Image detection via ss_base_model_version or modelspec.architecture
        if target_ecosystem == 'Unknown':
            ss_base = training_metadata.get('ss_base_model_version')
            if ss_base and ss_base.lower() == 'zimage':
                target_ecosystem = 'Z-Image'
                architecture_notes.append('Z-Image (zimage base model)')
            else:
                modelspec_arch = all_metadata.get('modelspec.architecture')
                if isinstance(modelspec_arch, str) and 'Z-Image' in modelspec_arch:
                    target_ecosystem = 'Z-Image'
                    architecture_notes.append('Z-Image (modelspec.architecture)')

        # If no matched patterns, try to infer from primary_structure
        if target_ecosystem == 'Unknown':
            if primary_structure == 'flux_double' or primary_structure == 'flux_single':
                target_ecosystem = 'Flux.1-Dev/S'
                architecture_notes.append('Flux architecture (detected via structure)')
            elif primary_structure == 'z_image_style':
                target_ecosystem = 'Z-Image'
                architecture_notes.append('Z-Image style (detected via structure)')
            else:
                target_ecosystem = 'SDXL/Unknown'

        return target_ecosystem, architecture_notes

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        default_folder = ""
        lora_folders = folder_paths.get_folder_paths("loras")
        if lora_folders:
            default_folder = str(lora_folders[0])

        return {
            "required": {
                # ── SECTION 1: INPUTS ───────────────────────────────────
                "lora": (["None"] + loras,),

                # ── SECTION 3: SETTINGS ─────────────────────────────────
                "target_format": (["auto", "standard_webui", "comfy_native", "forge_optimized"],
                                 {"default": "auto",
                                  "tooltip": "Auto‑select optimal format based on LoRA structure; otherwise choose manually. Standard (WebUI/Neo) = underscore + lora_down/up; Comfy Native = dot + lora_A/B; Forge‑Optimized = underscore + prefix mapping for Flux"}),
                "compression_mode": (["original", "auto-fast", "auto-full", "manual"], {
                    "default": "original",
                    "tooltip": "Original: no compression. Auto-fast: fast randomized SVD preserving 95% energy. Auto-full: full precision SVD preserving 95% energy (slower). Manual: use target rank slider."
                }),
                "target_rank": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 320,
                    "step": 1,
                    "tooltip": "Target rank for manual compression (1‑320). Ignored if compression_mode is not manual."
                }),
                "bake_custom_scale": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Multiply up‑weight tensors (lora_up.weight/lora_B.weight) by this factor before baking (e.g., 0.5 weakens the LoRA, 2.0 strengthens). Note: down‑weights are not scaled to avoid double‑strength effect."}),

                # ── SECTION 6: OUTPUT ───────────────────────────────────
                "save_trigger": ("BOOLEAN", {"default": False}),
                "filename": ("STRING", {"default": "converted_lora", "multiline": False}),
            },
            "optional": {
                # ── SECTION 1: INPUTS (continued) ───────────────────────
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_data": ("LORA",),

                # ── SECTION 3: SETTINGS (continued) ─────────────────────
                "te_mode": (["original", "remove", "scale"], {
                    "default": "original",
                    "tooltip": "Control Text Encoder weights: original (keep as‑is), remove (strip TE keys), scale (multiply by te_weight)"
                }),
                "te_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Multiplier for TE weights when te_mode is 'scale'"
                }),

                # ── SECTION 5: HARDWARE ─────────────────────────────────
                "precision": (PRECISION_STANDARD, {"default": "auto"}),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),

                # ── SECTION 6: OUTPUT (continued) ───────────────────────
                "save_folder": ("STRING", {"default": default_folder, "multiline": False}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Create cache key from relevant inputs
        cache_key = (
            kwargs.get('lora', 'None'),
            kwargs.get('save_trigger', False),
            kwargs.get('target_format', 'standard_webui'),
            kwargs.get('compression_mode', 'original'),
            kwargs.get('target_rank', 128),
            kwargs.get('bake_custom_scale', 1.0),
            kwargs.get('te_mode', 'original'),
            kwargs.get('te_weight', 1.0),
            kwargs.get('precision', 'auto'),
            kwargs.get('device', 'auto'),
            kwargs.get('model', None),
            kwargs.get('clip', None),
        )
        # Generate hash
        key_str = str(cache_key).encode('utf-8')
        return hashlib.md5(key_str).hexdigest()

    def _apply_svd_compression(self, normalized_sd, compression_mode, target_rank, device=None):
        """
        Apply SVD compression to normalized state dict using batched SVD.
        
        LoRA layers within a model architecture typically share dimensions
        (e.g., all DiT blocks have QKV projections of the same size).
        We group pairs by (out_features, in_features) and call
        torch.linalg.svd once per group with a batched 3D tensor,
        dramatically reducing GPU kernel launch overhead.
        
        Returns (compressed_sd, compression_info) where compression_info dict contains:
            - compression_mode
            - original_total_params
            - compressed_total_params
            - size_saved_mb
            - final_rank (average)
        """
        # Copy the state dict to modify
        compressed_sd = normalized_sd.copy()
        
        # Detect pairs: lora_down/up or lora_A/B
        down_keys = []
        up_keys = []
        for key in list(compressed_sd.keys()):
            if key.endswith('.lora_down.weight') or key.endswith('.lora_A.weight'):
                down_keys.append(key)
            elif key.endswith('.lora_up.weight') or key.endswith('.lora_B.weight'):
                up_keys.append(key)
        
        # Match pairs by base name
        pairs = []
        for dk in down_keys:
            base = dk.replace('.lora_down.weight', '').replace('.lora_A.weight', '')
            # find corresponding up key
            uk = None
            for uk_candidate in up_keys:
                if uk_candidate.startswith(base) and (uk_candidate.endswith('.lora_up.weight') or uk_candidate.endswith('.lora_B.weight')):
                    uk = uk_candidate
                    break
            if uk is None:
                continue
            pairs.append((dk, uk, base))
        
        total_original_params = 0
        total_compressed_params = 0
        rank_sum = 0
        layer_count = 0
        skipped_conv = 0
        
        # ================================================================
        # PHASE 1: Analyze all pairs, detect orientation, group by shape
        # ================================================================
        # Groups are keyed by (out_features, in_features) — pairs sharing
        # identical matrix dimensions are processed together in one batched
        # SVD call, which is dramatically faster than one-by-one on GPU.
        shape_groups = {}       # {(out, in): [(dk, uk, base, transposed, rank), ...]}
        conv_pairs = []         # handled separately (no SVD)
        skipped_mismatch = []   # rank mismatch warnings
        
        for dk, uk, base in pairs:
            down = compressed_sd[dk]
            up = compressed_sd[uk]

            # Skip convolutional layers (spatial dimensions >2) to avoid shape errors
            if down.ndim > 2 or up.ndim > 2:
                conv_pairs.append((dk, uk, base))
                # Ensure tensors are contiguous for safetensors
                if not down.is_contiguous():
                    compressed_sd[dk] = down.contiguous()
                if not up.is_contiguous():
                    compressed_sd[uk] = up.contiguous()
                continue

            # Detect orientation: down shape (out_features, rank) or (rank, out_features)
            transposed = False
            if down.shape[1] == up.shape[0] and down.shape[0] == up.shape[1]:
                # Both orientations possible (square matrices). Decide based on typical LoRA rank being smaller dimension.
                rank_standard = down.shape[1]
                if rank_standard < down.shape[0] and rank_standard < up.shape[1]:
                    out_features, rank = down.shape
                    rank2, in_features = up.shape
                else:
                    out_features, rank = down.T.shape
                    rank2, in_features = up.T.shape
                    transposed = True
            elif down.shape[1] == up.shape[0]:
                out_features, rank = down.shape
                rank2, in_features = up.shape
            elif down.shape[0] == up.shape[1]:
                out_features, rank = down.T.shape
                rank2, in_features = up.T.shape
                transposed = True
            else:
                skipped_mismatch.append((dk, uk, base))
                continue

            assert rank == rank2, f"Rank mismatch {rank} != {rank2} for pair {dk}, {uk}"

            # Group by (out_features, in_features) for batched SVD
            shape_key = (out_features, in_features)
            if shape_key not in shape_groups:
                shape_groups[shape_key] = []
            shape_groups[shape_key].append((dk, uk, base, transposed, rank))

        if skipped_mismatch:
            for dk, uk, base in skipped_mismatch:
                print(f"⚠️ Rank mismatch for pair {dk}, {uk}; skipping")

        # ================================================================
        # PHASE 2: Micro-batched SVD per shape group
        # ================================================================
        # Large shape groups (e.g., 105 identical QKV projections in DiT models)
        # produce W = down @ up matrices of size (out_features, in_features).
        # For 4096×4096 W, stacking ALL 105 into one W_batch would consume
        # ~7 GB + SVD intermediates — exceeding most consumer GPUs.
        #
        # Instead, process in micro-batches of SVD_MICRO_BATCH pairs each.
        # This caps peak VRAM while still amortizing kernel launch overhead.
        SVD_MICRO_BATCH = 8  # pairs per batched SVD call
        total_pairs = len(pairs)

        with ProgressTracker(total=total_pairs, desc="[SVD] Processing layers") as svd_progress:

            for (out_features, in_features), group in shape_groups.items():
                group_size = len(group)

                # Process this shape group in micro-batches to cap peak VRAM
                for start_idx in range(0, group_size, SVD_MICRO_BATCH):
                    micro_batch = group[start_idx:start_idx + SVD_MICRO_BATCH]
                    mb_size = len(micro_batch)

                    # ── Step 1: Compute W matrices for this micro-batch ──
                    W_list = []
                    for dk, uk, base, transposed, rank in micro_batch:
                        down = compressed_sd[dk]
                        up = compressed_sd[uk]

                        # On-demand GPU transfer (lazy, per-pair)
                        down_on_gpu = False
                        up_on_gpu = False
                        if device is not None and down.device != device:
                            down = down.to(device=device)
                            down_on_gpu = True
                        if device is not None and up.device != device:
                            up = up.to(device=device)
                            up_on_gpu = True

                        if transposed:
                            W = down.T.float() @ up.T.float()
                        else:
                            W = down.float() @ up.float()

                        # Free GPU copies of down/up immediately — no longer needed
                        if down_on_gpu:
                            del down
                        if up_on_gpu:
                            del up

                        W_list.append(W)

                    # ── Step 2: Batched SVD for this micro-batch ──
                    if compression_mode == 'auto-fast':
                        # Randomized SVD (fast): compute only top-q singular values.
                        # For LoRA, W = down @ up has rank <= training rank.
                        # q = rank + 10 oversampling ensures >99.9% energy capture.
                        pair_rank = micro_batch[0][4]  # original training rank
                        q = min(pair_rank + 10, out_features, in_features)
                        q = max(q, 1)
                        if mb_size == 1:
                            W = W_list[0]
                            U, S, V = torch.svd_lowrank(W, q=q)
                            Vh = V.mT  # svd_lowrank returns V; convert to Vh convention
                            U_batch, S_batch, Vh_batch = U.unsqueeze(0), S.unsqueeze(0), Vh.unsqueeze(0)
                            del W
                        else:
                            W_batch = torch.stack(W_list, dim=0)
                            del W_list
                            U_batch, S_batch, V_batch = torch.svd_lowrank(W_batch, q=q)
                            Vh_batch = V_batch.mT  # convert to Vh convention
                            del W_batch
                    else:
                        # Exact SVD (auto-full, original, manual)
                        if mb_size == 1:
                            W = W_list[0]
                            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                            U_batch, S_batch, Vh_batch = U.unsqueeze(0), S.unsqueeze(0), Vh.unsqueeze(0)
                            del W  # free single W immediately
                        else:
                            W_batch = torch.stack(W_list, dim=0)  # (B, M, N)
                            del W_list  # free list before SVD to save memory
                            U_batch, S_batch, Vh_batch = torch.linalg.svd(W_batch, full_matrices=False)
                            del W_batch  # free stacked W immediately after SVD

                    # ── Step 3: Process results (truncate and refactorize) ──
                    for i, (dk, uk, base, transposed, rank) in enumerate(micro_batch):
                        S = S_batch[i]
                        U = U_batch[i]
                        Vh = Vh_batch[i]

                        # Determine target rank
                        if compression_mode == 'original':
                            k = rank
                        elif compression_mode == 'manual':
                            k = min(target_rank, rank)
                            k = max(k, 1)
                        else:  # auto-fast or auto-full (same 95% energy threshold)
                            total_energy = torch.sum(S ** 2)
                            cumulative = torch.cumsum(S ** 2, dim=0)
                            k = torch.searchsorted(cumulative, 0.95 * total_energy).item() + 1
                            k = min(k, rank)
                            k = max(k, 1)

                        # Truncate and re-factorize
                        sqrt_Sk = torch.sqrt(S[:k])
                        down_new = (U[:, :k] * sqrt_Sk).contiguous()  # (out_features, k)
                        up_new = (sqrt_Sk[:, None] * Vh[:k, :]).contiguous()  # (k, in_features)

                        # Cast back to original dtype
                        down_dtype = compressed_sd[dk].dtype
                        up_dtype = compressed_sd[uk].dtype
                        down_new = down_new.to(dtype=down_dtype)
                        up_new = up_new.to(dtype=up_dtype)

                        if transposed:
                            down_new = down_new.T.contiguous()
                            up_new = up_new.T.contiguous()

                        compressed_sd[dk] = down_new
                        compressed_sd[uk] = up_new

                        # Update alpha if present
                        alpha_key = f"{base}.alpha"
                        if alpha_key in compressed_sd:
                            alpha = compressed_sd[alpha_key]
                            alpha = alpha * (k / rank)
                            compressed_sd[alpha_key] = alpha

                        # Statistics
                        total_original_params += out_features * rank + rank * in_features
                        total_compressed_params += out_features * k + k * in_features
                        rank_sum += k
                        layer_count += 1
                        svd_progress += 1

                    # Free SVD result tensors for this micro-batch
                    del U_batch, S_batch, Vh_batch

                    # Yield to ComfyUI scheduler between micro-batches
                    comfyui_yield()

                # Clear GPU cache between shape groups
                cleanup_memory()

            # Handle convolutional pairs (no SVD, just ensure contiguous)
            for dk, uk, base in conv_pairs:
                skipped_conv += 1
                print(f"ℹ️ Skipping convolutional layer {base} (shape {compressed_sd[dk].shape}, {compressed_sd[uk].shape})")
                svd_progress += 1

        if skipped_conv > 0:
            print(f"ℹ️ Skipped {skipped_conv} convolutional layers (normal for SDXL/UNet).")

        # Compute size saved (assuming float32 = 4 bytes)
        size_saved_mb = (total_original_params - total_compressed_params) * 4 / (1024 * 1024)
        avg_final_rank = rank_sum / layer_count if layer_count > 0 else 0

        compression_info = {
            'compression_mode': compression_mode,
            'original_total_params': total_original_params,
            'compressed_total_params': total_compressed_params,
            'size_saved_mb': size_saved_mb,
            'final_rank': avg_final_rank,
        }

        return compressed_sd, compression_info

    RETURN_TYPES = ("LORA", "MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("lora", "model", "clip", "output_path", "forensic_report")
    FUNCTION = "convert"
    CATEGORY = "LoRA/Universal"


    # ── Extracted helpers for convert() ────────────────────────────────────────

    @staticmethod
    def _resolve_input(lora, lora_data):
        """Resolve input source (lora_data dict or file path).

        Returns (sd, metadata, source_type) or sentinel (None, None, None, "", "")
        if no input is provided.
        """
        if lora_data is not None:
            lora_dict = lora_data[0] if isinstance(lora_data, (tuple, list)) else lora_data
            sd = lora_dict
            metadata = {}
            source_type = "LORA data input (in-memory)"
            print(f"\U0001f4c4 Source: {source_type} ({len(sd)} tensors)")
            return sd, metadata, source_type
        elif lora != "None" and lora:
            path = folder_paths.get_full_path("loras", lora)
            source_type = f"dropdown: {lora}"
            print(f"\U0001f4c4 Source: {source_type}")
            print(f"\U0001f4c1 Path: {path}")
            sd, metadata = load_lora_with_metadata(Path(path))
            return sd, metadata, source_type
        else:
            print("\u274c No LoRA input provided")
            return (None, None, None, "", "")

    def _handle_normalization(self, sd, metadata, target_format):
        """Normalize LoRA state dict to master format.

        Returns (normalized_sd, key_map).
        SD1.5 Diffusers LoRA detected via bridge for block-structure preservation.
        """
        from .diffusers_bridge import detect_diffusers_sd15_lora, normalize_diffusers_preserving

        if detect_diffusers_sd15_lora(sd) and target_format != "comfy_native":
            print("   SD1.5 Diffusers LoRA detected -- using Diffusers-preserving normalization")
            normalized_sd, key_map = normalize_diffusers_preserving(sd)
        else:
            normalized_sd, key_map = identity_normalize(sd, metadata)
        print(f"Normalized to master format: {len(normalized_sd)} keys")
        return normalized_sd, key_map

    def _handle_compression(self, normalized_sd, compression_mode, target_rank, device, info):
        """Apply SVD compression if requested. Modifies info in-place.

        Returns (normalized_sd, compression_info).
        Tensors are moved to the target device on-demand inside
        _apply_svd_compression (lazy, per-pair) instead of eagerly
        moving all tensors upfront — reduces PCIe traffic.
        """
        target_device = DeviceManager.get_device(device)

        if compression_mode != "original":
            print(f"\U0001f527 Applying SVD compression ({compression_mode})...")
            normalized_sd, compression_info = self._apply_svd_compression(
                normalized_sd, compression_mode, target_rank, device=target_device
            )
            info.update(compression_info)
            print(f"\U0001f527 Compression applied. Final rank: {compression_info.get('final_rank', 'N/A'):.1f}, "
                  f"size saved: {compression_info.get('size_saved_mb', 0):.2f} MB")
        else:
            compression_info = {
                'compression_mode': 'original',
                'final_rank': info.get('avg_rank', 0),
                'size_saved_mb': 0.0,
            }
            info.update(compression_info)
        return normalized_sd, compression_info

    @staticmethod
    def _handle_te_control(normalized_sd, te_mode, te_weight):
        """Apply Text Encoder modifications (remove or scale) in-place."""
        if te_mode != "original":
            te_keys = [k for k in normalized_sd.keys() if categorize_key(k) == 'te']
            if not te_keys:
                print("\U0001f524 No Text Encoder keys detected, skipping TE modification")
            else:
                if te_mode == "remove" or (te_mode == "scale" and te_weight == 0.0):
                    for key in te_keys:
                        del normalized_sd[key]
                    label = "Removed" if te_mode == "remove" else "Weight is 0.0, removed"
                    print(f"\U0001f524 {label} {len(te_keys)} TE keys")
                elif te_mode == "scale":
                    scaled_count = 0
                    for key in te_keys:
                        normalized_sd[key] = normalized_sd[key] * te_weight
                        scaled_count += 1
                    print(f"\U0001f524 Scaled {scaled_count} TE keys by {te_weight}x")

    @staticmethod
    def _handle_weight_scaling(normalized_sd, bake_custom_scale):
        """Apply up-weight scaling to LoRA up-weights. Returns effective_scaling bool."""
        effective_scaling = abs(bake_custom_scale - 1.0) > 1e-6
        if effective_scaling:
            print(f"\u2696\ufe0f Applying up-weight scaling: {bake_custom_scale}x")
            scaled_keys = 0
            for key in list(normalized_sd.keys()):
                if key.endswith(".weight") and ("lora_B" in key or "lora_up" in key):
                    normalized_sd[key] = normalized_sd[key] * bake_custom_scale
                    scaled_keys += 1
            print(f"   Scaled {scaled_keys} up-weight tensors")
        return effective_scaling

    def _handle_format_conversion(self, normalized_sd, key_map, target_format, info):
        """Auto-detect target format, convert keys, apply alpha baking.

        Returns (converted_sd, effective_target_format, bake_alphas_flag).
        """
        effective_target_format = target_format
        if target_format == "auto":
            separator = infer_separator_style(list(normalized_sd.keys()))
            naming = infer_naming_style(list(normalized_sd.keys()))
            target_ecosystem = info.get('target_ecosystem', 'Unknown')
            primary_structure = info.get('primary_structure', '')

            if target_ecosystem in ("Flux.1-Dev/S", "Z-Image"):
                if separator == "underscore" and naming == "lora_down_up":
                    effective_target_format = "forge_optimized"
                else:
                    effective_target_format = "comfy_native"
            elif separator == "dot" and naming == "lora_a_b":
                effective_target_format = "comfy_native"
            else:
                effective_target_format = "standard_webui"
            print(f"\U0001f50d Auto-selected target format: {effective_target_format} "
                  f"(separator={separator}, naming={naming}, ecosystem={target_ecosystem})")

        target_ecosystem = info.get('target_ecosystem', 'Unknown')
        if target_ecosystem in ("Flux.1-Dev/S", "Z-Image"):
            bake_alphas_flag = True
        else:
            bake_alphas_flag = False
        print(f"\U0001f4dd Legacy-aware alpha baking: alphas will be "
              f"{'baked' if bake_alphas_flag else 'kept'} (detected {target_ecosystem})")

        restored_sd, _ = finalize_for_save(
            normalized_sd, key_map, target_format=effective_target_format
        )

        if effective_target_format == 'forge_optimized':
            prefix_mapped = {}
            for key, tensor in restored_sd.items():
                if ('double_blocks' in key or 'single_blocks' in key) and key.startswith('diffusion_model.'):
                    key = 'transformer.' + key[len('diffusion_model.'):]
                prefix_mapped[key] = tensor
            restored_sd = prefix_mapped

        if bake_alphas_flag:
            from ..utils import bake_alphas
            baking_naming_style = infer_naming_style(list(restored_sd.keys()))
            restored_sd = bake_alphas(restored_sd, naming_style=baking_naming_style)

        converted_sd = restored_sd
        print(f"\U0001f3af Converted to {effective_target_format}: {len(converted_sd)} keys")
        return converted_sd, effective_target_format, bake_alphas_flag

    @staticmethod
    def _handle_precision_cast(converted_sd, precision, device):
        """Apply precision/device casting if not auto."""
        if precision != "auto" or device != "auto":
            target_device = DeviceManager.get_device(device)
            if precision != "auto":
                target_dtype = DeviceManager.get_dtype(precision, target_device)
            else:
                target_dtype = None
            cast_count = 0
            for key in list(converted_sd.keys()):
                tensor = converted_sd[key]
                if tensor.device != target_device or (target_dtype is not None and tensor.dtype != target_dtype):
                    converted_sd[key] = tensor.to(
                        device=target_device,
                        dtype=target_dtype if target_dtype is not None else tensor.dtype
                    )
                    cast_count += 1
            if cast_count > 0:
                print(f"\U0001f527 Cast {cast_count} tensors to "
                      f"{target_dtype if target_dtype else 'same dtype'} on {target_device}")

    @staticmethod
    def _build_final_metadata(metadata, effective_target_format, detected_format, source_type, converted_sd):
        """Build unified metadata dict for the output safetensors file."""
        return finalize_metadata(
            metadata=metadata,
            mode="preserve_a",
            component="converter",
            extra_fields={
                "target_format": effective_target_format,
                "converted_key_count": str(len(converted_sd)),
                "original_format": detected_format,
                "source_type": source_type,
            }
        )

    @staticmethod
    def _handle_save(save_trigger, save_folder, filename, converted_sd, final_metadata):
        """Save converted LoRA to safetensors file if requested. Returns save_path."""
        save_path = ""
        if save_trigger:
            try:
                if save_folder:
                    output_folder = Path(save_folder)
                else:
                    lora_folders = folder_paths.get_folder_paths("loras")
                    output_folder = Path(lora_folders[0]) if lora_folders else Path.cwd()

                output_folder.mkdir(parents=True, exist_ok=True)
                base_name = filename.replace('.safetensors', '')
                output_path = output_folder / f"{base_name}_converted.safetensors"

                counter = 1
                while output_path.exists():
                    output_path = output_folder / f"{base_name}_converted_{counter}.safetensors"
                    counter += 1

                contiguous_count = 0
                for key in list(converted_sd.keys()):
                    tensor = converted_sd[key]
                    if not tensor.is_contiguous():
                        converted_sd[key] = tensor.contiguous()
                        contiguous_count += 1
                if contiguous_count > 0:
                    print(f"\U0001f527 Made {contiguous_count} tensors contiguous for saving")

                save_file(converted_sd, str(output_path), metadata=final_metadata)
                save_path = str(output_path)
                print(f"\U0001f4be Saved to: {save_path}")

            except Exception as e:
                print(f"\u274c Save failed: {e}")
        else:
            print("\U0001f9e0 LoRA kept in RAM, no file written")
        return save_path

    @staticmethod
    def _load_preview(model, clip, converted_sd):
        """Load converted LoRA into model + clip for preview."""
        preview_model = model
        preview_clip = clip
        if model is not None and clip is not None and converted_sd:
            try:
                preview_model, preview_clip = comfy.sd.load_lora_for_models(
                    model, clip, converted_sd, 1.0, 1.0
                )
                print("\U0001f5bc\ufe0f LoRA preview: applied to model and clip")
            except Exception as e:
                print(f"\u26a0\ufe0f Failed to apply LoRA preview: {e}")
        return preview_model, preview_clip

    def convert(self, save_trigger=False, filename="converted_lora", target_format="auto",
                precision="auto", device="auto", compression_mode="none",
                target_rank=128, te_mode="original", te_weight=0.5,
                bake_custom_scale=2.0, lora=None, lora_data=None,
                model=None, clip=None, save_folder="outputs", **kwargs):
        print("\n" + "="*50)
        print("\U0001f504 Universal EasyLoRA Converter")
        print("="*50)

        # \u2500\u2500 0. Defensive input normalization \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

        # \u2550\u2550 Runtime precision guard \u2014 protect against stale workflow JSONs \u2550\u2550
        if precision not in PRECISION_STANDARD:
            print(f"   \u26a0\ufe0f Precision '{precision}' is no longer available, falling back to 'auto'")
            precision = "auto"
        if isinstance(target_rank, str):
            try:
                target_rank = int(target_rank)
            except ValueError:
                target_rank = 128
                print(f"\u26a0\ufe0f target_rank could not be parsed as integer, defaulting to {target_rank}")
        if isinstance(te_mode, str) and te_mode.lower() == 'false':
            te_mode = 'original'
        if compression_mode == '1' or compression_mode == 1:
            compression_mode = 'auto-full'
        if compression_mode == 'auto':
            compression_mode = 'auto-full'

        # \u2500\u2500 1. Resolve input source \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        sd, metadata, source_type = self._resolve_input(lora, lora_data)
        if sd is None:
            return (None, None, None, "", "")

        # \u2500\u2500 2. Detect format + analyze structure \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        detected_format = detect_lora_format(sd)
        print(f"\U0001f50d Detected format: {detected_format}")
        info = self.analyze_lora_structure(sd, metadata)

        # \u2500\u2500 3. Normalize to master format \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        normalized_sd, key_map = self._handle_normalization(sd, metadata, target_format)

        # \u2500\u2500 4. SVD compression \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        normalized_sd, compression_info = self._handle_compression(
            normalized_sd, compression_mode, target_rank, device, info
        )

        # \u2500\u2500 5. TE control (remove / scale) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self._handle_te_control(normalized_sd, te_mode, te_weight)

        # \u2500\u2500 6. Weight scaling \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        apply_weight_scaling = self._handle_weight_scaling(normalized_sd, bake_custom_scale)

        # \u2500\u2500 7. Format conversion + alpha baking \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        converted_sd, effective_target_format, bake_alphas_flag = \
            self._handle_format_conversion(normalized_sd, key_map, target_format, info)

        # \u2500\u2500 8. Precision / device casting \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self._handle_precision_cast(converted_sd, precision, device)

        # \u2500\u2500 9. Build metadata \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        final_metadata = self._build_final_metadata(
            metadata, effective_target_format, detected_format, source_type, converted_sd
        )

        # \u2500\u2500 10. Save to file \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        save_path = self._handle_save(save_trigger, save_folder, filename, converted_sd, final_metadata)

        # \u2500\u2500 11. Build output tuple + info string \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        lora_tuple = (converted_sd, 1.0, 1.0)

        source = info['primary_structure'] if info['primary_structure'] else 'unknown'
        hidden_dims_csv = ','.join(map(str, info['hidden_dims'][:3]))
        info_parts = [
            f"Source: {source}",
            f"Keys: {len(sd)} \u2192 {len(converted_sd)}",
            f"Rank: {info['avg_rank']:.0f}",
            f"Dims: {hidden_dims_csv}",
            f"Alphas: {'baked' if bake_alphas_flag else 'kept'}"
        ]
        if apply_weight_scaling:
            info_parts.append(f"Scale: {bake_custom_scale}x")
        info_str = " | ".join(f"[{part}]" for part in info_parts)

        info['te_mode'] = te_mode
        info['te_weight'] = te_weight
        info['bake_alphas_flag'] = bake_alphas_flag

        # \u2500\u2500 12. Forensic report \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        forensic_report = self._generate_forensic_report(
            info, target_format, apply_weight_scaling, bake_custom_scale,
            save_path, len(converted_sd)
        )

        # \u2500\u2500 13. Preview (load into model + clip) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        preview_model, preview_clip = self._load_preview(model, clip, converted_sd)

        print("="*50)
        return (lora_tuple, preview_model, preview_clip, save_path, forensic_report)
    def _parse_metadata(self, metadata):
        """Parse metadata dictionary, flatten JSON strings, and extract additional fields."""
        def flatten(obj, prefix=''):
            """Recursively flatten dict/list into a dict with dot‑notation keys."""
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    result.update(flatten(v, new_key))
                return result
            elif isinstance(obj, list):
                # Convert lists to dict with indices
                result = {}
                for i, v in enumerate(obj):
                    new_key = f"{prefix}.{i}"
                    result.update(flatten(v, new_key))
                return result
            else:
                return {prefix: obj}
        
        def deep_json_parse(obj):
            """Recursively parse JSON strings inside dict/list structures."""
            if isinstance(obj, str):
                stripped = obj.strip()
                if stripped.startswith(('{', '[')):
                    try:
                        parsed = json.loads(obj)
                        return deep_json_parse(parsed)
                    except (json.JSONDecodeError, TypeError):
                        return obj
            elif isinstance(obj, dict):
                return {k: deep_json_parse(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_json_parse(v) for v in obj]
            return obj
        
        parsed = {}
        if not metadata:
            return parsed
        
        for key, value in metadata.items():
            # Deep JSON parsing
            value = deep_json_parse(value)
            # Keep the parsed value as a whole (useful for dict fields like training_info, software)
            if isinstance(value, (dict, list)):
                parsed[key] = value
            # Flatten nested structures
            flattened = flatten(value, key)
            parsed.update(flattened)
        
        return parsed

    def _extract_trigger_words(self, all_metadata):
        """Extract potential trigger words from flattened metadata."""
        trigger_field_patterns = [
            'caption',
            'ss_caption',
            'sample_prompt',
            'user_prompt',
            'instance_prompt',
            'prompt',
            'tags',
            'tag_frequency'
        ]
        
        # Collect text from fields that match any pattern
        texts = []
        for key, value in all_metadata.items():
            if not isinstance(value, str):
                continue
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in trigger_field_patterns):
                texts.append(value)
        
        if not texts:
            return [], 'none'
        
        # Concatenate first 200 chars of each text
        combined = ' '.join(texts)[:500]
        # Extract words (simple regex)
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_]*\b', combined)
        if not words:
            return [], 'text_no_words'
        
        # Count frequency
        word_counts = Counter(words)
        top_words = [word for word, _ in word_counts.most_common(3)]
        
        # Determine source
        source = 'caption'
        if 'ss_tag_frequency' in all_metadata:
            source = 'ss_tag_frequency'
        elif any('caption' in k.lower() for k in all_metadata.keys()):
            source = 'caption'
        elif any('prompt' in k.lower() for k in all_metadata.keys()):
            source = 'prompt'
        else:
            source = 'unknown'
        
        return top_words, source

    def _extract_concept_analysis(self, training_metadata, all_metadata):
        """
        Extract concept analysis from training metadata.

        Returns a dict with keys:
        - has_concept_metadata (bool)
        - trigger_word (str or None)
        - trigger_count (int or None)
        - trigger_ratio (float or None)
        - top_tags (list of (tag, count))
        - supporting_tags (list of str)
        - total_unique_tags (int)
        - source (str)
        """
        # Priority 1: ss_tag_frequency dict
        tag_freq = training_metadata.get('ss_tag_frequency')
        if isinstance(tag_freq, dict) and tag_freq:
            # Already flattened
            sorted_items = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
            total_unique = len(sorted_items)
            if not sorted_items:
                # empty dict
                return {
                    'has_concept_metadata': False,
                    'trigger_word': None,
                    'trigger_count': None,
                    'trigger_ratio': None,
                    'top_tags': [],
                    'supporting_tags': [],
                    'total_unique_tags': 0,
                    'source': 'ss_tag_frequency_empty'
                }
            # Determine trigger word (highest count)
            trigger_word, trigger_count = sorted_items[0]
            # Compute ratio if total images known
            total_images = training_metadata.get('ss_num_train_images')
            trigger_ratio = None
            if isinstance(total_images, (int, float)) and total_images > 0:
                trigger_ratio = trigger_count / total_images
            # Top 10 tags
            top_tags = sorted_items[:10]
            # Supporting tags (top 5 after removing trigger word)
            supporting = [tag for tag, _ in sorted_items[1:6] if tag != trigger_word]
            # Ensure we have at most 5
            supporting_tags = supporting[:5]
            return {
                'has_concept_metadata': True,
                'trigger_word': trigger_word,
                'trigger_count': trigger_count,
                'trigger_ratio': trigger_ratio,
                'top_tags': top_tags,
                'supporting_tags': supporting_tags,
                'total_unique_tags': total_unique,
                'source': 'ss_tag_frequency'
            }
        # Priority 2: fallback to text extraction
        trigger_words, source = self._extract_trigger_words(all_metadata)
        if trigger_words:
            # treat each word as count 1
            top_tags = [(word, 1) for word in trigger_words[:10]]
            trigger_word = trigger_words[0] if trigger_words else None
            supporting_tags = trigger_words[1:6]
            return {
                'has_concept_metadata': True,
                'trigger_word': trigger_word,
                'trigger_count': 1,
                'trigger_ratio': None,
                'top_tags': top_tags,
                'supporting_tags': supporting_tags,
                'total_unique_tags': len(trigger_words),
                'source': source
            }
        # No concept metadata found
        return {
            'has_concept_metadata': False,
            'trigger_word': None,
            'trigger_count': None,
            'trigger_ratio': None,
            'top_tags': [],
            'supporting_tags': [],
            'total_unique_tags': 0,
            'source': 'none'
        }

    def analyze_lora_structure(self, sd, metadata=None):
        """Analyze LoRA structure to determine conversion needs."""
        import math
        
        keys = list(sd.keys())
        
        # TE detection
        te_keys = [k for k in keys if categorize_key(k) == 'te']
        has_te = len(te_keys) > 0
        te_key_count = len(te_keys)
        
        # Sample keys
        sample_keys = keys[:3]
        # Metadata keys
        metadata_keys = list(metadata.keys()) if metadata else []

        # Extract training metadata (ss_* keys) with JSON parsing
        training_metadata = {}
        all_metadata = self._parse_metadata(metadata) if metadata else {}
        if metadata:
            for key, value in metadata.items():
                if key.startswith('ss_') or key.startswith('sshs_'):
                    # Attempt to parse JSON strings
                    parsed = value
                    if isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    training_metadata[key] = parsed
        
        # Detect base model
        base_model = 'Unknown'
        base_source = 'unknown'
        ss_model = training_metadata.get('ss_sd_model_name')
        if ss_model and ss_model != 'Unknown':
            base_model = ss_model
            base_source = 'ss_sd_model_name'
        else:
            ss_base = training_metadata.get('ss_base_model_version')
            if ss_base:
                if ss_base.lower() == 'zimage':
                    base_model = 'Z-Image (zimage)'
                    base_source = 'ss_base_model_version (zimage)'
                else:
                    base_model = ss_base
                    base_source = 'ss_base_model_version'
            else:
                modelspec_arch = all_metadata.get('modelspec.architecture')
                if modelspec_arch and isinstance(modelspec_arch, str):
                    if 'Z-Image' in modelspec_arch:
                        base_model = 'Z-Image'
                        base_source = 'modelspec.architecture'
                    elif 'flux' in modelspec_arch.lower():
                        base_model = 'Flux'
                        base_source = 'modelspec.architecture'
                    else:
                        base_model = modelspec_arch
                        base_source = 'modelspec.architecture'
        
        # Flatten nested ss_tag_frequency dict for tag extraction
        tag_freq = training_metadata.get('ss_tag_frequency')
        if isinstance(tag_freq, dict):
            def flatten_tag_freq(d):
                result = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        result.update(flatten_tag_freq(v))
                    elif isinstance(v, (int, float)):
                        result[k] = v
                return result
            flat = flatten_tag_freq(tag_freq)
            training_metadata['ss_tag_frequency'] = flat
        
        # Extract training_info and software fields
        training_steps = 'Unknown'
        training_epochs = 'Unknown'
        training_info = all_metadata.get('training_info')
        if isinstance(training_info, dict):
            # Try various key naming conventions
            training_steps = self._first_of(training_info, 'step', 'steps', 'total_steps')
            training_epochs = self._first_of(training_info, 'epoch', 'epochs', 'total_epochs')
        else:
            # Fallback to flattened keys (if parsing flattened them)
            training_steps = self._first_of(all_metadata, 'training_info.step', 'training_info.steps')
            # Fallback to ss_steps / ss_max_train_steps if training_steps still Unknown
            if training_steps == 'Unknown':
                training_steps = self._first_of(training_metadata, 'ss_steps', 'ss_max_train_steps')

            training_epochs = self._first_of(all_metadata, 'training_info.epoch', 'training_info.epochs')
            if training_epochs == 'Unknown':
                training_epochs = self._first_of(training_metadata, 'ss_num_epochs')
        
        # Software extraction (used for trainer name)
        software = all_metadata.get('software', 'Unknown')
        if isinstance(software, dict):
            name = software.get('name', '')
            version = software.get('version', '')
            trainer_name = f"{name} {version}".strip() if name or version else 'Unknown'
        elif isinstance(software, str):
            trainer_name = software
        else:
            trainer_name = 'Unknown'
        # Fallback to ss_network_module or modelspec.implementation if trainer_name still Unknown
        if trainer_name == 'Unknown':
            trainer_name = training_metadata.get('ss_network_module',
                                 all_metadata.get('modelspec.implementation', 'Unknown'))
        # Store in training_metadata for backward compatibility
        training_metadata['training_steps'] = training_steps
        training_metadata['training_epochs'] = training_epochs
        training_metadata['trainer_name'] = trainer_name
        
        # Extract trigger words from all_metadata (fallback if ss_tag_frequency missing)
        trigger_words, trigger_source = self._extract_trigger_words(all_metadata)
        
        # Dataset info fallback
        dataset_images_fallback = self._first_of(
            all_metadata, 'dataset_images', 'ss_dataset_info', 'ss_num_train_images',
            default='Unknown'
        )
        
        # Fallback for missing ss_num_train_images using ss_num_train_items
        if dataset_images_fallback == 'Unknown':
            dataset_images_fallback = all_metadata.get('ss_num_train_items', 'Unknown')
        # Ensure training_metadata has ss_num_train_images for report
        if training_metadata.get('ss_num_train_images', 'Unknown') == 'Unknown':
            training_metadata['ss_num_train_images'] = dataset_images_fallback
        
        # Fallback for missing ss_resolution using modelspec.resolution
        if training_metadata.get('ss_resolution', 'Unknown') == 'Unknown':
            training_metadata['ss_resolution'] = all_metadata.get('modelspec.resolution', 'Unknown')
        
        # Get hidden dimensions
        hidden_dims = set()
        ranks = []
        total_abs = 0.0
        total_sq = 0.0
        total_elements = 0
        dtype_counts = {}
        for key, tensor in sd.items():
            # Count dtype for all tensors
            dtype_counts[tensor.dtype] = dtype_counts.get(tensor.dtype, 0) + 1
            
            if len(tensor.shape) >= 2:
                hidden_dims.add(tensor.shape[0])
                hidden_dims.add(tensor.shape[1])
                # Compute rank for weight tensors
                if key.endswith('.weight'):
                    rank = safe_get_rank(tensor, key)
                    if rank > 0:
                        ranks.append(rank)
                    # Energy contribution
                    total_abs += torch.abs(tensor).sum().item()
                    total_sq += (tensor ** 2).sum().item()
                    total_elements += tensor.numel()
        
        # Determine dominant dtype (precision)
        if dtype_counts:
            dominant_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0]
            # Map to human-readable string
            dtype_map = {
                torch.float32: 'fp32',
                torch.bfloat16: 'bf16',
                torch.float16: 'fp16',
                torch.float8_e4m3fn: 'fp8',
                torch.float8_e5m2: 'fp8',
            }
            precision = dtype_map.get(dominant_dtype, str(dominant_dtype))
            # If multiple dtypes present, note mixed
            if len(dtype_counts) > 1:
                precision = f'mixed ({precision})'
        else:
            precision = 'unknown'
        
        # Detect structure types
        structure = []
        if any('double_blocks' in k for k in keys):
            structure.append('flux_double')
        if any('single_blocks' in k for k in keys):
            structure.append('flux_single')
        if any('lora_unet_layers' in k for k in keys):
            structure.append('z_image_style')
        primary_structure = structure[0] if structure else ''
        
        # Detect target ecosystem and architecture notes
        target_ecosystem, architecture_notes = self._detect_ecosystem(
            keys, training_metadata, all_metadata, primary_structure
        )
        
        # Get unique prefixes
        prefixes = set()
        for key in keys[:10]:  # Sample first 10
            parts = key.split('.')
            if parts:
                prefixes.add(parts[0])
        # Compute average rank
        avg_rank = sum(ranks) / len(ranks) if ranks else 0

        # Compute energy scores (mean absolute and RMS)
        energy_score = (total_abs / total_elements * 1000) if total_elements > 0 else 0.0
        energy_score_rms = (math.sqrt(total_sq / total_elements) * 1000) if total_elements > 0 else 0.0

        # Concept analysis
        concept_analysis = self._extract_concept_analysis(training_metadata, all_metadata)
        
        return {
            'key_count': len(keys),
            'hidden_dims': sorted(list(hidden_dims))[:10],
            'structure': structure,
            'has_alphas': any('.alpha' in k for k in keys),
            'has_te': has_te,
            'te_key_count': te_key_count,
            'te_keys_sample': te_keys[:3] if te_keys else [],
            'prefixes': list(prefixes)[:5],
            'sample_keys': sample_keys,
            'metadata_keys': metadata_keys,
            'ranks': ranks,
            'avg_rank': avg_rank,
            'primary_structure': primary_structure,
            'training_metadata': training_metadata,
            'all_metadata': all_metadata,
            'trigger_words': trigger_words,
            'trigger_source': trigger_source,
            'concept_analysis': concept_analysis,
            'dataset_images_fallback': dataset_images_fallback,
            'energy_score_rms': energy_score_rms,
            'energy_score': energy_score,
            'precision': precision,
            'target_ecosystem': target_ecosystem,
            'architecture_notes': architecture_notes,
            'base_model': base_model,
            'base_source': base_source,
            'training_steps': training_steps,
            'training_epochs': training_epochs,
            'trainer_name': trainer_name,
        }

    def _generate_forensic_report(self, info, target_format, apply_weight_scaling, bake_custom_scale, save_path, converted_key_count):
        """Generate the formatted forensic report string."""
        # Extract metadata
        tm = info.get('training_metadata', {})
        all_metadata = info.get('all_metadata', {})
        # New fields
        trigger_words = info.get('trigger_words', [])
        trigger_source = info.get('trigger_source', 'none')
        concept_analysis = info.get('concept_analysis', {})
        # TE fields
        te_mode = info.get('te_mode', 'original')
        te_weight = info.get('te_weight', 1.0)
        has_te = info.get('has_te', False)
        te_key_count = info.get('te_key_count', 0)
        energy_rms = info.get('energy_score_rms', 0.0)
        precision = info.get('precision', 'unknown')
        target_ecosystem = info.get('target_ecosystem', 'Unknown')
        architecture_notes = info.get('architecture_notes', [])
        # Compression fields
        compression_mode = info.get('compression_mode', 'original')
        final_rank = info.get('final_rank', 0)
        size_saved_mb = info.get('size_saved_mb', 0.0)
        # Base model
        base_model = info.get('base_model', 'Unknown')
        # Architecture
        arch = info.get('primary_structure', 'Unknown')
        dims = ','.join(map(str, info.get('hidden_dims', [])[:3]))
        rank = info.get('avg_rank', 0)
        # Trigger tags
        tag_freq = tm.get('ss_tag_frequency')
        tag_text = 'Unknown'
        if isinstance(tag_freq, dict):
            sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
            top_tags = [tag for tag, _ in sorted_tags[:3]]
            tag_text = ', '.join(top_tags) if top_tags else 'None'
        elif tag_freq is not None:
            tag_text = str(tag_freq)
        # Dataset
        num_images = tm.get('ss_num_train_images', 'Unknown')
        resolution = tm.get('ss_resolution', 'Unknown')
        # Training
        lr = tm.get('ss_learning_rate', 'Unknown')
        optimizer = tm.get('ss_optimizer', 'Unknown')
        epochs = tm.get('ss_num_epochs', 'Unknown')
        # Training info (from training_info/software)
        training_steps = info.get('training_steps', 'Unknown')
        training_epochs = info.get('training_epochs', 'Unknown')
        trainer_name = info.get('trainer_name', 'Unknown')
        # Energy score
        energy = info.get('energy_score', 0.0)
        # Conversion
        format_a = info.get('primary_structure', 'Unknown')
        format_b = target_format
        # Modification flags
        bake_flag = info.get('bake_alphas_flag', True)
        baking = 'Yes' if bake_flag else 'No'
        scaling = f'{bake_custom_scale}x' if apply_weight_scaling else 'None'
        # Extract additional metadata for enhanced report
        model_hash = tm.get('sshs_legacy_hash') or tm.get('sshs_model_hash') or all_metadata.get('modelspec.hash_sha256')
        network_dim = tm.get('ss_network_dim')
        network_alpha = tm.get('ss_network_alpha')
        scheduler = tm.get('ss_lr_scheduler')
        warmup = tm.get('ss_lr_warmup_steps')
        accumulation = tm.get('ss_gradient_accumulation_steps')
        checkpointing = tm.get('ss_gradient_checkpointing')
        mixed_precision = tm.get('ss_mixed_precision')
        vae = tm.get('ss_vae_name')
        seed = tm.get('ss_seed')
        # Additional metadata from modelspec and OneTrainer
        modelspec_hash = all_metadata.get('modelspec.hash_sha256')
        modelspec_date = all_metadata.get('modelspec.date')
        modelspec_sai = all_metadata.get('modelspec.sai_model_spec')
        modelspec_title = all_metadata.get('modelspec.title')
        modelspec_architecture = all_metadata.get('modelspec.architecture')
        modelspec_timestep_range = all_metadata.get('modelspec.timestep_range')
        modelspec_implementation = all_metadata.get('modelspec.implementation')
        ot_revision = all_metadata.get('ot_revision')
        ot_branch = all_metadata.get('ot_branch')
        
        # Improve primary structure detection with modelspec.architecture fallback
        if arch in ('Unknown', '') and modelspec_architecture:
            arch = modelspec_architecture
        
        # Parse ss_datasets for batch size and bucket info
        dataset_details = ''
        ss_datasets = tm.get('ss_datasets')
        if isinstance(ss_datasets, list) and len(ss_datasets) > 0:
            ds = ss_datasets[0]
            batch = ds.get('batch_size_per_device', 'Unknown')
            bucket = ds.get('enable_bucket', 'Unknown')
            bucket_no_upscale = ds.get('bucket_no_upscale', 'Unknown')
            dataset_details = f', batch {batch}, bucket {bucket}' if batch != 'Unknown' else ''
        
        # Build report lines
        lines = []
        lines.append('🛡️ --- EASY LORA STUDIO: FORENSIC REPORT --- 🛡️')
        # Method and settings summary
        detected_ecosystem = info.get('target_ecosystem', 'Unknown')
        detected_structure = info.get('primary_structure', 'Unknown')
        target_fmt = info.get('target_format', target_format)
        lines.append(f'🌐 ECOSYSTEM: {detected_ecosystem} | Structure: {detected_structure}')
        lines.append(f'🎯 TARGET FORMAT: {target_fmt}')
        summary_parts = [f'Keys: {converted_key_count}', f'Rank: {info.get("avg_rank", 0):.0f}']
        if compression_mode != 'original':
            summary_parts.append(f'Compression: {compression_mode}')
        if info.get('te_mode', 'original') != 'original':
            summary_parts.append(f'TE: {info.get("te_mode")}')
        lines.append(f'📋 SUMMARY: {" | ".join(summary_parts)}')
        lines.append(f'🔧 SETTINGS: ecosystem={detected_ecosystem} | target={target_fmt} | compression={compression_mode} | precision={precision}')
        lines.append('-' * 50)
        # Add hash to base model line if available
        if model_hash:
            # Truncate hash for display (keep first 12 chars, strip 0x)
            display_hash = str(model_hash)
            if display_hash.startswith('0x'):
                display_hash = display_hash[2:]
            if len(display_hash) > 12:
                display_hash = display_hash[:12] + '...'
            lines.append(f'📦 BASE MODEL: {base_model} (hash {display_hash})')
        else:
            lines.append(f'📦 BASE MODEL: {base_model}')
        # Add network dim and alpha if available
        rank_info = f'{rank:.0f}'
        if network_dim and network_alpha:
            rank_info = f'{rank:.0f} (dim {network_dim}, alpha {network_alpha})'
        elif network_dim:
            rank_info = f'{rank:.0f} (dim {network_dim})'
        elif network_alpha:
            rank_info = f'{rank:.0f} (alpha {network_alpha})'
        lines.append(f'🎯 ARCHITECTURE: {arch} | DIMS: {dims} | RANK: {rank_info}')
        lines.append(f'🌐 TARGET ECOSYSTEM: {target_ecosystem}')
        # Metadata line
        metadata_parts = []
        if modelspec_title:
            metadata_parts.append(f'Title: {modelspec_title}')
        if modelspec_date:
            # Strip time component if present
            date_str = modelspec_date.split('T')[0] if 'T' in modelspec_date else modelspec_date
            metadata_parts.append(f'Date: {date_str}')
        if modelspec_sai:
            metadata_parts.append(f'Spec: {modelspec_sai}')
        if modelspec_timestep_range:
            metadata_parts.append(f'Timestep: {modelspec_timestep_range}')
        if modelspec_implementation:
            # Truncate long implementation strings
            impl = modelspec_implementation
            if len(impl) > 30:
                impl = impl[:27] + '...'
            metadata_parts.append(f'Implementation: {impl}')
        if ot_revision:
            ot_info = f'OneTrainer: {ot_revision}'
            if ot_branch:
                ot_info += f' ({ot_branch})'
            metadata_parts.append(ot_info)
        if modelspec_hash and modelspec_hash != model_hash:
            # Show truncated hash
            hash_str = str(modelspec_hash)
            truncated = hash_str[:12] + '...' if len(hash_str) > 12 else hash_str
            metadata_parts.append(f'SHA‑256: {truncated}')
        if metadata_parts:
            lines.append(f'📄 METADATA: {", ".join(metadata_parts)}')
        if architecture_notes:
            lines.append(f'   📝 Architecture notes: {"; ".join(architecture_notes)}')
        lines.append(f'🏷️ TRIGGER TAGS: {tag_text}')
        # Concept analysis
        if concept_analysis.get('has_concept_metadata'):
            trigger_word = concept_analysis.get('trigger_word')
            trigger_count = concept_analysis.get('trigger_count')
            trigger_ratio = concept_analysis.get('trigger_ratio')
            total_images = tm.get('ss_num_train_images', 'Unknown')
            ratio_text = ''
            if trigger_ratio is not None:
                ratio_text = f', {trigger_ratio:.0%}'
            elif isinstance(total_images, (int, float)) and total_images > 0 and trigger_count is not None:
                ratio_text = f', {trigger_count/total_images:.0%}'
            trigger_display = f'{trigger_word}'
            if trigger_count is not None:
                trigger_display += f' ({trigger_count}'
                if isinstance(total_images, (int, float)) and total_images > 0:
                    trigger_display += f' of {total_images} images{ratio_text}'
                else:
                    trigger_display += ' appearances'
                trigger_display += ')'
            lines.append(f'🏷️ CONCEPT ANALYSIS')
            lines.append(f'   Trigger Word: {trigger_display}')
            # Top 10 tags
            top_tags = concept_analysis.get('top_tags', [])
            if top_tags:
                top_str = ', '.join(f'{tag} ({count})' for tag, count in top_tags[:10])
                lines.append(f'   Top 10 Tags: {top_str}')
            # Supporting tags
            supporting = concept_analysis.get('supporting_tags', [])
            if supporting:
                lines.append(f'   Supporting Tags: {", ".join(supporting[:5])}')
        else:
            lines.append(f'🏷️ CONCEPT ANALYSIS: Not Found')
        if trigger_words and tag_text in ('Unknown', 'None'):
            lines.append(f'   🔍 Trigger words (fallback): {", ".join(trigger_words)} (source: {trigger_source})')
        # Append dataset details if available
        dataset_line = f'📸 DATASET: {num_images} images @ {resolution}'
        if dataset_details:
            dataset_line += dataset_details
        lines.append(dataset_line)
        lines.append(f'🔬 PRECISION: {precision}')
        lines.append('-' * 50)
        lines.append(f'🧬 TRAINING: {lr} | {optimizer} | {epochs}')
        lines.append(f'🔧 TRAINING INFO: {training_steps} steps, {training_epochs} epochs (via {trainer_name})')
        # Build training config line
        config_parts = []
        if scheduler and scheduler != 'Unknown':
            config_parts.append(f'Sched {scheduler}')
        if warmup and warmup != 'Unknown' and warmup != '0':
            config_parts.append(f'Warmup {warmup}')
        if accumulation and accumulation != 'Unknown' and accumulation != '1':
            config_parts.append(f'Accum {accumulation}')
        if checkpointing and checkpointing != 'Unknown':
            config_parts.append(f'Checkpoint {checkpointing}')
        if mixed_precision and mixed_precision != 'Unknown':
            config_parts.append(f'Precision {mixed_precision}')
        if vae and vae != 'Unknown':
            config_parts.append(f'VAE {vae}')
        if seed and seed != 'Unknown':
            config_parts.append(f'Seed {seed}')
        if config_parts:
            lines.append(f'⚙️ TRAINING CONFIG: {", ".join(config_parts)}')
        lines.append(f'⚡ ENERGY SCORE: {energy:.2f} (Higher = Stronger Effect)')
        lines.append(f'📊 RMS ENERGY: {energy_rms:.2f} (spikiness indicator)')
        lines.append('-' * 50)
        lines.append(f'✅ CONVERSION: {format_a} -> {format_b}')
        lines.append(f'⚖️ MODIFICATION: Baking: {baking} | Scale: {scaling}')
        lines.append(f'🔧 COMPRESSION: {compression_mode} | Final Rank: {final_rank:.0f} | Size Saved: {size_saved_mb:.2f} MB')
        # TE status
        if has_te:
            if te_mode == "original":
                te_status = "Original"
            elif te_mode == "remove":
                te_status = "Removed"
            else:  # scale
                if te_weight == 0.0:
                    te_status = "Removed (weight=0.0)"
                else:
                    te_status = f"Scaled (×{te_weight})"
            lines.append(f'🔤 TEXT ENCODER: Detected: Yes ({te_key_count} keys) | Status: {te_status}')
        else:
            lines.append(f'🔤 TEXT ENCODER: Detected: No | Status: Not applicable')
        lines.append('-' * 50)
        if save_path:
            lines.append(f'📁 SAVED TO: {save_path}')
        else:
            lines.append('📁 SAVED TO: RAM (no file written)')
        lines.append(f'📅 DATE: {__import__("time").strftime("%Y-%m-%d %H:%M:%S")}')
        return '\n'.join(lines)
