#!/usr/bin/env python3
"""
Easy Checkpoint Studio – Modular node for precision surgery and structural conversion of full .safetensors checkpoints.
Extends the EasyLoRA ecosystem to handle SDXL, Flux, Z‑Image, etc. with streaming I/O, component stripping,
precision conversion (FP8/BF16), and forensic audit.
"""

import warnings
from pathlib import Path
import time
import json
import hashlib
from typing import Dict, Optional

import torch
from safetensors import safe_open

import folder_paths
import comfy.sd

# Local imports
try:
    from ..config import PRECISION_STUDIO, DEVICE_OPTIONS, FORMAT_OPTIONS
    from ..utils import (
        load_lora_with_metadata,
        get_experiment_temp_path,
        DeviceManager,
        ProgressTracker,
        cleanup_memory,
        get_available_ram,
        load_state_dict_as_model_objects,
        categorize_checkpoint_key,
        save_safetensors_stream,
        ThreadSafeCleanup,
        check_ram_guard,
        print_ram_delta,
        memory_guard,
        read_safetensors_header_only,
    )
except ImportError:
    from config import PRECISION_STUDIO, DEVICE_OPTIONS, FORMAT_OPTIONS
    from utils import (
        load_lora_with_metadata,
        get_experiment_temp_path,
        DeviceManager,
        ProgressTracker,
        cleanup_memory,
        get_available_ram,
        load_state_dict_as_model_objects,
        categorize_checkpoint_key,
        save_safetensors_stream,
        ThreadSafeCleanup,
        check_ram_guard,
        print_ram_delta,
        memory_guard,
        read_safetensors_header_only,
    )

# FP8 quantization — shared module (single source of truth)
from .fp8_quantizer import (
    FP8_PRESERVE_BF16_PATTERNS as _FP8_PRESERVE_BF16_PATTERNS_MODULE,
    should_preserve_bf16,
    compute_weight_scale,
    quantize_to_fp8,
    dequant_fp8_tensor,
    get_dequant_dtype,
)

# INT8 quantization — per-channel symmetric quantization for legacy GPUs
from .int8_quantizer import (
    compute_int8_scale,
    quantize_to_int8,
    dequant_int8_tensor,
    quantize_weight_to_int8_with_scales,
    should_skip_int8 as _should_skip_int8_module,
    should_preserve_fp16_for_int8 as _should_preserve_fp16_for_int8_module,
)

# SVD preservation — ALL SVD logic (patterns, thresholds, decomposition, preprocessing)
# Single source of truth: engine/svd_quantizer.py
from .svd_quantizer import (
    should_skip_svd,
    get_svd_threshold,
    apply_svd_to_tensor,
    apply_svd_preprocess,
)

# GGUF writer — block-wise quantization output for ComfyUI-GGUF
from .gguf_writer import (
    GGUFSaveWriter,
    gguf_arch_from_arch,
    is_gguf_precision,
    GGUF_SUPPORTED_ARCHS,
)

try:
    from .metadata_factory import finalize_metadata
except ImportError:
    from metadata_factory import finalize_metadata

try:
    from .checkpoint_normalizer import detect_checkpoint_architecture
except ImportError:
    try:
        from checkpoint_normalizer import detect_checkpoint_architecture
    except ImportError:
        detect_checkpoint_architecture = None  # Signal "no fallback available"

# Module-level constant for architecture display name mapping
ARCHITECTURE_DISPLAY_MAP = {
    "anima": "Anima",
    "flux": "Flux",
    "lumina2": "Lumina2",
    "sdxl": "SDXL",
    "sd15": "SD1.5",
    "z_image": "Z-Image",
}

# Suppress warnings
warnings.filterwarnings("ignore", message="lora key not loaded")


class MusubiCheckpointStudio:
    OUTPUT_NODE = True
    """
    Universal checkpoint converter: precision casting, component stripping, structural remapping,
    and forensic analysis for full .safetensors checkpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of checkpoints from ComfyUI's folder paths
        checkpoints = folder_paths.get_filename_list("checkpoints")
        default_folder = ""
        checkpoint_folders = folder_paths.get_folder_paths("checkpoints")
        if checkpoint_folders:
            default_folder = str(checkpoint_folders[0])

        return {
            "required": {
                # ── SECTION 1: INPUTS ───────────────────────────────────
                "checkpoint": (["None"] + checkpoints,),

                # ── SECTION 3: SETTINGS ─────────────────────────────────
                "strip_vae": ("BOOLEAN", {"default": False,
                                          "tooltip": "Remove VAE weights from the checkpoint"}),
                "strip_te": ("BOOLEAN", {"default": False,
                                         "tooltip": "Remove Text Encoder weights (CLIP/T5)"}),
                "strip_clip": ("BOOLEAN", {"default": False,
                                           "tooltip": "Remove CLIP visual encoder (if present)"}),

                # ── SECTION 5: HARDWARE ─────────────────────────────────
                "precision": (PRECISION_STUDIO, {
                    "default": "auto",
                    "tooltip": "Precision: int8 / int8_convrot save ~50% smaller files but require "
                               "ComfyUI-Flux2-INT8 custom node to load. ConvRot uses Hadamard "
                               "rotation for better quality at the same file size.",
                }),
                "output_format": (FORMAT_OPTIONS, {
                    "default": "safetensors",
                    "tooltip": "Output file format: safetensors (standard) or GGUF with "
                               "block-wise quantization (load with ComfyUI-GGUF nodes).",
                }),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),

                # ── SECTION 6: OUTPUT ───────────────────────────────────
                "save_trigger": ("BOOLEAN", {"default": False}),
                "filename": ("STRING", {"default": "converted_checkpoint", "multiline": False}),
            },
            "optional": {
                # ── SECTION 1: INPUTS (continued) ───────────────────────
                "checkpoint_data": ("CHECKPOINT",),

                # ── SECTION 3: SETTINGS (continued) ─────────────────────
                "keep_metadata": ("BOOLEAN", {"default": True,
                                              "tooltip": "Preserve original metadata"}),
                "svd_mode": (["none", "selective", "full"], {"default": "none",
                                     "tooltip": "SVD compression mode: none, selective (large weight matrices only), full (all weight matrices)"}),
                "svd_energy_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                                                   "tooltip": "Energy threshold for automatic rank selection (0.95 = keep 95% of energy)"}),

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
            kwargs.get('checkpoint', 'None'),
            kwargs.get('save_trigger', False),
            kwargs.get('precision', 'auto'),
            kwargs.get('device', 'auto'),
            kwargs.get('strip_vae', False),
            kwargs.get('strip_te', False),
            kwargs.get('strip_clip', False),
            kwargs.get('keep_metadata', True),
            kwargs.get('svd_mode', 'none'),
            kwargs.get('svd_energy_threshold', 0.95),
        )
        # Generate hash
        key_str = str(cache_key).encode('utf-8')
        return hashlib.md5(key_str).hexdigest()

    # ── Shared helper methods (used by both inner classes and outer methods) ──

    @staticmethod
    def _should_strip_key(key, strip_vae, strip_te, strip_clip, stripped_counts):
        """Check if a key should be stripped and increment the counter. Returns True if stripped."""
        from ..engine.key_utils import is_vae_key, is_te_key, is_clip_key

        if strip_vae and is_vae_key(key):
            stripped_counts["vae"] += 1
            return True
        if strip_te and is_te_key(key):
            stripped_counts["te"] += 1
            return True
        if strip_clip and is_clip_key(key):
            stripped_counts["clip"] += 1
            return True
        return False

    @staticmethod
    def _categorize_key_component(key, numel, component_counts, parameter_counts):
        """Categorize key via shared utility, update component + parameter counts in-place."""
        component = categorize_checkpoint_key(key)
        component_counts[component] = component_counts.get(component, 0) + 1
        parameter_counts[component] = parameter_counts.get(component, 0) + numel

    # 🔥 FP8 IMPROVEMENT: Patterns that should be preserved in BF16 instead of quantized to FP8
    #
    # Creator's FP8 selectively keeps critical layers in BF16:
    #   - Input projections: img_in, time_in, txt_in, vector_in, guidance_in
    #   - Output projections: final_layer
    #   - Modulation layers: single_stream_modulation, double_stream_modulation_*
    #   - Norm scale/bias tensors
    #
    # Patterns support two key formats:
    #   - Diffusers-style: uses '/' path separators (model/diffusion_model/...)
    #   - FLUX-style: uses '.' dot notation (double_blocks.0.img_attn...)
    # Delegated to engine/fp8_quantizer.py — single source of truth.
    _FP8_PRESERVE_BF16_PATTERNS = _FP8_PRESERVE_BF16_PATTERNS_MODULE

    @staticmethod
    def _should_preserve_bf16(key: str) -> bool:
        """Return True if this key should stay in BF16 when target precision is FP8.

        Delegated to engine/fp8_quantizer.should_preserve_bf16().
        """
        return should_preserve_bf16(key)

    @staticmethod
    def _build_quantization_metadata(tensor_list: list, fp8_dtype_str: str) -> dict:
        """Build _quantization_metadata dict describing the FP8 quantization format.

        Args:
            tensor_list: The list of tensor entries from survey, including scale tensors.
            fp8_dtype_str: "F8_E4M3" or "F8_E5M2"

        Returns:
            Dict with format_version and per-layer quantization format info,
            or empty dict if no FP8 tensors are present.
        """
        # Map our dtype strings to inference-engine format names
        format_map = {
            "F8_E4M3": "float8_e4m3fn",
            "F8_E5M2": "float8_e5m2",
        }
        fp8_format = format_map.get(fp8_dtype_str, "float8_e4m3fn")

        # Collect all FP8 weight keys (exclude scale tensors themselves)
        fp8_layers = {}
        for item in tensor_list:
            key = item.get('new_key', item.get('original_key', ''))
            dtype_str = item['dtype']
            if dtype_str in ("F8_E4M3", "F8_E5M2") and not key.endswith(('.weight_scale', '.input_scale')):
                # Strip trailing .weight/.bias to get base layer name
                for suffix in ('.weight', '.bias'):
                    if key.endswith(suffix):
                        layer_name = key[:-len(suffix)]
                        break
                else:
                    layer_name = key
                fp8_layers[layer_name] = {
                    "format": fp8_format,
                }

        if not fp8_layers:
            return {}

        return {
            "format_version": "1.0",
            "layers": fp8_layers,
        }

    @staticmethod
    def _compute_weight_scale(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        """Compute the per-tensor weight_scale for FP8 quantization.

        Delegated to engine/fp8_quantizer.compute_weight_scale().
        """
        return compute_weight_scale(tensor, target_dtype)

    @staticmethod
    def _quantize_to_fp8(tensor: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
        """Quantize tensor to FP8 using scale-then-cast (matching Creator's algorithm).

        Delegated to engine/fp8_quantizer.quantize_to_fp8().
        """
        return quantize_to_fp8(tensor, fp8_dtype)

    @staticmethod
    def _should_skip_int8(key: str) -> bool:
        """Return True if this key should skip INT8 quantization (LoRA weights etc.).

        Delegated to engine/int8_quantizer.should_skip_int8().
        """
        return _should_skip_int8_module(key)

    @staticmethod
    def _should_preserve_fp16_for_int8(key: str) -> bool:
        """Return True if this key should stay in FP16 when target precision is INT8.

        INT8 has zero mantissa bits — input projections (img_in, time_in, txt_in),
        output layers (final_layer), and modulation tensors are the most sensitive
        to quantization error. Mirrors FP8's ``should_preserve_bf16()``.

        For INT8 dequant math (``int8 * scale``), FP16's 10 mantissa bits produce
        more accurate results than BF16's 7 mantissa bits — so FP16 is universally
        correct regardless of GPU generation (Pascal/Turing/Ampere+).

        Delegated to engine/int8_quantizer.should_preserve_fp16_for_int8().
        """
        return _should_preserve_fp16_for_int8_module(key)

    @staticmethod
    def _resolve_output_path(save_folder, filename, extension=".safetensors"):
        """Determine output path with auto-increment dedup logic.

        Args:
            save_folder: Output directory path.
            filename: Base filename (may include existing extension).
            extension: Desired file extension, e.g. ``.safetensors`` or ``.gguf``.
        """
        if save_folder:
            output_folder = Path(save_folder)
        else:
            checkpoint_folders = folder_paths.get_folder_paths("checkpoints")
            output_folder = Path(checkpoint_folders[0]) if checkpoint_folders else Path.cwd()
        output_folder.mkdir(parents=True, exist_ok=True)
        base_name = filename.rsplit('.', 1)[0]  # strip any existing extension
        output_path = output_folder / f"{base_name}_converted{extension}"
        counter = 1
        while output_path.exists():
            output_path = output_folder / f"{base_name}_converted_{counter}{extension}"
            counter += 1
        return output_path

    @staticmethod
    def _format_ui_summary(tensor_count, target_dtype, quality_pct, stripped_counts, svd_info,
                           output_format="safetensors"):
        """Build a compact UI summary string."""
        if output_format.startswith("gguf_"):
            format_label = output_format.replace("gguf_", "GGUF ").upper()
            ui_summary = f"[{tensor_count} tensors, {format_label}]"
        else:
            ui_summary = f"[{tensor_count} tensors, {target_dtype}]"
        if quality_pct is not None:
            ui_summary += f" Q:{quality_pct}%"
        if any(stripped_counts.values()):
            ui_summary += f" stripped:{stripped_counts['vae']}vae{stripped_counts['te']}te{stripped_counts['clip']}clip"
        if svd_info and svd_info.get('svd_applied', False):
            rank_ratio = svd_info.get('avg_rank_ratio', 0)
            ui_summary += f" svd:{svd_info.get('compressed_layers', 0)}layers"
            if rank_ratio > 0:
                ui_summary += f",rr={rank_ratio:.3f}"
        return ui_summary

    @staticmethod
    def _build_architecture_extra(architecture: str) -> Optional[Dict[str, str]]:
        """Build extra metadata fields for architecture.

        Returns ``{"modelspec.architecture": arch}`` only for SD-family
        architectures where the SAI model spec field is meaningful.
        Returns ``None`` for Flux, Z-Image, Unknown, etc.
        """
        sd_architectures = {"SD1.5", "SDXL", "Anima", "Lumina2"}
        if architecture in sd_architectures:
            return {"modelspec.architecture": architecture}
        return None

    @staticmethod
    def _compute_size_savings_gb(base_bytes, total_parameters, target_dtype):
        """Compute size_savings_gb from base bytes and target dtype."""
        if target_dtype is None:
            return 0.0
        if target_dtype in (torch.float32, torch.float):
            new_size_bytes = base_bytes
        elif target_dtype == torch.bfloat16:
            new_size_bytes = total_parameters * 2
        elif target_dtype == torch.float16:
            new_size_bytes = total_parameters * 2
        elif target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            new_size_bytes = total_parameters * 1
        else:
            new_size_bytes = base_bytes
        return (base_bytes - new_size_bytes) / (1024**3)

    class _LazyCheckpointMapping:
        """
        A dict‑like wrapper that loads tensors from a safetensors file on demand.

        WHY THIS EXISTS (instead of using safe_open):
            safetensors.safe_open() uses mmap internally to map the entire file into
            memory. While individual tensors are only loaded on first access (lazy
            page faults), the mmap pages remain resident in RAM and cannot be reliably
            freed between batches. This causes cumulative RAM growth in workflows that
            create and destroy multiple lazy mappings (e.g., checkpoint baker with
            per-batch source reopening, or checkpoint weaver with batched merging).

            There is no safe_open API to force-release mmap pages. Closing the safe_open
            handle and deleting the object does NOT guarantee the OS will reclaim the
            pages. Over many batches, this leads to OOM.

        HOW THIS WORKS INSTEAD:
            1. Opens the file with built-in open() in binary mode (no mmap)
            2. On first access, manually parses the safetensors header (magic bytes,
               header length, JSON metadata) — ~50 KB read, negligible cost
            3. On __getitem__, seeks to the absolute data offset and reads the exact
               byte range for the requested tensor, then reconstructs a torch.Tensor
               from the bytes via torch.frombuffer
            4. Exposes close() that actually closes the binary file handle — no mmap
               pages to leak
            5. Re-opening after close re-reads the header (cheap, ~50 KB)

        LIMITATIONS:
            - This is NOT a general replacement for safe_open. It's optimized for
              sequential, per-tensor access patterns (baking, merging, conversion).
            - Full tensor data is read into memory on each access (same as safe_open
              get_tensor). For random-access patterns across very large files, mmap
              may be more efficient.
            - Thread safety: None (same as safe_open).

        Supports write-through via _write_cache for keys that are set (not from file),
        enabling compatibility with code that writes to the mapping (e.g. ComfyUI's
        model loader during temp fallback, or _assemble_output_lazy which overlays
        baked keys).
        """
        _sentinel = object()

        # Safetensors dtype string → torch.dtype mapping
        # NOTE: Imported from key_utils to eliminate fragile cross-module coupling.
        # 'F8' is included for backward compatibility with files created by older
        # versions of this node that used the wrong dtype string.
        # The correct identifier per safetensors spec is 'F8_E4M3'.
        from ..engine.key_utils import SAFETENSORS_DTYPE_MAP as _SAFETENSORS_DTYPE_MAP

        def __init__(self, filepath, metadata=None, target_dtype=None):
            self.filepath = Path(filepath)
            self.metadata = metadata or {}
            self._handle = None          # binary file handle (not safe_open)
            self._header = None          # parsed tensor headers: {name: {dtype, shape, data_offsets}}
            self._keys = None            # list of tensor keys (preserves file order)
            self._metadata_from_file = {}  # metadata from safetensors file header
            self._data_start = 0         # absolute byte offset where tensor data begins
            self._popped = set()
            self._write_cache: Optional[Dict[str, torch.Tensor]] = None
            self._target_dtype = target_dtype  # Optional uniform dtype conversion (Fix FP8)
            self._permanent_closed = False  # SSD Fix: prevents auto-reopen after permanent close

        @staticmethod
        def _is_companion_scale_key(key: str) -> bool:
            """Check if a key is a FP8 companion scale (weight_scale or input_scale).

            Companion scales are stored alongside FP8 weight tensors for
            per-channel dequantization. They must remain accessible via
            __getitem__ (for _load_fp8_scale) but must NOT appear in iteration
            methods (__iter__, keys, items, values, __len__) to prevent leaking
            as unrecognized model weights ("unet unexpected" keys in ComfyUI).
            """
            return key.endswith(('.weight_scale', '.input_scale'))

        def _ensure_open(self):
            """
            Open the safetensors file in binary mode and parse the header.

            NOTE: We intentionally do NOT use safetensors.safe_open() here because
            its mmap-based implementation cannot release pages between batches.
            See class docstring for full rationale.
            """
            # SSD Fix: prevent auto-reopen after permanent_close()
            if self._permanent_closed:
                raise RuntimeError(
                    f"Cannot reopen permanently closed mapping: {self.filepath}. "
                    "This mapping was closed after model loading completed."
                )
            if self._handle is not None:
                return
            # Open in binary mode — no mmap
            self._handle = open(self.filepath, 'rb')
            # Read header length (8 bytes, little-endian uint64)
            header_len_bytes = self._handle.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError(f"File too small or invalid safetensors: {self.filepath}")
            header_len = int.from_bytes(header_len_bytes, 'little')
            # Read and parse JSON header
            header_bytes = self._handle.read(header_len)
            if len(header_bytes) < header_len:
                raise ValueError(f"Truncated header in safetensors file: {self.filepath}")
            header = json.loads(header_bytes)
            # Extract file metadata
            self._metadata_from_file = header.get('__metadata__', {})
            # Store tensor headers only (preserves file insertion order via Python 3.7+ dict)
            self._header = {k: v for k, v in header.items() if k != '__metadata__'}
            self._keys = list(self._header.keys())
            # Data starts after the 8-byte length prefix + header payload
            self._data_start = 8 + header_len

        def __getitem__(self, key):
            # Check write cache first (keys set via __setitem__)
            if self._write_cache and key in self._write_cache:
                return self._apply_target_dtype(self._write_cache[key], key)
            self._ensure_open()
            if key not in self._header:
                raise KeyError(f"Key {key} not found in {self.filepath}")
            info = self._header[key]
            dtype_str = info['dtype']
            shape = info['shape']
            start, end = info['data_offsets']
            # Seek to the absolute data offset
            absolute_offset = self._data_start + start
            tensor_size = end - start
            self._handle.seek(absolute_offset)
            tensor_bytes = self._handle.read(tensor_size)
            dtype = self._SAFETENSORS_DTYPE_MAP[dtype_str]
            # Use bytearray for writable buffer compatibility across PyTorch versions
            tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=dtype).reshape(shape)
            return self._apply_target_dtype(tensor, key)

        def _apply_target_dtype(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
            """
            Apply uniform dtype conversion to a tensor, handling FP8 dequantization.

            When target_dtype is set and the tensor is FP8, this:
              1. Casts FP8 → target dtype (e.g. bf16) — giving scaled values
              2. Loads the per-channel scale factor from the companion
                 .weight_scale / .input_scale tensor
              3. Multiplies: tensor_out = tensor_fp8_as_bf16 * scale

            For non-FP8 dtypes, performs a simple cast.

            Returns the original tensor unchanged if target_dtype is None
            or the tensor already matches target_dtype.
            """
            if self._target_dtype is None or tensor.dtype == self._target_dtype:
                return tensor
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # Dequantize FP8→target_dtype using companion scale factors
                # Pass self as scale_store (supports __contains__ for write cache + file)
                tensor = dequant_fp8_tensor(tensor, key, self._target_dtype, self)
            elif tensor.dtype == torch.int8:
                # Dequantize INT8→target_dtype using companion .weight_scale factor
                # Same pattern as FP8: pass self as scale_store for lazy file-backed lookup
                tensor = dequant_int8_tensor(tensor, key, self._target_dtype, self)
            else:
                tensor = tensor.to(dtype=self._target_dtype)
            return tensor

        def _load_fp8_scale(self, key: str) -> Optional[torch.Tensor]:
            """
            Load the per-channel scale factor for an fp8 weight tensor.

            FP8 checkpoints store weights with per-channel quantization:
              weight key:     double_blocks.0.img_attn.proj.weight    (fp8_e4m3fn)
              weight_scale:   double_blocks.0.img_attn.proj.weight_scale  (float32)
              input_scale:    double_blocks.0.img_attn.proj.input_scale   (float32)

            The scale factor is a 1-D tensor [out_channels] that must be
            multiplied element-wise after the fp8→bf16 cast to recover the
            original weight values.

            Returns None for keys that are not per-channel-quantized weights
            (biases, norms, non-diffusion keys, or scale keys themselves).
            """
            # Scale keys are not weights — no self-referential lookup
            if key.endswith(('.weight_scale', '.input_scale')):
                return None

            # Derive the base prefix by stripping trailing .weight or .bias
            if key.endswith('.weight'):
                prefix = key[:-len('.weight')]
            elif key.endswith('.bias'):
                prefix = key[:-len('.bias')]
            else:
                return None

            # Check write cache first — companion scales stored by baking processor
            # (e.g. _assemble_output_lazy in baking_processor_baking.py) live in the
            # write cache, not the underlying file. Without this check, dequantization
            # of write-cache-backed FP8 tensors silently skips the scale factor,
            # producing noise.
            if self._write_cache:
                for suffix in ('.weight_scale', '.input_scale'):
                    scale_key = prefix + suffix
                    if scale_key in self._write_cache:
                        return self._write_cache[scale_key]

            self._ensure_open()

            # Try weight_scale first (primary per-channel scale), then input_scale
            for suffix in ('.weight_scale', '.input_scale'):
                scale_key = prefix + suffix
                if scale_key in self._header:
                    info = self._header[scale_key]
                    dtype_str = info['dtype']
                    shape = info['shape']
                    start, end = info['data_offsets']
                    offset = self._data_start + start
                    self._handle.seek(offset)
                    raw = self._handle.read(end - start)
                    dtype = self._SAFETENSORS_DTYPE_MAP[dtype_str]
                    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)

            return None

        def __setitem__(self, key, value):
            """Store a tensor in the write cache (does not modify the underlying file)."""
            if self._write_cache is None:
                self._write_cache = {}
            self._write_cache[key] = value

        def __iter__(self):
            """Iterate keys — includes .weight_scale for MixedPrecisionOps when _user_fp8_mode."""
            self._ensure_open()
            user_fp8 = getattr(self, '_user_fp8_mode', False)
            for k in self._iter_keys_unfiltered():
                if user_fp8 and k.endswith('.weight_scale'):
                    yield k  # MixedPrecisionOps needs these for FP8 dequant
                elif not self._is_companion_scale_key(k):
                    yield k

        def _iter_keys_unfiltered(self):
            """Generator: all keys (write cache + file), respecting popped status.

            Used by items() and __len__() which MUST include companion scales
            for downstream materialization (baker_node.py:465 uses dict(items())).
            """
            cached = list(self._write_cache.keys()) if self._write_cache else []
            for k in cached:
                yield k
            for k in self._keys:
                if k not in self._popped and k not in cached:
                    yield k

        def __len__(self):
            self._ensure_open()
            return sum(1 for _ in self._iter_keys_unfiltered())

        def keys(self):
            """Return list of keys — includes .weight_scale for MixedPrecisionOps when _user_fp8_mode."""
            self._ensure_open()
            user_fp8 = getattr(self, '_user_fp8_mode', False)
            return [k for k in self._iter_keys_unfiltered()
                    if (user_fp8 and k.endswith('.weight_scale'))
                    or not self._is_companion_scale_key(k)]

        def items(self):
            """Iterate (key, tensor) pairs — INCLUDES companion scales (needed by baker materialization).

            NOTE: Companion scales ARE included here because baker_node.py:465
            uses ``dict(output_sd.items())`` to materialize a _LazyCheckpointMapping
            to file, and the saved file MUST contain companion scales for subsequent
            FP8→BF16 dequantization via _LazyCheckpointMapping.
            """
            self._ensure_open()
            for k in self._iter_keys_unfiltered():
                yield k, self.__getitem__(k)

        def values(self):
            """Iterate tensors — INCLUDES companion scales (via items())."""
            for _, v in self.items():
                yield v

        def __contains__(self, key):
            if self._write_cache and key in self._write_cache:
                return True
            self._ensure_open()
            return key in self._header and key not in self._popped

        def update(self, other: Dict[str, torch.Tensor]) -> None:
            """Batch-update write cache from another dict."""
            if self._write_cache is None:
                self._write_cache = {}
            self._write_cache.update(other)

        # NOTE: Kept for dict protocol compatibility — downstream code
        # (e.g. comfy.sd.load_state_dict_guess_config) may call .get().
        def get(self, key, default=None):
            try:
                return self[key]
            except KeyError:
                return default

        # NOTE: Kept for dict protocol compatibility — comfy.sd
        # load_state_dict_guess_config is known to pop() keys from
        # the state dict during model loading.
        def pop(self, key, default=_sentinel):
            if key in self:
                value = self[key]
                self._popped.add(key)
                return value
            if default is not self._sentinel:
                return default
            raise KeyError(key)

        def close(self):
            """
            Release the binary file handle.

            Unlike safe_open.close(), this actually releases the file descriptor.
            After close(), the file can be reopened by the next __getitem__ call
            (header is re-read from disk — cheap at ~50 KB).

            Call this between batches to prevent RAM accumulation from mmap pages.

            NOTE: This is a TEMPORARY close — the file handle will be reopened
            automatically on the next __getitem__ access. For a permanent close
            that prevents reopening, call permanent_close() instead.
            """
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            # Header stays cached — keys() etc. still work after close.
            # Re-opening will re-parse the header (~50 KB, cheap).

        def permanent_close(self):
            """
            Permanently close the file handle and prevent all future reopening.

            Unlike close(), which allows the file to be reopened on the next access,
            permanent_close() sets a flag that causes _ensure_open() to raise an error.
            This prevents file handles from leaking after the mapping is no longer needed.

            Call this after model objects have been fully loaded from the mapping.
            Any subsequent __getitem__ attempts will raise RuntimeError.
            """
            self._permanent_closed = True
            self.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def __del__(self):
            # SSD Fix: Use permanent close in __del__ to prevent auto-reopen
            # if GC triggers __del__ while the mapping is temporarily closed.
            self.permanent_close()

        def copy(self):
            """Return a real dict with all tensors materialized, companion scales EXCLUDED.

            Required for dict protocol compatibility — ComfyUI's
            ``load_state_dict_guess_config`` may call .copy() on the state dict.
            Companion scales are stripped from the copy to prevent ``unet unexpected``
            key warnings during model loading.

            NOTE: This differs from ``dict(self.items())`` which INCLUDES companion
            scales (needed by baker_node.py materialization to file).
            """
            return {k: v for k, v in self.items() if not self._is_companion_scale_key(k)}

    def _parse_checkpoint_data(self, checkpoint_data):
        """
        Parse checkpoint_data input (CHECKPOINT type) into tensors dict and metadata.
        Supports:
        - tuple (state_dict, metadata?) similar to LORA data
        - dict (state_dict)
        Returns (tensors, metadata).
        """
        if checkpoint_data is None:
            return {}, {}
        if isinstance(checkpoint_data, tuple):
            # Possibly (state_dict, metadata) or (state_dict, something)
            if len(checkpoint_data) >= 1:
                state_dict = checkpoint_data[0]
                metadata = checkpoint_data[1] if len(checkpoint_data) > 1 else {}
            else:
                state_dict = {}
                metadata = {}
        elif isinstance(checkpoint_data, dict) or (
            hasattr(checkpoint_data, '__getitem__') and
            hasattr(checkpoint_data, 'keys')
        ):
            state_dict = checkpoint_data
            metadata = {}
        else:
            raise ValueError(f"Unsupported checkpoint_data type: {type(checkpoint_data)}")
        # Ensure state_dict is a dict of tensors
        if not isinstance(state_dict, dict) and not (
            hasattr(state_dict, '__getitem__') and
            hasattr(state_dict, 'keys')
        ):
            raise ValueError("Checkpoint state dict must be a dictionary or dict-like object")
        # Ensure metadata is a dict
        if not isinstance(metadata, dict):
            metadata = {}
        return state_dict, metadata

    def _estimate_quality_retention(self, target_dtype, output_format="safetensors"):
        """
        Rough estimate of quality retention after precision conversion.
        Returns integer percentage or None if unknown.
        """
        # GGUF block-wise quant quality estimates
        if output_format == "gguf_q8_0":
            return 97  # 8-bit block-wise, very high quality retention
        elif output_format == "gguf_q5_0":
            return 94  # 5-bit block-wise, good quality
        elif output_format == "gguf_q4_0":
            return 90  # 4-bit block-wise, reasonable quality
        elif target_dtype == torch.float32:
            return 100
        elif target_dtype == torch.bfloat16:
            return 99
        elif target_dtype == torch.float16:
            return 98
        elif target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return 97  # Selective BF16 preservation protects critical layers/norms
        else:
            return None

    def _analyze_checkpoint_structure(self, tensors, target_dtype=None):
        """
        Analyze checkpoint structure: architecture detection, parameter counts,
        component breakdown, size estimation.
        Returns dict with keys:
            architecture (str): "SD1.5", "SDXL", "Flux", "Z-Image", "Unknown"
            total_parameters (int): total number of parameters (billions)
            total_size_gb (float): size in GB assuming float32
            size_savings_gb (float): saved GB after conversion (if target_dtype provided)
            component_counts (dict): counts per component (VAE, UNet, Text Encoder, CLIP)
            parameter_counts (dict): parameter counts per component
        """
        component_counts = {"vae": 0, "unet": 0, "te": 0, "clip": 0, "other": 0}
        parameter_counts = {"vae": 0, "unet": 0, "te": 0, "clip": 0, "other": 0}
        total_parameters = 0
        total_size_bytes = 0

        for key, tensor in tensors.items():
            numel = tensor.numel()
            total_parameters += numel
            total_size_bytes += numel * tensor.element_size()  # actual dtype size
            # Categorize component via shared helper
            self._categorize_key_component(key, numel, component_counts, parameter_counts)

        # Architecture detection — centralized via checkpoint_normalizer
        keys_list = list(tensors.keys())
        if detect_checkpoint_architecture is not None:
            arch_raw = detect_checkpoint_architecture(keys_list)
        else:
            arch_raw = "Unknown"
        architecture = ARCHITECTURE_DISPLAY_MAP.get(arch_raw, "Unknown")

        total_size_gb = total_size_bytes / (1024**3)
        size_savings_gb = self._compute_size_savings_gb(total_size_bytes, total_parameters, target_dtype)

        return {
            "architecture": architecture,
            "total_parameters": total_parameters,
            "total_parameters_billions": total_parameters / 1e9,
            "total_size_gb": total_size_gb,
            "size_savings_gb": size_savings_gb,
            "component_counts": component_counts,
            "parameter_counts": parameter_counts,
        }

    def _generate_forensic_report_from_analysis(self, analysis, processed_tensors, target_dtype,
                                                 stripped_counts, svd_info=None,
                                                 output_format="safetensors"):
        """
        Generate standardized forensic report from a pre‑computed analysis dict
        (no access to the original tensor dict needed).

        Used by the in‑memory processing path so that ``tensors`` can be freed
        (:keyword:`del tensors`) immediately after conversion, avoiding the
        ~16 GB overlap between ``tensors`` and ``processed`` during SVD,
        metadata, save, and model‑loading phases.
        """
        summary_prefix = f"→ {len(processed_tensors)} tensors"
        return self._format_forensic_report_lines(
            total_parameters=analysis['total_parameters'],
            total_size_gb=analysis['total_size_gb'],
            size_savings_gb=analysis['size_savings_gb'],
            component_counts=analysis['component_counts'],
            stripped_counts=stripped_counts,
            architecture=analysis['architecture'],
            target_dtype=target_dtype,
            svd_info=svd_info,
            summary_prefix=summary_prefix,
            output_format=output_format,
        )

    def _format_forensic_report_lines(self, total_parameters, total_size_gb, size_savings_gb,
                                       component_counts, stripped_counts, architecture,
                                       target_dtype, svd_info=None, summary_prefix="",
                                       output_format="safetensors"):
        """
        Shared formatting for forensic report lines.
        """
        lines = []
        lines.append("🛡️ --- CHECKPOINT STUDIO: FORENSIC REPORT --- 🛡️")
        if output_format.startswith("gguf_"):
            dtype_label = output_format.replace("gguf_", "GGUF ").upper()
        else:
            dtype_label = str(target_dtype).split('.')[-1] if target_dtype else 'unknown'
        lines.append(f"🎯 PRECISION: {dtype_label}")
        stripped_parts = []
        if stripped_counts['vae'] > 0:
            stripped_parts.append('VAE')
        if stripped_counts['te'] > 0:
            stripped_parts.append('TE')
        if stripped_counts['clip'] > 0:
            stripped_parts.append('CLIP')
        strip_info = f" | Stripped: {', '.join(stripped_parts)}" if stripped_parts else ""
        svd_info_line = f" | SVD: {svd_info.get('compressed_layers', 0)} layers" if svd_info and svd_info.get('svd_applied') else ""
        lines.append(f"📋 SUMMARY: {summary_prefix} | {dtype_label}{strip_info}{svd_info_line}")
        lines.append('-' * 50)
        lines.append(f"🏗️ ARCHITECTURE: {architecture}")
        lines.append(f"📊 TOTAL PARAMETERS: {total_parameters / 1e9:.2f}B")
        lines.append(f"💾 ORIGINAL SIZE: {total_size_gb:.2f} GB")
        if size_savings_gb > 0:
            lines.append(f"💿 SIZE SAVINGS: {size_savings_gb:.2f} GB")
        # Component breakdown (optional)
        comp = component_counts
        if comp['vae'] > 0 or comp['unet'] > 0 or comp['te'] > 0 or comp['clip'] > 0:
            lines.append(f"🧱 COMPONENTS: VAE {comp['vae']}, UNet {comp['unet']}, TE {comp['te']}, CLIP {comp['clip']}, other {comp['other']}")
        # Quality retention estimate
        quality_pct = self._estimate_quality_retention(target_dtype, output_format=output_format)
        if quality_pct is not None:
            lines.append(f"✅ ESTIMATED QUALITY: {quality_pct}%")
        if stripped_counts['vae'] > 0 or stripped_counts['te'] > 0 or stripped_counts['clip'] > 0:
            lines.append(f"🗑️ STRIPPED: VAE {stripped_counts['vae']}, TE {stripped_counts['te']}, CLIP {stripped_counts['clip']}")
        if svd_info and svd_info.get('svd_applied'):
            rank_ratio = svd_info.get('avg_rank_ratio', 0)
            lines.append(f"🔧 SVD COMPRESSED: {svd_info.get('compressed_layers', 0)} layers "
                         f"(rank ratio {rank_ratio:.3f})")
        lines.append(f"📅 DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    def _convert_in_memory(self, tensors, source_metadata, save_trigger, filename,
                           precision, device, strip_vae, strip_te, strip_clip,
                           save_folder, keep_metadata,
                           svd_mode, svd_energy_threshold,
                           output_format, node_id):
        """
        Convert directly from in-memory tensor dict — NO temp file written.
        Used when ``save_trigger=False`` and RAM is sufficient.
        Falls back to temp file + lazy mapping if model loading fails.

        When *output_format* starts with ``"gguf_"``, the save path writes a GGUF
        file instead of .safetensors. Preview path returns FP16 tensors (GGUF is a
        save-only format; ComfyUI-GGUF handles GGUF loading separately).

        Returns (checkpoint, model, clip, vae, save_path, forensic_report)
        where ``save_path`` is always ``""`` (no file written) unless saving.
        """
        start_time = time.time()

        # ── Detect GGUF output format early ──
        gguf_format = is_gguf_precision(output_format)

        # Determine target device and dtype
        target_device = DeviceManager.get_device(device)
        target_dtype = DeviceManager.get_dtype(precision, target_device)
        if gguf_format:
            print(f"🔧 Target device: {target_device}, GGUF format: {output_format}")
        else:
            print(f"🔧 Target device: {target_device}, dtype: {target_dtype}")

        # ── 1. Capture forensic data BEFORE processing ──
        _forensic_analysis = self._analyze_checkpoint_structure(tensors, target_dtype)
        print(f"   🏗️ Architecture: {_forensic_analysis['architecture']}")
        print(f"   📊 {_forensic_analysis['total_parameters_billions']:.2f}B parameters, "
              f"{_forensic_analysis['total_size_gb']:.2f} GB")

        # Architecture-aware tip: Flux on old GPU — suggest T5 stripping
        if _forensic_analysis.get('architecture') == 'flux':
            if precision == "old_gpu":
                print(f"💡 Tip: Flux on old GPU — consider also enabling `strip_te=True` "
                      f"to remove the T5 text encoder (~3 GB saved)")
            elif precision == "device_optimized" and target_dtype == torch.int8:
                print(f"💡 Tip: This machine is an old GPU (CC < 8). For Flux, consider "
                      f"also enabling `strip_te=True` (~3 GB saved)")

        # ── Detect raw architecture for GGUF writer (needs "flux", "sd15", etc.) ──
        gguf_arch = None
        if gguf_format:
            keys_list = list(tensors.keys())
            if detect_checkpoint_architecture is not None:
                arch_raw = detect_checkpoint_architecture(keys_list)
            else:
                arch_raw = "Unknown"
            gguf_arch = gguf_arch_from_arch(arch_raw)
            if gguf_arch is None:
                print(f"   ⚠️ GGUF export not supported for architecture '{arch_raw}'. "
                      f"Supported: {sorted(GGUF_SUPPORTED_ARCHS)}")
                print(f"   ⚠️ Falling back to safetensors output.")
                gguf_format = False  # disable GGUF, proceed as normal safetensors
                output_format = "safetensors"
            else:
                print(f"   🏗️ GGUF architecture: {gguf_arch} (raw: {arch_raw})")

        # ── 2a. SVD preprocessing (applied to ORIGINAL float32 tensors) ──
        # This runs BEFORE quantization so SVD operates on meaningful float values,
        # not on quantized integers. Companion scales computed during quantization
        # will automatically reflect SVD-corrected values.
        # NOTE: GGUF mode also benefits from SVD preprocessing when the processed
        # tensors are converted to FP16 for preview (the GGUF file itself goes
        # through GGUFSaveWriter which does its own block-wise quantization).
        svd_info = None
        if svd_mode != "none":
            print(f"🔧 Applying SVD preprocessing ({svd_mode})...")
            svd_info = apply_svd_preprocess(
                tensors, svd_mode, svd_energy_threshold,
                target_device=target_device,
            )
            if svd_info['compressed_layers'] > 0:
                print(f"🔧 SVD processed {svd_info['compressed_layers']} layers "
                      f"(rank ratio {svd_info['avg_rank_ratio']:.3f})")

        # ── 2b. Process tensors (strip + convert dtype) ──
        processed = {}
        stripped_counts = {"vae": 0, "te": 0, "clip": 0}
        total_tensors = len(tensors)
        if gguf_format:
            # GGUF mode: strip components, convert to FP16 for preview/model loading.
            # No INT8/FP8 quantization — GGUFSaveWriter handles block-wise quant.
            # No companion scales needed.
            for key, tensor in tensors.items():
                if self._should_strip_key(key, strip_vae, strip_te,
                                          strip_clip, stripped_counts):
                    continue
                # Convert to FP16 for ComfyUI preview (GGUF is save-only format)
                t = tensor.detach().cpu()
                if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    t = t.float()
                elif t.dtype == torch.bfloat16:
                    t = t.half()  # BF16 → FP16 for ComfyUI compatibility
                processed[key] = t
            print(f"   📋 GGUF mode: {len(processed)} tensors (FP16 preview)")
        elif precision == "svd_only":
            # SVD-only mode: skip quantization, copy SVD'd float32 tensors directly.
            # SVD preprocessing already ran above (step 2a), replacing qualifying
            # tensors with low-rank float32 reconstructions in-place.
            # No companion scales, no dtype conversion, no quantization needed.
            for key, tensor in tensors.items():
                if self._should_strip_key(key, strip_vae, strip_te,
                                          strip_clip, stripped_counts):
                    continue
                processed[key] = tensor
            print(f"   📋 SVD-only mode: {len(processed)} tensors "
                  f"(float32, SVD compressed)")
        elif total_tensors > 0:
            node_prefix = f"[{node_id}] " if node_id else ""
            with ProgressTracker(total=total_tensors,
                                 desc=f"{node_prefix}Processing tensors") as convert_progress:
                for key, tensor in tensors.items():
                    # Component stripping
                    if self._should_strip_key(key, strip_vae, strip_te,
                                              strip_clip, stripped_counts):
                        convert_progress += 1
                        continue
                    new_key = key

                    # FP8: preserve BF16 for sensitive layers
                    effective_dtype = target_dtype
                    if target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        # Sensitive layers (by pattern) — the same patterns used by
                        # Creator's FP8 and aligned with INT8_PRESERVE_FP16_PATTERNS.
                        if self._should_preserve_bf16(key):
                            effective_dtype = torch.bfloat16
                        # 1D weight tensors (norms, biases, scale factors) are critical
                        # for quality and tiny in memory (~KB each). INT8 auto-preserves
                        # them in FP16 via the `tensor.dim() >= 2` guard at line 1111;
                        # FP8 mirrors this by preserving ALL dim<2 tensors in BF16,
                        # not just pattern-matched ones. This catches Anima's 169 norm
                        # weight tensors (k_norm.weight, q_norm.weight, norm_*.weight)
                        # and similar 1D parameters in all architectures.
                        elif tensor.dim() < 2:
                            effective_dtype = torch.bfloat16

                    # Determine if this tensor needs FP8 companion scales
                    # Scales must be computed from the ORIGINAL float tensor
                    # BEFORE quantization to FP8 (max/abs operations fail on fp8)
                    _needs_fp8_scales = (
                        effective_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                        and target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                        and new_key.endswith('.weight')
                    )

                    # INT8: companion scale determination
                    # Sensitive layers (input/output projections, modulation, norm
                    # weights) are preserved in FP16 — INT8's zero mantissa bits
                    # corrupt them disproportionately.
                    _needs_int8_scales = (
                        target_dtype == torch.int8
                        and effective_dtype == torch.int8
                        and new_key.endswith('.weight')
                        and tensor.dim() >= 2  # 1D weights cannot use per-channel quantization
                        and not self._should_skip_int8(key)
                        and not self._should_preserve_fp16_for_int8(key)
                    )
                    if target_dtype == torch.int8 and not _needs_int8_scales:
                        effective_dtype = torch.float16

                    # ── GPU-accelerated conversion ──────────────────────────────
                    # Move tensor to GPU before dtype conversion / FP8 quantization
                    # so that max/abs (weight scale) and quantize_to_fp8 run on GPU.
                    # Move back to CPU for storage to prevent GPU OOM accumulation.
                    moved_to_gpu = False
                    if target_device is not None and target_device.type == 'cuda':
                        tensor = tensor.to(device=target_device, non_blocking=True)
                        moved_to_gpu = True
                    # ────────────────────────────────────────────────────────────

                    # Cast dtype if needed — compute scales BEFORE quantization
                    if tensor.dtype != effective_dtype:
                        if _needs_fp8_scales:
                            weight_scale = self._compute_weight_scale(tensor, effective_dtype)
                            input_scale = self._compute_weight_scale(tensor, effective_dtype)
                            tensor = self._quantize_to_fp8(tensor, effective_dtype)
                        elif _needs_int8_scales:
                            q, wscale = quantize_to_int8(
                                tensor,
                                use_convrot=(precision == "int8_convrot"),
                            )
                            weight_scale = wscale
                            tensor = q
                        else:
                            tensor = tensor.to(dtype=effective_dtype)
                    elif _needs_fp8_scales:
                        weight_scale = self._compute_weight_scale(tensor, effective_dtype)
                        input_scale = self._compute_weight_scale(tensor, effective_dtype)
                    elif _needs_int8_scales:
                        # Per-channel scale via compute_int8_scale (applies percentile clipping
                        # for outlier reduction, same as the quantize_to_int8 path)
                        weight_scale = compute_int8_scale(tensor)

                    # Move back to CPU for storage (prevents GPU OOM from accumulating tensors)
                    if moved_to_gpu:
                        tensor = tensor.cpu()
                        if _needs_fp8_scales:
                            weight_scale = weight_scale.cpu()
                            input_scale = input_scale.cpu()
                        if _needs_int8_scales:
                            weight_scale = weight_scale.cpu()

                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()

                    # Store companion scale tensors (harmless extra keys for ComfyUI)
                    if _needs_fp8_scales:
                        # 🔥 FIX: Strip '.weight' from key so scale tensors are named
                        # `{layer}.weight_scale` instead of `{layer}.weight.weight_scale`.
                        # ComfyUI's MixedPrecisionOps expects {prefix}weight_scale.
                        base_key = new_key[:-len('.weight')]  # new_key.endswith('.weight') guaranteed
                        processed[base_key + '.weight_scale'] = weight_scale.cpu()
                        processed[base_key + '.input_scale'] = input_scale.cpu()
                    elif _needs_int8_scales:
                        base_key = new_key[:-len('.weight')]
                        processed[base_key + '.weight_scale'] = weight_scale.cpu()

                    processed[new_key] = tensor
                    convert_progress += 1
        else:
            print("⚠️ No tensors to process")

        # ── 3. Free original tensors — recover RAM before SVD/metadata ──
        del tensors
        cleanup_memory()
        print(f"   ✅ Freed original tensors — "
              f"{_forensic_analysis['total_size_gb']:.2f} GB recovered")

        # Report stripping
        if any(stripped_counts.values()):
            print(f"🔪 Stripped components: VAE {stripped_counts['vae']}, "
                  f"TE {stripped_counts['te']}, CLIP {stripped_counts['clip']}")

        # ── 3. Metadata ──
        mode = "preserve_a" if keep_metadata else "none"
        extra = self._build_architecture_extra(_forensic_analysis['architecture'])
        final_metadata = finalize_metadata(
            metadata=source_metadata,
            mode=mode,
            component="checkpoint_studio",
            extra_fields=extra,
        )

        # Quantization metadata + dequant_target hint
        # Handles both FP8 (E4M3/E5M2) and INT8 quantized formats.
        # GGUF skips this entirely — it has its own metadata in the GGUF file.
        # NOTE: dequant_target is stripped from save_metadata (internal-only key).
        # Downstream loaders get_dequant_dtype() auto-detects from GPU capability.
        if not gguf_format and target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8):
            if target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                final_metadata['dequant_target'] = 'bfloat16'
            else:  # INT8
                final_metadata['dequant_target'] = 'float16'
            print(f"   🏷️ dequant_target={final_metadata['dequant_target']}")

            # FP8-only: build quantization metadata (INT8 does not use this)
            if target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                fp8_dtype_str = 'F8_E4M3' if target_dtype == torch.float8_e4m3fn else 'F8_E5M2'
                tensor_meta_list = []
                for k, v in processed.items():
                    is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                    tensor_meta_list.append({
                        'new_key': k,
                        'dtype': fp8_dtype_str if is_fp8 else 'BF16',
                    })
                qmeta = self._build_quantization_metadata(tensor_meta_list, fp8_dtype_str)
                if qmeta:
                    final_metadata['_quantization_metadata'] = json.dumps(qmeta)
                    print(f"   📋 Added quantization metadata "
                          f"({len(qmeta.get('layers', {}))} FP8 layers)")

        # ── 4. Save or preview dispatch ─────────────────────────────────
        save_path = ""
        output_vae = not strip_vae
        output_clip = not (strip_te or strip_clip)
        model_loading_metadata = final_metadata.copy()
        # Both paths keep _quantization_metadata for MixedPrecisionOps detection.
        # Save path: uses in-memory processed dict (raw FP8 tensors + metadata).
        # Preview path: same — in-memory FP8 tensors + metadata.
        # INT8 is the only exception: strips _quantization_metadata because
        # ComfyUI does not support INT8 natively (saves native INT8 format,
        # requires ComfyUI-Flux2-INT8 to load).
        # GGUF: metadata lives inside the .gguf file, not in safetensors header.

        # Build save-time metadata: strip internal-only keys (`dequant_target`)
        # and project signature keys (`modelspec.*`) so saved files match the
        # Creator's metadata layout — no extra keys that confuse external loaders.
        save_metadata = {
            k: v for k, v in final_metadata.items()
            if k not in ('dequant_target',)
            and not k.startswith('modelspec.')
        }

        if save_trigger:
            # ── GGUF save path ───────────────────────────────────────────
            # GGUF is a save-only format. The writer handles its own block-wise
            # quantization internally, including:
            #   - dtype conversion (BF16/FP16/FP8 → FP32 → quantise)
            #   - 1D tensor preservation (F32)
            #   - Shape rearrangement for CNN models (SD1.5/SDXL)
            #   - Metadata (architecture, file type, quant version)
            if gguf_format:
                from .gguf_writer import GGUFSaveWriter, gguf_qtype_from_precision
                gguf_qtype = gguf_qtype_from_precision(output_format)
                if gguf_qtype is None:
                    print(f"   ⚠️ Unknown GGUF format '{output_format}', "
                          f"falling back to safetensors.")
                else:
                    # Determine if shape rearrangement is needed
                    # Flux/transformer models: shape_fix=False
                    # SD1.5/SDXL (CNN UNets): shape_fix=True
                    needs_shape_fix = gguf_arch in ("sd1", "sdxl")
                    writer = GGUFSaveWriter(
                        output_path="",  # will set after resolving path
                        arch_raw=gguf_arch,
                        quant_type=gguf_qtype,
                        shape_fix=needs_shape_fix,
                    )
                    # Write all tensors through the GGUF writer
                    for key, tensor in processed.items():
                        writer.add_tensor(key, tensor)
                    # Resolve output path with .gguf extension
                    output_path = self._resolve_output_path(
                        save_folder, filename, extension=".gguf"
                    )
                    writer._output_path = str(output_path)
                    writer.finalize()
                    save_path = str(output_path)
                    print(f"💾 Saved to: {save_path} (GGUF {output_format})")
                    print(f"   ✅ GGUF file — load with ComfyUI-GGUF Unet Loader node.")
                    # For model loading, return FP16 tensors (same as preview)
                    converted_checkpoint = processed
                    # Skip safetensors save logic below
                    # (fall through to model loading section)

            # ── Safetensors save path (existing logic) ───────────────────
            if not gguf_format:
                # Save path: keep _quantization_metadata in model_loading_metadata for
                # MixedPrecisionOps detection. Use in-memory processed dict directly
                # (no _LazyCheckpointMapping) so FP8 tensors reach the model loader
                # intact — matching CheckpointLoaderSimple's loading path exactly.
                if target_dtype == torch.int8:
                    # Native INT8 save — requires ComfyUI-Flux2-INT8 custom node to load.
                    is_convrot = (precision == "int8_convrot")

                    # 1. Reshape weight_scale [C] → [C, 1] for Flux2-INT8 per-row detection
                    for k in list(processed.keys()):
                        if k.endswith('.weight_scale'):
                            processed[k] = processed[k].unsqueeze(-1)

                    # 2. Generate comfy_quant metadata per INT8 weight layer
                    _convrot_group_size = 256
                    for k in list(processed.keys()):
                        if k.endswith('.weight') and processed[k].dtype == torch.int8:
                            base = k[:-len('.weight')]
                            layer_in_features = processed[k].shape[-1]
                            layer_convrot = (
                                is_convrot
                                and layer_in_features % _convrot_group_size == 0
                            )
                            conf = {"convrot": layer_convrot}
                            if layer_convrot:
                                conf["convrot_groupsize"] = _convrot_group_size
                            processed[base + '.comfy_quant'] = torch.tensor(
                                list(json.dumps(conf).encode('utf-8')), dtype=torch.uint8
                            )

                    # 3. Strip _quantization_metadata
                    save_metadata.pop('_quantization_metadata', None)
                    model_loading_metadata.pop('_quantization_metadata', None)

                    int8_count = sum(1 for v in processed.values() if v.dtype == torch.int8)
                    print(f"   ✅ INT8 native: {int8_count} INT8 tensors + companion scales. ~50% disk savings.")
                    print(f"   ⚠️  IMPORTANT: This file contains native INT8 tensors.")
                    print(f"   ⚠️  You MUST have ComfyUI-Flux2-INT8 custom node installed to load it.")
                    print(f"   ⚠️  https://github.com/BobJohnson24/ComfyUI-INT8-Fast")

                # FP8 save: strip .input_scale companion tensors before writing.
                if target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    input_scale_keys = [k for k in processed if k.endswith('.input_scale')]
                    for k in input_scale_keys:
                        del processed[k]
                    if input_scale_keys:
                        print(f"   🧹 Stripped {len(input_scale_keys)} .input_scale tensors "
                              f"(MixedPrecisionOps falls back to correct .weight_scale)")

                # Save mode: serialize to safetensors file
                output_path = self._resolve_output_path(save_folder, filename)
                save_safetensors_stream(processed, output_path, metadata=save_metadata)
                save_path = str(output_path)
                print(f"💾 Saved to: {save_path}")

                # Use in-memory processed dict directly for model loading.
                converted_checkpoint = processed
        else:
            # Preview mode: handle quantized tensor types for direct ComfyUI loading.
            # FP8: kept as-is containing FP8 tensors + .weight_scale + _quantization_metadata
            #      → model loader detects MixedPrecisionOps → uses QuantizedTensors
            # INT8: dequant to FP16 (ComfyUI does not support INT8 natively)
            # GGUF: already FP16 from processing step
            if gguf_format:
                # Already FP16 from processing step — pass directly to model loader
                converted_checkpoint = processed
                print(f"🧠 Preview — GGUF mode: FP16 tensors passed to model loader. "
                      f"(GGUF file written during save_trigger=True)")
            elif target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # Strip .input_scale to match save path behavior
                input_scale_keys = [k for k in processed if k.endswith('.input_scale')]
                for k in input_scale_keys:
                    del processed[k]
                converted_checkpoint = processed
                print(f"🧠 Preview — FP8 tensors passed directly to MixedPrecisionOps. "
                      f"Stripped {len(input_scale_keys)} .input_scale tensors.")
            elif target_dtype == torch.int8:
                is_convrot_preview = (precision == "int8_convrot")
                for k in list(processed.keys()):
                    v = processed[k]
                    if v.dtype == torch.int8:
                        deq = dequant_int8_tensor(v, k, torch.float16, processed)
                        if is_convrot_preview:
                            try:
                                import importlib
                                _convrot_mod = importlib.import_module(
                                    "custom_nodes.ComfyUI-Flux2-INT8.convrot"
                                )
                                H = _convrot_mod.build_hadamard(
                                    256, device=deq.device, dtype=deq.dtype
                                )
                                deq = _convrot_mod.rotate_weight(
                                    deq, H, group_size=256
                                )
                            except (ImportError, AttributeError):
                                print("   ⚠️ ConvRot preview: Flux2-INT8 not found, "
                                      "skipping inverse rotation.")
                        processed[k] = deq
                for k in list(processed.keys()):
                    if k.endswith('.weight_scale'):
                        del processed[k]
                mode_str = "ConvRot " if is_convrot_preview else ""
                print(f"🧠 Preview — INT8 tensors dequantized to FP16 in-memory. ({mode_str}dequant)")
                converted_checkpoint = processed
            else:
                converted_checkpoint = processed

        # ── 5. Load MODEL/CLIP/VAE objects ──────────────────────────────
        _pre_load_ram = get_available_ram()
        model, clip, vae = None, None, None
        try:
            model, clip, vae = load_state_dict_as_model_objects(
                converted_checkpoint,
                metadata=model_loading_metadata,
                output_vae=output_vae, output_clip=output_clip,
            )
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            model, clip, vae = None, None, None

        print_ram_delta(_pre_load_ram, get_available_ram(), "load")

        # ── 6. Generate forensic report ──
        forensic_report = self._generate_forensic_report_from_analysis(
            _forensic_analysis, converted_checkpoint, target_dtype,
            stripped_counts, svd_info,
            output_format=output_format,
        )

        # ── 7. UI summary ──
        if gguf_format:
            quality_pct = self._estimate_quality_retention(None, output_format=output_format)
        else:
            quality_pct = self._estimate_quality_retention(target_dtype)
        ui_summary = self._format_ui_summary(
            len(converted_checkpoint) if isinstance(converted_checkpoint, dict) else 0,
            target_dtype, quality_pct,
            stripped_counts, svd_info,
            output_format=output_format,
        )
        print(f"   {ui_summary}")

        print(f"⏱️  In-memory conversion total: {time.time() - start_time:.2f}s")

        return (converted_checkpoint, model, clip, vae, save_path, forensic_report)

    RETURN_TYPES = ("CHECKPOINT", "MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("checkpoint", "model", "clip", "vae", "output_path", "forensic_report")
    FUNCTION = "convert"
    CATEGORY = "Checkpoint/Universal"

    @staticmethod
    def _make_ram_insufficient_return(tensors_dict, meta, device, label="checkpoint_studio"):
        """Save source to temp, wrap in _LazyCheckpointMapping, return as-is
        (no conversion applied — user should increase RAM or use save_trigger=True).

        Uses GPU‑aware dequant target so the temp mapping is compatible
        with the local GPU (handles cross‑compilation scenario implicitly).
        
        ``get_experiment_temp_path()`` auto-registers cleanup via ThreadSafeCleanup,
        so callers do not need to register the temp file manually.
        """
        temp_path = get_experiment_temp_path(label)
        save_safetensors_stream(tensors_dict, temp_path, metadata=meta)
        del tensors_dict
        cleanup_memory()
        # Resolve dequant target from local GPU capability; no file metadata
        # yet, so get_dequant_dtype falls back to auto‑detect.
        _dequant_target = get_dequant_dtype(target_device=device)
        converted = MusubiCheckpointStudio._LazyCheckpointMapping(
            temp_path, meta, target_dtype=_dequant_target,
        )
        # Generate minimal forensic report (no conversion happened)
        forensic_report = (
            f"🛡️ --- CHECKPOINT STUDIO: FORENSIC REPORT --- 🛡️\n"
            f"⚠️ INSUFFICIENT RAM — no conversion applied\n"
            f"📅 DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return (converted, None, None, None, "", forensic_report)

    def _convert_from_lazy_mapping(self, tensors, metadata, studio_kwargs, device):
        """Handle ``_LazyCheckpointMapping`` input: estimate size from header,
        load into RAM if sufficient, otherwise passthrough without conversion.

        Args:
            tensors: A ``_LazyCheckpointMapping`` instance.
            metadata: Source metadata dict (may be updated with lazy mapping metadata).
            studio_kwargs: Shared kwargs dict for ``_convert_in_memory()``.
            device: Target device for dequant dtype resolution.
        """
        filepath = tensors.filepath
        _dtype_byte_map = {
            'F32': 4, 'F16': 2, 'BF16': 2,
            'F8_E4M3': 1, 'F8_E5M2': 1, 'F8': 1,
        }
        try:
            file_header, file_meta = read_safetensors_header_only(Path(filepath))
            total_bytes = 0
            for info in file_header.values():
                if isinstance(info, dict):
                    shape = info.get('shape', [])
                    elem = _dtype_byte_map.get(info.get('dtype', 'F32'), 4)
                    numel = 1
                    for d in shape:
                        numel *= d
                    total_bytes += numel * elem

            if check_ram_guard(
                total_bytes, len(file_header), 1,
                label="checkpoint_studio_lazy", use_dict=True,
            ):
                print(f"   🧠 RAM sufficient — loading file-backed tensors into memory")
                from safetensors import safe_open
                loaded = {}
                with safe_open(filepath, framework='pt') as sf:
                    for key in sf.keys():
                        loaded[key] = sf.get_tensor(key)
                # Merge metadata from lazy mapping if available
                lazy_meta = getattr(tensors, '_metadata', {}) or {}
                if lazy_meta:
                    metadata = lazy_meta
                return self._convert_in_memory(
                    tensors=loaded, source_metadata=metadata, **studio_kwargs,
                )
            else:
                print(f"   ℹ️ Insufficient RAM — passing through lazy mapping without conversion")
                return self._make_ram_insufficient_return(
                    tensors, metadata, device, label="checkpoint_studio_lazy"
                )
        except Exception as e:
            print(f"   ⚠️ Header read failed ({e}) — trying direct in-memory conversion")
            # Attempt conversion anyway; _convert_in_memory may OOM
            return self._convert_in_memory(
                tensors=tensors, source_metadata=metadata, **studio_kwargs,
            )

    def _handle_file_preview(self, path, studio_kwargs, device):
        """Preview mode for a ``.safetensors`` file: load into RAM if sufficient,
        otherwise return a lazy mapping directly on the source file.

        No temp file is created — the lazy mapping reads from the original source.
        """
        _dtype_byte_map = {
            'F32': 4, 'F16': 2, 'BF16': 2,
            'F8_E4M3': 1, 'F8_E5M2': 1, 'F8': 1,
        }
        try:
            file_header, file_metadata = read_safetensors_header_only(path)
            total_bytes = 0
            for info in file_header.values():
                if isinstance(info, dict):
                    shape = info.get('shape', [])
                    elem = _dtype_byte_map.get(info.get('dtype', 'F32'), 4)
                    numel = 1
                    for d in shape:
                        numel *= d
                    total_bytes += numel * elem

            if check_ram_guard(
                total_bytes, len(file_header), 1,
                label="checkpoint_studio_file_preview", use_dict=True,
            ):
                print(f"   🧠 RAM sufficient — loading into memory for preview (no temp file)")
                from safetensors import safe_open
                loaded = {}
                with safe_open(str(path), framework='pt') as sf:
                    for key in sf.keys():
                        loaded[key] = sf.get_tensor(key)
                return self._convert_in_memory(
                    tensors=loaded, source_metadata=file_metadata, **studio_kwargs,
                )
            else:
                print(f"   ℹ️ Insufficient RAM for preview — returning lazy mapping on source file")
                _dequant_target = get_dequant_dtype(target_device=device)
                converted = self._LazyCheckpointMapping(
                    path, file_metadata, target_dtype=_dequant_target,
                )
                forensic_report = (
                    f"🛡️ --- CHECKPOINT STUDIO: FORENSIC REPORT --- 🛡️\n"
                    f"⚠️ INSUFFICIENT RAM — no conversion applied\n"
                    f"📅 DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                return (converted, None, None, None, "", forensic_report)
        except Exception as e:
            print(f"   ⚠️ RAM check failed ({e}) — returning lazy mapping on source file")
            _dequant_target = get_dequant_dtype(target_device=device)
            converted = self._LazyCheckpointMapping(
                path, {}, target_dtype=_dequant_target,
            )
            forensic_report = (
                f"🛡️ --- CHECKPOINT STUDIO: FORENSIC REPORT --- 🛡️\n"
                f"⚠️ RAM CHECK FAILED — no conversion applied\n"
                f"📅 DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return (converted, None, None, None, "", forensic_report)

    def _handle_file_save(self, path, studio_kwargs):
        """Save mode for a ``.safetensors`` file: load into dict and convert."""
        from safetensors import safe_open
        loaded = {}
        with safe_open(str(path), framework='pt') as sf:
            for key in sf.keys():
                loaded[key] = sf.get_tensor(key)
        # Read metadata from file header
        try:
            _, file_metadata = read_safetensors_header_only(path)
        except Exception:
            file_metadata = {}
        return self._convert_in_memory(
            tensors=loaded, source_metadata=file_metadata, **studio_kwargs,
        )

    def _handle_non_safetensors_file(self, path, studio_kwargs):
        """Convert a non-safetensors checkpoint by loading into memory first."""
        tensors, metadata = load_lora_with_metadata(path)
        return self._convert_in_memory(
            tensors=tensors, source_metadata=metadata, **studio_kwargs,
        )

    def convert(self, checkpoint="None", save_trigger=False, filename="converted_checkpoint",
                precision="auto", device="auto", strip_vae=False, strip_te=False, strip_clip=False,
                checkpoint_data=None, save_folder="", keep_metadata=True,
                svd_mode="none", svd_energy_threshold=0.95,
                node_id=None,
                output_format="safetensors"):
        """
        Main conversion routine.
        All processing is routed through ``_convert_in_memory()``.
        The buggy ``_SafetensorsStreamWriter`` has been removed (was the root cause of
        safetensors corruption during FP8→FP8 re-conversion).
        """
        print("\n" + "="*50)
        print(">>> Easy Checkpoint Studio")
        print("="*50)

        # ══ Runtime precision guard — protect against stale workflow JSONs ══
        if precision not in PRECISION_STUDIO:
            print(f"   ⚠️ Precision '{precision}' is no longer available, falling back to 'auto'")
            precision = "auto"

        # ── Shared kwargs for _convert_in_memory() ──────────────────────────
        # All non-varying parameters are captured once here, eliminating
        # the repetitive parameter call signature at every dispatch branch.
        _studio_kwargs = dict(
            save_trigger=save_trigger, filename=filename,
            precision=precision, device=device,
            strip_vae=strip_vae, strip_te=strip_te, strip_clip=strip_clip,
            save_folder=save_folder, keep_metadata=keep_metadata,
            svd_mode=svd_mode, svd_energy_threshold=svd_energy_threshold,
            node_id=node_id,
            output_format=output_format,
        )

        # Determine source
        tensors = {}
        metadata = {}
        if checkpoint_data is not None:
            # Parse checkpoint_data (CHECKPOINT type)
            try:
                tensors, metadata = self._parse_checkpoint_data(checkpoint_data)
                print(f"📄 Source: CHECKPOINT data input ({len(tensors)} tensors, type={type(tensors).__name__})")
            except Exception as e:
                print(f"❌ Failed to parse checkpoint_data: {e}")
                return (None, None, None, None, "", "Invalid checkpoint_data")

            # ── Handle _LazyCheckpointMapping ────────────────────────────
            # Delegate to dedicated helper (extracted for clarity — see above).
            if isinstance(tensors, self._LazyCheckpointMapping):
                return self._convert_from_lazy_mapping(
                    tensors, metadata, _studio_kwargs, device
                )

            # ── Plain dict: check RAM guard ──────────────────────────────
            total_bytes = sum(
                t.numel() * t.element_size() for t in tensors.values()
            )
            if check_ram_guard(
                total_bytes, len(tensors), 1,
                label="checkpoint_studio", use_dict=True,
            ):
                return self._convert_in_memory(
                    tensors=tensors, source_metadata=metadata, **_studio_kwargs,
                )
            else:
                print(f"   ℹ️ RAM insufficient — saving to temp file for lazy passthrough")
                return self._make_ram_insufficient_return(tensors, metadata, device)
        else:
            if checkpoint == "None" or not checkpoint:
                print("❌ No checkpoint input provided")
                return (None, None, None, None, "", "No input")
            path = folder_paths.get_full_path("checkpoints", checkpoint)
            if path is None:
                print(f"❌ Checkpoint file not found: {checkpoint}")
                return (None, None, None, None, "", "File not found")
            print(f"📄 Source: {checkpoint}")
            print(f"📁 Path: {path}")

            if path.lower().endswith('.safetensors'):
                if not save_trigger:
                    return self._handle_file_preview(Path(path), _studio_kwargs, device)
                return self._handle_file_save(Path(path), _studio_kwargs)

            return self._handle_non_safetensors_file(Path(path), _studio_kwargs)


# Module-level alias for easy import from other engine modules
_LazyCheckpointMapping = MusubiCheckpointStudio._LazyCheckpointMapping

# For testing
if __name__ == "__main__":
    # Quick sanity check
    print("Easy Checkpoint Studio skeleton loaded.")
    print("Input types:", MusubiCheckpointStudio.INPUT_TYPES())