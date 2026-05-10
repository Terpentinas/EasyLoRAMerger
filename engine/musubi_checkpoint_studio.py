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
import os
from typing import Dict, Optional

import torch
from safetensors.torch import save_file
from safetensors import safe_open

import folder_paths
import comfy.sd

# Local imports
try:
    from ..config import PRECISION_OPTIONS, DEVICE_OPTIONS
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
    )
except ImportError:
    from config import PRECISION_OPTIONS, DEVICE_OPTIONS
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
        # Fallback: define a simple version if module not available
        def detect_checkpoint_architecture(keys):
            if any("net.blocks" in k for k in keys):
                return "Anima"
            elif any("double_blocks" in k or "single_blocks" in k for k in keys):
                return "Flux"
            elif any("cap_embedder" in k for k in keys):
                return "Lumina2"
            elif any("z_image" in k.lower() for k in keys):
                return "Z-Image"
            elif any("layers" in k and "attention" in k for k in keys):
                return "Z-Image"
            elif any("model.diffusion_model" in k for k in keys):
                return "SDXL"
            elif any("diffusion_model" in k for k in keys):
                return "SD1.5"
            return "Unknown"

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
                "precision": (PRECISION_OPTIONS, {"default": "auto"}),
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

                # ── DEBUG ───────────────────────────────────────────────
                "debug": ("BOOLEAN", {"default": False, "tooltip": "Show detailed conversion info"}),
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

    @staticmethod
    def _is_vae_key(key: str) -> bool:
        """Return True if key belongs to VAE component.

        Delegates to :func:`engine.key_utils.is_vae_key`.
        """
        from ..engine.key_utils import is_vae_key as _is_vae
        return _is_vae(key)

    @staticmethod
    def _is_te_key(key: str) -> bool:
        """Return True if key belongs to Text Encoder (CLIP/T5).

        Delegates to :func:`engine.key_utils.is_te_key`.
        """
        from ..engine.key_utils import is_te_key as _is_te
        return _is_te(key)

    @staticmethod
    def _is_clip_key(key: str) -> bool:
        """Return True if key belongs to CLIP visual encoder.

        Delegates to :func:`engine.key_utils.is_clip_key`.
        """
        from ..engine.key_utils import is_clip_key as _is_clip
        return _is_clip(key)

    # ── Shared helper methods (used by both inner classes and outer methods) ──

    @staticmethod
    def _should_strip_key(key, strip_vae, strip_te, strip_clip, stripped_counts):
        """Check if a key should be stripped and increment the counter. Returns True if stripped."""
        if strip_vae and MusubiCheckpointStudio._is_vae_key(key):
            stripped_counts["vae"] += 1
            return True
        if strip_te and MusubiCheckpointStudio._is_te_key(key):
            stripped_counts["te"] += 1
            return True
        if strip_clip and MusubiCheckpointStudio._is_clip_key(key):
            stripped_counts["clip"] += 1
            return True
        return False

    @staticmethod
    def _categorize_key_component(key, numel, component_counts, parameter_counts):
        """Categorize key via shared utility, update component + parameter counts in-place."""
        component = categorize_checkpoint_key(key)
        component_counts[component] = component_counts.get(component, 0) + 1
        parameter_counts[component] = parameter_counts.get(component, 0) + numel

    @staticmethod
    def _resolve_output_path(save_folder, filename):
        """Determine output path with auto-increment dedup logic."""
        if save_folder:
            output_folder = Path(save_folder)
        else:
            checkpoint_folders = folder_paths.get_folder_paths("checkpoints")
            output_folder = Path(checkpoint_folders[0]) if checkpoint_folders else Path.cwd()
        output_folder.mkdir(parents=True, exist_ok=True)
        base_name = filename.replace('.safetensors', '')
        output_path = output_folder / f"{base_name}_converted.safetensors"
        counter = 1
        while output_path.exists():
            output_path = output_folder / f"{base_name}_converted_{counter}.safetensors"
            counter += 1
        return output_path

    @staticmethod
    def _format_ui_summary(tensor_count, target_dtype, quality_pct, stripped_counts, svd_info):
        """Build a compact UI summary string."""
        ui_summary = f"[{tensor_count} tensors, {target_dtype}]"
        if quality_pct is not None:
            ui_summary += f" Q:{quality_pct}%"
        if any(stripped_counts.values()):
            ui_summary += f" stripped:{stripped_counts['vae']}vae{stripped_counts['te']}te{stripped_counts['clip']}clip"
        if svd_info and svd_info.get('svd_applied', False):
            ui_summary += f" svd:{svd_info.get('compressed_layers', 0)}layers"
            size_saved = svd_info.get('size_saved_mb', 0)
            if size_saved:
                ui_summary += f",{size_saved:.1f}MB"
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

    class _SafetensorsStreamWriter:
        """
        Two‑pass incremental writer for safetensors checkpoints.
        """
        @staticmethod
        def _read_header(filepath):
            """
            Read safetensors header from file, return dict with shape/dtype per key.
            """
            with open(filepath, 'rb') as f:
                magic = f.read(8)
                if magic != b'__safet':
                    raise ValueError('Not a safetensors file')
                header_size_bytes = f.read(8)
                if len(header_size_bytes) != 8:
                    raise ValueError('Invalid header size')
                header_size = int.from_bytes(header_size_bytes, 'little')
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes)
                # Filter out '__metadata__'
                tensors = {k: v for k, v in header.items() if k != '__metadata__'}
                return tensors, header.get('__metadata__', {})

        # Mapping from safetensors dtype strings to torch dtypes and element sizes
        DTYPE_MAP = {
            "F64": (torch.float64, 8),
            "F32": (torch.float32, 4),
            "F16": (torch.float16, 2),
            "BF16": (torch.bfloat16, 2),
            "F8_E4M3": (torch.float8_e4m3fn, 1),
            "F8_E5M2": (torch.float8_e5m2, 1),
            "I64": (torch.int64, 8),
            "I32": (torch.int32, 4),
            "I16": (torch.int16, 2),
            "I8": (torch.int8, 1),
            "U8": (torch.uint8, 1),
            "BOOL": (torch.bool, 1),
        }
        # Reverse mapping: torch dtype -> safetensors dtype string
        DTYPE_MAP_REVERSE = {v[0]: k for k, v in DTYPE_MAP.items()}

        def __init__(self, parent, source_path=None, output_path=None, metadata=None, options=None, tensors_dict=None):
            self.parent = parent
            self.source_path = Path(source_path) if source_path else None
            self.tensors_dict = tensors_dict  # NEW: optional in-memory dict source
            self.output_path = Path(output_path)
            self.metadata = metadata
            self.options = options
            self.header = None
            self.offsets = None
            self.total_parameters = 0
            self.component_counts = {"vae": 0, "unet": 0, "te": 0, "clip": 0, "other": 0}
            self.parameter_counts = {"vae": 0, "unet": 0, "te": 0, "clip": 0, "other": 0}
            self.stripped_counts = {"vae": 0, "te": 0, "clip": 0}
            self.architecture = "Unknown"
            self.svd_info = {"svd_applied": False, "compressed_layers": 0, "skipped_layers": 0, "total_original_params": 0, "total_compressed_params": 0, "size_saved_mb": 0.0}

        def _init_accumulators(self):
            """Reset all per-survey accumulator state."""
            self.total_parameters = 0
            self.component_counts = {"vae": 0, "unet": 0, "te": 0, "clip": 0, "other": 0}
            self.parameter_counts = {"vae": 0, "unet": 0, "te": 0, "clip": 0, "other": 0}
            self.stripped_counts = {"vae": 0, "te": 0, "clip": 0}
            self.architecture = "Unknown"
            self.svd_info = {"svd_applied": False, "compressed_layers": 0, "skipped_layers": 0, "total_original_params": 0, "total_compressed_params": 0, "size_saved_mb": 0.0}

        def _compute_output_dtype_str(self, original_dtype_str, precision, device, target_dtype_str):
            """Determine the output dtype string for a tensor based on precision settings.
            
            Preserves integer dtypes; converts float tensors based on precision/device config.
            """
            if original_dtype_str.startswith('I') or original_dtype_str.startswith('U'):
                return original_dtype_str
            if precision == 'auto' and device == 'auto':
                return original_dtype_str
            return target_dtype_str

        def _finalize_header_and_store(self, tensor_list):
            """Build header JSON from tensor_list, compute offsets with padding iteration,
            store results in self, and return (header_dict, total_parameters, component_counts).

            Shared by file-based survey() and dict-based _survey_from_dict() — eliminates
            the ~140-line code duplication risk flagged in the Phase 2 plan.
            """
            # Build placeholder header with dummy offsets (0)
            header = {}
            for item in tensor_list:
                header[item['new_key']] = {
                    "dtype": item['dtype'],
                    "shape": list(item['shape']),
                    "data_offsets": [0, item['size']]  # placeholder
                }
            # Add metadata
            if self.metadata:
                header['__metadata__'] = self.metadata

            # Compute header length and padding iteratively until stable
            MAX_ITER = 5
            for iteration in range(MAX_ITER):
                header_json = json.dumps(header, separators=(',', ':'))
                header_len = len(header_json)
                pad_len = (8 - (header_len % 8)) % 8
                header_len_padded = header_len + pad_len
                data_start = 8 + header_len_padded

                # Update header with correct data_offsets relative to data_start
                for item in tensor_list:
                    rel_start = item['offset']
                    rel_end = rel_start + item['size']
                    header[item['new_key']]['data_offsets'] = [rel_start, rel_end]

                # Recompute header JSON with updated offsets
                header_json = json.dumps(header, separators=(',', ':'))
                # Validate JSON
                try:
                    json.loads(header_json)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Generated header JSON is invalid: {e}")
                new_header_len = len(header_json)
                new_pad_len = (8 - (new_header_len % 8)) % 8
                new_header_len_padded = new_header_len + new_pad_len

                if new_header_len_padded == header_len_padded:
                    # Converged
                    break
                # Adjust header_len_padded for next iteration (relative offsets unchanged)
                header_len_padded = new_header_len_padded
            else:
                # Loop completed without convergence (should be rare)
                print(f"Warning: Header length did not converge after {MAX_ITER} iterations")

            # Now compute absolute offsets
            for item in tensor_list:
                item['offset'] += data_start

            # Final header JSON with padding
            header_json_padded = header_json + ' ' * pad_len

            # Store results
            self.header = header
            self.header_json = header_json_padded
            self.header_len_padded = header_len_padded
            self.data_start = data_start
            if tensor_list:
                first = tensor_list[0]
                print(f"[DEBUG] data_start = {data_start} (alignment {data_start % 8})")
                print(f"[DEBUG] first tensor '{first['new_key']}' offset = {first['offset']} (absolute), size = {first['size']}, dtype = {first['dtype']}")
            self.tensor_list = tensor_list

            return header, self.total_parameters, self.component_counts

        def _build_tensor_list_from_entries(self, tensor_entries, precision, device, target_dtype_str,
                                             strip_vae, strip_te, strip_clip):
            """Build sorted tensor_list with offsets from an iterable of tensor metadata entries.

            Args:
                tensor_entries: iterable of (key, shape_tuple, dtype_str, numel)
                (other params): same as survey() options

            Returns:
                List of tensor item dicts with original_key, new_key, shape, dtype, size, offset, pad_after.

            Shared helper used by both file-based survey() and dict-based _survey_from_dict().
            """
            tensor_list = []
            total_data_size = 0

            for key, shape, original_dtype_str, numel in sorted(tensor_entries, key=lambda x: x[0]):
                # Component stripping
                if self.parent._should_strip_key(key, strip_vae, strip_te, strip_clip, self.stripped_counts):
                    continue

                # No key mapping — always ComfyUI original format
                new_key = key

                # Determine output dtype
                output_dtype_str = self._compute_output_dtype_str(
                    original_dtype_str, precision, device, target_dtype_str
                )
                elem_size = self.DTYPE_MAP.get(output_dtype_str, (None, 4))[1]

                tensor_size = numel * elem_size
                pad_after = (8 - (tensor_size % 8)) % 8
                padded_size = tensor_size + pad_after
                offset = total_data_size
                total_data_size += padded_size

                # Component counting for forensic audit
                self.parent._categorize_key_component(key, numel, self.component_counts, self.parameter_counts)
                self.total_parameters += numel

                tensor_list.append({
                    "original_key": key,
                    "new_key": new_key,
                    "shape": shape,
                    "dtype": output_dtype_str,
                    "size": tensor_size,
                    "offset": offset,
                    "pad_after": pad_after,
                })

            return tensor_list

        def survey(self):
            """
            First pass: compute output shape/dtype, build header with offsets.
            Returns (header_dict, total_parameters, component_counts)
            """
            if self.tensors_dict is not None:
                return self._survey_from_dict()

            # Read header tensors and metadata
            tensors, metadata = self._read_header(self.source_path)

            # Detect architecture early (needed for metadata)
            keys_list = list(tensors.keys())
            arch_raw = detect_checkpoint_architecture(keys_list)
            self.architecture = ARCHITECTURE_DISPLAY_MAP.get(arch_raw, "Unknown")

            # Determine final metadata using unified metadata factory
            mode = "preserve_a" if self.options.get('keep_metadata', True) else "none"
            extra = self.parent._build_architecture_extra(self.architecture)
            self.metadata = finalize_metadata(
                metadata=metadata,
                mode=mode,
                component="checkpoint_studio",
                extra_fields=extra,
            )

            # Determine target dtype and element size
            precision = self.options.get('precision', 'auto')
            device = self.options.get('device', 'auto')
            target_dtype = DeviceManager.get_dtype(precision, device)
            target_dtype_str = self.DTYPE_MAP_REVERSE.get(target_dtype, "F32")

            strip_vae = self.options.get('strip_vae', False)
            strip_te = self.options.get('strip_te', False)
            strip_clip = self.options.get('strip_clip', False)

            # Reset accumulators
            self._init_accumulators()

            # Build sorted tensor entries from file header: (key, shape, dtype_str, numel)
            tensor_entries = []
            for key in sorted(tensors.keys()):
                info = tensors[key]
                shape = tuple(info['shape'])
                dtype_str = info['dtype']
                numel = 1
                for dim in shape:
                    numel *= dim
                tensor_entries.append((key, shape, dtype_str, numel))

            # Build tensor list with offsets (shared with dict path)
            tensor_list = self._build_tensor_list_from_entries(
                tensor_entries, precision, device, target_dtype_str,
                strip_vae, strip_te, strip_clip
            )

            # Build header from tensor list (shared with dict path)
            return self._finalize_header_and_store(tensor_list)

        def _survey_from_dict(self):
            """
            Dict-source survey: build header from in-memory tensor dict metadata.
            No tensor data is materialized — only .shape, .dtype, .numel() are accessed.
            Metadata is finalized via the unified factory (same as file-based survey).
            """
            tensors = self.tensors_dict

            # Detect architecture early (needed for metadata)
            keys_list = list(tensors.keys())
            arch_raw = detect_checkpoint_architecture(keys_list)
            self.architecture = ARCHITECTURE_DISPLAY_MAP.get(arch_raw, "Unknown")

            # Finalize metadata using unified factory (same as file-based survey)
            mode = "preserve_a" if self.options.get('keep_metadata', True) else "none"
            extra = self.parent._build_architecture_extra(self.architecture)
            self.metadata = finalize_metadata(
                metadata=self.metadata if self.metadata else {},
                mode=mode,
                component="checkpoint_studio",
                extra_fields=extra,
            )

            precision = self.options.get('precision', 'auto')
            device = self.options.get('device', 'auto')
            target_dtype = DeviceManager.get_dtype(precision, device)
            target_dtype_str = self.DTYPE_MAP_REVERSE.get(target_dtype, "F32")

            strip_vae = self.options.get('strip_vae', False)
            strip_te = self.options.get('strip_te', False)
            strip_clip = self.options.get('strip_clip', False)

            # Reset accumulators
            self._init_accumulators()

            # Build sorted tensor entries from dict: (key, shape, dtype_str, numel)
            # Access .shape, .dtype, .numel() — metadata only, no tensor data materialization
            tensor_entries = []
            for key in sorted(tensors.keys()):
                tensor = tensors[key]
                shape = tuple(tensor.shape)
                # Map torch dtype to safetensors dtype string
                original_dtype_str = self.DTYPE_MAP_REVERSE.get(tensor.dtype, "F32")
                numel = tensor.numel()
                tensor_entries.append((key, shape, original_dtype_str, numel))

            # Build tensor list with offsets (shared with file path)
            tensor_list = self._build_tensor_list_from_entries(
                tensor_entries, precision, device, target_dtype_str,
                strip_vae, strip_te, strip_clip
            )

            # Build header from tensor list (shared with file path)
            return self._finalize_header_and_store(tensor_list)

        def weave(self):
            """
            Second pass: load, process, write tensors sequentially.
            """
            # Ensure survey has been called
            if not hasattr(self, 'header_json'):
                raise RuntimeError('Survey must be called before weave')

            if self.tensors_dict is not None:
                return self._weave_from_dict()

            # Open source file for reading tensors one by one
            source_path_str = str(self.source_path)

            # Open output file for writing
            with open(self.output_path, 'wb', buffering=0) as out_f:
                # Write header length (8 bytes little‑endian)
                header_len = self.header_len_padded
                out_f.write(header_len.to_bytes(8, 'little'))
                # Write header JSON (with padding)
                out_f.write(self.header_json.encode('utf-8'))

                # Open source safetensors file for reading
                with safe_open(source_path_str, framework='pt') as sf:
                    # Iterate over tensors in the same order as survey
                    for item in self.tensor_list:
                        original_key = item['original_key']
                        new_key = item['new_key']
                        offset = item['offset']
                        dtype_str = item['dtype']

                        # Load tensor (only this tensor)
                        tensor = sf.get_tensor(original_key)

                        # Determine target dtype based on stored dtype_str (preserves integer dtypes)
                        target_torch_dtype = self.DTYPE_MAP.get(dtype_str, (torch.float32, None))[0]
                        if tensor.dtype != target_torch_dtype:
                            tensor = tensor.to(dtype=target_torch_dtype)
                        # Move to target device if needed (with VRAM guard)
                        target_device = DeviceManager.get_device(self.options.get('device', 'auto'))
                        if tensor.device != target_device:
                            if target_device.type != "cpu":
                                free_vram = DeviceManager.get_free_vram(target_device)
                                if free_vram is not None:
                                    tensor_bytes = tensor.numel() * tensor.element_size()
                                    needed_bytes = tensor_bytes * 3
                                    if free_vram < needed_bytes:
                                        short_key = original_key[-48:]
                                        print(f"         ⚠️  VRAM guard: keeping [{short_key}] on CPU "
                                              f"(need {needed_bytes / (1024**2):.0f} MB, "
                                              f"free {free_vram / (1024**2):.0f} MB)")
                                        target_device = torch.device("cpu")
                            tensor = tensor.to(device=target_device)

                        # Apply SVD compression if requested
                        svd_mode = self.options.get('svd_mode', 'none')
                        if svd_mode != 'none':
                            svd_tensors, svd_info = self.parent._apply_svd_to_checkpoint(
                                {new_key: tensor}, svd_mode,
                                self.options.get('svd_energy_threshold', 0.95)
                            )
                            tensor = svd_tensors[new_key]
                            self.svd_info['svd_applied'] = self.svd_info['svd_applied'] or svd_info['svd_applied']
                            self.svd_info['compressed_layers'] += svd_info['compressed_layers']
                            self.svd_info['skipped_layers'] += svd_info['skipped_layers']
                            self.svd_info['total_original_params'] += svd_info['total_original_params']
                            self.svd_info['total_compressed_params'] += svd_info['total_compressed_params']
                            self.svd_info['size_saved_mb'] += svd_info['size_saved_mb']

                        # Ensure tensor is contiguous and on CPU for writing
                        if not tensor.is_contiguous():
                            tensor = tensor.contiguous()
                        tensor_cpu = tensor.cpu()

                        # Write tensor bytes at the pre‑calculated offset
                        current_pos = out_f.tell()
                        if current_pos < offset:
                            padding = offset - current_pos
                            if padding > 0:
                                print(f"[DEBUG PADDING] Writing {padding} zero bytes before tensor {new_key}")
                                out_f.write(b'\x00' * padding)
                        elif current_pos > offset:
                            print(f"[WARNING] File position {current_pos} ahead of offset {offset}, seeking back")
                            out_f.seek(offset)
                        arr = tensor_cpu.numpy()
                        out_f.write(arr.tobytes())
                        pad_after = item.get('pad_after', 0)
                        if pad_after:
                            out_f.write(b'\x00' * pad_after)
                        tensor_size = arr.nbytes
                        print(f"[DEBUG PADDING] tensor {new_key} size {tensor_size}")
                        out_f.flush()

                        del tensor, tensor_cpu, arr
                        cleanup_memory()

                    out_f.flush()
                    os.fsync(out_f.fileno())
                    self._validate_safetensors_file(self.output_path)

            print(f"✅ Stream writing completed: {self.output_path}")

        def _weave_from_dict(self):
            """
            Dict-source weave: read tensors from in-memory dict instead of safe_open,
            process (dtype/device/SVD), write sequentially, then `del` the original
            key from the dict to best-effort free memory during iteration.
            """
            if not hasattr(self, 'header_json'):
                raise RuntimeError('Survey must be called before weave')

            with open(self.output_path, 'wb', buffering=0) as out_f:
                # Write header (same as file-based weave)
                out_f.write(self.header_len_padded.to_bytes(8, 'little'))
                out_f.write(self.header_json.encode('utf-8'))

                for item in self.tensor_list:
                    original_key = item['original_key']
                    new_key = item['new_key']
                    offset = item['offset']
                    dtype_str = item['dtype']

                    # Read tensor from in-memory dict
                    tensor = self.tensors_dict[original_key]

                    # Dtype/device conversion (same logic as file-based weave)
                    target_torch_dtype = self.DTYPE_MAP.get(dtype_str, (torch.float32, None))[0]
                    if tensor.dtype != target_torch_dtype:
                        tensor = tensor.to(dtype=target_torch_dtype)
                    target_device = DeviceManager.get_device(self.options.get('device', 'auto'))
                    if tensor.device != target_device:
                        if target_device.type != "cpu":
                            free_vram = DeviceManager.get_free_vram(target_device)
                            if free_vram is not None:
                                tensor_bytes = tensor.numel() * tensor.element_size()
                                needed_bytes = tensor_bytes * 3
                                if free_vram < needed_bytes:
                                    short_key = original_key[-48:]
                                    print(f"         ⚠️  VRAM guard: keeping [{short_key}] on CPU "
                                          f"(need {needed_bytes / (1024**2):.0f} MB, "
                                          f"free {free_vram / (1024**2):.0f} MB)")
                                    target_device = torch.device("cpu")
                        tensor = tensor.to(device=target_device)

                    # SVD (same logic as file-based weave)
                    svd_mode = self.options.get('svd_mode', 'none')
                    if svd_mode != 'none':
                        svd_tensors, svd_info = self.parent._apply_svd_to_checkpoint(
                            {new_key: tensor}, svd_mode,
                            self.options.get('svd_energy_threshold', 0.95)
                        )
                        tensor = svd_tensors[new_key]
                        self.svd_info['svd_applied'] = self.svd_info['svd_applied'] or svd_info['svd_applied']
                        self.svd_info['compressed_layers'] += svd_info['compressed_layers']
                        self.svd_info['skipped_layers'] += svd_info['skipped_layers']
                        self.svd_info['total_original_params'] += svd_info['total_original_params']
                        self.svd_info['total_compressed_params'] += svd_info['total_compressed_params']
                        self.svd_info['size_saved_mb'] += svd_info['size_saved_mb']

                    # Ensure tensor is contiguous and on CPU for writing
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    tensor_cpu = tensor.cpu()

                    # Write at pre-calculated offset
                    current_pos = out_f.tell()
                    if current_pos < offset:
                        padding = offset - current_pos
                        if padding > 0:
                            out_f.write(b'\x00' * padding)
                    elif current_pos > offset:
                        out_f.seek(offset)
                    arr = tensor_cpu.numpy()
                    out_f.write(arr.tobytes())
                    pad_after = item.get('pad_after', 0)
                    if pad_after:
                        out_f.write(b'\x00' * pad_after)
                    out_f.flush()

                    # Best-effort: delete original key from dict to free memory during iteration
                    # (Finding 13 — incremental memory reduction during weave)
                    if original_key in self.tensors_dict:
                        del self.tensors_dict[original_key]

                    del tensor, tensor_cpu, arr
                    cleanup_memory()

                out_f.flush()
                os.fsync(out_f.fileno())
                self._validate_safetensors_file(self.output_path)

            print(f"✅ Dict stream writing completed: {self.output_path}")

        def _validate_safetensors_file(self, filepath):
            """
            Validate the safetensors file after writing:
            - Read header length and ensure it matches the computed length.
            - Parse JSON header and ensure it's valid.
            - Verify that each tensor's data_offsets are within file bounds.
            - Verify 8‑byte alignment of each tensor offset.
            """
            with open(filepath, 'rb') as f:
                # Read header length
                header_len_bytes = f.read(8)
                if len(header_len_bytes) != 8:
                    raise RuntimeError(f"File too short: {filepath}")
                header_len = int.from_bytes(header_len_bytes, 'little')
                # Read header JSON
                header_json_bytes = f.read(header_len)
                if len(header_json_bytes) != header_len:
                    raise RuntimeError(f"Header length mismatch: expected {header_len}, got {len(header_json_bytes)}")
                # Decode and parse JSON
                try:
                    header = json.loads(header_json_bytes.decode('utf-8'))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Invalid JSON in header: {e}")
                # Ensure header is a dict
                if not isinstance(header, dict):
                    raise RuntimeError("Header is not a JSON object")
                # Remove __metadata__ if present (it's not a tensor)
                tensor_headers = {k: v for k, v in header.items() if k != '__metadata__'}
                # Compute data start offset
                data_start = 8 + header_len
                # Get file size
                f.seek(0, 2)  # seek to end
                file_size = f.tell()
                for key, info in tensor_headers.items():
                    if 'data_offsets' not in info:
                        raise RuntimeError(f"Tensor {key} missing data_offsets")
                    start, end = info['data_offsets']
                    # Verify offsets are relative to data_start
                    absolute_start = data_start + start
                    absolute_end = data_start + end
                    if absolute_start < data_start:
                        raise RuntimeError(f"Tensor {key} start offset is negative relative")
                    if absolute_end > file_size:
                        raise RuntimeError(f"Tensor {key} ends beyond file size: {absolute_end} > {file_size}")
                    # Verify 8‑byte alignment (warning only, source checkpoints may be misaligned)
                    if absolute_start % 8 != 0:
                        print(f"[WARNING] Tensor {key} start offset not 8‑byte aligned: {absolute_start}")
                # Success
                return True

    class _LazyCheckpointMapping:
        """
        A dict‑like wrapper that loads tensors from a safetensors file on demand.
        """
        _sentinel = object()
        def __init__(self, filepath, metadata=None):
            self.filepath = Path(filepath)
            self.metadata = metadata or {}
            self._handle = None
            self._keys = None
            self._popped = set()
        
        def _ensure_open(self):
            if self._handle is None:
                self._handle = safe_open(str(self.filepath), framework='pt')
        
        def __getitem__(self, key):
            self._ensure_open()
            try:
                return self._handle.get_tensor(key)
            except Exception as e:
                raise KeyError(f"Key {key} not found in {self.filepath}") from e
        
        def __iter__(self):
            self._ensure_open()
            return iter([k for k in self._handle.keys() if k not in self._popped])
        
        def __len__(self):
            self._ensure_open()
            return len([k for k in self._handle.keys() if k not in self._popped])
        def keys(self):
            self._ensure_open()
            return [k for k in self._handle.keys() if k not in self._popped]

        def items(self):
            """Iterate over (key, tensor) pairs, skipping popped keys."""
            self._ensure_open()
            for k in self._handle.keys():
                if k not in self._popped:
                    yield k, self._handle.get_tensor(k)

        def values(self):
            """Iterate over tensors, skipping popped keys."""
            for _, v in self.items():
                yield v

        def __contains__(self, key):
            self._ensure_open()
            return key in self._handle.keys() and key not in self._popped

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
            if self._handle is not None:
                if hasattr(self._handle, 'close'):
                    self._handle.close()
                self._handle = None
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()
        
        def __del__(self):
            self.close()

    def _apply_svd_to_checkpoint(self, tensors, svd_mode, energy_threshold,
                                  target_device: Optional[torch.device] = None):
        """
        Apply SVD compression to weight tensors in the checkpoint.
        Returns (compressed_tensors, compression_info)

        When ``target_device`` is provided and is CUDA, each tensor is moved
        one-at-a-time to GPU for accelerated SVD, then moved back to CPU before
        processing the next tensor. This avoids loading all checkpoint weights
        onto GPU simultaneously (key VRAM optimization). A VRAM guard checks
        that sufficient free memory exists before each GPU SVD.
        """
        if svd_mode == "none":
            return tensors, {"svd_applied": False, "compressed_layers": 0}
        
        total_original_params = 0
        total_compressed_params = 0
        compressed_layers = 0
        skipped_layers = 0
        
        # Minimum dimension threshold for SVD (skip tiny matrices)
        SVD_MIN_DIM = 128
        
        for key, tensor in tensors.items():
            # Decide if tensor is a weight matrix suitable for SVD
            is_weight = key.endswith('.weight') and tensor.ndim == 2
            if not is_weight:
                # Keep original tensor (no change)
                continue
            
            out_features, in_features = tensor.shape
            
            # Heuristic for selective mode: only compress large matrices
            if svd_mode == "selective" and (out_features <= 1024 or in_features <= 1024):
                skipped_layers += 1
                continue
            
            # Global shape guard: skip tiny matrices regardless of mode
            if out_features < SVD_MIN_DIM and in_features < SVD_MIN_DIM:
                skipped_layers += 1
                continue
            
            # Determine if rank would be full rank (no compression benefit)
            min_dim = min(out_features, in_features)
            # For selective/full mode we'll compute rank later, but we can still skip if min_dim == 1
            if min_dim == 1:
                skipped_layers += 1
                continue
            
            # Keep tensor on its current device, convert to float32 for SVD
            orig_device = tensor.device
            orig_dtype = tensor.dtype
            tensor_fp32 = tensor.float()  # same device initially
            
            # ── VRAM-aware per-tensor GPU SVD ─────────────────────────────────
            # Move ONE tensor to GPU for SVD, then move result back to CPU
            # before processing the next tensor. This avoids loading all
            # checkpoint weights onto GPU simultaneously.
            moved_to_gpu_for_svd = False
            if target_device is not None and target_device.type != "cpu":
                free_vram = DeviceManager.get_free_vram(target_device)
                tensor_bytes = tensor_fp32.numel() * tensor_fp32.element_size()
                # SVD needs ~3× tensor size for working memory (U, S, Vh)
                needed_bytes = tensor_bytes * 3
                if free_vram is not None and free_vram >= needed_bytes:
                    if tensor_fp32.device != target_device:
                        tensor_fp32 = tensor_fp32.to(target_device)
                        moved_to_gpu_for_svd = True
                        short_key = key[-48:]
                        print(f"      ⚡ GPU SVD [{short_key}] "
                              f"{out_features}×{in_features} | "
                              f"{tensor_bytes / (1024**2):.1f} MB → "
                              f"free VRAM {free_vram / (1024**2):.0f} MB")
                elif free_vram is not None:
                    short_key = key[-48:]
                    print(f"      ⚠️ CPU SVD [{short_key}] "
                          f"({out_features}×{in_features}, "
                          f"need ≥{needed_bytes / (1024**2):.0f} MB, "
                          f"free VRAM {free_vram / (1024**2):.0f} MB)")
            
            try:
                U, S, Vh = torch.linalg.svd(tensor_fp32, full_matrices=False)
            except Exception as e:
                print(f"[WARNING] SVD failed for {key}: {e}, skipping")
                skipped_layers += 1
                continue
            
            # Determine target rank
            total_energy = torch.sum(S ** 2)
            # auto rank based on energy threshold
            cumulative = torch.cumsum(S ** 2, dim=0)
            k = torch.searchsorted(cumulative, energy_threshold * total_energy).item() + 1
            k = min(k, len(S))
            # Ensure at least rank 1
            k = max(k, 1)
            
            # Skip if compressed representation would be larger than original
            original_params = out_features * in_features
            compressed_params = out_features * k + k * in_features
            if compressed_params >= original_params:
                skipped_layers += 1
                continue
            
            # Truncate and reconstruct
            sqrt_Sk = torch.sqrt(S[:k])
            reconstructed = (U[:, :k] * sqrt_Sk) @ (sqrt_Sk[:, None] * Vh[:k, :])
            # Move back to CPU if we accelerated on GPU, then cast to original dtype
            if moved_to_gpu_for_svd:
                reconstructed = reconstructed.cpu()
            reconstructed = reconstructed.to(dtype=orig_dtype)
            
            # Replace tensor in-place
            tensors[key] = reconstructed
            compressed_layers += 1
            total_original_params += out_features * in_features
            total_compressed_params += out_features * k + k * in_features  # low-rank representation size
            
            if svd_mode == "selective" or svd_mode == "full":
                energy_ratio = torch.sum(S[:k]**2) / total_energy
                print(f"[SVD] compressed {key}: {out_features}x{in_features} -> rank {k} (energy {energy_ratio:.3f})")
            
            # Delete intermediate variables to free memory (only for large tensors)
            if out_features * in_features > 1024 * 1024:  # 1M parameters
                cleanup_memory(U, S, Vh, sqrt_Sk, tensor_fp32)
        
        compression_info = {
            "svd_applied": compressed_layers > 0,
            "compressed_layers": compressed_layers,
            "skipped_layers": skipped_layers,
            "total_original_params": total_original_params,
            "total_compressed_params": total_compressed_params,
            "size_saved_mb": (total_original_params - total_compressed_params) * 4 / (1024 * 1024),
        }
        return tensors, compression_info

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

    def _estimate_quality_retention(self, target_dtype):
        """
        Rough estimate of quality retention after precision conversion.
        Returns integer percentage or None if unknown.
        """
        if target_dtype == torch.float32:
            return 100
        elif target_dtype == torch.bfloat16:
            return 99
        elif target_dtype == torch.float16:
            return 98
        elif target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return 95
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
        arch_raw = detect_checkpoint_architecture(keys_list)
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

    def _generate_forensic_report_from_analysis(self, analysis, processed_tensors, target_dtype, stripped_counts, svd_info=None):
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
        )

    def _generate_forensic_report_from_survey(self, total_parameters, component_counts, parameter_counts, stripped_counts, architecture, target_dtype, svd_info=None):
        """
        Generate standardized forensic report using pre‑computed survey metrics (streaming path).
        """
        total_tensors = sum(component_counts.values())
        total_size_gb = total_parameters * 4 / (1024**3)
        # Size savings based on target dtype
        size_savings_gb = self._compute_size_savings_gb(total_parameters * 4, total_parameters, target_dtype)
        return self._format_forensic_report_lines(
            total_parameters=total_parameters,
            total_size_gb=total_size_gb,
            size_savings_gb=size_savings_gb,
            component_counts=component_counts,
            stripped_counts=stripped_counts,
            architecture=architecture,
            target_dtype=target_dtype,
            svd_info=svd_info,
            summary_prefix=f"{total_tensors} tensors",
        )

    def _format_forensic_report_lines(self, total_parameters, total_size_gb, size_savings_gb,
                                       component_counts, stripped_counts, architecture,
                                       target_dtype, svd_info=None, summary_prefix=""):
        """
        Shared formatting for forensic report lines.
        """
        lines = []
        lines.append("🛡️ --- CHECKPOINT STUDIO: FORENSIC REPORT --- 🛡️")
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
        quality_pct = self._estimate_quality_retention(target_dtype)
        if quality_pct is not None:
            lines.append(f"✅ ESTIMATED QUALITY: {quality_pct}%")
        if stripped_counts['vae'] > 0 or stripped_counts['te'] > 0 or stripped_counts['clip'] > 0:
            lines.append(f"🗑️ STRIPPED: VAE {stripped_counts['vae']}, TE {stripped_counts['te']}, CLIP {stripped_counts['clip']}")
        if svd_info and svd_info.get('svd_applied'):
            lines.append(f"🔧 SVD COMPRESSED: {svd_info.get('compressed_layers', 0)} layers, {svd_info.get('size_saved_mb', 0):.2f} MB saved")
        lines.append(f"📅 DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    def _convert_with_streaming(self, source_path, save_trigger, filename, precision, device,
                                strip_vae, strip_te, strip_clip, save_folder, keep_metadata,
                                svd_mode, svd_energy_threshold, debug, node_id):
        """
        Perform conversion via streaming writer (flat‑RAM guarantee).
        Returns (checkpoint, output, forensic_report).
        """
        start_time = time.time()

        # Determine target device and dtype
        target_device = DeviceManager.get_device(device)
        target_dtype = DeviceManager.get_dtype(precision, target_device)

        # Build options dict
        options = {
            'precision': precision,
            'device': device,
            'strip_vae': strip_vae,
            'strip_te': strip_te,
            'strip_clip': strip_clip,
            'svd_mode': svd_mode,
            'svd_energy_threshold': svd_energy_threshold,
            'keep_metadata': keep_metadata,
        }

        # Determine output path
        if save_trigger:
            output_path = self._resolve_output_path(save_folder, filename)
        else:
            output_path = get_experiment_temp_path("checkpoint_studio")
            output_path = output_path.with_suffix('.safetensors')
            # Register temp file for cleanup (Finding 12 — get_experiment_temp_path
            # does NOT auto-register; explicit registration required)
            ThreadSafeCleanup.register_temp_file(output_path)

        # Create streaming writer
        writer = self._SafetensorsStreamWriter(
            parent=self,
            source_path=source_path,
            output_path=output_path,
            metadata={},  # will be populated from source metadata
            options=options,
        )
        # First pass: survey
        header, total_parameters, component_counts = writer.survey()
        # Metadata already finalized in survey, use writer.metadata
        final_metadata = writer.metadata
        # Second pass: weave
        writer.weave()

        # Wrap output file in lazy mapping
        converted_checkpoint = self._LazyCheckpointMapping(output_path, final_metadata)

        # Generate forensic report from survey metrics
        forensic_report = self._generate_forensic_report_from_survey(
            total_parameters=writer.total_parameters,
            component_counts=writer.component_counts,
            parameter_counts=writer.parameter_counts,
            stripped_counts=writer.stripped_counts,
            architecture=writer.architecture,
            target_dtype=target_dtype,
            svd_info=writer.svd_info,  # SVD info aggregated across tensors
        )

        # UI summary
        quality_pct = self._estimate_quality_retention(target_dtype)
        ui_summary = self._format_ui_summary(len(writer.tensor_list), target_dtype, quality_pct, writer.stripped_counts, writer.svd_info)

        # Load MODEL/CLIP/VAE objects from lazy mapping
        output_vae = not strip_vae
        output_clip = not (strip_te or strip_clip)
        try:
            model, clip, vae = load_state_dict_as_model_objects(
                converted_checkpoint,
                metadata=final_metadata,
                output_vae=output_vae,
                output_clip=output_clip,
            )
        except Exception as e:
            print(f"⚠️ Failed to load state dict as model objects: {e}")
            model, clip, vae = None, None, None

        save_path = str(output_path) if save_trigger else ""
        total_time = time.time() - start_time
        if debug:
            print(f"⏱️  Performance timings: streaming total {total_time:.2f}s")

        return (converted_checkpoint, model, clip, vae, save_path, forensic_report)

    def _convert_from_dict(self, tensors, source_metadata, save_trigger, filename,
                           precision, device, strip_vae, strip_te, strip_clip,
                           save_folder, keep_metadata,
                           svd_mode, svd_energy_threshold, debug, node_id):
        """
        Convert directly from an in-memory tensor dict — no intermediate temp file.
        Mirrors _convert_with_streaming() exactly in structure; only the writer
        creation differs (passes tensors_dict instead of source_path).

        Returns (checkpoint, model, clip, vae, save_path, forensic_report).
        """
        start_time = time.time()

        # Determine target device and dtype
        target_device = DeviceManager.get_device(device)
        target_dtype = DeviceManager.get_dtype(precision, target_device)

        # Build options dict (same as _convert_with_streaming)
        options = {
            'precision': precision,
            'device': device,
            'strip_vae': strip_vae,
            'strip_te': strip_te,
            'strip_clip': strip_clip,
            'svd_mode': svd_mode,
            'svd_energy_threshold': svd_energy_threshold,
            'keep_metadata': keep_metadata,
        }

        # Determine output path (same logic as _convert_with_streaming)
        if save_trigger:
            output_path = self._resolve_output_path(save_folder, filename)
        else:
            output_path = get_experiment_temp_path("checkpoint_studio")
            output_path = output_path.with_suffix('.safetensors')
            ThreadSafeCleanup.register_temp_file(output_path)

        # Create writer with tensors_dict instead of source_path
        print(f"   ↪ Direct dict-to-output streaming (no temp file)")
        writer = self._SafetensorsStreamWriter(
            parent=self,
            tensors_dict=tensors,           # KEY DIFFERENCE from _convert_with_streaming
            output_path=output_path,
            metadata=source_metadata,       # from _parse_checkpoint_data
            options=options,
        )
        # First pass: survey
        header, total_parameters, component_counts = writer.survey()
        # Metadata already finalized in survey, use writer.metadata
        final_metadata = writer.metadata
        # Second pass: weave (reads from dict, writes to file)
        writer.weave()

        # Wrap output file in lazy mapping
        converted_checkpoint = self._LazyCheckpointMapping(output_path, final_metadata)

        # Generate forensic report from survey metrics (same as _convert_with_streaming)
        forensic_report = self._generate_forensic_report_from_survey(
            total_parameters=writer.total_parameters,
            component_counts=writer.component_counts,
            parameter_counts=writer.parameter_counts,
            stripped_counts=writer.stripped_counts,
            architecture=writer.architecture,
            target_dtype=target_dtype,
            svd_info=writer.svd_info,
        )

        # UI summary
        quality_pct = self._estimate_quality_retention(target_dtype)
        ui_summary = self._format_ui_summary(len(writer.tensor_list), target_dtype, quality_pct, writer.stripped_counts, writer.svd_info)

        # Load MODEL/CLIP/VAE objects from lazy mapping (same as _convert_with_streaming)
        output_vae = not strip_vae
        output_clip = not (strip_te or strip_clip)
        try:
            model, clip, vae = load_state_dict_as_model_objects(
                converted_checkpoint,
                metadata=final_metadata,
                output_vae=output_vae,
                output_clip=output_clip,
            )
        except Exception as e:
            print(f"⚠️ Failed to load state dict as model objects: {e}")
            model, clip, vae = None, None, None

        save_path = str(output_path) if save_trigger else ""
        total_time = time.time() - start_time
        if debug:
            print(f"⏱️  Performance timings: dict-stream total {total_time:.2f}s")

        return (converted_checkpoint, model, clip, vae, save_path, forensic_report)

    RETURN_TYPES = ("CHECKPOINT", "MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("checkpoint", "model", "clip", "vae", "output_path", "forensic_report")
    FUNCTION = "convert"
    CATEGORY = "Checkpoint/Universal"

    def convert(self, checkpoint="None", save_trigger=False, filename="converted_checkpoint",
                precision="auto", device="auto", strip_vae=False, strip_te=False, strip_clip=False,
                checkpoint_data=None, save_folder="", keep_metadata=True,
                svd_mode="none", svd_energy_threshold=0.95,
                debug=False, node_id=None):
        """
        Main conversion routine.
        """
        print("\n" + "="*50)
        print(">>> Easy Checkpoint Studio")
        print("="*50)

        start_time = time.time()
        timings = {}
        stage_start = start_time

        # Determine source
        tensors = {}
        metadata = {}
        if checkpoint_data is not None:
            # Parse checkpoint_data (CHECKPOINT type)
            try:
                tensors, metadata = self._parse_checkpoint_data(checkpoint_data)
                print(f"📄 Source: CHECKPOINT data input ({len(tensors)} tensors)")
            except Exception as e:
                print(f"❌ Failed to parse checkpoint_data: {e}")
                return (None, None, None, None, "", "Invalid checkpoint_data")

            # ── Phase 2: Three-way dispatch ──────────────────────────────────
            # Priority:
            #   1. LazyCheckpointMapping short-circuit → file-based streaming
            #   2. RAM guard pass → dict-to-output streaming (no temp file)
            #   3. RAM guard fail → temp file fallback → file-based streaming

            # 1. Short-circuit: if input is already a _LazyCheckpointMapping,
            #    extract its filepath and use file-based streaming directly.
            #    This avoids materializing lazy-mapping tensors just for
            #    metadata/survey (see plan for rationale).
            if isinstance(tensors, self._LazyCheckpointMapping):
                print(f"   ↪ Input is already file-backed — streaming directly from file")
                return self._convert_with_streaming(
                    source_path=tensors.filepath,
                    save_trigger=save_trigger, filename=filename,
                    precision=precision, device=device,
                    strip_vae=strip_vae, strip_te=strip_te, strip_clip=strip_clip,
                    save_folder=save_folder, keep_metadata=keep_metadata,
                    svd_mode=svd_mode, svd_energy_threshold=svd_energy_threshold,
                    debug=debug, node_id=node_id,
                )

            # 2. RAM guard: calculate total data size and check if
            #    direct dict streaming is safe.
            # NOTE: For real dicts, .numel() and .element_size() are
            # metadata-only — no tensor data materialization.
            total_bytes = sum(
                t.numel() * t.element_size() for t in tensors.values()
            )
            if check_ram_guard(
                total_bytes, len(tensors), 1,
                label="checkpoint_studio", use_dict=True,
            ):
                # ✅ Direct dict-to-output streaming — no temp file
                return self._convert_from_dict(
                    tensors=tensors, source_metadata=metadata,
                    save_trigger=save_trigger, filename=filename,
                    precision=precision, device=device,
                    strip_vae=strip_vae, strip_te=strip_te, strip_clip=strip_clip,
                    save_folder=save_folder, keep_metadata=keep_metadata,
                    svd_mode=svd_mode, svd_energy_threshold=svd_energy_threshold,
                    debug=debug, node_id=node_id,
                )
            else:
                # ❌ RAM too low — save to temp file, free dict, stream from file
                print(f"   ℹ️  RAM Guard triggered: falling back to temp-file streaming")
                temp_path = get_experiment_temp_path("checkpoint_studio")
                temp_path = temp_path.with_suffix('.safetensors')
                save_safetensors_stream(tensors, temp_path, metadata=metadata)
                del tensors
                cleanup_memory()
                ThreadSafeCleanup.register_temp_file(temp_path)
                return self._convert_with_streaming(
                    source_path=temp_path,
                    save_trigger=save_trigger, filename=filename,
                    precision=precision, device=device,
                    strip_vae=strip_vae, strip_te=strip_te, strip_clip=strip_clip,
                    save_folder=save_folder, keep_metadata=keep_metadata,
                    svd_mode=svd_mode, svd_energy_threshold=svd_energy_threshold,
                    debug=debug, node_id=node_id,
                )
        else:
            if checkpoint == "None" or not checkpoint:
                print("❌ No checkpoint input provided")
                return (None, None, None, None, "", "No input")
            # Get full path
            path = folder_paths.get_full_path("checkpoints", checkpoint)
            if path is None:
                print(f"❌ Checkpoint file not found: {checkpoint}")
                return (None, None, None, None, "", "File not found")
            print(f"📄 Source: {checkpoint}")
            print(f"📁 Path: {path}")
            # Check if file is safetensors (magic bytes)
            try:
                with open(path, 'rb') as f:
                    magic = f.read(8)
            except Exception:
                magic = b''
            if magic[:7] == b'__safet':
                # Use streaming writer for file source (flat‑RAM guarantee)
                return self._convert_with_streaming(
                    source_path=path,
                    save_trigger=save_trigger,
                    filename=filename,
                    precision=precision,
                    device=device,
                    strip_vae=strip_vae,
                    strip_te=strip_te,
                    strip_clip=strip_clip,
                    save_folder=save_folder,
                    keep_metadata=keep_metadata,
                    svd_mode=svd_mode,
                    svd_energy_threshold=svd_energy_threshold,
                    debug=debug,
                    node_id=node_id,
                )
            else:
                # Fallback: load tensors into memory (original path)
                tensors, metadata = load_lora_with_metadata(Path(path))
        
        timings['loading'] = time.time() - stage_start
        stage_start = time.time()
        print(f"✅ Loaded {len(tensors)} tensors")
        if debug:
            print(f"   Sample keys: {list(tensors.keys())[:3]}")

        # ── RAM Guard: report available memory vs data size ────────────────
        _total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        _avail = get_available_ram()
        if _avail is not None:
            _data_gb = _total_bytes / (1024**3)
            _ram_gb = _avail / (1024**3)
            print(f"🧠 RAM Guard: Data ~{_data_gb:.2f} GB in {len(tensors)} tensors, "
                  f"Available RAM ~{_ram_gb:.2f} GB")
            if _data_gb > _ram_gb * 0.85:
                print(f"   ⚠️ Data exceeds 85% of available RAM — risk of OOM")
        
        # Determine target device and dtype
        target_device = DeviceManager.get_device(device)
        target_dtype = DeviceManager.get_dtype(precision, target_device)
        print(f"🔧 Target device: {target_device}, dtype: {target_dtype}")
        
        # Process tensors
        processed = {}
        stripped_counts = {"vae": 0, "te": 0, "clip": 0}
        total_tensors = len(tensors)
        if total_tensors > 0:
            node_prefix = f"[{node_id}] " if node_id else ""
            with ProgressTracker(total=total_tensors, desc=f"{node_prefix}Processing tensors") as convert_progress:
                for key, tensor in tensors.items():
                    # Component stripping
                    if self._should_strip_key(key, strip_vae, strip_te, strip_clip, stripped_counts):
                        convert_progress += 1
                        continue
                    # No key mapping — always ComfyUI original format
                    new_key = key
                    
                    
                    # Cast dtype if needed — keep on CPU (safetensors stores CPU bytes)
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(dtype=target_dtype)
                    
                    # Ensure contiguous for safetensors
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    
                    processed[new_key] = tensor
                    convert_progress += 1
        else:
            print("⚠️ No tensors to process")
        
        timings['processing'] = time.time() - stage_start
        stage_start = time.time()

        # ── Capture forensic data BEFORE freeing original tensors ──────────
        _forensic_analysis = self._analyze_checkpoint_structure(tensors, target_dtype)
        # Free original tensors — no longer needed after conversion.
        # This recovers ~16 GB (fp16) before SVD / metadata / save / model-loading,
        # eliminating the ~24 GB peak (tensors + processed coexist).
        del tensors
        cleanup_memory()
        print(f"   ✅ Freed original tensors — {_forensic_analysis['total_size_gb']:.2f} GB recovered")

        # Report stripping
        if any(stripped_counts.values()):
            print(f"🔪 Stripped components: VAE {stripped_counts['vae']}, TE {stripped_counts['te']}, CLIP {stripped_counts['clip']}")
        
        # Apply SVD compression if requested
        svd_info = None
        if svd_mode != "none":
            print(f"🔧 Applying SVD compression ({svd_mode})...")
            processed, svd_info = self._apply_svd_to_checkpoint(
                processed, svd_mode, svd_energy_threshold,
                target_device=target_device,
            )
            print(f"🔧 SVD compressed {svd_info['compressed_layers']} layers, saved {svd_info['size_saved_mb']:.2f} MB")
        
        if svd_mode != "none":
            timings['svd'] = time.time() - stage_start
            stage_start = time.time()

        # Metadata handling via unified factory
        mode = "preserve_a" if keep_metadata else "none"
        extra = self._build_architecture_extra(_forensic_analysis['architecture'])
        final_metadata = finalize_metadata(
            metadata=metadata,
            mode=mode,
            component="checkpoint_studio",
            extra_fields=extra,
        )
        
        # Save if triggered
        save_path = ""
        if save_trigger:
            try:
                # Determine output path with auto-increment
                output_path = self._resolve_output_path(save_folder, filename)
                
                # Save tensors and metadata using streaming write (avoids BytesIO 2x spike)
                save_safetensors_stream(processed, output_path, metadata=final_metadata)
                save_path = str(output_path)
                print(f"💾 Saved to: {save_path}")
            except Exception as e:
                print(f"❌ Save failed: {e}")
                save_path = ""
        else:
            print("🧠 Checkpoint kept in RAM, no file written")

        # ── Post-conversion RAM summary ────────────────────────────────────
        _avail = get_available_ram()
        if _avail is not None:
            print(f"   ✅ RAM post-conversion: {_avail / (1024**3):.2f} GB available")
        
        if save_trigger:
            timings['saving'] = time.time() - stage_start
            stage_start = time.time()

        # Generate UI summary
        quality_pct = self._estimate_quality_retention(target_dtype)
        ui_summary = self._format_ui_summary(len(processed), target_dtype, quality_pct, stripped_counts, svd_info)
        
        # Generate forensic report from pre‑computed analysis (tensors already freed)
        forensic_report = self._generate_forensic_report_from_analysis(
            _forensic_analysis, processed, target_dtype, stripped_counts, svd_info
        )
        
        # Return converted checkpoint (processed state dict)
        converted_checkpoint = processed
        
        timings['model_loading'] = time.time() - stage_start
        stage_start = time.time()

        # Convert to MODEL/CLIP/VAE objects
        output_vae = not strip_vae
        output_clip = not (strip_te or strip_clip)
        try:
            model, clip, vae = load_state_dict_as_model_objects(
                processed.copy(),  # preserve original for checkpoint_data output (ComfyUI pops keys)
                metadata=final_metadata,
                output_vae=output_vae,
                output_clip=output_clip
            )
        except Exception as e:
            print(f"⚠️ Failed to load state dict as model objects: {e}")
            model, clip, vae = None, None, None
        
        total_time = time.time() - start_time
        if debug:
            print(f"⏱️  Performance timings:")
            for stage, duration in timings.items():
                print(f"   {stage}: {duration:.2f}s")
            print(f"   total: {total_time:.2f}s")

        return (converted_checkpoint, model, clip, vae, save_path, forensic_report)


# Module-level alias for easy import from other engine modules
_LazyCheckpointMapping = MusubiCheckpointStudio._LazyCheckpointMapping

# For testing
if __name__ == "__main__":
    # Quick sanity check
    print("Easy Checkpoint Studio skeleton loaded.")
    print("Input types:", MusubiCheckpointStudio.INPUT_TYPES())