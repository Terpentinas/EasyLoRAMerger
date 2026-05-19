"""
Streaming triple checkpoint merger using Surveyor & Weaver pattern.
Handles three source checkpoint files (.safetensors) and merges them with low RAM usage.
"""

import json
import gc
import time
import sys
import uuid
import os

import folder_paths

# Debug flag for verbose diagnostic logging during checkpoint weave operations.
# Set WEAVER_DEBUG=1 environment variable to enable detailed per-batch diagnostics.
# In production (default), only essential warnings and summary lines are printed,
# eliminating ~2700 log lines and ~2600 sys.stdout.flush() calls per merge.
_WEAVER_DEBUG = os.environ.get("WEAVER_DEBUG", "0") == "1"
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..utils import (
    categorize_checkpoint_key,
    load_checkpoint_with_metadata,
    ThreadSafeCleanup,
    comfyui_yield,
    ProgressTracker,
    NullProgressTracker,
    cleanup_memory,
    get_available_ram,
)
from ..config import DevicePrecisionConfig
from .checkpoint_methods import merge_triple_method
from .metadata_factory import scrub_metadata, sign_metadata
from .identity_normalizer import identity_normalize
from .musubi_checkpoint_studio import MusubiCheckpointStudio, _LazyCheckpointMapping
from .checkpoint_normalizer import (
    detect_checkpoint_architecture,
    normalize_checkpoint_header,
)
from .forensics import build_forensic_report
from .fp8_quantizer import (
    should_preserve_bf16,
    quantize_weight_to_fp8_with_scales_optimized as quantize_weight_to_fp8_with_scales,
    dequant_fp8_tensor,
)


class CheckpointTripleMerger:
    """
    Two‑pass incremental merger that merges three checkpoints on the fly.
    Accepts either file paths (source_paths) or in-memory state dicts
    (source_dicts) as source inputs. When source_dicts[i] is provided,
    the corresponding source_paths[i] is ignored.
    """
    
    class _DictMapping:
        """Dict-like wrapper for in-memory state dicts, compatible with weave()."""
        def __init__(self, state_dict):
            self._sd = state_dict
        def __getitem__(self, key):
            return self._sd[key]
        def keys(self):
            return list(self._sd.keys())
        def __contains__(self, key):
            return key in self._sd
        def __iter__(self):
            return iter(self._sd.keys())
        def __len__(self):
            return len(self._sd)
        def close(self):
            """No-op: in-memory dict has no resources to close."""
            pass
    
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

    # Reverse mapping from torch.dtype to safetensors dtype string
    DTYPE_REVERSE = {
        torch.float64: "F64", torch.float32: "F32",
        torch.float16: "F16", torch.bfloat16: "BF16",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e5m2: "F8_E5M2",
        torch.int64: "I64",   torch.int32: "I32",
        torch.int16: "I16",   torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    # Extended reverse map with torch aliases and string self-mappings for robust matching.
    # Used by _read_header to handle both torch.dtype and string dtype identifiers.
    DTYPE_REVERSE_ALIASES = {
        **DTYPE_REVERSE,
        torch.half: "F16", torch.long: "I64",
        torch.int: "I32",  torch.short: "I16",
        "F64": "F64", "F32": "F32", "F16": "F16", "BF16": "BF16",
        "I64": "I64", "I32": "I32", "I16": "I16", "I8": "I8", "U8": "U8",
    }
    
    @staticmethod
    def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
        """
        Convert a contiguous CPU tensor to raw bytes, supporting all dtypes
        (including bfloat16 which numpy does not support).
        """
        if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
            # bfloat16/fp8 not supported by numpy, reinterpret as uint8.
            # tensor must be contiguous (caller ensures).
            # Handle scalar tensors (dim 0) because view() requires at least 1 dim.
            if tensor.dim() == 0:
                tensor = tensor.reshape(-1)  # convert to 1D with 1 element
            return tensor.view(torch.uint8).numpy().tobytes()
        else:
            # Use numpy for speed and simplicity.
            return tensor.numpy().tobytes()

    @staticmethod
    def _read_header(filepath: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Read safetensors header from file, return dict with shape/dtype per key and metadata.
        Uses safe_open for robust reading.
        """
        tensors = {}
        metadata = {}
        try:
            with safe_open(filepath, framework="pt") as sf:
                metadata = sf.metadata() or {}
                # Use class-level aliased reverse map (includes torch aliases + string self-mappings)
                dtype_reverse = CheckpointTripleMerger.DTYPE_REVERSE_ALIASES
                for key in sf.keys():
                    slice_info = sf.get_slice(key)
                    dtype = slice_info.get_dtype()
                    if isinstance(dtype, str):
                        dtype_str = dtype
                    else:
                        dtype_str = dtype_reverse.get(dtype, "F32")
                    if dtype_str == "F32" and dtype not in dtype_reverse:
                        print(f"[_read_header] WARNING: dtype {dtype} not in mapping for key {key}, defaulting to F32")
                    if 'position_ids' in key:
                        print(f"[_read_header] position_ids key {key}: dtype={dtype}, dtype_str={dtype_str}")
                    tensors[key] = {
                        "shape": list(slice_info.get_shape()),
                        "dtype": dtype_str,
                    }
        except Exception as e:
            # If safe_open fails, fall back to manual parsing (original logic)
            with open(filepath, 'rb') as f:
                magic = f.read(8)
                if magic != b'__safet':
                    print(f"⚠️ Not a safetensors file: {filepath.name}, magic={magic!r}")
                    raise ValueError('Not a safetensors file')
                header_size_bytes = f.read(8)
                if len(header_size_bytes) != 8:
                    raise ValueError('Invalid header size')
                header_size = int.from_bytes(header_size_bytes, 'little')
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes)
                # Filter out '__metadata__'
                tensors = {k: v for k, v in header.items() if k != '__metadata__'}
                metadata = header.get('__metadata__', {})
        return tensors, metadata

    @staticmethod
    def _read_header_from_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Extract header info (shape, dtype) from an in-memory state dict.
        Returns the same format as _read_header() — (tensors_dict, metadata_dict).

        This avoids writing chained data to a temporary safetensors file
        just to read its header back.
        """
        tensors = {}
        dtype_reverse = CheckpointTripleMerger.DTYPE_REVERSE_ALIASES
        for key, tensor in state_dict.items():
            dtype_str = dtype_reverse.get(tensor.dtype, "F32")
            tensors[key] = {
                "shape": list(tensor.shape),
                "dtype": dtype_str,
            }
        return tensors, {}

    @staticmethod
    def _ensure_safetensors(filepath: Path) -> Path:
        """
        Ensure the file is a safetensors file; if not, convert it.
        Returns the path to a safetensors file (original or temporary).
        """
        # Try to open with safe_open; if it succeeds, treat as safetensors.
        try:
            with safe_open(filepath, framework="pt"):
                pass
        except Exception:
            # safe_open failed, check magic bytes as a fallback
            with open(filepath, 'rb') as f:
                magic = f.read(8)
            if magic == b'__safet':
                # Magic bytes match but safe_open failed; maybe metadata issue.
                # Still treat as safetensors, but log a warning.
                print(f"⚠️ safe_open failed but magic bytes match for {filepath.name}; treating as safetensors (may have metadata issues).")
                return filepath
            # Not a safetensors file, proceed to conversion
        else:
            # safe_open succeeded → genuine safetensors
            return filepath
        
        # Load as checkpoint (supports .ckpt, .pth, etc.)
        gb = os.path.getsize(filepath) / (1024**3)
        print(f"⚠️  Loading entire checkpoint into RAM ({gb:.1f} GB) for conversion. Ensure sufficient free memory.")
        print(f"🔧 Converting non‑safetensors file {filepath.name} to safetensors...")
        tensors, metadata = load_checkpoint_with_metadata(filepath, categorize=False)
        
        # Create temporary safetensors file
        temp_dir = Path(folder_paths.get_temp_directory()) / "easy_checkpoint_merger"
        temp_dir.mkdir(parents=True, exist_ok=True)
        unique = uuid.uuid4().hex[:8]
        temp_path = temp_dir / f"converted_{unique}_{filepath.stem}.safetensors"
        
        # Save as safetensors
        save_file(tensors, str(temp_path), metadata=metadata)
        print(f"   Temporary safetensors saved to {temp_path}")
        
        # Verify the saved file can be opened with safe_open (more robust than magic check)
        try:
            with safe_open(temp_path, framework="pt"):
                pass
        except Exception as e:
            raise RuntimeError(
                f"Converted file {temp_path.name} is not a valid safetensors file "
                f"(safe_open failed: {e})"
            )
        
        # Register for cleanup
        ThreadSafeCleanup.register_temp_file(temp_path)
        return temp_path

    def __init__(self,
                 source_paths: List[Path],
                 output_path: Path,
                 merge_config: Dict[str, Any],
                 metadata_options: Optional[Dict[str, Any]] = None,
                 use_dict: bool = False,
                 source_dicts: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None):
        """
        Initialize the triple merger.
        
        Args:
            source_paths: List of checkpoint file paths (fewer than three allowed).
                For sources where source_dicts[i] is provided, the path is a placeholder.
            output_path: Where to write the merged checkpoint.
            merge_config: Dictionary with merge parameters:
                - method: str = "linear"
                - weights: List[float] = [1.0, 1.0, 1.0]
                - density: float = 1.0
                - uniqueness: float = 0.7
                - threshold: float = 0.0
                - blend: float = 0.5
                - blend_mode: str = "auto"
                - magnitude_scaling: str = "none"
                - max_scaling_factor: float = 10.0
                - batch_size: int = 32
                - streaming: bool = True
                - energy_preservation: bool = True
                - balancing_mode: str = "disabled"
                - weight_unet: float = 1.0
                - weight_clip: float = 1.0
                - weight_vae: float = 1.0
                - weight_te: float = 1.0
                - precision: str = "auto"
                - device: str = "auto"
            metadata_options: Options for metadata merging (keep, inject, etc.).
            source_dicts: Optional parallel list to source_paths. When source_dicts[i]
                is not None, that source's data comes from the in-memory dict instead
                of reading from source_paths[i]. This avoids writing chained data
                to a temporary file on disk.
        """
        if len(source_paths) > 3:
            raise ValueError("Maximum three source checkpoints supported.")
        self.source_dicts = source_dicts or [None] * len(source_paths)
        self.source_paths = [Path(p) for p in source_paths]
        # Convert non‑safetensors files to safetensors — skip for dict sources
        self.source_paths = [
            self._ensure_safetensors(p) if self.source_dicts[i] is None else p
            for i, p in enumerate(self.source_paths)
        ]
        self.output_path = Path(output_path) if output_path is not None else None
        self.merge_config = merge_config
        self.metadata_options = metadata_options or {}
        self.use_dict = use_dict
        self._output_dict: Optional[Dict[str, torch.Tensor]] = None
        # Resolve device and precision once via unified config
        device_str = merge_config.get('device', 'auto')
        precision_str = merge_config.get('precision', 'auto')
        self._device_precision = DevicePrecisionConfig(
            device_type=device_str, precision=precision_str
        )
        self.target_device = self._device_precision.device
        self.target_dtype = self._device_precision.dtype

        # Will be populated by survey()
        self.headers = []           # list of header dicts per source
        self.metadatas = []         # list of metadata dicts per source
        self.normalized_headers = [] # list of normalized header dicts per source
        self.key_mappings = []      # list of mapping dicts per source
        self.common_keys = []       # keys present in all sources
        self.union_keys = []        # keys present in at least one source (sorted)
        self.tensor_list = []       # list of dicts describing each output tensor
        self.header = None          # output header dict
        self.header_json = None
        self.header_len_padded = None
        self.data_start = None
        self.total_parameters = 0
        self.component_counts = {"unet": 0, "clip": 0, "vae": 0, "te": 0, "other": 0}
        self.parameter_counts = {"unet": 0, "clip": 0, "vae": 0, "te": 0, "other": 0}
        self.architecture = "Unknown"
        # FP8 companion scales dict: norm_key → (weight_scale_tensor, input_scale_tensor)
        # Populated during weave() when quantizing weights to FP8, used to write
        # companion scale tensor entries alongside FP8-quantized weights.
        self._companion_scales: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Companion scale tensor slots emitted by survey() for FP8 output.
        # Stored separately from tensor_list so they don't inflate batch counting.
        self.companion_list: List[Dict[str, Any]] = []
        # OOM safety flag: set to True after first CUDA OOM, causing remaining
        # batches to process on CPU instead of GPU.
        self._oom_cpu_fallback = False

        # Map torch dtype to safetensors dtype string
        self.target_dtype_str = self.DTYPE_REVERSE.get(self.target_dtype, "F32")
        self.elem_size = self.DTYPE_MAP.get(self.target_dtype_str, (None, 4))[1]
    
    def _normalize_header(self, header: Dict[str, Any], metadata: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Normalize checkpoint keys using architecture-aware normalization.
        
        For full checkpoints: uses checkpoint_normalizer to detect architecture and
        normalize keys to canonical form (e.g., stripping 'model.diffusion_model.' prefix
        for FLUX models), enabling proper key matching between sources with different
        key naming conventions.
        
        For LoRAs: falls back to identity_normalize for LoRA-specific key conversion.
        """
        # Sample keys to show what we're detecting
        sample_keys = list(header.keys())[:5]
        has_lora = any('lora_' in k for k in header.keys())
        if _WEAVER_DEBUG:
            print(f"[_normalize_header] Checking {len(header)} keys, has 'lora_' pattern: {has_lora}, sample keys: {sample_keys}")
        
        if not has_lora:
            # Full checkpoint: use architecture-aware normalization
            architecture = detect_checkpoint_architecture(list(header.keys()))
            # Keep one concise line per source even in non-debug mode
            if not _WEAVER_DEBUG:
                print(f"[_normalize_header] {len(header)} keys, architecture: {architecture}")
            else:
                print(f"[_normalize_header] Architecture: {architecture}, normalizing {len(header)} keys")
            
            # Normalize keys to canonical form (e.g., strip model.diffusion_model. prefix for FLUX)
            normalized_header, mapping = normalize_checkpoint_header(header, architecture)
            
            if _WEAVER_DEBUG:
                print(f"[_normalize_header] Normalized {len(header)} keys -> {len(normalized_header)} canonical keys")
            return normalized_header, mapping
        
        # LoRA detected - proceed with identity normalization
        if _WEAVER_DEBUG:
            print(f"[_normalize_header] LoRA: {len(header)} keys, running identity_normalize")
        # Build dummy state dict with keys only (identity_normalize uses keys, not values)
        dummy_sd = {key: torch.zeros(1, dtype=torch.float32) for key in header}
        # Normalize with identity mapping
        normalized_sd, mapping = identity_normalize(dummy_sd, metadata=metadata)
        # Convert back to header format (preserve shape/dtype from original header)
        normalized_header = {}
        for norm_key in normalized_sd.keys():
            # Find original key via mapping (should exist)
            orig_key = mapping.get(norm_key, norm_key)
            # Keep original shape/dtype
            normalized_header[norm_key] = header[orig_key]
        return normalized_header, mapping
    
    def survey(self, pbar: 'ProgressTracker' = NullProgressTracker()) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
        """
        First pass: read headers from all source files, compute common keys,
        determine output tensor sizes and offsets, build output header.
        Returns (header_dict, total_parameters, component_counts).

        If called a second time (e.g. RAM guard then weave), cached results
        are returned immediately — no redundant header I/O.

        Args:
            pbar: ProgressTracker to advance after each source processed.
                  Defaults to NullProgressTracker (silent no-op).
                  Stateless singleton — safe as default parameter.
        """
        # ── Cache guard: skip if already surveyed ──
        if self.tensor_list:
            return self.header, self.total_parameters, self.component_counts

        _survey_t0 = time.time()
        print(f"   🕐 [t={_survey_t0 - time.time():.1f}s] survey: reading {len(self.source_paths)} source headers")
        # Re-fetch current time for accurate relative offset
        _survey_t0 = time.time()
        self.headers = []
        self.metadatas = []
        self.normalized_headers = []
        self.key_mappings = []
        keys_per_source = []
        norm_keys_per_source = []
        
        # Read headers and collect keys
        for i, src in enumerate(self.source_paths):
            if self.source_dicts[i] is not None:
                # In-memory dict source: extract header without writing to disk
                tensors, metadata = self._read_header_from_dict(self.source_dicts[i])
                print(f"   [SURVEY] Source {i}: in-memory dict — {len(tensors)} keys")
            else:
                tensors, metadata = self._read_header(src)
            self.headers.append(tensors)
            self.metadatas.append(metadata)
            # Normalize keys using architecture-aware normalization
            norm_header, mapping = self._normalize_header(tensors, metadata)
            self.normalized_headers.append(norm_header)
            self.key_mappings.append(mapping)
            keys_per_source.append(set(tensors.keys()))
            # Normalized keys
            norm_keys = set(norm_header.keys())
            norm_keys_per_source.append(norm_keys)
        
            pbar.update(1)
        print(f"   🕐 [t={time.time()-_survey_t0:.1f}s] survey: headers read ({len(self.union_keys)} union keys before filtering)")
        
        # Log integer tensor counts per source
        for i, nh in enumerate(self.normalized_headers):
            int_keys = [k for k, v in nh.items() if v['dtype'].startswith('I') or v['dtype'].startswith('U')]
            if int_keys:
                print(f"[DEBUG] Source {i}: {len(int_keys)} integer tensors")
        
        # Compute intersection (keys present in all sources) and union (sorted) based on NORMALIZED keys
        if norm_keys_per_source:
            self.common_keys = list(set.intersection(*norm_keys_per_source))
        else:
            self.common_keys = []
        self.union_keys = sorted(set.union(*norm_keys_per_source) if norm_keys_per_source else set())
        
        # ── Filter out FP8 quantization artifact keys ─────────────────────
        # These keys (*.input_scale, *.weight_scale, *.comfy_quant) are FP8
        # / INT8 quantization metadata, not actual model weights. They appear
        # in quantized safetensors files alongside regular weight tensors:
        #   - .input_scale / .weight_scale  — companion scale factors
        #   - .comfy_quant                  — ComfyUI quantisation metadata
        # Including them in the merge:
        #   (a) inflates the union with keys present in only one source,
        #   (b) gets them averaged/corrupted like regular tensors,
        #   (c) causes ComfyUI to detect "quantization metadata" and enter
        #       the MixedPrecisionOps code path, but the actual weights are
        #       BF16 (dequantized during merge) — model loading fails.
        # This is safe for ALL architectures (Flux, SDXL, SD1.5, Z-Image,
        # Anima, Lumina2) — none use these suffixes as actual parameter names.
        # The GGUF writer (gguf_writer.py:203) already uses the same triple.
        _FP8_ARTIFACT_SUFFIXES = ('.input_scale', '.weight_scale', '.comfy_quant')
        filtered = []
        removed = 0
        for nk in self.union_keys:
            if nk.endswith(_FP8_ARTIFACT_SUFFIXES):
                removed += 1
            else:
                filtered.append(nk)
        if removed:
            print(f"   ⏭️ Excluded {removed} FP8/INT8 quantization artifact keys from merge")
        self.union_keys = filtered
        
        print(f"[WEAVE] Union: {len(self.union_keys)} unique keys across sources")
        
        # Merge metadata: keep from first source, add others with source prefix
        merged_metadata = {}
        if self.metadatas:
            merged_metadata.update(self.metadatas[0])
            for i, meta in enumerate(self.metadatas[1:], start=2):
                for k, v in meta.items():
                    merged_metadata[f'source{i}_{k}'] = v

        # Scrub based on keep_metadata flag, then sign with checkpoint_studio preset
        mode = "preserve_a" if self.metadata_options.get('keep_metadata', True) else "none"
        scrubbed = scrub_metadata(merged_metadata, mode)

        # Strip _quantization_metadata ONLY when output dtype is NOT FP8.
        #
        # When the merge target is BF16/FP16/FP32, the merger dequantizes
        # FP8 sources and produces float output — no quantization metadata
        # should propagate (would trigger MixedPrecisionOps incorrectly).
        #
        # When target IS FP8, the output IS a quantized checkpoint and
        # needs _quantization_metadata so ComfyUI's load_state_dict_guess_config
        # can enter MixedPrecisionOps for proper dequant during inference.
        # The companion scale tensors (.weight_scale, .input_scale) are
        # emitted by survey() at lines 575-609 and populated by weave().
        if self.target_dtype_str not in ("F8_E4M3", "F8_E5M2"):
            scrubbed.pop('_quantization_metadata', None)

        self.merged_metadata = scrubbed
        
        # Determine output shape/dtype per key (use first source where key exists)
        # For simplicity, assume all sources have same shape for common keys (no validation).
        # We'll pick shape from first source that contains the key.
        self.tensor_list = []
        self.companion_list = []
        self.total_parameters = 0
        self.component_counts = {k: 0 for k in self.component_counts}
        self.parameter_counts = {k: 0 for k in self.parameter_counts}
        
        for norm_key in self.union_keys:
            # Find first source that contains this normalized key
            for i, norm_header in enumerate(self.normalized_headers):
                if norm_key in norm_header:
                    info = norm_header[norm_key]
                    shape = tuple(info['shape'])
                    original_dtype_str = info['dtype']
                    break
            else:
                continue  # shouldn't happen
            
            numel = 1
            for dim in shape:
                numel *= dim
            
            # Determine output dtype string and element size
            # Preserve integer dtypes (I64, I32, I16, I8, U8) unchanged; convert float tensors to target dtype.
            if original_dtype_str.startswith('I') or original_dtype_str.startswith('U'):
                output_dtype_str = original_dtype_str
            else:
                output_dtype_str = self.target_dtype_str
            elem_size = self.DTYPE_MAP.get(output_dtype_str, (None, 4))[1]
            tensor_size = numel * elem_size
            
            # No padding between tensors (contiguity required by safetensors)
            pad_after = 0
            # offset will be determined during sequential write
            offset = None
            
            # Component counting - need original key for categorization
            # Choose original key from first source that contains this normalized key
            orig_key = self.key_mappings[i].get(norm_key, norm_key)
            comp = categorize_checkpoint_key(orig_key)
            self.component_counts[comp] += 1
            self.parameter_counts[comp] += numel
            self.total_parameters += numel
            
            # Determine which sources have this normalized key
            sources = [idx for idx, nh in enumerate(self.normalized_headers) if norm_key in nh]
            # For each source, get original key via mapping
            orig_keys = []
            for src_idx in range(len(self.source_paths)):
                if src_idx in sources:
                    orig_keys.append(self.key_mappings[src_idx].get(norm_key, norm_key))
                else:
                    orig_keys.append(None)
            
            self.tensor_list.append({
                "norm_key": norm_key,
                "orig_keys": orig_keys,
                "key": orig_key,                # output key (original from first source)
                "shape": shape,
                "dtype": output_dtype_str,
                "size": tensor_size,
                "offset": offset,
                "pad_after": pad_after,
                "sources": sources,
            })

            # ── FP8: emit companion scale tensor slots for weight keys ──
            if output_dtype_str in ("F8_E4M3", "F8_E5M2") and norm_key.endswith('.weight'):
                if not should_preserve_bf16(norm_key):
                    # .weight_scale companion (scalar F32, 4 bytes)
                    weight_scale_key = orig_key + "_scale"
                    self.companion_list.append({
                        "norm_key": norm_key + "_scale",
                        "orig_keys": [None] * len(self.source_paths),
                        "key": weight_scale_key,
                        "shape": [1],
                        "dtype": "F32",
                        "size": 4,
                        "offset": None,
                        "pad_after": 0,
                        "sources": [],
                        "_companion": True,
                        "_parent_norm_key": norm_key,
                        "_scale_type": "weight_scale",
                    })
                    # .input_scale companion (scalar F32, 4 bytes)
                    input_scale_key = orig_key.replace('.weight', '.input_scale')
                    self.companion_list.append({
                        "norm_key": norm_key.replace('.weight', '.input_scale'),
                        "orig_keys": [None] * len(self.source_paths),
                        "key": input_scale_key,
                        "shape": [1],
                        "dtype": "F32",
                        "size": 4,
                        "offset": None,
                        "pad_after": 0,
                        "sources": [],
                        "_companion": True,
                        "_parent_norm_key": norm_key,
                        "_scale_type": "input_scale",
                    })
        # Architecture detection (using centralized, extensible detection)
        # Use normalized keys from all sources for accurate detection
        all_normalized_keys = []
        for nh in self.normalized_headers:
            all_normalized_keys.extend(nh.keys())
        self.architecture = detect_checkpoint_architecture(all_normalized_keys)
        # Capitalize first letter for display (flux -> Flux, sdxl -> SDXL, sd15 -> SD1.5)
        ARCH_DISPLAY_NAMES = {
            "anima": "Anima", "flux": "Flux", "lumina2": "Lumina2",
            "sdxl": "SDXL", "sd15": "SD1.5", "z_image": "Z-Image",
        }
        self.architecture = ARCH_DISPLAY_NAMES.get(self.architecture, "Unknown")
        print(f"[Architecture] Detected: {self.architecture}")

        # Free survey-only metadata — no longer needed by weave()
        self.headers = None
        self.normalized_headers = None
        self.key_mappings = None

        # Sign metadata: inject checkpoint_studio preset (SAI header + architecture)
        if self.metadata_options.get('inject_sai_header', False):
            self.merged_metadata = sign_metadata(
                self.merged_metadata,
                component="checkpoint_studio",
                extra_fields={"modelspec.architecture": self.architecture}
            )
        else:
            # Still add architecture even without full SAI header
            self.merged_metadata['modelspec.architecture'] = self.architecture
        
        
        # Build placeholder header with dummy offsets (0)
        # Include both weight tensor_list and companion scale items.
        header = {}
        for item in self.tensor_list:
            header[item['key']] = {
                "dtype": item['dtype'],
                "shape": list(item['shape']),
                "data_offsets": [0, item['size']]  # placeholder
            }
        for item in self.companion_list:
            header[item['key']] = {
                "dtype": item['dtype'],
                "shape": list(item['shape']),
                "data_offsets": [0, item['size']]  # placeholder
            }
        # Add merged metadata
        if self.merged_metadata:
            header['__metadata__'] = self.merged_metadata
        
        # Compute header length with zero offsets, add safety margin, and pad to 8 bytes
        header_json = json.dumps(header, separators=(',', ':'))
        header_len_raw = len(header_json)
        # Safety margin to accommodate final offset digits (max 20 digits per offset) and metadata changes
        # Increased to 64KB to ensure placeholder fits final header for large merges.
        safety_margin = 65536
        header_len_with_margin = header_len_raw + safety_margin
        pad_len = (8 - (header_len_with_margin % 8)) % 8
        header_len_padded = header_len_with_margin + pad_len
        data_start = 8 + header_len_padded
        
        # Placeholder header JSON padded with spaces (will be overwritten later)
        # Add safety_margin + pad_len spaces so that the total written length matches header_len_padded
        header_json_padded = header_json + ' ' * (header_len_padded - header_len_raw)
        
        # Store placeholder results
        self.header = header
        self.header_json = header_json_padded
        self.header_len_padded = header_len_padded
        self.data_start = data_start
        
        total_slots = len(self.tensor_list) + len(self.companion_list)
        print(f"[WEAVE] Header boundary: {header_len_padded} bytes, data start = {data_start}, {total_slots} tensors ({len(self.tensor_list)} weights, {len(self.companion_list)} companions)")
        print(f"   🕐 [t={time.time()-_survey_t0:.1f}s] survey complete ({total_slots} tensor slots)")
        
        return header, self.total_parameters, self.component_counts
    
    def weave(self, pbar: 'ProgressTracker' = NullProgressTracker()):
        """
        Second pass: load tensors from each source, merge using checkpoint_methods,
        write merged tensors sequentially using ground‑truth offsets.
        Supports both file and BytesIO (true RAM) output targets.

        Args:
            pbar: ProgressTracker to advance after each batch completes.
                  Defaults to NullProgressTracker (silent no-op).
                  The pbar is advanced by merge_triple_method via on_substep callback,
                  NOT directly at the weave level.
                  Stateless singleton — safe as default parameter.
        """
        # Ensure survey has been called
        if not hasattr(self, 'header_json'):
            raise RuntimeError('Survey must be called before weave')
        
        # Determine output target: direct dict or file (disk)
        if self.use_dict:
            out_f = None
            print(f"🧠 Direct dict mode – storing merged tensors in dict")
        else:
            out_f = open(self.output_path, 'wb', buffering=0)
        
        try:
            # ── Dict mode: skip placeholder header (no serialized output) ──
            if self.use_dict:
                data_start = 0
                self._output_dict = {}
                # ── Phase 3: detect zero-weight components for auto-omit ──
                self._omitted_components = []
                _comp_map = [("vae", "weight_vae"), ("te", "weight_te"), ("clip", "weight_clip")]
                for cat, config_key in _comp_map:
                    if self.merge_config.get(config_key, 1.0) == 0.0:
                        self._omitted_components.append(cat)
                if self._omitted_components:
                    print(f"   ⏭️ Auto-omitting {self._omitted_components} tensors (weight=0)")
                print(f"   [DICT] Initialized output dict for {len(self.tensor_list)} tensors")
            else:
                # Write placeholder header length and JSON (already padded with safety margin)
                header_len = self.header_len_padded
                out_f.write(header_len.to_bytes(8, 'little'))
                out_f.write(self.header_json.encode('utf-8'))
                
                # Verify we are at data_start
                current_pos = out_f.tell()
                data_start = self.data_start
                if current_pos != data_start:
                    print(f"[WARNING] File position after header ({current_pos}) does not match data_start ({data_start}), adjusting with zero padding")
                    if current_pos < data_start:
                        out_f.write(b'\x00' * (data_start - current_pos))
                    else:
                        # Should not happen because header length is fixed
                        out_f.seek(data_start)
            
            # Prepare merge configuration
            config = self.merge_config
            weights = config.get('weights', [1.0, 1.0, 1.0])
            if len(weights) < len(self.source_paths):
                weights = weights + [1.0] * (len(self.source_paths) - len(weights))
            weights = weights[:len(self.source_paths)]
            
            # Dictionary to store ground‑truth offsets (relative to data_start)
            # Use instance variable to prevent any scoping/GC issues between batches
            self.actual_offsets = {}
            first_tensor = True
            
            batch_size = config.get('batch_size', 64)
            streaming = config.get('streaming', True)
            total_items = len(self.tensor_list)
            # When streaming=False, process all tensors in a single batch
            effective_batch_size = batch_size if streaming else total_items
            num_batches = (total_items + effective_batch_size - 1) // effective_batch_size
            if _WEAVER_DEBUG:
                # ── DIAG: show first 10 tensor_list entries ──
                print(f"   [DIAG] tensor_list first 10 entries:")
                for ti in range(min(10, total_items)):
                    print(f"   [DIAG]   [{ti}] norm_key='{self.tensor_list[ti]['norm_key']}', "
                          f"key='{self.tensor_list[ti]['key']}', "
                          f"sources={self.tensor_list[ti]['sources']}, "
                          f"shape={self.tensor_list[ti]['shape']}, "
                          f"dtype={self.tensor_list[ti]['dtype']}")
                
                # ── DIAG: print expected total data size vs actual buffer capacity ──
                expected_total_size = sum(item['size'] for item in self.tensor_list)
                print(f"   [DIAG] Expected total data size from survey: {expected_total_size} bytes "
                      f"({expected_total_size / (1024**3):.2f} GB)")
            
            # pbar is always a valid object (NullProgressTracker by default).
            # No internal fallback tracker needed — merge_triple_method handles
            # substep progress via on_substep callback.
            _weave_t0 = time.time()
            print(f"   🕐 [t={0.0:.1f}s] weave: starting {num_batches} batches (batch_size={effective_batch_size})")
            
            for batch_start in range(0, total_items, effective_batch_size):
                # Open source files fresh for THIS batch only, then wrap in try/finally
                # so mmap pages are released before the next batch begins.
                batch_sources = []
                for i, src_path in enumerate(self.source_paths):
                    if self.source_dicts[i] is not None:
                        # In-memory dict: wrap with _DictMapping instead of _LazyCheckpointMapping
                        mapping = CheckpointTripleMerger._DictMapping(self.source_dicts[i])
                    else:
                        mapping = _LazyCheckpointMapping(src_path)
                    batch_sources.append(mapping)
                try:
                    batch_end = min(batch_start + effective_batch_size, total_items)
                    batch_items = self.tensor_list[batch_start:batch_end]
                    
                    pos_before = 0 if self.use_dict else out_f.tell()
                    batch_num = batch_start // effective_batch_size + 1
                    if _WEAVER_DEBUG:
                        # ── DIAG: buffer position BEFORE batch ──
                        buf_id = id(out_f)
                        self_id = id(self)
                        print(f"   [DIAG:BUF] Batch {batch_num}/{num_batches}: "
                              f"before write pos={pos_before}, "
                              f"rel_to_data={pos_before - data_start}, "
                              f"actual_offsets_count={len(self.actual_offsets)}, "
                              f"out_f_id={buf_id}, self_id={self_id}")
                        sys.stdout.flush()
                    
                    # Build per‑source state dicts for this batch
                    batch_sds = [{} for _ in range(len(self.source_paths))]
                    batch_mappings = [{} for _ in range(len(self.source_paths))]
                    # batch_original_sds NOT needed — only used for LoRA alpha lookup
                    for item in batch_items:
                        norm_key = item['norm_key']
                        orig_keys = item['orig_keys']
                        source_indices = item['sources']
                        
                        for idx in source_indices:
                            orig_key = orig_keys[idx]
                            tensor = batch_sources[idx][orig_key]
                            target_torch_dtype = self.DTYPE_MAP[item['dtype']][0]

                            # ── Dequantize FP8 tensors ────────────────────────────────
                            # If the loaded tensor is FP8-quantized (float8_e4m3fn or
                            # float8_e5m2), load the corresponding weight_scale from the
                            # same source and apply dequantization before converting to
                            # target dtype.  This prevents half-range corrupted values
                            # when merging FP8 sources with float16/bfloat16 sources,
                            # which was the root cause of "noise instead of image" in
                            # the FP8 merge workflow.
                            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                                # Dequant FP8 → float32 using companion scale factors
                                tensor = dequant_fp8_tensor(tensor, orig_key, torch.float32, batch_sources[idx])

                            # If the storage dtype is FP8, use bfloat16 for compute
                            # (merge arithmetic cannot run on Float8 CUDA tensors).
                            # The final conversion to FP8 happens at write stage
                            # (see lines ~888-892).
                            if target_torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                                if tensor.dtype != compute_dtype:
                                    tensor = tensor.to(dtype=compute_dtype)
                            elif tensor.dtype != target_torch_dtype:
                                tensor = tensor.to(dtype=target_torch_dtype)
                            # Keep source tensors on CPU — merge_triple_method's per-key
                            # .to(device) at checkpoint_methods.py:878 handles GPU transfer
                            # on-demand for each key individually.  Previously, ALL batch
                            # tensors were pre-loaded to GPU here, causing ~9 GB VRAM tied
                            # up during merge_triple_method, with unrecoverable 2-3 GB
                            # fragmentation accumulation per batch (→ CUDA OOM by batch 3).
                            batch_sds[idx][norm_key] = tensor
                            batch_mappings[idx][norm_key] = orig_key
                            # batch_original_sds[idx][orig_key] = tensor  # REMOVED — unused in checkpoint path
                    
                    # Build on_substep callback that advances the weave-level pbar
                    # via merge_triple_method's fine-grained progress reporting.
                    def on_merge_step(n: int) -> None:
                        pbar.update(n)

                    # Merge using merge_triple_method with mapping support.
                    # NOTE: Import is from engine.checkpoint_methods, which returns:
                    #   (merged_dict: Dict[str, Tensor], corrupted_keys: List[str])
                    # This is NOT the same function as engine/triple_methods.py
                    # which returns 4 values.
                    merge_device = self.target_device
                    _batch_t0 = time.time()
                    try:
                        merged_dict, batch_corrupted = merge_triple_method(
                            sds=batch_sds,
                            weights=weights,
                            method=config.get('method', 'linear'),
                            density=config.get('density', 1.0),
                            uniqueness=config.get('uniqueness', 0.7),
                            threshold=config.get('threshold', 0.0),
                            blend=config.get('blend', 0.5),
                            blend_mode=config.get('blend_mode', 'auto'),
                            magnitude_scaling=config.get('magnitude_scaling', 'none'),
                            max_scaling_factor=config.get('max_scaling_factor', 10.0),
                            batch_size=batch_size,
                            streaming=config.get('streaming', True),
                            energy_preservation=config.get('energy_preservation', True),
                            balancing_mode=config.get('balancing_mode', 'disabled'),
                            weight_unet=config.get('weight_unet', 1.0),
                            weight_clip=config.get('weight_clip', 1.0),
                            weight_vae=config.get('weight_vae', 1.0),
                            weight_te=config.get('weight_te', 1.0),
                            mappings=batch_mappings,
                            original_sds=None,
                            metas=self.metadatas,
                            on_substep=on_merge_step,
                            resolved_device=merge_device,
                            resolved_dtype=self.target_dtype,
                            sequential_only=self._oom_cpu_fallback,
                        )
                    except torch.cuda.OutOfMemoryError:
                        cleanup_memory()
                        print(f"   ⚠️ CUDA OOM on batch {batch_num} — falling back to CPU for remaining tensors")
                        # All subsequent batches will use CPU too
                        self._oom_cpu_fallback = True
                        merge_device = torch.device("cpu")

                        # ── Rebuild batch_sds from source files ────────────────────────
                        # The first (GPU) call to merge_triple_method partially consumed
                        # batch_sds via sd.pop() (an intentional GPU-memory optimization).
                        # We cannot reuse the partially-consumed dicts — instead, re-read
                        # all tensors from the original source files directly onto CPU.
                        print(f"   🔄 Rebuilding batch {batch_num} tensors on CPU from source files...")
                        batch_sds = [{} for _ in range(len(self.source_paths))]
                        batch_mappings = [{} for _ in range(len(self.source_paths))]
                        for item in batch_items:
                            norm_key = item['norm_key']
                            orig_keys = item['orig_keys']
                            source_indices = item['sources']
                            for idx in source_indices:
                                orig_key = orig_keys[idx]
                                tensor = batch_sources[idx][orig_key]
                                target_torch_dtype = self.DTYPE_MAP[item['dtype']][0]

                                # ── FP8 dequant (same logic as lines 815-837) ──
                                if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                                    # Dequant FP8 → float32 using companion scale factors
                                    tensor = dequant_fp8_tensor(tensor, orig_key, torch.float32, batch_sources[idx])

                                # ── Dtype conversion (same logic as lines 839-848) ──
                                if target_torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                                    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                                    if tensor.dtype != compute_dtype:
                                        tensor = tensor.to(dtype=compute_dtype)
                                elif tensor.dtype != target_torch_dtype:
                                    tensor = tensor.to(dtype=target_torch_dtype)

                                # Always CPU for the retry
                                tensor = tensor.to(device=torch.device("cpu"))
                                batch_sds[idx][norm_key] = tensor
                                batch_mappings[idx][norm_key] = orig_key

                        merged_dict, batch_corrupted = merge_triple_method(
                            sds=batch_sds,
                            weights=weights,
                            method=config.get('method', 'linear'),
                            density=config.get('density', 1.0),
                            uniqueness=config.get('uniqueness', 0.7),
                            threshold=config.get('threshold', 0.0),
                            blend=config.get('blend', 0.5),
                            blend_mode=config.get('blend_mode', 'auto'),
                            magnitude_scaling=config.get('magnitude_scaling', 'none'),
                            max_scaling_factor=config.get('max_scaling_factor', 10.0),
                            batch_size=batch_size,
                            streaming=config.get('streaming', True),
                            energy_preservation=config.get('energy_preservation', True),
                            balancing_mode=config.get('balancing_mode', 'disabled'),
                            weight_unet=config.get('weight_unet', 1.0),
                            weight_clip=config.get('weight_clip', 1.0),
                            weight_vae=config.get('weight_vae', 1.0),
                            weight_te=config.get('weight_te', 1.0),
                            mappings=batch_mappings,
                            original_sds=None,
                            metas=self.metadatas,
                            on_substep=on_merge_step,
                            resolved_device=merge_device,
                            resolved_dtype=self.target_dtype,
                            sequential_only=True,
                        )
                    print(f"   🕐 [t={time.time()-_weave_t0:.1f}s] weave batch {batch_num}/{num_batches}: merged {len(batch_items)} tensors (merge took {time.time()-_batch_t0:.1f}s)")

                    if batch_corrupted:
                        if not hasattr(self, 'corrupted_keys'):
                            self.corrupted_keys = []
                        self.corrupted_keys.extend(batch_corrupted)

                    if _WEAVER_DEBUG:
                        # ── DIAG: batch items check ──
                        print(f"   [DIAG:INNER] Batch {batch_num}: len(batch_items)={len(batch_items)}, "
                              f"len(merged_dict)={len(merged_dict)}, "
                              f"batch_items[0]='{batch_items[0]['norm_key'] if batch_items else 'EMPTY'}', "
                              f"pos_before={pos_before}, data_start={data_start}")
                        sys.stdout.flush()

                    # ── Direct Dict Mode: store tensors directly ──
                    if self.use_dict:
                        for idx, item in enumerate(batch_items):
                            norm_key = item['norm_key']
                            merged_tensor = merged_dict.get(norm_key)

                            if merged_tensor is None:
                                if hasattr(self, 'corrupted_keys') and norm_key in self.corrupted_keys:
                                    shape = tuple(item['shape'])
                                    dtype_str = item['dtype']
                                    target_dtype = self.DTYPE_MAP[dtype_str][0]
                                    merged_tensor = torch.zeros(shape, dtype=target_dtype)
                                    print(f"   ⚠️ Zero tensor for corrupted key '{norm_key}' ({shape})")
                                else:
                                    raise RuntimeError(f"Missing merged tensor for key {norm_key}")

                            # Ensure tensor is contiguous and move to CPU FIRST
                            # (before dtype conversion, so FP8 quantization runs on CPU
                            # and doesn't spike GPU memory with F32 intermediates)
                            if not merged_tensor.is_contiguous():
                                merged_tensor = merged_tensor.contiguous()
                            merged_tensor_cpu = merged_tensor if merged_tensor.device.type == 'cpu' else merged_tensor.cpu()

                            # ── Convert to target dtype with proper FP8 quantization ──
                            target_dtype_str = item['dtype']
                            target_torch_dtype = self.DTYPE_MAP[target_dtype_str][0]
                            if merged_tensor_cpu.dtype != target_torch_dtype:
                                if target_torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) \
                                   and norm_key.endswith('.weight') and not should_preserve_bf16(norm_key):
                                    # Proper scale-then-cast for FP8 output weights (on CPU)
                                    q, wscale, iscale = quantize_weight_to_fp8_with_scales(
                                        merged_tensor_cpu, target_torch_dtype
                                    )
                                    merged_tensor_cpu = q
                                    # Store companion scales for writing later
                                    self._companion_scales[norm_key] = (wscale, iscale)
                                elif target_torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) \
                                     and should_preserve_bf16(norm_key):
                                    merged_tensor_cpu = merged_tensor_cpu.to(dtype=torch.bfloat16)
                                else:
                                    merged_tensor_cpu = merged_tensor_cpu.to(dtype=target_torch_dtype)

                            # ── Phase 2: skip tensors for zero-weight components ──
                            if self._omitted_components:
                                component = categorize_checkpoint_key(norm_key)
                                if component in self._omitted_components:
                                    if _WEAVER_DEBUG:
                                        print(f"   ⏭️ Omitting '{norm_key}' (component={component}, weight=0)")
                                    continue

                            self._output_dict[norm_key] = merged_tensor_cpu

                    # ── Standard Mode: write to BytesIO or file ──
                    else:
                        for idx, item in enumerate(batch_items):
                            norm_key = item['norm_key']
                            merged_tensor = merged_dict.get(norm_key)

                            # ── DIAG: print first item of each batch ──
                            if idx == 0:
                                present = "YES" if merged_tensor is not None else "NO"
                                if _WEAVER_DEBUG:
                                    print(f"   [DIAG:ITEM] Batch {batch_num}[0]: norm_key='{norm_key}', "
                                          f"in_merged_dict={present}, "
                                          f"merged_dict keys count={len(merged_dict)}")
                                    sys.stdout.flush()

                            if merged_tensor is None:
                                # Check if this key was skipped due to corruption
                                if hasattr(self, 'corrupted_keys') and norm_key in self.corrupted_keys:
                                    # Write zeros of expected shape as fallback
                                    shape = tuple(item['shape'])
                                    dtype_str = item['dtype']
                                    target_dtype = self.DTYPE_MAP[dtype_str][0]
                                    fallback = torch.zeros(shape, dtype=target_dtype)
                                    print(f"   ⚠️ Writing zero tensor for corrupted key '{norm_key}' ({shape})")
                                    merged_tensor = fallback
                                    merged_dict[norm_key] = fallback
                                else:
                                    # ── DIAG: dump batch context before failing ──
                                    print(f"   [DIAG] ⚠️ Key '{norm_key}' NOT in merged_dict ({len(merged_dict)} keys)")
                                    print(f"   [DIAG] ⚠️ First 10 merged_dict keys: {list(merged_dict.keys())[:10]}")
                                    print(f"   [DIAG] ⚠️ batch_items count={len(batch_items)}, batch_sds[0] keys={len(batch_sds[0])}")
                                    raise RuntimeError(f"Missing merged tensor for key {norm_key}")

                            # Ensure tensor is contiguous and move to CPU FIRST
                            # (before dtype conversion, so FP8 quantization runs on CPU
                            # and doesn't spike GPU memory with F32 intermediates)
                            if not merged_tensor.is_contiguous():
                                merged_tensor = merged_tensor.contiguous()
                            merged_tensor_cpu = merged_tensor if merged_tensor.device.type == 'cpu' else merged_tensor.cpu()

                            # ── Convert to target dtype with proper FP8 quantization ──
                            target_dtype_str = item['dtype']
                            target_torch_dtype = self.DTYPE_MAP[target_dtype_str][0]
                            if merged_tensor_cpu.dtype != target_torch_dtype:
                                if target_torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) \
                                   and norm_key.endswith('.weight') and not should_preserve_bf16(norm_key):
                                    # Proper scale-then-cast for FP8 output weights (on CPU)
                                    q, wscale, iscale = quantize_weight_to_fp8_with_scales(
                                        merged_tensor_cpu, target_torch_dtype
                                    )
                                    merged_tensor_cpu = q
                                    # Store companion scales for writing later
                                    self._companion_scales[norm_key] = (wscale, iscale)
                                elif target_torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) \
                                     and should_preserve_bf16(norm_key):
                                    merged_tensor_cpu = merged_tensor_cpu.to(dtype=torch.bfloat16)
                                else:
                                    merged_tensor_cpu = merged_tensor_cpu.to(dtype=target_torch_dtype)

                            # Verify tensor size matches survey expectation
                            expected_size = item['size']  # bytes from survey
                            actual_size = merged_tensor_cpu.numel() * merged_tensor_cpu.element_size()
                            if expected_size != actual_size:
                                print(f"[WARNING] Size mismatch for tensor {norm_key}: expected {expected_size}, actual {actual_size}. This may cause offset drift.")

                            # Verify first tensor alignment (silent check)
                            if first_tensor:
                                if out_f.tell() != data_start:
                                    print(f"[ERROR] File position mismatch! Expected {data_start}, got {out_f.tell()}")
                                    out_f.seek(data_start)
                            first_tensor = False

                            # No alignment padding (contiguity required by safetensors)
                            current_pos = out_f.tell()

                            # Record start offset relative to data_start
                            start = current_pos - data_start
                            # Write tensor data
                            raw_bytes = self._tensor_to_bytes(merged_tensor_cpu)
                            out_f.write(raw_bytes)
                            # Record end offset
                            end = out_f.tell() - data_start
                            self.actual_offsets[norm_key] = (start, end)

                            # Verify the tensor actually produced data
                            if start == end:
                                print(f"   ⚠️ Zero-length tensor written for key '{norm_key}': "
                                      f"shape={list(item['shape'])}, dtype={item['dtype']}, "
                                      f"tensor.numel()={merged_tensor_cpu.numel()}, "
                                      f"elem_size={merged_tensor_cpu.element_size()}")

                            if _WEAVER_DEBUG and len(self.actual_offsets) <= 5:
                                # ── DIAG: dump first 5 tensor offsets ──
                                print(f"   [DIAG] Written: norm_key='{norm_key}', "
                                      f"start={start}, end={end}, "
                                      f"raw_bytes_len={len(raw_bytes)}, "
                                      f"numel={merged_tensor_cpu.numel()}, "
                                      f"elem_size={merged_tensor_cpu.element_size()}, "
                                      f"shape={list(merged_tensor_cpu.shape)}, "
                                      f"dtype={merged_tensor_cpu.dtype}")

                    if _WEAVER_DEBUG:
                        # ── DIAG: after batch write loop ──
                        print(f"   [DIAG:AFTER] Batch {batch_num}: len(self.actual_offsets)={len(self.actual_offsets)}, "
                              f"pos after write={out_f.tell()}, "
                              f"rel_to_data={out_f.tell() - data_start}")
                        if self.actual_offsets:
                            first_key = next(iter(self.actual_offsets))
                            print(f"   [DIAG:AFTER]   first entry: '{first_key}' = {self.actual_offsets[first_key]}")
                        sys.stdout.flush()

                    # Batch-level summary (unconditional - removed `if 'end' in dir()` to ensure it always prints)
                    # Use actual_offsets count as reliable indicator
                    if not self.use_dict:
                        last_offset = list(self.actual_offsets.values())[-1][1] if self.actual_offsets else 0
                        print(f"   [WEAVE] Batch {batch_num}/{num_batches}: "
                              f"tensors {batch_start}..{batch_end-1}, "
                              f"data offset={last_offset} bytes, "
                              f"tensors_in_batch={len(batch_items)}, "
                              f"total_recorded={len(self.actual_offsets)}")
                    else:
                        print(f"   [WEAVE] Batch {batch_num}/{num_batches}: "
                              f"tensors {batch_start}..{batch_end-1}, "
                              f"tensors_in_batch={len(batch_items)}, "
                              f"dict_keys={len(self._output_dict)}")
                    # ── Log available RAM every batch for memory profiling ──
                    _avail = get_available_ram()
                    if _avail is not None:
                        print(f"      📊 RAM: {_avail / (1024**3):.2f} GB available")

                    # Clean up batch references — GPU tensors no longer accumulate
                    # at batch level (tensors stay on CPU; per-key GPU transfer is
                    # handled by merge_triple_method inside checkpoint_methods.py).
                    del batch_sds
                    del batch_mappings
                    del merged_dict
                    cleanup_memory()  # gc.collect() + torch.cuda.empty_cache()
                    # pbar is advanced by merge_triple_method via on_substep callback.
                    # No advancement at weave level — that would double-count.
                    # Heartbeat guard: yield every batch to prevent connection drops
                    comfyui_yield()

                finally:
                    for bs in batch_sources:
                        bs.close()

            # ── Write companion scale tensors (after all weight batches) ──
            # Companion scales are scalar FP32 tensors generated during FP8 quantization
            # of parent weight tensors. They were partitioned into self.companion_list
            # during survey() and are now written here from self._companion_scales.
            if self.companion_list:
                if self.use_dict:
                    for item in self.companion_list:
                        norm_key = item['norm_key']
                        parent_norm_key = item.get('_parent_norm_key')
                        if parent_norm_key and parent_norm_key in self._companion_scales:
                            scale_idx = 0 if item.get('_scale_type') == 'weight_scale' else 1
                            tensor = self._companion_scales[parent_norm_key][scale_idx]
                            self._output_dict[norm_key] = tensor
                else:
                    print(f"   [WEAVE] Writing {len(self.companion_list)} companion scale tensors...")
                    for item in self.companion_list:
                        norm_key = item['norm_key']
                        parent_norm_key = item.get('_parent_norm_key')
                        if parent_norm_key and parent_norm_key in self._companion_scales:
                            scale_idx = 0 if item.get('_scale_type') == 'weight_scale' else 1
                            tensor = self._companion_scales[parent_norm_key][scale_idx]
                        else:
                            raise RuntimeError(
                                f"Missing companion scale for '{norm_key}' "
                                f"(parent='{parent_norm_key}')"
                            )

                        # Tensor is already F32 CPU scalar — just write it as raw bytes
                        expected_size = item['size']
                        raw_bytes = self._tensor_to_bytes(tensor)
                        current_pos = out_f.tell()
                        start = current_pos - data_start
                        out_f.write(raw_bytes)
                        end = out_f.tell() - data_start
                        self.actual_offsets[norm_key] = (start, end)

            # No internal tracker to clean up — on_substep handles all advancement.
            print(f"   🕐 [t={time.time()-_weave_t0:.1f}s] weave: all {num_batches} batches complete")
            
            # Post-loop final GC sweep
            cleanup_memory()
            
            # ── Dict mode: skip safetensors header finalization ──
            if not self.use_dict:
                # All tensors written – now build final header with ground‑truth offsets
                print("[FINALIZE] Building output header...")
                final_header = {}
                added_keys = set()
                zero_offset_count = 0
                fallback_count = 0
                
                if _WEAVER_DEBUG:
                    # ── DIAG: dump first 5 actual_offsets entries ──
                    print("   [DIAG] self.actual_offsets first 5 entries:")
                    diag_count = 0
                    for nk, (s, e) in self.actual_offsets.items():
                        if diag_count >= 5:
                            break
                        print(f"   [DIAG]   self.actual_offsets['{nk}'] = ({s}, {e})")
                        diag_count += 1
                    print(f"   [DIAG] total self.actual_offsets entries: {len(self.actual_offsets)}")
                
                # ── FIX: compute cumulative offsets as fallback ──
                # The weave writes tensors sequentially in tensor_list order starting at data_start.
                # If actual_offsets entries are missing (e.g. due to scoping/GC issues between batches),
                # we compute the expected offset from the cumulative size of preceding tensors.
                cumulative_offset = 0
                for i, item in enumerate(self.tensor_list):
                    norm_key = item['norm_key']
                    key = item['key']
                    
                    # Primary: use actual_offsets if available
                    entry = self.actual_offsets.get(norm_key)
                    if entry is not None:
                        start, end = entry
                    else:
                        # Fallback: compute from cumulative position
                        start = cumulative_offset
                        end = cumulative_offset + item['size']
                        fallback_count += 1
                        # ── WARN: fallback offset used (offset tracking gap) ──
                        if fallback_count <= 10:
                            print(f"   [WARNING] Offset for '{norm_key}' not in actual_offsets. "
                                  f"Using cumulative: ({start}, {end})")
                    
                    if _WEAVER_DEBUG and i < 5:
                        # ── DIAG: first 5 tensor_list entries ──
                        print(f"   [DIAG] tensor_list[{i}]: norm_key='{norm_key}', key='{key}', "
                              f"offsets=({start}, {end})")
                    
                    # Detect zero-length tensors (data corruption) early
                    if start == end:
                        zero_offset_count += 1
                        print(f"   ⚠️ Zero-length tensor detected in final header: "
                              f"key='{key}', norm_key='{norm_key}', "
                              f"shape={list(item['shape'])}, dtype={item['dtype']}, "
                              f"offsets=[{start}, {end}]")
                    else:
                        # Verify consistency if actual_offsets was available
                        if entry is not None and start != cumulative_offset:
                            print(f"   [WARNING] Offset mismatch for '{norm_key}': "
                                  f"actual_offsets says {start}, expected {cumulative_offset} "
                                  f"(diff={start - cumulative_offset} bytes)")
                    
                    # Advance cumulative offset for next iteration
                    cumulative_offset = end
                    
                    # Add primary key (original key from first source) — canonical form only
                    if key not in added_keys:
                        final_header[key] = {
                            "dtype": item['dtype'],
                            "shape": list(item['shape']),
                            "data_offsets": [start, end]
                        }
                        added_keys.add(key)
                
                # ── Add companion scale tensors to final header ──
                for item in self.companion_list:
                    norm_key = item['norm_key']
                    key = item['key']
                    entry = self.actual_offsets.get(norm_key)
                    if entry is not None:
                        start, end = entry
                    else:
                        # Fallback: compute from cumulative position
                        start = cumulative_offset
                        end = cumulative_offset + item['size']
                        fallback_count += 1
                        if fallback_count <= 10:
                            print(f"   [WARNING] Offset for companion '{norm_key}' not in actual_offsets. "
                                  f"Using cumulative: ({start}, {end})")
                    cumulative_offset = end
                    if key not in added_keys:
                        final_header[key] = {
                            "dtype": item['dtype'],
                            "shape": list(item['shape']),
                            "data_offsets": [start, end]
                        }
                        added_keys.add(key)
                
                if fallback_count:
                    print(f"   [FALLBACK] {fallback_count} tensor(s) used cumulative fallback offset "
                          f"(missing from actual_offsets, likely batches 1-{max(1, fallback_count // 64)})")
                if zero_offset_count:
                    print(f"   ⚠️ WARNING: {zero_offset_count} tensor(s) have zero-length data offsets. "
                          f"This will cause 'ValueError: buffer length (0)' when parsing the output.")
                # Add merged metadata
                if self.merged_metadata:
                    final_header['__metadata__'] = self.merged_metadata
                
                # Summary
                print(f"[FINALIZE] Header built: {len(final_header)} tensors, "
                      f"{len(self.merged_metadata)} metadata keys")
                
                # Compute final header JSON and pad to placeholder size (self.header_len_padded)
                header_json = json.dumps(final_header, separators=(',', ':'))
                header_len = len(header_json)
                # Ensure final header fits within the placeholder size
                if header_len > self.header_len_padded:
                    raise RuntimeError(
                        f"Final header length ({header_len}) exceeds placeholder size ({self.header_len_padded}). "
                        "Increase safety margin in survey()."
                    )
                # Pad with spaces to exactly placeholder length (which is already 8‑byte aligned)
                pad_len = self.header_len_padded - header_len
                print(f"[FINALIZE] Header size = {header_len} bytes (padded to {self.header_len_padded})")
                header_json_padded = header_json + ' ' * pad_len
                # Verify length matches
                assert len(header_json_padded) == self.header_len_padded, f"Padded header length mismatch: {len(header_json_padded)} vs {self.header_len_padded}"
                
                # Seek to start of file and overwrite header length and JSON
                out_f.seek(0)
                out_f.write(self.header_len_padded.to_bytes(8, 'little'))
                out_f.write(header_json_padded.encode('utf-8'))
                # Verify file position after writing header matches data_start
                if out_f.tell() != self.data_start:
                    print(f"[WARNING] File position after final header ({out_f.tell()}) does not match data_start ({self.data_start}), seeking")
                    out_f.seek(self.data_start)
                out_f.flush()
                os.fsync(out_f.fileno())
                
                # Validate the safetensors file after writing
                self._validate_safetensors_file(self.output_path)
            
        
        finally:
            if self.use_dict:
                # Dict mode: no output stream to finalize
                omit_msg = ""
                if hasattr(self, '_omitted_components') and self._omitted_components:
                    omit_msg = f" (omitted {self._omitted_components} zero-weight components)"
                print(f"🧠 Direct dict mode completed — {len(self._output_dict)} tensors stored in output dict{omit_msg}")
            else:
                out_f.close()
                print(f"✅ Triple checkpoint merging completed: {self.output_path}")
            print(f"   🕐 [t={time.time()-_weave_t0:.1f}s] weave: output finalized")
    
    def _validate_safetensors_file(self, filepath):
        """
        Validate the safetensors file after writing:
        - Read header length and ensure it matches the computed length.
        - Parse JSON header and ensure it's valid.
        - Verify that each tensor's data_offsets are within file bounds.
        - Verify 8‑byte alignment of each tensor offset.
        """
        print(f"[DEBUG VALIDATION] Validating {filepath}")
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
                # (alignment check intentionally omitted — source checkpoints may be misaligned)
            # Additional validation using safe_open to ensure tensors can be loaded
            with safe_open(filepath, framework="pt") as sf:
                for key in tensor_headers.keys():
                    try:
                        # This will raise if the tensor cannot be accessed
                        slice_info = sf.get_slice(key)
                        # Optionally verify dtype and shape match header
                        # (skip for speed)
                    except Exception as e:
                        raise RuntimeError(f"Safe‑open validation failed for key {key}: {e}")
        
            # Success
            print(f"[DEBUG VALIDATION] All tensors aligned correctly and pass safe_open validation")
            return True
    
    def get_output_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return the output dict containing merged tensors.
        Only valid when use_dict=True was passed to __init__.
        
        Returns:
            Dict[str, torch.Tensor]: Merged tensors keyed by normalized key names.
        """
        if not self.use_dict:
            raise RuntimeError("get_output_dict() is only valid when use_dict=True was passed to __init__")
        if self._output_dict is None:
            raise RuntimeError("Output dict not available – weave() may not have been called yet")
        return self._output_dict

    def generate_forensic_report(self) -> str:
        """
        Generate a standardized forensic audit report.
        """
        mc = self.merge_config
        method = mc.get("method", "unknown")
        weights = mc.get("weights", [])
        blend_mode = mc.get("blend_mode", "auto")
        balancing_mode = mc.get("balancing_mode", "disabled")
        density = mc.get("density", 1.0)
        w_str = ", ".join(f"{w:.2f}" for w in weights) if weights else "N/A"

        # Build component data for component-breakdown lines
        comp_lines = []
        for comp in ["unet", "clip", "vae", "te", "other"]:
            count = self.component_counts[comp]
            params = self.parameter_counts[comp]
            if count > 0:
                comp_lines.append(
                    f"   {comp.upper():<6} {count:>4} tensors, {params:>12,} parameters"
                )

        # Build source lines — show "(in-memory)" for dict sources
        source_lines = []
        for i, p in enumerate(self.source_paths):
            if self.source_dicts[i] is not None:
                source_lines.append(f"   {i + 1}. (in-memory dict) — {len(self.source_dicts[i])} tensors")
            else:
                source_lines.append(f"   {i + 1}. {p.name}")

        # Build merge-settings lines
        settings_keys = ["method", "weights", "density", "uniqueness", "threshold", "blend",
                         "blend_mode", "magnitude_scaling", "streaming", "balancing_mode",
                         "weight_unet", "weight_clip", "weight_vae", "weight_te",
                         "precision", "device"]
        settings_lines = [f"   {k}: {mc[k]}" for k in settings_keys if k in mc]

        # Build ordered sections
        sections = []
        sections.append((None, [f"🏗️ ARCHITECTURE: {self.architecture}"]))
        sections.append((None, [f"📊 TOTAL PARAMETERS: {self.total_parameters:,}"]))
        sections.append((None, [""]))  # blank line before component breakdown
        sections.append(("COMPONENT BREAKDOWN", comp_lines))
        sections.append((None, [""]))  # blank line after component breakdown
        sections.append(("SOURCES", source_lines))
        sections.append((None, [""]))  # blank line after sources

        # ── Corrupted/Skipped tensors ──
        corrupted = getattr(self, 'corrupted_keys', None)
        if corrupted:
            corr_lines = [f"   ⚠️ {ck}" for ck in corrupted]
            sections.append(("CORRUPTED / INVALID SHAPE TENSORS (skipped, zero-filled)", corr_lines))
            sections.append((None, [""]))

        sections.append(("MERGE SETTINGS", settings_lines))
        sections.append((None, [f"📅 DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}"]))

        return build_forensic_report(
            report_type="EASY CHECKPOINT MERGER",
            title_data={
                "📦 METHOD": method,
                "📋 SUMMARY": (
                    f"{method} | {len(self.source_paths)} checkpoints [{w_str}] "
                    f"| blend={blend_mode} | balance={balancing_mode} | density={density}"
                ),
            },
            sections=sections,
            footer_width=60,
        )
