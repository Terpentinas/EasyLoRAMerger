"""
Utility functions and classes for Easy LoRA Merger.
"""

import warnings
import time
import uuid
import hashlib
import gc
import json
import struct
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import folder_paths
import comfy.utils
import comfy.sd


# Suppress warnings
warnings.filterwarnings("ignore", message="lora key not loaded")


def comfyui_yield():
    """Yield control to ComfyUI if possible, otherwise sleep briefly."""
    try:
        # Try different ComfyUI yield methods
        if hasattr(comfy.utils, 'yield_for_comfyui'):
            comfy.utils.yield_for_comfyui()
        elif hasattr(comfy.utils, 'yield_current'):
            comfy.utils.yield_current()
        else:
            time.sleep(0.001)
    except:
        time.sleep(0.001)


def cleanup_memory(*objs):
    """
    Run garbage collection and clear CUDA cache.

    NOTE: Passing objects to this function does NOT delete them from the
    caller's scope.  Python's `del obj` inside a function only deletes the
    local parameter reference.  To actually free memory, callers MUST use
    explicit `del` at the call site:

        del batch_sds, batch_mappings, merged_dict
        cleanup_memory()  # gc.collect() + torch.cuda.empty_cache()
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class NodeCache:
    """MD5-hash based change detection for ComfyUI IS_CHANGED."""

    _CACHE: Dict[tuple, str] = {}

    @classmethod
    def check_changed(cls, *cache_values) -> str:
        """
        Hash-based change detection for IS_CHANGED.

        Returns the MD5 hash of cache_values.
        If the same values produce the same hash, ComfyUI treats it as unchanged.

        Usage in IS_CHANGED:
            @classmethod
            def IS_CHANGED(cls, **kwargs):
                return NodeCache.check_changed(
                    cls.__name__,
                    kwargs.get('param1', ''),
                    kwargs.get('param2', ''),
                )
        """
        key_str = str(cache_values).encode('utf-8')
        current_hash = hashlib.md5(key_str).hexdigest()

        if cache_values in cls._CACHE and cls._CACHE[cache_values] == current_hash:
            return cls._CACHE[cache_values]

        cls._CACHE[cache_values] = current_hash
        return current_hash

    @classmethod
    def is_changed(cls, class_name: str, **kwargs) -> str:
        """
        Unified IS_CHANGED helper for ComfyUI nodes.

        Data-guard params (wired inputs that can't be cheaply hashed) should
        be checked inline before calling this method — see baker_node.py or
        checkpoint_merger_node.py for the recommended pattern.

        Usage:
            @classmethod
            def IS_CHANGED(cls, **kwargs):
                # Inline data guard FIRST
                if kwargs.get('lora_data_a') is not None:
                    return float("nan")
                # Then hash-based caching for the rest
                return NodeCache.is_changed(
                    cls.__name__,
                    lora_a=kwargs.get('lora_a', 'None'),
                    method=kwargs.get('method', 'linear'),
                )

        Args:
            class_name: cls.__name__ for namespacing the cache key.
            **kwargs: Named parameters to hash for change detection.

        Returns:
            str: MD5 hash of parameters for cache hit detection.
        """
        key_str = str((class_name, kwargs)).encode('utf-8')
        current_hash = hashlib.md5(key_str).hexdigest()

        cache_key = (class_name, tuple(sorted(kwargs.items())))
        if cache_key in cls._CACHE and cls._CACHE[cache_key] == current_hash:
            return cls._CACHE[cache_key]

        cls._CACHE[cache_key] = current_hash
        return current_hash


class ProgressTracker:
    """
    Wraps ComfyUI's ProgressBar with console fallback + ETA.

    - If comfy.utils.ProgressBar is available, uses the visual UI progress bar.
    - Otherwise, prints percentage updates to console at intervals.
    - Displays ETA for operations exceeding 5 seconds.
    - Thread-safe guard: single-threaded use within ComfyUI pipeline.

    Singleton awareness:
    - ComfyUI only supports ONE comfy.utils.ProgressBar instance at a time.
    - The class tracks _active_pbar so nested ProgressTrackers don't compete.
    - When a child tracker is created (via parent=), it delegates to the parent.
    - When a standalone tracker is created while one is already active,
      it uses console fallback without creating a competing UI bar.
    """
    # Class-level: tracks the ONE active ComfyUI ProgressBar instance.
    # Prevents nested ProgressTrackers from creating competing UI bars.
    _active_pbar = None

    def __init__(self, total: int, desc: str = "", update_interval: float = 0.5,
                 parent: 'ProgressTracker' = None):
        self.total = max(total, 1)
        self.desc = desc
        self.update_interval = update_interval
        self.current = 0
        self.start_time = time.time()
        self.last_print = 0.0
        self._pbar = None
        self._parent = parent
        self._owns_pbar = False

        if parent is not None:
            # Child tracker: delegate to parent, never create own ComfyUI bar
            self._pbar = None
        elif ProgressTracker._active_pbar is None:
            # Top-level tracker: create the ONE and only ComfyUI bar
            try:
                if hasattr(comfy.utils, 'ProgressBar'):
                    self._pbar = comfy.utils.ProgressBar(self.total)
                    ProgressTracker._active_pbar = self
                    self._owns_pbar = True
            except Exception:
                self._pbar = None
        else:
            # Another standalone tracker is already active — use console fallback
            # to avoid competing with the existing ComfyUI ProgressBar.
            self._pbar = None

    def update(self, n: int = 1):
        """Advance progress by n steps. Updates UI bar or prints to console."""
        self.current += n
        if self._parent is not None:
            # Delegate to parent tracker
            self._parent.update(n)
        elif self._pbar is not None:
            self._pbar.update(n)
        else:
            # Console fallback — print at intervals
            now = time.time()
            if now - self.last_print >= self.update_interval:
                pct = self.current / self.total * 100
                elapsed = now - self.start_time
                eta_str = ""
                if self.current > 0 and elapsed > 5:
                    rate = elapsed / self.current
                    remaining = (self.total - self.current) * rate
                    if remaining > 5:
                        eta_str = f" [ETA: {remaining:.0f}s]"
                print(f"\r{self.desc}: {self.current}/{self.total} ({pct:.0f}%){eta_str}", end="", flush=True)
                self.last_print = now

    def complete(self):
        """Mark progress as complete. Ensures final display."""
        if self._parent is not None:
            # Child: nothing to finalize — parent handles its own display
            return
        if self._pbar is None:
            elapsed = time.time() - self.start_time
            remaining = self.total - self.current
            if remaining > 0:
                self.current = self.total
            print(f"\r{self.desc}: {self.total}/{self.total} (100%) — {elapsed:.1f}s")
        else:
            remaining = self.total - self.current
            if remaining > 0:
                self._pbar.update(remaining)
                self.current = self.total
            # Release class-level ref so the next standalone tracker can create a bar
            if self._owns_pbar:
                ProgressTracker._active_pbar = None

    def __enter__(self):
        """Context manager entry — returns self for use as `with ProgressTracker(...) as p:`."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — auto-completes progress on normal exit or exception."""
        self.complete()
        # Do not suppress exceptions
        return False

    def __iadd__(self, n: int):
        """Support `progress += 1` syntax as sugar for `progress.update(1)`."""
        self.update(n)
        return self


class NullProgressTracker:
    """Silent progress tracker — accepts all calls but does nothing.
    
    Used as a default parameter value in place of None to eliminate
    `if tracker is None` conditional branches throughout the codebase.
    Implements the same public interface as ProgressTracker.
    Stateless — all instances behave identically, safe as default param.
    """
    def update(self, n: int = 1) -> None:
        pass

    def __iadd__(self, n: int) -> 'NullProgressTracker':
        return self

    def __enter__(self) -> 'NullProgressTracker':
        return self

    def __exit__(self, *args) -> None:
        pass

def load_lora_with_metadata(path: Path):
    """Load LoRA with metadata and validation."""
    print(f"📥 Loading {path.name}...")
    
    # First try standard safetensors loading
    try:
        with safe_open(path, framework="pt") as f:
            tensors = {}
            failed_keys = []
            
            keys = list(f.keys())
            print(f"   Found {len(keys)} keys in safetensors")
            
            # Show sample of keys for debugging
            # Filter to show actual weight/bias keys instead of noise schedule params
            weight_keys = [k for k in keys if '.weight' in k or '.bias' in k]
            if weight_keys:
                print(f"   Sample keys: {weight_keys[:3]}")
            else:
                print(f"   Sample keys: {keys[:3]}")
            if len(keys) > 3:
                print(f"   ... and {len(keys) - 3} more")
            
            for k in keys:
                try:
                    tensors[k] = f.get_tensor(k)
                except Exception as e:
                    print(f"⚠️ Failed to load tensor {k}: {e}")
                    failed_keys.append(k)
            
            if failed_keys:
                print(f"⚠️ Failed to load {len(failed_keys)} keys")
            
            try:
                metadata = f.metadata() or {}
                if metadata:
                    print(f"   Metadata: {list(metadata.keys())[:5]}...")
            except Exception as e:
                print(f"⚠️ Failed to load metadata: {e}")
                metadata = {}
            
            print(f"✅ Loaded {len(tensors)} tensors from {path.name}")
            return tensors, metadata
    except Exception as e:
        print(f"⚠️ Failed to open {path} with safe_open: {e}")
        
        # Try fallback to load_file (handles more formats)
        try:
            print(f"🔧 Trying fallback loading for {path.name}...")
            tensors = load_file(str(path))
            print(f"   Loaded {len(tensors)} tensors via fallback")
            
            # Try to extract metadata if possible
            metadata = {}
            try:
                # Some LoRA files store metadata in special keys
                if '__metadata__' in tensors:
                    metadata = tensors['__metadata__']
                    del tensors['__metadata__']
                    print(f"   Found metadata with {len(metadata)} entries")
            except:
                pass
            
            return tensors, metadata
        except Exception as e2:
            # Last resort: try as PyTorch pickle file
            try:
                print(f"🔧 Trying PyTorch loading for {path.name}...")
                tensors = torch.load(str(path), map_location='cpu', weights_only=False)
                
                # Convert to dictionary if it's not already
                if not isinstance(tensors, dict):
                    print(f"❌ Unknown format in {path.name}")
                    raise ValueError(f"Unknown file format: {path.name}")
                
                print(f"✅ Loaded as PyTorch file: {len(tensors)} tensors")
                
                metadata = {}
                return tensors, metadata
            except Exception as e3:
                # All loading methods failed
                error_msg = f"Cannot load LoRA file {path.name}:\n"
                error_msg += f"1. safetensors error: {e}\n"
                error_msg += f"2. load_file error: {e2}\n"
                if 'weights_only' in str(e3):
                    error_msg += f"3. torch.load error: File may be unsafe (try enabling weights_only=False)\n"
                else:
                    error_msg += f"3. torch.load error: {e3}\n"
                raise ValueError(error_msg)




class DeviceManager:
    """Manages device and precision settings with optimal defaults."""

    # Minimum free VRAM (bytes) to prefer CUDA over CPU in "auto" mode
    VRAM_AUTO_THRESHOLD_BYTES: int = 2 * (1024**3)  # 2 GB

    @staticmethod
    def get_device(device_type: str = "auto", architecture: str = None) -> torch.device:
        """Get torch device with fallback logic.

        Args:
            device_type: ``"auto"``, ``"cuda"``, or ``"cpu"``.
            architecture: Optional model architecture name for architecture-aware VRAM
                threshold. If provided, uses the architecture-specific minimum VRAM
                instead of the flat 2 GB threshold.

        Returns:
            torch.device: Resolved device.
        """
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "auto":
            if torch.cuda.is_available():
                free_vram = DeviceManager.get_free_vram()
                if free_vram is not None:
                    # Use architecture-specific threshold if provided
                    threshold = (DeviceManager.get_vram_threshold_for_architecture(architecture)
                                 if architecture else DeviceManager.VRAM_AUTO_THRESHOLD_BYTES)
                    if free_vram < threshold:
                        arch_label = f" for {architecture}" if architecture else ""
                        print(f"⚠️ [DeviceManager] Only {free_vram / (1024**3):.2f} GB free VRAM "
                              f"(minimum {threshold / (1024**3):.1f} GB{arch_label}) — "
                              f"falling back to CPU")
                        return torch.device("cpu")
                return torch.device("cuda")
            return torch.device("cpu")
        else:
            if device_type == "cuda" and not torch.cuda.is_available():
                print("⚠️ [DeviceManager] CUDA explicitly selected but not available — falling back to CPU")
            return torch.device("cpu")

    @staticmethod
    def get_free_vram(device: Optional[torch.device] = None) -> Optional[int]:
        """Return free VRAM in bytes, or None if not CUDA / info unavailable."""
        if device is None:
            if not torch.cuda.is_available():
                return None
            device = torch.device("cuda")
        if device.type != "cuda":
            return None
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            return total - reserved
        except Exception:
            return None

    @staticmethod
    def get_vram_threshold_for_architecture(architecture: str) -> int:
        """Return minimum free VRAM bytes required for given model architecture.

        Used by :meth:`get_device` when an architecture name is available.
        Thresholds account for: model weights on GPU + peak merge overhead + safety margin.

        Args:
            architecture: Model architecture name (e.g. ``"sdxl"``, ``"flux"``).

        Returns:
            int: Minimum free VRAM in bytes.
        """
        thresholds = {
            "sd15":     3 * (1024**3),   # 3 GB  — SD1.5: ~2 GB model + 1 GB overhead
            "sdxl":     8 * (1024**3),   # 8 GB  — SDXL: ~6 GB model + 2 GB overhead
            "flux":    16 * (1024**3),   # 16 GB — Flux: ~12 GB model + 4 GB overhead
            "z_image":  8 * (1024**3),   # 8 GB  — Z-Image: varies 4-8 GB
            "anima":    4 * (1024**3),   # 4 GB  — Anima: varies 2-4 GB
            "lumina2": 12 * (1024**3),   # 12 GB — Lumina2: ~8 GB model + 4 GB overhead
        }
        return thresholds.get(architecture, 6 * (1024**3))  # default 6 GB for unknown

    @staticmethod
    def suggest_batch_size(
        device: torch.device,
        architecture: Optional[str] = None,
    ) -> int:
        """Suggest an appropriate batch_size based on free VRAM and architecture.

        Args:
            device: Target torch device.
            architecture: Optional model architecture name for per-tensor size estimation.

        Returns:
            int: Suggested batch_size (clamped to 1-128).
        """
        if device.type != "cuda":
            return 64  # CPU: sequential is fine, batch matters less

        free_vram = DeviceManager.get_free_vram(device)
        if free_vram is None:
            return 32  # safe default when VRAM info unavailable

        # Estimate per-tensor memory (architecture-dependent)
        if architecture == "flux":
            per_tensor_estimate = 8 * 1024 * 1024  # 8 MB per tensor
        elif architecture in ("sdxl", "z_image"):
            per_tensor_estimate = 4 * 1024 * 1024  # 4 MB per tensor
        else:
            per_tensor_estimate = 2 * 1024 * 1024  # 2 MB per tensor (SD1.5 typical)

        # 3 sources + 1 output = 4× per batch during merge
        # Baker: 1 source + 1 delta = 2× per batch
        # Use 3× as a middle-ground estimate
        batch_overhead = per_tensor_estimate * 3
        max_batch = free_vram // batch_overhead if batch_overhead > 0 else 32

        # Clamp to reasonable range
        return max(1, min(max_batch, 128))
    
    @staticmethod
    def get_dtype(precision: str = "auto", device: torch.device = None) -> torch.dtype:
        """Get optimal dtype with FP8/NVFP4 support detection and fallback."""
        if device is None:
            device = DeviceManager.get_device()
        
        # Convert string device type to torch.device
        if isinstance(device, str):
            device = DeviceManager.get_device(device)
        
        if precision == "float32":
            return torch.float32
        elif precision == "bfloat16":
            if device.type == "cuda" and not torch.cuda.is_bf16_supported():
                print("⚠️ bfloat16 not supported on this GPU, falling back to float16")
                return torch.float16
            return torch.bfloat16
        elif precision == "float16":
            return torch.float16
        elif precision == "fp8_e4m3fn":
            result = _try_fp8_dtype('float8_e4m3fn', "fp8_e4m3fn", device)
            if result is not None:
                return result
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif precision == "fp8_e5m2":
            result = _try_fp8_dtype('float8_e5m2', "fp8_e5m2", device)
            if result is not None:
                return result
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif precision == "fp8":
            result = _try_fp8_dtype('float8_e4m3fn', "fp8", device)
            if result is not None:
                return result
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:  # "auto"
            if device.type == "cuda":
                # Auto-select best precision
                # Default to bfloat16 for supported GPUs, otherwise float16
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float32
    
    @staticmethod
    def get_fallback_dtype(current_dtype: torch.dtype) -> torch.dtype:
        """Get a fallback dtype when current dtype operations fail."""
        if current_dtype == torch.float8_e4m3fn:
            # FP8 failed, try bfloat16
            if torch.cuda.is_bf16_supported():
                print("🔄 FP8 (e4m3fn) operation failed, falling back to bfloat16")
                return torch.bfloat16
            else:
                print("🔄 FP8 (e4m3fn) operation failed, falling back to float16")
                return torch.float16
        elif current_dtype == torch.float8_e5m2:
            # FP8 failed, try bfloat16
            if torch.cuda.is_bf16_supported():
                print("🔄 FP8 (e5m2) operation failed, falling back to bfloat16")
                return torch.bfloat16
            else:
                print("🔄 FP8 (e5m2) operation failed, falling back to float16")
                return torch.float16
        elif current_dtype == torch.bfloat16:
            # bfloat16 failed, try float16
            print("🔄 bfloat16 operation failed, falling back to float16")
            return torch.float16
        else:
            # float16 failed, try float32
            print("🔄 float16 operation failed, falling back to float32")
            return torch.float32


def _try_fp8_dtype(dtype_attr: str, label: str, device: torch.device) -> Optional[torch.dtype]:
    """Attempt to return an FP8 dtype if hardware supports it, otherwise None."""
    if device.type == "cuda" and hasattr(torch, dtype_attr):
        device_cap = torch.cuda.get_device_capability(device)
        # FP8 requires compute capability 8.9+ (Hopper)
        if device_cap[0] >= 9 or (device_cap[0] == 8 and device_cap[1] >= 9):
            print(f"⚠️ {label} selected but operations may be limited")
            print("   Will attempt FP8 but fall back to bfloat16 if operations fail")
            return getattr(torch, dtype_attr)
        else:
            print(f"⚠️ {label} not supported on device (compute {device_cap}), falling back to bfloat16")
    else:
        print(f"⚠️ {label} not available in this PyTorch version, falling back to bfloat16")
    return None


@contextmanager
def memory_optimized_merge():
    """Context manager for memory-optimized merging."""
    original_state = {
        "grad_enabled": torch.is_grad_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    
    try:
        # Disable gradients for merging
        torch.set_grad_enabled(False)
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        yield
    finally:
        # Restore original state
        torch.set_grad_enabled(original_state["grad_enabled"])
        torch.backends.cudnn.benchmark = original_state["cudnn_benchmark"]
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def memory_guard():
    """
    Check CUDA VRAM usage and empty cache if above 80% threshold.
    
    Called between batches during baking / checkpoint merging to prevent
    OOM from accumulated intermediate tensors.
    Safe to call on CPU (no-op when CUDA is unavailable).
    """
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            if total > 0 and allocated / total > 0.8:
                torch.cuda.empty_cache()
        except Exception:
            pass


def get_available_ram() -> Optional[int]:
    """
    Returns available physical RAM in bytes, or None if detection fails.
    Tries psutil first (cross-platform), falls back to Windows ctypes.
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass

    try:
        # Windows fallback via ctypes kernel32.GlobalMemoryStatusEx
        import ctypes
        kernel32 = ctypes.windll.kernel32
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        memoryStatus = MEMORYSTATUSEX()
        memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
        return memoryStatus.ullAvailPhys
    except Exception:
        pass

    return None  # Detection unavailable


def estimate_ram_mode_peak(total_data_size: int, num_tensors: int, batch_size: int, use_dict: bool = False) -> int:
    """
    Estimate peak memory for in-memory modes.
    
    Peak depends on which mode is active:
    - Dict mode (use_dict=True): ~1.5x total_data_size
      No BytesIO/clone overhead — state dict IS the merge output.
      During weave: output dict grows to ~1x total_data_size (system RAM).
      During model loading: output dict (~1x) partially overlaps with
      model objects (~1x GPU, negligible system RAM impact via get_available_ram).
      Conservative peak: 1.5x accounts for brief overlap during ComfyUI loading
      when state dict is partially consumed and model objects partially built.
      When precision=bf16/fp16, dict uses ~half the fp32 memory.
      Phase 3 (zero-weight auto-omit) further reduces dict size proportionally.
    - Legacy buffer mode (use_buffer=True): ~2x total_data_size
      Phase 1 (parse): buffer (~1x) + growing state dict (~0.5x partial) = ~1.5x
      Phase 2 (del writer -> ComfyUI load): state dict (~1x) + model objects (~1x) = ~2x
    - Batch processing during weave: negligible vs parse/load phases
    
    We conservatively estimate 2x for legacy buffer mode, 1.5x for dict mode,
    plus batch overhead. Dict mode users get ~25% lower estimate automatically
    (more with low-precision or zero-weight omission), reducing unnecessary
    fallbacks to disk mode on moderately RAM-constrained systems.
    """
    # Batch processing overhead during weave
    batch_fraction = min(batch_size / max(num_tensors, 1), 1.0)
    batch_memory = int(total_data_size * batch_fraction * 0.5)
    
    if use_dict:
        # Dict mode: state dict IS the merge output (no BytesIO copy).
        # Model objects use GPU VRAM, not system RAM (measured by get_available_ram).
        # During ComfyUI loading, ~0.5x state_dict + ~1x model objects overlap briefly
        # on CPU, but model objects are GPU-resident so system RAM impact is minimal.
        # Conservative: 1.5x accounts for partial system RAM overlap.
        operation_peak = int(total_data_size * 1.5)
    else:
        # Legacy buffer mode: BytesIO (~1x) + state_dict (~1x) + model objects (~1x)
        # can briefly coexist during parse+load on system RAM. Conservative: 2x.
        operation_peak = int(total_data_size * 2.0)
    
    return operation_peak + batch_memory


def check_ram_guard(
    total_bytes: int,
    num_tensors: int,
    batch_size: int,
    label: str = "operation",
    threshold_ratio: Optional[float] = None,
    use_dict: bool = False,
) -> bool:
    """
    Check if RAM is sufficient for an in-memory operation.

    Returns True if safe (within threshold), False if fallback recommended.

    Args:
        total_bytes: Total size of data to load (in bytes)
        num_tensors: Number of tensors (for peak estimation)
        batch_size: Batch size for peak estimation
        label: Human-readable label for diagnostic messages
        threshold_ratio: Fraction of available RAM to use as threshold
                         (default sourced from config.RAM_GUARD_THRESHOLD = 0.85)
        use_dict: If True, use dict-mode peak estimate (~1.5x) instead of
                  legacy buffer mode (~2x). Dict mode avoids the BytesIO
                  serialization spike, lowering system RAM peak.
    """
    if threshold_ratio is None:
        from .config import RAM_GUARD_THRESHOLD
        threshold_ratio = RAM_GUARD_THRESHOLD

    available_ram = get_available_ram()
    if available_ram is None:
        print(f"⚠️ RAM Guard ({label}): Cannot detect available RAM — proceeding without guard")
        return True  # Can't determine, assume safe

    peak_estimate = estimate_ram_mode_peak(total_bytes, num_tensors, batch_size, use_dict=use_dict)
    threshold_ram = int(available_ram * threshold_ratio)

    print(f"🧠 RAM Guard ({label}): Estimated peak ~{peak_estimate / (1024**3):.2f} GB, "
          f"Available RAM ~{available_ram / (1024**3):.2f} GB, "
          f"{threshold_ratio*100:.0f}% threshold ~{threshold_ram / (1024**3):.2f} GB")

    if peak_estimate > threshold_ram:
        print(f"   ⚠️ RAM Guard ({label}): Peak estimate exceeds {threshold_ratio*100:.0f}% threshold — fallback recommended")
        return False
    else:
        print(f"✅ RAM Guard ({label}): Peak estimate within safe limits")
        return True


def estimate_disk_mode_peak(total_data_size: int, num_tensors: int, batch_size: int) -> int:
    """
    Estimate peak memory for disk mode.
    - Streamed to file (disk buffers, negligible RAM for output)
    - Batch processing: batch_size/num_tensors * total_data_size
    - Lazy loading: negligible until ComfyUI access
    """
    batch_fraction = min(batch_size / max(num_tensors, 1), 1.0)
    return int(total_data_size * batch_fraction * 1.2)  # 20% overhead


def get_free_disk_space(path: Path) -> Optional[int]:
    """
    Returns free disk space in bytes on the drive containing `path`,
    or None if detection fails.
    """
    try:
        if hasattr(Path, 'statvfs'):  # Unix-like
            stat = path.statvfs()
            return stat.f_frsize * stat.f_bavail
        else:  # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_t(str(path.absolute())),
                None, None, ctypes.pointer(free_bytes)
            )
            return free_bytes.value
    except Exception:
        return None

class ThreadSafeCleanup:
    """Thread-safe temporary file cleanup."""
    _lock = threading.Lock()
    _registered_files = set()
    
    @classmethod
    def cleanup_temp_directory(cls):
        """Main cleanup function for all temp files with thread safety."""
        with cls._lock:
            temp_dir = Path(folder_paths.get_temp_directory())
            
            # 1. Clean easy_lora folder (keep last 10)
            easy_lora_dir = temp_dir / "easy_lora"
            if easy_lora_dir.exists():
                files = list(easy_lora_dir.glob("*.safetensors"))
                if len(files) > 10:
                    files.sort(key=lambda x: x.stat().st_mtime)
                    for old_file in files[:-10]:
                        try:
                            old_file.unlink()
                        except Exception as e:
                            print(f"⚠️ Failed to delete {old_file}: {e}")
            
            # 2. Clean temp_* files from LORA data conversion (keep last 5)
            temp_files = list(temp_dir.glob("temp_*.safetensors"))
            if len(temp_files) > 5:
                temp_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in temp_files[:-5]:
                    try:
                        old_file.unlink()
                    except Exception as e:
                        print(f"⚠️ Failed to delete {old_file}: {e}")
            
            # 3. Clean checkpoint_temp_* files from chained data (keep last 5)
            ckpt_temp_files = list(temp_dir.glob("checkpoint_temp_*.safetensors"))
            if len(ckpt_temp_files) > 5:
                ckpt_temp_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in ckpt_temp_files[:-5]:
                    try:
                        old_file.unlink()
                    except Exception as e:
                        print(f"⚠️ Failed to delete {old_file}: {e}")
            
            # 4. Clean old cache files (older than 1 hour)
            cache_files = list(temp_dir.glob("*.cache"))
            current_time = time.time()
            for cache_file in cache_files:
                try:
                    if current_time - cache_file.stat().st_mtime > 3600:
                        cache_file.unlink()
                except Exception:
                    pass

    @classmethod
    def register_temp_file(cls, path):
        """Register a temporary file for later cleanup."""
        with cls._lock:
            cls._registered_files.add(Path(path))


def get_experiment_temp_path(node_type: str = "main") -> Path:
    """Get temp path with cleanup."""
    # Run cleanup in background thread
    cleanup_thread = threading.Thread(target=ThreadSafeCleanup.cleanup_temp_directory)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Create path
    temp_dir = Path(folder_paths.get_temp_directory()) / "easy_lora"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    random_str = uuid.uuid4().hex[:6]
    
    if node_type == "main":
        filename = f"experiment_main_{timestamp}_{random_str}.safetensors"
    else:
        filename = f"experiment_only_{timestamp}_{random_str}.safetensors"
    
    return temp_dir / filename


def _get_output_path(
    save_folder: str,
    filename: str,
    default_name: str,
    folder_type: str,
) -> Path:
    """Shared output path resolution with auto-increment for LoRAs and checkpoints."""
    # Default filename
    if not filename or filename.strip() == "":
        filename = default_name

    # Ensure .safetensors extension
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"

    # Clean filename
    safe_filename = ''.join(c for c in filename
                           if c.isalnum() or c in '._- ').rstrip()

    # Get default base folder
    folders = folder_paths.get_folder_paths(folder_type)
    default_base = Path(folders[0]) if folders else Path.cwd()

    # Resolve output folder
    if save_folder and isinstance(save_folder, str) and save_folder.strip():
        user_path = save_folder.strip()
        print(f"📁 User requested save folder: '{user_path}'")

        is_windows_absolute = (len(user_path) >= 2 and
                              user_path[1] == ':' and
                              user_path[2] == '\\')
        is_network_path = user_path.startswith('\\\\')

        if is_windows_absolute or is_network_path:
            try:
                output_folder = Path(user_path)
                if '..' in user_path or user_path.count(':\\') > 1:
                    print(f"⚠️ Suspicious path pattern, using default")
                    output_folder = default_base
                else:
                    output_folder.mkdir(parents=True, exist_ok=True)
                    print(f"✅ Using absolute path: {output_folder}")
            except Exception as e:
                print(f"⚠️ Could not use absolute path '{user_path}': {e}")
                output_folder = default_base
        else:
            try:
                clean_path = user_path.strip('\\/')
                output_folder = default_base / clean_path
                output_folder.mkdir(parents=True, exist_ok=True)
                print(f"✅ Using relative path: {output_folder}")
            except Exception as e:
                print(f"⚠️ Could not create subfolder '{user_path}': {e}")
                output_folder = default_base
    else:
        output_folder = default_base

    # Auto-increment logic
    base_path = output_folder / safe_filename
    counter = 1
    final_path = base_path
    while final_path.exists():
        stem = base_path.stem
        parts = stem.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            stem = '_'.join(parts[:-1])
        final_path = base_path.parent / f"{stem}_{counter}{base_path.suffix}"
        counter += 1

    print(f"💾 Final save path: {final_path}")
    return final_path


def get_user_output_path(save_folder: str, filename: str) -> Path:
    """Get output path for a merged LoRA with auto-increment."""
    return _get_output_path(save_folder, filename, "merged_lora", "loras")


def get_checkpoint_output_path(save_folder: str, filename: str) -> Path:
    """Get output path for a merged checkpoint with auto-increment."""
    return _get_output_path(save_folder, filename, "merged_checkpoint", "checkpoints")


# ── Streaming safetensors writer (avoids BytesIO 2× memory spike) ──

_DTYPE_SIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "F8": 1, "BOOL": 1,
}

DTYPE_REVERSE_STREAM = {
    torch.float64: "F64", torch.float32: "F32",
    torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64",   torch.int32: "I32",
    torch.int16: "I16",   torch.int8: "I8",
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8",
    torch.float8_e5m2: "F8",
    torch.bool: "BOOL",
}


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a tensor to raw bytes for safetensors storage.

    Automatically moves the tensor to CPU first — safetensors always
    stores CPU bytes. Supports all dtypes including bfloat16 and fp8
    (which numpy cannot represent natively) via .view(torch.uint8)
    reinterpretation. Handles scalar tensors (dim=0) for the .view() path.
    """
    # Defensive CPU transfer — safetensors stores CPU bytes only
    tensor = tensor.cpu()

    # Defensive contiguity check — prevents cryptic numpy errors
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Dtypes needing .view(torch.uint8) workaround:
    # bfloat16, fp8_e4m3fn, fp8_e5m2 — numpy has no native support
    if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        if tensor.dim() == 0:
            tensor = tensor.reshape(-1)  # view() requires at least 1 dim
        return tensor.view(torch.uint8).numpy().tobytes()

    return tensor.numpy().tobytes()


def save_safetensors_stream(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write safetensors file by streaming tensors directly to disk,
    avoiding the BytesIO intermediate buffer used by save_file().

    Peak memory: state_dict only (no 2× BytesIO spike).

    Args:
        tensors: Dict of key -> contiguous CPU tensor.
        filename: Output file path.
        metadata: Optional metadata dict (must be JSON-serializable).

    Raises:
        RuntimeError: On write failure after partial file creation.
    """
    # Phase 1: Build header and compute cumulative offsets
    sorted_keys = sorted(tensors.keys())
    header: Dict[str, Any] = {"__metadata__": metadata or {}}
    offset = 0
    for key in sorted_keys:
        tensor = tensors[key]
        dtype_str = DTYPE_REVERSE_STREAM.get(tensor.dtype, "F32")
        shape = list(tensor.shape)
        numel = tensor.numel()
        elem_size = _DTYPE_SIZE.get(dtype_str, 4)
        data_size = numel * elem_size
        header[key] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + data_size],
        }
        offset += data_size

    # Phase 2: Serialize to file, one tensor at a time
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_size = struct.pack("<Q", len(header_bytes))

    try:
        with open(filename, "wb") as f:
            f.write(header_size)
            f.write(header_bytes)
            for key in sorted_keys:
                tensor = tensors[key]
                f.write(_tensor_to_bytes(tensor))
    except BaseException:
        # Clean up partial file — catch BaseException to handle
        # KeyboardInterrupt (Ctrl+C) during long writes
        try:
            Path(filename).unlink(missing_ok=True)
        except Exception:
            pass
        raise


def save_safetensors_file(
    state_dict: Dict[str, torch.Tensor],
    save_folder: str,
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
    folder_type: str = "loras",
    default_name: str = "output",
) -> Tuple[Optional[Path], str]:
    """
    Unified save utility for safetensors files with error handling.

    Resolves the output path via _get_output_path, creates parent dirs,
    attempts save_file, and returns the result.

    Args:
        state_dict: The tensor dict to save.
        save_folder: User-specified folder override (empty = default).
        filename: Desired filename (without extension or with .safetensors).
        metadata: Optional metadata dict to embed in the safetensors file.
        folder_type: ComfyUI folder type ('loras', 'checkpoints', etc.).
        default_name: Fallback name if filename is empty.

    Returns:
        Tuple of (output_path or None on failure, path_string or error message).
    """
    output_path = _get_output_path(save_folder, filename, default_name, folder_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    try:
        save_file(state_dict, str(output_path), metadata=metadata)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved: {output_path} ({file_size_mb:.1f} MB)")
        return output_path, str(output_path)
    except Exception as e:
        error_msg = f"❌ Failed to save {default_name}: {e}"
        print(error_msg)
        return None, error_msg


def safe_get_rank(tensor: torch.Tensor, key: str) -> int:
    """Safely get rank from any LoRA tensor."""
    if len(tensor.shape) < 2:
        return 1  # Alpha or bias

    key_lower = key.lower()

    # lora_A types
    if any(x in key_lower for x in ['lora_a', 'lora.down', 'lora_down']):
        return tensor.shape[0]  # [rank, in_dim]

    # lora_B types
    elif any(x in key_lower for x in ['lora_b', 'lora.up', 'lora_up']):
        if len(tensor.shape) > 1:
            return tensor.shape[1]
        else:
            return tensor.shape[0]

    # Default fallback
    else:
        return min(tensor.shape[0], tensor.shape[1])


def categorize_key(key: str) -> str:
    """
    Categorize a key into 'model', 'te', or 'other'.

    **Note:** This is a backward-compatible wrapper that delegates to
    :func:`engine.key_utils.categorize_key` but maps ``'unet'`` → ``'model'``
    to preserve the original API (callers expect ``'model'`` rather than
    ``'unet'``).
    """
    from .engine.key_utils import categorize_key as _cat
    comp = _cat(key)
    # backward compat: old API returned 'model' not 'unet'
    if comp == 'unet':
        return 'model'
    return comp


def get_tensor_energy(tensor: torch.Tensor, mode: str) -> float:
    """
    Compute the energy (RMS or percentile) of a tensor.
    Modes: "none" (returns 1.0), "rms", "top_5%", "top_10%", "top_20%", "top_30%".
    """
    if mode == "none":
        return 1.0
    abs_tensor = torch.abs(tensor)
    if mode == "rms":
        return torch.sqrt(torch.mean(abs_tensor ** 2))
    # Parse percentile from mode string, e.g., "top_5%" -> quantile = 0.95
    if mode.startswith("top_"):
        try:
            percent = float(mode.split('_')[1].rstrip('%'))
            quantile = 1.0 - (percent / 100.0)
            # Flatten tensor for quantile computation
            flat = abs_tensor.flatten().float()
            return torch.quantile(flat, quantile)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid percentile mode: {mode}")
    raise ValueError(f"Unknown magnitude scaling mode: {mode}")


def silent_pad_or_truncate(tensor: torch.Tensor, target_rank: int, key: str) -> torch.Tensor:
    """Pad or truncate tensor to target rank."""
    current_rank = safe_get_rank(tensor, key)

    if current_rank == target_rank:
        return tensor

    # Determine if this is lora_A or lora_B
    is_lora_b = any(x in key.lower() for x in ['lora_b', 'lora.up', 'lora_up'])

    if len(tensor.shape) <= 1:
        # Scalar or 1D tensor (alpha?) - no change
        return tensor

    if not is_lora_b:
        # lora_A: rank is axis 0
        new_shape = (target_rank,) + tensor.shape[1:]
        new_tensor = torch.zeros(new_shape, device=tensor.device, dtype=tensor.dtype)
        min_rank = min(current_rank, target_rank)
        # Slice along axis 0
        new_tensor[:min_rank, ...] = tensor[:min_rank, ...]
    else:
        # lora_B: rank is axis 1
        new_shape = (tensor.shape[0], target_rank) + tensor.shape[2:]
        new_tensor = torch.zeros(new_shape, device=tensor.device, dtype=tensor.dtype)
        min_rank = min(current_rank, target_rank)
        # Slice along axis 1
        new_tensor[:, :min_rank, ...] = tensor[:, :min_rank, ...]

    return new_tensor


def save_lora_data_to_temp(lora_data, name: str) -> Optional[Path]:
    """Save LORA data tuple to temporary safetensors file."""
    if lora_data is None:
        return None
    
    try:
        lora_dict, _, _ = lora_data
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create secure temp filename
        temp_filename = f"temp_{name}_{uuid.uuid4().hex}.safetensors"
        temp_path = temp_dir / temp_filename
        
        # Validate before saving
        if not isinstance(lora_dict, dict):
            raise ValueError("LORA data must be a dictionary")
        
        save_file(lora_dict, str(temp_path))
        return temp_path
    except Exception as e:
        print(f"❌ Failed to save temp LORA data: {e}")
        return None


def save_checkpoint_data_to_temp(state_dict, name: str, metadata=None) -> Optional[Path]:
    """Save a checkpoint state dict to a temporary safetensors file (for chaining/preview)."""
    if state_dict is None:
        return None
    try:
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_filename = f"checkpoint_temp_{name}_{uuid.uuid4().hex}.safetensors"
        temp_path = temp_dir / temp_filename
        save_file(state_dict, str(temp_path), metadata=metadata or {})
        ThreadSafeCleanup.register_temp_file(temp_path)
        return temp_path
    except Exception as e:
        print(f"❌ Failed to save checkpoint temp file: {e}")
        return None

def bake_alphas(state_dict, naming_style='lora_a_b'):
    """
    Bake alpha values into weight tensors and delete .alpha keys.
    Applies sqrt(alpha/rank) scaling to both down and up weight tensors.
    
    Delegates tensor scaling to :func:`engine.scale_utils.bake_alpha_key`.
    
    naming_style: either 'lora_a_b' (lora_A.weight, lora_B.weight) or
                  'lora_down_up' (lora_down.weight, lora_up.weight).
    """
    from .engine.scale_utils import resolve_scale, bake_alpha_key
    
    # Determine suffixes for down and up weight keys
    if naming_style == 'lora_a_b':
        down_suffix = '.lora_A.weight'
        up_suffix = '.lora_B.weight'
    else:
        down_suffix = '.lora_down.weight'
        up_suffix = '.lora_up.weight'
    
    # Collect alpha keys to delete
    alpha_keys = [k for k in state_dict.keys() if k.endswith('.alpha')]
    if not alpha_keys:
        # No alpha keys, nothing to bake
        return state_dict
    
    print(f"🧪 Baking {len(alpha_keys)} alpha keys...")
    scaled_count = 0
    total_scale = 0.0
    mismatch_count = 0
    mismatch_details = []
    
    # Iterate over alpha keys
    for alpha_key in alpha_keys:
        # Derive weight keys
        base = alpha_key[:-6]  # remove '.alpha'
        down_key = base + down_suffix
        up_key = base + up_suffix
        
        # Ensure both keys exist
        if down_key not in state_dict or up_key not in state_dict:
            print(f"⚠️ Missing weight tensors for alpha key {alpha_key}, skipping")
            continue
        
        down_tensor = state_dict[down_key]
        up_tensor = state_dict[up_key]
        alpha_tensor = state_dict[alpha_key]
        # Alpha value (scalar)
        if isinstance(alpha_tensor, torch.Tensor):
            alpha_val = alpha_tensor.item()
        else:
            alpha_val = float(alpha_tensor)
        
        # Rank from weight shapes (min dimension)
        rank = safe_get_rank(down_tensor, down_key)
        if rank == 0:
            rank = 1  # safety
        
        # Scaling factor and tensor baking via shared scale_utils
        scale = resolve_scale(alpha_val, rank, mode="sqrt")
        orig_down_norm = torch.norm(down_tensor).item()
        orig_up_norm = torch.norm(up_tensor).item()
        state_dict[down_key], state_dict[up_key] = bake_alpha_key(
            down_tensor, up_tensor, alpha_val, rank
        )
        actual_down_scale = torch.norm(state_dict[down_key]).item() / orig_down_norm
        actual_up_scale = torch.norm(state_dict[up_key]).item() / orig_up_norm
        
        # Update statistics
        total_scale += scale
        scaled_count += 1
        
        # Check for significant mismatch (relative tolerance 1%)
        tolerance = 0.01
        rel_down = abs(actual_down_scale - scale) / scale
        rel_up = abs(actual_up_scale - scale) / scale
        if rel_down > tolerance or rel_up > tolerance:
            mismatch_count += 1
            mismatch_details.append((alpha_key, scale, actual_down_scale, actual_up_scale, alpha_val, rank))
    
    # Delete alpha keys
    for alpha_key in alpha_keys:
        if alpha_key in state_dict:
            del state_dict[alpha_key]
    
    # Summary
    if scaled_count > 0:
        avg_scale = total_scale / scaled_count
        if abs(avg_scale - 1.0) > 1e-6:
            print(f"   Average scale detected: {avg_scale:.6f}")
    
    if mismatch_count > 0:
        print(f"   ⚠️  {mismatch_count} keys exceeded 1% tolerance")
        # Print up to 5 mismatches for debugging
        for i, (alpha_key, scale, down_act, up_act, alpha_val, rank) in enumerate(mismatch_details[:5]):
            print(f"      {alpha_key}: expected {scale:.6f}, down actual {down_act:.6f}, up actual {up_act:.6f} (alpha={alpha_val}, rank={rank})")
        if mismatch_count > 5:
            print(f"      ... and {mismatch_count - 5} more")
    # No mismatches: remain silent (no extra line)
    
    print(f"✅ Baked {len(alpha_keys)} alpha keys ({scaled_count} scaled)")
    return state_dict


def categorize_checkpoint_key(key: str) -> str:
    """
    Categorize a checkpoint tensor key into component: 'unet', 'clip', 'vae', 'te', or 'other'.

    Delegates to :func:`engine.key_utils.categorize_key` with the same return
    values (``'unet'``, ``'clip'``, ``'vae'``, ``'te'``, ``'other'``).
    """
    from .engine.key_utils import categorize_key as _cat
    return _cat(key)


def load_state_dict_as_model_objects(
    state_dict,
    metadata=None,
    output_vae=True,
    output_clip=True,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Convert a state dict or lazy mapping to ComfyUI MODEL, CLIP, VAE objects.

    Returns (model, clip, vae).
    NOTE: ComfyUI pops keys from the input state_dict. Use .copy() if preserving the original.
    """
    model, clip, vae, _ = comfy.sd.load_state_dict_guess_config(
        state_dict,
        output_vae=output_vae,
        output_clip=output_clip,
        output_clipvision=False,
        embedding_directory=None,
        model_options={},
        te_model_options={},
        metadata=metadata,
        disable_dynamic=False,
    )
    return model, clip, vae


def load_checkpoint_with_metadata(path: Path, categorize: bool = False):
    """
    Load checkpoint tensors and metadata from a safetensors file.
    Returns (tensors, metadata) if categorize=False.
    If categorize=True, returns (tensors, metadata, component_counts) where
    component_counts is a dict mapping component to number of keys.
    """
    tensors, metadata = load_lora_with_metadata(path)
    if not categorize:
        return tensors, metadata
    # Compute component counts
    counts = {'unet': 0, 'clip': 0, 'vae': 0, 'te': 0, 'other': 0}
    for key in tensors.keys():
        comp = categorize_checkpoint_key(key)
        counts[comp] += 1
    return tensors, metadata, counts


def apply_component_scaling(tensor: torch.Tensor, key: str,
                            weight_unet: float = 1.0,
                            weight_clip: float = 1.0,
                            weight_vae: float = 1.0,
                            weight_te: float = 1.0) -> torch.Tensor:
    """
    Scale a tensor according to its component category.
    Used for Weight Block Map scaling before merging.
    """
    comp = categorize_checkpoint_key(key)
    if comp == 'unet':
        scale = weight_unet
    elif comp == 'clip':
        scale = weight_clip
    elif comp == 'vae':
        scale = weight_vae
    elif comp == 'te':
        scale = weight_te
    else:
        # 'other' components: no scaling (or default 1.0)
        scale = 1.0
    if scale != 1.0:
        tensor = tensor * scale
    return tensor




def compute_component_energy_ratios(
    norm_sds: List[Dict[str, torch.Tensor]],
    common_keys: List[str],
    original_sds: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
    mappings: Optional[List[Optional[Dict[str, str]]]] = None,
    converted_flags: Optional[List[bool]] = None,
    key_categorizer: callable = categorize_key,
) -> Dict[str, List[float]]:
    """
    Compute per-component mean-squared-energy for N input models across common keys.

    Uses per-element mean(tensor^2) instead of sum, making the metric rank-independent
    (no zero-padding bias from rank mismatches). Only shared keys (common_keys) are
    analyzed, preventing architectural mismatches from affecting the energy ratio.

    Args:
        norm_sds: List of normalized state dicts (one per input model).
        common_keys: Keys present in ALL input state dicts (shared layers only).
        original_sds: Optional original SDs for alpha/rank scaling lookup.
        mappings: Optional key mappings (normalized -> original) for alpha lookup.
        converted_flags: Optional list of bools indicating if each model is 'converted'.
        key_categorizer: Function to categorize keys into components.

    Returns:
        Dict mapping component name -> list of accumulated mean energies
        [energy_0, energy_1, ...] for each input model.
    """
    num_inputs = len(norm_sds)
    if original_sds is None:
        original_sds = [None] * num_inputs
    if mappings is None:
        mappings = [None] * num_inputs
    if converted_flags is None:
        converted_flags = [False] * num_inputs

    energy_by_component: Dict[str, List[float]] = {}

    for key in common_keys:
        tensors = []
        for i in range(num_inputs):
            tensor = norm_sds[i][key]
            # Apply LoRA alpha/rank scaling if original data is available
            # (for unconverted LoRAs with known alpha values)
            rank = safe_get_rank(tensor, key)
            rank = max(1, rank)

            if original_sds[i] is not None:
                from ..engine.scale_utils import find_alpha_value
                alpha_value = find_alpha_value(original_sds[i], key, mapping=mappings[i])
                if not converted_flags[i] and alpha_value is not None:
                    scale_factor = alpha_value / rank
                    if abs(scale_factor - 1.0) > 1e-6:
                        # Apply scaling only to up-weights (same logic as merge engine)
                        is_down = any(s in key for s in ['.lora_A.weight', '.lora_down.weight'])
                        if not is_down:
                            tensor = tensor * scale_factor

            tensors.append(tensor)

        # Categorize key
        component = key_categorizer(key)
        if component not in energy_by_component:
            energy_by_component[component] = [0.0] * num_inputs

        # Accumulate per-element mean energy (rank-independent)
        # Using mean(t^2) instead of sum(t^2) avoids bias from rank mismatches
        # Skip integer tensors (e.g. position_ids) which have no meaningful energy
        for i, t in enumerate(tensors):
            if t.dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
                continue
            energy_by_component[component][i] += torch.mean(t ** 2).item()

    return energy_by_component


def compute_primary_driver_intensity_metric(
    norm_sds: List[Dict[str, torch.Tensor]],
    common_keys: List[str],
    energy_concentration: float = 0.80,
    original_sds: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
    mappings: Optional[List[Optional[Dict[str, str]]]] = None,
    converted_flags: Optional[List[bool]] = None,
    key_categorizer: callable = categorize_key,
) -> Dict[str, dict]:
    """
    Compute per-LoRA 'Primary Driver Intensity' using adaptive energy concentration.

    Instead of a fixed top_percent (like compute_lora_intensity_metric), this function
    dynamically determines which keys carry the majority (80%) of each LoRA's total
    energy and compares them on a JOINT key set (union) for a fair apples-to-apples
    comparison.

    Why this is better than fixed top-percentile:
      - Sparse LoRAs (Anima/Z-Image): 80% energy concentrated in ~5-10% of keys
        → focuses on truly important keys only
      - Dense LoRAs: 80% energy spread over ~30-40% of keys
        → captures adequate context
      - Cross-concept merges: joint union set ensures same-key comparison,
        preventing the "apples to oranges" bias where each LoRA's peak
        is measured on its own strongest keys where the other is weak.

    Args:
        norm_sds: List of normalized state dicts (one per input model).
        common_keys: Keys present in ALL input state dicts (shared layers only).
        energy_concentration: Fraction of total energy to capture (default 0.80 = 80%).
        original_sds: Optional original SDs for alpha/rank scaling lookup.
        mappings: Optional key mappings (normalized -> original) for alpha lookup.
        converted_flags: Optional list of bools indicating if each model is 'converted'.
        key_categorizer: Function to categorize keys into components.

    Returns:
        Dict mapping component name -> dict with keys:
            'peaks': [peak_A, peak_B]
                Mean energy on each LoRA's OWN primary driver set.
            'joint_peaks': [joint_A, joint_B]
                Mean energy on UNION of both LoRAs' primary driver sets (fair comparison).
            'primary_driver_counts': [count_A, count_B]
                Number of primary driver keys for each LoRA.
            'joint_count': int
                Number of keys in the union set.
            'energy_concentration': [conc_A, conc_B]
                Actual fraction of total energy captured by primary drivers.
            'ratio_joint': float
                sqrt(joint_A / joint_B) — THE metric to use for global scaling.
            'overlap_count': int
                Number of keys that are primary drivers for BOTH LoRAs.
    """
    import math
    num_inputs = len(norm_sds)
    if original_sds is None:
        original_sds = [None] * num_inputs
    if mappings is None:
        mappings = [None] * num_inputs
    if converted_flags is None:
        converted_flags = [False] * num_inputs

    # Phase 1: Collect per-key energies per component per model
    # {component: [[(key, energy), ...], [(key, energy), ...]]}
    per_key_energies: Dict[str, List[List[tuple]]] = {}

    for key in common_keys:
        component = key_categorizer(key)
        if component not in per_key_energies:
            per_key_energies[component] = [[] for _ in range(num_inputs)]

        for i in range(num_inputs):
            tensor = norm_sds[i][key]
            rank = safe_get_rank(tensor, key)
            rank = max(1, rank)

            # Alpha/rank scaling for unconverted LoRAs (same logic as existing functions)
            if original_sds[i] is not None:
                from ..engine.scale_utils import find_alpha_value
                alpha_value = find_alpha_value(original_sds[i], key, mapping=mappings[i])
                if not converted_flags[i] and alpha_value is not None:
                    scale_factor = alpha_value / rank
                    if abs(scale_factor - 1.0) > 1e-6:
                        is_down = any(s in key for s in ['.lora_A.weight', '.lora_down.weight'])
                        if not is_down:
                            tensor = tensor * scale_factor

            # Per-element mean energy (rank-independent)
            energy = torch.mean(tensor ** 2).item()
            per_key_energies[component][i].append((key, energy))

    # Phase 2-4: Find Primary Drivers, form Joint Set, compute metrics
    result: Dict[str, dict] = {}
    for component, model_key_energies in per_key_energies.items():
        primary_sets: List[set] = []   # list of sets of primary driver keys per model
        peaks: List[float] = []        # mean energy on own primary drivers
        concentrations: List[float] = []# actual energy fraction captured
        all_energies_list: List[dict] = []  # key->energy lookup per model

        for key_energies in model_key_energies:
            # Sort by energy descending
            sorted_items = sorted(key_energies, key=lambda x: x[1], reverse=True)
            total_energy = sum(e for _, e in sorted_items)

            # Build lookup dict for later use
            energy_lookup = {k: e for k, e in key_energies}
            all_energies_list.append(energy_lookup)

            # Find smallest N where cumulative energy >= concentration_threshold * total
            cumsum = 0.0
            cutoff_idx = 0
            for idx, (k, e) in enumerate(sorted_items):
                cumsum += e
                if cumsum >= energy_concentration * total_energy:
                    cutoff_idx = idx + 1
                    break
            else:
                cutoff_idx = len(sorted_items)

            # Primary driver keys = first cutoff_idx keys
            primary_keys = set(k for k, _ in sorted_items[:cutoff_idx])
            primary_sets.append(primary_keys)

            # Peak = mean energy of primary drivers
            if primary_keys:
                peak = sum(e for k, e in sorted_items[:cutoff_idx]) / cutoff_idx
            else:
                peak = 0.0
            peaks.append(peak)
            concentrations.append(cumsum / total_energy if total_energy > 0 else 0.0)

        # Phase 3: Joint union set
        joint_set: set = set()
        for ps in primary_sets:
            joint_set |= ps

        # Phase 4: Compute joint mean energies on the union set
        joint_peaks: List[float] = []
        for i, lookup in enumerate(all_energies_list):
            if joint_set:
                joint_total = sum(lookup.get(k, 0.0) for k in joint_set)
                joint_peak = joint_total / len(joint_set)
            else:
                joint_peak = 0.0
            joint_peaks.append(joint_peak)

        # Ratio on joint set
        eps = 1e-12
        if joint_peaks[1] > eps:
            ratio_joint = math.sqrt(joint_peaks[0] / joint_peaks[1])
        else:
            ratio_joint = 1.0

        # Overlap count
        overlap_count = 0
        if len(primary_sets) >= 2:
            overlap_count = len(primary_sets[0] & primary_sets[1])

        result[component] = {
            'peaks': peaks,
            'joint_peaks': joint_peaks,
            'primary_driver_counts': [len(ps) for ps in primary_sets],
            'joint_count': len(joint_set),
            'energy_concentration': concentrations,
            'ratio_joint': ratio_joint,
            'overlap_count': overlap_count,
        }

    return result


def _lookup_alpha_value(orig_key: str, original_sd: Dict[str, torch.Tensor]) -> Optional[float]:
    """DEPRECATED: Use engine.scale_utils.find_alpha_value instead."""
    import warnings
    warnings.warn(
        "_lookup_alpha_value is deprecated, use engine.scale_utils.find_alpha_value",
        DeprecationWarning, stacklevel=2,
    )
    """
    Look up the alpha value for a given original key.
    Tries multiple candidate alpha key patterns to handle various naming conventions.
    """
    candidates = []
    # Pattern 1: replace .weight with .alpha
    if orig_key.endswith(".weight"):
        candidates.append(orig_key.replace(".weight", ".alpha"))
    # Pattern 2: replace lora_A/B.weight with alpha
    for pat in ['lora_A.weight', 'lora_B.weight', 'lora_down.weight', 'lora_up.weight']:
        if pat in orig_key:
            candidates.append(orig_key.replace(pat, "alpha"))
    # Pattern 3: strip suffix after last .lora_
    if ".lora_" in orig_key:
        base = orig_key.split(".lora_")[0]
        candidates.append(f"{base}.alpha")
    if "_lora_" in orig_key:
        base = orig_key.split("_lora_")[0]
        candidates.append(f"{base}.alpha")

    for ak in candidates:
        if ak in original_sd:
            alpha_tensor = original_sd[ak]
            if isinstance(alpha_tensor, torch.Tensor):
                return alpha_tensor.item() if alpha_tensor.numel() == 1 else alpha_tensor.mean().item()
    return None
