EASY_LORA_MERGER_VERSION = "1.1.0"
EASY_LORA_MERGER_DATE = "2026-02-15"
import warnings
from pathlib import Path
import time
import uuid
import json
import hashlib
import threading
import re  # ADD THIS IMPORT
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Set, Literal, Tuple, Union
from contextlib import contextmanager

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import numpy as np

import folder_paths
import comfy.sd
import comfy.utils

from .methods import (
    merge_linear, merge_ties_strict, merge_ties_gentle, merge_ties_gentle, merge_ties_gentle, merge_subtract, merge_magnitude, merge_feature_mix, merge_svd_preserve, merge_noise_aware, merge_gradient_alignment, apply_density, merge_dare_lite, merge_dare_rescale, universal_merge_executor
)
from .klein_normalizer import (
    normalize_all_klein_formats, 
    pad_or_truncate,
    universal_normalize, 
    universal_finalize, 
    safe_get_rank, 
    detect_lora_format
)

# Suppress warnings
warnings.filterwarnings("ignore", message="lora key not loaded")

# Let's also add better error handling for the yield function:
def comfyui_yield():
    """Yield control to ComfyUI if possible, otherwise sleep briefly."""
    try:
        # Try different ComfyUI yield methods
        if hasattr(comfy.utils, 'yield_for_comfyui'):
            comfyui_yield()
        elif hasattr(comfy.utils, 'yield_current'):
            comfy.utils.yield_current()
        else:
            time.sleep(0.001)
    except:
        time.sleep(0.001)

# ==================== CONFIGURATION CLASSES ====================

@dataclass
class MergeConfig:
    """Configuration for LoRA merging operations."""
    method: str = "linear"
    density: float = 1.0
    weight_a: float = 1.0
    weight_b: float = 1.0
    confidence_a: float = 0.5  # Keep for backward compatibility
    confidence_b: float = 0.5  # Keep for backward compatibility
    attn_weight: float = 1.0    # Keep for backward compatibility
    mlp_weight: float = 0.7     # Keep for backward compatibility
    device_type: Literal["auto", "cuda", "cpu"] = "auto"
    precision: Literal["auto", "float32", "bfloat16", "float16", "fp8"] = "auto"
    metadata_mode: Literal["none", "preserve_a", "preserve_b", "merge_basic"] = "merge_basic"
    batch_size: int = 32
    streaming: bool = True
    
    # NEW ATTRIBUTES for new methods
    uniqueness: float = 0.7      # For feature_mix
    threshold: float = 0.0       # For subtract
    blend: float = 0.5           # For magnitude
    
    @classmethod
    def from_inputs(cls, **kwargs) -> 'MergeConfig':
        """Create config from node inputs."""
        valid_fields = cls.__annotations__.keys()
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered)

@dataclass
class ModelMetadata:
    """Structured metadata for LoRA models."""
    ss_base_model: Optional[str] = None
    ss_sd_model_name: Optional[str] = None
    ss_network_module: Optional[str] = None
    ss_network_dim: Optional[int] = None
    ss_network_alpha: Optional[int] = None
    ss_training_started_at: Optional[str] = None
    lora_merge_tool: str = "EasyLoRAMerger"
    merge_method: Optional[str] = None
    merged_date: Optional[str] = None
    
    @classmethod
    def from_safetensors(cls, metadata: Dict[str, str]) -> 'ModelMetadata':
        """Parse metadata from safetensors file."""
        args = {}
        for field in cls.__annotations__.keys():
            if field in metadata:
                args[field] = metadata[field]
        return cls(**args)

# ==================== METHOD REGISTRY ====================

class MergeMethodRegistry:
    """Registry for all merge methods with consistent interface."""
    
    _methods: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a merge method."""
        def decorator(func):
            cls._methods[name] = func
            return func
        return decorator
    
    @classmethod
    def get_method(cls, name: str):
        """Get merge method by name."""
        method = cls._methods.get(name)
        if not method:
            raise ValueError(f"Unknown merge method: {name}")
        return method
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """List all available merge methods."""
        return list(cls._methods.keys())

# In your easy_lora_merger.py, update the method registrations:
MergeMethodRegistry.register("linear")(merge_linear)
MergeMethodRegistry.register("ties_strict")(merge_ties_strict)
MergeMethodRegistry.register("ties_gentle")(merge_ties_gentle)
MergeMethodRegistry.register("dare_lite")(merge_dare_lite)
MergeMethodRegistry.register("dare_rescale")(merge_dare_rescale)
MergeMethodRegistry.register("subtract")(merge_subtract)  # NEW
MergeMethodRegistry.register("magnitude")(merge_magnitude)  # NEW
MergeMethodRegistry.register("feature_mix")(merge_feature_mix)  # NEW
MergeMethodRegistry.register("svd_preserve")(merge_svd_preserve)
MergeMethodRegistry.register("noise_aware")(merge_noise_aware)
MergeMethodRegistry.register("gradient_alignment")(merge_gradient_alignment)

# ==================== SECURITY FUNCTIONS ====================

class SecurityError(Exception):
    """Security-related exceptions."""
    pass

def sanitize_path(user_path: str, allowed_base: Path) -> Path:
    """
    Sanitize user input to prevent path traversal attacks.
    
    Args:
        user_path: User-provided path
        allowed_base: Base directory the path must stay within
        
    Returns:
        Sanitized Path object
        
    Raises:
        SecurityError: If path traversal is detected
    """
    if not user_path or not isinstance(user_path, str):
        return allowed_base
    
    # Clean up path
    clean_path = user_path.strip().replace('\\', '/').replace('//', '/')
    
    # Remove null bytes and control characters
    clean_path = ''.join(c for c in clean_path if ord(c) >= 32 and ord(c) != 127)
    
    try:
        # Create relative path
        relative = Path(clean_path)
        if relative.is_absolute():
            raise SecurityError(f"Abolute paths not allowed: {clean_path}")
        
        # Resolve against allowed base
        resolved = (allowed_base / relative).resolve()
        
        # Ensure the resolved path is within allowed base
        if not resolved.is_relative_to(allowed_base.resolve()):
            raise SecurityError(f"Path traversal attempt: {clean_path}")
        
        # Additional security: check for dangerous patterns
        dangerous_patterns = [
            '..', '../', '/..', '~', '/etc/', '/bin/', '/usr/', 
            'cmd.exe', 'powershell', 'bash', 'sh'
        ]
        
        path_str = str(resolved).lower()
        for pattern in dangerous_patterns:
            if pattern in path_str:
                raise SecurityError(f"Dangerous path pattern detected: {pattern}")
        
        return resolved
    except (ValueError, RuntimeError) as e:
        raise SecurityError(f"Invalid path: {clean_path}") from e

# Update the validate_safetensors_file function to be more permissive:

def validate_safetensors_file(path: Path) -> bool:
    """
    Validate safetensors file with relaxed checking.
    Many LoRA files have non-standard headers, so we need to be more permissive.
    
    Args:
        path: Path to safetensors file
        
    Returns:
        True if file appears to be a valid safetensors or LoRA file
    """
    try:
        if not path.exists():
            return False
        
        # Check file size
        file_size = path.stat().st_size
        if file_size < 16:  # Minimum size for any safetensors
            return False
        
        with open(path, 'rb') as f:
            # Try to read magic bytes
            try:
                magic = f.read(8)
            except:
                return False
            
            # Check for safetensors magic OR accept any file that passes basic checks
            if magic == b'__safet':
                # Valid safetensors magic
                try:
                    # Try to read header size
                    header_size_bytes = f.read(8)
                    if len(header_size_bytes) != 8:
                        return False
                    
                    header_size = int.from_bytes(header_size_bytes, 'little')
                    
                    # Validate header size is reasonable
                    if header_size < 2 or header_size > 100 * 1024 * 1024:  # 100MB max
                        return False
                    
                    # Read header
                    header_bytes = f.read(header_size)
                    if len(header_bytes) != header_size:
                        return False
                    
                    try:
                        header = json.loads(header_bytes)
                    except json.JSONDecodeError:
                        # Could be a LoRA with non-JSON header, still accept it
                        # Many LoRA files have non-standard formats
                        return True  # Accept anyway, let safetensors library handle it
                    
                    # Basic validation
                    if not isinstance(header, dict):
                        return False
                    
                    # Check for at least one tensor entry
                    has_tensors = False
                    for key, info in header.items():
                        if key == '__metadata__':
                            continue
                        
                        if isinstance(info, dict) and 'dtype' in info and 'shape' in info:
                            has_tensors = True
                            break
                    
                    return has_tensors
                except Exception:
                    # If we can't parse as standard safetensors, still accept it
                    # Many LoRA files are non-standard
                    return True
            
            else:
                # Not standard safetensors magic, but could still be a LoRA file
                # Many LoRA files have different headers
                
                # Check if it might be a PyTorch .pt file masquerading as safetensors
                f.seek(0)
                first_bytes = f.read(100)
                
                # Check for common LoRA patterns in the first 100 bytes
                patterns_to_check = [
                    b'lora_', b'diffusion_model', b'transformer',
                    b'input_blocks', b'output_blocks', b'middle_block'
                ]
                
                for pattern in patterns_to_check:
                    if pattern in first_bytes:
                        # Likely a valid LoRA file
                        return True
                
                # Try to load with safetensors anyway (some files have corrupt headers but valid data)
                return True  # Let the safetensors library decide
    except Exception:
        # If any error occurs during validation, be permissive
        # The actual loading code will catch invalid files
        return True  # Accept and let loading fail with a better error message

# Also update the load_lora_with_metadata function to be more informative:

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
            if len(keys) > 0:
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

# Also, we should add a warning about Flux model merging since the error mentions "Flux2TEModel":

def validate_flux_model_compatibility(sd_a: Dict[str, torch.Tensor], 
                                     sd_b: Dict[str, torch.Tensor]) -> bool:
    """Special validation for Flux model merging."""
    is_flux_a = any('diffusion_model' in k for k in sd_a.keys())
    is_flux_b = any('diffusion_model' in k for k in sd_b.keys())
    
    if is_flux_a and is_flux_b:
        print("🔬 Detected Flux model merging")
        
        # Check Flux architecture versions
        has_double_a = any('double_blocks' in k for k in sd_a.keys())
        has_double_b = any('double_blocks' in k for k in sd_b.keys())
        
        if has_double_a != has_double_b:
            print(f"⚠️ Different Flux architectures detected!")
            print(f"   A has double_blocks: {has_double_a}")
            print(f"   B has double_blocks: {has_double_b}")
            print("   This merge may produce unexpected results!")
        
        # Check for img_attn vs txt_attn patterns
        img_attn_a = any('img_attn' in k for k in sd_a.keys())
        txt_attn_a = any('txt_attn' in k for k in sd_a.keys())
        img_attn_b = any('img_attn' in k for k in sd_b.keys())
        txt_attn_b = any('txt_attn' in k for k in sd_b.keys())
        
        if (img_attn_a != img_attn_b) or (txt_attn_a != txt_attn_b):
            print(f"⚠️ Different attention patterns in Flux models")
    
    return True

# ==================== PERFORMANCE OPTIMIZATIONS ====================

class DeviceManager:
    """Manages device and precision settings with optimal defaults."""
    
    @staticmethod
    def get_device(device_type: str = "auto") -> torch.device:
        """Get torch device with fallback logic."""
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device("cpu")
    
    @staticmethod
    def get_dtype(precision: str = "auto", device: torch.device = None) -> torch.dtype:
        """Get optimal dtype with FP8/NVFP4 support detection and fallback."""
        if device is None:
            device = DeviceManager.get_device()
        
        if precision == "float32":
            return torch.float32
        elif precision == "bfloat16":
            if device.type == "cuda" and not torch.cuda.is_bf16_supported():
                print("⚠️ bfloat16 not supported on this GPU, falling back to float16")
                return torch.float16
            return torch.bfloat16
        elif precision == "float16":
            return torch.float16
        elif precision == "fp8":
            # Check for FP8 support (Hopper+ GPUs)
            if device.type == "cuda" and hasattr(torch, 'float8_e4m3fn'):
                cuda_version = torch.version.cuda
                device_cap = torch.cuda.get_device_capability(device)
                
                # FP8 requires compute capability 8.9+ (Hopper)
                # But even on supported hardware, FP8 ops may not be fully implemented
                if device_cap[0] >= 9 or (device_cap[0] == 8 and device_cap[1] >= 9):
                    print("⚠️ FP8 selected but operations may be limited")
                    print("   Will attempt FP8 but fall back to bfloat16 if operations fail")
                    return torch.float8_e4m3fn
                else:
                    print(f"⚠️ FP8 not supported on device (compute {device_cap}), falling back to bfloat16")
            else:
                print("⚠️ FP8 not available in this PyTorch version, falling back to bfloat16")
            
            # Fallback
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
                print("🔄 FP8 operation failed, falling back to bfloat16")
                return torch.bfloat16
            else:
                print("🔄 FP8 operation failed, falling back to float16")
                return torch.float16
        elif current_dtype == torch.bfloat16:
            # bfloat16 failed, try float16
            print("🔄 bfloat16 operation failed, falling back to float16")
            return torch.float16
        else:
            # float16 failed, try float32
            print("🔄 float16 operation failed, falling back to float32")
            return torch.float32

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

# ==================== HELPER FUNCTIONS ====================

class ThreadSafeCleanup:
    """Thread-safe temporary file cleanup."""
    _lock = threading.Lock()
    
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
            
            # 3. Clean old cache files (older than 1 hour)
            cache_files = list(temp_dir.glob("*.cache"))
            current_time = time.time()
            for cache_file in cache_files:
                try:
                    if current_time - cache_file.stat().st_mtime > 3600:
                        cache_file.unlink()
                except Exception:
                    pass

def get_experiment_temp_path(node_type: str = "main") -> Path:
    """Get temp path with cleanup."""
    # Run cleanup in background thread
    import threading
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

def get_user_output_path(save_folder: str, filename: str) -> Path:
    """
    Get user output path with auto-increment.
    """
    # Default filename
    if not filename or filename.strip() == "":
        filename = "merged_lora"
    
    # Ensure .safetensors extension
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"
    
    # Clean filename
    safe_filename = ''.join(c for c in filename 
                           if c.isalnum() or c in '._- ').rstrip()
    
    # Get base output directory (default fallback)
    lora_folders = folder_paths.get_folder_paths("loras")
    default_base = Path(lora_folders[0]) if lora_folders else Path.cwd()
    
    # If user provided a save folder
    if save_folder and isinstance(save_folder, str) and save_folder.strip():
        user_path = save_folder.strip()
        
        print(f"📁 User requested save folder: '{user_path}'")
        
        # Check if it looks like an absolute Windows path
        is_windows_absolute = (len(user_path) >= 2 and 
                              user_path[1] == ':' and 
                              user_path[2] == '\\')
        
        # Check if it looks like a network path
        is_network_path = user_path.startswith('\\\\')
        
        if is_windows_absolute or is_network_path:
            # Absolute path - use it directly
            try:
                output_folder = Path(user_path)
                
                # Basic security: check for obvious path traversal
                if '..' in user_path or user_path.count(':\\') > 1:
                    print(f"⚠️ Suspicious path pattern, using default")
                    output_folder = default_base
                else:
                    # Create directory if it doesn't exist
                    output_folder.mkdir(parents=True, exist_ok=True)
                    print(f"✅ Using absolute path: {output_folder}")
            except Exception as e:
                print(f"⚠️ Could not use absolute path '{user_path}': {e}")
                output_folder = default_base
        else:
            # Relative path - treat as subfolder of default lora folder
            try:
                # Remove any leading/trailing slashes
                clean_path = user_path.strip('\\/')
                output_folder = default_base / clean_path
                output_folder.mkdir(parents=True, exist_ok=True)
                print(f"✅ Using relative path: {output_folder}")
            except Exception as e:
                print(f"⚠️ Could not create subfolder '{user_path}': {e}")
                output_folder = default_base
    else:
        # No custom folder specified
        output_folder = default_base
        print(f"📁 Using default lora folder: {output_folder}")
    
    # Create final path with auto-increment
    base_path = output_folder / safe_filename
    
    # Auto-increment logic
    counter = 1
    final_path = base_path
    while final_path.exists():
        stem = base_path.stem
        # Remove existing counter
        parts = stem.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            stem = '_'.join(parts[:-1])
        final_path = base_path.parent / f"{stem}_{counter}{base_path.suffix}"
        counter += 1
    
    print(f"💾 Final save path: {final_path}")
    return final_path

def silent_pad_or_truncate(tensor: torch.Tensor, target_rank: int, key: str) -> torch.Tensor:
    """Silent version that doesn't print."""
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = pad_or_truncate(tensor, target_rank, key)
    finally:
        sys.stdout = old_stdout
    return result

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

# ==================== MODEL VALIDATION ====================

class ModelValidator:
    """Validates LoRA model compatibility and integrity."""
    
    @staticmethod
    def detect_model_type(state_dict: Dict[str, torch.Tensor]) -> str:
        """Detect LoRA model architecture type."""
        keys = list(state_dict.keys())
        if not keys:
            return "unknown"
        
        # Check for specific patterns
        if any("diffusion_model.double_blocks" in k for k in keys):
            return "flux_klein"
        elif any("diffusion_model.single_blocks" in k for k in keys):
            return "flux_klein_4b"
        elif any("input_blocks" in k or "output_blocks" in k for k in keys):
            return "sd15_or_sdxl"
        elif any("transformer" in k for k in keys):
            return "transformer_based"
        elif any("lora_unet_" in k for k in keys):
            return "musubi_klein"
        
        return "unknown"
    
    @staticmethod
    def extract_layer_name(key: str) -> str:
        """Extract base layer name from LoRA key."""
        patterns = [
            r'(.*)\.lora_A\.weight$',
            r'(.*)\.lora_B\.weight$',
            r'(.*)\.lora\.down\.weight$',
            r'(.*)\.lora\.up\.weight$',
            r'(.*)\.alpha$',
            r'(.*)\.lora_down\.weight$',
            r'(.*)\.lora_up\.weight$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, key)
            if match:
                return match.group(1)
        return key
    
    @staticmethod
    def validate_compatibility(sd_a: Dict[str, torch.Tensor], 
                              sd_b: Dict[str, torch.Tensor]) -> bool:
        """
        Validate if two LoRAs are compatible for merging.
        
        Returns:
            True if compatible, raises ValueError if not
        """
        # Extract layer names (without lora_A/lora_B suffix)
        layers_a = {ModelValidator.extract_layer_name(k) for k in sd_a.keys()}
        layers_b = {ModelValidator.extract_layer_name(k) for k in sd_b.keys()}
        
        # Check model types
        type_a = ModelValidator.detect_model_type(sd_a)
        type_b = ModelValidator.detect_model_type(sd_b)
        
        if type_a != type_b:
            print(f"⚠️ Different model architectures: {type_a} vs {type_b}")
            print("   Attempting merge anyway, but results may be unstable")
        
        # Check layer overlap
        overlap = layers_a.intersection(layers_b)
        total_layers = min(len(layers_a), len(layers_b))
        
        if total_layers == 0:
            raise ValueError("No layers found in one or both models")
        
        overlap_ratio = len(overlap) / total_layers
        
        if overlap_ratio < 0.3:
            raise ValueError(
                f"Models have only {overlap_ratio:.1%} layer overlap "
                f"({len(overlap)}/{total_layers}) - likely incompatible"
            )
        
        if overlap_ratio < 0.7:
            print(f"⚠️ Low layer overlap: {overlap_ratio:.1%}")
            print("   Some layers may not be merged")
        
        # Check rank consistency
        ranks_a = ModelValidator.get_model_ranks(sd_a)
        ranks_b = ModelValidator.get_model_ranks(sd_b)
        
        if ranks_a and ranks_b:
            avg_rank_a = sum(ranks_a) / len(ranks_a)
            avg_rank_b = sum(ranks_b) / len(ranks_b)
            
            if abs(avg_rank_a - avg_rank_b) > max(avg_rank_a, avg_rank_b) * 0.5:
                print(f"⚠️ Significant rank difference: {avg_rank_a:.1f} vs {avg_rank_b:.1f}")
        
        return True
    
    @staticmethod
    def get_model_ranks(state_dict: Dict[str, torch.Tensor]) -> List[int]:
        """Get all unique ranks in a model."""
        ranks = set()
        for key, tensor in state_dict.items():
            if len(tensor.shape) >= 2:
                rank = safe_get_rank(tensor, key)
                ranks.add(rank)
        return list(ranks)

# ==================== METADATA HANDLING ====================

class MetadataMerger:
    """Handles merging of metadata from multiple LoRA files."""
    
    @staticmethod
    def merge(meta_a: Dict[str, str], meta_b: Dict[str, str], 
              mode: str = "merge_basic") -> Dict[str, str]:
        """
        Merge metadata from two LoRA files with conflict resolution.
        
        Args:
            meta_a: Metadata from first LoRA
            meta_b: Metadata from second LoRA
            mode: Merge strategy
            
        Returns:
            Merged metadata dictionary
        """
        if mode == "none":
            return {}
        elif mode == "preserve_a":
            return {f"lora_a_{k}": v for k, v in meta_a.items()}
        elif mode == "preserve_b":
            return {f"lora_b_{k}": v for k, v in meta_b.items()}
        
        # merge_basic mode
        merged = {}
        conflicts = []
        
        # Important fields to preserve separately
        important_fields = [
            "ss_base_model", "ss_sd_model_name", "ss_network_module",
            "ss_network_dim", "ss_network_alpha", "ss_training_started_at"
        ]
        
        # Merge with conflict detection
        all_keys = set(meta_a.keys()) | set(meta_b.keys())
        
        for key in all_keys:
            val_a = meta_a.get(key)
            val_b = meta_b.get(key)
            
            if val_a is not None and val_b is not None:
                if val_a != val_b:
                    conflicts.append((key, val_a, val_b))
                    
                    # Handle important fields specially
                    if key in important_fields:
                        merged[f"lora_a_{key}"] = val_a
                        merged[f"lora_b_{key}"] = val_b
                    else:
                        # Try to merge values
                        try:
                            # Numeric values: average
                            num_a = float(val_a)
                            num_b = float(val_b)
                            merged[f"{key}_avg"] = str((num_a + num_b) / 2)
                            merged[f"{key}_a"] = val_a
                            merged[f"{key}_b"] = val_b
                        except ValueError:
                            # String values: concatenate
                            merged[f"{key}_a"] = val_a
                            merged[f"{key}_b"] = val_b
                else:
                    merged[key] = val_a
            elif val_a is not None:
                if key in important_fields:
                    merged[f"lora_a_{key}"] = val_a
                else:
                    merged[f"{key}_a"] = val_a
            elif val_b is not None:
                if key in important_fields:
                    merged[f"lora_b_{key}"] = val_b
                else:
                    merged[f"{key}_b"] = val_b
        
        if conflicts:
            print(f"📝 Resolved {len(conflicts)} metadata conflicts")
        
        return merged
    
# ==================== SIMPLIFIED STREAMING MERGE ENGINE ====================

# In easy_lora_merger.py, update the StreamingMergeEngine class:

class StreamingMergeEngine:
    """Performs memory-efficient streaming merges."""
    
    def __init__(self, config: MergeConfig):
        self.config = config
        self.device = DeviceManager.get_device(config.device_type)
        self.dtype = DeviceManager.get_dtype(config.precision, self.device)
        
        # Optimize device settings
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if torch.cuda.get_device_capability(self.device)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"💻 Device: {self.device}, Precision: {self.dtype}")
    
    def _detect_converted_lora(self, sd, original_sd):
        """Detect if a LoRA is already converted (alpha baked in)"""
        # Method 1: Check for alpha keys in original
        has_alpha = any('.alpha' in k for k in original_sd.keys())
        
        # Method 2: Check key count ratio (unconverted have ~1.5x more keys)
        key_count = len(sd)
        
        # Unconverted Musubi patterns:
        # - 4B: ~240 keys unconverted, ~160 converted
        # - 9B: ~336 keys unconverted, ~224 converted
        # - Z-image: ~630 keys unconverted, ~420 converted
        
        # If it has alpha keys AND high key count, it's unconverted
        if has_alpha and key_count > 200:
            # Check if it's in the unconverted range
            if (220 <= key_count <= 260) or (320 <= key_count <= 350) or (600 <= key_count <= 650):
                print(f"   📊 Detected unconverted LoRA ({key_count} keys, has alphas)")
                return False
        
        # If no alpha keys, it's definitely converted
        if not has_alpha:
            print(f"   📊 Detected converted LoRA (no alpha keys)")
            return True
        
        # Default: assume converted if uncertain
        print(f"   📊 LoRA status uncertain ({key_count} keys, has_alphas={has_alpha}) - assuming converted")
        return True
    
    
    def merge_with_streaming(self, path_a: Path, path_b: Path, 
                            output_path: Path) -> Dict[str, torch.Tensor]:
        """
        Simplified merge function that works with all formats.
        Includes alpha scaling to ensure sliders work correctly.
        """
        print("🔍 Loading LoRAs with streaming...")
        
        # First, let's load and normalize both models fully
        print("🔄 Loading and normalizing models...")
        
        # Load both models completely (they should fit in RAM)
        with safe_open(path_a, framework="pt") as fa:
            sd_a = {k: fa.get_tensor(k) for k in fa.keys()}
        
        with safe_open(path_b, framework="pt") as fb:
            sd_b = {k: fb.get_tensor(k) for k in fb.keys()}
        
        # Store original state dicts for alpha lookups
        original_sd_a = dict(sd_a)  # Keep original for alpha keys
        original_sd_b = dict(sd_b)
        
        # Normalize using the existing universal_normalize function
        sd_a = universal_normalize(sd_a)
        sd_b = universal_normalize(sd_b)
        
        print(f"📊 Normalized A: {len(sd_a)} keys, B: {len(sd_b)} keys")
        
        # Detect Z-Image architectures
        def detect_zimage_architecture(sd, name):
            keys = list(sd.keys())
            key_count = len(sd)
            
            # Musubi-tuner (630 keys, has alphas)
            if key_count >= 600 and any('lora_unet_layers' in k for k in keys):
                return f"{name}: Z-Image (Musubi-tuner)"
            
            # AI Toolkit (480 keys, no alphas)  
            elif key_count >= 450 and any('adaLN_modulation' in k for k in keys):
                return f"{name}: Z-Image Base (AI Toolkit)"
            elif key_count >= 450:
                # Without adaLN, it might be Turbo
                return f"{name}: Z-Image Turbo (AI Toolkit)"
            
            elif any('double_blocks' in k for k in keys):
                return f"{name}: Flux/Klein"
            else:
                return f"{name}: Unknown"
        
        arch_a = detect_zimage_architecture(sd_a, "A")
        arch_b = detect_zimage_architecture(sd_b, "B")
        print(f"🏗️ {arch_a}")
        print(f"🏗️ {arch_b}")
 
        # --- ADD MODEL TYPE DETECTION ---
        def detect_model_type(sd):
            """Detect which base model this LoRA was trained for"""
            shapes = []
            # Check first 20 relevant keys
            for key in list(sd.keys())[:20]:
                if any(x in key for x in ['linear1', 'linear2', 'img_attn.qkv']):
                    tensor = sd[key]
                    if len(tensor.shape) >= 2:
                        shapes.append((tensor.shape[0], tensor.shape[1]))
            
            if not shapes:
                return "unknown"
            
            avg_in = sum(s[0] for s in shapes) / len(shapes)
            avg_out = sum(s[1] for s in shapes) / len(shapes)
            
            # Flux 9B standard
            if abs(avg_in - 4096) < 100 and abs(avg_out - 36864) < 1000:
                return "flux_9b_standard"
            # Flux 2 Dev (what you just saw)
            elif abs(avg_in - 6144) < 100 and abs(avg_out - 24576) < 1000:
                return "flux_2_dev"
            # Z-Image
            elif abs(avg_in - 3840) < 100 and abs(avg_out - 10240) < 1000:
                return "z_image"
            # SDXL range
            elif avg_in > 2000 and avg_in < 3000 and avg_out > 2000 and avg_out < 3000:
                return "sdxl"
            # flux_2_dev range
            elif abs(avg_in - 9232) < 100 and abs(avg_out - 3088) < 100:
                return "flux_2_dev"
            else:
                return f"unknown_{int(avg_in)}x{int(avg_out)}"
        
        model_a = detect_model_type(sd_a)
        model_b = detect_model_type(sd_b)
        print(f"📐 Model A: {model_a}")
        print(f"📐 Model B: {model_b}")
        
        if model_a != model_b and "unknown" not in model_a and "unknown" not in model_b:
            print(f"⚠️ WARNING: LoRAs trained for different models!")
            print(f"   A: {model_a}")
            print(f"   B: {model_b}")
            print(f"   Merge will work but may not load correctly in your model")
 
        # --- END OF BLOCK ---

        print("🔍 Detecting LoRA conversion status...")
        sd_a_converted = self._detect_converted_lora(sd_a, original_sd_a)
        sd_b_converted = self._detect_converted_lora(sd_b, original_sd_b)
        print(f"📊 LoRA A converted: {sd_a_converted}, LoRA B converted: {sd_b_converted}")

        # Find common keys
        keys_a = set(sd_a.keys())
        keys_b = set(sd_b.keys())
        common_keys = keys_a & keys_b
        
        # Calculate unique keys before using them
        unique_keys_a = keys_a - keys_b
        unique_keys_b = keys_b - keys_a
        
        print(f"🧩 Found {len(common_keys)} common layers to merge")
        print(f"📝 Unique to A: {len(unique_keys_a)}, Unique to B: {len(unique_keys_b)}")
        
        if len(common_keys) == 0:
            raise ValueError("No common layers found between the two LoRAs")
        
        # Process in batches
        merged_dict = {}
        keys_list = list(common_keys)
        batch_size = self.config.batch_size
        
        # Create progress bar
        try:
            pbar = comfy.utils.ProgressBar(len(keys_list))
        except:
            pbar = None
        
        print("⚙️ Merging layers...")
        
        # --- HELPER FUNCTION FOR ALPHA SCALING ---
        def apply_lora_scaling(tensor, original_sd, key, is_converted=False):
            """Apply alpha/rank scaling correction to LoRA tensors"""
            
            # For converted LoRAs (no alpha keys), don't scale
            if is_converted:
                return tensor
            
            # Store original stats for debug
            orig_mean = tensor.abs().mean().item()
            
            # Try multiple alpha key patterns
            alpha_key_candidates = [
                key.replace(".weight", ".alpha"),
                key.replace("lora_A.weight", "alpha"),
                key.replace("lora_B.weight", "alpha"),
                key.replace("lora_down.weight", "alpha"),
                key.replace("lora_up.weight", "alpha"),
                key.replace("diffusion_model.", "").replace(".lora_A.weight", ".alpha"),
                key.replace("diffusion_model.", "").replace(".lora_B.weight", ".alpha"),
            ]
            
            # Also try with the base layer name
            if '.lora_' in key:
                base_key = key.split('.lora_')[0]
                alpha_key_candidates.append(f"{base_key}.alpha")
                alpha_key_candidates.append(f"{base_key}_alpha")
            
            alpha_value = None
            alpha_key_used = None
            for alpha_key in alpha_key_candidates:
                if alpha_key in original_sd:
                    try:
                        alpha_tensor = original_sd[alpha_key]
                        if isinstance(alpha_tensor, torch.Tensor):
                            if alpha_tensor.numel() == 1:
                                alpha_value = alpha_tensor.item()
                            else:
                                alpha_value = alpha_tensor.mean().item()
                            alpha_key_used = alpha_key
                            break
                    except:
                        continue
            
            if alpha_value is not None:
                alpha_value = abs(alpha_value)
                
                if len(tensor.shape) >= 2:
                    rank = min(tensor.shape[0], tensor.shape[1])
                else:
                    rank = 1
                
                rank = max(1, rank)
                scale_factor = alpha_value / rank
                
                # Check if this is a converted LoRA (alpha should be baked in)
                # If scale_factor is very small, it might be incorrectly applied
                if scale_factor < 0.01 and alpha_value < 1.0:
                    print(f"⚠️ Very small scale factor {scale_factor:.6f} for {key}")
                    print(f"   alpha={alpha_value:.4f}, rank={rank}")
                    print(f"   This LoRA may already be converted - using raw tensor")
                    return tensor  # Return unscaled for converted LoRAs
                
                # For valid scaling factors, apply
                if scale_factor > 0.01:
                    return tensor * scale_factor
                else:
                    # Too small, probably shouldn't scale
                    return tensor
            
            # No alpha found - this is a converted LoRA
            return tensor
        
        
        for i in range(0, len(keys_list), batch_size):
            batch_keys = keys_list[i:i + batch_size]
            
            for key in batch_keys:
                try:
                    # Get base tensors
                    raw_a = sd_a[key].to(self.device).to(self.dtype)
                    raw_b = sd_b[key].to(self.device).to(self.dtype)
                    
                    # --- APPLY ALPHA SCALING FIRST (CRITICAL FOR SLIDER ACCURACY) ---
                    # This ensures both LoRAs are at the same "volume" before merging
                    t_a = apply_lora_scaling(raw_a, original_sd_a, key, is_converted=sd_a_converted)
                    t_b = apply_lora_scaling(raw_b, original_sd_b, key, is_converted=sd_b_converted)
                    
                    # --- THEN DO RANK ADJUSTMENT ---
                    rank_a = safe_get_rank(t_a, key)
                    rank_b = safe_get_rank(t_b, key)
                    target_rank = max(rank_a, rank_b)
                    
                    if rank_a != rank_b:
                        t_a = silent_pad_or_truncate(t_a, target_rank, key)
                        t_b = silent_pad_or_truncate(t_b, target_rank, key)
                    
                    # --- DEBUG INFO FOR FIRST FEW LAYERS ---
                    if i == 0 and len(batch_keys) > 0 and key in batch_keys[:2]:
                        # Choose a good sample key (prefer lora_A over lora_B)
                        sample_key = key
                        if 'lora_B' in key:
                            # Try to find corresponding A weight
                            a_key = key.replace('lora_B', 'lora_A')
                            if a_key in sd_a:
                                sample_key = a_key
                        
                        # Get stats on the SCALED tensors
                        a_stats = {
                            'mean': t_a.abs().mean().item(),
                            'std': t_a.std().item(),
                            'max': t_a.abs().max().item(),
                            'min': t_a.abs().min().item(),
                            'non_zero': (t_a != 0).float().mean().item() * 100
                        }
                        b_stats = {
                            'mean': t_b.abs().mean().item(),
                            'std': t_b.std().item(),
                            'max': t_b.abs().max().item(),
                            'min': t_b.abs().min().item(),
                            'non_zero': (t_b != 0).float().mean().item() * 100
                        }
                        
                        print(f"\n📊 Sample Layer: {sample_key}")
                        print(f"   LoRA A - mean: {a_stats['mean']:.6f}, max: {a_stats['max']:.6f}, non-zero: {a_stats['non_zero']:.1f}%")
                        print(f"   LoRA B - mean: {b_stats['mean']:.6f}, max: {b_stats['max']:.6f}, non-zero: {b_stats['non_zero']:.1f}%")
                        
                        # Show slider values
                        if self.config.method == "feature_mix":
                            print(f"   🔧 uniqueness={self.config.uniqueness}")
                        elif self.config.method == "subtract":
                            print(f"   🔧 threshold={self.config.threshold}")
                        elif self.config.method == "magnitude":
                            print(f"   🔧 blend={self.config.blend}")
                        
                        # Also show raw (unscaled) stats for comparison
                        if 'raw_a' in locals() and 'raw_b' in locals():
                            raw_a_mean = raw_a.abs().mean().item()
                            raw_b_mean = raw_b.abs().mean().item()
                            print(f"   Raw (unscaled) - A: {raw_a_mean:.6f}, B: {raw_b_mean:.6f}")
                            
                            # Calculate scaling effect
                            if raw_a_mean > 0:
                                scale_a = a_stats['mean'] / raw_a_mean
                                print(f"   Scaling factor A: {scale_a:.3f}")
                            if raw_b_mean > 0:
                                scale_b = b_stats['mean'] / raw_b_mean
                                print(f"   Scaling factor B: {scale_b:.3f}")
                    
                    # --- END DEBUG INFO ---
                    
                    # Get merge method
                    merge_func = MergeMethodRegistry.get_method(self.config.method)
                    
                    # Prepare arguments with SCALED tensors
                    method_args = {
                        'a': t_a, 'b': t_b, 
                        'wa': self.config.weight_a, 'wb': self.config.weight_b
                    }
                    
                    # Add method-specific parameters from self.config
                    if self.config.method == "subtract":
                        method_args['threshold'] = self.config.threshold
                    elif self.config.method == "magnitude":
                        method_args['blend'] = self.config.blend
                    elif self.config.method == "feature_mix":
                        method_args['uniqueness'] = self.config.uniqueness
                    elif self.config.method == "svd_preserve":
                        method_args['preserve_ratio'] = self.config.density
                    elif self.config.method == "noise_aware":
                        method_args['noise_threshold'] = 0.01 * self.config.density
                    elif self.config.method == "dare_rescale":
                        method_args['drop_rate'] = 1.0 - self.config.density
                    elif self.config.method == "ties_gentle":
                        method_args['agreement_threshold'] = 0.3
                    
                    # --- ADDITIONAL SCALING CORRECTION (if needed) ---
                    # This handles dimension mismatches
                    if t_a.shape != t_b.shape:
                        scale_correction = max(t_a.shape[-1], t_b.shape[-1]) / min(t_a.shape[-1], t_b.shape[-1])
                        if t_a.numel() < t_b.numel():
                            method_args['wa'] *= scale_correction
                        else:
                            method_args['wb'] *= scale_correction
                    
                    # Merge
                    # Extract weights properly
                    wa_final = method_args.pop('wa')
                    wb_final = method_args.pop('wb')
                    a_tensor = method_args.pop('a')
                    b_tensor = method_args.pop('b')
                    
                    # Use universal_merge_executor for safety with shape mismatches
                    from .methods import universal_merge_executor
                    
                    merged = universal_merge_executor(
                        merge_func, 
                        a_tensor, 
                        b_tensor, 
                        wa_final, 
                        wb_final, 
                        **method_args
                    )
                    
                    # Apply density if needed
                    if "dare" not in self.config.method and self.config.density < 1.0:
                        merged = apply_density(merged, self.config.density)


                    # Store result
                    merged_dict[key] = merged.cpu()
                    
                    # Cleanup
                    del raw_a, raw_b, t_a, t_b, merged
                    
                except Exception as e:
                    print(f"⚠️ Failed to merge {key}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Update progress
            if pbar:
                pbar.update(min(batch_size, len(keys_list) - i))
            
            # Clean GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Yield for UI
            if i % (batch_size * 4) == 0:
                try:
                    comfyui_yield()
                except:
                    pass
        
        # Add unique keys (with scaling applied)
        if unique_keys_a:
            print(f"📝 Adding {len(unique_keys_a)} unique keys from A")
            for key in unique_keys_a:
                try:
                    raw_tensor = sd_a[key].to(self.device).to(self.dtype)
                    # Apply scaling to unique keys too
                    tensor = apply_lora_scaling(raw_tensor, original_sd_a, key) * self.config.weight_a
                    if self.config.density < 1.0 and "dare" not in self.config.method:
                        tensor = apply_density(tensor, self.config.density)
                    merged_dict[key] = tensor.cpu()
                except Exception as e:
                    print(f"⚠️ Failed to add unique key {key}: {e}")
        
        if unique_keys_b:
            print(f"📝 Adding {len(unique_keys_b)} unique keys from B")
            for key in unique_keys_b:
                try:
                    raw_tensor = sd_b[key].to(self.device).to(self.dtype)
                    # Apply scaling to unique keys too
                    tensor = apply_lora_scaling(raw_tensor, original_sd_b, key) * self.config.weight_b
                    if self.config.density < 1.0 and "dare" not in self.config.method:
                        tensor = apply_density(tensor, self.config.density)
                    merged_dict[key] = tensor.cpu()
                except Exception as e:
                    print(f"⚠️ Failed to add unique key {key}: {e}")
        
        print(f"✅ Merged {len(merged_dict)} total keys")
        return merged_dict
        
        def _merge_single(self, t_a: torch.Tensor, t_b: torch.Tensor, key: str) -> torch.Tensor:
            """Merge single tensor pair."""
            t_a = t_a.to(self.device).to(self.dtype)
            t_b = t_b.to(self.device).to(self.dtype)
            
            # Rank adjustment
            rank_a = safe_get_rank(t_a, key)
            rank_b = safe_get_rank(t_b, key)
            target_rank = max(rank_a, rank_b)
            
            t_a = silent_pad_or_truncate(t_a, target_rank, key)
            t_b = silent_pad_or_truncate(t_b, target_rank, key)
            
            # Merge (simplified version)
            if self.config.method == "linear":
                merged = merge_linear(t_a, t_b, self.config.weight_a, self.config.weight_b)
            # ... other methods ...
            else:
                merged = merge_linear(t_a, t_b, self.config.weight_a, self.config.weight_b)
            
            if "dare" not in self.config.method and self.config.density < 1.0:
                merged = apply_density(merged, self.config.density)
            
            return merged

# ==================== CORE MERGE FUNCTION ====================

def merge_loras(path_a: Union[str, Path], path_b: Union[str, Path], 
                config: MergeConfig, output_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Core merge function with security, performance, and validation.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)
    output_path = Path(output_path)
    
    # Security: validate paths exist
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(f"LoRA file not found: {path_a if not path_a.exists() else path_b}")
    
    print("🔍 Loading LoRAs...")
    sd_a, meta_a = load_lora_with_metadata(path_a)
    sd_b, meta_b = load_lora_with_metadata(path_b)
    
    # Detect original formats
    format_a = detect_lora_format(sd_a)
    format_b = detect_lora_format(sd_b)
    print(f"📁 Original formats: A={format_a}, B={format_b}")
    
    # Determine target format based on inputs
    # If both use the same format, keep that format
    # If mixed, prefer lora_A/lora_B format as it's more common
    if format_a == format_b:
        target_format = format_a
    elif "lora_a_b" in format_a or "lora_a_b" in format_b:
        target_format = "lora_a_b"
    else:
        target_format = "lora_down_up"
    
    print(f"🎯 Target format for merge: {target_format}")
    
    # Validate model compatibility
    try:
        ModelValidator.validate_compatibility(sd_a, sd_b)
    except ValueError as e:
        print(f"⚠️ Model compatibility warning: {e}")
    
    # Special Flux model validation
    validate_flux_model_compatibility(sd_a, sd_b)
    
    # Normalize BOTH to the same intermediate format for merging
    print("🔄 Normalizing formats for merging...")
    sd_a_norm = universal_normalize(sd_a, metadata=meta_a)
    sd_b_norm = universal_normalize(sd_b, metadata=meta_b)
    
    # In your merge_loras function, after normalization:
    print("🔍 FIRST 10 KEYS AFTER NORMALIZATION:")
    for i, key in enumerate(list(sd_a_norm.keys())[:10]):
        print(f"  A{i}: {key}")
    for i, key in enumerate(list(sd_b_norm.keys())[:10]):
        print(f"  B{i}: {key}")
    
    # Merge metadata
    preserved_metadata = MetadataMerger.merge(meta_a, meta_b, config.metadata_mode)
    
    # Create merge engine and perform merge
    with memory_optimized_merge():
        engine = StreamingMergeEngine(config)
        merged_dict = engine.merge_with_streaming(path_a, path_b, output_path)
    
    # Now convert merged result to target format for saving
    print(f"🔄 Converting to target format: {target_format}")
    
    if target_format in ["flux_klein_lora_down_up", "pony_diffusion_lora_down_up", "z_image_lora_down_up"]:
        # Convert back to lora_down/lora_up format
        final_dict = {}
        for key, value in merged_dict.items():
            new_key = key
            if "lora_A" in key and "weight" in key:
                new_key = key.replace("lora_A", "lora_down")
            elif "lora_B" in key and "weight" in key:
                new_key = key.replace("lora_B", "lora_up")
            elif "alpha" in key:
                # Keep alpha key but update if it references lora_A
                new_key = key.replace("lora_A", "lora_down")
            final_dict[new_key] = value
    else:
        # Keep as lora_A/lora_B format
        final_dict = merged_dict
    
    # Finalize with the correct format
    if "lora_down_up" in target_format:
        # Need to ensure we're in lora_down/lora_up format for finalization
        final_dict = universal_finalize(final_dict)
    else:
        final_dict = universal_finalize(final_dict)
    
    # Create final metadata
    merge_metadata = {
        "lora_merge_tool": "EasyLoRAMerger",
        "easy_lora_merger_version": EASY_LORA_MERGER_VERSION,  # ADD THIS
        "easy_lora_merger_date": EASY_LORA_MERGER_DATE,        # ADD THIS
        "merge_method": config.method,
        "density": str(config.density),
        "lora_a_weight": str(config.weight_a),
        "lora_b_weight": str(config.weight_b),
        "merged_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata_mode": config.metadata_mode,
        "source_a": path_a.name,
        "source_b": path_b.name,
        "source_format_a": format_a,
        "source_format_b": format_b,
        "target_format": target_format,
        "total_keys": str(len(final_dict)),
    }
    
    if config.method in ["confidence_weighted", "ties_strict", "ties_gentle", "ties_consensus"]:
        merge_metadata["confidence_a"] = str(config.confidence_a)
        merge_metadata["confidence_b"] = str(config.confidence_b)
    
    if config.method == "layer_selective":
        merge_metadata["attn_weight"] = str(config.attn_weight)
        merge_metadata["mlp_weight"] = str(config.mlp_weight)
    
    final_metadata = {**preserved_metadata, **merge_metadata}
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(final_dict, str(output_path), metadata=final_metadata)
    
    # Verify the saved file
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"✅ Saved: {output_path.name} ({file_size:.1f} MB)")
    
    # Try to validate the saved file can be loaded
    try:
        test_load = load_file(str(output_path))
        print(f"🔍 Validation: Loaded {len(test_load)} keys from saved file")
        
        # Check key patterns in saved file
        lora_down_keys = [k for k in test_load.keys() if "lora_down" in k]
        lora_up_keys = [k for k in test_load.keys() if "lora_up" in k]
        lora_a_keys = [k for k in test_load.keys() if "lora_A" in k]
        lora_b_keys = [k for k in test_load.keys() if "lora_B" in k]
        
        print(f"🔑 Key patterns in saved file:")
        if lora_down_keys:
            print(f"   lora_down: {len(lora_down_keys)} keys")
        if lora_up_keys:
            print(f"   lora_up: {len(lora_up_keys)} keys")
        if lora_a_keys:
            print(f"   lora_A: {len(lora_a_keys)} keys")
        if lora_b_keys:
            print(f"   lora_B: {len(lora_b_keys)} keys")
        
    except Exception as e:
        print(f"⚠️ Saved file validation warning: {e}")
    
    return final_dict


# ==================== MAIN NODE ====================

class EasyLoRAmergerNode:
    """ComfyUI node for merging LoRAs with security and performance enhancements."""
    
    _CACHE = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        default_folder = ""
        lora_folders = folder_paths.get_folder_paths("loras")
        if lora_folders:
            default_folder = str(lora_folders[0])
        
        # Add tooltip about FP8 limitations
        precision_options = ["auto", "float32", "bfloat16", "float16", "fp8"]
        precision_tooltips = {
            "fp8": "⚠️ Experimental: FP8 operations may not be fully implemented. Will fall back to bfloat16 if needed."
        }
        
        # Method tooltips for better UX
        method_tooltips = {
            "linear": "Simple weighted average - good starting point",
            "ties_strict": "Keep only where signs agree - good for conflicting styles",
            "ties_gentle": "Apply TIES only for strong disagreements",
            "dare_lite": "Random dropout without rescaling - experimental",
            "dare_rescale": "Random dropout with rescaling - maintains magnitude",
            "subtract": "Subtract B from A - remove unwanted styles",
            "magnitude": "Keep larger magnitude from either LoRA - blend controls strictness",
            "feature_mix": "Preserve unique features from each LoRA - uniqueness controls preservation",
            "svd_preserve": "SVD-based rank reduction - preserves structure",
            "noise_aware": "Reduce small noise values before merging",
            "gradient_alignment": "Weight by directional similarity"
        }
        
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "method": (["linear", "ties_strict", "ties_gentle", "dare_lite", 
                           "dare_rescale", "subtract", "magnitude", "feature_mix",
                           "svd_preserve", "noise_aware", "gradient_alignment"], {
                    "default": "linear",
                    "tooltip": "Choose merging method"
                }),
                "density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05,
                                      "tooltip": "Keep top % of weights (1.0 = all)"}),
            },
            "optional": {
                "lora_a": (["None"] + loras,),
                "lora_b": (["None"] + loras,),
                "lora_data_a": ("LORA",),
                "lora_data_b": ("LORA",),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -5, "max": 5, "step": 0.05,
                                       "tooltip": "Strength of first LoRA"}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -5, "max": 5, "step": 0.05,
                                       "tooltip": "Strength of second LoRA"}),
                
                # NEW PARAMETERS FOR NEW METHODS
                "uniqueness": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                                        "tooltip": "For feature_mix: higher = preserve more unique features"}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "For subtract: minimum magnitude to subtract (0 = all)"}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "For magnitude: 0=strict selection, 1=equal blend"}),
                
                # SAVE OPTIONS
                "save_trigger": ("BOOLEAN", {"default": False,
                                             "tooltip": "Save permanently? (temp file if false)"}),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False,
                                          "tooltip": "Folder to save in (full path or subfolder)"}),
                "filename": ("STRING", {"default": "merged_lora", "multiline": False,
                                       "tooltip": "Filename (auto-increments if exists)"}),
                
                # DEVICE OPTIONS
                "device": (["auto", "cuda", "cpu"], {"default": "auto",
                           "tooltip": "Device to use for merging"}),
                "precision": (precision_options, {
                    "default": "auto",
                    "tooltip": precision_tooltips.get("fp8", "Select precision")
                }),
                
                # METADATA OPTIONS
                "metadata_mode": (["none", "preserve_a", "preserve_b", "merge_basic"], {
                    "default": "merge_basic",
                    "tooltip": "How to handle metadata from source LoRAs"
                }),
                
                # PERFORMANCE OPTIONS
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 8,
                                       "tooltip": "Tensors per batch (lower = less VRAM)"}),
                "streaming": ("BOOLEAN", {"default": True,
                                         "tooltip": "Stream tensors to save VRAM"}),
                # METHOD INFO DISPLAY (read-only)
                "method_info": ("STRING", {
                    "default": "Select a method to see details here...", 
                    "multiline": True, 
                    "dynamicPrompts": False
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "save_path")
    FUNCTION = "merge"
    CATEGORY = "LoRA"
    
    _CACHE = {}  # THIS LINE MUST EXIST AT CLASS LEVEL
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache implementation for ComfyUI change detection."""
        # Create cache key from relevant inputs
        cache_key = (
            kwargs.get('lora_a', 'None'),
            kwargs.get('lora_b', 'None'),
            kwargs.get('weight_a', 1.0),
            kwargs.get('weight_b', 1.0),
            kwargs.get('method', 'linear'),
            kwargs.get('density', 1.0),
            kwargs.get('save_trigger', False),
            kwargs.get('streaming', True),
        )
        
        # Generate hash
        key_str = str(cache_key).encode('utf-8')
        current_hash = hashlib.md5(key_str).hexdigest()
        
        # Check cache - use cls._CACHE (class attribute)
        if cache_key in cls._CACHE and cls._CACHE[cache_key] == current_hash:
            return cls._CACHE[cache_key]
        
        # Update cache
        cls._CACHE[cache_key] = current_hash
        return current_hash
    
    def merge(self, model, clip, method="linear", density=1.0, 
              lora_a="None", lora_b="None", lora_data_a=None, lora_data_b=None,
              weight_a=1.0, weight_b=1.0, 
              uniqueness=0.7, threshold=0.0, blend=0.5,
              save_trigger=False, save_folder="", filename="merged_lora",
              device="auto", precision="auto", metadata_mode="merge_basic",
              batch_size=32, streaming=True,
              method_info=None):
        
        print("\n" + "="*50)
        print("🧩 Easy LoRA Merger (Secure & Optimized)")
        print("="*50)
        
        # Create config from inputs
        config = MergeConfig.from_inputs(
            method=method,
            density=density,
            weight_a=weight_a,
            weight_b=weight_b,
            uniqueness=uniqueness,
            threshold=threshold,
            blend=blend,
            device_type=device,
            precision=precision,
            metadata_mode=metadata_mode,
            batch_size=batch_size,
            streaming=streaming
        )
        
        # Get input paths - LORA DATA INPUTS TAKE PRIORITY
        path_a = None
        path_b = None
        
        # Handle input A: LORA data takes priority over dropdown
        if lora_data_a is not None:
            path_a = save_lora_data_to_temp(lora_data_a, "A")
            if path_a:
                print(f"📄 A: LORA data (temp: {path_a.name})")
                # Show what would have been selected from dropdown for reference
                if lora_a != "None":
                    print(f"   (dropdown selection '{lora_a}' ignored due to LORA data input)")
        elif lora_a != "None" and lora_a:
            try:
                path_a = folder_paths.get_full_path("loras", lora_a)
                print(f"📄 A: {lora_a}")
            except Exception as e:
                print(f"❌ Failed to get path for {lora_a}: {e}")
                path_a = None
        else:
            print("📄 A: Not specified")
        
        # Handle input B: LORA data takes priority over dropdown
        if lora_data_b is not None:
            path_b = save_lora_data_to_temp(lora_data_b, "B")
            if path_b:
                print(f"📄 B: LORA data (temp: {path_b.name})")
                # Show what would have been selected from dropdown for reference
                if lora_b != "None":
                    print(f"   (dropdown selection '{lora_b}' ignored due to LORA data input)")
        elif lora_b != "None" and lora_b:
            try:
                path_b = folder_paths.get_full_path("loras", lora_b)
                print(f"📄 B: {lora_b}")
            except Exception as e:
                print(f"❌ Failed to get path for {lora_b}: {e}")
                path_b = None
        else:
            print("📄 B: Not specified")
        
        if not path_a or not path_b:
            print("❌ Need 2 valid LoRA inputs")
            return (model, clip, "")
        
        # Validate weight sanity
        total_weight = abs(weight_a) + abs(weight_b)
        if total_weight > 3.0:
            print(f"⚠️ High total weight ({total_weight}), results may be unstable")
        
        if weight_a * weight_b < 0 and "ties" not in method and "dare" not in method:
            print("⚠️ Negative weights with non-TIES method may cancel effects")
        
        # Get output path
        if save_trigger:
            try:
                output_path = get_user_output_path(save_folder, filename)
                print(f"💾 Will save to: {output_path}")
            except SecurityError as e:
                print(f"❌ Security error: {e}")
                return (model, clip, "")
        else:
            output_path = get_experiment_temp_path("main")
            print(f"📁 Temp file: {output_path.name}")
            print(f"💡 Location: {output_path.parent}")
        
        print(f"⚖️  Weights: {weight_a} : {weight_b}")
        print(f"🎯 Method: {method}, Density: {density}")
        print(f"🚀 Streaming: {streaming}, Batch size: {batch_size}")
        
        # Merge
        try:
            merged_dict = merge_loras(
                path_a, path_b, config, output_path
            )
            print(f"✅ Merged {len(merged_dict)} keys")
        except Exception as e:
            print(f"❌ Merge failed: {e}")
            import traceback
            traceback.print_exc()
            return (model, clip, "")
        
        # Load into model
        try:
            lora_data = comfy.utils.load_torch_file(str(output_path))
            
            # Check if this LoRA has any Text Encoder keys
            te_key_patterns = ["lora_te", "conditioner", "text_model", "transformer.text"]
            has_te_keys = any(any(pattern in k for pattern in te_key_patterns) for k in lora_data.keys())
            
            if has_te_keys:
                print("🔤 LoRA contains Text Encoder keys - applying to CLIP")
                model_lora, clip_lora = comfy.sd.load_lora_for_models(
                    model, clip, lora_data, 1.0, 1.0
                )
            else:
                print("ℹ️ No Text Encoder keys found - applying only to model")
                # Apply only to model, keep clip unchanged
                model_lora, _ = comfy.sd.load_lora_for_models(
                    model, clip, lora_data, 1.0, 1.0
                )
                clip_lora = clip  # Return original clip
                
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return (model, clip, "")
        
        # Final message
        if save_trigger:
            print(f"💾 Saved permanently to {output_path}")
        else:
            print(f"📁 Temp file will auto-cleanup (keeps last 10)")
        
        # SAFE cache clearing - only if cache exists
        if save_trigger:
            try:
                # Check if cache exists before clearing
                if hasattr(EasyLoRAmergerNode, '_CACHE'):
                    EasyLoRAmergerNode._CACHE.clear()
                elif hasattr(self, '_CACHE'):
                    self._CACHE.clear()
            except:
                pass  # Ignore cache errors - not critical
        
        return (model_lora, clip_lora, str(output_path))

# ==================== LORA-ONLY NODE ====================

class EasyLoRAonlyMerger:
    OUTPUT_NODE = True
    """Merge LoRAs without loading into model."""
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        default_folder = ""
        lora_folders = folder_paths.get_folder_paths("loras")
        if lora_folders:
            default_folder = str(lora_folders[0])
        
        # Add tooltip about FP8 limitations
        precision_options = ["auto", "float32", "bfloat16", "float16", "fp8"]
        precision_tooltips = {
            "fp8": "⚠️ Experimental: FP8 operations may not be fully implemented. Will fall back to bfloat16 if needed."
        }
        
        # Method tooltips for better UX
        method_tooltips = {
            "linear": "Simple weighted average - good starting point",
            "ties_strict": "Keep only where signs agree - good for conflicting styles",
            "ties_gentle": "Apply TIES only for strong disagreements",
            "dare_lite": "Random dropout without rescaling - experimental",
            "dare_rescale": "Random dropout with rescaling - maintains magnitude",
            "subtract": "Subtract B from A - remove unwanted styles",
            "magnitude": "Keep larger magnitude from either LoRA - blend controls strictness",
            "feature_mix": "Preserve unique features from each LoRA - uniqueness controls preservation",
            "svd_preserve": "SVD-based rank reduction - preserves structure",
            "noise_aware": "Reduce small noise values before merging",
            "gradient_alignment": "Weight by directional similarity"
        }
        
        return {
            "required": {
                "method": (["linear", "ties_strict", "ties_gentle", "dare_lite", 
                           "dare_rescale", "subtract", "magnitude", "feature_mix",
                           "svd_preserve", "noise_aware", "gradient_alignment"], {
                    "default": "linear",
                    "tooltip": "Choose merging method"
                }),
                "density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05,
                                      "tooltip": "Keep top % of weights (1.0 = all)"}),
            },
            "optional": {
                "lora_a": (["None"] + loras,),
                "lora_b": (["None"] + loras,),
                "lora_data_a": ("LORA",),
                "lora_data_b": ("LORA",),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -5, "max": 5, "step": 0.05,
                                       "tooltip": "Strength of first LoRA"}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -5, "max": 5, "step": 0.05,
                                       "tooltip": "Strength of second LoRA"}),
                
                # NEW PARAMETERS FOR NEW METHODS
                "uniqueness": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                                        "tooltip": "For feature_mix: higher = preserve more unique features"}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "For subtract: minimum magnitude to subtract (0 = all)"}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "For magnitude: 0=strict selection, 1=equal blend"}),
                
                # SAVE OPTIONS
                "save_trigger": ("BOOLEAN", {"default": False,
                                             "tooltip": "Save permanently? (temp file if false)"}),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False,
                                          "tooltip": "Folder to save in (full path or subfolder)"}),
                "filename": ("STRING", {"default": "merged_lora", "multiline": False,
                                       "tooltip": "Filename (auto-increments if exists)"}),
                
                # DEVICE OPTIONS
                "device": (["auto", "cuda", "cpu"], {"default": "auto",
                           "tooltip": "Device to use for merging"}),
                "precision": (precision_options, {
                    "default": "auto",
                    "tooltip": precision_tooltips.get("fp8", "Select precision")
                }),
                
                # METADATA OPTIONS
                "metadata_mode": (["none", "preserve_a", "preserve_b", "merge_basic"], {
                    "default": "merge_basic",
                    "tooltip": "How to handle metadata from source LoRAs"
                }),
                
                # PERFORMANCE OPTIONS
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 8,
                                       "tooltip": "Tensors per batch (lower = less VRAM)"}),
                "streaming": ("BOOLEAN", {"default": True,
                                         "tooltip": "Stream tensors to save VRAM"}),
                # METHOD INFO DISPLAY (read-only)
                "method_info": ("STRING", {
                    "default": "Select a method to see details here...", 
                    "multiline": True, 
                    "dynamicPrompts": False
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("LORA", "STRING")
    RETURN_NAMES = ("merged_lora", "save_path")
    FUNCTION = "merge_only"
    CATEGORY = "LoRA"
    
    
    def merge_only(self, method="linear", density=1.0,
                   lora_a="None", lora_b="None", lora_data_a=None, lora_data_b=None,
                   weight_a=1.0, weight_b=1.0, 
                   uniqueness=0.7, threshold=0.0, blend=0.5,
                   save_trigger=False, save_folder="", filename="merged_lora",
                   device="auto", precision="auto", metadata_mode="merge_basic",
                   batch_size=32, streaming=True,
                   method_info=None, node_id=None):
        
        print("\n" + "="*50)
        print("🧩 LoRA-Only Merger")
        print("="*50)
        
        # Create config - REMOVE the undefined variables
        config = MergeConfig.from_inputs(
            method=method,
            density=density,
            weight_a=weight_a,
            weight_b=weight_b,
            uniqueness=uniqueness,
            threshold=threshold,
            blend=blend,
            # REMOVED: attn_weight, mlp_weight, confidence_a, confidence_b
            device_type=device,
            precision=precision,
            metadata_mode=metadata_mode,
            batch_size=batch_size,
            streaming=streaming
        )
        
        # Get input paths - LORA DATA INPUTS TAKE PRIORITY
        path_a = None
        path_b = None
        
        # Handle input A: LORA data takes priority over dropdown
        if lora_data_a is not None:
            path_a = save_lora_data_to_temp(lora_data_a, "A")
            if path_a:
                print(f"📄 A: LORA data (temp: {path_a.name})")
                if lora_a != "None":
                    print(f"   (dropdown selection '{lora_a}' ignored due to LORA data input)")
        elif lora_a != "None" and lora_a:
            try:
                path_a = folder_paths.get_full_path("loras", lora_a)
                print(f"📄 A: {lora_a}")
            except Exception as e:
                print(f"❌ Failed to get path for {lora_a}: {e}")
                path_a = None
        else:
            print("📄 A: Not specified")
        
        # Handle input B: LORA data takes priority over dropdown
        if lora_data_b is not None:
            path_b = save_lora_data_to_temp(lora_data_b, "B")
            if path_b:
                print(f"📄 B: LORA data (temp: {path_b.name})")
                if lora_b != "None":
                    print(f"   (dropdown selection '{lora_b}' ignored due to LORA data input)")
        elif lora_b != "None" and lora_b:
            try:
                path_b = folder_paths.get_full_path("loras", lora_b)
                print(f"📄 B: {lora_b}")
            except Exception as e:
                print(f"❌ Failed to get path for {lora_b}: {e}")
                path_b = None
        else:
            print("📄 B: Not specified")
        
        if not path_a or not path_b:
            print("❌ Need 2 valid LoRA inputs")
            return (None, "")
        
        # Output path
        if save_trigger:
            try:
                output_path = get_user_output_path(save_folder, filename)
                print(f"💾 Will save to: {output_path}")
            except SecurityError as e:
                print(f"❌ Security error: {e}")
                return (None, "")
        else:
            output_path = get_experiment_temp_path("only")
            print(f"📁 Temp file: {output_path.name}")
            print(f"💡 Location: {output_path.parent}")
        
        print(f"⚖️  Weights: {weight_a} : {weight_b}")
        print(f"🎯 Method: {method}, Density: {density}")
        print(f"🚀 Streaming: {streaming}, Batch size: {batch_size}")
        
        # Merge
        try:
            merged_dict = merge_loras(
                path_a, path_b, config, output_path
            )
            print(f"✅ Merged {len(merged_dict)} keys")
        except Exception as e:
            print(f"❌ Merge failed: {e}")
            import traceback
            traceback.print_exc()
            return (None, "")
        
        # Load as LORA data
        try:
            lora_data = comfy.utils.load_torch_file(str(output_path))
            lora_tuple = (lora_data, 1.0, 1.0)
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return (None, "")
        
        # Final message
        if save_trigger:
            print(f"💾 Saved permanently to {output_path}")
        else:
            print(f"📁 Temp file will auto-cleanup (keeps last 10)")
        
        print("="*50)
        print("🎉 Done!")
        print("="*50)
        
        return (lora_tuple, str(output_path))

# ==================== New Triple Merger Node ====================

class EasyLoRATripleMerger:
    """Merge THREE LoRAs at once and apply to model (experimental)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        default_folder = ""
        lora_folders = folder_paths.get_folder_paths("loras")
        if lora_folders:
            default_folder = str(lora_folders[0])
        
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "method": (["linear", "ties_strict", "feature_mix", "magnitude", "subtract", "dare_rescale"], 
                          {"default": "linear"}),
                "density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "lora_a": (["None"] + loras,),
                "lora_b": (["None"] + loras,),
                "lora_c": (["None"] + loras,),
                "lora_data_a": ("LORA",),
                "lora_data_b": ("LORA",),
                "lora_data_c": ("LORA",),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "weight_c": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "uniqueness": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                                        "tooltip": "For feature_mix: higher = preserve more unique features"}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "For subtract: minimum magnitude to subtract"}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "For magnitude: 0=strict, 1=blended"}),
                "save_trigger": ("BOOLEAN", {"default": False}),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False}),
                "filename": ("STRING", {"default": "triple_merged", "multiline": False}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["auto", "float32", "bfloat16", "float16"], {"default": "auto"}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 8}),
                "method_info": ("STRING", {
                    "default": "Select a method to see details...", 
                    "multiline": True, 
                    "dynamicPrompts": False
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "LORA")
    RETURN_NAMES = ("model", "clip", "save_path", "lora_data")
    FUNCTION = "merge_triple"
    CATEGORY = "LoRA/Experimental"
    
    def merge_triple(self, model, clip, method="linear", density=1.0,
                    lora_a="None", lora_b="None", lora_c="None",
                    lora_data_a=None, lora_data_b=None, lora_data_c=None,
                    weight_a=1.0, weight_b=1.0, weight_c=1.0,
                    uniqueness=0.7, threshold=0.0, blend=0.5,
                    save_trigger=False, save_folder="", filename="triple_merged",
                    device="auto", precision="auto", batch_size=32,
                    method_info=None):
        
        print("\n" + "="*50)
        print("🎨 Easy LoRA Triple Merger (Experimental)")
        print("="*50)
        
        # Create config with only the variables we have
        config = MergeConfig(
            method=method,
            density=density,
            weight_a=weight_a,
            weight_b=weight_b,
            uniqueness=uniqueness,
            threshold=threshold,
            blend=blend,
            device_type=device,
            precision=precision,
            batch_size=batch_size,
            streaming=True
        )
        
        # Get paths for all three LoRAs
        paths = []
        loras_data = [lora_data_a, lora_data_b, lora_data_c]
        loras_dropdown = [lora_a, lora_b, lora_c]
        names = ["A", "B", "C"]
        
        for i, (data, dropdown, name) in enumerate(zip(loras_data, loras_dropdown, names)):
            if data is not None:
                path = save_lora_data_to_temp(data, name)
                print(f"📄 {name}: LORA data (temp)")
                paths.append(path)
            elif dropdown != "None" and dropdown:
                path = folder_paths.get_full_path("loras", dropdown)
                print(f"📄 {name}: {dropdown}")
                paths.append(path)
            else:
                print(f"❌ {name}: Missing input")
                return (model, clip, "", None)
        
        # Output path logic
        if save_trigger:
            output_path = get_user_output_path(save_folder, filename)
        else:
            output_path = get_experiment_temp_path("triple")
        
        # Load all three
        print("📥 Loading LoRAs...")
        sds = []
        metas = []
        for path in paths:
            sd, meta = load_lora_with_metadata(Path(path))
            sds.append(sd)
            metas.append(meta)
        
        # Check if they're all the same type
        print("🔍 Detecting formats...")
        formats = [detect_lora_format(sd) for sd in sds]
        print(f"   Formats: {formats}")
        
        # Normalize all to common format
        print("🔄 Normalizing...")
        normalized_sds = [universal_normalize(sd) for sd in sds]
        
        # Merge using triple method
        print(f"🎯 Method: {method}")
        merged_dict = self._merge_triple_method(
            normalized_sds, 
            [weight_a, weight_b, weight_c],
            method, density, uniqueness, threshold, blend
        )
        
        # Finalize and save
        final_dict = universal_finalize(merged_dict)
        save_file(final_dict, str(output_path))
        
        print(f"✅ Saved: {output_path.name}")
        
        # Load into model
        try:
            lora_data = comfy.utils.load_torch_file(str(output_path))
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora_data, 1.0, 1.0
            )
            lora_tuple = (lora_data, 1.0, 1.0)
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return (model, clip, "", None)
        
        print("="*50)
        print("🎉 Triple Merge Complete!")
        print("="*50)
        
        return (model_lora, clip_lora, str(output_path), lora_tuple)
    
    def _merge_triple_method(self, sds, weights, method, density, uniqueness, threshold, blend):
        """Core triple merge logic with rank adjustment"""
        all_keys = set()
        for sd in sds:
            all_keys.update(sd.keys())
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merged = {}
        
        # Import rank utilities
        from .klein_normalizer import safe_get_rank, pad_or_truncate
        
        for key in all_keys:
            tensors = []
            valid_weights = []
            
            # Collect tensors that exist in each SD
            for i, sd in enumerate(sds):
                if key in sd:
                    tensors.append(sd[key].to(device).to(torch.bfloat16))
                    valid_weights.append(weights[i])
            
            if len(tensors) < 2:
                if tensors:
                    merged[key] = (tensors[0] * valid_weights[0]).cpu()
                continue
            
            # --- RANK ADJUSTMENT (FIX THE DIMENSION MISMATCH) ---
            # Get ranks for all tensors
            ranks = [safe_get_rank(t, key) for t in tensors]
            target_rank = max(ranks)
            
            # Pad/truncate all tensors to same rank
            adjusted_tensors = []
            for i, t in enumerate(tensors):
                if ranks[i] != target_rank:
                    t = pad_or_truncate(t, target_rank, f"{key}_{i}")
                adjusted_tensors.append(t)
            
            tensors = adjusted_tensors
            
            # Apply method with adjusted tensors
            if method == "linear":
                result = sum(t * w for t, w in zip(tensors, valid_weights))
            
            elif method == "ties_strict":
                signs = [torch.sign(t * w) for t, w in zip(tensors, valid_weights)]
                agreement = torch.stack(signs).float().mean(dim=0).abs() > 0.5
                result = sum(t * w for t, w in zip(tensors, valid_weights)) * agreement
            
            elif method == "ties_gentle":
                # Simple version for triple
                signs = [torch.sign(t * w) for t, w in zip(tensors, valid_weights)]
                agreement = torch.stack(signs).float().mean(dim=0).abs() > 0.3
                result = sum(t * w for t, w in zip(tensors, valid_weights)) * agreement
            
            elif method == "feature_mix":
                result = tensors[0] * valid_weights[0]
                for i in range(1, len(tensors)):
                    current = tensors[i] * valid_weights[i]
                    ratio = current.abs() / (result.abs() + 1e-8)
                    unique_mask = (ratio > (1/uniqueness)).float()
                    result = result * (1 - unique_mask) + current * unique_mask
            
            elif method == "magnitude":
                stacked = torch.stack([t * w for t, w in zip(tensors, valid_weights)])
                magnitudes = stacked.abs()
                winner = magnitudes.argmax(dim=0)
                selected = torch.gather(stacked, 0, winner.unsqueeze(0)).squeeze(0)
                averaged = stacked.mean(dim=0)
                result = selected * blend + averaged * (1 - blend)
            
            elif method == "subtract":
                result = tensors[0] * valid_weights[0]
                for i in range(1, len(tensors)):
                    current = tensors[i] * valid_weights[i]
                    if threshold > 0:
                        result_magnitude = result.abs()
                        current_magnitude = current.abs()
                        significant = current_magnitude > (threshold * result_magnitude + 1e-8)
                        result = result - current * significant.float()
                    else:
                        result = result - current
            
            elif method == "dare_rescale":
                drop_rate = 1.0 - density
                results = []
                for t, w in zip(tensors, valid_weights):
                    mask = (torch.rand_like(t) > drop_rate).float()
                    rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
                    results.append(t * mask * rescale * w)
                result = sum(results)
            
            else:  # Default to linear
                result = sum(t * w for t, w in zip(tensors, valid_weights))
            
            # Apply density if needed
            if density < 1.0 and method not in ["dare_rescale"]:
                flat = result.abs().flatten()
                k = max(1, int(flat.numel() * density))
                threshold_val = torch.topk(flat, k).values.min()
                mask = result.abs() >= threshold_val
                result = result * mask
            
            merged[key] = result.cpu()
            
            # Cleanup
            del tensors
        
        return merged


# ==================== MUSUBI LORA CONVERSION NODE ====================

class MusubiLoraConverter:
    OUTPUT_NODE = True
    """
    Dedicated node for converting Musubi Tuner LoRAs to standard format.
    Supports: Z-Image (base/turbo), Klein 4B, Klein 9B
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        default_folder = ""
        lora_folders = folder_paths.get_folder_paths("loras")
        if lora_folders:
            default_folder = str(lora_folders[0])
        
        return {
            "required": {
                "mode": (["auto_detect", "z_image", "klein_4b", "klein_9b"], 
                        {"default": "auto_detect"}),
                "save_trigger": ("BOOLEAN", {"default": False}),
                "filename": ("STRING", {"default": "converted_lora", "multiline": False}),
            },
            "optional": {
                "lora": (["None"] + loras,),
                "lora_data": ("LORA",),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False}),
                "keep_alphas": ("BOOLEAN", {"default": False, 
                                           "tooltip": "Keep alpha keys (usually not needed after conversion)"}),
                "debug": ("BOOLEAN", {"default": False, "tooltip": "Show detailed conversion info"}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("LORA", "STRING", "STRING")
    RETURN_NAMES = ("converted_lora", "save_path", "conversion_info")
    FUNCTION = "convert"
    CATEGORY = "LoRA/Musubi"
    
    def convert(self, mode="auto_detect", save_trigger=False, filename="converted_lora",
                lora="None", lora_data=None, save_folder="", keep_alphas=False, debug=False, node_id=None):
        
        print("\n" + "="*50)
        print("🔄 Musubi LoRA Converter")
        print("="*50)
        
        # Get input path
        if lora_data is not None:
            path = save_lora_data_to_temp(lora_data, "musubi")
            source_type = "LORA data input"
        elif lora != "None" and lora:
            path = folder_paths.get_full_path("loras", lora)
            source_type = f"dropdown: {lora}"
        else:
            print("❌ No LoRA input provided")
            return (None, "", "No input")
        
        print(f"📄 Source: {source_type}")
        print(f"📁 Path: {path}")
        
        # Load the LoRA
        sd, metadata = load_lora_with_metadata(Path(path))
        
        # Detect actual format
        detected_format = detect_lora_format(sd)
        print(f"🔍 Detected format: {detected_format}")
        
        # Analyze structure
        info = self.analyze_lora_structure(sd)
        if debug:
            print("\n📊 Structure analysis:")
            print(f"   Total keys: {info['key_count']}")
            print(f"   Has alphas: {info['has_alphas']}")
            print(f"   Hidden dims: {info['hidden_dims'][:5]}")
            print(f"   Structure: {info['structure'][:5]}")
        
        # Determine if conversion is needed
        needs_conversion = any([
            "musubi" in detected_format.lower() and "needs_conversion" in detected_format,
            info['has_alphas'] and any('lora_unet_' in k for k in sd.keys()),
            mode != "auto_detect"  # Manual override
        ])
        
        if not needs_conversion and mode == "auto_detect":
            print("✅ LoRA already in standard format, no conversion needed")
            # Still return as LORA data
            lora_tuple = (sd, 1.0, 1.0)
            return (lora_tuple, "", "Already in standard format")
        
        # Determine conversion type
        if mode != "auto_detect":
            conversion_type = mode
        elif "z_image" in detected_format or any('lora_unet_layers' in k for k in sd.keys()):
            conversion_type = "z_image"
        elif "klein" in detected_format or any('lora_unet_double_blocks' in k for k in sd.keys()):
            # Check if it's 9B by looking at hidden dimensions
            if 4096 in info['hidden_dims']:
                conversion_type = "klein_9b"
            else:
                conversion_type = "klein_4b"
        else:
            conversion_type = "z_image"  # Default fallback
        
        print(f"🔄 Converting as: {conversion_type}")
        
        # Perform conversion
        if conversion_type == "z_image":
            converted_sd = self.convert_zimage(sd, keep_alphas)
        elif conversion_type == "klein_4b":
            converted_sd = self.convert_klein_4b(sd, keep_alphas)
        elif conversion_type == "klein_9b":
            converted_sd = self.convert_klein_9b(sd, keep_alphas)
        else:
            converted_sd = self.convert_zimage(sd, keep_alphas)
        
        print(f"✅ Converted: {info['key_count']} → {len(converted_sd)} keys")
        
        # Save if requested
        save_path = ""
        if save_trigger:
            try:
                if save_folder:
                    output_folder = Path(save_folder)
                else:
                    lora_folders = folder_paths.get_folder_paths("loras")
                    output_folder = Path(lora_folders[0]) if lora_folders else Path.cwd()
                
                output_folder.mkdir(parents=True, exist_ok=True)
                
                # Add suffix based on conversion
                base_name = filename.replace('.safetensors', '')
                output_path = output_folder / f"{base_name}_converted.safetensors"
                
                # Auto-increment
                counter = 1
                while output_path.exists():
                    output_path = output_folder / f"{base_name}_converted_{counter}.safetensors"
                    counter += 1
                
                # Save
                save_file(converted_sd, str(output_path))
                save_path = str(output_path)
                print(f"💾 Saved to: {save_path}")
                
            except Exception as e:
                print(f"❌ Save failed: {e}")
        
        # Create LORA tuple for output
        lora_tuple = (converted_sd, 1.0, 1.0)
        
        # Create info string
        info_str = (f"Converted {conversion_type}: {info['key_count']} → {len(converted_sd)} keys, "
                   f"Alphas: {'kept' if keep_alphas else 'baked'}")
        
        print("="*50)
        return (lora_tuple, save_path, info_str)
    
    def analyze_lora_structure(self, sd):
        """Analyze LoRA structure to determine conversion needs."""
        keys = list(sd.keys())
        
        # Get hidden dimensions
        hidden_dims = set()
        for key, tensor in sd.items():
            if len(tensor.shape) >= 2:
                hidden_dims.add(tensor.shape[0])
                hidden_dims.add(tensor.shape[1])
        
        # Detect structure types
        structure = []
        if any('double_blocks' in k for k in keys):
            structure.append('flux_double')
        if any('single_blocks' in k for k in keys):
            structure.append('flux_single')
        if any('lora_unet_layers' in k for k in keys):
            structure.append('z_image_style')
        
        # Get unique prefixes
        prefixes = set()
        for key in keys[:10]:  # Sample first 10
            parts = key.split('.')
            if parts:
                prefixes.add(parts[0])
        
        return {
            'key_count': len(keys),
            'hidden_dims': sorted(list(hidden_dims))[:10],
            'structure': structure,
            'has_alphas': any('.alpha' in k for k in keys),
            'prefixes': list(prefixes)[:5]
        }
    
    def convert_zimage(self, sd, keep_alphas=False):
        """Convert Z-Image (base/turbo) format to standard."""
        converted = {}
        
        # Pattern: lora_unet_layers_X_attention_to_k.lora_down.weight
        # Target: diffusion_model.layers.X.attention.to_k.lora_A.weight
        
        for key, tensor in sd.items():
            new_key = key
            
            # Skip alpha keys if we're baking them
            if '.alpha' in key and not keep_alphas:
                continue
            
            # Convert prefix
            new_key = new_key.replace('lora_unet_', 'diffusion_model.')
            
            # Convert layer pattern: layers_X_ → layers.X.
            new_key = re.sub(r'layers_(\d+)_', r'layers.\1.', new_key)
            
            # Convert attention: attention_to_k → attention.to_k
            new_key = re.sub(r'attention_to_([kqv])', r'attention.to_\1', new_key)
            new_key = new_key.replace('attention_to_out_0', 'attention.to_out.0')
            
            # Convert feed forward: feed_forward_w1 → feed_forward.w1
            new_key = re.sub(r'feed_forward_w(\d)', r'feed_forward.w\1', new_key)
            
            # Convert lora naming
            if 'lora_down' in new_key:
                new_key = new_key.replace('lora_down.weight', 'lora_A.weight')
            elif 'lora_up' in new_key:
                new_key = new_key.replace('lora_up.weight', 'lora_B.weight')
            
            # Clean up double dots
            while '..' in new_key:
                new_key = new_key.replace('..', '.')
            
            # If this is an alpha key and we're not keeping it, bake it
            if '.alpha' in key and not keep_alphas:
                # Find corresponding weight key
                base_key = key.replace('.alpha', '')
                weight_key = None
                
                # Try both lora_down and lora_up
                for suffix in ['lora_down.weight', 'lora_up.weight']:
                    candidate = f"{base_key}.{suffix}"
                    if candidate in sd:
                        weight_key = candidate
                        break
                
                if weight_key and weight_key in sd:
                    # Alpha will be baked during merge, so we skip it
                    continue
            else:
                converted[new_key] = tensor
        
        return converted
    
    def convert_klein_4b(self, sd, keep_alphas=False):
        """Convert Klein 4B format to standard."""
        converted = {}
        
        # Pattern: lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight
        # Target: diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight
        
        for key, tensor in sd.items():
            new_key = key
            
            # Skip alpha keys if we're baking them
            if '.alpha' in key and not keep_alphas:
                continue
            
            # Convert prefix
            new_key = new_key.replace('lora_unet_', 'diffusion_model.')
            
            # Convert double/single blocks
            new_key = re.sub(r'(double|single)_blocks_(\d+)_', r'\1_blocks.\2.', new_key)
            
            # Convert attention: img_attn_qkv → img_attn.qkv
            new_key = re.sub(r'(img|txt)_attn_(proj|qkv)', r'\1_attn.\2', new_key)
            
            # Convert MLP: img_mlp_0 → img_mlp.0
            new_key = re.sub(r'(img|txt)_mlp_(\d+)', r'\1_mlp.\2', new_key)
            
            # Convert lora naming
            if 'lora_down' in new_key:
                new_key = new_key.replace('lora_down.weight', 'lora_A.weight')
            elif 'lora_up' in new_key:
                new_key = new_key.replace('lora_up.weight', 'lora_B.weight')
            
            # Clean up double dots
            while '..' in new_key:
                new_key = new_key.replace('..', '.')
            
            # If this is an alpha key and we're not keeping it, bake it
            if '.alpha' in key and not keep_alphas:
                # Find corresponding weight key
                base_key = key.replace('.alpha', '')
                weight_key = None
                
                for suffix in ['lora_down.weight', 'lora_up.weight']:
                    candidate = f"{base_key}.{suffix}"
                    if candidate in sd:
                        weight_key = candidate
                        break
                
                if weight_key and weight_key in sd:
                    # Alpha will be baked during merge
                    continue
            else:
                converted[new_key] = tensor
        
        return converted
    
    def convert_klein_9b(self, sd, keep_alphas=False):
        """Convert Klein 9B format to standard."""
        # 9B has larger hidden dimensions (4096 vs 3072)
        converted = {}
        
        # Same patterns as 4B, just different dimensions
        for key, tensor in sd.items():
            new_key = key
            
            if '.alpha' in key and not keep_alphas:
                continue
            
            new_key = new_key.replace('lora_unet_', 'diffusion_model.')
            new_key = re.sub(r'(double|single)_blocks_(\d+)_', r'\1_blocks.\2.', new_key)
            new_key = re.sub(r'(img|txt)_attn_(proj|qkv)', r'\1_attn.\2', new_key)
            new_key = re.sub(r'(img|txt)_mlp_(\d+)', r'\1_mlp.\2', new_key)
            
            if 'lora_down' in new_key:
                new_key = new_key.replace('lora_down.weight', 'lora_A.weight')
            elif 'lora_up' in new_key:
                new_key = new_key.replace('lora_up.weight', 'lora_B.weight')
            
            while '..' in new_key:
                new_key = new_key.replace('..', '.')
            
            if '.alpha' in key and not keep_alphas:
                base_key = key.replace('.alpha', '')
                weight_key = None
                
                for suffix in ['lora_down.weight', 'lora_up.weight']:
                    candidate = f"{base_key}.{suffix}"
                    if candidate in sd:
                        weight_key = candidate
                        break
                
                if weight_key and weight_key in sd:
                    continue
            else:
                converted[new_key] = tensor
        
        return converted

# ==================== Z-IMAGE NORMALIZER NODE ====================

class ZImageNormalizer:
    OUTPUT_NODE = True
    """Simple Z-Image LoRA normalizer"""
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        lora_folders = folder_paths.get_folder_paths("loras")
        default_folder = str(lora_folders[0]) if lora_folders else ""
        
        return {
            "required": {
                "mode": (["standardize_only", "detect_only"], 
                        {"default": "standardize_only"}),
                "target_weight": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 3.0, "step": 0.1,
                                           "tooltip": "Recommended weight (user decides)"}),
            },
            "optional": {
                "lora": (["None"] + loras,),
                "lora_data": ("LORA",),
                "save_trigger": ("BOOLEAN", {"default": False}),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False}),
                "filename": ("STRING", {"default": "normalized_zimage", "multiline": False}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("LORA", "STRING", "STRING")
    RETURN_NAMES = ("normalized_lora", "info", "recommended_weight")
    FUNCTION = "normalize"
    CATEGORY = "LoRA/Z-Image"
    
    def normalize(self, mode="standardize_only", target_weight=1.5,
                  lora="None", lora_data=None, save_trigger=False, 
                  save_folder="", filename="normalized_zimage", node_id=None):
        
        print("\n" + "="*50)
        print("🔄 Z-Image Normalizer")
        print("="*50)
        
        # Load LoRA
        from pathlib import Path
        import folder_paths
        from safetensors.torch import load_file, save_file
        
        if lora_data is not None:
            sd, _ = lora_data
            source = "LORA data"
            source_name = "data_input"
        elif lora != "None" and lora:
            path = folder_paths.get_full_path("loras", lora)
            sd = load_file(str(path))
            source = f"file: {lora}"
            source_name = Path(lora).stem
        else:
            print("❌ No LoRA input")
            return (None, "No input", "1.0")
        
        print(f"📄 Source: {source}")
        print(f"📊 Keys: {len(sd)}")
        
        # Detect trainer
        info = []
        recommended = "1.0"
        
        if any('lora_unet_layers' in k for k in sd.keys()):
            info.append("Musubi-tuner format")
            if len(sd) > 600:
                info.append("unconverted (has alphas)")
            else:
                info.append("converted")
        elif any('adaLN_modulation' in k for k in sd.keys()):
            info.append("AI Toolkit format (likely Base)")
            recommended = str(target_weight)
        elif any('attention.to' in k for k in sd.keys()):
            info.append("AI Toolkit format (likely Turbo)")
        else:
            info.append("Unknown format")
        
        # Apply standardization if requested
        if mode == "standardize_only":
            # Just ensure consistent naming (no conversion)
            new_sd = {}
            for key, tensor in sd.items():
                # Simple cleanup - remove double dots if any
                new_key = key.replace('..', '.')
                new_sd[new_key] = tensor
            sd = new_sd
            info.append("standardized")
        
        info_str = ", ".join(info)
        print(f"🔍 Detection: {info_str}")
        print(f"💡 Recommended weight: {recommended}")
        
        # Save if requested
        save_path = ""
        if save_trigger:
            try:
                # Handle save folder
                if save_folder and save_folder.strip():
                    output_folder = Path(save_folder)
                else:
                    lora_folders = folder_paths.get_folder_paths("loras")
                    output_folder = Path(lora_folders[0]) if lora_folders else Path.cwd()
                
                output_folder.mkdir(parents=True, exist_ok=True)
                
                # Create filename
                base_name = filename if filename else source_name
                if not base_name.endswith('.safetensors'):
                    base_name += '.safetensors'
                
                output_path = output_folder / base_name
                
                # Auto-increment
                counter = 1
                while output_path.exists():
                    stem = output_path.stem
                    if '_' in stem and stem.split('_')[-1].isdigit():
                        stem = '_'.join(stem.split('_')[:-1])
                    output_path = output_folder / f"{stem}_{counter}.safetensors"
                    counter += 1
                
                save_file(sd, str(output_path))
                save_path = str(output_path)
                print(f"💾 Saved to: {save_path}")
                
            except Exception as e:
                print(f"❌ Save failed: {e}")
        
        print("="*50)
        return ((sd, 1.0, 1.0), info_str, recommended)


# ==================== EXPERIMENTAL: BASE MODEL MERGER ====================

class EasyLoRABaseMerger:
    """EXPERIMENTAL: Merge LoRAs directly into base model (bakes them in)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "method": (["linear", "ties_strict", "ties_gentle", "feature_mix", 
                           "magnitude", "subtract", "dare_rescale"], 
                          {"default": "linear"}),
                "density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "lora_a": (["None"] + loras,),
                "lora_b": (["None"] + loras,),
                "lora_data_a": ("LORA",),
                "lora_data_b": ("LORA",),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                
                # Method-specific sliders
                "uniqueness": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05}),
                "threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # Device options
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["auto", "float32", "bfloat16", "float16"], {"default": "auto"}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "merge_into_base"
    CATEGORY = "LoRA/Experimental"
    
    def merge_into_base(self, model, clip, method="linear", density=1.0,
                        lora_a="None", lora_b="None", 
                        lora_data_a=None, lora_data_b=None,
                        weight_a=1.0, weight_b=1.0,
                        uniqueness=0.7, threshold=0.2, blend=0.5,
                        device="auto", precision="auto", **kwargs):
        
        print("\n" + "="*50)
        print("🔥 BASE MODEL MERGER (Experimental)")
        print("="*50)
        print("⚠️  This bakes LoRAs directly into the model!")
        print("   The result is a new model with LoRAs permanently merged")
        print("="*50)
        
        # Create config
        config = MergeConfig(
            method=method,
            density=density,
            weight_a=weight_a,
            weight_b=weight_b,
            uniqueness=uniqueness,
            threshold=threshold,
            blend=blend,
            device_type=device,
            precision=precision,
            batch_size=32,
            streaming=True
        )
        
        # Get LoRA paths
        from .easy_lora_merger import save_lora_data_to_temp, load_lora_with_metadata, merge_loras, get_experiment_temp_path
        
        path_a = None
        path_b = None
        
        if lora_data_a is not None:
            path_a = save_lora_data_to_temp(lora_data_a, "A")
            print(f"📄 A: LORA data")
        elif lora_a != "None" and lora_a:
            path_a = folder_paths.get_full_path("loras", lora_a)
            print(f"📄 A: {lora_a}")
        
        if lora_data_b is not None:
            path_b = save_lora_data_to_temp(lora_data_b, "B")
            print(f"📄 B: LORA data")
        elif lora_b != "None" and lora_b:
            path_b = folder_paths.get_full_path("loras", lora_b)
            print(f"📄 B: {lora_b}")
        
        if not path_a or not path_b:
            print("❌ Need two LoRAs")
            return (model, clip, "Error: Need two LoRAs")
        
        print(f"⚖️  Weights: {weight_a} : {weight_b}")
        print(f"🎯 Method: {method}")
        
        # Create temp path for merged LoRA
        temp_path = get_experiment_temp_path("base_merge")
        
        # Merge the LoRAs
        try:
            merged_dict = merge_loras(path_a, path_b, config, temp_path)
            print(f"✅ LoRAs merged: {len(merged_dict)} keys")
        except Exception as e:
            print(f"❌ Merge failed: {e}")
            return (model, clip, f"Error: {e}")
        
        # Load the merged LoRA
        try:
            lora_data = comfy.utils.load_torch_file(str(temp_path))
            print(f"📥 Loaded merged LoRA")
        except Exception as e:
            print(f"❌ Failed to load merged LoRA: {e}")
            return (model, clip, f"Error: {e}")
        
        # BAKE IT INTO THE MODEL (strength 1.0 = full bake)
        print("🔥 Baking LoRA into base model...")
        try:
            model_baked, clip_baked = comfy.sd.load_lora_for_models(
                model, clip, lora_data, 1.0, 1.0
            )
            print("✅ LoRA baked successfully!")
        except Exception as e:
            print(f"❌ Failed to bake LoRA: {e}")
            return (model, clip, f"Error: {e}")
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except:
            pass
        
        info = f"Baked {method} merge (w={weight_a},{weight_b}) into base model"
        print("="*50)
        print("🎉 Base model ready with baked LoRAs!")
        print("="*50)
        
        return (model_baked, clip_baked, info)

# ==================== REGISTRATION ====================

NODE_CLASS_MAPPINGS = {
    "EasyLoRAmerger": EasyLoRAmergerNode,
    "EasyLoRAonlyMerger": EasyLoRAonlyMerger,
    "EasyLoRATripleMerger": EasyLoRATripleMerger,
    "MusubiLoraConverter": MusubiLoraConverter,
    "ZImageNormalizer": ZImageNormalizer,
    "EasyLoRABaseMerger": EasyLoRABaseMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyLoRAmerger": "Easy LoRA Merger",
    "EasyLoRAonlyMerger": "Easy LoRA-Only Merger",
    "EasyLoRATripleMerger": "🎨 Easy LoRA Triple Merger (Experimental)",
    "MusubiLoraConverter": "🔄 Musubi LoRA Converter (Experimental)",
    "ZImageNormalizer": "🔄 Z-Image Normalizer (Experimental)",
    "EasyLoRABaseMerger": "🔥 Bake LoRAs into Base Model (Experimental)",
}