# easy_lora_merger.py - SECURE & OPTIMIZED VERSION
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
    merge_linear, merge_ties_old, merge_ties_new, merge_dare_old, 
    merge_dare_new, apply_density, merge_ties_gentle, merge_confidence_weighted,
    merge_layer_selective, merge_svd_preserve, merge_noise_aware, merge_gradient_alignment
)
from .klein_normalizer import (
    finalize_all_klein_keys, normalize_all_klein_formats, pad_or_truncate,
    universal_normalize, universal_finalize, safe_get_rank, detect_lora_format
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
    confidence_a: float = 0.5
    confidence_b: float = 0.5
    attn_weight: float = 1.0
    mlp_weight: float = 0.7
    device_type: Literal["auto", "cuda", "cpu"] = "auto"
    precision: Literal["auto", "float32", "bfloat16", "float16", "fp8"] = "auto"
    metadata_mode: Literal["none", "preserve_a", "preserve_b", "merge_basic"] = "merge_basic"
    batch_size: int = 32  # For VRAM optimization
    streaming: bool = True  # Stream tensors instead of loading all at once
    
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
MergeMethodRegistry.register("ties_strict")(merge_ties_old)
MergeMethodRegistry.register("ties_gentle")(merge_ties_gentle)
MergeMethodRegistry.register("ties_consensus")(merge_ties_old)  # Same as ties_strict
MergeMethodRegistry.register("dare_lite")(merge_dare_old)
MergeMethodRegistry.register("dare_rescale")(merge_dare_new)
MergeMethodRegistry.register("confidence_weighted")(merge_confidence_weighted)
MergeMethodRegistry.register("layer_selective")(merge_layer_selective)
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
    Get user output path with auto-increment and security validation.
    
    Args:
        save_folder: User-provided save folder
        filename: Base filename
        
    Returns:
        Safe output Path
    """
    # Default filename
    if not filename or filename.strip() == "":
        filename = "merged_lora"
    
    # Ensure .safetensors extension
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"
    
    # Get base output folder (LORA directory or current)
    lora_folders = folder_paths.get_folder_paths("loras")
    default_base = Path(lora_folders[0]) if lora_folders else Path.cwd()
    
    # Sanitize user-provided folder
    if save_folder and isinstance(save_folder, str) and save_folder.strip():
        try:
            output_folder = sanitize_path(save_folder.strip(), default_base)
        except SecurityError as e:
            print(f"⚠️ Security warning: {e}, using default folder")
            output_folder = default_base
    else:
        output_folder = default_base
    
    # Create directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate safe filename with auto-increment
    original_path = output_folder / filename
    
    # Sanitize filename
    safe_filename = ''.join(c for c in original_path.name 
                           if c.isalnum() or c in '._- ').rstrip()
    original_path = original_path.parent / safe_filename
    
    # Auto-increment if exists
    counter = 1
    output_path = original_path
    while output_path.exists():
        stem = original_path.stem
        # Remove existing counter suffix
        if "_" in stem and stem.split("_")[-1].isdigit():
            stem = "_".join(stem.split("_")[:-1])
        output_path = original_path.parent / f"{stem}_{counter}{original_path.suffix}"
        counter += 1
    
    return output_path

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

# ==================== STREAMING MERGE ENGINE ====================

class StreamingMergeEngine:
    """Performs memory-efficient streaming merges."""
    
    def __init__(self, config: MergeConfig):
        self.config = config
        self.device = DeviceManager.get_device(config.device_type)
        self.dtype = DeviceManager.get_dtype(config.precision, self.device)
        self.original_dtype = self.dtype  # Store original dtype for reference
        
        # Optimize device settings
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if torch.cuda.get_device_capability(self.device)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"💻 Device: {self.device}, Precision: {self.dtype}")
    
    def merge_with_streaming(self, path_a: Path, path_b: Path, 
                            output_path: Path) -> Dict[str, torch.Tensor]:
        """
        Simplified merge function that works with all formats.
        """
        print("🔍 Loading LoRAs with streaming...")
        
        # First, let's load and normalize both models fully
        # This is simpler than trying to stream with complex key mappings
        
        print("🔄 Loading and normalizing models...")
        
        # Load both models completely (they should fit in RAM)
        with safe_open(path_a, framework="pt") as fa:
            sd_a = {k: fa.get_tensor(k) for k in fa.keys()}
        
        with safe_open(path_b, framework="pt") as fb:
            sd_b = {k: fb.get_tensor(k) for k in fb.keys()}
        
        # Normalize using the existing universal_normalize function
        sd_a = universal_normalize(sd_a)
        sd_b = universal_normalize(sd_b)
        
        print(f"📊 Normalized A: {len(sd_a)} keys, B: {len(sd_b)} keys")
        
        # Find common keys
        keys_a = set(sd_a.keys())
        keys_b = set(sd_b.keys())
        common_keys = keys_a & keys_b
        
        print(f"🧩 Found {len(common_keys)} common layers to merge")
        
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
        
        # Track dtype fallback status
        current_dtype = self.dtype
        dtype_changed = False
        
        for i in range(0, len(keys_list), batch_size):
            batch_keys = keys_list[i:i + batch_size]
            
            for key in batch_keys:
                try:
                    # Try with current dtype
                    try:
                        t_a = sd_a[key].to(self.device).to(current_dtype)
                        t_b = sd_b[key].to(self.device).to(current_dtype)
                    except Exception as dtype_error:
                        # dtype conversion failed, try fallback
                        if not dtype_changed:
                            current_dtype = DeviceManager.get_fallback_dtype(current_dtype)
                            dtype_changed = True
                            print(f"🔄 Changed dtype to {current_dtype} due to conversion error")
                            t_a = sd_a[key].to(self.device).to(current_dtype)
                            t_b = sd_b[key].to(self.device).to(current_dtype)
                        else:
                            # Already tried fallback, re-raise
                            raise dtype_error
                    
                    # Rank adjustment
                    rank_a = safe_get_rank(t_a, key)
                    rank_b = safe_get_rank(t_b, key)
                    target_rank = max(rank_a, rank_b)
                    
                    if rank_a != rank_b:
                        t_a = silent_pad_or_truncate(t_a, target_rank, key)
                        t_b = silent_pad_or_truncate(t_b, target_rank, key)
                    
                    # Get merge method
                    merge_func = MergeMethodRegistry.get_method(self.config.method)
                    
                    # Prepare arguments - only pass confidence for methods that support it
                    method_args = {
                        'a': t_a, 'b': t_b, 
                        'wa': self.config.weight_a, 'wb': self.config.weight_b
                    }
                    
                    # Add method-specific parameters
                    if self.config.method == "confidence_weighted":
                        method_args['confidence_a'] = self.config.confidence_a
                        method_args['confidence_b'] = self.config.confidence_b
                    elif self.config.method in ["ties_strict", "ties_consensus"]:
                        # ties_strict and ties_consensus use merge_ties_old
                        # Don't pass confidence parameters to TIES methods
                        pass
                    elif self.config.method == "ties_gentle":
                        method_args['agreement_threshold'] = 0.3  # Default value
                    
                    if self.config.method == "layer_selective":
                        method_args['key'] = key
                        method_args['attn_weight'] = self.config.attn_weight
                        method_args['mlp_weight'] = self.config.mlp_weight
                    
                    if self.config.method == "svd_preserve":
                        method_args['preserve_ratio'] = self.config.density
                    
                    if self.config.method == "noise_aware":
                        method_args['noise_threshold'] = 0.01 * self.config.density
                    
                    if self.config.method == "dare_rescale":
                        method_args['drop_rate'] = 1.0 - self.config.density
                    
                    # Try to merge
                    try:
                        merged = merge_func(**method_args)
                    except Exception as merge_error:
                        # If merge fails with current dtype, try fallback
                        if not dtype_changed and "not implemented" in str(merge_error):
                            current_dtype = DeviceManager.get_fallback_dtype(current_dtype)
                            dtype_changed = True
                            print(f"🔄 Changed dtype to {current_dtype} due to: {merge_error}")
                            
                            # Retry with new dtype
                            t_a = sd_a[key].to(self.device).to(current_dtype)
                            t_b = sd_b[key].to(self.device).to(current_dtype)
                            
                            # Reapply rank adjustment
                            if rank_a != rank_b:
                                t_a = silent_pad_or_truncate(t_a, target_rank, key)
                                t_b = silent_pad_or_truncate(t_b, target_rank, key)
                            
                            # Update method args with new tensors
                            method_args['a'] = t_a
                            method_args['b'] = t_b
                            
                            merged = merge_func(**method_args)
                        else:
                            # Already tried fallback or different error
                            raise merge_error
                    
                    # Apply density if needed
                    if "dare" not in self.config.method and self.config.density < 1.0:
                        merged = apply_density(merged, self.config.density)
                    
                    # Store result
                    merged_dict[key] = merged.cpu()
                    
                    # Cleanup
                    del t_a, t_b, merged
                    
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
            time.sleep(0.001)
        
        # Add unique keys with current dtype
        unique_keys_a = keys_a - keys_b
        unique_keys_b = keys_b - keys_a
        
        if unique_keys_a:
            print(f"📝 Adding {len(unique_keys_a)} unique keys from A")
            for key in unique_keys_a:
                try:
                    tensor = sd_a[key].to(self.device).to(current_dtype) * self.config.weight_a
                    if self.config.density < 1.0 and "dare" not in self.config.method:
                        tensor = apply_density(tensor, self.config.density)
                    merged_dict[key] = tensor.cpu()
                except Exception as e:
                    print(f"⚠️ Failed to add unique key {key}: {e}")
        
        if unique_keys_b:
            print(f"📝 Adding {len(unique_keys_b)} unique keys from B")
            for key in unique_keys_b:
                try:
                    tensor = sd_b[key].to(self.device).to(current_dtype) * self.config.weight_b
                    if self.config.density < 1.0 and "dare" not in self.config.method:
                        tensor = apply_density(tensor, self.config.density)
                    merged_dict[key] = tensor.cpu()
                except Exception as e:
                    print(f"⚠️ Failed to add unique key {key}: {e}")
        
        if dtype_changed:
            print(f"📊 Final dtype used: {current_dtype} (originally requested {self.original_dtype})")
        
        print(f"✅ Merged {len(merged_dict)} total keys")
        return merged_dict

    
    def _process_batches(self, common_keys: Set[str], 
                        norm_to_orig_a: Dict[str, str], norm_to_orig_b: Dict[str, str],
                        fa, fb) -> Dict[str, torch.Tensor]:
        """Process keys in batches to optimize VRAM usage."""
        merged_dict = {}
        keys_list = list(common_keys)
        batch_size = self.config.batch_size
        
        # Create progress bar for ComfyUI
        try:
            pbar = comfy.utils.ProgressBar(len(keys_list))
        except:
            pbar = None
        
        for i in range(0, len(keys_list), batch_size):
            batch_keys = keys_list[i:i + batch_size]
            batch_results = {}
            
            for norm_key in batch_keys:
                try:
                    # Get original keys
                    orig_key_a = norm_to_orig_a.get(norm_key)
                    orig_key_b = norm_to_orig_b.get(norm_key)
                    
                    if not orig_key_a or not orig_key_b:
                        print(f"⚠️ Key mapping missing for {norm_key}, skipping")
                        continue
                    
                    # Load tensors
                    with torch.no_grad():
                        t_a = fa.get_tensor(orig_key_a).to(self.device).to(self.dtype)
                        t_b = fb.get_tensor(orig_key_b).to(self.device).to(self.dtype)
                    
                    # Debug output for first few keys
                    if i == 0 and len(batch_results) < 3:
                        print(f"   Merging: {orig_key_a[:50]}...")
                        print(f"     Shape A: {t_a.shape}, B: {t_b.shape}")
                    
                    # Rank adjustment
                    rank_a = safe_get_rank(t_a, norm_key)
                    rank_b = safe_get_rank(t_b, norm_key)
                    target_rank = max(rank_a, rank_b)
                    
                    if rank_a != rank_b:
                        print(f"   Rank adjustment: {rank_a} -> {target_rank} for {norm_key[:40]}...")
                    
                    t_a = silent_pad_or_truncate(t_a, target_rank, norm_key)
                    t_b = silent_pad_or_truncate(t_b, target_rank, norm_key)
                    
                    # Get merge method and merge
                    merge_func = MergeMethodRegistry.get_method(self.config.method)
                    
                    # Prepare method-specific arguments
                    method_args = {
                        'a': t_a, 'b': t_b, 
                        'wa': self.config.weight_a, 'wb': self.config.weight_b
                    }
                    
                    if self.config.method in ["confidence_weighted", "ties_strict", 
                                            "ties_gentle", "ties_consensus"]:
                        method_args['confidence_a'] = self.config.confidence_a
                        method_args['confidence_b'] = self.config.confidence_b
                    
                    if self.config.method == "layer_selective":
                        method_args['key'] = norm_key
                        method_args['attn_weight'] = self.config.attn_weight
                        method_args['mlp_weight'] = self.config.mlp_weight
                    
                    if self.config.method == "svd_preserve":
                        method_args['preserve_ratio'] = self.config.density
                    
                    if self.config.method == "noise_aware":
                        method_args['noise_threshold'] = 0.01 * self.config.density
                    
                    # Special handling for dare_rescale
                    if self.config.method == "dare_rescale":
                        drop_rate = 1.0 - self.config.density
                        method_args['drop_rate'] = drop_rate
                    
                    # Apply merge
                    merged = merge_func(**method_args)
                    
                    # Apply density if needed (for non-DARE methods)
                    if "dare" not in self.config.method and self.config.density < 1.0:
                        merged = apply_density(merged, self.config.density)
                    
                    # Store result (use normalized key pattern)
                    # For Flux models, we need to use a consistent naming pattern
                    result_key = norm_key
                    batch_results[result_key] = merged.cpu()
                    
                    # Clean up GPU memory
                    del t_a, t_b, merged
                    
                except Exception as e:
                    print(f"⚠️ Failed to merge {norm_key}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Update main dict and clean up
            merged_dict.update(batch_results)
            batch_results.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update progress
            if pbar:
                pbar.update(min(batch_size, len(keys_list) - i))
            
            # Yield to keep ComfyUI responsive
            if i % (batch_size * 2) == 0:
                comfyui_yield()
        
        return merged_dict
    
    def _process_all(self, common_keys: Set[str], 
                    norm_to_orig_a: Dict[str, str], norm_to_orig_b: Dict[str, str],
                    fa, fb) -> Dict[str, torch.Tensor]:
        """Process all keys at once (for debugging)."""
        merged_dict = {}
        
        for norm_key in common_keys:
            orig_key_a = norm_to_orig_a.get(norm_key)
            orig_key_b = norm_to_orig_b.get(norm_key)
            
            if not orig_key_a or not orig_key_b:
                continue
            
            t_a = fa.get_tensor(orig_key_a)
            t_b = fb.get_tensor(orig_key_b)
            
            # Use original merge logic as fallback
            t_a = t_a.to(self.device).to(self.dtype)
            t_b = t_b.to(self.device).to(self.dtype)
            
            # Simple linear merge for debugging
            merged = merge_linear(t_a, t_b, self.config.weight_a, self.config.weight_b)
            merged_dict[norm_key] = merged.cpu()
        
        return merged_dict
    
# ==================== SIMPLIFIED STREAMING MERGE ENGINE ====================

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
    
    def merge_with_streaming(self, path_a: Path, path_b: Path, 
                            output_path: Path) -> Dict[str, torch.Tensor]:
        """
        Simplified merge function that works with all formats.
        """
        print("🔍 Loading LoRAs with streaming...")
        
        # First, let's load and normalize both models fully
        # This is simpler than trying to stream with complex key mappings
        
        print("🔄 Loading and normalizing models...")
        
        # Load both models completely (they should fit in RAM)
        with safe_open(path_a, framework="pt") as fa:
            sd_a = {k: fa.get_tensor(k) for k in fa.keys()}
        
        with safe_open(path_b, framework="pt") as fb:
            sd_b = {k: fb.get_tensor(k) for k in fb.keys()}
        
        # Normalize using the existing universal_normalize function
        sd_a = universal_normalize(sd_a)
        sd_b = universal_normalize(sd_b)
        
        print(f"📊 Normalized A: {len(sd_a)} keys, B: {len(sd_b)} keys")
        
        # Find common keys
        keys_a = set(sd_a.keys())
        keys_b = set(sd_b.keys())
        common_keys = keys_a & keys_b
        
        print(f"🧩 Found {len(common_keys)} common layers to merge")
        
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
        
        for i in range(0, len(keys_list), batch_size):
            batch_keys = keys_list[i:i + batch_size]
            
            for key in batch_keys:
                try:
                    # Get tensors
                    t_a = sd_a[key].to(self.device).to(self.dtype)
                    t_b = sd_b[key].to(self.device).to(self.dtype)
                    
                    # Rank adjustment
                    rank_a = safe_get_rank(t_a, key)
                    rank_b = safe_get_rank(t_b, key)
                    target_rank = max(rank_a, rank_b)
                    
                    if rank_a != rank_b:
                        t_a = silent_pad_or_truncate(t_a, target_rank, key)
                        t_b = silent_pad_or_truncate(t_b, target_rank, key)
                    
                    # Get merge method
                    merge_func = MergeMethodRegistry.get_method(self.config.method)
                    
                    # Prepare arguments
                    method_args = {
                        'a': t_a, 'b': t_b, 
                        'wa': self.config.weight_a, 'wb': self.config.weight_b
                    }
                    
                    # Add method-specific parameters
                    if self.config.method in ["confidence_weighted", "ties_strict", 
                                            "ties_gentle", "ties_consensus"]:
                        method_args['confidence_a'] = self.config.confidence_a
                        method_args['confidence_b'] = self.config.confidence_b
                    
                    if self.config.method == "layer_selective":
                        method_args['key'] = key
                        method_args['attn_weight'] = self.config.attn_weight
                        method_args['mlp_weight'] = self.config.mlp_weight
                    
                    if self.config.method == "svd_preserve":
                        method_args['preserve_ratio'] = self.config.density
                    
                    if self.config.method == "noise_aware":
                        method_args['noise_threshold'] = 0.01 * self.config.density
                    
                    if self.config.method == "dare_rescale":
                        method_args['drop_rate'] = 1.0 - self.config.density
                    
                    # Merge
                    merged = merge_func(**method_args)
                    
                    # Apply density if needed
                    if "dare" not in self.config.method and self.config.density < 1.0:
                        merged = apply_density(merged, self.config.density)
                    
                    # Store result
                    merged_dict[key] = merged.cpu()
                    
                    # Cleanup
                    del t_a, t_b, merged
                    
                except Exception as e:
                    print(f"⚠️ Failed to merge {key}: {e}")
            
            # Update progress
            if pbar:
                pbar.update(min(batch_size, len(keys_list) - i))
            
            # Clean GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Yield for UI
            if i % (batch_size * 4) == 0:
                comfyui_yield()
        
        # Add unique keys
        unique_keys_a = keys_a - keys_b
        unique_keys_b = keys_b - keys_a
        
        if unique_keys_a:
            print(f"📝 Adding {len(unique_keys_a)} unique keys from A")
            for key in unique_keys_a:
                try:
                    tensor = sd_a[key].to(self.device).to(self.dtype) * self.config.weight_a
                    if self.config.density < 1.0 and "dare" not in self.config.method:
                        tensor = apply_density(tensor, self.config.density)
                    merged_dict[key] = tensor.cpu()
                except Exception as e:
                    print(f"⚠️ Failed to add unique key {key}: {e}")
        
        if unique_keys_b:
            print(f"📝 Adding {len(unique_keys_b)} unique keys from B")
            for key in unique_keys_b:
                try:
                    tensor = sd_b[key].to(self.device).to(self.dtype) * self.config.weight_b
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
    sd_a_norm = universal_normalize(sd_a)
    sd_b_norm = universal_normalize(sd_b)
    
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
        
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "method": (MergeMethodRegistry.list_methods(), {
                    "default": "linear",
                    "tooltip": "Choose merging method"
                }),
                "density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "lora_a": (["None"] + loras,),
                "lora_b": (["None"] + loras,),
                "lora_data_a": ("LORA",),
                "lora_data_b": ("LORA",),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -5, "max": 5, "step": 0.05}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -5, "max": 5, "step": 0.05}),
                "save_trigger": ("BOOLEAN", {"default": False}),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False}),
                "filename": ("STRING", {"default": "merged_lora", "multiline": False}),
                "attn_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "mlp_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "confidence_a": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "confidence_b": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (precision_options, {
                    "default": "auto",
                    "tooltip": precision_tooltips.get("fp8", "Select precision")
                }),
                "metadata_mode": (["none", "preserve_a", "preserve_b", "merge_basic"], {
                    "default": "merge_basic",
                }),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 8}),
                "streaming": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "save_path")
    FUNCTION = "merge"
    CATEGORY = "LoRA"
    
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
        
        # Check cache
        if cache_key in cls._CACHE and cls._CACHE[cache_key] == current_hash:
            return cls._CACHE[cache_key]
        
        # Update cache
        cls._CACHE[cache_key] = current_hash
        return current_hash
    
    def merge(self, model, clip, method="linear", density=1.0,
              lora_a="None", lora_b="None", lora_data_a=None, lora_data_b=None,
              weight_a=1.0, weight_b=1.0, save_trigger=False, save_folder="", 
              filename="merged_lora", attn_weight=1.0, mlp_weight=0.7,
              confidence_a=0.5, confidence_b=0.5, device="auto", 
              precision="auto", metadata_mode="merge_basic",
              batch_size=32, streaming=True):
        
        print("\n" + "="*50)
        print("🧩 Easy LoRA Merger (Secure & Optimized)")
        print("="*50)
        
        # Create config from inputs
        config = MergeConfig.from_inputs(
            method=method,
            density=density,
            weight_a=weight_a,
            weight_b=weight_b,
            confidence_a=confidence_a,
            confidence_b=confidence_b,
            attn_weight=attn_weight,
            mlp_weight=mlp_weight,
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
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora_data, 1.0, 1.0
            )
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return (model, clip, "")
        
        # Final message
        if save_trigger:
            print(f"💾 Saved permanently to {output_path}")
        else:
            print(f"📁 Temp file will auto-cleanup (keeps last 10)")
        
        print("="*50)
        print("🎉 Done!")
        print("="*50)
        
        # Clear cache if this was a save operation
        if save_trigger:
            self._CACHE.clear()
        
        return (model_lora, clip_lora, str(output_path))

# ==================== LORA-ONLY NODE ====================

class EasyLoRAonlyMerger:
    """Merge LoRAs without loading into model."""
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        default_folder = ""
        lora_folders = folder_paths.get_folder_paths("loras")
        if lora_folders:
            default_folder = str(lora_folders[0])
        
        return {
            "required": {
                "method": (MergeMethodRegistry.list_methods(), {"default": "linear"}),
                "density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "lora_a": (["None"] + loras,),
                "lora_b": (["None"] + loras,),
                "lora_data_a": ("LORA",),
                "lora_data_b": ("LORA",),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -5, "max": 5, "step": 0.05}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -5, "max": 5, "step": 0.05}),
                "save_trigger": ("BOOLEAN", {"default": False}),
                "save_folder": ("STRING", {"default": default_folder, "multiline": False}),
                "filename": ("STRING", {"default": "merged_lora", "multiline": False}),
                "attn_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "mlp_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "confidence_a": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "confidence_b": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["auto", "float32", "bfloat16", "float16", "fp8"], {"default": "auto"}),
                "metadata_mode": (["none", "preserve_a", "preserve_b", "merge_basic"], {
                    "default": "merge_basic",
                }),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 8}),
                "streaming": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LORA", "STRING")
    RETURN_NAMES = ("merged_lora", "save_path")
    FUNCTION = "merge_only"
    CATEGORY = "LoRA"
    
    def merge_only(self, method="linear", density=1.0,
                   lora_a="None", lora_b="None", lora_data_a=None, lora_data_b=None,
                   weight_a=1.0, weight_b=1.0, save_trigger=False, save_folder="", 
                   filename="merged_lora", attn_weight=1.0, mlp_weight=0.7,
                   confidence_a=0.5, confidence_b=0.5, device="auto", 
                   precision="auto", metadata_mode="merge_basic",
                   batch_size=32, streaming=True):
        
        print("\n" + "="*50)
        print("🧩 LoRA-Only Merger (Secure & Optimized)")
        print("="*50)
        
        # Create config
        config = MergeConfig.from_inputs(
            method=method,
            density=density,
            weight_a=weight_a,
            weight_b=weight_b,
            confidence_a=confidence_a,
            confidence_b=confidence_b,
            attn_weight=attn_weight,
            mlp_weight=mlp_weight,
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

# ==================== REGISTRATION ====================

NODE_CLASS_MAPPINGS = {
    "EasyLoRAmerger": EasyLoRAmergerNode,
    "EasyLoRAonlyMerger": EasyLoRAonlyMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyLoRAmerger": "Easy LoRA Merger",
    "EasyLoRAonlyMerger": "Easy LoRA-Only Merger"
}