"""
Configuration classes and constants for Easy LoRA Merger.
"""

import warnings
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal

try:
    from .engine.methods import (
        merge_linear, merge_cross, merge_ties_strict, merge_ties_gentle, merge_ties_contrast,
        merge_subtract, merge_magnitude, merge_feature_mix, merge_svd_preserve, merge_block_swap,
        merge_noise_aware, merge_gradient_alignment, merge_slerp, apply_density,
        merge_dare_lite, merge_dare_rescale, universal_merge_executor
    )
except ImportError:
    from engine.methods import (
        merge_linear, merge_cross, merge_ties_strict, merge_ties_gentle, merge_ties_contrast,
        merge_subtract, merge_magnitude, merge_feature_mix, merge_svd_preserve, merge_block_swap,
        merge_noise_aware, merge_gradient_alignment, merge_slerp, apply_density,
        merge_dare_lite, merge_dare_rescale, universal_merge_executor
    )

# Version constants
EASY_LORA_MERGER_VERSION = "1.1.0"
EASY_LORA_MERGER_DATE = "2026-02-15"

# ── RAM Guard threshold ──
# Controls when in-memory dict mode falls back to disk mode.
# Default 0.85 = 85% of available system RAM is the safe upper limit.
#
# How to tune (advanced users, edit this file):
#   Increase to 0.90-1.00  → More aggressive dict mode if you have RAM headroom
#   Decrease to 0.50-0.70  → Forces disk mode earlier for low-RAM systems
RAM_GUARD_THRESHOLD: float = 0.85

# Default threshold for active region detection in triple merge.
# When active_threshold is enabled (BOOL = True), this value is used.
# Setting to 0.0 disables filtering (all non-zero values treated as active).
ACTIVE_THRESHOLD_DEFAULT: float = 1e-8

# Suppress warnings
warnings.filterwarnings("ignore", message="lora key not loaded")


# ── Shared UI option lists (single source of truth for all nodes) ──
PRECISION_OPTIONS = ["auto", "float32", "bfloat16", "float16",
                     "fp8", "fp8_e4m3fn", "fp8_e5m2"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]


# ── Unified device/precision resolver ──
@dataclass
class DevicePrecisionConfig:
    """Resolves device/precision strings to torch objects once.

    All three nodes (Baker, Triple Merger, Checkpoint Merger) and
    all engines share this single resolver, ensuring consistent
    device/dtype resolution across the entire codebase.
    """
    device_type: str = "auto"
    precision: str = "auto"

    # Resolved on construction (not dataclass init fields)
    device: torch.device = field(init=False)
    dtype: torch.dtype = field(init=False)

    def __post_init__(self):
        # Lazy import to avoid circular dependency with utils
        from .utils import DeviceManager
        self.device = DeviceManager.get_device(self.device_type)
        self.dtype = DeviceManager.get_dtype(self.precision, self.device)


@dataclass
class MergeConfig:
    """Configuration for LoRA merging operations."""
    method: str = "linear"
    density: float = 1.0
    weight_a: float = 1.0
    weight_b: float = 1.0
    device_type: str = "auto"
    precision: str = "auto"
    metadata_mode: Literal["none", "preserve_a", "preserve_b", "merge_basic"] = "merge_basic"
    batch_size: int = 64
    streaming: bool = True
    energy_preservation: bool = True

    # NEW ATTRIBUTES for new methods
    uniqueness: float = 0.7      # For feature_mix
    threshold: float = 0.0       # For subtract
    blend: float = 0.5           # For magnitude
    blend_mode: str = "auto"  # "auto", "balanced", "dense", "fun_mode"
    active_threshold: float = 1e-8  # threshold for active region detection
    magnitude_scaling: Literal["none", "rms", "top_5%", "top_10%", "top_20%", "top_30%"] = "none"  # Signal magnitude scaling mode
    balancing_mode: Literal["safe", "creative", "disabled", "intensity", "impact"] = "safe"  # Auto-weight-balancing mode
    max_scaling_factor: float = 200.0  # Maximum allowed scaling factor for magnitude equalization
    z_image_max_scaling_factor: float = 10.0  # Safety clamp for Z-Image LoRAs
    unify_dtype_to_engine: bool = True  # Ensure uniform dtype across merged tensors (prevent Frankenstein files)

    def __post_init__(self):
        """Resolve device/precision once via DevicePrecisionConfig."""
        self._device_precision = DevicePrecisionConfig(
            device_type=self.device_type, precision=self.precision
        )

    @property
    def device_precision(self) -> DevicePrecisionConfig:
        """Access pre-resolved device/precision config."""
        return self._device_precision

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

# Register all merge methods
MergeMethodRegistry.register("linear")(merge_linear)
MergeMethodRegistry.register("cross")(merge_cross)
MergeMethodRegistry.register("ties_strict")(merge_ties_strict)
MergeMethodRegistry.register("ties_gentle")(merge_ties_gentle)
MergeMethodRegistry.register("ties_contrast")(merge_ties_contrast)
MergeMethodRegistry.register("dare_lite")(merge_dare_lite)
MergeMethodRegistry.register("dare_rescale")(merge_dare_rescale)
MergeMethodRegistry.register("subtract")(merge_subtract)
MergeMethodRegistry.register("magnitude")(merge_magnitude)
MergeMethodRegistry.register("feature_mix")(merge_feature_mix)
MergeMethodRegistry.register("svd_preserve")(merge_svd_preserve)
MergeMethodRegistry.register("block_swap")(merge_block_swap)
MergeMethodRegistry.register("noise_aware")(merge_noise_aware)
MergeMethodRegistry.register("gradient_alignment")(merge_gradient_alignment)
MergeMethodRegistry.register("slerp")(merge_slerp)