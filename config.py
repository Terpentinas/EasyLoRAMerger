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

# ── INT8 percentile clipping for outlier reduction ─────────────────────
# Controls how compute_int8_scale() handles outlier weights during
# per-channel INT8 quantization. When set to < 1.0, the per-channel
# scale is computed from the value at this percentile (e.g. 0.999 =
# 99.9th percentile) instead of the absolute maximum, preventing a
# single extreme outlier from compressing the effective INT8 range.
#
# Example: 99.9% of weights in [-0.5, 0.5], one outlier at 10.0
#   Without clipping: scale = 10.0/127 ≈ 0.079 → bulk maps to ±6 (5%)
#   With 99.9% clip:  scale =  0.5/127 ≈ 0.004 → bulk maps to ±127 (100%)
#
# How to tune (advanced users, edit this file):
#   1.0   → Disable clipping (revert to max-based, original behavior)
#   0.999 → Clip at 99.9th percentile (default, recommended)
#   0.99  → Clip more aggressively (higher saturation risk)
#   0.9   → Very aggressive clipping (only if outliers dominate)
INT8_CLIP_PERCENTILE: float = 1.0

# ── SVD minimum dimension threshold ──────────────────────────────
# Controls the minimum dimension for SVD compression.
# Tensors where both dimensions are below this value are skipped.
# Set higher to only compress large matrices; set lower to compress more.
# Default 128 matches the original hardcoded value in _apply_svd_to_tensor().
SVD_MIN_DIMENSION: int = 128

# ── SVD selective mode minimum dimension ─────────────────────────
# In "selective" mode, weight matrices with both dimensions below
# this value are skipped. Only large matrices benefit from SVD in
# selective mode.
# Default 1024 matches the original hardcoded value.
SVD_SELECTIVE_MIN_DIM: int = 1024

# ── SVD parameter threshold for intermediate cleanup ─────────────
# Tensors with more than this many parameters trigger explicit
# intermediate variable cleanup (del U, S, Vh, sqrt_Sk, tensor_fp32)
# to free GPU memory during SVD.
# Default 1,048,576 (1024*1024) matches original hardcoded value.
SVD_CLEANUP_PARAM_THRESHOLD: int = 1048576

# Suppress warnings
warnings.filterwarnings("ignore", message="lora key not loaded")


# ── Shared UI option lists (single source of truth for all nodes) ──
# Standard float precisions — safe for all nodes without dequant.
PRECISION_STANDARD = ["auto", "float32", "bfloat16", "float16"]

# Extended with FP8 — only for nodes with proper dequant pipelines.
PRECISION_EXTENDED = PRECISION_STANDARD + ["fp8_e4m3fn", "fp8_e5m2"]

# Full studio toolkit — includes INT8 and SVD for precision conversion.
# Only MusubiCheckpointStudio uses this tier; it implements the actual
# conversion pipelines for int8, int8_convrot, and svd_only.
PRECISION_STUDIO = PRECISION_EXTENDED + ["int8", "int8_convrot", "svd_only"]

DEVICE_OPTIONS = ["auto", "cuda", "cpu"]

# GGUF output format options for Checkpoint Studio.
# Each entry maps to a GGMLQuantizationType in engine/gguf_writer.py.
# "safetensors" retains the existing output path (FP8/BF16/INT8 safetensors file).
FORMAT_OPTIONS = ["safetensors", "gguf_q8_0", "gguf_q5_0", "gguf_q4_0"]


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