"""
FP8 Quantizer — Shared Module for Scale-Then-Cast Quantization

Provides the canonical FP8 quantization algorithm used across EasyLoRAMerger:
  - compute_weight_scale()   — Per-tensor scale factor for FP8 quantization
  - quantize_to_fp8()        — Scale-then-cast quantization (matching Creator's algorithm)
  - should_preserve_bf16()   — BF16 preservation for sensitive layer types
  - get_dequant_dtype()      — GPU-aware dequant target resolution (FP16 for legacy GPUs)
  - FP8_PRESERVE_BF16_PATTERNS — The pattern tuple (source of truth)
  - build_fp8_quantization_metadata() — Inject _quantization_metadata into metadata dict
  - strip_input_scale_keys()         — Remove .input_scale companion keys for ComfyUI compat

All stateless functions operating on tensors and keys.
This module is the single source of truth; musubi_checkpoint_studio.py
and baker_node.py delegate to it.
"""

import json
from typing import Dict, Optional

import torch

# ── FP8 format constants ──────────────────────────────────────────────────────
# Max representable values for each FP8 format (used in scale computation).
#   float8_e4m3fn:  max = 448.0  (3 mantissa bits, 4 exponent bits)
#   float8_e5m2:    max = 57344.0 (2 mantissa bits, 5 exponent bits)
FP8_MAX_VALUES = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}


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
#
# SDXL/SD1.5 and Anima patterns were validated in INT8_PRESERVE_FP16_PATTERNS
# (engine/int8_quantizer.py) — ported here for equivalent FP8 protection.
FP8_PRESERVE_BF16_PATTERNS: tuple = (
    # ── Diffusers-style path separators ──
    '/final_layer/',
    '/time_in/',
    '/txt_in/',
    '/vector_in/',
    '/guidance_in/',
    '/norm.',
    # ── FLUX dot-notation keys ──
    'final_layer.',
    'time_in.',
    'txt_in.',
    'vector_in.',
    'guidance_in.',
    'img_in.',
    'single_stream_modulation.',
    # FLUX modulation tensors use underscore suffixes (_img, _txt), not dots
    'double_stream_modulation_img.',
    'double_stream_modulation_txt.',

    # ── SDXL/SD1.5 dot-notation keys ──
    # Time embedding projection — SDXL/SD1.5 equivalent of Flux's time_in.
    # Controls how the denoising timestep signal enters the model.
    # Errors here compound through all denoising steps.
    'time_embed.',
    # Label embedding (SDXL only) — conditioning signal entry point.
    # SDXL uses pooled text embeddings from both CLIP models.
    'label_emb.',
    # First input block — the very first convolution processing the
    # noisy latent. Signal entry point — errors propagate everywhere.
    'input_blocks.0.',
    # UNet middle block (bottleneck) — all features pass through.
    # Small number of tensors but critical for global coherence.
    'middle_block.',
    # Output projection — final convolution before VAE decode.
    # Signal exit point — errors directly affect pixel output.
    'out.',

    # ── SDXL diffusers-style path equivalents ──
    '/time_embed/',
    '/label_emb/',
    '/input_blocks/0/',
    '/middle_block/',
    '/out/',

    # ── Anima architecture patterns ──
    # Input patch embedding — the very first layer processing the noisy latent.
    # Equivalent of Flux's img_in. and SDXL's input_blocks.0.
    # Signal entry — errors here propagate through all 35 DiT blocks.
    # Matches: net.x_embedder.proj.weight
    'net.x_embedder.',
    # Time embedding projection — timestep conditioning signal entry.
    # Equivalent of Flux's time_in. and SDXL's time_embed.
    # Errors here compound through all 35 denoising steps.
    # Matches: net.t_embedder.0.weight, net.t_embedder.2.weight
    'net.t_embedder.',
    # First DiT block — signal entry into the transformer core.
    # All embedded signals (latent, timestep, text conditioning) flow
    # through blocks.0 first. Errors here corrupt every subsequent block.
    # Matches: net.blocks.0.adaln_modulation_cross_attn.1.weight
    'net.blocks.0.',
    # Runtime format variant — ComfyUI wraps UNet under diffusion_model.
    # When a baked/converted checkpoint is re-loaded, keys use the
    # diffusion_model. prefix instead of net. — but the same first-block
    # signal entry rationale applies.
    # Matches: diffusion_model.blocks.0.adaln_modulation_cross_attn.1.weight
    'diffusion_model.blocks.0.',
    # Output projection — final linear layer before VAE decode.
    # Equivalent of Flux's final_layer. and SDXL's out.
    # Signal exit — errors directly affect pixel output.
    # Matches: net.final_layer.linear.weight
    'net.final_layer.',

    # ── LLM Adapter — Anima's T5 cross-attention conditioning interface ──
    # First llm_adapter block — signal entry for T5 cross-attention conditioning.
    # Similar rationale to net.blocks.0.: the first block processes raw T5
    # hidden states before they propagate to downstream blocks. Errors here
    # corrupt all cross-attention conditioning across every DiT block.
    # Matches: net.llm_adapter.blocks.0.cross_attn.k_proj.weight
    'net.llm_adapter.blocks.0.',
    # LLM Adapter input embedding — projects T5 hidden states into the
    # adapter's feature space. Signal entry point.
    # Matches: net.llm_adapter.embed.weight
    'net.llm_adapter.embed.',
    # LLM Adapter output norm — final normalization before the output
    # projection. Should be preserved alongside other norm parameters.
    # Matches: net.llm_adapter.norm.weight
    'net.llm_adapter.norm.',
    # LLM Adapter output projection — projects adapter features back into
    # the main DiT block's feature space. Signal exit point.
    # Matches: net.llm_adapter.out_proj.weight, net.llm_adapter.out_proj.bias
    'net.llm_adapter.out_proj.',

    # Suffix-based patterns (work for both formats)
    '_norm.scale',
    '_norm.bias',
)


# ── Public API ────────────────────────────────────────────────────────────────


def get_dequant_dtype(
    target_device: Optional[torch.device] = None,
    file_metadata: Optional[Dict] = None,
) -> torch.dtype:
    """Return optimal dequantization dtype for FP8 tensors on the target device.

    Resolution order:
    1. If ``file_metadata`` has a ``'dequant_target'`` hint → use that
       (honors the baker's intent for cross‑compilation scenarios).
    2. Else → auto‑detect from device capability:
       - CC >= 8.0 (Ampere+): ``bfloat16`` (natively supported)
       - CC < 8.0 (Pascal/Turing): ``float16`` (only natively‑supported 16‑bit option)
       - CPU: ``float32``

    Args:
        target_device: The torch device that will run inference.
            If ``None``, uses ``DeviceManager.get_device()``.
        file_metadata: Optional metadata dict from the safetensors file.
            If present and contains ``'dequant_target'``, that value takes
            priority over auto‑detection.

    Returns:
        torch.dtype: ``torch.bfloat16``, ``torch.float16``, or ``torch.float32``.
    """
    # Layer 1: File metadata hint (cross‑compilation support)
    if file_metadata and 'dequant_target' in file_metadata:
        hint = file_metadata['dequant_target']
        if hint == 'float16':
            return torch.float16
        elif hint == 'bfloat16':
            return torch.bfloat16
        elif hint == 'float32':
            return torch.float32
        # Unknown hint → fall through to auto‑detect

    # Layer 2: Auto‑detect from device capability
    if target_device is None:
        # Lazy import to avoid circular dependency
        from ..utils import DeviceManager as _DevMgr
        target_device = _DevMgr.get_device()
    elif isinstance(target_device, str):
        from ..utils import DeviceManager as _DevMgr
        target_device = _DevMgr.get_device(target_device)

    if target_device.type == 'cuda':
        try:
            cap = torch.cuda.get_device_capability(target_device)
            # CC >= 8.0 (Ampere, Ada, Hopper, Blackwell) → BF16 native
            if cap[0] >= 8:
                return torch.bfloat16
            # CC < 8.0 (Pascal, Turing, Volta) → FP16 native or emulated
            return torch.float16
        except Exception:
            return torch.float16  # safe fallback

    return torch.float32


def should_preserve_bf16(key: str) -> bool:
    """Return True if this key should stay in BF16 when target precision is FP8.

    Per FP8 comparison analysis (flux-test/FP8_COMPARISON_REPORT.md), the Creator's FP8
    preserves these layers in BF16 because they are sensitive small tensors (norm scales,
    input projections) where FP8 quantization introduces measurable precision loss.
    """
    key_lower = key.lower()
    for pattern in FP8_PRESERVE_BF16_PATTERNS:
        if pattern in key_lower:
            return True
    return False


def compute_weight_scale(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Compute the per-tensor weight_scale for FP8 quantization.

    The scale factor maps the max absolute value of the tensor to the
    max representable value of the FP8 format:
        weight_scale = max(abs(tensor)) / max_fp8

    Args:
        tensor: Original float tensor (before FP8 quantization).
        target_dtype: torch.float8_e4m3fn or torch.float8_e5m2.

    Returns:
        F32 scalar tensor containing the scale factor.
    """
    max_fp8 = FP8_MAX_VALUES.get(target_dtype, 448.0)
    # Convert to float32 first — abs() and max() are not implemented for
    # Float8 dtypes on all PyTorch versions (e.g. ComfyUI's CUDA build).
    tensor_f32 = tensor.float()
    # Guard: sanitize NaN/Inf → 0 to prevent corrupted scale factors.
    # A single NaN in max_abs → scale=NaN → tensor/NaN = NaN during dequant.
    if not torch.isfinite(tensor_f32).all():
        tensor_f32 = torch.nan_to_num(tensor_f32, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = tensor_f32.abs().max()
    # Force F32 tensor division to match Creator's precision (closes ~2% gap)
    scale = max_abs / torch.tensor(max_fp8, dtype=torch.float32, device=max_abs.device)
    scale = torch.clamp(scale, min=1e-12)
    return scale.to(dtype=torch.float32)


def quantize_to_fp8(tensor: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """Quantize tensor to FP8 using scale-then-cast (matching Creator's algorithm).

    Creator's approach (validated at 100.00% match):
      1. Compute scale = max_abs / max_fp8 (in F32 precision)
      2. Divide tensor by scale
      3. Clamp to [-max_fp8, max_fp8]
      4. Cast to FP8

    This differs from PyTorch's .to(fp8) which uses hardware rounding directly
    without scaling — producing ~98% different byte values.

    Edge case guards:
      - NaN/Inf in weights → sanitized to 0.0 before scale computation.
        (Without this, a single NaN would produce scale=NaN, corrupting the
        entire block during dequant via `fp8_value * NaN = NaN`.)
      - All-zero tensors → scale clamped to 1e-12, quantized to 0. Correct.
    """
    max_fp8 = FP8_MAX_VALUES.get(fp8_dtype, 448.0)
    tensor_f32 = tensor.float()

    # Guard: sanitize NaN/Inf → 0 to prevent corrupted scale factors.
    if not torch.isfinite(tensor_f32).all():
        tensor_f32 = torch.nan_to_num(tensor_f32, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1: Compute scale from cleaned tensor (F32 precision)
    max_abs = tensor_f32.abs().max()
    scale = max_abs / torch.tensor(max_fp8, dtype=torch.float32, device=max_abs.device)
    scale = torch.clamp(scale, min=1e-12)  # All-zero tensor → 1e-12, not 0

    # Steps 2-4: scale-then-cast
    tensor_f32.div_(scale)          # Normalize to [-max_fp8, max_fp8] range
    tensor_f32.clamp_(-max_fp8, max_fp8)  # Clamp to FP8 representable range
    result = tensor_f32.to(dtype=fp8_dtype)

    return result


def quantize_weight_to_fp8_with_scales(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> tuple:
    """Quantize a weight tensor to FP8 and return (quantized_tensor, weight_scale, input_scale).

    This is a convenience wrapper that performs all three steps together:
      1. Compute weight_scale
      2. Compute input_scale (same as weight_scale per Creator's convention)
      3. Quantize the tensor

    Args:
        tensor: Original float tensor (before FP8 quantization).
        fp8_dtype: torch.float8_e4m3fn or torch.float8_e5m2.

    Returns:
        Tuple of (quantized_fp8_tensor, weight_scale_f32_scalar, input_scale_f32_scalar).
    """
    wscale = compute_weight_scale(tensor, fp8_dtype)
    iscale = compute_weight_scale(tensor, fp8_dtype)  # Same computation per Creator
    quantized = quantize_to_fp8(tensor, fp8_dtype)
    return quantized, wscale, iscale


def quantize_weight_to_fp8_with_scales_optimized(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> tuple:
    """Single-pass FP8 quantization — computes weight_scale and quantize in ONE scan.

    Eliminates the 2 redundant tensor scans from the original implementation
    (which called compute_weight_scale twice + quantize_to_fp8 separately).

    Args:
        tensor: Original float tensor (before FP8 quantization).
        fp8_dtype: torch.float8_e4m3fn or torch.float8_e5m2.

    Returns:
        Tuple of (quantized_fp8_tensor, weight_scale_f32_scalar, input_scale_f32_scalar).
    """
    max_fp8 = FP8_MAX_VALUES.get(fp8_dtype, 448.0)

    # ── Single scan: float → abs().max() → scale ──
    tensor_f32 = tensor.float()
    if not torch.isfinite(tensor_f32).all():
        tensor_f32 = torch.nan_to_num(tensor_f32, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs = tensor_f32.abs().max()
    scale = max_abs / torch.tensor(max_fp8, dtype=torch.float32, device=max_abs.device)
    scale = torch.clamp(scale, min=1e-12)

    # ── Scale-then-cast (in-place on the F32 copy) ──
    tensor_f32.div_(scale)
    tensor_f32.clamp_(-max_fp8, max_fp8)
    quantized = tensor_f32.to(dtype=fp8_dtype)

    # weight_scale and input_scale are identical per Creator's convention
    wscale = scale.to(dtype=torch.float32)
    iscale = scale.to(dtype=torch.float32)  # same value, separate tensor

    return quantized, wscale, iscale


def dequant_fp8_tensor(
    tensor: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
    scale_store: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Dequantize an FP8 tensor to target dtype using companion scale factors.

    FP8 tensors store per-channel quantization scales as companion tensors
    (e.g. '.weight_scale' and '.input_scale' suffixes). This function:
      1. Casts FP8 → target dtype (e.g. bf16) — gives scaled values
      2. Looks up the per-channel scale factor from scale_store
      3. Multiplies: out = fp8_as_target * scale (broadcasted)

    Args:
        tensor: FP8 tensor (float8_e4m3fn or float8_e5m2).
        key: Checkpoint key for the tensor (used to derive scale key names).
        target_dtype: Target dtype for dequantization (e.g. bfloat16).
        scale_store: Dict-like containing companion scale tensors.
            Supports plain dict and _LazyCheckpointMapping (both have __contains__).

    Returns:
        Dequantized tensor at target_dtype. Returns tensor unchanged if no
        companion scale is found (fallback for non-quantized keys).
    """
    tensor = tensor.to(dtype=target_dtype)
    # Derive the base prefix for scale key lookup
    base = key
    if key.endswith('.weight'):
        base = key[:-len('.weight')]
    # Look for companion scale (weight_scale or input_scale)
    for suffix in ('.weight_scale', '.input_scale'):
        scale_key = base + suffix
        if scale_key in scale_store:
            scale = scale_store[scale_key].to(dtype=target_dtype)
            # Broadcast: weight [C, K1, K2, ...] * scale [C, 1, 1, ...]
            for _ in range(tensor.dim() - scale.dim()):
                scale = scale.unsqueeze(-1)
            tensor = tensor * scale
            break
    return tensor


# ══════════════════════════════════════════════════════════════════════════════
# Shared FP8 metadata helpers — used by both baker_node.py and checkpoint_merger_node.py
# Extracted here to follow "One Pattern Per Concept" (working_principles.md).
# ══════════════════════════════════════════════════════════════════════════════


def build_fp8_quantization_metadata(
    output_sd,
    metadata: dict,
    label: str = "",
) -> dict:
    """Build _quantization_metadata for MixedPrecisionOps and inject into metadata.

    Handles both _LazyCheckpointMapping (scans header/write_cache for dtype)
    and plain dicts (scans tensor values directly).

    Returns the (possibly updated) metadata dict.
    """
    fp8_dtype_str = getattr(output_sd, '_user_fp8_dtype_str', 'F8_E4M3')
    fp8_format = {'F8_E4M3': 'float8_e4m3fn', 'F8_E5M2': 'float8_e5m2'}.get(
        fp8_dtype_str, 'float8_e4m3fn')

    fp8_layers = {}

    if hasattr(output_sd, '_iter_keys_unfiltered'):
        # _LazyCheckpointMapping: check write cache and file header
        for k in output_sd._iter_keys_unfiltered():
            if k.endswith(('.weight_scale', '.input_scale')):
                continue
            if output_sd._write_cache and k in output_sd._write_cache:
                dtype = output_sd._write_cache[k].dtype
            else:
                output_sd._ensure_open()
                info = output_sd._header.get(k, {})
                dtype_str = info.get('dtype', 'BF16')
                dtype = output_sd._SAFETENSORS_DTYPE_MAP.get(dtype_str, torch.bfloat16)
            is_fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            if is_fp8:
                layer_name = k[:-len('.weight')] if k.endswith('.weight') else k
                fp8_layers[layer_name] = {'format': fp8_format}
    else:
        # Plain dict: check tensor dtype directly
        for k, v in output_sd.items():
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                if not k.endswith(('.weight_scale', '.input_scale')):
                    layer_name = k[:-len('.weight')] if k.endswith('.weight') else k
                    # Derive format from actual tensor dtype — more reliable than
                    # _user_fp8_dtype_str (which can't be set on plain dicts).
                    _layer_fmt = 'float8_e4m3fn' if v.dtype == torch.float8_e4m3fn else 'float8_e5m2'
                    fp8_layers[layer_name] = {'format': _layer_fmt}

    if fp8_layers:
        qmeta = {'format_version': '1.0', 'layers': fp8_layers}
        metadata = dict(metadata) if metadata else {}
        metadata['_quantization_metadata'] = json.dumps(qmeta)
        suffix = f" ({label})" if label else ""
        print(f"   📋 Added quantization metadata ({len(fp8_layers)} FP8 layers){suffix}")

    return metadata


def strip_input_scale_keys(output_sd) -> int:
    """Strip .input_scale keys (MixedPrecisionOps only needs .weight_scale).

    Returns the number of keys stripped.
    """
    input_scale_keys = [k for k in output_sd if k.endswith('.input_scale')]
    for k in input_scale_keys:
        output_sd.pop(k, None)
    if input_scale_keys:
        print(f"   🧹 Stripped {len(input_scale_keys)} .input_scale tensors for MixedPrecisionOps")
    return len(input_scale_keys)
