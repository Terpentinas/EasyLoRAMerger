"""
INT8 Quantizer — Per-Channel Symmetric Quantization for Legacy GPUs

Provides symmetric INT8 quantization optimized for Turing/Pascal GPUs where
FP8 is emulated and slow. Uses per-channel scales to minimize precision loss.

All functions are stateless pure functions operating on tensors and keys.
This module mirrors the API shape of engine/fp8_quantizer.py.
"""

from typing import Dict, Optional, Tuple

import torch

try:
    from ..config import INT8_CLIP_PERCENTILE
except ImportError:
    from config import INT8_CLIP_PERCENTILE

# Optional ConvRot (Hadamard rotation) support for INT8 quantization quality.
# Requires ComfyUI-Flux2-INT8 custom node to be installed.
# ConvRot applies an orthogonal Hadamard transform before quantization,
# spreading per-channel outliers uniformly to reduce quantization error.
_HAS_CONVROT = False
_HADAMARD_CACHE = {}  # (size, device_str, dtype_str) -> H matrix
try:
    import importlib
    # Use importlib.import_module() because the module name contains
    # hyphens ('ComfyUI-Flux2-INT8') which are not valid in Python's
    # from X import Y syntax (SyntaxError on '-' identifier).
    _convrot_mod = importlib.import_module("custom_nodes.ComfyUI-Flux2-INT8.convrot")
    build_hadamard = _convrot_mod.build_hadamard
    rotate_weight = _convrot_mod.rotate_weight
    _HAS_CONVROT = True
except (ImportError, AttributeError):
    pass


# ── INT8 skip patterns (tensors too small to benefit from INT8) ──────
# LoRA weights are tiny (typically 1-50K elements each); the per-channel
# scale overhead (~2 bytes per output channel) is comparable to the data
# savings, making INT8 quantization counterproductive.
INT8_SKIP_PATTERNS: tuple = (
    'lora.',        # All LoRA weight patterns start with a module containing 'lora'
)


# 🔥 INT8 IMPROVEMENT: Patterns that should be preserved in FP16 instead of quantized to INT8
#
# INT8 has zero mantissa bits — step size is 1.0 in integer space. After scaling,
# effective precision is `scale` per channel. Input/output projections, modulation
# layers, and norm weights are the most sensitive to quantization error because they
# handle signal entry/exit from the transformer blocks.
#
# This mirrors FP8_PRESERVE_BF16_PATTERNS from engine/fp8_quantizer.py. The key
# difference: FP8 uses BF16 (3 mantissa bits preserved), while INT8 uses FP16
# (10 mantissa bits). For INT8 dequant math (int8 * scale), FP16's higher mantissa
# precision produces more accurate results than BF16 — so FP16 is universally correct
# regardless of GPU generation (Pascal/Turing/Ampere+).
#
# Patterns support two key formats:
#   - Diffusers-style: uses '/' path separators (model/diffusion_model/...)
#   - FLUX-style: uses '.' dot notation (double_blocks.0.img_attn...)
INT8_PRESERVE_FP16_PATTERNS: tuple = (
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

    # ── SDXL CLIP Text Encoder — embeddings & first encoder layer ──
    # Token/position embedding tables encode semantic vocabulary.
    # Matches: conditioner.embedders.0.transformer.text_model.embeddings.*
    # Note: We do NOT add 'conditioner.embedders.' here — that broad pattern
    # would match BOTH CLIP models (embedders.0 ~123M, embedders.1 ~694M),
    # forcing the entire 694M-param OpenCLIP ViT-bigG to FP16 (+670 MB).
    # The two specific patterns below already protect the sensitive layers
    # (token embeddings + first encoder layer) in BOTH CLIP models.
    'text_model.embeddings.',
    # First text encoder layer — signal entry for text processing.
    'text_model.encoder.layers.0.',

    # ── Output block embedding layers (time embedding projections) ──
    # Similar criticality to time_embed. but on the output side.
    # Matches: output_blocks.*.0.emb_layers.*
    'emb_layers.',
)


# ── Public API ────────────────────────────────────────────────────────


def should_skip_int8(key: str) -> bool:
    """Return True if this key should skip INT8 quantization.

    Args:
        key: The checkpoint key to check.

    Returns:
        True if the key matches a skip pattern (e.g. LoRA weights).
    """
    key_lower = key.lower()
    for pattern in INT8_SKIP_PATTERNS:
        if pattern in key_lower:
            return True
    return False


def should_preserve_fp16_for_int8(key: str) -> bool:
    """Return True if this key should stay in FP16 when target precision is INT8.

    INT8 has zero mantissa bits — input projections (img_in, time_in, txt_in),
    output layers (final_layer), and modulation tensors are the most sensitive
    to quantization error. This mirrors FP8's ``should_preserve_bf16()`` but
    uses FP16 preservation instead of BF16.

    For INT8 dequant math (``int8 * scale``), FP16's 10 mantissa bits produce
    more accurate results than BF16's 7 mantissa bits — so FP16 is universally
    better regardless of GPU generation.

    Args:
        key: The checkpoint key to check.

    Returns:
        True if this key should be preserved in FP16 instead of INT8.
    """
    key_lower = key.lower()
    for pattern in INT8_PRESERVE_FP16_PATTERNS:
        if pattern in key_lower:
            return True
    return False


def compute_int8_scale(
    tensor: torch.Tensor,
    percentile: Optional[float] = None,
) -> torch.Tensor:
    """Per-channel INT8 scale factor per output channel.

    Computes ``threshold / 127.0`` where ``threshold`` is the per-channel
    absolute value at the given ``percentile`` (default: module-level
    :data:`INT8_CLIP_PERCENTILE`). Clipping outliers via percentile
    (e.g. 99.9th) prevents a single extreme outlier from compressing the
    effective INT8 range for the bulk of weights.

    Symmetric quantization maps the range [-threshold, +threshold] to
    [-127, +127] per output channel. Scale is clamped to a minimum of
    1e-12 to prevent division by zero for all-zero tensors.

    Handles 1D tensors (``(C,)``) by using a per-tensor (scalar) scale — there
    is no "output channel" dimension to quantize per-channel. This is a defensive
    guard; callers should route 1D weights through float16, but if they don't,
    the function still produces correct output instead of crashing.

    Args:
        tensor: Float tensor, shape ``(out_channels, ...)`` or ``(C,)``.
        percentile: If set to < 1.0, clips outliers at this percentile
            before computing scale. E.g. 0.999 = 99.9th percentile. Default
            (None) uses :data:`INT8_CLIP_PERCENTILE`. Set to 1.0 to use
            max (disable clipping).

    Returns:
        Scale tensor, dtype ``float32``. Shape is ``(out_channels,)`` for 2D+
        input, or scalar (0-dim) for 1D input.
    """
    # Use module-level default if not specified
    if percentile is None:
        percentile = INT8_CLIP_PERCENTILE

    # Handle 1D tensors: use per-tensor (scalar) scale — flatten(1) would crash
    if tensor.dim() <= 1:
        flat = tensor.float()  # shape (C,)
        is_1d = True
    else:
        # Flatten all dimensions except the first (output channel)
        flat = tensor.float().flatten(1)  # shape (C, K*...)
        is_1d = False

    # Guard: sanitize NaN/Inf -> 0 to prevent corrupted scale factors
    if not torch.isfinite(flat).all():
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    if is_1d:
        # 1D: per-tensor (scalar) scale — no channel dimension to reduce
        abs_vals = flat.abs()
        if percentile < 1.0:
            k = max(1, int(abs_vals.numel() * percentile))
            threshold = abs_vals.kthvalue(k).values
            flat = flat.clamp(-threshold, threshold)
            # Recompute abs after clipping (outliers now saturated at threshold)
            abs_vals = flat.abs()
        abs_max = abs_vals.max()
    else:
        # 2D+: per-channel scale — max across all flattened non-channel dims
        abs_vals = flat.abs()  # shape (C, K*...)
        if percentile < 1.0:
            # Per-channel percentile: independent threshold per output channel
            k = max(1, int(abs_vals.size(-1) * percentile))
            threshold = abs_vals.kthvalue(k, dim=-1).values  # shape (C,)
            # Clamp each channel independently
            flat = flat.clamp(
                -threshold.unsqueeze(-1),
                threshold.unsqueeze(-1),
            )
            # Recompute abs after clipping (outliers now saturated at threshold)
            abs_vals = flat.abs()
        abs_max = abs_vals.max(dim=1).values
    scale = abs_max / 127.0
    scale = torch.clamp(scale, min=1e-12)
    return scale.to(dtype=torch.float32)


def quantize_to_int8(
    tensor: torch.Tensor,
    percentile: Optional[float] = None,
    use_convrot: bool = False,
    convrot_group_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel INT8 quantization.

    Steps:
        1. (Optional) Apply Hadamard rotation if ``use_convrot=True``,
           spreading outlier channels to reduce quantization error.
        2. Compute per-channel scale = threshold / 127.0 (with optional
           percentile clipping for outlier reduction)
        3. Divide tensor by scale (broadcasted across non-channel dims)
        4. Round and clamp to [-127, +127]
        5. Cast to ``torch.int8``

    Args:
        tensor: Float tensor, shape ``(out_channels, ...)``.
        percentile: Passed through to :func:`compute_int8_scale`. If < 1.0,
            clips outliers at this percentile before computing scale.
            Default (None) uses :data:`INT8_CLIP_PERCENTILE`.
        use_convrot: If True, apply Hadamard rotation before quantization.
            Requires ComfyUI-Flux2-INT8 custom node to be installed.
        convrot_group_size: Group size for Hadamard matrix (must be power of 4).
            Only used when ``use_convrot=True``.

    Returns:
        Tuple of ``(quantized_int8_tensor, scale_f32_1d)``.

    Raises:
        ImportError: If ``use_convrot=True`` but ComfyUI-Flux2-INT8 is not installed.
    """
    if use_convrot:
        if not _HAS_CONVROT:
            raise ImportError(
                "ConvRot requires ComfyUI-Flux2-INT8 custom node. "
                "Install from: https://github.com/BobJohnson24/ComfyUI-INT8-Fast"
            )
        # Skip ConvRot for layers where in_features is not divisible by
        # group_size (e.g. small embedding projections like [64, 64]).
        # rotate_weight raises ValueError otherwise.
        in_features = tensor.shape[-1]
        if in_features % convrot_group_size != 0:
            print(f"   ⚠️ ConvRot: skipping layer with in_features={in_features} "
                  f"(not divisible by group_size={convrot_group_size})")
        else:
            # Cache Hadamard matrix per (size, device, dtype) to avoid recomputation
            cache_key = (convrot_group_size, str(tensor.device), str(tensor.dtype))
            if cache_key not in _HADAMARD_CACHE:
                H = build_hadamard(convrot_group_size, device=tensor.device, dtype=tensor.dtype)
                _HADAMARD_CACHE[cache_key] = H
            else:
                H = _HADAMARD_CACHE[cache_key]
            # Rotate weight: W_rot = W @ H^T (orthogonal, preserves magnitude)
            tensor = rotate_weight(tensor, H, group_size=convrot_group_size)

    scale = compute_int8_scale(tensor, percentile=percentile)
    # Reshape scale for broadcasting: [C] -> [C, 1, 1, ...]
    scale_view = scale.view(-1, *([1] * (tensor.dim() - 1)))
    q = (tensor / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_weight_to_int8_with_scales(
    tensor: torch.Tensor,
    percentile: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: quantize a weight tensor to INT8.

    Identical to :func:`quantize_to_int8` but named to mirror the FP8 API
    convention from :mod:`engine.fp8_quantizer.quantize_weight_to_fp8_with_scales`.

    Args:
        tensor: Original float tensor (before INT8 quantization).
        percentile: Passed through to :func:`quantize_to_int8`. If < 1.0,
            clips outliers at this percentile before computing scale.
            Default (None) uses :data:`INT8_CLIP_PERCENTILE`.

    Returns:
        Tuple of ``(quantized_int8_tensor, weight_scale_f32_1d)``.
    """
    return quantize_to_int8(tensor, percentile=percentile)


def dequant_int8_tensor(
    tensor: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
    scale_store: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Dequantize an INT8 tensor to target dtype using companion scale factors.

    INT8 tensors store per-channel quantization scales as companion tensors
    (``.weight_scale`` suffix, matching the FP8 convention). This function:
        1. Casts INT8 -> target dtype (e.g. fp16) — gives scaled values
        2. Looks up the per-channel scale factor from ``scale_store``
        3. Broadcasts and multiplies: out = int8_as_target * scale

    Args:
        tensor: INT8 tensor (``torch.int8``).
        key: Checkpoint key for the tensor (used to derive scale key names).
        target_dtype: Target dtype for dequantization (e.g. ``torch.float16``).
        scale_store: Dict-like containing companion scale tensors.
            Supports plain dict and ``_LazyCheckpointMapping`` (both have
            ``__contains__``).

    Returns:
        Dequantized tensor at ``target_dtype``. Returns tensor unchanged
        (cast to target_dtype) if no companion scale is found.
    """
    if tensor.dtype != torch.int8:
        return tensor.to(dtype=target_dtype) if tensor.dtype != target_dtype else tensor

    result = tensor.to(dtype=target_dtype)

    # Derive the base prefix for scale key lookup
    base = key[:-len('.weight')] if key.endswith('.weight') else key
    scale_key = base + '.weight_scale'

    if scale_key in scale_store:
        scale = scale_store[scale_key].to(dtype=target_dtype)
        # Broadcast: weight [C, K1, K2, ...] * scale [C, 1, 1, ...]
        for _ in range(result.dim() - scale.dim()):
            scale = scale.unsqueeze(-1)
        result = result * scale

    return result
