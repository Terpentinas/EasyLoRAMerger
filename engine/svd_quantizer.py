"""
SVD Quantizer — Full SVD Compression Logic for Checkpoint Studio

Provides the canonical SVD compression logic used across EasyLoRAMerger:
  - SVD_SKIP_PATTERNS         — Critical layers to skip SVD entirely
  - SVD_ENERGY_TIERS          — Sensitive layer patterns with elevated threshold
  - should_skip_svd()         — Skip SVD for critical layer types
  - get_svd_threshold()      — Tiered energy threshold (0.999 for sensitive layers)
  - apply_svd_to_tensor()    — Per-tensor SVD decomposition + reconstruction
  - apply_svd_preprocess()   — Batch SVD preprocessing before quantization

All functions are stateless pure functions operating on keys, floats, and tensors.
This module is the single source of truth; musubi_checkpoint_studio.py
delegates to it — matching the pattern of fp8_quantizer.py and int8_quantizer.py.

The patterns are validated across all six supported architectures:
  - Flux (dot-notation: double_blocks.0.img_attn.qkv.weight)
  - Anima (dot-notation: net.blocks.0.adaln_modulation_cross_attn.1.weight)
  - SDXL (diffusers: model.diffusion_model.input_blocks.0.0.weight)
  - SD1.5 (same SDXL format, no label_emb.)
  - Lumina2, Z-Image (same principles apply)
"""

from typing import Dict, Optional

import torch

try:
    from ..config import (
        SVD_MIN_DIMENSION,
        SVD_SELECTIVE_MIN_DIM,
        SVD_CLEANUP_PARAM_THRESHOLD,
    )
    from ..utils import DeviceManager, cleanup_memory, memory_guard
except ImportError:
    from config import (
        SVD_MIN_DIMENSION,
        SVD_SELECTIVE_MIN_DIM,
        SVD_CLEANUP_PARAM_THRESHOLD,
    )
    from utils import DeviceManager, cleanup_memory, memory_guard



# 🔥 SVD IMPROVEMENT: Patterns that should skip SVD compression entirely
#
# SVD compresses weight matrices by truncating small singular values.
# Signal-entry/exit layers (input projections, output projections,
# modulation layers, norm weights) are catastrophic to compress because:
#
#   - Input projections (img_in, time_in, txt_in, vector_in, guidance_in,
#     x_embedder, t_embedder): Signal entry — errors compound through ALL
#     downstream layers.
#
#   - Output projections (final_layer, out): Signal exit — errors directly
#     visible in pixels.
#
#   - First block (blocks.0, input_blocks.0, diffusion_model.blocks.0):
#     Everything flows through the first block first. Errors here corrupt
#     every subsequent block.
#
#   - Modulation layers (single_stream_modulation, double_stream_modulation_*):
#     Controls normalization scale/bias per timestep — tiny errors cause
#     large output shifts. These matrices are typically small (~256×256)
#     with uniformly distributed energy, so 0.95 threshold keeps only
#     30-50% of vectors — disastrous.
#
#   - LLM Adapter (llm_adapter.*): T5 cross-attention conditioning interface.
#     First block processes raw T5 hidden states before propagation.
#     Errors corrupt all cross-attention conditioning across every DiT block.
#
#   - Norm weights (_norm.scale, _norm.bias): Control scale/bias of
#     normalization. Tiny errors → large output shifts because norm
#     statistics are computed per-token.
#
# Patterns support two key formats (same as FP8_PRESERVE_BF16_PATTERNS):
#   - Diffusers-style: uses '/' path separators (model/diffusion_model/...)
#   - FLUX-style: uses '.' dot notation (double_blocks.0.img_attn...)
SVD_SKIP_PATTERNS: tuple = (
    # ── Diffusers-style path separators ──
    '/final_layer/',
    '/time_in/',
    '/txt_in/',
    '/vector_in/',
    '/guidance_in/',
    '/norm.',
    '/input_blocks/0/',
    '/middle_block/',
    '/out/',

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

    # ── SDXL CLIP Text Encoder — embeddings & first encoder layer ──
    # Token/position embedding tables encode semantic vocabulary.
    # SVD compression corrupts the learned token representations,
    # causing text understanding degradation.
    # Matches: conditioner.embedders.0.transformer.text_model.embeddings.*
    # Note: We do NOT add 'conditioner.embedders.' here — that broad pattern
    # matches BOTH CLIP models (embedders.0 ~123M, embedders.1 ~694M) and is
    # redundant with the two specific patterns below which protect sensitive
    # layers (token embeddings + first encoder layer) in BOTH CLIP models.
    'text_model.embeddings.',
    # First text encoder layer — signal entry for text processing.
    # Similar rationale to net.blocks.0. and input_blocks.0.
    # Matches: ...text_model.encoder.layers.0.*
    'text_model.encoder.layers.0.',

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


# 🔥 SVD ENERGY TIERS: Sensitive-but-large layers get a higher threshold
#
# Some layers are too large to skip entirely (they would defeat the purpose
# of SVD compression) but are still signal-critical enough that the default
# 0.95 threshold is too aggressive. These get 0.999 instead.
#
# The tiered threshold only applies to layers NOT matching SVD_SKIP_PATTERNS.
# If a layer matches BOTH a skip pattern and a sensitive pattern, the skip
# takes precedence (it's excluded from SVD entirely).
SVD_ENERGY_TIERS: dict = {
    # Sensitive layer patterns — signal entry/exit points that are large
    # enough to benefit from SVD but need near-full rank preservation.
    'sensitive_patterns': (
        # ── Diffusers-style path separators ──
        '/final_layer/',
        '/time_in/',
        '/txt_in/',
        '/vector_in/',
        '/guidance_in/',
        # ── FLUX dot-notation keys ──
        'final_layer.',
        'time_in.',
        'txt_in.',
        'vector_in.',
        'guidance_in.',
        'img_in.',
        # ── Anima — signal entry ──
        'net.x_embedder.',
        'net.t_embedder.',
        'net.final_layer.',
        # ── Anima — LLM Adapter first block only (signal entry for T5 cross-attn) ──
        # We do NOT set 0.999 for the entire module — only blocks.0 needs elevated
        # threshold; blocks.1–23 can safely use the default 0.95.
        'net.llm_adapter.blocks.0.',
        # ── SDXL/SD1.5 — signal entry/exit ──
        'time_embed.',
        'label_emb.',
        'out.',
        # ── SDXL output block embedding layers (time embedding projections) ──
        # Similar criticality to time_embed. but on the output side.
        'emb_layers.',
    ),
    # Threshold for sensitive layers — 0.999 preserves near-full rank
    # while still allowing some compression for truly redundant modes.
    'sensitive_threshold': 0.999,
}


# ── Public API ────────────────────────────────────────────────────────────────


def should_skip_svd(key: str) -> bool:
    """Return True if this key should skip SVD compression entirely.

    Critical signal-entry/exit layers (img_in, time_in, final_layer,
    modulation, norm weights, first block, LLM adapter entry points)
    are excluded from SVD to prevent quality degradation.

    Patterns are matched case-insensitively as a substring of the key,
    supporting both Diffusers-style (/) and FLUX/SDXL dot-notation (.).

    Args:
        key: The checkpoint key to check (e.g. ``double_blocks.0.img_attn.qkv.weight``).

    Returns:
        True if the key matches a SVD skip pattern.
    """
    key_lower = key.lower()
    for pattern in SVD_SKIP_PATTERNS:
        if pattern in key_lower:
            return True
    return False


def get_svd_threshold(key: str, default_threshold: float = 0.95) -> float:
    """Return the energy threshold for SVD compression of a given key.

    Sensitive layers (signal entry/exit projections) get a higher threshold
    (0.999) to preserve near-full rank. All other layers use the user-provided
    default threshold.

    NOTE: This function is only called for layers that did NOT match
    ``should_skip_svd()``. A layer matching both a skip pattern and a
    sensitive pattern is skipped entirely (skip takes precedence).

    Args:
        key: The checkpoint key to check.
        default_threshold: The user-specified energy threshold (default 0.95).

    Returns:
        Energy threshold to use for this key (0.999 for sensitive layers,
        ``default_threshold`` otherwise).
    """
    key_lower = key.lower()
    for pattern in SVD_ENERGY_TIERS['sensitive_patterns']:
        if pattern in key_lower:
            return SVD_ENERGY_TIERS['sensitive_threshold']
    return default_threshold


# ── Per-Tensor SVD Application ────────────────────────────────────────────────


def apply_svd_to_tensor(
    tensor: torch.Tensor,
    key: str,
    svd_mode: str,
    energy_threshold: float,
    target_device: Optional[torch.device] = None,
) -> tuple:
    """
    Apply SVD compression to a SINGLE weight tensor and return the
    SVD-reconstructed float32 tensor (or the original if not applicable).

    This is a pure function — the input tensor is not modified.

    Args:
        tensor: The weight tensor to compress (must be float dtype).
        key: The checkpoint key (e.g. ``double_blocks.0.img_attn.qkv.weight``).
        svd_mode: ``"none"``, ``"selective"``, or ``"full"``.
        energy_threshold: Energy threshold for rank selection (0.0–1.0).
        target_device: Optional GPU device for accelerated SVD.

    Returns:
        Tuple of ``(reconstructed_tensor, was_compressed, original_params, compressed_params)``.
        If the tensor does not qualify for SVD, returns ``(original_tensor, False, 0, 0)``.
    """
    # Must be a 2D weight matrix
    is_weight = key.endswith('.weight') and tensor.ndim == 2
    if not is_weight:
        return tensor, False, 0, 0

    # Skip critical signal-entry/exit layers entirely.
    if should_skip_svd(key):
        return tensor, False, 0, 0

    out_features, in_features = tensor.shape

    # Heuristic for selective mode: only compress large matrices
    if svd_mode == "selective" and (out_features <= SVD_SELECTIVE_MIN_DIM or in_features <= SVD_SELECTIVE_MIN_DIM):
        return tensor, False, 0, 0

    # Global shape guard: skip tiny matrices regardless of mode
    if out_features < SVD_MIN_DIMENSION and in_features < SVD_MIN_DIMENSION:
        return tensor, False, 0, 0

    # Skip if rank would always be full rank (no compression benefit)
    min_dim = min(out_features, in_features)
    if min_dim == 1:
        return tensor, False, 0, 0

    # Convert to float32 for SVD (always operates on float32)
    tensor_fp32 = tensor.float()

    # ── VRAM-aware per-tensor GPU SVD ──────────────────────────────────
    moved_to_gpu_for_svd = False
    if target_device is not None and target_device.type != "cpu":
        free_vram = DeviceManager.get_free_vram(target_device)
        tensor_bytes = tensor_fp32.numel() * tensor_fp32.element_size()
        needed_bytes = tensor_bytes * 3  # SVD needs ~3× tensor size
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
        return tensor, False, 0, 0

    # Clamp singular values to prevent NaN from numerically negative tiny SVs
    S = S.clamp(min=0.0)

    # Determine target rank with tiered threshold
    effective_threshold = get_svd_threshold(key, energy_threshold)
    total_energy = torch.sum(S ** 2)
    cumulative = torch.cumsum(S ** 2, dim=0)
    k = torch.searchsorted(cumulative, effective_threshold * total_energy).item() + 1
    k = min(k, len(S))
    k = max(k, 1)  # Ensure at least rank 1

    # Skip if compressed representation would be larger than original
    original_params = out_features * in_features
    compressed_params = out_features * k + k * in_features
    if compressed_params >= original_params:
        return tensor, False, 0, 0

    # Truncate and reconstruct
    sqrt_Sk = torch.sqrt(S[:k])
    reconstructed = (U[:, :k] * sqrt_Sk) @ (sqrt_Sk[:, None] * Vh[:k, :])

    # Move back to CPU if we accelerated on GPU
    if moved_to_gpu_for_svd:
        reconstructed = reconstructed.cpu()

    if svd_mode == "selective" or svd_mode == "full":
        energy_ratio = torch.sum(S[:k]**2) / total_energy
        print(f"[SVD] compressed {key}: {out_features}x{in_features} -> rank {k} (energy {energy_ratio:.3f})")

    # Delete intermediate variables to free memory (only for large tensors)
    if out_features * in_features > SVD_CLEANUP_PARAM_THRESHOLD:
        cleanup_memory(U, S, Vh, sqrt_Sk, tensor_fp32)
    memory_guard()

    return reconstructed, True, original_params, compressed_params


# ── Batch SVD Preprocessing ──────────────────────────────────────────────────


def apply_svd_preprocess(
    tensors: Dict[str, torch.Tensor],
    svd_mode: str,
    energy_threshold: float,
    target_device: Optional[torch.device] = None,
) -> dict:
    """
    Apply SVD to original float32 tensors IN PLACE before quantization.

    Each qualifying weight tensor is decomposed and reconstructed at
    reduced rank. The ``tensors`` dict is mutated: qualifying keys are
    replaced with SVD-reconstructed float32 tensors.

    Because this runs BEFORE the quantization loop, companion scales
    computed during quantization will AUTOMATICALLY reflect the
    SVD-corrected values — fixing the companion scale invalidation bug.

    Args:
        tensors: Dict of ``{key: tensor}`` to mutate in-place.
        svd_mode: ``"none"``, ``"selective"``, or ``"full"``.
        energy_threshold: Energy threshold for rank selection.
        target_device: Optional GPU device for accelerated SVD.

    Returns:
        Dict with svd_info keys for reporting:
        - svd_applied: bool
        - compressed_layers: int
        - skipped_layers: int
        - total_original_params: int
        - total_compressed_params: int
        - size_saved_mb: 0 (SVD does not change storage size)
        - avg_rank_ratio: float
    """
    if svd_mode == "none":
        return {"svd_applied": False, "compressed_layers": 0,
                "skipped_layers": 0, "size_saved_mb": 0}

    total_original_params = 0
    total_compressed_params = 0
    compressed_layers = 0
    skipped_layers = 0

    for key in list(tensors.keys()):
        tensor = tensors[key]
        result, was_compressed, orig_p, comp_p = apply_svd_to_tensor(
            tensor, key, svd_mode, energy_threshold,
            target_device=target_device,
        )
        if was_compressed:
            # Replace with SVD-reconstructed float32 tensor
            tensors[key] = result
            compressed_layers += 1
            total_original_params += orig_p
            total_compressed_params += comp_p
        else:
            skipped_layers += 1

    # Compute average rank reduction ratio
    avg_rank_ratio = 0.0
    if compressed_layers > 0 and total_original_params > 0:
        # compressed_params is low-rank: out*k + k*in
        # original_params is full: out*in
        # rank ratio ≈ compressed_params / original_params
        avg_rank_ratio = total_compressed_params / total_original_params

    return {
        "svd_applied": compressed_layers > 0,
        "compressed_layers": compressed_layers,
        "skipped_layers": skipped_layers,
        "total_original_params": total_original_params,
        "total_compressed_params": total_compressed_params,
        # SVD does NOT reduce file size when stored as INT8/FP8 full matrix.
        # Reporting 0 avoids misleading "498 MB saved" messages.
        # The rank_ratio shows the actual compression achieved.
        "size_saved_mb": 0,
        "avg_rank_ratio": avg_rank_ratio,
    }
