"""
Key matching: reverse key map, multi-strategy cascade.

Extracted from baking_processor.py. Contains all key matching logic:
  - ComfyUI-native reverse key map generation
  - 8 matching strategies (reverse map, exact, prefix, underscore-to-dot,
    TE2, suffix, shape+block, shape match)
  - Fused QKV resolution, structural orphan detection
  - Component categorization and diagnostics
"""

import torch
import re
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

from ..utils import ProgressTracker
from .baking_processor_constants import (
    UNDERSCORE_TO_DOT_PATTERNS,
    LORA_TO_CHECKPOINT_PREFIXES,
)
from .baking_processor_baking import _check_shape_compatible
from .baking_processor_delta import _LazyDeltaDict, _MatchedDeltas, _ShapeInfo


def _strip_tensor_suffix(key: str) -> str:
    """Strip .weight, .bias, or .alpha suffix from a tensor key.

    Shared helper used by _build_reverse_key_map(), _global_search_by_shape_block(),
    and _find_matching_keys() to normalize checkpoint keys to their base form.
    """
    for suffix in ('.weight', '.bias', '.alpha'):
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return key


# ===================================================================
# Strategy 2: Prefix-Aware Conversion
# ===================================================================

def _lora_key_to_checkpoint_key(lora_base: str) -> Optional[str]:
    """
    Convert a LoRA key to a checkpoint key by trying known prefix mappings.

    Iterates through LORA_TO_CHECKPOINT_PREFIXES and tries each mapping.
    Returns the converted key or None if no mapping applies.

    Example:
        lora_unet_input_blocks.3.1.attn1.to_k
        → model.diffusion_model.input_blocks.3.1.attn1.to_k
    """
    for lora_prefix, ckpt_prefix, _desc in LORA_TO_CHECKPOINT_PREFIXES:
        if lora_base.startswith(lora_prefix):
            rest = lora_base[len(lora_prefix):]
            return ckpt_prefix + rest
    return None


# ===================================================================
# Strategy 2.5a: Underscore-to-Dot Fuzzy Conversion
# ===================================================================

def _try_underscore_to_dot_conversion(
    lora_base: str,
    ckpt_sd: Dict[str, torch.Tensor],  # kept for API consistency with other strategies
    ckpt_base_lookup: Dict[str, str],
) -> Optional[str]:
    """
    Try fuzzy underscore-to-dot conversion patterns.

    Applies each pattern from UNDERSCORE_TO_DOT_PATTERNS to the LoRA base key.
    If the converted key exists in the checkpoint base lookup, returns the
    full checkpoint key.

    Catches common naming inconsistencies:
      attn1_to_k → attn1.to.k
      ff_net_0 → ff.net.0
      etc.
    """
    for pattern, replacement, _desc in UNDERSCORE_TO_DOT_PATTERNS:
        converted = re.sub(pattern, replacement, lora_base)
        if converted != lora_base:
            if converted in ckpt_base_lookup:
                return ckpt_base_lookup[converted]
    return None


# ===================================================================
# Strategy 2.5b: SDXL TE Conditioner Prefix (handles both te1. and te2.)
# ===================================================================

def _try_te_conditioner_prefix(
    lora_base: str,
    ckpt_sd: Dict[str, torch.Tensor],
    ckpt_base_lookup: Dict[str, str],
    lora_deltas: Dict[str, torch.Tensor],
) -> Optional[str]:
    """
    Try to match SDXL TE1/TE2 keys by testing the conditioner.embedders prefix.

    SDXL uses conditioner.embedders.{0|1}.transformer.text_model for its
    text encoders, while Kohya-style LoRAs may use te1. or te2. prefix.

    Uses shape-based disambiguation to choose between embedders.0 and
    embedders.1 when both exist.

    Returns:
        Full checkpoint key if matched, None otherwise.
    """
    # Handle both te1. and te2. prefix keys
    if not lora_base.startswith(('te1.', 'te2.')):
        return None

    # Determine embedder hint: te1. prefers embedders.0, te2. prefers embedders.1
    if lora_base.startswith('te1.'):
        prefix = 'te1.'
        embedder_hint = 0
    else:
        prefix = 'te2.'
        embedder_hint = 1

    # Extract the inner key (everything after the prefix)
    inner_key = lora_base[len(prefix):]  # e.g., 'text_model.encoder.layers.0.self_attn.k_proj'

    # Try hinted embedder first with shape check
    candidate_hint = f"conditioner.embedders.{embedder_hint}.transformer.{inner_key}"
    if candidate_hint in ckpt_base_lookup:
        full_key = ckpt_base_lookup[candidate_hint]
        if full_key in ckpt_sd:
            target_delta = lora_deltas.get(lora_base)
            if target_delta is None or _check_shape_compatible(target_delta, ckpt_sd[full_key]):
                return full_key
        else:
            return full_key

    # Try the other embedder with shape check
    other = 1 - embedder_hint
    candidate_other = f"conditioner.embedders.{other}.transformer.{inner_key}"
    if candidate_other in ckpt_base_lookup:
        full_key = ckpt_base_lookup[candidate_other]
        if full_key in ckpt_sd:
            target_delta = lora_deltas.get(lora_base)
            if target_delta is None or _check_shape_compatible(target_delta, ckpt_sd[full_key]):
                return full_key
        else:
            return full_key

    # Shape-based disambiguation: find which embedder has matching shapes
    target_delta = lora_deltas.get(lora_base)
    if target_delta is None:
        return None

    for embedder_idx in range(2):  # Try embedders.0 and embedders.1
        check_key = f"conditioner.embedders.{embedder_idx}.transformer.{inner_key}"
        full_key = ckpt_base_lookup.get(check_key)

        if full_key is not None and full_key in ckpt_sd:
            ckpt_tensor = ckpt_sd[full_key]
            if isinstance(ckpt_tensor, torch.Tensor):
                # Check if shapes are compatible
                if ckpt_tensor.shape == target_delta.shape or \
                   (ckpt_tensor.dim() == target_delta.dim() and
                    ckpt_tensor.shape[0] == target_delta.shape[0]):
                    return full_key

    return None


# ===================================================================
# Strategy 2.5d: Deterministic TE Direct-Mapping (Kohya-inspired)
# ===================================================================

def _try_deterministic_te_mapping(
    lora_base: str,
    ckpt_sd: Dict[str, torch.Tensor],  # kept for API consistency with other strategies
    ckpt_base_lookup: Dict[str, str],
    delta: torch.Tensor,
    is_shape_compatible_fn: Callable[[str, torch.Tensor], bool],
    verbose: bool = False,
) -> Optional[str]:
    """
    Deterministic TE key mapping (Kohya-inspired, Strategy 2.5d).

    Maps LoRA TE keys to checkpoint keys using known structural patterns
    rather than relying on the reverse_map or shape-based fallbacks.
    Mirrors Kohya's approach of knowing the architecture layout a priori.

    Handles:
      - SD1.5 TE: lora_te_text_model.encoder.layers.N.*
        → cond_stage_model.transformer.text_model.encoder.layers.N.*
      - SDXL TE1 (lora_te1_): lora_te1_text_model.encoder.layers.N.*
        → cond_stage_model.transformer.text_model.encoder.layers.N.*
      - SDXL TE1 (normalized te1.): te1.text_model.encoder.layers.N.*
        → conditioner.embedders.{0|1}.transformer.text_model.encoder.layers.N.*
      - SDXL TE2 (lora_te2_): lora_te2_text_model.encoder.layers.N.*
        → conditioner.embedders.{0|1}.transformer.text_model.encoder.layers.N.*

    Returns:
        Full checkpoint key if matched, None otherwise.
    """
    # Handle both raw LoRA keys (lora_te_*) and normalized keys (text_model.*, te1.)
    # After identity_normalize() → convert_sd15_diffusers_to_comfyui(), the
    # lora_te_ prefix is stripped and keys use text_model.encoder.layers.N.* format.
    # After normalize_sdxl_kohya_with_te(), keys use te1.text_model.encoder.layers.N.* format.
    if not lora_base.startswith(('lora_te_', 'lora_te1_', 'lora_te2_', 'te1.', 'te2.', 'text_model.')):
        return None

    # Extract the suffix after the prefix
    # Raw:  lora_te_text_model.encoder.layers.0.self_attn.k_proj
    #        → suffix = "text_model.encoder.layers.0.self_attn.k_proj"
    # Norm: text_model.encoder.layers.0.self_attn.k_proj
    #        → suffix = "encoder.layers.0.self_attn.k_proj"
    # Post-norm: te1.text_model.encoder.layers.0.self_attn.k_proj
    #        → suffix = "text_model.encoder.layers.0.self_attn.k_proj"
    for prefix in ('lora_te2_text_model.', 'lora_te1_text_model.',
                   'lora_te_text_model.', 'text_model.'):
        if lora_base.startswith(prefix):
            suffix = lora_base[len(prefix):]
            break
    else:
        # Check for post-normalization te1./te2. prefix (from normalize_sdxl_kohya_with_te)
        if lora_base.startswith('te1.'):
            suffix = lora_base[4:]  # Strip 'te1.' → 'text_model.encoder.layers.N.*'
        elif lora_base.startswith('te2.'):
            suffix = lora_base[4:]  # Strip 'te2.' → 'text_model.encoder.layers.N.*'
        else:
            return None

    # --- SDXL TE2 (lora_te2_) ---
    if lora_base.startswith('lora_te2_'):
        # Try embedders.0 first, then embedders.1
        for embedder_idx in ('0', '1'):
            ckpt_candidate = (
                f"conditioner.embedders.{embedder_idx}."
                f"transformer.text_model.{suffix}"
            )
            if ckpt_candidate in ckpt_base_lookup:
                full_key = ckpt_base_lookup[ckpt_candidate]
                if is_shape_compatible_fn(full_key, delta):
                    if verbose:
                        print(f"      🎯 Deterministic TE2 (embedder {embedder_idx}): "
                              f"{lora_base} → {full_key}")
                    return full_key
        return None

    # --- SDXL TE1 (normalized te1. prefix) → conditioner.embedders ---
    # After normalize_sdxl_kohya_with_te(), the key uses te1. prefix.
    # In SDXL, both CLIP-L (TE1) and CLIP-G (TE2) live under conditioner.embedders.*.
    # Try embedders.0 first (primary CLIP-L placement), then embedders.1.
    if lora_base.startswith('te1.'):
        for embedder_idx in ('0', '1'):
            ckpt_candidate = (
                f"conditioner.embedders.{embedder_idx}."
                f"transformer.text_model.{suffix}"
            )
            if ckpt_candidate in ckpt_base_lookup:
                full_key = ckpt_base_lookup[ckpt_candidate]
                if is_shape_compatible_fn(full_key, delta):
                    if verbose:
                        print(f"      🎯 Deterministic TE1 (embedder {embedder_idx}): "
                              f"{lora_base} → {full_key}")
                    return full_key
        return None

    # --- SDXL TE2 (normalized te2. prefix) → conditioner.embedders ---
    if lora_base.startswith('te2.'):
        for embedder_idx in ('0', '1'):
            ckpt_candidate = (
                f"conditioner.embedders.{embedder_idx}."
                f"transformer.text_model.{suffix}"
            )
            if ckpt_candidate in ckpt_base_lookup:
                full_key = ckpt_base_lookup[ckpt_candidate]
                if is_shape_compatible_fn(full_key, delta):
                    if verbose:
                        print(f"      🎯 Deterministic TE2 (embedder {embedder_idx}): "
                              f"{lora_base} → {full_key}")
                    return full_key
        return None

    # --- SD1.5 TE / SDXL TE1 (lora_te_ or lora_te1_) ---
    ckpt_candidate = f"cond_stage_model.transformer.text_model.{suffix}"
    if ckpt_candidate in ckpt_base_lookup:
        full_key = ckpt_base_lookup[ckpt_candidate]
        if is_shape_compatible_fn(full_key, delta):
            if verbose:
                print(f"      🎯 Deterministic TE: {lora_base} → {full_key}")
            return full_key

    return None


# ===================================================================
# Strategy 0: Reverse Key Map (ComfyUI-native, exhaustive)
# ===================================================================

def _build_reverse_key_map(ckpt_sd: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """
    Build a reverse key map: for each possible LoRA key variant, map it
    to the corresponding checkpoint base key (without .weight suffix).

    This mirrors ComfyUI's model_lora_keys_unet() and model_lora_keys_clip()
    logic. For each checkpoint key, it generates ALL known LoRA naming
    variants and registers them in the map.

    Architecture support:
      - SDXL UNet (model.diffusion_model.input_blocks.*)
      - SDXL TE1 (cond_stage_model.transformer.text_model.*)
      - SDXL TE2 (conditioner.embedders.*.transformer.text_model.*)
      - SD1.5 UNet (model.diffusion_model.input_blocks.*)
      - SD1.5 TE (cond_stage_model.transformer.text_model.*)
      - Flux (model.diffusion_model.transformer.double_blocks.* / single_blocks.*)
      - Z-Image / Musubi (model.diffusion_model.layers.*)
      - Anima (model.* with long block prefixes)

    Returns:
        Dict mapping lora_key_variant -> ckpt_base_key (without .suffix)
    """
    reverse_map: Dict[str, str] = {}

    # Pre-process: collect all base keys (without .weight/.bias/.alpha suffix)
    ckpt_bases: List[str] = []
    for k in ckpt_sd.keys():
        base = _strip_tensor_suffix(k)
        if base not in ckpt_bases:
            ckpt_bases.append(base)

    # ================================================================
    # Section 1: Standard SDXL UNet entries (model.diffusion_model.*)
    # ================================================================
    for base_key in ckpt_bases:
        # Skip non-UNet keys early
        if not base_key.startswith('model.diffusion_model.'):
            continue

        # Strip the 'model.' prefix to get the inner path
        inner = base_key[len('model.'):]  # diffusion_model.input_blocks.3.1.attn1.to.k

        # Register direct variant (diffusion_model.*)
        reverse_map[inner] = base_key

        # Register without any prefix (pure path)
        pure_path = inner[len('diffusion_model.'):]  # input_blocks.3.1.attn1.to.k
        reverse_map[pure_path] = base_key

        # Register 'lora_unet_' prefix variant (Kohya style)
        reverse_map[f"lora_unet_{pure_path}"] = base_key

        # Also try underscore variant for the pure path
        underscorized = pure_path.replace('.', '_')
        if underscorized != pure_path:
            reverse_map[underscorized] = base_key
            reverse_map[f"lora_unet_{underscorized}"] = base_key


    # ================================================================
    # Section 2a: SDXL / SD1.5 TE entries (cond_stage_model.transformer.*)
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith('cond_stage_model.transformer.'):
            continue

        inner = base_key[len('cond_stage_model.'):]  # transformer.text_model.encoder.layers.0...
        pure_te = base_key[len('cond_stage_model.transformer.'):]  # text_model.encoder.layers.0...

        # SD1.5: text_model.* → direct
        reverse_map[pure_te] = base_key
        reverse_map[inner] = base_key

        # Kohya: lora_te_ prefix
        reverse_map[f"lora_te_{pure_te}"] = base_key
        reverse_map[f"lora_te1_{pure_te}"] = base_key

        # te. prefix, te1. prefix
        reverse_map[f"te.{pure_te}"] = base_key
        reverse_map[f"te1.{pure_te}"] = base_key

        # Underscore variants
        underscorized = pure_te.replace('.', '_')
        if underscorized != pure_te:
            reverse_map[underscorized] = base_key
            reverse_map[f"lora_te_{underscorized}"] = base_key
            reverse_map[f"lora_te1_{underscorized}"] = base_key  # ← Item 4: missing te1 variant

    # ================================================================
    # Section 2a-ii: CLIP-L / CLIP-G text encoder entries
    # SDXL dual-TE: clip_l.text_model.* and clip_g.text_model.*
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith(('clip_l.text_model.', 'clip_g.text_model.')):
            continue

        # clip_l.text_model.encoder.layers.0.self_attn.k_proj
        pure_clip = base_key[len('clip_l.'):]  # text_model.encoder.layers.0...

        # Direct: text_model.* (shared with Section 2a pure_te path)
        reverse_map[pure_clip] = base_key

        # lora_te_ prefix (some trainers use lora_te_clip_l_text_model_...)
        reverse_map[f"lora_te_{pure_clip}"] = base_key

        # te. prefix
        reverse_map[f"te.{pure_clip}"] = base_key

        # Original prefix variants
        reverse_map[base_key] = base_key

        # lora_te_clip_l_ / lora_te_clip_g_ prefix variant
        clip_prefix = base_key.split('.')[0]  # clip_l or clip_g
        reverse_map[f"lora_te_{clip_prefix}_{pure_clip}"] = base_key

        # Underscore variants
        underscorized = pure_clip.replace('.', '_')
        if underscorized != pure_clip:
            reverse_map[underscorized] = base_key
            reverse_map[f"lora_te_{underscorized}"] = base_key

    # ================================================================
    # Section 2b: SDXL TE2 entries (conditioner.embedders.*)
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith('conditioner.embedders.'):
            continue

        # Extract embedder index
        # conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.k_proj
        parts = base_key.split('.')
        if len(parts) >= 4 and parts[2].isdigit():
            embedder_idx = parts[2]
            inner_key = '.'.join(parts[3:])  # transformer.text_model.encoder.layers.0...
            pure_te2 = '.'.join(parts[4:])  # text_model.encoder.layers.0...

            # Direct: conditioner.embedders.N.* → conditioner.embedders.N.*
            reverse_map[f"conditioner.embedders.{embedder_idx}.{inner_key}"] = base_key

            # te2. prefix
            reverse_map[f"te2.{pure_te2}"] = base_key

            # te1. prefix (some SDXL LoRAs use te1. for the first embedder)
            reverse_map[f"te1.{pure_te2}"] = base_key

            # lora_te2_ prefix
            reverse_map[f"lora_te2_{pure_te2}"] = base_key

            # conditioner prefix (without embedder index — generic)
            reverse_map[f"conditioner.{inner_key}"] = base_key

    # ================================================================
    # Section 2c: T5 text encoder entries (t5xxl.transformer.*)
    # Used by Flux and other architectures with T5 text encoder.
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith('t5xxl.transformer.'):
            continue

        # t5xxl.transformer.encoder.layers.0.self_attn.k_proj
        # Strip 't5xxl.' prefix to get the transformer path
        inner = base_key[len('t5xxl.'):]  # transformer.encoder.layers.0.self_attn.k_proj

        # Direct: transformer.encoder.layers.*
        reverse_map[inner] = base_key

        # Pure path without transformer prefix: encoder.layers.0.self_attn.k_proj
        pure_t5 = base_key[len('t5xxl.transformer.'):]  # encoder.layers.0.self_attn.k_proj
        reverse_map[pure_t5] = base_key

        # lora_te_ prefix variant
        reverse_map[f"lora_te_{pure_t5}"] = base_key

        # te. prefix variant (some trainers use te.t5xxl.transformer.*)
        reverse_map[f"te.{pure_t5}"] = base_key

        # Underscore variants
        underscorized = pure_t5.replace('.', '_')
        if underscorized != pure_t5:
            reverse_map[underscorized] = base_key
            reverse_map[f"lora_te_{underscorized}"] = base_key
            reverse_map[f"te_{underscorized}"] = base_key

    # ================================================================
    # Section 3: Flux entries (model.diffusion_model.transformer.*)
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith('model.diffusion_model.transformer.'):
            continue

        inner = base_key[len('model.diffusion_model.'):]  # transformer.double_blocks.0.attn.to_q

        # Direct: transformer.*
        reverse_map[inner] = base_key

        # Strip 'model.diffusion_model.' prefix
        pure_flux = base_key[len('model.diffusion_model.transformer.'):]  # double_blocks.0.attn.to_q
        reverse_map[pure_flux] = base_key

        # lora_unet_ prefix
        reverse_map[f"lora_unet_{pure_flux}"] = base_key
        # lora_unet_transformer.* variant (rare Kohya/Flux convention)
        reverse_map[f"lora_unet_transformer.{pure_flux}"] = base_key

        # Flux Klein convention: model.diffusion_model.transformer. → (some use different prefixes)
        # Klein models may use 'transformer.' directly
        reverse_map[f"transformer.{pure_flux}"] = base_key

    # ================================================================
    # Section 4: Z-Image / Musubi entries (model.diffusion_model.layers.*)
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith('model.diffusion_model.layers.'):
            continue

        inner = base_key[len('model.diffusion_model.'):]  # layers.0.attention.to_q
        pure_zi = base_key[len('model.diffusion_model.layers.'):]  # 0.attention.to_q

        # Direct: layers.*
        reverse_map[inner] = base_key
        reverse_map[pure_zi] = base_key

        # lora_unet_ prefix
        reverse_map[f"lora_unet_{inner}"] = base_key
        reverse_map[f"lora_unet_layers_{pure_zi}"] = base_key

    # ================================================================
    # Section 4b: Bare Z-Image / Musubi entries (layers.* without model prefix)
    # Some LoRAs may skip the model.diffusion_model prefix entirely for Z-Image
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith('layers.'):
            continue
        # Already registered by Section 4?
        # But catch keys that are NOT under model.diffusion_model.layers
        # e.g., standalone 'layers.0.attention.to_q'
        reverse_map[base_key] = base_key

    # ================================================================
    # Section 4c: Flux transformer entries (transformer.double_blocks.*, transformer.single_blocks.*)
    # Flux checkpoints use bare transformer.* keys without model.diffusion_model prefix.
    # These need to be registered so LoRA keys (which may have diffusion_model. prefix
    # after normalization) can find their checkpoint counterparts.
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith('transformer.'):
            continue
        
        # Register bare key (for LoRAs that use raw Flux key naming)
        reverse_map[base_key] = base_key
        
        # Register with model.diffusion_model prefix (for LoRAs normalized to ComfyUI format)
        reverse_map[f"model.diffusion_model.{base_key}"] = base_key
        
        # Register with diffusion_model prefix (shorthand variant)
        reverse_map[f"diffusion_model.{base_key}"] = base_key
        
        # Also register 'lora_unet_' prefix variant for ComfyUI-style LoRA keys
        pure_path = base_key[len('transformer.'):]  # double_blocks.0.img_attn.qkv
        reverse_map[f"lora_unet_{pure_path}"] = base_key

    # ================================================================
    # Section 4d: Flux keys with diffusion_model.{pure_path} prefix (Klein normalized format)
    # After identity_normalize() → normalize_all_klein_formats(), Flux Klein LoRA
    # keys use diffusion_model.double_blocks.* / diffusion_model.single_blocks.*
    # format (without transformer. sub-path). Map these to the checkpoint's
    # transformer.double_blocks.* / transformer.single_blocks.* keys.
    #
    # Example:
    #   LoRA (normalized):  diffusion_model.double_blocks.0.img_attn.qkv
    #   Checkpoint:         transformer.double_blocks.0.img_attn.qkv
    #   Reverse map entry:  diffusion_model.double_blocks.0.img_attn.qkv
    #                       → transformer.double_blocks.0.img_attn.qkv
    #
    # NOTE: This only applies to vanilla Flux checkpoints that use the
    #       transformer. prefix. For Klein checkpoints which use bare
    #       double_blocks.* / single_blocks.* keys, see Section 4e below.
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith('transformer.'):
            continue
        pure_path = base_key[len('transformer.'):]  # double_blocks.0.img_attn.qkv
        
        # Map diffusion_model.{pure_path} (Klein normalized format)
        reverse_map[f"diffusion_model.{pure_path}"] = base_key
        
        # Map model.diffusion_model.{pure_path} (full prefix variant)
        reverse_map[f"model.diffusion_model.{pure_path}"] = base_key

    # ================================================================
    # Section 4e: Flux bare double_blocks/single_blocks keys (Klein 4B/9B format)
    # Klein checkpoints store keys WITHOUT the transformer. prefix, using bare
    # double_blocks.* / single_blocks.* format. Normalized LoRA keys use
    # diffusion_model.double_blocks.* / diffusion_model.single_blocks.* which
    # need to map to these bare checkpoint keys.
    #
    # Example:
    #   LoRA (normalized):  diffusion_model.double_blocks.0.img_attn.qkv
    #   Checkpoint:         double_blocks.0.img_attn.qkv
    #   Reverse map entry:  diffusion_model.double_blocks.0.img_attn.qkv
    #                       → double_blocks.0.img_attn.qkv
    # ================================================================
    for base_key in ckpt_bases:
        if not base_key.startswith(('double_blocks.', 'single_blocks.')):
            continue
        
        # Map diffusion_model.{base_key} (Klein normalized LoRA format)
        reverse_map[f"diffusion_model.{base_key}"] = base_key
        
        # Map model.diffusion_model.{base_key} (full prefix variant)
        reverse_map[f"model.diffusion_model.{base_key}"] = base_key
        
        # Map lora_unet_{base_key} (ComfyUI-style LoRA key format)
        reverse_map[f"lora_unet_{base_key}"] = base_key

    # ================================================================
    # Section 5: Bare layers (generic fallback for un-prefixed keys)
    # ================================================================
    # Handle keys that don't have any standard prefix
    for base_key in ckpt_bases:
        if base_key.startswith(('model.', 'diffusion_model.', 'cond_stage_model.',
                                'conditioner.', 'transformer.', 'layers.',
                                'double_blocks.', 'single_blocks.')):
            continue

        # Bare key: register as-is
        reverse_map[base_key] = base_key

    # ================================================================
    # Section 6: Anima block entries (long block prefixes)
    # ================================================================
    for base_key in ckpt_bases:
        # Anima blocks use long descriptive prefixes that don't fit
        # standard patterns. They're typically registered as bare keys
        # by Section 5, but we also add lora_unet_ variants.
        if base_key.startswith('model.diffusion_model.'):
            inner = base_key[len('model.diffusion_model.'):]
            if not inner.startswith(('input_blocks.', 'output_blocks.', 'middle_block.',
                                     'transformer.', 'layers.')):
                # Anima-style block path
                reverse_map[f"lora_unet_{inner}"] = base_key

    # ================================================================
    # Additional: Anima llm_adapter entries
    # ================================================================
    for base_key in ckpt_bases:
        if 'llm_adapter' in base_key or 'llama' in base_key:
            reverse_map[base_key] = base_key

    return reverse_map


# ===================================================================
# Block ID Extraction (used by multiple strategies)
# ===================================================================

def _is_spatial_depth_block(block_id: str) -> bool:
    """
    Check if a block ID represents a UNet spatial depth level.

    Spatial depth blocks (input_blocks, output_blocks) have resolution
    levels where adjacent blocks represent different spatial scales.
    Cross-depth redirection between non-adjacent levels causes Picasso
    effect (pattern from one resolution appears at wrong resolution).

    NOT spatial depth blocks:
      - double_blocks (Flux) — same resolution throughout
      - single_blocks (Flux) — same resolution throughout
      - encoder.layers (TE) — not resolution-based
      - middle_block — single block, no depth variation
    """
    return block_id.startswith('input_blocks.') or block_id.startswith('output_blocks.')


def _extract_block_id(key: str) -> Optional[str]:
    """
    Extract a canonical 'block ID' from a model key.

    The block ID identifies which structural block a tensor belongs to,
    abstracting away the specific tensor function (weight, bias, etc.).

    For UNet keys (input_blocks/output_blocks/middle_block):
        input_blocks.3.1.attn1  → input_blocks.3.1.attn1
    For TE encoder layers:
        encoder.layers.0.self_attn → encoder.layers.0.self_attn
    For Flux blocks:
        double_blocks.0.attn → double_blocks.0.attn

    Strips model prefixes (model.diffusion_model., model.text_model., etc.)
    and tensor suffixes (.weight, .bias).
    """
    if key is None:
        return None

    # Strip model prefixes
    k = key
    if k.startswith('model.diffusion_model.'):
        k = k[len('model.diffusion_model.'):]
    elif k.startswith('model.text_model.'):
        k = k[len('model.text_model.'):]
    elif k.startswith('model.'):
        k = k[len('model.'):]
    elif k.startswith('diffusion_model.'):
        k = k[len('diffusion_model.'):]

    # For Flux transformer blocks
    if k.startswith('transformer.'):
        k = k[len('transformer.'):]

    # For conditioner embedders
    if k.startswith('conditioner.embedders.'):
        parts = k.split('.')
        if len(parts) >= 4 and parts[2].isdigit():
            k = '.'.join(parts[3:])

    # For conditioner transformer.text_model prefix
    if k.startswith('transformer.text_model.'):
        k = k[len('transformer.text_model.'):]

    # For text_model prefix (from conditioner)
    if k.startswith('text_model.'):
        k = k[len('text_model.'):]

    parts = k.split('.')

    # For input_blocks/output_blocks/middle_block (UNet):
    if parts[0] in ('input_blocks', 'output_blocks', 'middle_block'):
        if len(parts) >= 3 and parts[1].isdigit():
            block_type = parts[0]
            block_num = parts[1]
            rest = parts[2:]
            sub_block = parts[2] if parts[2].isdigit() else ''
            component = ''
            for p in rest:
                if p in ('attn1', 'attn2', 'mlp', 'fc', 'proj', 'conv'):
                    component = p
                    break
            if component:
                if sub_block:
                    return f"{block_type}.{block_num}.{sub_block}.{component}"
                return f"{block_type}.{block_num}.{component}"
            return f"{block_type}.{block_num}"

    # For TE encoder layers:
    if parts[0] == 'encoder' and len(parts) >= 3:
        if parts[1] == 'layers' and parts[2].isdigit():
            layer_num = parts[2]
            component = parts[3] if len(parts) > 3 else ''
            return f"encoder.layers.{layer_num}.{component}"

    # For Flux double/single blocks:
    if parts[0] in ('double_blocks', 'single_blocks') and len(parts) >= 2:
        if parts[1].isdigit():
            block_num = parts[1]
            component = parts[2] if len(parts) > 2 else ''
            return f"{parts[0]}.{block_num}.{component}"

    # Fallback: just return first 3 segments
    if len(parts) >= 3:
        return '.'.join(parts[:3])
    return '.'.join(parts) if parts else None


# ===================================================================
# Shape+Block Index Builder (O(1) lookup for Strategy 4)
# ===================================================================

def _build_shape_block_index(
    ckpt_sd: Dict[str, torch.Tensor],
) -> Dict[Tuple[Tuple[int, ...], str], List[str]]:
    """
    Build index: (shape_tuple, block_type) -> [full_ckpt_keys].

    Enables O(1) candidate lookup in _global_search_by_shape_block()
    instead of O(N) scan over all checkpoint keys.

    Only indexes keys where _extract_block_id() returns a valid block ID.
    Keys with unrecognizable structure (e.g., embeddings, scalars) are
    excluded — they would never match any LoRA key via shape+block anyway.

    Model support:
      - SDXL:     input_blocks.*, output_blocks.*, middle_block.*
      - SD1.5:    input_blocks.*, output_blocks.*, middle_block.* (same format)
      - Flux:     double_blocks.*, single_blocks.*
      - Z-Image:  layers.* (via fallback block_id)
      - Anima:    custom prefixes (via fallback block_id)
    """
    index: Dict[Tuple[Tuple[int, ...], str], List[str]] = {}
    for ckpt_key, tensor in ckpt_sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = tuple(tensor.shape)
        base = _strip_tensor_suffix(ckpt_key)
        block_id = _extract_block_id(base)
        if block_id is None:
            continue
        block_type = block_id.split('.')[0]  # 'input_blocks', 'double_blocks', 'layers', etc.
        key = (shape, block_type)
        index.setdefault(key, []).append(ckpt_key)  # store FULL key (with suffix)
    return index


def _build_shape_block_index_from_header(
    header: Dict[str, Any],
) -> Dict[Tuple[Tuple[int, ...], str], List[str]]:
    """
    Build shape+block index from safetensors header dict (no tensor loading).
    
    Reads shapes from the header JSON instead of loading tensor data.
    Memory: ~50 KB for 425 keys vs 17+ GiB for full tensor load.
    """
    index: Dict[Tuple[Tuple[int, ...], str], List[str]] = {}
    for ckpt_key, info in header.items():
        if not isinstance(info, dict) or 'shape' not in info:
            continue
        shape = tuple(info['shape'])
        base = _strip_tensor_suffix(ckpt_key)
        block_id = _extract_block_id(base)
        if block_id is None:
            continue
        block_type = block_id.split('.')[0]
        key = (shape, block_type)
        index.setdefault(key, []).append(ckpt_key)
    return index


def _build_shape_index(
    ckpt_sd: Dict[str, torch.Tensor],
) -> Dict[Tuple[int, ...], List[str]]:
    """
    Build index: shape_tuple -> [full_ckpt_keys].

    Used by Strategy 5 (shape match last resort) for O(1) lookup.
    
    NOTE: When ckpt_sd is a _LazyCheckpointMapping, this calls .items() which
    loads ALL tensors. For mmap mode, use _build_shape_index_from_header() instead.
    """
    index: Dict[Tuple[int, ...], List[str]] = {}
    for ckpt_key, tensor in ckpt_sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = tuple(tensor.shape)
        index.setdefault(shape, []).append(ckpt_key)
    return index


def _build_shape_index_from_header(
    header: Dict[str, Any],
) -> Dict[Tuple[int, ...], List[str]]:
    """
    Build shape index from safetensors header dict (no tensor loading).
    
    The header maps key -> {dtype, shape, data_offsets}. This reads shapes
    directly from the header without loading any tensor data.
    Memory: ~50 KB for 425 keys vs 17+ GiB for full tensor load.
    """
    index: Dict[Tuple[int, ...], List[str]] = {}
    for ckpt_key, info in header.items():
        if not isinstance(info, dict) or 'shape' not in info:
            continue
        shape = tuple(info['shape'])
        index.setdefault(shape, []).append(ckpt_key)
    return index


# ===================================================================
# Strategy 4: Global Search by Shape + Block ID
# ===================================================================

def _global_search_by_shape_block(
    lora_base: str,
    delta: torch.Tensor,
    ckpt_sd: Dict[str, torch.Tensor],
    ckpt_base_lookup: Dict[str, str],
    matched_keys: Set[str],
    shape_block_index: Dict[Tuple[Tuple[int, ...], str], List[str]],  # NEW
) -> Optional[str]:
    """
    Global search: match an orphan LoRA key to a checkpoint key by
    tensor shape + block ID similarity.

    Uses pre-computed shape_block_index for O(1) candidate filtering
    instead of scanning all checkpoint keys.

    Pipeline:
      1. Extract the block ID from the LoRA key
      2. Look up candidates by (delta_shape, block_type) in index
      3. Among filtered candidates, rank by block ID similarity
      4. Return the best match not already used

    Matching logic is IDENTICAL to the original — only the lookup changes.
    """
    delta_shape = delta.shape
    if delta.dim() < 2:
        return None

    lora_block_id = _extract_block_id(lora_base)
    if lora_block_id is None:
        return None

    block_type = lora_block_id.split('.')[0]
    candidates: List[Tuple[str, float]] = []

    # O(1) lookup instead of O(N) scan over all checkpoint keys
    candidate_keys = shape_block_index.get((delta_shape, block_type), [])

    for ckpt_key in candidate_keys:
        ckpt_tensor = ckpt_sd[ckpt_key]
        if not isinstance(ckpt_tensor, torch.Tensor):
            continue
        if ckpt_key in matched_keys:
            continue
        # Shape already matches by index — no need to check again

        ckpt_base = _strip_tensor_suffix(ckpt_key)

        ckpt_block_id = _extract_block_id(ckpt_base)
        if ckpt_block_id is None:
            continue

        # Simple string overlap score
        lora_parts = set(lora_block_id.split('.'))
        ckpt_parts = set(ckpt_block_id.split('.'))
        overlap = len(lora_parts & ckpt_parts)
        total = len(lora_parts | ckpt_parts)
        similarity = overlap / total if total > 0 else 0.0

        # Bonus: same block type prefix
        if lora_block_id.split('.')[0] == ckpt_block_id.split('.')[0]:
            similarity += 0.3

        # Bonus: same numeric block index
        lora_nums = [p for p in lora_block_id.split('.') if p.isdigit()]
        ckpt_nums = [p for p in ckpt_block_id.split('.') if p.isdigit()]
        if lora_nums and ckpt_nums and lora_nums[0] == ckpt_nums[0]:
            similarity += 0.2

        # 🚫 Architecture-gated cross-depth rejection (F1)
        if _is_spatial_depth_block(lora_block_id):
            if lora_nums and ckpt_nums:
                offset = abs(int(ckpt_nums[0]) - int(lora_nums[0]))
                if offset >= 2:
                    continue
                elif offset == 1:
                    similarity -= 0.3

        # 🔧 Shape proportion compatibility check (F2)
        if delta.dim() == 2 and ckpt_tensor.dim() == 2:
            out_ratio = max(delta.shape[0], ckpt_tensor.shape[0]) / max(min(delta.shape[0], ckpt_tensor.shape[0]), 1)
            if out_ratio > 2.0:
                continue

        # 🔧 VRAM-1: Early-exit at >0.8 similarity — sufficiently good match found,
        # no need to scan remaining checkpoint keys.
        if similarity > 0.8:
            ckpt_base = _strip_tensor_suffix(ckpt_key)
            if ckpt_base in ckpt_base_lookup:
                return ckpt_base_lookup[ckpt_base]
            return ckpt_key

        candidates.append((ckpt_key, similarity))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_key, best_score = candidates[0]

    if best_score > 0.3:
        ckpt_base = _strip_tensor_suffix(best_key)
        if ckpt_base in ckpt_base_lookup:
            return ckpt_base_lookup[ckpt_base]
        return best_key

    return None


# ===================================================================
# Key Categorization
# ===================================================================

def _categorize_lora_key(lora_base: str) -> str:
    """
    Categorize a normalized LoRA base key into a component.

    Delegates to :func:`engine.key_utils.categorize_key`.

    Returns one of: ``'unet'``, ``'te'``, ``'clip'``, ``'vae'``
    (falls back to ``'unet'`` if :func:`categorize_key` returns ``'other'``).
    """
    from ..engine.key_utils import categorize_key as _cat
    comp = _cat(lora_base)
    return comp if comp != 'other' else 'unet'


# ===================================================================
# Diagnostic: Effect Bar Visualization
# ===================================================================

def _create_effect_bar(mean_percent: float, max_percent: float, width: int = 20) -> str:
    """Build a compact visual bar representing the effect magnitude."""
    if not math.isfinite(mean_percent) or not math.isfinite(max_percent):
        return "████████████████████ mean=NaN max=NaN"
    filled = int((mean_percent / 5.0) * width)
    filled = max(0, min(filled, width))
    remaining = width - filled
    bar = '█' * filled + '▁' * remaining
    return f"{bar} mean={mean_percent}% max={max_percent}%"



# ===================================================================
# Main Matching Entry Point (instance method)
# ===================================================================

def _find_matching_keys(
    self,
    ckpt_sd: Dict[str, torch.Tensor],
    lora_deltas: Dict[str, torch.Tensor],
    return_stats: bool = False,
    ckpt_header: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Match LoRA delta keys to checkpoint keys using ComfyUI-native reverse
    key map as the primary strategy, with multiple fallback layers.

    After identity_normalize() converts LoRA keys to master format,
    this method tries a cascade of strategies:

    PRIMARY:
      0. **Reverse Key Map** — Build a map from each checkpoint key
         to ALL possible LoRA naming variants.

    FALLBACK (kept for edge cases):
      1. Exact match (normalized lookup without .weight suffix)
      2. Prefix-aware conversion
      2.5a. Underscore-to-dot fuzzy conversion
      2.5b. SDXL TE2 conditioner prefix
      2.5d. Deterministic TE Direct-Mapping (Kohya-inspired)
      2.5e. Post-normalization TE Direct Mapping
      3. Suffix match (lowercase, normalized)
      4. Global Search by Shape + Block ID
      5. Shape match (last resort)

    NOTE: This function receives `self` for access to self.device, self._verbose, etc.

    When `ckpt_header` is provided, shape/block indices are built from the
    header JSON (no tensors loaded) instead of from the full state dict.
    This enables the Survey→Match→Weave pipeline where matching happens
    after only header reading (~50 KB instead of 17+ GiB for Klein 9B).
    """
    print("   🔗 Matching LoRA keys to checkpoint keys...")

    # Quick diagnostic: count keys by prefix pattern
    sorted_ckpt_keys = sorted(ckpt_sd.keys())
    model_prefix = sum(1 for k in sorted_ckpt_keys if k.startswith('model.'))
    diff_model_prefix = sum(1 for k in sorted_ckpt_keys if k.startswith('diffusion_model.'))
    layers_prefix = sum(1 for k in sorted_ckpt_keys if 'layers.' in k)
    print(f"   📊 Checkpoint key prefixes: model.*={model_prefix}, diffusion_model.*={diff_model_prefix}, layers.*={layers_prefix}")

    matched: Dict[str, torch.Tensor] = {}
    unmatched_lora: List[str] = []

    # Build normalized lookup: key_without_suffix -> original_key
    ckpt_base_lookup: Dict[str, str] = {}
    for k in ckpt_sd.keys():
        base = _strip_tensor_suffix(k)
        suffix_found = next((s for s in ('.weight', '.bias', '.alpha') if k.endswith(s)), None)
        if base in ckpt_base_lookup:
            existing_key = ckpt_base_lookup[base]
            if suffix_found == '.weight' and not existing_key.endswith('.weight'):
                ckpt_base_lookup[base] = k
        else:
            ckpt_base_lookup[base] = k

    # Build lowercase lookup for suffix matching
    ckpt_base_lower: Dict[str, str] = {}
    for base, orig_key in ckpt_base_lookup.items():
        ckpt_base_lower[base.lower()] = orig_key

    # Build suffix index for O(1) candidate filtering in Strategy 3
    # Indexes checkpoint base keys by their last path segment (after final '.')
    # so that suffix matching only iterates relevant candidates instead of
    # scanning all checkpoint keys per unmatched LoRA key.
    ckpt_suffix_index: Dict[str, List[str]] = {}
    for base_lower in ckpt_base_lower:
        suffix_key = base_lower.rsplit('.', 1)[-1]
        ckpt_suffix_index.setdefault(suffix_key, []).append(base_lower)

    # Build shape+block index for O(1) Strategy 4 lookup
    # Use header-based indices when available (no tensor loading needed)
    if ckpt_header is not None:
        shape_block_index = _build_shape_block_index_from_header(ckpt_header)
    else:
        shape_block_index = _build_shape_block_index(ckpt_sd)

    # Build shape index for O(1) Strategy 5 lookup
    if ckpt_header is not None:
        shape_index = _build_shape_index_from_header(ckpt_header)
    else:
        shape_index = _build_shape_index(ckpt_sd)

    # =====================================================================
    # PHASE 2: Build reverse key map (ComfyUI-native, exhaustive)
    # =====================================================================
    reverse_map = _build_reverse_key_map(ckpt_sd)

    # =====================================================================
    # 🔍 DIAGNOSTIC: Comprehensive reverse_map coverage (runs always)
    # =====================================================================
    print(f"   📊 Reverse map entries: {len(reverse_map)}")

    if lora_deltas:
        # Count ALL keys NOT in reverse_map
        rmap_misses_total = sum(1 for k in lora_deltas if k not in reverse_map)
        rmap_hits_total = len(lora_deltas) - rmap_misses_total
        print(f"   📊 Reverse map coverage: {rmap_hits_total}/{len(lora_deltas)} keys in map ({rmap_misses_total} misses)")

        # Special focus on diffusion_model.* keys (Flux Klein normalized format)
        diffusion_keys = [k for k in lora_deltas if k.startswith('diffusion_model.')]
        if diffusion_keys:
            diffusion_misses = [k for k in diffusion_keys if k not in reverse_map]
            diffusion_hits = len(diffusion_keys) - len(diffusion_misses)
            rmap_iter_msg = "❌" if diffusion_misses else "✅"
            print(f"   {rmap_iter_msg} diffusion_model.* keys: {diffusion_hits}/{len(diffusion_keys)} in reverse_map")
            if diffusion_misses:
                print(f"      Sample misses (first 5): {diffusion_misses[:5]}")
                print(f"      ⚠️  {len(diffusion_misses)} diffusion_model.* keys will use fallback strategies")

        # Show first 5 misses for any key type
        all_misses = [k for k in lora_deltas if k not in reverse_map]
        if all_misses:
            for k in all_misses[:5]:
                print(f"      ❌ Not in reverse_map: '{k}'")
        else:
            print(f"   ✅ All {len(lora_deltas)} LoRA keys in reverse_map")

    # =====================================================================
    # 🔍 DIAGNOSTIC: Comprehensive key format trace (verbose-only)
    # =====================================================================
    if self._verbose:
        has_unet_deltas = any('input_blocks.' in k or 'output_blocks.' in k or 'middle_block.' in k for k in lora_deltas)
        if has_unet_deltas:
            ckpt_unet_bases = []
            for k in sorted(ckpt_sd.keys()):
                if 'input_blocks' in k or 'output_blocks' in k or 'middle_block' in k:
                    base = _strip_tensor_suffix(k)
                    if base not in ckpt_unet_bases:
                        ckpt_unet_bases.append(base)
            print(f"   🔍 [DIAG] Sample checkpoint UNet bases (first 5):")
            for b in ckpt_unet_bases[:5]:
                print(f"         '{b}'")

            lora_unet_deltas = sorted([k for k in lora_deltas if 'input_blocks.' in k or 'output_blocks.' in k or 'middle_block.' in k])
            print(f"   🔍 [DIAG] Sample LoRA UNet delta bases (first 5):")
            for b in lora_unet_deltas[:5]:
                in_rmap = b in reverse_map
                if not in_rmap and b.startswith('diffusion_model.'):
                    mdl_version = f"model.{b}"
                    mdl_in_rmap = mdl_version in reverse_map
                    print(f"         '{b}'  → in reverse_map: {in_rmap}  → model. version: {mdl_in_rmap}")
                else:
                    print(f"         '{b}'  → in reverse_map: {in_rmap}")

            # Detailed mismatch analysis
            unmatched = [b for b in lora_unet_deltas if b not in reverse_map]
            if unmatched:
                print(f"   🔍 [DIAG] === KEY-BY-KEY MISMATCH ANALYSIS ({len(unmatched)} unmatched) ===")
                pattern_counts = {}
                for b in unmatched[:30]:
                    segs = b.split('.')
                    block_prefix = '.'.join(segs[:4]) if len(segs) >= 4 else b

                    ckpt_prefixes = [block_prefix]
                    if block_prefix.startswith('diffusion_model.'):
                        ckpt_prefixes.append(f'model.{block_prefix}')
                    matching_ckpt = [ck for ck in ckpt_unet_bases if any(ck.startswith(p) for p in ckpt_prefixes)]
                    if matching_ckpt:
                        pattern_counts.setdefault("block_exists", 0)
                        pattern_counts["block_exists"] += 1
                        if pattern_counts["block_exists"] <= 5:
                            ckpt_suffix = matching_ckpt[0][len(block_prefix):]
                            lora_suffix = b[len(block_prefix):]
                            print(f"         BLOCK MATCH: '{block_prefix}'")
                            print(f"           LoRA suffix: '{lora_suffix}'")
                            print(f"           Ckpt suffix:  '{ckpt_suffix}'")
                            print(f"           LoRA key:     '{b}'")
                            print(f"           Ckpt key:     '{matching_ckpt[0]}'")
                    else:
                        pattern_counts.setdefault("block_missing", 0)
                        pattern_counts["block_missing"] += 1
                        if len(segs) >= 4:
                            for offset in range(-1, 2):
                                try:
                                    idx = int(segs[3]) + offset
                                except (ValueError, IndexError):
                                    continue
                                alt_prefix = '.'.join(segs[:3] + [str(idx)])
                                alt_prefixes = [alt_prefix]
                                if alt_prefix.startswith('diffusion_model.'):
                                    alt_prefixes.append(f'model.{alt_prefix}')
                                alt_match = [ck for ck in ckpt_unet_bases if any(ck.startswith(p) for p in alt_prefixes)]
                                if alt_match:
                                    if pattern_counts.get("block_missing", 0) <= 3:
                                        ckpt_label = alt_match[0]
                                        print(f"         BLOCK MISMATCH: LoRA '{block_prefix}' → Ckpt has '{alt_prefix}' (offset {offset:+d})")
                                        print(f"           LoRA key: '{b}'")
                                        print(f"           Ckpt key: '{ckpt_label}'")
                                    pattern_counts.setdefault("block_shift", 0)
                                    pattern_counts["block_shift"] += 1
                                    break
                            else:
                                dotted_variant = b.replace('ff_net.', 'ff.net.')
                                if dotted_variant != b and dotted_variant in reverse_map:
                                    if pattern_counts.get("block_missing", 0) <= 3:
                                        print(f"         NAMING MISMATCH (ff_net → ff.net):")
                                        print(f"           LoRA key:     '{b}'")
                                        print(f"           Dotted in RM: '{dotted_variant}'")
                                    pattern_counts.setdefault("naming_mismatch_ff_net", 0)
                                    pattern_counts["naming_mismatch_ff_net"] += 1

                print(f"   🔍 [DIAG] Mismatch categories:")
                for cat, count in sorted(pattern_counts.items()):
                    print(f"         {cat}: {count}")
            else:
                print(f"   🔍 [DIAG] All UNet deltas matched via reverse map! ✅")

            unet_in_rmap = len(lora_unet_deltas) - len(unmatched)
            print(f"   🔍 [DIAG] UNet delta keys in reverse map: {unet_in_rmap}/{len(lora_unet_deltas)} (total deltas: {len(lora_deltas)})")
        else:
            print(f"   🔍 [DIAG] No UNet-style keys in lora deltas (total deltas: {len(lora_deltas)})")
    matched_ckpt_keys: Set[str] = set()
    # Track ckpt_key → lora_base for _LazyDeltaDict consumers.
    # When lora_deltas is a lazy dict, `matched` stores _ShapeInfo values
    # (not tensors), so the bake loop needs a reverse mapping to reconstruct
    # deltas on demand from the stored A/B components.
    _ckpt_to_lora: Dict[str, str] = {}
    # Pre-computed tensors (fused QKV, to_out.0 fallback) that cannot be
    # reconstructed from _LazyDeltaDict components alone.
    _precomputed: Dict[str, torch.Tensor] = {}

    # =====================================================================
    # FUSED QKV DETECTION
    # =====================================================================
    fused_qkv_heads: Dict[str, torch.Tensor] = {}
    _head_re = re.compile(
        r'(?:diffusion_model\.|model\.diffusion_model\.)?'
        r'layers\.(\d+)\.attention\.(to_q|to_k|to_v)$'
    )
    has_fused_qkv = any(
        re.search(r'layers\.\d+\.attention\.qkv\.weight$', k)
        for k in ckpt_sd
    )
    if has_fused_qkv:
        for lora_base in list(lora_deltas.keys()):
            if _head_re.match(lora_base):
                if isinstance(lora_deltas, _LazyDeltaDict):
                    # pop from _LazyDeltaDict reconstructs the delta
                    fused_qkv_heads[lora_base] = lora_deltas.pop(lora_base)
                else:
                    fused_qkv_heads[lora_base] = lora_deltas.pop(lora_base)

    total_lora_keys = len(lora_deltas) + len(fused_qkv_heads)

    # =====================================================================
    # COMPONENT MATCHING ANALYSIS
    # =====================================================================
    self._component_breakdown = {
        "unet": {"lora_keys": 0, "matched": 0},
        "te": {"lora_keys": 0, "matched": 0},
        "clip": {"lora_keys": 0, "matched": 0},
        "vae": {"lora_keys": 0, "matched": 0},
    }
    lora_component_map: Dict[str, str] = {}
    all_lora_bases: Set[str] = set()

    def _count_lora_key(lb: str) -> None:
        all_lora_bases.add(lb)
        comp = _categorize_lora_key(lb)
        lora_component_map[lb] = comp
        self._component_breakdown[comp]["lora_keys"] += 1

    for lb in lora_deltas:
        _count_lora_key(lb)
    for lb in fused_qkv_heads:
        _count_lora_key(lb)

    with ProgressTracker(total=len(lora_deltas), desc="Matching LoRA keys") as match_progress:

        strategy_counts: Dict[str, int] = {}

        def _track_strategy(name: str) -> None:
            strategy_counts[name] = strategy_counts.get(name, 0) + 1

        def _is_shape_compatible(full_key: str, delta_tensor: torch.Tensor) -> bool:
            if full_key not in ckpt_sd:
                return False
            ckpt_t = ckpt_sd[full_key]
            if not isinstance(ckpt_t, torch.Tensor):
                return False
            return _check_shape_compatible(delta_tensor, ckpt_t)

        for lora_base, delta in lora_deltas.items():
            if not isinstance(delta, (torch.Tensor, _ShapeInfo)):
                continue

            # =============================================================
            # Strategy 0 (PRIMARY): Reverse Key Map lookup
            # =============================================================
            if lora_base in reverse_map:
                ckpt_base = reverse_map[lora_base]
                if ckpt_base in ckpt_base_lookup:
                    full_key = ckpt_base_lookup[ckpt_base]
                    if _is_shape_compatible(full_key, delta):
                        matched[full_key] = delta
                        matched_ckpt_keys.add(full_key)
                        if isinstance(lora_deltas, _LazyDeltaDict):
                            _ckpt_to_lora[full_key] = lora_base
                        _track_strategy("reverse_map")
                        continue

            # =============================================================
            # Strategy 1: exact match
            # =============================================================
            if lora_base in ckpt_base_lookup:
                full_key = ckpt_base_lookup[lora_base]
                if _is_shape_compatible(full_key, delta):
                    matched[full_key] = delta
                    matched_ckpt_keys.add(full_key)
                    if isinstance(lora_deltas, _LazyDeltaDict):
                        _ckpt_to_lora[full_key] = lora_base
                    _track_strategy("exact")
                    continue

            # =============================================================
            # Strategy 2: prefix-aware conversion
            # =============================================================
            ckpt_candidate = _lora_key_to_checkpoint_key(lora_base)
            if ckpt_candidate is not None:
                if ckpt_candidate in ckpt_base_lookup:
                    full_key = ckpt_base_lookup[ckpt_candidate]
                    if _is_shape_compatible(full_key, delta):
                        matched[full_key] = delta
                        matched_ckpt_keys.add(full_key)
                        if isinstance(lora_deltas, _LazyDeltaDict):
                            _ckpt_to_lora[full_key] = lora_base
                        _track_strategy("prefix")
                        continue

            # =============================================================
            # Strategy 2.5a: Underscore-to-dot fuzzy conversion
            # =============================================================
            fuzzy_match = _try_underscore_to_dot_conversion(lora_base, ckpt_sd, ckpt_base_lookup)
            if fuzzy_match is not None:
                if _is_shape_compatible(fuzzy_match, delta):
                    matched[fuzzy_match] = delta
                    matched_ckpt_keys.add(fuzzy_match)
                    if isinstance(lora_deltas, _LazyDeltaDict):
                        _ckpt_to_lora[fuzzy_match] = lora_base
                    _track_strategy("underscore_to_dot")
                    continue

            # =============================================================
            # Strategy 2.5b: SDXL TE conditioner prefix (handles te1. and te2.)
            # =============================================================
            te_cond_match = _try_te_conditioner_prefix(lora_base, ckpt_sd, ckpt_base_lookup, lora_deltas)
            if te_cond_match is not None:
                matched[te_cond_match] = delta
                matched_ckpt_keys.add(te_cond_match)
                if isinstance(lora_deltas, _LazyDeltaDict):
                    _ckpt_to_lora[te_cond_match] = lora_base
                _track_strategy("te_conditioner")
                continue

            # =============================================================
            # Strategy 2.5d: Deterministic TE Direct-Mapping (Kohya-inspired)
            # =============================================================
            te_direct_match = _try_deterministic_te_mapping(
                lora_base, ckpt_sd, ckpt_base_lookup, delta, _is_shape_compatible,
                verbose=getattr(self, '_verbose', False),
            )
            if te_direct_match is not None:
                matched[te_direct_match] = delta
                matched_ckpt_keys.add(te_direct_match)
                if isinstance(lora_deltas, _LazyDeltaDict):
                    _ckpt_to_lora[te_direct_match] = lora_base
                _track_strategy("deterministic_te")
                continue

            # =============================================================
            # Strategy 2.5e: Post-normalization TE Direct Mapping
            # =============================================================
            # After identity_normalize() → convert_sd15_diffusers_to_comfyui()
            # converts lora_te_* to text_model.* format, the reverse_map
            # (Section 2a) should already register these via pure_te entries.
            # But as a safety net, directly construct the checkpoint key for
            # any text_model.* keys that slipped through the above strategies.
            #
            # Mirrors Kohya's approach: know the architecture layout a priori
            # and construct the checkpoint key directly from the LoRA key.
            if lora_base.startswith('text_model.'):
                # 🔧 Strip trailing dots from lora_base — delta keys from
                # _reconstruct_lora_delta may have trailing '.' artifacts from
                # the SD1.5 Diffusers→ComfyUI conversion that prevent exact
                # key matching against ckpt_base_lookup (which has clean keys).
                clean_base = lora_base.rstrip('.')
                ckpt_candidate = f"cond_stage_model.transformer.{clean_base}"
                # 🔍 PHASE A2 DIAGNOSTIC: Log first 3 TE keys hitting Strategy 2.5e
                _te_25e_count = getattr(self, '_te_25e_count', 0)
                if _te_25e_count < 3:
                    self._te_25e_count = _te_25e_count + 1
                    in_lookup = ckpt_candidate in ckpt_base_lookup
                    print(f"      🔍 [A2] Strategy 2.5e check: '{lora_base}'")
                    print(f"           candidate (clean): '{ckpt_candidate}' (in ckpt_base_lookup: {in_lookup})")
                    if not in_lookup:
                        similar = [k for k in list(ckpt_base_lookup.keys())[:30] if 'text_model' in k and 'encoder' in k and 'layers' in k][:3]
                        if similar:
                            for sk in similar:
                                print(f"           similar ckpt key: '{sk}'")
                    print(f"           delta shape: {list(delta.shape)}")
                if ckpt_candidate in ckpt_base_lookup:
                    full_key = ckpt_base_lookup[ckpt_candidate]
                    if _is_shape_compatible(full_key, delta):
                        matched[full_key] = delta
                        matched_ckpt_keys.add(full_key)
                        if isinstance(lora_deltas, _LazyDeltaDict):
                            _ckpt_to_lora[full_key] = lora_base
                        _track_strategy("normalized_te_direct")
                        if getattr(self, '_verbose', False):
                            print(f"      ✅ Strategy 2.5e (normalized TE direct): "
                                  f"{lora_base} → {full_key}")
                        continue

            # 🔥 TE-key early exit: skip unmatched TE keys (not in checkpoint)
            # Uses key_utils substring matching (broader & more correct than startswith)
            from ..engine.key_utils import is_te_key as _is_te_key
            if _is_te_key(lora_base):
                # 🔧 SILENT-2: Track skipped TE keys for diagnostic summary
                _te_skip_count = getattr(self, '_te_skip_count', 0)
                self._te_skip_count = _te_skip_count + 1
                if getattr(self, '_verbose', False):
                    print(f"      ⏭️  Skipping TE key (not in checkpoint): {lora_base}")
                unmatched_lora.append(lora_base)
                continue

            # =============================================================
            # Strategy 3: suffix match (lowercase, normalized)
            # =============================================================
            lora_base_lower = lora_base.lower()
            matched_suffix = False

            # Quick guard: skip suffix scan if lora key lacks known structural blocks.
            # All supported architectures (Flux, SDXL, SD1.5, Z-Image, Anima)
            # use one of these block identifiers in their key hierarchy.
            # If none match, the key cannot suffix-match any checkpoint key.
            _KNOWN_BLOCKS = ('double_blocks', 'single_blocks', 'input_blocks',
                             'middle_blocks', 'output_blocks', 'layers.')
            if not any(b in lora_base for b in _KNOWN_BLOCKS):
                unmatched_lora.append(lora_base)
                continue

            # 🔧 VRAM-2: Use suffix index for O(1) candidate filtering instead of
            # scanning ALL ckpt_base_lower items per unmatched key.
            # Only checkpoint keys whose last path segment matches the LoRA key's
            # last segment are checked, reducing iteration from O(N) to O(M) where
            # M << N for most architectures.
            lora_last_seg = lora_base_lower.rsplit('.', 1)[-1]
            suffix_candidates = ckpt_suffix_index.get(lora_last_seg, [])
            for ckpt_lower in suffix_candidates:
                ckpt_key = ckpt_base_lower[ckpt_lower]
                if ckpt_lower.endswith(lora_base_lower):
                    if _is_shape_compatible(ckpt_key, delta):
                        matched[ckpt_key] = delta
                        matched_ckpt_keys.add(ckpt_key)
                        if isinstance(lora_deltas, _LazyDeltaDict):
                            _ckpt_to_lora[ckpt_key] = lora_base
                        matched_suffix = True
                        break
            if matched_suffix:
                _track_strategy("suffix")
                continue

            # Also try suffix match with underscore-to-dot converted lora_base
            for pattern, replacement, _desc in UNDERSCORE_TO_DOT_PATTERNS:
                converted = re.sub(pattern, replacement, lora_base)
                if converted != lora_base:
                    converted_lower = converted.lower()
                    converted_last_seg = converted_lower.rsplit('.', 1)[-1]
                    conv_candidates = ckpt_suffix_index.get(converted_last_seg, [])
                    for ckpt_lower in conv_candidates:
                        ckpt_key = ckpt_base_lower[ckpt_lower]
                        if ckpt_lower.endswith(converted_lower):
                            if _is_shape_compatible(ckpt_key, delta):
                                matched[ckpt_key] = delta
                                matched_ckpt_keys.add(ckpt_key)
                                if isinstance(lora_deltas, _LazyDeltaDict):
                                    _ckpt_to_lora[ckpt_key] = lora_base
                                matched_suffix = True
                                break
                    if matched_suffix:
                        break
            if matched_suffix:
                _track_strategy("suffix_underscore_to_dot")
                continue

            # =============================================================
            # Strategy 4: Global Search by Shape + Block ID
            # =============================================================
            shape_block_match = _global_search_by_shape_block(
                lora_base, delta, ckpt_sd, ckpt_base_lookup, matched_ckpt_keys,
                shape_block_index,
            )
            if shape_block_match is not None:
                matched[shape_block_match] = delta
                matched_ckpt_keys.add(shape_block_match)
                if isinstance(lora_deltas, _LazyDeltaDict):
                    _ckpt_to_lora[shape_block_match] = lora_base
                if getattr(self, '_verbose', False):
                    print(f"      🔍 Shape+Block match: {lora_base} → {shape_block_match}")
                _track_strategy("shape_block")
                continue

            # =============================================================
            # Strategy 6: shape match as last resort (O(1) indexed)
            # =============================================================
            shape_candidates = shape_index.get(delta.shape, [])
            for ckpt_key in shape_candidates:
                if ckpt_key in matched_ckpt_keys:
                    continue
                if delta.dim() >= 2:
                    matched[ckpt_key] = delta
                    matched_ckpt_keys.add(ckpt_key)
                    if isinstance(lora_deltas, _LazyDeltaDict):
                        _ckpt_to_lora[ckpt_key] = lora_base
                    _track_strategy("shape_match")
                    break
            else:
                _track_strategy("unmatched")
                unmatched_lora.append(lora_base)

            match_progress += 1


    # =====================================================================
    # POST-PROCESSING: Fused QKV Resolution
    # =====================================================================
    fused_qkv_resolved = 0

    if has_fused_qkv and fused_qkv_heads:
        fused_pending: Dict[str, Dict[str, str]] = {}
        for lora_base in fused_qkv_heads:
            m = _head_re.match(lora_base)
            if m:
                layer_num = m.group(1)
                head = m.group(2)
                group_key = f"layers.{layer_num}"
                if group_key not in fused_pending:
                    fused_pending[group_key] = {}
                fused_pending[group_key][head] = lora_base

        for group_key, heads in fused_pending.items():
            if len(heads) >= 3:
                layer_num = group_key.split('.')[1]
                ckpt_qkv_base = f"layers.{layer_num}.attention.qkv"

                qkv_full_key = None
                for ckpt_key in ckpt_sd:
                    ckpt_base = _strip_tensor_suffix(ckpt_key)
                    if ckpt_base == ckpt_qkv_base:
                        qkv_full_key = ckpt_key
                        break

                if qkv_full_key and qkv_full_key not in matched_ckpt_keys:
                    delta_q = fused_qkv_heads[heads['to_q']]
                    delta_k = fused_qkv_heads[heads['to_k']]
                    delta_v = fused_qkv_heads[heads['to_v']]

                    qkv_weight = ckpt_sd[qkv_full_key]
                    if delta_q.dim() >= 2 and qkv_weight.dim() >= 2:
                        if delta_q.shape[0] * 3 == qkv_weight.shape[0]:
                            cat_dim = 0
                        elif delta_q.shape[1] * 3 == qkv_weight.shape[1]:
                            cat_dim = 1
                        else:
                            cat_dim = 0

                        # Ensure dtype consistency before cat
                        target_dtype = delta_q.dtype
                        fused_delta = torch.cat([
                            delta_q.to(target_dtype),
                            delta_k.to(target_dtype),
                            delta_v.to(target_dtype),
                        ], dim=cat_dim)

                        matched[qkv_full_key] = fused_delta
                        matched_ckpt_keys.add(qkv_full_key)
                        if isinstance(lora_deltas, _LazyDeltaDict):
                            # QKV fused delta cannot be reconstructed from
                            # a single _LazyDeltaDict entry — store precomputed.
                            _precomputed[qkv_full_key] = fused_delta
                        fused_qkv_resolved += 3
                        if getattr(self, '_verbose', False):
                            print(f"      🔄 Fused QKV: layers.{layer_num}.attention "
                                  f"to_q+to_k+to_v → {qkv_full_key} "
                                  f"(cat_dim={cat_dim}, shapes: {delta_q.shape}*3 → {fused_delta.shape})")
                        continue
                    else:
                        for head_name, lora_key in heads.items():
                            unmatched_lora.append(lora_key)
                        continue
                else:
                    for head_name, lora_key in heads.items():
                        unmatched_lora.append(lora_key)
                    continue

            for head_name, lora_key in heads.items():
                unmatched_lora.append(lora_key)

    # --- Phase B: Handle to_out.0 fallback from unmatched ---
    if unmatched_lora:
        remaining_unmatched = []
        for lora_base in unmatched_lora:
            m_out = re.match(
                r'(?:diffusion_model\.|model\.diffusion_model\.)?'
                r'layers\.(\d+)\.attention\.to_out\.0$',
                lora_base
            )
            if m_out:
                layer_num = m_out.group(1)
                ckpt_out_base = f"layers.{layer_num}.attention.out"
                out_full_key = None
                for ckpt_key in ckpt_sd:
                    ckpt_base = _strip_tensor_suffix(ckpt_key)
                    if ckpt_base == ckpt_out_base:
                        out_full_key = ckpt_key
                        break
                if out_full_key and out_full_key not in matched_ckpt_keys:
                    delta = lora_deltas[lora_base]
                    matched[out_full_key] = delta
                    matched_ckpt_keys.add(out_full_key)
                    if isinstance(lora_deltas, _LazyDeltaDict):
                        # Already reconstructed by __getitem__ above — store precomputed.
                        _precomputed[out_full_key] = delta
                    fused_qkv_resolved += 1
                    if getattr(self, '_verbose', False):
                        print(f"      🔄 to_out.0→out: {lora_base} → {out_full_key}")
                    continue
            remaining_unmatched.append(lora_base)
        unmatched_lora = remaining_unmatched

    # --- Compute component breakdown from matched vs unmatched ---
    matched_lora_bases = all_lora_bases - set(unmatched_lora)
    for lb in matched_lora_bases:
        comp = lora_component_map.get(lb, 'unet')
        self._component_breakdown[comp]["matched"] += 1

    for comp, data in self._component_breakdown.items():
        if data["lora_keys"] > 0 and data["matched"] == 0:
            data["reason"] = f"no matching {comp} keys found in checkpoint"

    # --- Print results ---
    if unmatched_lora:
        print(f"   ⚠️  Could not match {len(unmatched_lora)} LoRA keys:")
        for uk in unmatched_lora[:10]:
            print(f"      - {uk}")
            in_rmap = uk in reverse_map
            in_ckpt_lookup = uk in ckpt_base_lookup
            in_ckpt_lower = uk.lower() in ckpt_base_lower

            suffix = uk.split('.')[-1] if '.' in uk else uk
            close_matches = []
            for ckpt_lower_key, ckpt_orig_key in ckpt_base_lower.items():
                if ckpt_lower_key.endswith('.' + suffix.lower()):
                    close_matches.append(ckpt_orig_key)
                elif ckpt_lower_key.endswith(suffix.lower()):
                    close_matches.append(ckpt_orig_key)

            print(f"         In reverse_map: {in_rmap}  |  In ckpt_lookup: {in_ckpt_lookup}  |  In ckpt_lower: {in_ckpt_lower}")
            if close_matches:
                seen = set()
                unique_matches = []
                for cm in close_matches:
                    if cm not in seen:
                        seen.add(cm)
                        unique_matches.append(cm)
                print(f"         📋 Close ckpt matches (suffix='{suffix}', showing {min(3, len(unique_matches))}):")
                for cm in unique_matches[:3]:
                    print(f"           - {cm}")
            print(f"         💡 Suggestion: Check if suffix '{suffix}' needs a new reverse_map entry")

        if len(unmatched_lora) > 10:
            print(f"         ... and {len(unmatched_lora) - 10} more unmatched keys (see above for truncated list)")

        diffusers_hints = [uk for uk in unmatched_lora if "down_blocks_" in uk or "up_blocks_" in uk or "mid_block_" in uk]
        if diffusers_hints:
            print(f"   💡 TIP: {len(diffusers_hints)} unmatched key(s) contain SD1.5 Diffusers naming")
            print(f"      (down_blocks_/up_blocks_/mid_block_). This suggests the LoRA wasn't")
            print(f"      properly detected as SD1.5 Diffusers format during normalization.")
    else:
        print(f"   ✅ All {total_lora_keys} LoRA keys matched successfully!")

    if fused_qkv_resolved > 0:
        print(f"   🔄 Fused QKV resolution: {fused_qkv_resolved} attention heads combined into fused deltas")

    # 🔧 SILENT-2: Report TE keys skipped due to absence in checkpoint
    _te_skip_total = getattr(self, '_te_skip_count', 0)
    if _te_skip_total > 0:
        print(f"   ⏭️  Skipped {_te_skip_total} TE keys (not present in checkpoint — expected for SDXL/Flux LoRA on mismatched base)")

    print(f"   ✅ Matched {len(matched)} LoRA deltas to checkpoint keys")

    # ── Wrap matched in _MatchedDeltas for on-demand reconstruction ──
    # When lora_deltas is a _LazyDeltaDict, matched contains _ShapeInfo
    # values instead of real tensors.  Wrap it in _MatchedDeltas so that
    # the bake loop can pop() one delta at a time, reconstructing on demand.
    use_lazy = isinstance(lora_deltas, _LazyDeltaDict)
    if use_lazy and _precomputed:
        # Some matched entries (fused QKV, to_out.0) have precomputed
        # tensors that must be stored directly.  Create the wrapper.
        lazy_matched = _MatchedDeltas(lora_deltas, _ckpt_to_lora, _precomputed)
    elif use_lazy:
        lazy_matched = _MatchedDeltas(lora_deltas, _ckpt_to_lora)
    else:
        lazy_matched = matched

    if return_stats:
        return lazy_matched, strategy_counts
    return lazy_matched
