"""
key_mapper.py — Shared Checkpoint ↔ LoRA Key Mapping

Single authoritative source for ALL key mapping logic, used by both the
extractor (:mod:`lora_extractor`) and the baker (:mod:`baking_processor_matching`).

Design follows ``plans/working_principles.md``:
    P1 (Delete First)  — replaces two separate implementations with one shared module
    P3 (One Pattern)   — single ``_ARCH_KEY_RULES`` registry derived from both directions
    P4 (Resist Guards) — clean prefix-matching logic, no special-case cascades
    P5 (Simple Dispatch) — each function has one clear responsibility

Architecture support:
    - SDXL UNet (model.diffusion_model.input_blocks.*)
    - SDXL TE1 (cond_stage_model.transformer.text_model.*)
    - SDXL TE2 (conditioner.embedders.*.transformer.text_model.*)
    - SD1.5 UNet (diffusion_model.input_blocks.*)
    - SD1.5 TE (cond_stage_model.transformer.text_model.*)
    - Flux standard (model.diffusion_model.transformer.double_blocks.* / single_blocks.*)
    - Flux Klein (bare double_blocks.* / single_blocks.* — adds transformer. prefix)
    - Z-Image / Lumina2 (layers.* / cap_embedder.* etc.)
    - Anima (net.blocks.*)
"""

import re
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Shared helpers
# ============================================================================

_TENSOR_SUFFIXES = ('.weight', '.bias', '.alpha')


def strip_tensor_suffix(key: str) -> str:
    """
    Strip ``.weight``, ``.bias``, or ``.alpha`` suffix from a tensor key.

    Shared helper used by key_mapper functions, the baker's shape-index
    builders, and the extractor's key matching.

    Example:
        >>> strip_tensor_suffix("double_blocks.0.img_attn.proj.weight")
        "double_blocks.0.img_attn.proj"
    """
    for suffix in _TENSOR_SUFFIXES:
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return key


# ============================================================================
# Pattern matching helpers (for diffusers key conversion)
# ============================================================================


def _match_pattern(key: str, pattern: str) -> Optional[Dict[str, str]]:
    """
    Match a ``{i}`` pattern against a key.

    E.g., ``_match_pattern("double_blocks.3.img_attn.qkv", "double_blocks.{i}")``
    returns ``{"i": "3"}``.

    Returns:
        Dict of ``{placeholder: value}`` if matched, ``None`` otherwise.
    """
    escaped = re.escape(pattern).replace(r"\{i\}", r"(\d+)")
    m = re.match(escaped, key)
    if m:
        return {"i": m.group(1)}
    return None


def _fill_pattern(pattern: str, vars: Dict[str, str]) -> str:
    """Fill ``{i}`` placeholders in a pattern with extracted values."""
    result = pattern
    for k, v in vars.items():
        result = result.replace("{" + k + "}", v)
    return result


def _strip_known_prefixes(key: str, arch: str) -> str:
    """
    Strip known LoRA/architecture prefixes for diffusers key matching.

    For Flux: strips ``transformer.``, ``model.diffusion_model.``, etc.
    """
    rules = _ARCH_KEY_RULES.get(arch, {})
    lora_prefix = rules.get("lora_prefix", "")
    if key.startswith(lora_prefix):
        return key[len(lora_prefix):]
    # Also try bare model.diffusion_model. prefix
    if key.startswith("model.diffusion_model."):
        return key[len("model.diffusion_model."):]
    if key.startswith("diffusion_model."):
        return key[len("diffusion_model."):]
    return key


# ============================================================================
# Architecture key rules — SINGLE source of truth
# ============================================================================
# Each entry defines:
#   checkpoint_prefixes: List of (checkpoint_prefix, lora_prefix).
#     Forward mapping: if a key starts with checkpoint_prefix,
#     replace it with lora_prefix.
#   bare_paths: List of path prefixes that may appear WITHOUT any model
#     prefix (e.g. Klein Flux "double_blocks.*").  For forward mapping,
#     these get ``lora_prefix`` prepended.
#   lora_prefix: Canonical LoRA prefix for this architecture.
#     Used when a bare_path is matched.

_ARCH_KEY_RULES: Dict[str, Dict] = {
    # ── Flux (standard + Klein) ──────────────────────────────────────────
    # Standard: model.diffusion_model.transformer.double_blocks.*
    # Normalized (after normalizer strips model.diffusion_model.):
    #   transformer.double_blocks.*
    # Klein (bare, no prefix): double_blocks.*
    "flux": {
        "checkpoint_prefixes": [
            # Order matters: more specific first
            ("model.diffusion_model.transformer.double_blocks.", "transformer.double_blocks."),
            ("model.diffusion_model.transformer.single_blocks.", "transformer.single_blocks."),
            ("transformer.double_blocks.", "transformer.double_blocks."),
            ("transformer.single_blocks.", "transformer.single_blocks."),
        ],
        "bare_paths": [
            "double_blocks.",    # Klein Flux bare key → transformer.double_blocks.*
            "single_blocks.",    # Klein Flux bare key → transformer.single_blocks.*
        ],
        "lora_prefix": "transformer.",
        # ── Diffusers-format LoRA keys (for ComfyUI load_lora compatibility) ──
        "diffusers": {
            "block_maps": {
                "double_blocks.{i}": "transformer_blocks.{i}",
                "single_blocks.{i}": "single_transformer_blocks.{i}",
            },
            "qkv_splits": {
                "double_blocks.{i}.img_attn.qkv": [
                    ("transformer_blocks.{i}.attn.to_q", (0, 1)),
                    ("transformer_blocks.{i}.attn.to_k", (1, 1)),
                    ("transformer_blocks.{i}.attn.to_v", (2, 1)),
                ],
                "double_blocks.{i}.txt_attn.qkv": [
                    ("transformer_blocks.{i}.attn.add_q_proj", (0, 1)),
                    ("transformer_blocks.{i}.attn.add_k_proj", (1, 1)),
                    ("transformer_blocks.{i}.attn.add_v_proj", (2, 1)),
                ],
                "single_blocks.{i}.linear1": [
                    ("single_transformer_blocks.{i}.attn.to_q", (0, 1)),
                    ("single_transformer_blocks.{i}.attn.to_k", (1, 1)),
                    ("single_transformer_blocks.{i}.attn.to_v", (2, 1)),
                    ("single_transformer_blocks.{i}.proj_mlp", (3, 4)),
                ],
            },
            "simple_map": {
                # Double block sub-keys
                "img_attn.proj": "attn.to_out.0",
                "txt_attn.proj": "attn.to_add_out",
                "img_mod.lin": "norm1.linear",
                "txt_mod.lin": "norm1_context.linear",
                "img_mlp.0": "ff.net.0.proj",
                "img_mlp.2": "ff.net.2",
                "txt_mlp.0": "ff_context.net.0.proj",
                "txt_mlp.2": "ff_context.net.2",
                "img_attn.norm.query_norm": "attn.norm_q",
                "img_attn.norm.key_norm": "attn.norm_k",
                "txt_attn.norm.query_norm": "attn.norm_added_q",
                "txt_attn.norm.key_norm": "attn.norm_added_k",
                # Single block sub-keys
                "linear2": "proj_out",
                "modulation.lin": "norm.linear",
                "norm.query_norm": "attn.norm_q",
                "norm.key_norm": "attn.norm_k",
            },
            "basic_map": {
                # CRITICAL: The checkpoint stores non-block keys using DIFFUSERS
                # naming (e.g. "final_layer.linear", "img_in", "txt_in",
                # "time_in.in_layer"), NOT Klein checkpoint naming ("proj_out",
                # "x_embedder", "context_embedder").  We match the diffusers
                # name and map to the CHECKPOINT name because ComfyUI's
                # model_lora_keys_unet() builds LoRA key_map entries using the
                # CHECKPOINT key name (from flux_to_diffusers() MAP_BASIC tuples
                # where k[1]=checkpoint_name is the dict key).
                #
                # Resulting LoRA key = lora_prefix + new_basic = "transformer.{checkpoint_name}"
                # which matches ComfyUI's key_map["transformer.{checkpoint_name}"].
                "final_layer.linear": "proj_out",
                "img_in": "x_embedder",
                "txt_in": "context_embedder",
                "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
                "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
                "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
                "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
                "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
                "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
                "norm_out.linear": "norm_out.linear",
                # final_layer.adaLN_modulation.1 → checkpoint name norm_out.linear
                # (with swap_scale_shift in ComfyUI; we don't apply the transform
                #  so the LoRA tensor is the raw delta)
                "final_layer.adaLN_modulation.1": "norm_out.linear",
                # Flux 2 modulation keys (not in current ComfyUI MAP_BASIC,
                # but properly prefixed for future compatibility)
                "double_stream_modulation_img.lin": "double_stream_modulation_img.lin",
                "double_stream_modulation_txt.lin": "double_stream_modulation_txt.lin",
                "single_stream_modulation.lin": "single_stream_modulation.lin",
            },
        },
    },
    # ── Lumina2 ──────────────────────────────────────────────────────────
    # Normalized keys are bare (layers.*, cap_embedder.*, etc.) after
    # normalizer strips model.diffusion_model.  LoRA keys need the
    # diffusion_model. prefix to match ComfyUI's model parameters.
    "lumina2": {
        "checkpoint_prefixes": [
            ("layers.",           "diffusion_model.layers."),
            ("cap_embedder.",     "diffusion_model.cap_embedder."),
            ("context_refiner.",  "diffusion_model.context_refiner."),
            ("noise_refiner.",    "diffusion_model.noise_refiner."),
            ("final_layer.",      "diffusion_model.final_layer."),
            ("x_embedder",        "diffusion_model.x_embedder"),
            ("x_pad_token",       "diffusion_model.x_pad_token"),
            ("cap_pad_token",     "diffusion_model.cap_pad_token"),
        ],
        "bare_paths": [],
        "lora_prefix": "diffusion_model.",
        # ── Diffusers-format LoRA keys (for ComfyUI load_lora compatibility) ──
        "diffusers": {
            "block_maps": {
                # Lumina2 block prefix is pass-through (same in checkpoint and diffusers)
                "layers.{i}": "layers.{i}",
                "context_refiner.{i}": "context_refiner.{i}",
                "noise_refiner.{i}": "noise_refiner.{i}",
            },
            "qkv_splits": {
                "layers.{i}.attention.qkv": [
                    ("layers.{i}.attention.to_q", (0, 1)),
                    ("layers.{i}.attention.to_k", (1, 1)),
                    ("layers.{i}.attention.to_v", (2, 1)),
                ],
                "context_refiner.{i}.attention.qkv": [
                    ("context_refiner.{i}.attention.to_q", (0, 1)),
                    ("context_refiner.{i}.attention.to_k", (1, 1)),
                    ("context_refiner.{i}.attention.to_v", (2, 1)),
                ],
                "noise_refiner.{i}.attention.qkv": [
                    ("noise_refiner.{i}.attention.to_q", (0, 1)),
                    ("noise_refiner.{i}.attention.to_k", (1, 1)),
                    ("noise_refiner.{i}.attention.to_v", (2, 1)),
                ],
            },
            "simple_map": {
                # Checkpoint sub-key → Diffusers sub-key
                # Derived from z_image_to_diffusers() block_map (reversed)
                "attention.out": "attention.to_out.0",
                "attention.q_norm": "attention.norm_q",
                "attention.k_norm": "attention.norm_k",
                # Pass-through entries (identity in ComfyUI's block_map)
                "attention_norm1": "attention_norm1",
                "attention_norm2": "attention_norm2",
                "feed_forward.w1": "feed_forward.w1",
                "feed_forward.w2": "feed_forward.w2",
                "feed_forward.w3": "feed_forward.w3",
                "ffn_norm1": "ffn_norm1",
                "ffn_norm2": "ffn_norm2",
                "adaLN_modulation.0": "adaLN_modulation.0",
            },
            "basic_map": {
                # CRITICAL: Identity mapping — ComfyUI's model_lora_keys_unet()
                # indexes by the KEY of z_image_to_diffusers() key_map, which
                # is the CHECKPOINT key name (second element of MAP_BASIC tuples).
                # The diffusers rename happens in the VALUE, not the LoRA key.
                "all_final_layer.2-1.linear": "all_final_layer.2-1.linear",
                "all_final_layer.2-1.adaLN_modulation.1": "all_final_layer.2-1.adaLN_modulation.1",
                "all_x_embedder.2-1": "all_x_embedder.2-1",
                "cap_embedder.0": "cap_embedder.0",
                "cap_embedder.1": "cap_embedder.1",
                "x_pad_token": "x_pad_token",
                "cap_pad_token": "cap_pad_token",
                "t_embedder.mlp.0": "t_embedder.mlp.0",
                "t_embedder.mlp.2": "t_embedder.mlp.2",
            },
        },
    },
    # ── SDXL ─────────────────────────────────────────────────────────────
    # Checkpoint keys keep model.diffusion_model.* prefix.
    # LoRA keys use diffusion_model.* (strip model. prefix).
    "sdxl": {
        "checkpoint_prefixes": [
            ("model.diffusion_model.", "diffusion_model."),
        ],
        "bare_paths": [],
        "lora_prefix": "diffusion_model.",
    },
    # ── SD1.5 ────────────────────────────────────────────────────────────
    # SD1.5 UNet: diffusion_model.xxx → diffusion_model.xxx (pass-through)
    # SD1.5 CLIP: cond_stage_model.transformer.text_model.xxx
    #             → text_encoders.clip_l.transformer.text_model.xxx
    "sd15": {
        "checkpoint_prefixes": [
            ("cond_stage_model.transformer.text_model.",
             "text_encoders.clip_l.transformer.text_model."),
            ("diffusion_model.", "diffusion_model."),  # pass-through
        ],
        "bare_paths": [],
        "lora_prefix": "diffusion_model.",
    },
    # ── Anima ────────────────────────────────────────────────────────────
    # Anima: net.blocks.xxx → diffusion_model.blocks.xxx
    # After checkpoint_normalizer strips "net." prefix, keys are bare
    # blocks.xxx. The bare_paths rule catches these normalized keys.
    "anima": {
        "checkpoint_prefixes": [
            ("net.blocks.", "diffusion_model.blocks."),
        ],
        "bare_paths": [
            "blocks.",  # normalized bare blocks.* → diffusion_model.blocks.*
        ],
        "lora_prefix": "diffusion_model.",
    },
    # ── Additional TE/CLIP prefix patterns ───────────────────────────────
    # These are shared across architectures for text-encoder keys.
    # SDXL TE2: model.conditioner.embedders.0.transformer.text_model.xxx
    #           → text_encoders.transformer.text_model.xxx
    # SDXL TE1: model.conditioner.embedders.1.transformer.text_model.xxx
    #           → text_encoders.transformer.text_model.xxx
    # SDXL / SD1.5: model.text_model.xxx → text_model.xxx
    "te": {  # special — matches any arch's text encoder
        "checkpoint_prefixes": [
            ("model.conditioner.embedders.0.transformer.text_model.",
             "text_encoders.transformer.text_model."),
            ("model.conditioner.embedders.1.transformer.text_model.",
             "text_encoders.transformer.text_model."),
            ("model.conditioner.", "conditioner."),
            ("model.text_model.", "text_model."),
        ],
        "bare_paths": [],
        "lora_prefix": "",
    },
}

# Architectures in detection order (most specific first)
_ARCH_ORDER = ["flux", "lumina2", "anima", "sdxl", "sd15"]


# ============================================================================
# Forward mapping: checkpoint key → LoRA key  (used by extractor)
# ============================================================================

def checkpoint_key_to_lora_key(
    ckpt_key: str,
    arch: Optional[str] = None,
) -> str:
    """
    Convert a normalized checkpoint key to the equivalent LoRA master-format key.

    After :func:`~engine.checkpoint_normalizer.normalize_checkpoint_key`, the
    key reflects the canonical form of the *checkpoint* naming.  This function
    maps it to the *LoRA* master format used by :func:`~engine.identity_normalizer.identity_normalize`
    and the rest of the merger suite.

    Args:
        ckpt_key: Checkpoint key (may include ``.weight`` / ``.bias`` / ``.alpha`` suffix).
        arch: Optional architecture hint (``"flux"``, ``"sdxl"``, ``"sd15"``, etc.).
            If ``None``, tries all known architectures.

    Returns:
        LoRA master-format key (without ``.weight/.bias/.alpha`` suffix).
        If no known conversion applies, the key is returned unchanged
        (pass-through for already-LoRA-format keys).
    """
    # ── Strip tensor-type suffix ──────────────────────────────────────────
    key = strip_tensor_suffix(ckpt_key)

    # ── Apply architecture-specific prefix rules ─────────────────────────
    architectures_to_try: List[Tuple[str, Dict]] = []
    if arch is not None and arch in _ARCH_KEY_RULES:
        architectures_to_try = [(arch, _ARCH_KEY_RULES[arch])]
    else:
        # Try all architectures in order
        for a in _ARCH_ORDER:
            if a in _ARCH_KEY_RULES:
                architectures_to_try.append((a, _ARCH_KEY_RULES[a]))
        # Also try the generic 'te' rule
        if 'te' in _ARCH_KEY_RULES:
            architectures_to_try.append(('te', _ARCH_KEY_RULES['te']))

    for arch_name, rules in architectures_to_try:
        # ── Checkpoint prefix match ──────────────────────────────────────
        for ckpt_prefix, lora_prefix in rules.get("checkpoint_prefixes", []):
            if key.startswith(ckpt_prefix):
                return lora_prefix + key[len(ckpt_prefix):]

        # ── Bare path match (Klein-style, no prefix at all) ──────────────
        for bare_path in rules.get("bare_paths", []):
            if key.startswith(bare_path):
                lora_prefix = rules.get("lora_prefix", "")
                return lora_prefix + key

    # Pass-through: already in LoRA format (Flux bare, Z-Image, SD1.5,
    # clip_l/clip_g/t5xxl/text_encoder/text_encoders, etc.)
    return key


# ============================================================================
# Diffusers key conversion: checkpoint-internal → diffusers LoRA format
# (used by extractor for Flux/Lumina2 where ComfyUI expects diffusers keys)
# ============================================================================


def checkpoint_key_to_diffusers_key(
    ckpt_base: str,
    arch: str,
) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    """
    Convert a checkpoint-internal key base to diffusers-format LoRA key(s).

    For QKV-split keys (merged qkv → to_q/to_k/to_v), returns multiple entries
    with ``(offset_in_h, width_in_h)`` tuples. Non-split keys return
    ``(diff_key, None)``.

    The ``(offset_in_h, width_in_h)`` tuple means:
        row_start = offset_in_h * hidden_size
        row_end   = (offset_in_h + width_in_h) * hidden_size

    This tells the caller how to slice the SVD output along the output
    dimension (lora_B row dimension).

    Args:
        ckpt_base: Checkpoint-internal key base, **with** the ``lora_prefix``
            already applied (e.g. ``"transformer.double_blocks.0.img_attn.qkv"``).
        arch: Architecture name from ``_ARCH_KEY_RULES``.

    Returns:
        List of ``(diffusers_key, (offset_in_h, width_in_h) or None)``.
        If no diffusers rules apply, returns ``[(ckpt_base, None)]``
        (pass-through for architectures that don't need conversion).
    """
    rules = _ARCH_KEY_RULES.get(arch, {}).get("diffusers")
    if not rules:
        return [(ckpt_base, None)]

    lora_prefix = _ARCH_KEY_RULES[arch].get("lora_prefix", "")
    inner = _strip_known_prefixes(ckpt_base, arch)

    # ── 1. Check qkv_splits (1→N expansion) ──────────────────────────────
    for pattern, splits in rules.get("qkv_splits", {}).items():
        vars = _match_pattern(inner, pattern)
        if vars is not None:
            result = []
            for diff_pattern, (offset_h, width_h) in splits:
                diff_inner = _fill_pattern(diff_pattern, vars)
                result.append((f"{lora_prefix}{diff_inner}", (offset_h, width_h)))
            return result

    # ── 2. Check block_maps + simple_map (1→1 rename) ────────────────────
    for block_old, block_new in rules.get("block_maps", {}).items():
        vars = _match_pattern(inner, block_old)
        if vars is not None:
            block_old_filled = _fill_pattern(block_old, vars)
            inner_suffix = inner[len(block_old_filled):]  # e.g. ".img_attn.proj"

            for old_sub, new_sub in rules.get("simple_map", {}).items():
                if inner_suffix == f".{old_sub}" or inner_suffix.startswith(f".{old_sub}."):
                    suffix = inner_suffix[len(f".{old_sub}"):]
                    block_new_filled = _fill_pattern(block_new, vars)
                    diff_inner = f"{block_new_filled}.{new_sub}{suffix}"
                    return [(f"{lora_prefix}{diff_inner}", None)]

            # Block matched but no sub-match — pass through sub-key as-is
            block_new_filled = _fill_pattern(block_new, vars)
            return [(f"{lora_prefix}{block_new_filled}{inner_suffix}", None)]

    # ── 3. Check basic_map (non-block root-level keys) ───────────────────
    for old_basic, new_basic in rules.get("basic_map", {}).items():
        if inner == old_basic or inner.startswith(old_basic + "."):
            suffix = inner[len(old_basic):]
            result_key = f"{lora_prefix}{new_basic}{suffix}"
            return [(result_key, None)]

    # Pass-through: no diffusers rule matched
    return [(f"{lora_prefix}{inner}", None)]


# ============================================================================
# Reverse mapping: generate all LoRA key variants  (used by baker)
# ============================================================================

def build_lora_key_variants(ckpt_base: str) -> List[str]:
    """
    Generate ALL known LoRA key variants for a given checkpoint base key.

    Used by the baker to build its reverse key map.  Mirrors ComfyUI's
    ``model_lora_keys_unet()`` and ``model_lora_keys_clip()`` logic.

    Args:
        ckpt_base: Checkpoint base key (without ``.weight`` / ``.bias`` / ``.alpha`` suffix).

    Returns:
        List of LoRA key variants (canonical/master format first).
        Empty list if the key doesn't match any known architecture pattern.
    """
    variants: List[str] = []
    seen: set = set()

    def _add(v: str) -> None:
        if v not in seen:
            seen.add(v)
            variants.append(v)

    # ── Section 1: SDXL UNet (model.diffusion_model.*) ───────────────────
    if ckpt_base.startswith('model.diffusion_model.'):
        inner = ckpt_base[len('model.'):]  # diffusion_model.input_blocks.X
        pure_path = inner[len('diffusion_model.'):]  # input_blocks.X

        _add(inner)                                          # diffusion_model.*
        _add(pure_path)                                      # pure path (input_blocks.X)
        _add(f"lora_unet_{pure_path}")                       # Kohya-style

        # Underscore variants
        underscorized = pure_path.replace('.', '_')
        if underscorized != pure_path:
            _add(underscorized)
            _add(f"lora_unet_{underscorized}")

    # ── Section 2a: SD1.5 TE (cond_stage_model.transformer.*) ────────────
    if ckpt_base.startswith('cond_stage_model.transformer.'):
        inner = ckpt_base[len('cond_stage_model.'):]  # transformer.text_model.X
        pure_te = ckpt_base[len('cond_stage_model.transformer.'):]  # text_model.X

        _add(pure_te)                                        # text_model.*
        _add(inner)                                          # transformer.text_model.*
        _add(f"lora_te_{pure_te}")                           # Kohya-style
        _add(f"lora_te1_{pure_te}")                          # te1 variant
        _add(f"te.{pure_te}")                                # te. prefix
        _add(f"te1.{pure_te}")                               # te1. prefix

        # Underscore variants
        underscorized = pure_te.replace('.', '_')
        if underscorized != pure_te:
            _add(underscorized)
            _add(f"lora_te_{underscorized}")
            _add(f"lora_te1_{underscorized}")

    # ── Section 2a-ii: CLIP-L / CLIP-G (clip_l.text_model.* / clip_g.text_model.*) ──
    if ckpt_base.startswith(('clip_l.text_model.', 'clip_g.text_model.')):
        clip_prefix = ckpt_base.split('.')[0]  # clip_l or clip_g
        pure_clip = ckpt_base[len(f"{clip_prefix}."):]  # text_model.X

        _add(pure_clip)                                      # text_model.*
        _add(f"lora_te_{pure_clip}")                         # lora_te_ variant
        _add(f"te.{pure_clip}")                              # te. prefix
        _add(ckpt_base)                                      # original
        _add(f"lora_te_{clip_prefix}_{pure_clip}")           # lora_te_clip_l_ variant

        # Underscore variants
        underscorized = pure_clip.replace('.', '_')
        if underscorized != pure_clip:
            _add(underscorized)
            _add(f"lora_te_{underscorized}")

    # ── Section 2b: SDXL TE2 (conditioner.embedders.*) ───────────────────
    if ckpt_base.startswith('conditioner.embedders.'):
        parts = ckpt_base.split('.')
        if len(parts) >= 4 and parts[2].isdigit():
            embedder_idx = parts[2]
            inner_key = '.'.join(parts[3:])  # transformer.text_model.X
            pure_te2 = '.'.join(parts[4:])   # text_model.X

            _add(f"conditioner.embedders.{embedder_idx}.{inner_key}")
            _add(f"te2.{pure_te2}")
            _add(f"te1.{pure_te2}")                           # some use te1 for first embedder
            _add(f"lora_te2_{pure_te2}")
            _add(f"conditioner.{inner_key}")

    # ── Section 2c: T5 text encoder (t5xxl.transformer.*) ────────────────
    if ckpt_base.startswith('t5xxl.transformer.'):
        inner = ckpt_base[len('t5xxl.'):]  # transformer.encoder.layers.X
        pure_t5 = ckpt_base[len('t5xxl.transformer.'):]  # encoder.layers.X

        _add(inner)                                          # transformer.encoder.layers.X
        _add(pure_t5)                                        # encoder.layers.X
        _add(f"lora_te_{pure_t5}")                           # lora_te_ variant
        _add(f"te.{pure_t5}")                                # te. prefix

        # Underscore variants
        underscorized = pure_t5.replace('.', '_')
        if underscorized != pure_t5:
            _add(underscorized)
            _add(f"lora_te_{underscorized}")
            _add(f"te_{underscorized}")

    # ── Section 3: Flux (model.diffusion_model.transformer.*) ────────────
    if ckpt_base.startswith('model.diffusion_model.transformer.'):
        inner = ckpt_base[len('model.diffusion_model.'):]  # transformer.double_blocks.X
        pure_flux = ckpt_base[len('model.diffusion_model.transformer.'):]  # double_blocks.X

        _add(inner)                                          # transformer.double_blocks.X
        _add(pure_flux)                                      # double_blocks.X (bare)
        _add(f"lora_unet_{pure_flux}")                       # Kohya-style
        _add(f"lora_unet_transformer.{pure_flux}")            # rare Kohya variant
        _add(f"transformer.{pure_flux}")                     # explicit transformer. prefix

    # ── Section 4a: Z-Image / Musubi (model.diffusion_model.layers.*) ───
    if ckpt_base.startswith('model.diffusion_model.layers.'):
        inner = ckpt_base[len('model.diffusion_model.'):]  # layers.X
        pure_zi = ckpt_base[len('model.diffusion_model.layers.'):]  # X

        _add(inner)                                          # layers.*
        _add(pure_zi)                                        # pure path
        _add(f"lora_unet_{pure_zi}")                         # lora_unet variant
        _add(f"lora_unet_layers_{pure_zi}")                  # lora_unet_layers variant

    # ── Section 4b: Bare Z-Image / Musubi (layers.* without model prefix) ──
    if ckpt_base.startswith('layers.') and not ckpt_base.startswith('model.diffusion_model.layers.'):
        _add(ckpt_base)                                      # bare layers.*

    # ── Section 4c: Flux transformer entries (transformer.double_blocks.* / single_blocks.*) ──
    if ckpt_base.startswith('transformer.double_blocks.') or ckpt_base.startswith('transformer.single_blocks.'):
        _add(ckpt_base)                                      # bare transformer.*
        _add(f"model.diffusion_model.{ckpt_base}")           # with model.diffusion_model prefix
        _add(f"diffusion_model.{ckpt_base}")                 # with diffusion_model prefix

        pure_path = ckpt_base[len('transformer.'):]  # double_blocks.X
        _add(f"lora_unet_{pure_path}")                       # lora_unet variant

    # ── Section 4d: Flux keys with diffusion_model.{pure_path} (Klein normalized) ──
    if ckpt_base.startswith('transformer.double_blocks.') or ckpt_base.startswith('transformer.single_blocks.'):
        pure_path = ckpt_base[len('transformer.'):]  # double_blocks.X
        _add(f"diffusion_model.{pure_path}")                # diffusion_model.double_blocks.X
        _add(f"model.diffusion_model.{pure_path}")          # model.diffusion_model.double_blocks.X

    # ── Section 4e: Flux bare double_blocks/single_blocks (Klein 4B/9B format) ──
    if ckpt_base.startswith(('double_blocks.', 'single_blocks.')):
        _add(f"diffusion_model.{ckpt_base}")                # diffusion_model.double_blocks.X
        _add(f"model.diffusion_model.{ckpt_base}")          # model.diffusion_model.double_blocks.X
        _add(f"lora_unet_{ckpt_base}")                      # lora_unet_double_blocks.X

    # ── Section 5: Bare keys (generic fallback) ───────────────────────────
    # If the key doesn't have any standard prefix, register it as-is.
    if not ckpt_base.startswith((
        'model.', 'diffusion_model.', 'cond_stage_model.',
        'conditioner.', 'transformer.', 'layers.',
        'double_blocks.', 'single_blocks.',
    )):
        _add(ckpt_base)

    # ── Section 6: Anima block entries ────────────────────────────────────
    if ckpt_base.startswith('model.diffusion_model.'):
        inner = ckpt_base[len('model.diffusion_model.'):]
        if not inner.startswith(('input_blocks.', 'output_blocks.', 'middle_block.',
                                 'transformer.', 'layers.')):
            _add(f"lora_unet_{inner}")  # Anima-style block path

    # ── Additional: llm_adapter / llama entries ───────────────────────────
    if 'llm_adapter' in ckpt_base or 'llama' in ckpt_base:
        _add(ckpt_base)

    return variants
