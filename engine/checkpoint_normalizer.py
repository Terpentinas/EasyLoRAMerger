"""
checkpoint_normalizer.py — Architecture-Aware Checkpoint Key Normalizer

Detects model architecture from checkpoint tensor keys and normalizes keys
to a canonical form so that checkpoints from different sources (e.g., with
or without the "model.diffusion_model." prefix) can be properly matched
and merged.

Extensible design: add new architectures to CHECKPOINT_ARCHITECTURES dict.
"""

from typing import Dict, List, Tuple, Optional

# ============================================================================
# ARCHITECTURE REGISTRY — Add new models here
# ============================================================================
# Each entry defines:
#   detect:     List of substrings; if ANY key in the checkpoint contains ALL
#               of these (AND-logic per entry), the architecture matches.
#               Multiple entries = OR logic (first matching entry wins).
#   strip_prefixes: List of prefixes to remove from keys to reach canonical form.
#   canonical_form: Description of canonical form for documentation.

CHECKPOINT_ARCHITECTURES = {
    "anima": {
        "detect": [
            # Anima models use net.blocks.* pattern (FLOW-based diffusion)
            ["net.blocks"],
            # Also detect by net.final_layer as secondary marker
            ["net.final_layer"],
        ],
        "strip_prefixes": [],
        "canonical_form": "with net. prefix — keys like net.blocks.0.adaln_modulation_cross_attn.1.weight",
        "description": "Anima / Anime Diffusion (FLOW-based model with net.blocks architecture)",
    },
    "flux": {
        "detect": ["double_blocks", "single_blocks"],
        "strip_prefixes": ["model.diffusion_model."],
        "canonical_form": "no_prefix — keys like double_blocks.0.img_attn.proj.weight",
        "description": "FLUX / FLUX.1-dev / FLUX.1-schnell (Black Forest Labs)",
    },
    "lumina2": {
        "detect": [
            # Primary: cap_embedder is unique to NextDiT / Lumina2
            ["cap_embedder"],
        ],
        "strip_prefixes": ["model.diffusion_model."],
        "canonical_form": "no prefix — keys like layers.0.attention.to_q.weight, cap_embedder.0.weight",
        "description": "Lumina2 / Lumina Image 2.0 / NextDiT (Alpha-VLLM, base for Z-Image Turbo)",
    },
    "z_image": {
        "detect": [
            # Primary: Z-Image has 'layers' patterns
            ["layers", "attention"],
            # Fallback: explicit z_image marker
            ["z_image"],
        ],
        "strip_prefixes": ["model.diffusion_model."],
        "canonical_form": "no_prefix — keys like diffusion_model.layers.0.attention.to_q.weight",
        "description": "Z-Image (Lightrick / Z-inversion models)",
    },
    "sdxl": {
        "detect": [
            # SDXL has input_blocks / middle_block / output_blocks WITH model.diffusion_model. prefix
            ["model.diffusion_model", "input_blocks"],
            ["model.diffusion_model", "middle_block"],
            ["model.diffusion_model", "output_blocks"],
            # Catch-all: any key with model.diffusion_model. prefix is SDXL
            # (FLUX and Z-Image are checked first, so they won't reach here)
            ["model.diffusion_model"],
        ],
        "strip_prefixes": [],
        "canonical_form": "with model.diffusion_model. prefix — keys like model.diffusion_model.input_blocks.0.0.weight",
        "description": "Stable Diffusion XL",
    },
    "sd15": {
        "detect": [
            # SD1.5 has input_blocks WITHOUT model.diffusion_model prefix (but has diffusion_model)
            # Only match if NOT sdxl (checked by detect order)
            ["diffusion_model", "input_blocks"],
            ["diffusion_model", "middle_block"],
            ["diffusion_model", "output_blocks"],
        ],
        "strip_prefixes": [],
        "canonical_form": "with diffusion_model. prefix — keys like diffusion_model.input_blocks.0.0.weight",
        "description": "Stable Diffusion 1.5",
    },
}


def _matches_architecture(keys: List[str], arch_config: dict) -> bool:
    """
    Check if any key pattern in arch_config['detect'] matches the given keys.
    Each entry in 'detect' is EITHER a string (all keys are checked for that
    substring) OR a list of strings (all substrings must be present, AND-logic).
    Multiple entries = OR logic.
    """
    detect_patterns = arch_config.get("detect", [])
    for pattern in detect_patterns:
        if isinstance(pattern, str):
            # Single substring — any key containing it
            if any(pattern in k for k in keys):
                return True
        elif isinstance(pattern, list):
            # List of substrings — ALL must be present in at least one key each
            # (but not necessarily the same key)
            if all(any(sub in k for k in keys) for sub in pattern):
                return True
    return False


def detect_checkpoint_architecture(keys: List[str]) -> str:
    """
    Detect the model architecture from a list of tensor keys.

    Checks architectures in order of specificity (most specific first):
        flux → lumina2 → z_image → baked_sd15 → sdxl → sd15 → unknown

    The ``baked_sd15`` pre-check catches SD1.5 checkpoints whose UNet keys
    have been prefixed with ``model.diffusion_model.`` (e.g. by the baking
    pipeline).  Without this, such checkpoints incorrectly match the SDXL
    catch-all detector (``["model.diffusion_model"]``) before the SD15
    detector can fire.

    Args:
        keys: List of tensor key strings from the checkpoint header.

    Returns:
        Architecture name string: "flux", "lumina2", "z_image", "sdxl",
        "sd15", or "unknown".
    """
    # Ordered by specificity: Anima (net.blocks is unique) first,
    # then FLUX (double_blocks are unique),
    # then Lumina2 (cap_embedder is unique to NextDiT),
    # then mixed patterns, then SD variants.
    check_order = ["anima", "flux", "lumina2", "z_image", "sdxl", "sd15"]

    # ── Pre-check: baked SD1.5 ─────────────────────────────────────────
    # Baked SD1.5 checkpoints have model.diffusion_model.input_blocks.xxx
    # (baker adds "model." prefix to UNet keys) which would match the SDXL
    # catch-all below.  Detect them by looking for SD1.5-specific CLIP keys.
    # "cond_stage_model.transformer.text_model" is exclusive to SD1.5 —
    # SDXL uses "model.conditioner.embedders.0.transformer.text_model" or
    # "model.text_model".
    if _matches_architecture(keys, {"detect": [
        ["cond_stage_model.transformer.text_model", "model.diffusion_model.input_blocks"],
    ]}):
        return "sd15"

    for arch_name in check_order:
        arch_config = CHECKPOINT_ARCHITECTURES.get(arch_name, {})
        if _matches_architecture(keys, arch_config):
            return arch_name

    return "unknown"


def normalize_checkpoint_key(key: str, architecture: str) -> str:
    """
    Normalize a single checkpoint key to canonical form for the given architecture.

    For FLUX / Z-Image: strips 'model.diffusion_model.' prefix.
    For SDXL / SD1.5: keeps existing prefix as-is.

    Args:
        key: Original tensor key.
        architecture: Architecture name from detect_checkpoint_architecture().

    Returns:
        Normalized key in canonical form.
    """
    arch_config = CHECKPOINT_ARCHITECTURES.get(architecture, {})
    strip_prefixes = arch_config.get("strip_prefixes", [])

    normalized = key
    for prefix in strip_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break  # only strip first matching prefix

    return normalized


def normalize_checkpoint_header(
    header: Dict[str, dict],
    architecture: Optional[str] = None,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Normalize all keys in a checkpoint header to canonical form.

    Args:
        header: Dict mapping key -> {shape, dtype, ...} as read from safetensors.
        architecture: Detected architecture. If None, auto-detect.

    Returns:
        (normalized_header, key_mapping) where:
            normalized_header: Same values, but with canonical keys.
            key_mapping: Dict mapping canonical_key -> original_key.
    """
    if not header:
        return {}, {}

    if architecture is None:
        architecture = detect_checkpoint_architecture(list(header.keys()))

    normalized = {}
    mapping = {}

    for orig_key, value in header.items():
        norm_key = normalize_checkpoint_key(orig_key, architecture)
        # Handle collisions: keep first occurrence with warning
        if norm_key not in normalized:
            normalized[norm_key] = value
            mapping[norm_key] = orig_key
        else:
            import logging
            logging.warning(
                f"normalize_checkpoint_header: key collision — "
                f"'{orig_key}' and '{mapping[norm_key]}' both normalize to '{norm_key}'. "
                f"Keeping first occurrence."
            )

    return normalized, mapping


