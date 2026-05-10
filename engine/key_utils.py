"""
Unified key categorisation and architecture detection.

Consolidates 7 implementations found across the codebase:

  1. ``categorize_key()``              – :file:`utils.py`
  2. ``categorize_checkpoint_key()``   – :file:`utils.py`
  3. ``_categorize_lora_key()``        – :file:`engine/baking_processor_matching.py`
  4. ``_is_te_key()``                  – :file:`engine/serialization_factory.py`
  5. ``_is_vae_key()``                 – :file:`engine/musubi_checkpoint_studio.py`
  6. ``_is_te_key()``                  – :file:`engine/musubi_checkpoint_studio.py`
  7. ``_is_clip_key()``                – :file:`engine/musubi_checkpoint_studio.py`
  8. ``detect_checkpoint_architecture`` – :file:`engine/musubi_checkpoint_studio.py` (fallback)

Usage
-----
>>> from engine.key_utils import categorize_key, is_te_key, is_vae_key, is_clip_key, detect_architecture

>>> categorize_key("model.diffusion_model.input_blocks.0.0.weight")
'unet'

>>> categorize_key("te1.text_model.encoder.layers.0.weight")
'te'

>>> categorize_key("first_stage_model.decoder.conv_in.weight")
'vae'

>>> categorize_key("visual.encoder.layers.0.weight")
'clip'

>>> is_te_key("te2.text_model.encoder.layers.0.weight")
True
"""

from typing import Iterable, Optional, Set


# ===================================================================
# Pattern collections
# ===================================================================

# ── Text Encoder patterns ──────────────────────────────────────────
# Checked FIRST to avoid 'encoder' in text_model.encoder.layers
# matching the VAE 'encoder' pattern.
_TE_PATTERNS: Set[str] = {
    # From categorize_checkpoint_key (utils.py)
    "te1.", "te.", "text_model", "conditioner",
    "clip_l", "t5", "transformer.text_model",
    "token_embedding", "positional_embedding",
    # From _categorize_lora_key (baking_processor_matching.py) – extra lora_te variants
    "lora_te", "lora_te1_", "lora_te2_",
    "lora_te1_text_model.", "lora_te2_text_model.",
    # From _is_te_key (serialization_factory.py)
    "transformer.model.layers",
    "text_encoder.",
    "clip_g.",
    # From _is_te_key (musubi_checkpoint_studio.py)
    "te2.",
}

# ── VAE patterns ───────────────────────────────────────────────────
_VAE_PATTERNS: Set[str] = {
    "first_stage_model", ".vae.", "vae.", "vae_", "vae/",
    "decoder", "encoder", "post_quant_conv",
    ".conv", ".conv1", ".conv2",    # broad – matches after TE check
    # From _is_vae_key (musubi_checkpoint_studio.py)
    "quant_conv.",
}

# ── CLIP visual patterns ───────────────────────────────────────────
_CLIP_PATTERNS: Set[str] = {
    "visual.", "clip.", "clip_visual", "vision_model",
    "image_encoder", "image_projection",
    # From _is_clip_key (musubi_checkpoint_studio.py)
    "clip.",
}

# ── UNet / diffusion model patterns ────────────────────────────────
_UNET_PATTERNS: Set[str] = {
    "model.", "model_", "model/", "unet", "diffusion",
    "time_embed", "input_blocks", "middle_block", "output_blocks",
    "out.", "out_", "conv_in", "conv_out", "norm", "attentions",
    "resnets", "down_blocks", "up_blocks",
    "double_blocks", "single_blocks", "img_attn", "txt_attn",
    "img_mlp", "txt_mlp", "feed_forward", "linear1", "linear2",
    "proj", "layers", "attention", "to_q", "to_k", "to_v", "to_out",
    "mlp.fc", "adaln", "modulation", "transformer", "diffusion_model",
}

# ── Architecture-detection patterns ────────────────────────────────
_ARCHITECTURE_PATTERNS = [
    ("Anima",   "net.blocks"),
    ("Flux",    "double_blocks"),
    ("Flux",    "single_blocks"),
    ("Lumina2", "cap_embedder"),
    ("Z-Image", "z_image"),
    ("Z-Image", "layers"),     # heuristic – Z-Image often has 'layers' in attention
    ("SDXL",    "model.diffusion_model"),
    ("SD1.5",   "diffusion_model"),
]


# ===================================================================
# Public API
# ===================================================================

def categorize_key(key: str) -> str:
    """
    Categorise a key (LoRA or checkpoint) into a component name.

    Returns one of ``'unet'``, ``'te'``, ``'clip'``, ``'vae'``, ``'other'``.

    **Order of checks (important)**
    The function checks TE patterns **before** VAE patterns so that
    ``text_model.encoder.layers`` is classified as ``'te'`` rather than
    ``'vae'`` (which also checks for ``'encoder'``).
    """
    key_lower = key.lower()

    # 1. Text Encoder (checked first – see docstring)
    if any(p in key_lower for p in _TE_PATTERNS):
        return 'te'

    # 2. VAE
    if any(p in key_lower for p in _VAE_PATTERNS):
        return 'vae'

    # 3. CLIP visual
    if any(p in key_lower for p in _CLIP_PATTERNS):
        return 'clip'

    # 4. UNet / diffusion model
    if any(p in key_lower for p in _UNET_PATTERNS):
        return 'unet'

    # 5. Unrecognised
    return 'other'


def is_te_key(key: str) -> bool:
    """Return ``True`` if *key* belongs to a text encoder (CLIP/T5)."""
    return categorize_key(key) == 'te'


def is_vae_key(key: str) -> bool:
    """Return ``True`` if *key* belongs to the VAE."""
    return categorize_key(key) == 'vae'


def is_clip_key(key: str) -> bool:
    """Return ``True`` if *key* belongs to the CLIP visual encoder."""
    return categorize_key(key) == 'clip'


def is_unet_key(key: str) -> bool:
    """Return ``True`` if *key* belongs to the UNet / diffusion model."""
    return categorize_key(key) == 'unet'


def detect_architecture(keys: Iterable[str]) -> str:
    """
    Detect checkpoint architecture from a collection of keys.

    Returns one of ``'Anima'``, ``'Flux'``, ``'Lumina2'``, ``'Z-Image'``,
    ``'SDXL'``, ``'SD1.5'``, ``'Unknown'``.

    Patterns are checked in order — the first match wins.
    """
    key_set: Set[str] = set(keys)
    for arch, pattern in _ARCHITECTURE_PATTERNS:
        if any(pattern in k for k in key_set):
            return arch
    return "Unknown"
