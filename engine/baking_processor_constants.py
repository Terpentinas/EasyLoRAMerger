"""
Architecture-aware registries for extensible key matching.

These module-level constants were extracted from baking_processor.py
to reduce file size and centralize registration tables used by
_build_reverse_key_map(), _find_matching_keys(), and key normalization.
"""

from typing import List, Tuple

# ------------------------------------------------------------------
# Registry of underscore-to-dot conversion patterns
# ------------------------------------------------------------------

# Each entry: (regex_pattern, replacement_template, description)
# Extend this list when adding support for new architectures.
UNDERSCORE_TO_DOT_PATTERNS: List[Tuple[str, str, str]] = [
    # SD1.5/SDXL UNet + TE attention: attn1_to_k → attn1.to.k
    (r'(attn\d)_(to)_(q|k|v)', r'\1.\2.\3', 'attn1_to_k → attn1.to.k'),
    # attn2_to_out → attn2.to_out
    (r'(attn\d)_(to_out)', r'\1.\2', 'attn2_to_out → attn2.to_out'),
    # ff_net → ff.net (SD1.5 Diffusers feed-forward)
    (r'ff_net_(\d+)', r'ff.net.\1', 'ff_net_0 → ff.net.0'),
    (r'ff_net\.', r'ff.net.', 'ff_net. → ff.net.'),
    # proj_in → proj.in (unlikely but safe)
    (r'(proj)_(in|out)', r'\1.\2', 'proj_in → proj.in'),
    # self_attn → self.attn
    (r'(self|cross)_attn', r'\1_attn', 'self_attn preserved'),
    # Transformer blocks: _transformer_ → .transformer.
    (r'_transformer_', r'.transformer.', '_transformer_ → .transformer.'),
    # text_model → text.model (SD1.5 Diffusers TE)
    (r'text_model_', r'text.model.', 'text_model_ → text.model.'),
    # Final fallback: double underscore → single dot
    (r'__', r'.', '__ → .'),
]

# ------------------------------------------------------------------
# Registry of LoRA prefix → checkpoint prefix mappings
# ------------------------------------------------------------------

# Each entry: (lora_prefix_to_strip, checkpoint_prefix_to_add, description)
# Used by _lora_key_to_checkpoint_key() during Strategy 2 (prefix-aware conversion).
LORA_TO_CHECKPOINT_PREFIXES: List[Tuple[str, str, str]] = [
    # Standard ComfyUI
    ('diffusion_model.', 'model.diffusion_model.', 'diffusion_model. → model.diffusion_model.'),
    # Standard SD1.5 checkpoint TE path
    ('text_model.', 'cond_stage_model.transformer.text_model.', 'text_model. → cond_stage_model.transformer.text_model.'),
    # Kohya-style lora_unet_ prefix → model.diffusion_model.
    ('lora_unet_', 'model.diffusion_model.', 'lora_unet_ → model.diffusion_model.'),
    # Kohya-style lora_te_ prefix → cond_stage_model.transformer.text_model.
    ('lora_te_', 'cond_stage_model.transformer.text_model.', 'lora_te_ → cond_stage_model.transformer.text_model.'),
    # Kohya-style lora_te1_ prefix → cond_stage_model.transformer.text_model.
    ('lora_te1_', 'cond_stage_model.transformer.text_model.', 'lora_te1_ → cond_stage_model.transformer.text_model.'),
    # Kohya-style lora_te2_ prefix → conditioner.embedders.0.transformer.text_model.
    ('lora_te2_', 'conditioner.embedders.0.transformer.text_model.', 'lora_te2_ → conditioner.embedders.0.transformer.text_model.'),
    # SDXL Conditioner
    ('conditioner.', 'conditioner.', 'conditioner. (passthrough)'),
    # Flux transformer
    ('transformer.', 'model.diffusion_model.transformer.', 'transformer. → model.diffusion_model.transformer.'),
    # Z-Image base model prefix
    ('model.', 'model.', 'model. (passthrough)'),
    # TE2 short form
    ('te2.', 'conditioner.embedders.0.transformer.text_model.', 'te2. → conditioner.embedders.0.transformer.text_model.'),
    # TE1 short form
    ('te1.', 'cond_stage_model.transformer.text_model.', 'te1. → cond_stage_model.transformer.text_model.'),
    # TE short form (alias for te1)
    ('te.', 'cond_stage_model.transformer.text_model.', 'te. → cond_stage_model.transformer.text_model.'),
]
