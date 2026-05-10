"""
Serialization Factory for Ariadne Project.
Restores original trainer keys using master key map, ensuring perfect compatibility.
"""
import torch
from typing import Dict, Optional, Tuple
try:
    from .klein_normalizer import get_format_style  # optional
except ImportError:
    from klein_normalizer import get_format_style  # optional

from ..utils import ProgressTracker


def _clean_double_underscores(s: str) -> str:
    """Collapse repeated underscores into single underscores."""
    while '__' in s:
        s = s.replace('__', '_')
    return s


# Patterns to match suffixes (in order of priority, shared by TE and non-TE branches)
_SUFFIX_PATTERNS = [
    (r'\.lora_A\.weight$', '.lora_A.weight'),
    (r'\.lora_B\.weight$', '.lora_B.weight'),
    (r'\.lora_down\.weight$', '.lora_down.weight'),
    (r'\.lora_up\.weight$', '.lora_up.weight'),
    (r'\.alpha$', '.alpha'),
    (r'\.weight$', '.weight'),  # Generic weight (should come after lora patterns)
    (r'\.bias$', '.bias'),
]


def _is_te_key(key: str) -> bool:
    """Return True if key matches a text encoder (CLIP/Llama/T5) pattern.

    Delegates to :func:`engine.key_utils.is_te_key`.
    """
    from ..engine.key_utils import is_te_key as _is_te
    return _is_te(key)


def _match_suffix(key: str, patterns):
    """Find the first matching suffix pattern for a key.

    Args:
        key: The key string to search in.
        patterns: List of (regex_pattern, suffix_string) tuples to match.

    Returns:
        Tuple of (matched_pattern, matched_suffix) or (None, "") if no match.
    """
    import re
    for pattern, suffix in patterns:
        if re.search(pattern, key):
            return pattern, suffix
    return None, ""


def _convert_to_target_format(key: str, separator_style: str, naming_style: str) -> str:
    """
    Convert normalized key to target format using specified separator and naming style.
    
    Args:
        key: Normalized key (dot notation, lora_A/lora_B naming).
        separator_style: Either "dot" (keep dot notation) or "underscore" (convert dot‑index patterns to underscores).
        naming_style: Either "lora_a_b" (keep lora_A/lora_B) or "lora_down_up" (convert to lora_down/lora_up).
    
    Returns:
        Key in target format.
    """
    import re
    
    new_key = key
    is_te = _is_te_key(key)
    is_flux = 'double_blocks' in key or 'single_blocks' in key
    
    if is_te:
        # Handle TE prefix based on separator style
        if separator_style in ("underscore", "hybrid"):
            if key.startswith("te1."):
                new_key = "lora_te1_" + key[4:]
            elif key.startswith("te."):
                new_key = "lora_te_" + key[3:]
            elif key.startswith("transformer."):
                # transformer.text_model.encoder.layers.N.xxx → lora_te_text_model.encoder.layers.N.xxx
                # (dots will be converted to underscores later for underscore style)
                new_key = "lora_te_" + key[len("transformer."):]
            elif key.startswith("text_model.encoder."):
                # text_model.encoder.layers.N.xxx → lora_te_layers.N.xxx
                # (dots will be converted to underscores later for underscore style)
                new_key = "lora_te_" + key[len("text_model.encoder."):]
            # If already has lora_te prefix, keep as is
            elif key.startswith("lora_te"):
                new_key = key
        else:  # dot style
            # For dot style (comfy_native), add text_encoders. prefix.
            # ComfyUI's model_lora_keys_clip() builds key_map entries as:
            #   "text_encoders.{model_key[:-len('.weight')]}" → model_key
            # The LoRA adapter then looks for "{key_map_key}.lora_B.weight" in
            # the merged LoRA, so we must match the text_encoders. prefix.
            # This is added AFTER pattern extraction below, not here.
            pass
        
        # Find matching suffix pattern using shared helper
        matched_pattern, matched_suffix = _match_suffix(new_key, _SUFFIX_PATTERNS)
        
        if matched_pattern:
            # Extract base part (everything before the matched pattern)
            base = re.sub(matched_pattern, '', new_key)
            # Apply separator conversion to base
            if separator_style in ("underscore", "hybrid"):
                # Convert all dots in base to underscores, then clean
                base = _clean_double_underscores(base.replace(".", "_"))
            else:  # dot style
                # Add text_encoders. prefix for ComfyUI's model_lora_keys_clip matching.
                # The key_map expects: "text_encoders.{model_key}" which means the LoRA
                # must have "{text_encoders.{model_key}}.lora_A.weight".
                # Skip if already has text_encoders. or lora_te prefix.
                if not base.startswith("text_encoders.") and not base.startswith("lora_te"):
                    base = "text_encoders." + base
            # Reconstruct with proper suffix
            new_key = base + matched_suffix
        else:
            # No pattern matched, apply separator conversion to whole key
            if separator_style in ("underscore", "hybrid"):
                new_key = _clean_double_underscores(new_key.replace(".", "_"))
            else:  # dot style
                # Add text_encoders. prefix for TE keys in dot style
                if not new_key.startswith("text_encoders.") and not new_key.startswith("lora_te"):
                    if "te." in new_key or "te1." in new_key or new_key.startswith("transformer.") or new_key.startswith("text_model.encoder."):
                        new_key = "text_encoders." + new_key
    else:
        # Non‑TE keys (UNet and other)
        unet_patterns = ["input_blocks", "middle_block", "output_blocks", "time_embed"]
        # Add lora_unet_ prefix only for underscore style (SDXL expectation)
        if separator_style in ("underscore", "hybrid") and any(p in key for p in unet_patterns) and not key.startswith("lora_unet_"):
            # Replace diffusion_model. with lora_unet_ if present, otherwise prepend
            if key.startswith("diffusion_model."):
                new_key = "lora_unet_" + key[len("diffusion_model."):]
            else:
                new_key = "lora_unet_" + key
        
        # Find matching suffix pattern using shared helper
        matched_pattern, matched_suffix = _match_suffix(new_key, _SUFFIX_PATTERNS)
        
        if matched_pattern:
            base = re.sub(matched_pattern, '', new_key)
            # Apply dot‑index conversion for underscore style
            if separator_style in ("underscore", "hybrid") and not is_flux:
                # Convert all dots in base to underscores, then clean
                base = _clean_double_underscores(base.replace(".", "_"))
            # else dot style: keep base unchanged
            new_key = base + matched_suffix
        else:
            # No pattern matched, apply dot‑index conversion for underscore style
            if separator_style in ("underscore", "hybrid") and not is_flux:
                # Convert all dots in key to underscores, then clean
                new_key = _clean_double_underscores(new_key.replace(".", "_"))
            # else dot style: keep unchanged
    
    # Apply naming style conversion
    if naming_style == "lora_down_up" and not is_flux:
        if 'lora_A.weight' in new_key:
            new_key = new_key.replace('lora_A.weight', 'lora_down.weight')
        if 'lora_B.weight' in new_key:
            new_key = new_key.replace('lora_B.weight', 'lora_up.weight')
    # else lora_a_b: keep as is
    
    # Remove any double underscores introduced by the above steps
    new_key = _clean_double_underscores(new_key)
    
    return new_key


# Moved from klein_normalizer.py
def _is_musubi_key(key: str) -> bool:
    """Return True if key matches Musubi native format (includes Anima)."""
    musubi_patterns = [
        "lora_unet_layers",
        "lora_unet_double_blocks",
        "lora_unet_blocks",  # Anima: lora_unet_blocks_N_... (trainer format)
        "lora_te_res_blocks",
        "lora_te_layers",    # Anima: lora_te_layers_N_... (TE trainer format)
    ]
    return any(pattern in key for pattern in musubi_patterns)


def _ensure_kohya_naming(key: str, naming_style: str = "lora_down_up") -> str:
    """
    Convert lora_A/lora_B naming to Kohya standard lora_down/lora_up,
    unless naming_style is "lora_a_b" (keep original naming).
    Leaves other keys unchanged.
    """
    is_flux = 'double_blocks' in key or 'single_blocks' in key
    if naming_style == "lora_down_up" and not is_flux:
        if 'lora_A.weight' in key:
            key = key.replace('lora_A.weight', 'lora_down.weight')
        if 'lora_B.weight' in key:
            key = key.replace('lora_B.weight', 'lora_up.weight')
    # If naming_style is "lora_a_b", do nothing.
    return key




def guess_original_key(norm_key: str, master_key_map: Dict[str, str], target_format: Optional[str] = None) -> str:
    """
    Heuristic fallback for normalized keys not present in master_key_map.
    Attempts to convert to a plausible original format based on target_format.
    If target_format is None, falls back to SDXL underscore style.
    """
    # If mapping exists, use it (should have been caught earlier)
    if norm_key in master_key_map:
        return master_key_map[norm_key]
    
    # Determine separator and naming style based on target_format
    if target_format is None:
        separator_style = "underscore"
        naming_style = "lora_down_up"
    else:
        separator_style, naming_style = get_format_style(target_format)
    
    guessed = _convert_to_target_format(norm_key, separator_style, naming_style)
    print(f"WARNING: No mapping for normalized key {norm_key}, guessing original as {guessed}")
    return guessed


def finalize_for_save(merged_tensors: Dict[str, torch.Tensor],
                      master_key_map: Dict[str, str],
                      target_format: Optional[str] = None,
                      meta_a: Optional[Dict[str, str]] = None,
                      meta_b: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Convert merged normalized tensors back to original trainer keys and produce merged metadata.
    
    Args:
        merged_tensors: State dict with normalized keys (mathematical keys).
        master_key_map: Mapping from normalized key -> original key.
        target_format: Detected format of primary LoRA (LoRA A) used to decide separator and naming style.
        meta_a: Metadata from LoRA A (optional).
        meta_b: Metadata from LoRA B (optional).
    
    Returns:
        restored_dict: State dict with original keys, ready for saving.
        final_metadata: Merged metadata to embed in the saved file.
    """
    print("🔄 Restoring original trainer keys...")
    # Determine separator and naming style based on target format
    if target_format is None:
        # fallback to SDXL underscore style for backward compatibility
        separator_style = "underscore"
        naming_style = "lora_down_up"
    else:
        separator_style, naming_style = get_format_style(target_format)
    restored = {}
    missing_map = 0
    guessed_map = 0
    overridden_count = 0

    total_keys = len(merged_tensors)
    with ProgressTracker(total=total_keys, desc="Restoring keys") as restore_progress:
        for norm_key, tensor in merged_tensors.items():
            if norm_key in master_key_map:
                orig_key = master_key_map[norm_key]
                if _is_musubi_key(orig_key):
                    # Musubi/Anima override: use converted target‑format nomenclature instead
                    # of restoring the original trainer key. The normalized key (norm_key) is
                    # already in the correct format (e.g. diffusion_model.blocks.N.* for Anima).
                    if separator_style == "underscore" and naming_style == "lora_down_up":
                        # SDXL-style musubi: convert norm_key to underscore/lora_down_up
                        guessed = _convert_to_target_format(norm_key, separator_style, naming_style)
                    else:
                        # Dot-style musubi (Anima): keep the normalized key format
                        # diffusion_model.blocks.N.cross_attn.out_proj.lora_A.weight
                        # with lora_A/lora_B naming (required by LoRAAdapter.load() diffusers2 format)
                        # The naming_style may be "lora_down_up" which would convert lora_A→lora_down,
                        # but LoRAAdapter.load() needs lora_A/lora_B (diffusers2 format), so force lora_a_b.
                        guessed = _ensure_kohya_naming(norm_key, "lora_a_b")
                    restored[guessed] = tensor
                    guessed_map += 1
                    overridden_count += 1
                else:
                    # Priority 1 (Standard): use Identity Map to restore 1:1 original keys.
                    # Apply naming style conversion (lora_A/lora_B → lora_down/lora_up) if needed.
                    orig_key = _ensure_kohya_naming(orig_key, naming_style)
                    restored[orig_key] = tensor
            else:
                # Fallback: guess original key using heuristic, respecting target format
                guessed = guess_original_key(norm_key, master_key_map, target_format)
                guessed = _ensure_kohya_naming(guessed, naming_style)
                restored[guessed] = tensor
                missing_map += 1
                if guessed != norm_key:
                    guessed_map += 1

            restore_progress += 1

    if missing_map or overridden_count:
        print(f"⚠️ {missing_map} keys missing mapping ({guessed_map} guessed), {overridden_count} Musubi keys overridden")
    
    # Ensure alpha keys are present (if any)
    # We rely on alpha keys already being in merged_tensors (if they were present).
    
    # Merge metadata using MetadataMerger, then sign via factory
    from ..validation import MetadataMerger
    from .metadata_factory import finalize_metadata
    meta_a_safe = meta_a or {}
    meta_b_safe = meta_b or {}
    merged_meta = MetadataMerger.merge(meta_a_safe, meta_b_safe, "preserve_b")
    final_metadata = finalize_metadata(
        metadata=merged_meta,
        mode="preserve_b",
        component="merger",
        extra_fields={"merged_key_count": str(len(merged_tensors))},
    )
    
    print(f"✅ Restored {len(restored)} keys to original format")
    return restored, final_metadata

