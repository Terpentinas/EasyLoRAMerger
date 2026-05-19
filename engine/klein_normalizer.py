# klein_normalizer.py - UNIVERSAL NORMALIZER WITH ALL FORMATS (CLEANED)
import torch
import re
from collections import defaultdict
from typing import Dict, List
try:
    from ..utils import safe_get_rank, silent_pad_or_truncate
except ImportError:
    from utils import safe_get_rank, silent_pad_or_truncate

# ==================== PERFORMANCE & REPAIR CONFIGURATION ====================

ENABLE_AUTO_REPAIR = True
ENABLE_BATCH_RANK_FIX = True
DEBUG_KEY_DIFF = False  # Enable for debugging conversions

_current_mapping_ref = None  # optional mapping dict for identity tracking

# Cache for conversion results keyed by (format_type, key_fingerprint)
# Stores the converted state dict to avoid redundant conversions.
_NORMALIZE_CACHE: dict = {}

# Increment this version counter to bust all cached normalization results.
# Use when normalization logic changes (e.g., new key conversion rules).
# Version 1 → 2: SD1.5 TE key normalization fix (text_model. prefix handling)
NORMALIZE_CACHE_VERSION = 2


# ==================== PRE-COMPILED REGEX PATTERNS ====================

LAYER_PATTERN = re.compile(r'layers_(\d+)_')
BLOCK_PATTERN = re.compile(r'(double|single)_blocks_(\d+)_')
ATTN_PATTERN = re.compile(r'(img|txt)_attn_(proj|qkv)')
TO_OUT_PATTERN = re.compile(r'to_out_(\d+)')
MLP_PATTERN = re.compile(r'(img|txt)_mlp_(\d+)')
FEED_FORWARD_PATTERN = re.compile(r'feed_forward_w(\d)')
RESBLOCK_PATTERN = re.compile(r'resblocks\.(\d+)_')
ANIMA_BLOCK_PATTERN = re.compile(r'blocks_(\d+)_')

# ==================== UTILITY FUNCTIONS ====================

def fast_cleanup_key(key):
    """Standardizes underscores into dots for Flux/Z-Image compatibility."""
    # 1. Remove common prefixes that shouldn't be there
    key = key.replace("lora_unet_", "diffusion_model.")
    
    # 2. Fix the 'attention_to' vs 'attention.to' (Z-Image & Flux)
    key = key.replace("attention_to_", "attention.to_")
    
    # 3. Fix the 'img_attn_qkv' and 'txt_attn_proj' (Flux 4B/9B)
    # This handles the mismatch where one side has dots and the other underscores
    replacements = {
        "img_attn_": "img_attn.",
        "txt_attn_": "txt_attn.",
        "img_mlp_": "img_mlp.",
        "txt_mlp_": "txt_mlp.",
        "to_out_": "to_out.",
        "feed_forward_w": "feed_forward.w"
    }
    
    for old, new in replacements.items():
        key = key.replace(old, new)

    # 4. Final safety: remove double dots if they were accidentally created
    while ".." in key:
        key = key.replace("..", ".")
        
    return key

def debug_key_change(old, new):
    """Print key transformations when debugging is enabled."""
    if DEBUG_KEY_DIFF and old != new:
        print(f"ðŸ”§ {old} â†’ {new}")

def safe_insert(converted, key, value, original_key=None):
    """
    Prevent silent overwriting if two keys collapse into one.
    Keeps first occurrence, warns on collision.
    """
    if key in converted:
        print("WARNING: Key collision detected!")
        if original_key:
            print(f"   Original: {original_key}")
        print(f"   Collides with existing: {key}")
        print("   Keeping first occurrence.")
        return
    converted[key] = value
    # Store mapping if global mapping ref exists and original_key provided
    if original_key is not None and _current_mapping_ref is not None:
        _current_mapping_ref[key] = original_key

def get_alpha_value(state_dict, key, default=16.0):
    """Determine alpha value, checking for prefixed metadata from previous merges."""
    
    # 1. Check for alpha key inside the state_dict (tensors)
    base_key = (key.replace(".lora_A.weight", "")
                    .replace(".lora_B.weight", "")
                    .replace(".lora_down.weight", "")
                    .replace(".lora_up.weight", "")
                    .replace(".lora.down.weight", "")
                    .replace(".lora.up.weight", ""))
    
    alpha_patterns = [
        f"{base_key}.alpha",
        f"{base_key}_alpha",
        key.replace("lora_A.weight", "alpha"),
        key.replace("lora_B.weight", "alpha"),
    ]
    
    for ak in alpha_patterns:
        if ak in state_dict:
            val = state_dict[ak]
            return val.item() if isinstance(val, torch.Tensor) else float(val)
    
    # 2. METADATA CHECK (3-way check for EasyLoRAMerger styles)
    meta = getattr(state_dict, 'metadata', {})
    if meta:
        # Check standard, then Preserve A prefix, then Preserve B prefix
        for mk in ['ss_network_alpha', 'lora_a_ss_network_alpha', 'lora_b_ss_network_alpha']:
            if mk in meta:
                try:
                    return float(meta[mk])
                except:
                    continue
    
# 3. FALLBACK: RANK INFERENCE (For 'None' mode merges)
    if key in state_dict:
        tensor = state_dict[key]
        if hasattr(tensor, 'shape') and len(tensor.shape) >= 2:
            # Pick dim 0 for A/down, dim 1 for B/up
            if "lora_A" in key or "lora_down" in key or "lora.down" in key:
                rank = tensor.shape[0]
            else:
                rank = tensor.shape[1]
            return float(rank)
    
    return default

# ==================== DETECTION & REPAIR ====================

def detect_broken_lora(state_dict):
    """Detect common LoRA corruption issues."""
    issues = []

    a_keys = [k for k in state_dict if "lora_A" in k]
    b_keys = [k for k in state_dict if "lora_B" in k]

    # If there are no LoRA keys, this is not a LoRA - skip detection
    if len(a_keys) == 0 and len(b_keys) == 0:
        return issues

    if len(a_keys) != len(b_keys):
        issues.append("Mismatch between lora_A and lora_B counts")

    for key, tensor in state_dict.items():
        if hasattr(tensor, "shape"):
            # Skip alpha tensors - they can be 0D (scalars)
            if '.alpha' in key and len(tensor.shape) == 0:
                continue  # This is fine!
            # Only check shape for LoRA-related tensors
            if "lora" in key.lower() and len(tensor.shape) < 2 and '.alpha' not in key:
                issues.append(f"Bad tensor shape: {key}")

    if issues:
        print("WARNING: Broken LoRA detected:")
        for i in issues[:5]:
            print("  -", i)

    return issues

def repair_missing_alpha(state_dict):
    """Add missing alpha keys."""
    repaired = dict(state_dict)
    added = 0

    for key in list(state_dict.keys()):
        if "lora_A.weight" in key:
            alpha_key = key.replace(".lora_A.weight", ".alpha")
            if alpha_key not in repaired:
                from .scale_utils import find_alpha_value
                meta = getattr(state_dict, 'metadata', {})
                alpha_val = find_alpha_value(state_dict, key, default=16.0, metadata=meta)
                if alpha_val is None:
                    # Rank inference fallback (original behavior)
                    alpha_val = 16.0
                repaired[alpha_key] = torch.tensor(alpha_val)
                added += 1

    if added:
        print(f"ðŸ›  Added {added} missing alpha keys")

    return repaired

def batch_fix_lora_ranks(state_dict, target_rank=None):
    """Fix mismatched LoRA ranks across a batch."""
    if not ENABLE_BATCH_RANK_FIX:
        return state_dict

    grouped = defaultdict(list)
    modified_count = 0

    for key, tensor in state_dict.items():
        if "lora_" in key and hasattr(tensor, "shape"):
            grouped[key.split(".lora_")[0]].append((key, tensor))

    fixed = dict(state_dict)

    for group, entries in grouped.items():
        ranks = []
        for k, t in entries:
            if len(t.shape) >= 2:
                ranks.append(min(t.shape[:2]))

        if not ranks:
            continue

        desired_rank = target_rank or min(ranks)
        
        # Check if any tensor needs adjustment
        for key, tensor in entries:
            current_rank = safe_get_rank(tensor, key)
            if current_rank != desired_rank:
                fixed[key] = silent_pad_or_truncate(tensor, desired_rank, key)
                modified_count += 1

    if modified_count > 0:
        print(f"WARNING: Batch rank fix adjusted {modified_count} tensors")
    else:
        print("* Batch rank fix complete (no changes needed)")
    
    return fixed

def auto_repair_lora(state_dict):
    """Automatically detect and repair common LoRA issues."""
    if not ENABLE_AUTO_REPAIR:
        return state_dict

    # Preserve metadata
    metadata = getattr(state_dict, "metadata", None)
    
    issues = detect_broken_lora(state_dict)
    repaired = dict(state_dict)  # Convert to dict to avoid metadata issues

    if issues:
        repaired = repair_missing_alpha(repaired)
        repaired = batch_fix_lora_ranks(repaired)
        
        # Restore metadata if it was a safetensors object
        if metadata and hasattr(state_dict, "metadata"):
            # If we need to preserve metadata, we'd need to return to safetensors format
            # This is complex - maybe just log a warning
            print("Note: Metadata preservation requires safetensors object")

    return repaired

def accelerate_lora_load(state_dict):
    """Run repair pipeline on LoRA load."""
    print("* Accelerating LoRA load...")
    state_dict = auto_repair_lora(state_dict)
    return state_dict

# ==================== FORMAT DETECTION ====================

def detect_lora_format(state_dict, mapping_ref=None):
    """Detect which LoRA format we're dealing with."""
    if not state_dict:
        return "unknown"
    
    keys = list(state_dict.keys())
    if not keys:
        return "unknown"

    # Infer separator and naming styles (for universal key normalization)
    if mapping_ref is not None:
        separator_style = infer_separator_style(keys)
        naming_style = infer_naming_style(keys)
        mapping_ref["__separator_style__"] = separator_style
        mapping_ref["__naming_style__"] = naming_style

    # Check for SDXL Kohya format (with or without text encoder)
    # Check for both original (underscore) and normalized (dot) notation
    has_sdxl_patterns = any("input_blocks_" in k or "middle_block_" in k or "output_blocks_" in k or
                          "input_blocks." in k or "middle_block." in k or "output_blocks." in k for k in keys)

    if has_sdxl_patterns:
        has_lora_a_b = any("lora_A" in k or "lora_B" in k for k in keys)

        if has_lora_a_b:
            # SDXL-specific patterns
            # Check for SDXL-specific architecture patterns
            # IMPORTANT: Check ALL keys for SDXL architecture, not just lora_A/lora_B keys
            # This fixes issue where wai-A3-V140.safetensors was misdetected as SD1.5
            has_emb_layers = any("emb_layers" in k for k in keys)
            has_proj_in_out = any("proj_in" in k or "proj_out" in k for k in keys)
            has_in_out_layers = any("in_layers" in k or "out_layers" in k for k in keys)
            has_time_embed = any("time_embed" in k for k in keys)
            has_transformer_blocks = any("transformer_blocks" in k for k in keys)

            has_sdxl_architecture = (
                has_emb_layers or
                has_proj_in_out or
                has_in_out_layers or
                has_time_embed or
                has_transformer_blocks
            )


            if has_sdxl_architecture:
                return "sdxl_kohya"
            else:
                return "sd15_kohya"

    # Count different key patterns
    lora_down_count = sum(1 for k in keys if "lora_down" in k and "weight" in k)
    lora_up_count = sum(1 for k in keys if "lora_up" in k and "weight" in k)
    lora_a_count = sum(1 for k in keys if "lora_A" in k and "weight" in k)
    lora_b_count = sum(1 for k in keys if "lora_B" in k and "weight" in k)

    # Check for SDXL with Text Encoder (must come before SDXL standard check)
    # Check for both original (lora_te) and normalized (te1., te.) TE patterns
    has_te_original = any("lora_te" in k for k in keys)
    has_te_normalized = any("te1." in k or "te." in k for k in keys)
    has_te = has_te_original or has_te_normalized

    if has_te:
        # Check if it has SDXL patterns (original or normalized)
        has_sdxl_original = any("lora_unet_" in k for k in keys) and \
                           any("input_blocks_" in k or "middle_block_" in k or "output_blocks_" in k for k in keys)
        has_sdxl_normalized = any("input_blocks." in k or "middle_block." in k or "output_blocks." in k for k in keys)
        has_sdxl = has_sdxl_original or has_sdxl_normalized

        if has_sdxl:
            # This is SDXL with TE, not Pony
            # Check for both original (lora_down/lora_up) and normalized (lora_A/lora_B) formats
            has_down_up = (lora_down_count > 0 and lora_up_count > 0)
            has_a_b = (lora_a_count > 0 and lora_b_count > 0)

            if has_down_up:
                return "sdxl_with_te_lora_down_up"
            elif has_a_b:
                return "sdxl_with_te_lora_a_b"
            else:
                # Fallback based on key count
                return "sdxl_with_te_lora_down_up" if lora_down_count > lora_a_count else "sdxl_with_te_lora_a_b"

    # Check for SDXL standard format (lora_unet_ prefix with lora_down/lora_up)
    if any("lora_unet_" in k for k in keys):
        if any("input_blocks_" in k or "middle_block_" in k or "output_blocks_" in k for k in keys):
            if any("lora_down" in k or "lora_up" in k for k in keys):
                # Check if it's a small/partial LoRA (Illustrious style)
                if len(keys) < 50:
                    return "illustrious_sdxl"
                return "sdxl_standard"

    # Check for Musubi Text Encoder format (needs conversion)
    if any("lora_te_res_blocks" in k for k in keys):
        if any("diffusion_model" in k for k in keys) or any("layers" in k for k in keys):
            return "musubi_zimage_te_needs_conversion"
        else:
            return "musubi_te_needs_conversion"

    # Check for Musubi format (needs conversion)
    if any("lora_unet_double_blocks" in k for k in keys):
        return "musubi_flux_needs_conversion"

    if any("lora_unet_layers" in k for k in keys):
        return "musubi_zimage_needs_conversion"

    # Check if partially converted
    if any("diffusion_model" in k for k in keys) and any("lora_unet" in k for k in keys):
        return "musubi_partially_converted"

    # Check for Anima format (lora_unet_blocks_ without double/single prefix)
    if any("lora_unet_blocks_" in k for k in keys):
        if not any("double_blocks" in k or "single_blocks" in k for k in keys):
            return "anima_needs_conversion"

    # =====================================================================
    # 🔥 SD1.5 Diffusers format detection
    # Detects LoRAs using HuggingFace Diffusers naming convention:
    #   down_blocks_X / up_blocks_X / mid_block_X
    # instead of ComfyUI-native:
    #   input_blocks_X / output_blocks_X / middle_block_X
    #
    # Must come BEFORE the generic musubi catch-all (line 375) because
    # these keys also contain "lora_unet_" and would be misrouted.
    #
    # Examples:
    #   lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_out.0
    #   lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q
    #   lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v
    #
    # Excludes Flux (double_blocks/single_blocks — different structure)
    # and Anima (blocks_N_ without down/up prefix — already caught above).
    # =====================================================================
    if any("lora_unet_" in k for k in keys):
        has_diffusers_sd15 = any(
            "_down_blocks_" in k or "_up_blocks_" in k or "_mid_block_" in k
            for k in keys
        )
        has_flux_patterns = any("double_blocks" in k or "single_blocks" in k for k in keys)

        if has_diffusers_sd15 and not has_flux_patterns:
            return "sd15_diffusers_needs_conversion"

    if any("lora_unet_" in k for k in keys):
        return "musubi_other_needs_conversion"

    # Check for Pony Diffusion (only if not SDXL with TE)
    if any("lora_te" in k for k in keys):
        if lora_down_count > 0 and lora_up_count > 0:
            return "pony_diffusion_lora_down_up"
        elif lora_a_count > 0 and lora_b_count > 0:
            return "pony_diffusion_lora_a_b"
    
    # Check for Flux Klein
    if any("diffusion_model" in k for k in keys):
        if any("double_blocks" in k for k in keys):
            if lora_down_count > 0 and lora_up_count > 0:
                return "flux_klein_lora_down_up"
            elif lora_a_count > 0 and lora_b_count > 0:
                return "flux_klein_lora_a_b"
    
    # Check for Z-Image format
    if any("diffusion_model.layers" in k for k in keys):
        if lora_a_count > 0 and lora_b_count > 0:
            return "z_image_lora_a_b"
        elif lora_down_count > 0 and lora_up_count > 0:
            return "z_image_lora_down_up"
    
    # Check for SDXL Diffusers format
    if any("lora.down" in k or "lora.up" in k for k in keys):
        return "sdxl_diffusers"
    
    # Check for Z-Image variants
    if any("diffusion_model.layers." in k for k in keys):
        if any("adaLN_modulation" in k for k in keys):
            return "z_image_ai_toolkit"
        else:
            return "z_image_other"
    
    if any(".lora_A.default.weight" in k or ".lora_B.default.weight" in k for k in keys):
        return "z_image_default"
    
    # Check for Flux Klein variants
    if any("diffusion_model.double_blocks." in k for k in keys):
        # Check for 9B by tensor shape
        for k in keys[:5]:
            if "lora_A" in k and state_dict[k].shape[1] == 4096:
                return "flux_klein_9b"
        return "flux_klein_4b"
    
    if any("diffusion_model.single_blocks." in k for k in keys):
        return "flux_klein_4b"
    
    if any("diffusion_model." in k and ("double_blocks" in k or "single_blocks" in k) for k in keys):
        return "flux_klein_4b"
    
    # SD1.5 Kohya
    if any("lora_A" in k or "lora_B" in k for k in keys):
        return "sd15_kohya"
    
    # LyCORIS
    if any("lora." in k and ("alpha" in k or "dyn" in k) for k in keys):
        return "lycoris"
    
    return "unknown"

# ==================== INFERENCE FUNCTIONS ====================

def infer_separator_style(keys):
    """
    Infer separator style from a list of keys.
    Returns "dot", "underscore", or "hybrid".
    """
    dot_pattern = re.compile(r'\.\d+\.')          # .0.
    hybrid_pattern = re.compile(r'_\d+\.')        # _0.
    underscore_pattern = re.compile(r'_\d+_')     # _0_
    
    dot_count = 0
    hybrid_count = 0
    underscore_count = 0
    
    for key in keys:
        if hybrid_pattern.search(key):
            hybrid_count += 1
        elif dot_pattern.search(key):
            dot_count += 1
        elif underscore_pattern.search(key):
            underscore_count += 1
    
    if hybrid_count > 0:
        return "hybrid"
    if dot_count > underscore_count:
        return "dot"
    else:
        return "underscore"


def infer_naming_style(keys):
    """
    Infer naming style from a list of keys.
    Returns "lora_a_b" or "lora_down_up".
    """
    lora_a_count = sum(1 for k in keys if "lora_A" in k)
    lora_b_count = sum(1 for k in keys if "lora_B" in k)
    lora_down_count = sum(1 for k in keys if "lora_down" in k)
    lora_up_count = sum(1 for k in keys if "lora_up" in k)
    
    if lora_a_count + lora_b_count > lora_down_count + lora_up_count:
        return "lora_a_b"
    else:
        return "lora_down_up"

def repair_hybrid_separators(key):
    """Convert hybrid underscore‑before‑index patterns to pure dot notation."""
    # Pattern: _(\d+)\. (underscore before index, dot after)
    hybrid = re.compile(r'_(\d+)\.')
    # Replace with .\1.
    new_key = hybrid.sub(r'.\1.', key)
    return new_key

# ==================== MUSUBI CONVERTERS ====================

def convert_musubi_te_to_standard(state_dict):
    """
    Convert Musubi Text Encoder format to standard ComfyUI format.
    
    Musubi TE: lora_te_res_blocks_0_mlp_c_fc.lora_down.weight
    Standard: conditioner.embedders.0.transformer.resblocks.0.mlp.c_fc.lora_A.weight
    """
    print("ðŸ”„ Converting Musubi Text Encoder format...")
    converted = {}

    for key, value in state_dict.items():
        new_key = key

        # Convert resblock pattern
        new_key = RESBLOCK_PATTERN.sub(r'resblocks.\1.', new_key)

        # Convert lora_down/up to lora_A/B
        new_key = (new_key
                   .replace("lora_down.weight", "lora_A.weight")
                   .replace("lora_up.weight", "lora_B.weight"))

        new_key = fast_cleanup_key(new_key)

        debug_key_change(key, new_key)
        safe_insert(converted, new_key, value, key)

    print(f"âœ… Converted {len(converted)} TE keys")
    return converted

def convert_musubi_to_standard(state_dict):
    """Convert Musubi Flux/Klein format to standard."""
    print("ðŸ”„ Converting Musubi format...")
    converted = {}

    # Check if this is actually Pony format
    is_pony = any("lora_te" in k for k in state_dict.keys())
    
    for key, value in state_dict.items():
        new_key = key
        
        # PONY BYPASS - don't convert these!
        if is_pony and "lora_te" in key:
            converted[key] = value
            continue

        # Prefix normalization
        if not new_key.startswith("diffusion_model."):
            new_key = new_key.replace("lora_unet_", "diffusion_model.")

        # Repair hybrid separators (e.g., _0. -> .0.)
        new_key = repair_hybrid_separators(new_key)

        # Pattern substitutions
        new_key = LAYER_PATTERN.sub(r'layers.\1.', new_key)
        new_key = BLOCK_PATTERN.sub(r'\1_blocks.\2.', new_key)
        new_key = ATTN_PATTERN.sub(r'\1_attn.\2', new_key)
        new_key = TO_OUT_PATTERN.sub(r'to_out.\1', new_key)
        new_key = MLP_PATTERN.sub(r'\1_mlp.\2', new_key)
        new_key = FEED_FORWARD_PATTERN.sub(r'feed_forward.w\1', new_key)

        # Attention routing
        new_key = (new_key
                   .replace("attention_to_k", "attention.to_k")
                   .replace("attention_to_q", "attention.to_q")
                   .replace("attention_to_v", "attention.to_v")
                   .replace("attention_to_out_0", "attention.to_out.0"))

        # Index normalization
        new_key = re.sub(r'\.(\d+)_', r'.\1.', new_key)
        new_key = re.sub(r'_(\d+)\.', r'.\1.', new_key)

        # LoRA naming
        new_key = (new_key
                   .replace("lora_down.weight", "lora_A.weight")
                   .replace("lora_up.weight", "lora_B.weight"))

        new_key = fast_cleanup_key(new_key)

        debug_key_change(key, new_key)
        safe_insert(converted, new_key, value, key)

    print(f"âœ… Converted {len(converted)} keys")
    return converted


def _convert_sd15_diffusers_key(key: str) -> str:
    """
    Convert a single SD1.5 Diffusers-format key to ComfyUI-native format.

    This is the core conversion logic extracted as a pure string transformation
    so it can be reused by convert_sd15_diffusers_to_comfyui() and tests.

    See convert_sd15_diffusers_to_comfyui() for full mapping documentation.
    """
    new_key = key

    # Step 1: Convert lora_unet_ prefix → diffusion_model.
    if new_key.startswith("lora_unet_"):
        new_key = "diffusion_model." + new_key[len("lora_unet_"):]

    # Step 1b: Handle lora_te_ prefix (Text Encoder keys for SD1.5 Diffusers)
    # Raw input:  lora_te_text_model_encoder_layers_0_self_attn_k_proj
    # Output:     text_model.encoder.layers.0.self_attn.k_proj
    # This normalizes the TE path so the reverse map (Strategy 0) can match
    # against the checkpoint's cond_stage_model.transformer.text_model.* keys.
    if new_key.startswith("lora_te_"):
        new_key = new_key[len("lora_te_"):]  # Remove lora_te_ prefix
        # Convert text_model_encoder_layers_N_ → text_model.encoder.layers.N.
        new_key = re.sub(
            r'text_model_encoder_layers_(\d+)_',
            r'text_model.encoder.layers.\1.',
            new_key
        )
        # Convert self_attn_k_proj → self_attn.k_proj
        new_key = re.sub(
            r'self_attn_(q|k|v|out)_proj',
            r'self_attn.\1_proj',
            new_key
        )
        # Convert mlp_fc1 → mlp.fc1, mlp_fc2 → mlp.fc2
        new_key = re.sub(r'mlp_fc(\d)', r'mlp.fc\1', new_key)

    # Step 2: Convert down_blocks_X → input_blocks.(X+1)
    # (input_blocks.0 = conv_in, so down_blocks.0 → input_blocks.1, etc.)
    def _shift_down_block(m):
        idx = int(m.group(1))
        return f"input_blocks_{idx + 1}_"
    new_key = re.sub(r'down_blocks_(\d+)_', _shift_down_block, new_key)

    # Step 3: Convert up_blocks_X → output_blocks_X (1:1)
    new_key = re.sub(r'up_blocks_(\d+)_', r'output_blocks_\1_', new_key)

    # Step 4: Convert mid_block_ → middle_block_ (1:1)
    new_key = new_key.replace("mid_block_", "middle_block_")

    # Step 5: Convert attentions_N_ → sub-block index 1_ (ComfyUI SD1.5)
    # 🔥 CRITICAL FIX: ComfyUI SD1.5 places attention at sub-block index 1
    # (e.g., input_blocks.1.1.transformer_blocks.0), NOT at the Diffusers
    # attention index N. Previous code used '\1_' (the attention index),
    # which produced sub-block .0 for all attention keys, making them
    # un-matchable via the reverse key map (Strategy 0). All attention
    # deltas then fell through to Energy Redirect, causing shape mismatches
    # and OOM crashes.
    # Reference: SD1.5 sub-block map — .0=resnet, .1=attention
    new_key = re.sub(r'attentions_(\d+)_', r'1_', new_key)

    # Step 5b: Convert resnets_N_ → sub-block index N_ (ComfyUI SD1.5)
    # ComfyUI SD1.5 places resnet at sub-block index 0 for N=0.
    # Keeping the original index N works for N=0 (most common case).
    # For N>0 (second resnet at sub-block .2), this is approximately correct
    # since LoRAs rarely target both resnet and attention in the same block.
    new_key = re.sub(r'resnets_(\d+)_', r'\1_', new_key)

    # Step 6: Convert remaining underscore numerics to dot numerics
    # e.g., input_blocks_1_1_ → input_blocks.1.1.
    new_key = re.sub(r'_(\d+)_', r'.\1.', new_key)

    # Step 6b: Fix missing dots from adjacent underscore-numeric patterns
    # After step 5 strips "attentions_N_" → "N_", we get patterns like
    # "1_1_transformer" where the middle "1_" isn't caught by step 6
    # because its leading underscore was consumed by the previous match.
    # This converts "1_transformer" → "1.transformer".
    # Also handles "1_resnets" → "1.resnets" for ResNet submodules.
    new_key = re.sub(r'(\d+)_([a-zA-Z])', r'\1.\2', new_key)

    # Step 7: Convert attention projection names (underscore → dot)
    # attn1_to_q → attn1.to.q, attn2_to_out_0 → attn2.to.out.0
    new_key = re.sub(r'attn(\d)_to_q\b', r'attn\1.to_q', new_key)
    new_key = re.sub(r'attn(\d)_to_k\b', r'attn\1.to_k', new_key)
    new_key = re.sub(r'attn(\d)_to_v\b', r'attn\1.to_v', new_key)
    new_key = re.sub(r'attn(\d)_to_out_(\d+)', r'attn\1.to_out.\2', new_key)
    new_key = re.sub(r'attn(\d)_to_out\b', r'attn\1.to_out', new_key)

    # Step 7b: Catch any remaining to_out_N (underscore) → to_out.N (dot)
    # The SD1.5 Diffusers format produces keys like "attn2_to_out.0" where
    # the dot is already present after "to_out", so Step 7's
    # attn(\d)_to_out_(\d+) regex doesn't match (no underscore after "to_out").
    # This cleanup catches any lingering to_out_N patterns from edge cases.
    new_key = re.sub(r'to_out_(\d+)', r'to_out.\1', new_key)

    # Step 7c: Convert ff_net → ff.net (ComfyUI uses dots for hierarchy)
    # In ComfyUI SD1.5, the feed-forward network module is named "ff.net"
    # (with a dot), not "ff_net" (with underscore). The Diffusers format
    # serializes this as "ff_net" which Step 6's _(\d+)_ regex can't convert
    # because it only targets numeric underscores. Without this fix, keys
    # like "ff_net.0.proj" fail to match the checkpoint's "ff.net.0.proj"
    # in the reverse key map (Strategy 0), falling through to Energy Redirect.
    # Reference: ComfyUI SD1.5 UNet — transformer_blocks.N.ff.net.0.proj
    #
    # 🔥 SUB-FIX: Handle ff_net_N (anime Diffusers second FF network)
    # Some anime Diffusers models have a second FF network (ff_net_2) per
    # transformer block. The key looks like ff_net_2.lora_down.weight — note
    # the DOT after the numeral (not underscore), so Step 6's _(\d+)_ regex
    # cannot convert it (requires trailing underscore). The boundary \b
    # between 't' (word) and '2' (word) in ff_net_2 also prevents
    # \bff_net\b from matching. This regex converts ff_net_N → ff.net.N.
    new_key = re.sub(r'\bff_net_(\d+)\b', r'ff.net.\1', new_key)
    # Secondary: handles ff_net (standard, followed by non-word like . or _)
    new_key = re.sub(r'\bff_net\b', r'ff.net', new_key)

    # Step 8: Convert lora_down/lora_up → lora_A/lora_B
    new_key = (new_key
               .replace("lora_down.weight", "lora_A.weight")
               .replace("lora_up.weight", "lora_B.weight"))

    # Step 9: Final cleanup — remove double dots
    while ".." in new_key:
        new_key = new_key.replace("..", ".")
    # Safety: remove ._.
    new_key = new_key.replace("._.", ".")

    return new_key


def convert_sd15_diffusers_to_comfyui(state_dict):
    """
    Convert SD1.5 Diffusers-format LoRA to ComfyUI-native format.

    Diffusers SD1.5 keys use down_blocks/up_blocks/mid_block naming
    (from HuggingFace diffusers library). ComfyUI uses input_blocks/
    output_blocks/middle_block naming.

    ⚠️  KEY COLLISION HANDLING (CRITICAL):
    SD1.5 Diffusers blocks can have MULTIPLE attention sub-modules
    (attentions_0, attentions_1, etc.) that all map to the same ComfyUI
    sub-block index .1.  When two (or more) Diffusers keys map to the same
    ComfyUI key, their tensors are SUMMED instead of dropped.  This preserves
    the total LoRA signal from all attention modules.

    Previously, safe_insert() kept only the first occurrence and silently
    dropped the second, causing ~41% key loss and weak baking.

    Block index mapping (SD1.5 specific):
      down_blocks.X    → input_blocks.(X+1)    (input_blocks.0 = conv_in)
      up_blocks.X      → output_blocks.X       (1:1)
      mid_block        → middle_block           (1:1)

    Attention submodule mapping (same for all blocks that have attention):
      down_blocks.X.attentions.N.transformer_blocks.M.*
        → input_blocks.(X+1).1.transformer_blocks.M.*

      mid_block.attentions.N.transformer_blocks.M.*
        → middle_block.1.transformer_blocks.M.*

      up_blocks.X.attentions.N.transformer_blocks.M.*
        → output_blocks.X.1.transformer_blocks.M.*

    ResNet submodule mapping:
      down_blocks.X.resnets.N.*  → input_blocks.(X+1).0.*
      up_blocks.X.resnets.N.*    → output_blocks.X.0.*
    """
    print("   🔄 Converting SD1.5 Diffusers format to ComfyUI-native...")

    # Phase 1: Convert all keys and group by output key
    # Groups collect tensors that map to the same ComfyUI key (collisions).
    groups: Dict[str, List[torch.Tensor]] = {}
    for key, value in state_dict.items():
        new_key = _convert_sd15_diffusers_key(key)
        if new_key not in groups:
            groups[new_key] = []
        groups[new_key].append(value)

    # Phase 2: Sum colliding groups, preserve unique entries
    converted = {}
    collision_count = 0
    for new_key, tensors in groups.items():
        if len(tensors) > 1:
            # Sum colliding tensors — multiple Diffusers attention modules
            # map to the same ComfyUI sub-block. Summing preserves the total
            # LoRA signal.
            converted[new_key] = sum(tensors[1:], tensors[0])
            collision_count += len(tensors) - 1
        else:
            converted[new_key] = tensors[0]

    total_original = len(state_dict)
    total_converted = len(converted)
    print(f"   ✅ Converted {total_converted} keys to ComfyUI SD1.5 format "
          f"({collision_count} collisions resolved via summation, "
          f"{total_original - total_converted - collision_count} duplicates dropped)")
    return converted


def convert_musubi_to_anima(state_dict):
    """Convert Musubi Anima format to standard.

    Anima model uses MiniTrainDIT internally (self.blocks = nn.ModuleList).
    In ComfyUI, BaseModel wraps the unet under self.diffusion_model, so
    runtime state_dict keys are diffusion_model.blocks.0... NOT net.blocks.*.
    The "net." prefix only exists in the trainer checkpoint file, not in the
    model's internal module structure.

    model_lora_keys_unet() builds key_map from diffusion_model.* keys, so
    the LoRA keys must match: diffusion_model.blocks.0.adaln_modulation...
    (without "net." prefix).

    Mapping guide (trainer_key → model_module_name):
      cross_attn_output_proj → cross_attn.output_proj   (Attention.output_proj)
      self_attn_output_proj  → self_attn.output_proj    (Attention.output_proj)
      mlp_layer1             → mlp.layer1                (GPT2FeedForward.layer1)
      mlp_layer2             → mlp.layer2                (GPT2FeedForward.layer2)
      cross_attn_q_proj      → cross_attn.q_proj         (Attention.q_proj)
      cross_attn_k_proj      → cross_attn.k_proj         (Attention.k_proj)
      cross_attn_v_proj      → cross_attn.v_proj         (Attention.v_proj)

    Input:  lora_unet_blocks_0_adaln_modulation_cross_attn_1.lora_down.weight
    Output: diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_A.weight
    """
    print("🔄 Converting Anima format...")
    converted = {}

    for key, value in state_dict.items():
        new_key = key

        # 0. Handle TE keys (Anima format: lora_te_layers_N_self_attn.X_proj.lora_A.weight)
        #    ComfyUI's model_lora_keys_clip() builds key_map from model CLIP state_dict keys.
        #    For SD1.5, the CLIP state dict has keys like:
        #      transformer.text_model.encoder.layers.N.self_attn.q_proj.weight
        #    model_lora_keys_clip() (line 101) creates:
        #      key_map["text_encoders.transformer.text_model.encoder.layers.N.self_attn.q_proj"]
        #      → "transformer.text_model.encoder.layers.N.self_attn.q_proj.weight"
        #    load_lora() then looks for "{key_map_key}.lora_B.weight" in the LoRA dict,
        #    so we must produce:
        #      text_encoders.transformer.text_model.encoder.layers.N.self_attn.X_proj.lora_A.weight
        #
        #    The musubi format lora_te_layers_N_ is shorthand for the model's actual
        #    transformer.text_model.encoder.layers.N. structure, so we must produce:
        #      transformer.text_model.encoder.layers.N.self_attn.X_proj (NOT te.layers.N.)
        if key.startswith("lora_te_"):
            new_key = key

            # a. Convert prefix: lora_te_layers → transformer.model.layers
            #    Anima TE uses Qwen3 (Llama2_ architecture), whose state dict keys are:
            #      qwen3_06b.transformer.model.layers.N.self_attn.q_proj.weight
            #    (via SD1ClipModel → SDClipModel → Qwen3_06BModel → Qwen3_06B → Llama2_)
            #    NOT SD1.5 CLIP which uses transformer.text_model.encoder.layers.N.*
            #    model_lora_keys_clip() strips the qwen3_06b. wrapper via .transformer. detection,
            #    producing: text_encoders.transformer.model.layers.N.self_attn.q_proj
            new_key = new_key.replace("lora_te_layers", "transformer.model.layers", 1)

            # b. Convert layers_N_ → layers.N. (now inside transformer.text_model.encoder.layers.N.)
            new_key = re.sub(r'layers_(\d+)_', r'layers.\1.', new_key)

            # c. Convert remaining underscore-to-dot for known submodule names
            #    o_proj is the short form of out_proj (self_attn_o_proj → self_attn.out_proj)
            new_key = re.sub(
                r'_(q_proj|k_proj|v_proj|output_proj|out_proj|o_proj|gate_proj|up_proj|down_proj|c_fc|c_proj)',
                r'.\1',
                new_key,
            )

            # d. Convert lora naming (master format: lora_A/lora_B)
            new_key = (new_key
                       .replace("lora_down.weight", "lora_A.weight")
                       .replace("lora_up.weight", "lora_B.weight"))

            # e. Final cleanup: remove double dots
            while ".." in new_key:
                new_key = new_key.replace("..", ".")

            debug_key_change(key, new_key)
            safe_insert(converted, new_key, value, key)
            continue  # Skip UNet handling for TE keys

        # 1. Change prefix: lora_unet_ → diffusion_model.
        #    ComfyUI's model_lora_keys_unet() iterates model state_dict keys
        #    starting with "diffusion_model." and builds key_map entries.
        #    The Anima model (MiniTrainDIT) creates self.blocks internally,
        #    so the model's state_dict keys are:
        #      diffusion_model.blocks.0.adaln_modulation_cross_attn.1.weight
        #    NOT diffusion_model.net.blocks.* — "net." is only in the trainer
        #    checkpoint format, not in the model's internal module structure.
        #    load_lora() then looks for "{key}.lora_A.weight" matching our output.
        new_key = new_key.replace("lora_unet_", "diffusion_model.", 1)

        # 2. Convert blocks_N_ → blocks.N.
        new_key = ANIMA_BLOCK_PATTERN.sub(r'blocks.\1.', new_key)

        # 3. Convert remaining index patterns (_N. → .N. and .N_ → .N.)
        new_key = re.sub(r'_(\d+)\.', r'.\1.', new_key)
        new_key = re.sub(r'\.(\d+)_', r'.\1.', new_key)

        # =====================================================================
        # 🔥 FIX: Handle llm_adapter and final_layer naming (Anima-specific)
        #
        # The main DIT blocks (lora_unet_blocks_0_...) are handled by step 2
        # (ANIMA_BLOCK_PATTERN).  But Anima also has:
        #
        #   1. LLMAdapter (llm_adapter.blocks.N.*) — a separate ModuleList of
        #      TransformerBlocks for T5 cross-attention injection.
        #      Trainer format: lora_unet_llm_adapter_blocks_0_cross_attn_q_proj
        #      Model format:   diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj
        #
        #      After step 2, llm_adapter_blocks_N_ becomes llm_adapter_blocks.N.
        #      (ANIMA_BLOCK_PATTERN matches blocks_0_ and converts to blocks.0.)
        #      We then need llm_adapter_blocks → llm_adapter.blocks
        #
        #   2. FinalLayer (net.final_layer.X) — single output projection + adaLN.
        #      Trainer format: lora_unet_final_layer_linear
        #      Model format:   diffusion_model.final_layer.linear
        #
        #      After step 1:  diffusion_model.final_layer_linear
        #      Needs:         diffusion_model.final_layer.linear
        # =====================================================================

        # 3a) llm_adapter_blocks → llm_adapter.blocks
        new_key = new_key.replace('llm_adapter_blocks', 'llm_adapter.blocks')

        # 3b) final_layer_N → final_layer.N (for numbered sub-parts like adaln_modulation.1)
        new_key = re.sub(r'final_layer_(\d+)', r'final_layer.\1', new_key)

        # 3c) final_layer_ → final_layer. (for non-digit suffixes like linear)
        new_key = new_key.replace('final_layer_', 'final_layer.')

        # 4. Map trainer-specific naming to model's actual module names.
        #    CRITICAL: Names must match predict2.py's module definitions:
        #      GPT2FeedForward uses layer1/layer2 (NOT c_fc/c_proj)
        #      Attention uses output_proj (NOT out_proj)
        new_key = (new_key
                   .replace("mlp_layer1", "mlp.layer1")
                   .replace("mlp_layer2", "mlp.layer2"))

        # 5. Convert structural underscores to dots for known submodule names.
        #    The trainer flattens ALL hierarchy separators to underscores, so
        #    'self_attn.q_proj' becomes 'self_attn_q_proj' in trainer format.
        #    The _N. regex above only handles digit-indexed parts; for named
        #    submodules we need to convert the preceding underscore to a dot.
        #    NOTE: output_proj must appear BEFORE out_proj in the alternation
        #    because out_proj is a substring of output_proj.
        new_key = re.sub(
            r'_(q_proj|k_proj|v_proj|output_proj|out_proj|gate_proj|up_proj|down_proj|c_fc|c_proj)',
            r'.\1',
            new_key,
        )

        # 6. Convert lora naming (master format: lora_A/lora_B)
        new_key = (new_key
                   .replace("lora_down.weight", "lora_A.weight")
                   .replace("lora_up.weight", "lora_B.weight"))

        # 7. Final cleanup: remove double dots
        while ".." in new_key:
            new_key = new_key.replace("..", ".")

        debug_key_change(key, new_key)
        safe_insert(converted, new_key, value, key)

    print(f"✅ Converted {len(converted)} Anima keys")
    return converted

def convert_musubi_zimage_to_standard(state_dict):
    """Convert Musubi Z-Image format to standard with strict dot-notation."""
    print("ðŸ”„ Converting Musubi Z-Image format...")
    converted = {}

    for key, value in state_dict.items():
        # 1. Primary prefix change
        new_key = key.replace("lora_unet_", "diffusion_model.")

        # Repair hybrid separators (e.g., _0. -> .0.)
        new_key = repair_hybrid_separators(new_key)

        # 2. Convert underscores in layers and blocks
        new_key = LAYER_PATTERN.sub(r'layers.\1.', new_key)
        new_key = TO_OUT_PATTERN.sub(r'to_out.\1', new_key)
        new_key = FEED_FORWARD_PATTERN.sub(r'feed_forward.w\1', new_key)

        # 3. FIX: Specific Z-Image underscore-to-dot mapping
        # This fixes the "attention_to_out" vs "attention.to_out" mismatch
        new_key = new_key.replace("attention_to_", "attention.to_")

        # 4. Standard LoRA naming conversion
        new_key = (new_key
                   .replace("lora_down.weight", "lora_A.weight")
                   .replace("lora_up.weight", "lora_B.weight"))

        new_key = fast_cleanup_key(new_key)

        debug_key_change(key, new_key)
        safe_insert(converted, new_key, value, key)

    print(f"âœ… Converted {len(converted)} Z-Image keys")
    return converted

# ==================== NORMALIZERS ====================

def normalize_z_image_context_refiner(state_dict):
    """Convert Z-Image context_refiner format to standard."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = (key
                   .replace("context_refiner.", "diffusion_model.layers.")
                   .replace(".default", ""))
        normalized[new_key] = value
    return normalized


def normalize_sdxl_kohya_with_te(state_dict, include_te=True):
    """Convert SDXL Kohya format with optional TE normalization."""
    normalized = {}

    for key, value in state_dict.items():
        new_key = key

        # Handle TE keys
        if include_te and new_key.startswith('lora_te'):
            # SDXL TE format: lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight
            # Remove lora_ prefix
            # ⚠️  ORDER MATTERS: lora_te2_ must be replaced BEFORE lora_te_ because
            #     'lora_te_' is a SUBSTRING of 'lora_te2_'.  If lora_te_ runs first,
            #     'lora_te2_text_model...' becomes 'te.2_text_model...' (corrupted!).
            new_key = new_key.replace('lora_te2_', 'te2.')
            new_key = new_key.replace('lora_te1_', 'te1.')
            new_key = new_key.replace('lora_te_', 'te.')

            # Convert underscores to dots for TE structure
            # text_model_encoder_layers_0_mlp_fc1 â†’ text_model.encoder.layers.0.mlp.fc1
            new_key = re.sub(r'text_model_encoder_layers_(\d+)_', r'text_model.encoder.layers.\1.', new_key)
            new_key = re.sub(r'mlp_fc(\d)', r'mlp.fc\1', new_key)
            new_key = re.sub(r'self_attn_', r'self_attn.', new_key)
            new_key = re.sub(r'(k|q|v|out)_proj', r'\1_proj', new_key)  # k_proj, q_proj etc.

            # Convert lora_down/lora_up to lora_A/lora_B for consistency
            new_key = new_key.replace('lora_down.', 'lora_A.')
            new_key = new_key.replace('lora_up.', 'lora_B.')

        # Handle UNet keys
        elif new_key.startswith('lora_unet_'):
            # Remove lora_unet_ prefix if present
            new_key = new_key.replace('lora_unet_', '', 1)

            # Convert underscores to dots in block patterns
            patterns = [
                (r'input_blocks_(\d+)_(\d+)_', r'input_blocks.\1.\2.'),
                (r'middle_block_(\d+)_', r'middle_block.\1.'),
                (r'output_blocks_(\d+)_(\d+)_', r'output_blocks.\1.\2.'),
                (r'time_embed_', r'time_embed.'),
            ]

            for pattern, replacement in patterns:
                new_key = re.sub(pattern, replacement, new_key)

            # Handle transformer blocks
            if 'transformer_blocks' in new_key:
                new_key = re.sub(r'transformer_blocks_(\d+)_', r'transformer_blocks.\1.', new_key)

            # Convert lora_down/lora_up to lora_A/lora_B for consistency
            # Use exact string replacement to catch all variations
            new_key = new_key.replace('lora_down', 'lora_A')
            new_key = new_key.replace('lora_up', 'lora_B')

        # Ensure proper suffix
        if not new_key.endswith(".weight") and ".alpha" not in new_key and "bias" not in new_key:
            if any(x in new_key for x in ['lora_down', 'lora_up', 'lora_A', 'lora_B']):
                new_key += ".weight"

        normalized[new_key] = value

    return normalized

def normalize_sdxl_diffusers(state_dict):
    """Convert SDXL diffusers format to standard."""
    normalized = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert lora.down â†’ lora_A, lora.up â†’ lora_B
        new_key = new_key.replace("lora.down", "lora_A").replace("lora.up", "lora_B")
        
        # Convert to standard ComfyUI naming
        if "to_k" in new_key or "to_v" in new_key or "to_q" in new_key or "to_out" in new_key:
            new_key = new_key.replace("processor.", "").replace("to_out.0", "to_out")
        
        # Add .weight suffix if missing
        if not new_key.endswith(".weight") and ".alpha" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

def normalize_z_image_ai_toolkit(state_dict):
    """Convert Z-Image AI-Toolkit format to standard."""
    normalized = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Already uses lora_A/lora_B format
        if not new_key.endswith(".weight") and ".alpha" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

def normalize_z_image_default(state_dict):
    """Convert Z-Image .default.weight format to standard."""
    normalized = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Remove .default from keys
        new_key = (new_key
                   .replace(".lora_A.default", ".lora_A")
                   .replace(".lora_B.default", ".lora_B"))
        
        # Ensure .weight suffix
        if not new_key.endswith(".weight") and ".alpha" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

def normalize_all_klein_formats(state_dict):
    """Standardizes Musubi and AI-Toolkit names into one format."""
    normalized = {}

    for key, value in state_dict.items():
        new_key = key
        new_key = repair_hybrid_separators(new_key)

        # Klein format handling
        new_key = re.sub(r'(img|txt)_attn_(proj|qkv)', r'\1_attn.\2', new_key)
        new_key = re.sub(r'(img|txt)_mlp_', r'\1_mlp.', new_key)

        if "single_blocks" in new_key:
            new_key = re.sub(r'single_blocks_(\d+)_attn_(proj|qkv)', r'single_blocks.\1.attn.\2', new_key)
            new_key = re.sub(r'single_blocks_(\d+)_mlp_', r'single_blocks.\1.mlp.', new_key)

        if "lora_unet_" in new_key:
            new_key = new_key.replace("lora_unet_", "diffusion_model.")

        # Convert lora_down/lora_up to lora_A/lora_B
        new_key = new_key.replace("lora_down", "lora_A").replace("lora_up", "lora_B")
        new_key = new_key.replace("attn_proj", "attn.proj").replace("attn_qkv", "attn.qkv")

        # Remove double dots
        while ".." in new_key:
            new_key = new_key.replace("..", ".")

        # Add .weight suffix
        if ("lora_A" in new_key or "lora_B" in new_key) and "weight" not in new_key and "alpha" not in new_key:
            new_key += ".weight"

        normalized[new_key] = value

    return normalized

def normalize_key_for_flux(key: str) -> str:
    """Normalize Flux model key to standard format."""
    normalized = key
    patterns = [
        (r'img_attn_proj', 'img_attn.proj'),
        (r'img_attn_qkv', 'img_attn.qkv'),
        (r'txt_attn_proj', 'txt_attn.proj'),
        (r'txt_attn_qkv', 'txt_attn.qkv'),
        (r'mlp_fc1', 'mlp.fc1'),
        (r'mlp_fc2', 'mlp.fc2'),
    ]
    
    for pattern, replacement in patterns:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Remove double dots
    while '..' in normalized:
        normalized = normalized.replace('..', '.')
    
    return normalized

# ==================== UNIVERSAL NORMALIZER ====================

def _compute_key_fingerprint(state_dict):
    """Compute a hashable fingerprint from sorted state dict keys for cache lookup."""
    return tuple(sorted(state_dict.keys()))


def universal_normalize(state_dict, metadata=None, mapping_ref=None):
    """Universal normalizer with format preservation and metadata awareness."""
    
    # Inject metadata into the object so get_alpha_value can find it
    if metadata is not None:
        # We use setattr because some state_dicts are custom objects
        try:
            state_dict.metadata = metadata
        except:
            # If it's a plain dict, we just attach it as a hidden property
            pass

    # B2: Skip format re-detection if already stored in mapping_ref
    if mapping_ref is not None and "__format__" in mapping_ref:
        format_type = mapping_ref["__format__"]
        print(f"Detected format (from mapping_ref): {format_type}")
    else:
        format_type = detect_lora_format(state_dict, mapping_ref)
        print(f"Detected format: {format_type}")
        # Store detected format in mapping_ref for later reuse (B2)
        if mapping_ref is not None:
            mapping_ref["__format__"] = format_type
    
    # Set global mapping reference if provided
    global _current_mapping_ref
    previous_mapping = _current_mapping_ref
    if mapping_ref is not None:
        _current_mapping_ref = mapping_ref
    else:
        _current_mapping_ref = None


    # 1. Run the Auto-Repair/Acceleration pipeline FIRST
    # This ensures Alphas are calculated before keys get renamed
    state_dict = accelerate_lora_load(state_dict)

    # B1: Check conversion cache — if we've already converted a LoRA with the
    # same key structure (same format + same set of keys), reuse the key mapping
    # to avoid redundant string processing and tensor iteration.
    fingerprint = _compute_key_fingerprint(state_dict)
    cache_key = (format_type, fingerprint, NORMALIZE_CACHE_VERSION)
    if cache_key in _NORMALIZE_CACHE:
        cached_mapping = _NORMALIZE_CACHE[cache_key]
        print(f"  [Cache] Reusing cached key mapping for format '{format_type}' ({len(cached_mapping)} keys)")
        result = {}
        for new_key, old_key in cached_mapping.items():
            result[new_key] = state_dict[old_key]
        return result

    # Bypass standard SD formats
    bypass_formats = ["pony", "illustrious", "sd15", "sd1.5", "sdxl_with_te", "illustrious_sdxl", "sdxl_kohya", "sdxl_standard", "sdxl_diffusers"]
    bypass_match = any(fmt in format_type.lower() for fmt in bypass_formats)

    # Special handling: SDXL formats should still be normalized (remove from bypass)
    if "sdxl" in format_type.lower():
        bypass_match = False

    # CRITICAL: "needs_conversion" formats MUST NEVER be bypassed — they require
    # active transformation to produce ComfyUI-native key format.  A format like
    # "sd15_diffusers_needs_conversion" would otherwise match "sd15" in the
    # bypass list (substring match) and skip conversion entirely.
    if "needs_conversion" in format_type.lower():
        bypass_match = False

    if bypass_match:
        print(f"OK: Keeping {format_type} format as-is (Standard SD format)")
        return state_dict
    
    # Auto-convert Anima format (must be checked before Musubi catch-all)
    if format_type == "anima_needs_conversion":
        print("Converting Anima format...")
        result = convert_musubi_to_anima(state_dict)
        # Cache the key mapping (new_key → old_key): post-conversion → pre-conversion
        _NORMALIZE_CACHE[cache_key] = {new: old for new, old in zip(result.keys(), state_dict.keys())}
        return result
    
    # Auto-convert SD1.5 Diffusers format (must be checked before Musubi catch-all)
    if format_type == "sd15_diffusers_needs_conversion":
        print("Converting SD1.5 Diffusers to ComfyUI format...")
        result = convert_sd15_diffusers_to_comfyui(state_dict)
        # Cache the key mapping (new_key → old_key): post-conversion → pre-conversion
        _NORMALIZE_CACHE[cache_key] = {new: old for new, old in zip(result.keys(), state_dict.keys())}
        return result
    
    # Auto-convert Musubi formats
    if "musubi" in format_type.lower() and "needs_conversion" in format_type:
        print("Converting Musubi to standard format...")
        
        if "zimage" in format_type:
            result = convert_musubi_zimage_to_standard(state_dict)
        elif "flux" in format_type or "klein" in format_type:
            result = convert_musubi_to_standard(state_dict)
        elif "te" in format_type:
            result = convert_musubi_te_to_standard(state_dict)
        else:
            result = convert_musubi_to_standard(state_dict)
        # Cache the key mapping (new_key → old_key): post-conversion → pre-conversion
        _NORMALIZE_CACHE[cache_key] = {new: old for new, old in zip(result.keys(), state_dict.keys())}
        return result
    
    # Already converted Musubi
    if "musubi" in format_type.lower() and "converted" in format_type:
        print("   Musubi already converted, using as-is")
        return state_dict
    
    # Keep Pony as-is
    if "pony" in format_type.lower():
        print("   Keeping Pony Diffusion format as-is")
        return state_dict
    
    # Apply format-specific normalization
    normalizers = {
        "sdxl_kohya": lambda sd: normalize_sdxl_kohya_with_te(sd, include_te=False),
        "sdxl_standard": lambda sd: normalize_sdxl_kohya_with_te(sd, include_te=False),
        "sdxl_with_te_lora_down_up": lambda sd: normalize_sdxl_kohya_with_te(sd, include_te=True),
        "sdxl_with_te_lora_a_b": lambda sd: normalize_sdxl_kohya_with_te(sd, include_te=True),
        "illustrious_sdxl": lambda sd: normalize_sdxl_kohya_with_te(sd, include_te=False),
        "sdxl_diffusers": normalize_sdxl_diffusers,
        "z_image_ai_toolkit": normalize_z_image_ai_toolkit,
        "z_image_default": normalize_z_image_default,
        "z_image_context_refiner": normalize_z_image_context_refiner,
        "z_image_lora_down_up": normalize_all_klein_formats,
        "z_image_lora_a_b": normalize_all_klein_formats,
        "flux_klein_9b": normalize_all_klein_formats,
        "flux_klein_4b": normalize_all_klein_formats,
        "flux_klein_lora_a_b": normalize_all_klein_formats,
        "flux_klein_lora_down_up": normalize_all_klein_formats,
        "ai_toolkit_klein": normalize_all_klein_formats,
    }
    
    if format_type in normalizers:
        normalized = normalizers[format_type](state_dict)
        # Apply Flux key normalization if needed
        if "flux" in format_type or "klein" in format_type:
            final = {}
            for key, value in normalized.items():
                new_key = normalize_key_for_flux(key)
                final[new_key] = value
            # Also cache the key mapping for normalizers
            _NORMALIZE_CACHE[cache_key] = {new: old for new, old in zip(final.keys(), normalized.keys())}
            return final
        return normalized


    
    # Unknown format - try Klein normalization
    print(f"WARNING: Unknown format, trying Klein normalization")
    try:
        normalized = normalize_all_klein_formats(state_dict)
        final = {}
        for key, value in normalized.items():
            new_key = normalize_key_for_flux(key)
            final[new_key] = value
        return final
    except:
        print(f"WARNING: Could not normalize, using raw format")
        return state_dict
    
# ==================== FORMAT INFO ====================

_FORMAT_STYLE_MAP = {
    # SDXL formats (underscore separator)
    "sdxl_standard": ("underscore", "lora_down_up"),
    "sdxl_kohya": ("underscore", "lora_down_up"),
    "sdxl_with_te_lora_down_up": ("underscore", "lora_down_up"),
    "sdxl_with_te_lora_a_b": ("underscore", "lora_a_b"),
    "illustrious_sdxl": ("underscore", "lora_down_up"),
    "sdxl_diffusers": ("underscore", "lora_down_up"),
    # SD1.5 Kohya
    "sd15_kohya": ("underscore", "lora_a_b"),
    # Z-Image formats (dot separator)
    "z_image_lora_a_b": ("dot", "lora_a_b"),
    "z_image_lora_down_up": ("dot", "lora_down_up"),
    "z_image_ai_toolkit": ("dot", "lora_a_b"),
    "z_image_default": ("dot", "lora_a_b"),
    "z_image_context_refiner": ("dot", "lora_a_b"),
    "z_image_other": ("dot", "lora_a_b"),
    # Flux Klein formats (dot separator)
    "flux_klein_lora_a_b": ("dot", "lora_a_b"),
    "flux_klein_lora_down_up": ("dot", "lora_down_up"),
    "flux_klein_9b": ("dot", "lora_a_b"),
    "flux_klein_4b": ("dot", "lora_a_b"),
    "ai_toolkit_klein": ("dot", "lora_a_b"),
    # Musubi formats (dot after conversion)
    "musubi_zimage_needs_conversion": ("dot", "lora_a_b"),
    "musubi_zimage_te_needs_conversion": ("dot", "lora_a_b"),
    "musubi_flux_needs_conversion": ("dot", "lora_a_b"),
    "musubi_te_needs_conversion": ("dot", "lora_a_b"),
    "musubi_partially_converted": ("dot", "lora_a_b"),
    "musubi_other_needs_conversion": ("dot", "lora_a_b"),
    # Anima format (dot separator, lora_down_up naming to match model keys)
    "anima_needs_conversion": ("dot", "lora_down_up"),
    # Pony Diffusion (underscore)
    "pony_diffusion_lora_down_up": ("underscore", "lora_down_up"),
    "pony_diffusion_lora_a_b": ("underscore", "lora_a_b"),
    # LyCORIS (unknown, default underscore)
    "lycoris": ("underscore", "lora_down_up"),
    # Unknown fallback
    "standard_webui": ("underscore", "lora_down_up"),
    "comfy_native": ("dot", "lora_a_b"),
    "forge_optimized": ("underscore", "lora_down_up"),
    "unknown": ("underscore", "lora_down_up"),
}

def get_format_style(format_str: str) -> tuple[str, str]:
    """
    Return (separator_style, naming_style) for a given format string.
    separator_style: "dot" or "underscore"
    naming_style: "lora_a_b" or "lora_down_up"
    """
    return _FORMAT_STYLE_MAP.get(format_str, ("underscore", "lora_down_up"))




