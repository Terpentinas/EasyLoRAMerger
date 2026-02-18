# klein_normalizer.py - UNIVERSAL NORMALIZER WITH ALL FORMATS (CLEANED)
import torch
import re
from pathlib import Path
import hashlib
from collections import defaultdict

# ==================== PERFORMANCE & REPAIR CONFIGURATION ====================

ENABLE_LAZY_CACHE = True
ENABLE_AUTO_REPAIR = True
ENABLE_BATCH_RANK_FIX = True
DEBUG_KEY_DIFF = False  # Enable for debugging conversions

# ==================== CACHE SYSTEM ====================

_lazy_conversion_cache = {}

def _state_dict_signature(state_dict):
    """Generate signature based on keys and tensor shapes to avoid collisions."""
    h = hashlib.sha1()
    keys = list(state_dict.keys())
    
    if not keys:
        return h.hexdigest()
    
    # Sample from beginning, middle, and end
    sample_points = [
        0,  # First
        len(keys) // 4,  # Quarter
        len(keys) // 2,  # Half
        3 * len(keys) // 4,  # Three quarters
        -1  # Last
    ]
    
    for idx in sample_points:
        if 0 <= idx < len(keys):
            k = keys[idx]
            h.update(f"{k}|".encode())
            if k in state_dict:
                tensor = state_dict[k]
                if hasattr(tensor, "shape"):
                    h.update(f"{tensor.shape}|".encode())
                if hasattr(tensor, "dtype"):
                    h.update(f"{tensor.dtype}|".encode())
    
    # Add total key count for uniqueness
    h.update(f"total_keys:{len(keys)}".encode())
    
    return h.hexdigest()

def cached_conversion(convert_fn, state_dict):
    """Cache conversion results to avoid reprocessing."""
    if not ENABLE_LAZY_CACHE:
        return convert_fn(state_dict)

    sig = _state_dict_signature(state_dict)

    if sig in _lazy_conversion_cache:
        print("⚡ Using cached conversion")
        return _lazy_conversion_cache[sig]

    result = convert_fn(state_dict)
    _lazy_conversion_cache[sig] = result
    return result

# ==================== PRE-COMPILED REGEX PATTERNS ====================

LAYER_PATTERN = re.compile(r'layers_(\d+)_')
BLOCK_PATTERN = re.compile(r'(double|single)_blocks_(\d+)_')
ATTN_PATTERN = re.compile(r'(img|txt)_attn_(proj|qkv)')
TO_OUT_PATTERN = re.compile(r'to_out_(\d+)')
MLP_PATTERN = re.compile(r'(img|txt)_mlp_(\d+)')
FEED_FORWARD_PATTERN = re.compile(r'feed_forward_w(\d)')
RESBLOCK_PATTERN = re.compile(r'resblocks\.(\d+)_')

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
        print(f"🔧 {old} → {new}")

def safe_insert(converted, key, value, original_key=None):
    """
    Prevent silent overwriting if two keys collapse into one.
    Keeps first occurrence, warns on collision.
    """
    if key in converted:
        print("⚠️ Key collision detected!")
        if original_key:
            print(f"   Original: {original_key}")
        print(f"   Collides with existing: {key}")
        print("   Keeping first occurrence.")
        return
    converted[key] = value

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

    if len(a_keys) != len(b_keys):
        issues.append("Mismatch between lora_A and lora_B counts")

    for key, tensor in state_dict.items():
        if hasattr(tensor, "shape"):
            # Skip alpha tensors - they can be 0D (scalars)
            if '.alpha' in key and len(tensor.shape) == 0:
                continue  # This is fine!
            if len(tensor.shape) < 2 and '.alpha' not in key:
                issues.append(f"Bad tensor shape: {key}")

    if issues:
        print("⚠️ Broken LoRA detected:")
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
                repaired[alpha_key] = torch.tensor(get_alpha_value(state_dict, key))
                added += 1

    if added:
        print(f"🛠 Added {added} missing alpha keys")

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
        print(f"⚠️ Batch rank fix adjusted {modified_count} tensors")
    else:
        print("⚡ Batch rank fix complete (no changes needed)")
    
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
            print("📋 Note: Metadata preservation requires safetensors object")

    return repaired

def accelerate_lora_load(state_dict):
    """Run repair pipeline on LoRA load."""
    print("⚡ Accelerating LoRA load...")
    state_dict = auto_repair_lora(state_dict)
    return state_dict

# ==================== FORMAT DETECTION ====================

def detect_lora_format(state_dict):
    """Detect which LoRA format we're dealing with."""
    if not state_dict:
        return "unknown"
    
    keys = list(state_dict.keys())
    if not keys:
        return "unknown"

    # Check for SDXL format (should NOT be converted)
    if any("lora_te1" in k for k in keys) and any("input_blocks" in k for k in keys):
        if len(keys) > 2000:  # SDXL has many keys
            return "sdxl_standard"  # Return this instead of musubi

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
    
    if any("lora_unet_" in k for k in keys):
        return "musubi_other_needs_conversion"
    
    # Count different key patterns
    lora_down_count = sum(1 for k in keys if "lora_down" in k and "weight" in k)
    lora_up_count = sum(1 for k in keys if "lora_up" in k and "weight" in k)
    lora_a_count = sum(1 for k in keys if "lora_A" in k and "weight" in k)
    lora_b_count = sum(1 for k in keys if "lora_B" in k and "weight" in k)
    
    # Check for Pony Diffusion
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
    
    # Check for SDXL Kohya format
    if any("input_blocks_" in k or "middle_block_" in k or "output_blocks_" in k for k in keys):
        if any("lora_A" in k or "lora_B" in k for k in keys):
            if any("proj_in" in k or "proj_out" in k for k in keys):
                return "sdxl_kohya"
            else:
                return "sd15_kohya"
    
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

# ==================== MUSUBI CONVERTERS ====================

def convert_musubi_te_to_standard(state_dict):
    """
    Convert Musubi Text Encoder format to standard ComfyUI format.
    
    Musubi TE: lora_te_res_blocks_0_mlp_c_fc.lora_down.weight
    Standard: conditioner.embedders.0.transformer.resblocks.0.mlp.c_fc.lora_A.weight
    """
    print("🔄 Converting Musubi Text Encoder format...")
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

    print(f"✅ Converted {len(converted)} TE keys")
    return converted

def convert_musubi_to_standard(state_dict):
    """Convert Musubi Flux/Klein format to standard."""
    print("🔄 Converting Musubi format...")
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

    print(f"✅ Converted {len(converted)} keys")
    return converted

def convert_musubi_zimage_to_standard(state_dict):
    """Convert Musubi Z-Image format to standard with strict dot-notation."""
    print("🔄 Converting Musubi Z-Image format...")
    converted = {}

    for key, value in state_dict.items():
        # 1. Primary prefix change
        new_key = key.replace("lora_unet_", "diffusion_model.")

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

    print(f"✅ Converted {len(converted)} Z-Image keys")
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

def normalize_sdxl_kohya(state_dict):
    """Convert SDXL Kohya format (with underscores) to standard dot notation."""
    normalized = {}
    
    for key, value in state_dict.items():
        new_key = key
        
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
        
        # Ensure proper suffix
        if not new_key.endswith(".weight") and ".alpha" not in new_key and "bias" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

def normalize_sdxl_diffusers(state_dict):
    """Convert SDXL diffusers format to standard."""
    normalized = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert lora.down → lora_A, lora.up → lora_B
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
    
    # Check if this is Pony Diffusion format
    is_pony = any("lora_te" in k for k in state_dict.keys())
    
    for key, value in state_dict.items():
        new_key = key
        
        if is_pony:
            # Pony Diffusion: Keep lora_down/lora_up
            new_key = new_key.replace("lora_", "")
            new_key = re.sub(r'_([0-9]+)_', r'.\1.', new_key)
            new_key = re.sub(r'_([0-9]+)\.', r'.\1.', new_key)
            new_key = re.sub(r'\.([0-9]+)_', r'.\1.', new_key)
            new_key = new_key.replace("_", ".")
            
            # Ensure .weight suffix
            if ("lora_down" in new_key or "lora_up" in new_key) and "weight" not in new_key and "alpha" not in new_key:
                new_key += ".weight"
        else:
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
        
        # Add .weight suffix for non-Pony
        if not is_pony and ("lora_A" in new_key or "lora_B" in new_key) and "weight" not in new_key and "alpha" not in new_key:
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

def universal_normalize(state_dict, metadata=None):
    """Universal normalizer with format preservation and metadata awareness."""
    
    # Inject metadata into the object so get_alpha_value can find it
    if metadata is not None:
        # We use setattr because some state_dicts are custom objects
        try:
            state_dict.metadata = metadata
        except:
            # If it's a plain dict, we just attach it as a hidden property
            pass

    format_type = detect_lora_format(state_dict)
    print(f"🔍 Detected format: {format_type}")

    # Bypass SDXL standard format
    if format_type == "sdxl_standard":
        print(f"✅ Keeping SDXL format as-is (standard format)")
        return state_dict

    # 1. Run the Auto-Repair/Acceleration pipeline FIRST
    # This ensures Alphas are calculated before keys get renamed
    state_dict = accelerate_lora_load(state_dict)
    
    # ... (rest of your existing bypass and auto-convert logic) ...
    # Bypass standard SD formats
    bypass_formats = ["pony", "illustrious", "sdxl", "sd15", "sd1.5", "kohya"]
    if any(fmt in format_type.lower() for fmt in bypass_formats):
        print(f"✅ Keeping {format_type} format as-is (Standard SD format)")
        return state_dict
    
    # Auto-convert Musubi formats
    if "musubi" in format_type.lower() and "needs_conversion" in format_type:
        print("🔄 Auto-converting Musubi to standard format...")
        
        if "zimage_te" in format_type:
            return convert_musubi_zimage_to_standard(state_dict)
        elif "zimage" in format_type:
            return convert_musubi_zimage_to_standard(state_dict)
        elif "flux" in format_type or "klein" in format_type:
            return convert_musubi_to_standard(state_dict)
        elif "te" in format_type:
            return convert_musubi_te_to_standard(state_dict)
        else:
            return convert_musubi_to_standard(state_dict)
    
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
        "sdxl_kohya": normalize_sdxl_kohya,
        "sdxl_diffusers": normalize_sdxl_diffusers,
        "z_image_ai_toolkit": normalize_z_image_ai_toolkit,
        "z_image_default": normalize_z_image_default,
        "z_image_context_refiner": normalize_z_image_context_refiner,
        "flux_klein_9b": normalize_all_klein_formats,
        "flux_klein_4b": normalize_all_klein_formats,
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
            return final
        return normalized


    
    # Unknown format - try Klein normalization
    print(f"⚠️  Unknown format, trying Klein normalization")
    try:
        normalized = normalize_all_klein_formats(state_dict)
        final = {}
        for key, value in normalized.items():
            new_key = normalize_key_for_flux(key)
            final[new_key] = value
        return final
    except:
        print(f"⚠️  Could not normalize, using raw format")
        return state_dict
    
def universal_finalize(state_dict):
    """Universal finalizer - assumes auto_repair already handled alpha."""
    # No alpha logic here - just final cleanup
    finalized = state_dict.copy()
    
    # Any final non-alpha cleanup (metadata, etc.)
    return finalized

# ==================== RANK UTILITIES ====================

def safe_get_rank(tensor, key):
    """Safely get rank from any LoRA tensor."""
    if len(tensor.shape) < 2:
        return 1  # Alpha or bias
    
    key_lower = key.lower()
    
    # lora_A types
    if any(x in key_lower for x in ['lora_a', 'lora.down', 'lora_down']):
        return tensor.shape[0]  # [rank, in_dim]
    
    # lora_B types
    elif any(x in key_lower for x in ['lora_b', 'lora.up', 'lora_up']):
        if len(tensor.shape) > 1:
            return tensor.shape[1]
        else:
            return tensor.shape[0]
    
    # Default fallback
    else:
        return min(tensor.shape[0], tensor.shape[1])

def silent_pad_or_truncate(tensor, target_rank, key):
    """Pad or truncate tensor to target rank."""
    current_rank = safe_get_rank(tensor, key)
    
    if current_rank == target_rank:
        return tensor
    
    # Determine if this is lora_A or lora_B
    is_lora_b = any(x in key.lower() for x in ['lora_b', 'lora.up', 'lora_up'])
    
    if not is_lora_b:
        if len(tensor.shape) > 1:
            new_shape = (target_rank, tensor.shape[1])
            new_tensor = torch.zeros(new_shape, device=tensor.device, dtype=tensor.dtype)
            min_rank = min(current_rank, target_rank)
            new_tensor[:min_rank, :] = tensor[:min_rank, :]
        else:
            new_tensor = tensor
    else:
        if len(tensor.shape) > 1:
            new_shape = (tensor.shape[0], target_rank)
            new_tensor = torch.zeros(new_shape, device=tensor.device, dtype=tensor.dtype)
            min_rank = min(current_rank, target_rank)
            new_tensor[:, :min_rank] = tensor[:, :min_rank]
        else:
            new_tensor = tensor
    
    return new_tensor

# Backward compatibility
pad_or_truncate = silent_pad_or_truncate

# ==================== FORMAT INFO ====================

def get_format_info(state_dict):
    """Get detailed information about the detected format."""
    format_type = detect_lora_format(state_dict)
    
    info = {
        "format": format_type,
        "total_keys": len(state_dict),
        "has_alpha": any(".alpha" in k for k in state_dict.keys()),
        "unique_ranks": set(),
        "sample_keys": list(state_dict.keys())[:3] if state_dict else []
    }
    
    # Collect unique ranks
    for key, tensor in state_dict.items():
        if len(tensor.shape) >= 2:
            rank = safe_get_rank(tensor, key)
            info["unique_ranks"].add(rank)
    
    info["unique_ranks"] = list(info["unique_ranks"])
    
    return info
