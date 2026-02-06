# klein_normalizer.py - UNIVERSAL NORMALIZER WITH ALL FORMATS
import torch
import re
from pathlib import Path

# ==================== FORMAT DETECTION ====================

def detect_lora_format(state_dict):
    """Detect which LoRA format we're dealing with."""
    if not state_dict:
        return "unknown"
    
    keys = list(state_dict.keys())
    if not keys:
        return "unknown"
    
    sample_key = keys[0]
    
    # 1. Check for Pony Diffusion (has lora_te prefix)
    if any("lora_te" in k for k in keys):
        return "pony_diffusion"  # Simplified, just one type
    
    # 2. Check for SDXL Kohya format (has input_blocks/middle_block with underscores)
    if any("input_blocks_" in k or "middle_block_" in k or "output_blocks_" in k for k in keys):
        if any("lora_A" in k or "lora_B" in k for k in keys):
            if any("proj_in" in k or "proj_out" in k for k in keys):
                return "sdxl_kohya"
            else:
                return "sd15_kohya"
    
    # Count different key patterns
    lora_down_count = sum(1 for k in keys if "lora_down" in k and "weight" in k)
    lora_up_count = sum(1 for k in keys if "lora_up" in k and "weight" in k)
    lora_a_count = sum(1 for k in keys if "lora_A" in k and "weight" in k)
    lora_b_count = sum(1 for k in keys if "lora_B" in k and "weight" in k)
    
    # Check for Pony Diffusion (has lora_te prefix and lora_down/lora_up)
    if any("lora_te" in k for k in keys):
        if lora_down_count > 0 and lora_up_count > 0:
            return "pony_diffusion_lora_down_up"
        elif lora_a_count > 0 and lora_b_count > 0:
            return "pony_diffusion_lora_a_b"
    
    # Check for Flux Klein with lora_down/lora_up
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
    
    # Fall back to original detection logic
    sample_key = keys[0]
    
    # 1. Check for Pony Diffusion (has lora_te prefix and lora_down/lora_up)
    if any("lora_te" in k for k in keys) and any("lora_down" in k or "lora_up" in k for k in keys):
        return "pony_diffusion"
    
    # 2. Check for SDXL Kohya format (has input_blocks/middle_block with underscores)
    if any("input_blocks_" in k or "middle_block_" in k or "output_blocks_" in k for k in keys):
        if any("lora_A" in k or "lora_B" in k for k in keys):
            if any("proj_in" in k or "proj_out" in k for k in keys):
                return "sdxl_kohya"
            else:
                return "sd15_kohya"
    
    # 2. Check for SDXL Diffusers format
    if any("lora.down" in k or "lora.up" in k for k in keys):
        return "sdxl_diffusers"
    
    # 3. Check for Z-Image (AI-Toolkit format)
    if any("diffusion_model.layers." in k for k in keys):
        if any("adaLN_modulation" in k for k in keys):
            return "z_image_ai_toolkit"
        else:
            return "z_image_other"
    
    # 4. Check for Z-Image (default.weight format)
    if any(".lora_A.default.weight" in k or ".lora_B.default.weight" in k for k in keys):
        return "z_image_default"
    
    # 5. Check for Flux Klein 9B (has 4096 inner dim)
    if any("diffusion_model.double_blocks." in k for k in keys):
        # Check shape of first few tensors
        for k in keys[:5]:
            if "lora_A" in k and state_dict[k].shape[1] == 4096:
                return "flux_klein_9b"
        return "flux_klein_4b"
    
    # 6. Check for Flux Klein 4B (with single_blocks)
    if any("diffusion_model.single_blocks." in k for k in keys):
        return "flux_klein_4b"
    
    # 7. Check for Flux Klein (general)
    if any("diffusion_model." in k and ("double_blocks" in k or "single_blocks" in k) for k in keys):
        return "flux_klein_4b"  # Default to 4B
    
    # 8. Check for Musubi Klein
    if "lora_unet_" in sample_key:
        return "musubi_klein"
    
    # 9. Check for SD1.5 Kohya
    if any("lora_A" in k or "lora_B" in k for k in keys):
        return "sd15_kohya"
    
    # 10. Check for LyCORIS
    if any("lora." in k and ("alpha" in k or "dyn" in k) for k in keys):
        return "lycoris"
    
    # 11. Check for Pony
    if any("transformer" in k and "lora" in k for k in keys):
        return "pony_diffusion"
    
    return "unknown"

# ==================== SDXL KOHYA NORMALIZER ====================

def normalize_sdxl_kohya(state_dict):
    """Convert SDXL Kohya format (with underscores) to standard dot notation."""
    normalized = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert underscores to dots in block patterns
        # Example: input_blocks_4_1_proj_in → input_blocks.4.1.proj_in
        patterns = [
            (r'input_blocks_(\d+)_(\d+)_', r'input_blocks.\1.\2.'),
            (r'middle_block_(\d+)_', r'middle_block.\1.'),
            (r'output_blocks_(\d+)_(\d+)_', r'output_blocks.\1.\2.'),
            (r'time_embed_', r'time_embed.'),
        ]
        
        for pattern, replacement in patterns:
            new_key = re.sub(pattern, replacement, new_key)
        
        # Handle transformer blocks within attention layers
        if 'transformer_blocks' in new_key:
            new_key = re.sub(r'transformer_blocks_(\d+)_', r'transformer_blocks.\1.', new_key)
        
        # Already uses lora_A/lora_B format
        # Ensure proper suffix
        if not new_key.endswith(".weight") and ".alpha" not in new_key and "bias" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

# ==================== SDXL DIFFUSERS NORMALIZER ====================

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

# ==================== Z-IMAGE NORMALIZERS ====================

def normalize_z_image_ai_toolkit(state_dict):
    """Convert Z-Image AI-Toolkit format to standard."""
    normalized = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Already uses lora_A/lora_B format, just ensure consistency
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
        new_key = new_key.replace(".lora_A.default", ".lora_A")
        new_key = new_key.replace(".lora_B.default", ".lora_B")
        
        # Ensure .weight suffix
        if not new_key.endswith(".weight") and ".alpha" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

# ==================== FLUX KLEIN NORMALIZERS ====================

def normalize_flux_klein_9b(state_dict):
    """Convert Flux Klein 9B format to standard (same as 4B)."""
    # Uses same normalization as Klein 4B
    return normalize_all_klein_formats(state_dict)

def normalize_all_klein_formats(state_dict):
    """Standardizes Musubi and AI-Toolkit names into one format."""
    normalized = {}
    
    # Check if this is Pony Diffusion format
    is_pony = any("lora_te" in k for k in state_dict.keys())
    
    for key, value in state_dict.items():
        new_key = key
        
        if is_pony:
            # Pony Diffusion format: lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight
            # Keep as lora_down/lora_up for Pony Diffusion (ComfyUI expects this)
            # Just convert underscores to dots for consistency
            new_key = new_key.replace("lora_", "")
            new_key = re.sub(r'_([0-9]+)_', r'.\1.', new_key)
            new_key = re.sub(r'_([0-9]+)\.', r'.\1.', new_key)
            new_key = re.sub(r'\.([0-9]+)_', r'.\1.', new_key)
            new_key = new_key.replace("_", ".")
            
            # DON'T convert lora_down/lora_up for Pony Diffusion!
            # Keep them as lora_down.weight and lora_up.weight
            # Only ensure .weight suffix if missing
            if ("lora_down" in new_key or "lora_up" in new_key) and "weight" not in new_key and "alpha" not in new_key:
                new_key += ".weight"
        else:
            # Original Klein format handling
            new_key = re.sub(r'(img|txt)_attn_(proj|qkv)', r'\1_attn.\2', new_key)
            new_key = re.sub(r'(img|txt)_mlp_', r'\1_mlp.', new_key)
            
            if "single_blocks" in new_key:
                new_key = re.sub(r'single_blocks_(\d+)_attn_(proj|qkv)', r'single_blocks.\1.attn.\2', new_key)
                new_key = re.sub(r'single_blocks_(\d+)_mlp_', r'single_blocks.\1.mlp.', new_key)
            
            if "lora_unet_" in new_key:
                new_key = new_key.replace("lora_unet_", "diffusion_model.")
            
            # Convert lora_down/lora_up to lora_A/lora_B for non-Pony formats
            new_key = new_key.replace("lora_down", "lora_A").replace("lora_up", "lora_B")
            new_key = new_key.replace("attn_proj", "attn.proj").replace("attn_qkv", "attn.qkv")
        
        # Remove any double dots
        while ".." in new_key:
            new_key = new_key.replace("..", ".")
        
        # Add .weight suffix if missing for weight tensors (for non-Pony)
        if not is_pony and ("lora_A" in new_key or "lora_B" in new_key) and "weight" not in new_key and "alpha" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

def finalize_all_klein_keys(state_dict):
    """Final check before saving to ensure ComfyUI can read it."""
    finalized = {}
    for key, value in state_dict.items():
        new_key = key
        
        # Ensure all attn layers use dots
        new_key = new_key.replace("attn_proj", "attn.proj").replace("attn_qkv", "attn.qkv")
        
        # Make sure single_blocks uses dots
        new_key = re.sub(r'single_blocks_(\d+)_', r'single_blocks.\1.', new_key)
        
        # Remove any double dots
        while ".." in new_key:
            new_key = new_key.replace("..", ".")
        
        finalized[new_key] = value
    
    return finalized

# ==================== SD1.5 & OTHER NORMALIZERS ====================

def normalize_sd15_kohya(state_dict):
    """Convert SD1.5 Kohya format to standard."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        
        # Already uses lora_A/lora_B format
        # Ensure proper suffix
        if not new_key.endswith(".weight") and ".alpha" not in new_key and "bias" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

def normalize_lycoris(state_dict):
    """Convert LyCORIS format to standard."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        
        # LyCORIS uses lora. prefix
        if "lora." in new_key and ("alpha" in new_key or "dyn" in new_key):
            # Keep as-is for LyCORIS specific keys
            pass
        elif not new_key.endswith(".weight") and ".alpha" not in new_key:
            new_key += ".weight"
        
        normalized[new_key] = value
    
    return normalized

# ==================== UNIVERSAL NORMALIZER ====================

def normalize_key_for_flux(key: str) -> str:
    """Normalize Flux model key to standard format."""
    # Replace img_attn_proj -> img_attn.proj
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
    
    # Remove any double dots
    while '..' in normalized:
        normalized = normalized.replace('..', '.')
    
    return normalized

# Update the universal_normalize function in klein_normalizer.py:
# (Add this to your existing klein_normalizer.py file)

def universal_normalize(state_dict):
    """Universal normalizer that detects format and applies appropriate conversion."""
    format_type = detect_lora_format(state_dict)
    print(f"🔍 Detected format: {format_type}")
    
    # CRITICAL FIX: Skip normalization for Pony Diffusion
    if "pony" in format_type.lower():
        print("   Keeping Pony Diffusion keys as-is")
        return state_dict
    
    # Apply appropriate normalization for other formats
    if format_type == "sdxl_kohya":
        return normalize_sdxl_kohya(state_dict)
    elif format_type == "sdxl_diffusers":
        return normalize_sdxl_diffusers(state_dict)
    elif format_type == "z_image_ai_toolkit":
        return normalize_z_image_ai_toolkit(state_dict)
    elif format_type == "z_image_default":
        return normalize_z_image_default(state_dict)
    elif format_type == "z_image_other":
        try:
            return normalize_z_image_ai_toolkit(state_dict)
        except:
            return state_dict
    elif format_type == "flux_klein_9b":
        return normalize_flux_klein_9b(state_dict)
    elif format_type in ["musubi_klein", "flux_klein_4b", "ai_toolkit_klein"]:
        normalized = normalize_all_klein_formats(state_dict)
        # Additional Flux-specific normalization
        final_normalized = {}
        for key, value in normalized.items():
            new_key = normalize_key_for_flux(key)
            final_normalized[new_key] = value
        return final_normalized
    elif format_type == "sd15_kohya":
        return normalize_sd15_kohya(state_dict)
    elif format_type == "lycoris":
        return normalize_lycoris(state_dict)
    elif format_type == "pony_diffusion":
        # This shouldn't be reached due to the check above, but as fallback
        return state_dict
    else:
        print(f"⚠️  Unknown format, trying Klein normalization")
        try:
            normalized = normalize_all_klein_formats(state_dict)
            # Try Flux normalization
            final_normalized = {}
            for key, value in normalized.items():
                new_key = normalize_key_for_flux(key)
                final_normalized[new_key] = value
            return final_normalized
        except:
            print(f"⚠️  Could not normalize, using raw format")
            return state_dict

def universal_finalize(state_dict):
    """Universal finalizer that ensures proper alpha values and ComfyUI compatibility."""
    # First detect what format we're working with
    format_type = detect_lora_format(state_dict)
    
    finalized = state_dict.copy()
    
    # Track what we've modified
    modifications = []
    
    if "lora_down_up" in format_type:
        # Format uses lora_down/lora_up - ensure we have alpha keys
        for key in list(state_dict.keys()):
            if "lora_down" in key and "weight" in key:
                # Create corresponding alpha key
                alpha_key = key.replace("lora_down.weight", "alpha")
                if alpha_key not in finalized:
                    # Determine appropriate alpha value
                    tensor = state_dict[key]
                    rank = tensor.shape[0] if len(tensor.shape) >= 2 else 16
                    alpha_value = min(rank, 16.0)
                    finalized[alpha_key] = torch.tensor(float(alpha_value))
                    modifications.append(f"Added alpha: {alpha_key}")
        
        # Also ensure lora_up keys exist
        for key in list(state_dict.keys()):
            if "lora_up" in key and "weight" in key:
                # Already have lora_up, keep it
                pass
        
        print(f"📝 Prepared {len(modifications)} alpha keys for lora_down/lora_up format")
        return finalized
    
    elif "lora_a_b" in format_type:
        # Format uses lora_A/lora_B - ensure we have alpha keys
        alpha_added = 0
        for key in list(state_dict.keys()):
            if "lora_A" in key and "weight" in key:
                alpha_key = key.replace("lora_A.weight", "alpha")
                if alpha_key not in finalized:
                    tensor = state_dict[key]
                    rank = tensor.shape[0] if len(tensor.shape) >= 2 else 16
                    alpha_value = min(rank, 16.0)
                    finalized[alpha_key] = torch.tensor(float(alpha_value))
                    alpha_added += 1
        
        if alpha_added > 0:
            print(f"📝 Added {alpha_added} alpha values for lora_A/lora_B format")
        
        return finalized
    
    else:
        # Unknown format, use default logic
        alpha_added = 0
        for key in list(state_dict.keys()):
            if ("lora_A" in key and "weight" in key) or \
               ("lora.down" in key and "weight" in key) or \
               ("lora_down" in key and "weight" in key):
                
                if "lora_A" in key:
                    alpha_key = key.replace("lora_A.weight", "alpha")
                elif "lora.down" in key:
                    alpha_key = key.replace("lora.down.weight", "alpha")
                elif "lora_down" in key:
                    alpha_key = key.replace("lora_down.weight", "alpha")
                
                if alpha_key not in finalized:
                    finalized[alpha_key] = torch.tensor(16.0)
                    alpha_added += 1
        
        if alpha_added > 0:
            print(f"📝 Added {alpha_added} alpha values")
        
        return finalized

# ==================== RANK DETECTION & ADJUSTMENT ====================

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
    """Silent version that doesn't print."""
    current_rank = safe_get_rank(tensor, key)
    
    if current_rank == target_rank:
        return tensor
    
    # Determine if this is lora_A or lora_B
    is_lora_b = any(x in key.lower() for x in ['lora_b', 'lora.up', 'lora_up'])
    
    # Zero padding
    if not is_lora_b:
        if len(tensor.shape) > 1:
            new_shape = (target_rank, tensor.shape[1])
            new_tensor = torch.zeros(new_shape, device=tensor.device, dtype=tensor.dtype)
            min_rank = min(current_rank, target_rank)
            new_tensor[:min_rank, :] = tensor[:min_rank, :]
        else:
            # 1D tensor (shouldn't happen for lora_A)
            new_tensor = tensor
    else:
        if len(tensor.shape) > 1:
            new_shape = (tensor.shape[0], target_rank)
            new_tensor = torch.zeros(new_shape, device=tensor.device, dtype=tensor.dtype)
            min_rank = min(current_rank, target_rank)
            new_tensor[:, :min_rank] = tensor[:, :min_rank]
        else:
            # Handle 1D tensors
            new_tensor = tensor
    
    return new_tensor

# ==================== BACKWARD COMPATIBILITY ====================

# Keep original functions for backward compatibility
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