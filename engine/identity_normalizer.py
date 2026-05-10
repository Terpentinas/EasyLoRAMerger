"""
Identity Normalizer for Ariadne Project.
Maps original LoRA keys to mathematical keys using universal_normalize with mapping capture.
"""
import re
import torch
from typing import Dict, List, Tuple, Optional

try:
    from .klein_normalizer import universal_normalize
except ImportError:
    from klein_normalizer import universal_normalize
from .scale_utils import build_alpha_mapping

def identity_normalize(state_dict: Dict[str, torch.Tensor], metadata=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Normalize LoRA keys while preserving a mapping from normalized key back to original key.
    Includes alpha keys in the mapping.
    
    Returns:
        normalized_dict: State dict with normalized keys (mathematical keys).
        key_map: Mapping from normalized_key -> original_key (includes weight and alpha keys).
    """
    # Create mapping dict
    mapping = {}
    # Call universal_normalize with mapping_ref
    normalized = universal_normalize(state_dict, metadata=metadata, mapping_ref=mapping)
    
    # Ensure mapping is populated (some normalizers may not use safe_insert).
    # Fallback: compute mapping by matching tensor ids.
    # Preserve special __format__ entry if present
    saved_format = mapping.get("__format__")
    
    if not mapping:
        mapping = compute_mapping_by_tensor_id(state_dict, normalized)
    else:
        # mapping may be incomplete (some keys not captured). Fill gaps.
        missing = set(normalized.keys()) - set(mapping.keys())
        if missing:
            extra_mapping = compute_mapping_by_tensor_id(state_dict, normalized, missing_keys=missing)
            mapping.update(extra_mapping)
    
    # Restore saved format if we had one
    if saved_format is not None:
        mapping["__format__"] = saved_format
    
    # Add explicit alpha‑key mapping (in case tensor‑id matching failed for alpha tensors)
    alpha_map = build_alpha_mapping(state_dict, mapping)
    mapping.update(alpha_map)  # alpha_map entries may already be present; that's fine
    
    # Verify mapping is bijective (no duplicate normalized keys)
    # This is already enforced by safe_insert collisions, but double-check.
    norm_to_orig = {}
    duplicates = []
    for norm_key, orig_key in mapping.items():
        if norm_key in norm_to_orig:
            duplicates.append((norm_key, orig_key, norm_to_orig[norm_key]))
        else:
            norm_to_orig[norm_key] = orig_key
    if duplicates:
        print(f"WARNING: Duplicate normalized keys in mapping (should not happen):")
        for norm_key, orig1, orig2 in duplicates[:5]:
            print(f"  {norm_key} -> {orig1} vs {orig2}")
    
    # Ensure every normalized key has a mapping (should be true)
    missing_mappings = set(normalized.keys()) - set(mapping.keys())
    if missing_mappings:
        print(f"WARNING: {len(missing_mappings)} normalized keys missing mapping; adding identity mapping")
        for k in missing_mappings:
            mapping[k] = k
    
    # Count alpha keys in mapping (original keys containing .alpha)
    alpha_count = sum(1 for orig_key in mapping.values() if '.alpha' in orig_key)
    print(f"[INFO] Identity mapping captured {len(mapping)} key pairs (including {alpha_count} alpha keys)")
    return normalized, mapping

def _extract_key_signature(key: str) -> Tuple[str, ...]:
    """
    Extract a structural signature from a key for fuzzy matching.
    
    Strips common prefixes and normalizes separators, then extracts
    the sequence of (type, value) pairs — layer numbers, operation names, etc.
    
    Example:
      "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.lora_A.weight"
      → ("input", "blocks", "NUM", 1, "NUM", 1, "transformer", "blocks", "NUM", 0, "attn1", "to", "q", "lora", "A", "weight")
    
      "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q.lora_down.weight"
      → ("input", "blocks", "NUM", 1, "NUM", 1, "transformer", "blocks", "NUM", 0, "attn1", "to", "q", "lora", "down", "weight")
    """
    # Strip known prefixes to get at the structural core
    for prefix in ('model.diffusion_model.', 'diffusion_model.', 'lora_unet_',
                   'lora_te_', 'lora_te1_', 'lora_te2_',
                   'cond_stage_model.transformer.', 'cond_stage_model.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    
    # Normalize: split on both '.' and '_'
    parts = re.split(r'[._]+', key)
    
    # Build signature: keep strings as-is, mark digits as ('NUM', int)
    sig_parts: List[Tuple[str, ...]] = []
    for p in parts:
        if p.isdigit():
            sig_parts.append(('NUM', p))
        else:
            sig_parts.append((p,))
    return tuple(sig_parts)


def _score_key_similarity(norm_sig: Tuple, orig_sig: Tuple) -> int:
    """
    Score similarity between two key signatures.
    Higher score = more similar. Considers matching structural elements
    in order, with exact string matches weighted higher than numeric matches.
    """
    score = 0
    # Compare element by element up to min length
    min_len = min(len(norm_sig), len(orig_sig))
    for i in range(min_len):
        n = norm_sig[i]
        o = orig_sig[i]
        if n == o:
            score += 2  # exact string match
        elif len(n) == 2 and len(o) == 2 and n[0] == 'NUM' and o[0] == 'NUM':
            # Both are numeric — check if close values
            if n[1] == o[1]:
                score += 1  # exact numeric value
            elif abs(int(n[1]) - int(o[1])) <= 1:
                score += 0  # adjacent layer (not a penalty)
    # Penalize length mismatch
    score -= abs(len(norm_sig) - len(orig_sig))
    return score


def _find_matching_original_key(
    norm_key: str,
    original: Dict[str, torch.Tensor],
    norm_tensor_shape: torch.Size,
) -> Optional[str]:
    """
    Find the best-matching original key for a normalized key by structural similarity.
    
    Used as fallback when tensor-ID matching fails (e.g., after collision resolution
    in convert_sd15_diffusers_to_comfyui which sums tensors).
    """
    norm_sig = _extract_key_signature(norm_key)
    best_score = -999
    best_key: Optional[str] = None
    
    for orig_key, orig_tensor in original.items():
        orig_sig = _extract_key_signature(orig_key)
        score = _score_key_similarity(norm_sig, orig_sig)
        
        # Boost score if tensor shapes match (strong signal)
        if norm_tensor_shape == orig_tensor.shape:
            score += 5
        
        if score > best_score:
            best_score = score
            best_key = orig_key
    
    # Only return if we have a reasonable match
    if best_score >= 0 and best_key is not None:
        return best_key
    return None


def compute_mapping_by_tensor_id(original: Dict[str, torch.Tensor],
                                 normalized: Dict[str, torch.Tensor],
                                 missing_keys: Optional[set] = None) -> Dict[str, str]:
    """
    Compute mapping by matching tensor objects via id().
    For each normalized key, find the original key with the same tensor object.
    
    When tensor-ID matching fails (e.g., after collision resolution that creates
    new tensor objects via sum()), falls back to key-signature matching.
    """
    # Build a map from tensor id to original key (handle duplicates by preferring first)
    id_to_original = {}
    for k, v in original.items():
        tid = id(v)
        # If duplicate tensor ids (rare), keep first.
        if tid not in id_to_original:
            id_to_original[tid] = k
    
    mapping = {}
    unmatched_keys: List[str] = []
    target_keys = missing_keys if missing_keys is not None else normalized.keys()
    
    for nk in target_keys:
        if nk not in normalized:
            continue
        v = normalized[nk]
        tid = id(v)
        if tid in id_to_original:
            mapping[nk] = id_to_original[tid]
        else:
            # Tensor may have been copied (e.g., after collision resolution in
            # convert_sd15_diffusers_to_comfyui which sums tensors).
            unmatched_keys.append(nk)
    
    if unmatched_keys:
        print(f"   [INFO] Tensor-ID matching failed for {len(unmatched_keys)} keys — "
              f"trying key-signature fallback...")
        matched_by_sig = 0
        for nk in unmatched_keys:
            orig_match = _find_matching_original_key(nk, original, normalized[nk].shape)
            if orig_match is not None:
                mapping[nk] = orig_match
                matched_by_sig += 1
            else:
                mapping[nk] = nk
                print(f"WARNING: Could not map normalized key {nk} to any original tensor")
        
        if matched_by_sig > 0:
            print(f"   ✅ Key-signature fallback matched {matched_by_sig}/{len(unmatched_keys)} keys")
    
    return mapping
