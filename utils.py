# utils.py - REMOVE apply_density function (keep it only in methods.py)
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Rank Dimension Detection ----------
def detect_rank_dimensions(shape, key):
    """
    Smart detection of which dimensions are rank dimensions.
    Uses shape heuristics first, key names as fallback.
    """
    if len(shape) != 2:
        return []  # Not a standard 2D LoRA weight
    
    # Heuristic 1: Rank is usually the smaller dimension
    # and model dimensions are typically >= 768
    if shape[0] <= 256 and shape[1] >= 512:
        return [0]  # rank is dim 0
    elif shape[1] <= 256 and shape[0] >= 512:
        return [1]  # rank is dim 1
    
    # Heuristic 2: Very rectangular tensors (rank << model dim)
    if shape[1] >= 4 * shape[0]:
        return [0]  # rank is dim 0
    elif shape[0] >= 4 * shape[1]:
        return [1]  # rank is dim 1
    
    # Fallback to key name hints
    key_lower = key.lower()
    if "down" in key_lower or "_a" in key_lower or ".a" in key_lower:
        return [0]  # Usually rank is dimension 0
    elif "up" in key_lower or "_b" in key_lower or ".b" in key_lower:
        return [1]  # Usually rank is dimension 1
    
    return []  # Couldn't determine


# ---------- Rank Projection ----------
def random_projection(out_rank, in_rank, device):
    """Create orthonormal projection matrix."""
    mat = torch.randn(out_rank, in_rank, device=device)
    q, _ = torch.linalg.qr(mat)
    return q


def project_rank(t1, t2, key):
    """
    Match tensor shapes via projection for rank dimensions,
    padding for non-rank dimensions.
    """
    s1, s2 = list(t1.shape), list(t2.shape)

    if s1 == s2:
        return t1, t2

    max_dims = [max(a, b) for a, b in zip(s1, s2)]
    
    def expand_tensor(t, target_shape, key):
        result = t
        
        # Get rank dimensions for this tensor
        rank_dims = detect_rank_dimensions(t.shape, key)
        
        for dim in range(len(target_shape)):
            if result.shape[dim] < target_shape[dim]:
                diff = target_shape[dim] - result.shape[dim]
                
                # Check if this is a rank dimension
                if dim in rank_dims:
                    # Project rank instead of padding (better quality)
                    proj = random_projection(
                        target_shape[dim],
                        result.shape[dim],
                        result.device,
                    )
                    
                    if dim == 0:
                        # lora_down style: [rank, in_dim]
                        result = proj @ result
                    elif dim == 1:
                        # lora_up style: [out_dim, rank]
                        result = result @ proj.T
                    else:
                        # Higher dimensions (rare) - use padding
                        pad_shape = list(result.shape)
                        pad_shape[dim] = diff
                        padding = torch.zeros(pad_shape, device=result.device)
                        result = torch.cat([result, padding], dim=dim)
                else:
                    # Non-rank dimension: use zero padding
                    pad_shape = list(result.shape)
                    pad_shape[dim] = diff
                    padding = torch.zeros(pad_shape, device=result.device)
                    result = torch.cat([result, padding], dim=dim)
        
        return result
    
    return expand_tensor(t1, max_dims, key), expand_tensor(t2, max_dims, key)