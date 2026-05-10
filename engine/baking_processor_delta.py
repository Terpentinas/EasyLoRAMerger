"""
Delta reconstruction, alpha/rank scaling, and shape alignment utilities.

Extracted from baking_processor.py to reduce file size and isolate
tensor-level operations from key matching logic.
"""

import torch
from typing import Optional, Union

from ..utils import ProgressTracker
from .scale_utils import apply_alpha_correction


# ===================================================================
# LoRA Delta Reconstruction
# ===================================================================

def _reconstruct_lora_delta(
    lora_sd: dict[str, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, torch.Tensor]:
    """
    Reconstruct LoRA weight deltas from decomposed A/B pairs.

    For each pair (lora_B, lora_A) sharing a base key:
        delta = B @ A

    Then applies (alpha / rank) scaling to produce the final delta
    that represents the weight change for this LoRA key.

    The rank is inferred from the A tensor (first dim = rank).
    Alpha is looked up from lora_sd using:
      - base_key + '.alpha'
      - scaled = (alpha / rank) * delta

    If no alpha key exists, defaults to alpha = rank (no scaling).

    Also handles:
      - Standalone tensors (pre-baked deltas) — passed through unchanged
      - Bias tensors — passed through unchanged (no A/B decomposition)

    NOTE: Suffix matching handles both `.lora_A.weight` (normalized output)
    and `.lora_A` (ComfyUI internal format without .weight).

    Args:
        lora_sd: LoRA state dict with A/B decomposed tensors.
        device: If provided (e.g. "cuda:0"), moves A/B tensors to this device
            before the matmul for GPU-accelerated delta reconstruction.
            If None, computes on CPU (original behavior).

    Returns:
        dict of base_key → delta tensor (on the specified device if provided).
    """

    deltas: dict[str, torch.Tensor] = {}

    # Step 1: Group tensor keys by their base LoRA key
    # A base key is everything before .lora_A[.weight]|.lora_B[.weight]|
    #   .lora_up[.weight]|.lora_down[.weight]|.alpha|.diff
    # For 'diff' (difference) LoRA Studio format, same grouping logic applies.

    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}

    for key, tensor in lora_sd.items():
        # Determine the base key by stripping the suffix
        base_key = None
        component = None  # 'A', 'B', 'alpha', etc.

        # Normalized keys use .lora_A.weight suffix (from convert_sd15_diffusers_to_comfyui)
        # ⚠️ The suffix includes the leading dot, so we must strip its full length.
        #    e.g. key[:-14] for 14-char '.lora_A.weight' removes the entire suffix.
        if key.endswith('.lora_A.weight'):
            base_key = key[:-14]  # '.lora_A.weight' = 14 chars
            component = 'A'
        elif key.endswith('.lora_B.weight'):
            base_key = key[:-14]  # '.lora_B.weight' = 14 chars
            component = 'B'
        elif key.endswith('.lora_up.weight'):
            base_key = key[:-15]  # '.lora_up.weight' = 15 chars
            component = 'A'
        elif key.endswith('.lora_down.weight'):
            base_key = key[:-17]  # '.lora_down.weight' = 17 chars
            component = 'B'
        # Raw/ComfyUI-internal keys may use .lora_A without .weight
        elif key.endswith('.lora_A'):
            base_key = key[:-7]
            component = 'A'
        elif key.endswith('.lora_B'):
            base_key = key[:-7]
            component = 'B'
        elif key.endswith('.lora_up'):
            base_key = key[:-8]
            component = 'A'
        elif key.endswith('.lora_down'):
            base_key = key[:-10]
            component = 'B'
        elif key.endswith('.alpha'):
            base_key = key[:-6]
            component = 'alpha'
        elif key.endswith('.diff'):
            # Pre-baked delta (LoRA Studio 'diff' format)
            base_key = key[:-5]
            component = 'diff'
        else:
            # Standalone tensor — pass through
            deltas[key] = tensor
            continue

        if base_key not in lora_pairs:
            lora_pairs[base_key] = {}
        lora_pairs[base_key][component] = tensor

    # Step 2: Reconstruct delta for each pair
    with ProgressTracker(total=len(lora_pairs), desc="Reconstructing deltas") as delta_progress:
        for base_key, components in lora_pairs.items():
            if 'A' in components and 'B' in components:
                B = components['B']  # lora_down — either (out_dim, rank) or (rank, in_dim)
                A = components['A']  # lora_up   — either (rank, in_dim) or (out_dim, rank)

                # Handle 4D Conv2d tensors (proj_in/proj_out) — squeeze spatial dims
                # Diffusers stores Conv2d LoRA weights as (out_dim, rank, 1, 1) / (rank, in_dim, 1, 1)
                if B.dim() == 4 and A.dim() == 4:
                    if B.shape[2] == 1 and B.shape[3] == 1 and A.shape[2] == 1 and A.shape[3] == 1:
                        B = B.squeeze(-1).squeeze(-1)  # 4D -> 2D
                        A = A.squeeze(-1).squeeze(-1)
                    else:
                        continue  # Non-(1,1) spatial dims — cannot squeeze safely

                # Verify shapes are compatible
                if B.dim() != 2 or A.dim() != 2:
                    continue

                # ===================================================================
                # GPU acceleration: move A/B to target device before matmul
                # ===================================================================
                # A and B tensors come from safe_open → CPU.  Moving them to GPU
                # before the matmul dramatically speeds up delta reconstruction
                # for large SDXL LoRAs (986 matmuls).  Each A/B pair is tiny
                # (rank=16, ~16KB each), so the transfer overhead is negligible.
                if device is not None:
                    A = A.to(device=device)
                    B = B.to(device=device)

                # ===================================================================
                # Convention-agnostic delta reconstruction
                # ===================================================================
                # Two conventions exist for LoRA A/B naming:
                #   Kohya/Diffusers: lora_up=(out_dim,rank), lora_down=(rank,in_dim)
                #     → A = lora_up, B = lora_down, delta = A @ B = (out_dim,rank)@(rank,in_dim)
                #   Transposed:      lora_up=(rank,in_dim), lora_down=(out_dim,rank)
                #     → A = lora_up, B = lora_down, delta = B @ A = (out_dim,rank)@(rank,in_dim)
                #
                # The rank is ALWAYS the smallest of the 4 dimensions
                # (out_dim, rank, rank, in_dim).  We use the ordering where the
                # shared inner dimension equals the rank.
                #
                # This is CRITICAL for square matrices (in_dim==out_dim, e.g. TE attention
                # 768→768) where both orderings pass the shape check but produce
                # different results:
                #   A @ B = (768,16)@(16,768) = (768,768)  ← correct
                #   B @ A = (16,768)@(768,16) = (16,16)    ← WRONG!
                # ===================================================================
                all_dims = [A.shape[0], A.shape[1], B.shape[0], B.shape[1]]
                rank = min(all_dims)

                if A.shape[1] == rank:
                    # A=(out_dim,rank), B=(rank,in_dim) → Kohya convention
                    delta = A @ B
                elif B.shape[1] == rank:
                    # A=(rank,in_dim), B=(out_dim,rank) → transposed convention
                    delta = B @ A
                else:
                    # Incompatible shapes — skip this pair
                    continue

                # Apply alpha/rank scaling
                # NOTE: rank was already correctly computed as min(all_dims) on
                # line 142.  DO NOT reassign rank here — for Kohya convention,
                # A = lora_up = (out_dim, rank), so A.shape[0] = out_dim (e.g.
                # 3072), NOT the rank (16).  Kohya uses
                # dim = down_weight.size()[0] (= rank) at merge_lora.py:87.
                alpha = components.get('alpha')

                if alpha is not None:
                    alpha_val = alpha.item() if hasattr(alpha, 'item') else float(alpha)
                    delta = apply_alpha_correction(delta, alpha_val, rank, mode="linear")

                deltas[base_key] = delta

            elif 'diff' in components:
                # Pre-baked delta — pass through unchanged
                deltas[base_key] = components['diff']

            delta_progress += 1

    return deltas
