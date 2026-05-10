"""
Unified alpha/rank scaling utilities for EasyLoRAMerger.

Consolidates 4 separate implementations of alpha/rank scaling logic into a
single shared module, eliminating duplicated code while preserving exact
behavior for all consumers.

=== Formula Reference ===

    Runtime scaling (merge_engine_v2, triple_methods):
        scale = alpha / rank          # linear mode
        result = up_tensor * scale    # only up-weight scaled

    Baking (bake_alphas in utils.py):
        scale = sqrt(alpha / rank)    # sqrt mode
        down_tensor *= scale          # both weights scaled
        up_tensor   *= scale          # because B * A * sqrt(a/r) * sqrt(a/r) = B * A * (a/r)

    Delta reconstruction (baking_processor_delta):
        scale = alpha / rank          # linear mode
        delta = B @ A                 # reconstructed weight delta
        delta *= scale                # scale the combined result

=== Consumer Mapping ===

    | Consumer                          | Mode      | Target | Data Source           |
    |-----------------------------------|-----------|--------|-----------------------|
    | merge_engine_v2._apply_lora_scaling | linear  | up     | original_sd + mapping |
    | triple_methods.apply_lora_scaling_triple | linear | up  | original_sd + mapping |
    | baking_processor_delta._reconstruct_lora_delta | linear | delta | lora_sd (grouped) |
    | utils.bake_alphas                  | sqrt     | both   | state_dict (in-place) |
"""

from typing import Any, Dict, List, Optional, Tuple

import torch


# ===================================================================
# Alpha Key Candidate Generation
# ===================================================================


def generate_alpha_candidates(key: str) -> List[str]:
    """
    Generate all possible alpha key candidates for a given weight key.

    Merges candidate generation logic from:
      - merge_engine_v2._apply_lora_scaling (7 patterns)
      - triple_methods.apply_lora_scaling_triple (7 patterns, identical)
      - utils._lookup_alpha_value (7 patterns, adds lora_A/B separately)

    Returns a list of candidate alpha keys ordered by specificity.
    The first match in ``find_alpha_value`` will be used.
    """
    candidates: List[str] = []

    # Pattern 1: replace .weight with .alpha
    if key.endswith(".weight"):
        candidates.append(key.replace(".weight", ".alpha"))

    # Pattern 2: replace specific LoRA suffix with 'alpha'
    if "lora_A.weight" in key:
        candidates.append(key.replace("lora_A.weight", "alpha"))
    if "lora_B.weight" in key:
        candidates.append(key.replace("lora_B.weight", "alpha"))
    if "lora_down.weight" in key:
        candidates.append(key.replace("lora_down.weight", "alpha"))
    if "lora_up.weight" in key:
        candidates.append(key.replace("lora_up.weight", "alpha"))

    # Pattern 3: strip suffix after last .lora_ or _lora_
    if ".lora_" in key:
        base = key.split(".lora_")[0]
        candidates.append(f"{base}.alpha")
    if "_lora_" in key:
        base = key.split("_lora_")[0]
        candidates.append(f"{base}.alpha")

    return candidates


# ===================================================================
# Alpha Value Lookup
# ===================================================================


def find_alpha_value(
    original_sd: Dict[str, torch.Tensor],
    key: str,
    mapping: Optional[Dict[str, str]] = None,
    alpha_one_is_rank: bool = False,
    rank: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Find the alpha value for a LoRA weight key in a state dict.

    Parameters
    ----------
    original_sd : Dict[str, torch.Tensor]
        The original LoRA state dict to search for alpha keys.
    key : str
        The weight key (e.g. ``some_block.lora_A.weight``).
    mapping : Optional[Dict[str, str]]
        Optional mapping from normalized key back to original weight key.
        If provided, ``mapping.get(key, key)`` is used as the base for
        candidate generation.
    alpha_one_is_rank : bool
        If True and the found alpha value is exactly 1.0, treat it as
        ``rank`` instead.  Used by ``IdentityMergeEngine`` for converted
        LoRAs where alpha=1 is meaningless but stored as metadata.
    rank : int
        The rank of the tensor.  Only used when ``alpha_one_is_rank=True``.
    metadata : Optional[Dict[str, Any]]
        Optional metadata dict to check for alpha values injected by
        EasyLoRAMerger during previous merges (e.g. ``ss_network_alpha``,
        ``lora_a_ss_network_alpha``, ``lora_b_ss_network_alpha``).
    default : Optional[float]
        Default value to return if no alpha key or metadata is found.
        If None, returns None on miss (original behavior).

    Returns
    -------
    Optional[float]
        The alpha value, or *default* (or None) if no matching alpha
        key was found.
    """
    # Resolve the original weight key via mapping
    orig_key = mapping.get(key, key) if mapping is not None else key

    # Generate candidates and search
    candidates = generate_alpha_candidates(orig_key)
    alpha_value: Optional[float] = None

    for ak in candidates:
        if ak in original_sd:
            alpha_tensor = original_sd[ak]
            if isinstance(alpha_tensor, torch.Tensor):
                if alpha_tensor.numel() == 1:
                    alpha_value = alpha_tensor.item()
                else:
                    alpha_value = alpha_tensor.mean().item()
                break

    # Apply alpha_one_is_rank correction
    if alpha_one_is_rank and alpha_value is not None and alpha_value == 1.0:
        alpha_value = float(rank)

    # Metadata check (EasyLoRAMerger metadata injection from previous merges)
    if alpha_value is None and metadata is not None:
        for mk in ['ss_network_alpha', 'lora_a_ss_network_alpha', 'lora_b_ss_network_alpha']:
            if mk in metadata:
                try:
                    alpha_value = float(metadata[mk])
                    break
                except (ValueError, TypeError):
                    continue

    # Default fallback
    if alpha_value is None and default is not None:
        alpha_value = default

    return alpha_value


# ===================================================================
# Scale Factor Resolution
# ===================================================================


def resolve_scale(
    alpha: float,
    rank: int,
    mode: str = "linear",
) -> float:
    """
    Resolve the alpha/rank scaling factor.

    Parameters
    ----------
    alpha : float
        The alpha value (from LoRA metadata).
    rank : int
        The rank of the LoRA tensor (>= 1).
    mode : str
        ``"linear"`` → returns ``alpha / rank``
        ``"sqrt"``   → returns ``sqrt(alpha / rank)``

    Returns
    -------
    float
        The scaling factor.  Returns 1.0 if ``rank <= 0``.
    """
    if rank <= 0:
        return 1.0

    if mode == "sqrt":
        return (alpha / rank) ** 0.5
    return alpha / rank


# ===================================================================
# Tensor-Level Alpha Correction
# ===================================================================


def apply_alpha_correction(
    tensor: torch.Tensor,
    alpha: float,
    rank: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    Apply alpha/rank scaling correction to a single tensor.

    Equivalent to ``tensor * resolve_scale(alpha, rank, mode)``
    but skips multiplication when the scale factor is essentially 1.0.

    Parameters
    ----------
    tensor : torch.Tensor
        The LoRA weight tensor to scale.
    alpha : float
        The alpha value.
    rank : int
        The rank of the tensor (>= 1).
    mode : str
        ``"linear"`` or ``"sqrt"``.

    Returns
    -------
    torch.Tensor
        The scaled tensor (or the original tensor if scaling ∼= 1.0).
    """
    scale = resolve_scale(alpha, rank, mode)
    if abs(scale - 1.0) > 1e-6:
        return tensor * scale
    return tensor


# ===================================================================
# Baking: Embed Alpha into Both Down and Up Weights
# ===================================================================


def bake_alpha_key(
    down_tensor: torch.Tensor,
    up_tensor: torch.Tensor,
    alpha: float,
    rank: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bake an alpha value into both down and up weight tensors.

    Uses ``sqrt(alpha / rank)`` scaling so that the overall delta
    ``B @ A`` is scaled by ``alpha / rank`` (since
    ``(B * s) @ (A * s) = B @ A * s^2 = B @ A * (alpha / rank)``).

    Parameters
    ----------
    down_tensor : torch.Tensor
        The down (A) weight tensor (e.g. ``lora_A.weight``).
    up_tensor : torch.Tensor
        The up (B) weight tensor (e.g. ``lora_B.weight``).
    alpha : float
        The alpha value to bake in.
    rank : Optional[int]
        The rank of the LoRA (inferred from ``down_tensor`` shape if None).
        For a 2D down tensor ``(R, in_dim)``, rank is ``R``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(scaled_down, scaled_up)`` — both tensors scaled by
        ``sqrt(alpha / rank)``.
    """
    if rank is None:
        # Infer rank: down tensor shape is typically (rank, in_dim)
        rank = down_tensor.shape[0] if down_tensor.dim() >= 2 else 1

    rank = max(1, rank)
    scale = resolve_scale(alpha, rank, mode="sqrt")

    if abs(scale - 1.0) > 1e-6:
        return down_tensor * scale, up_tensor * scale
    return down_tensor, up_tensor


def build_alpha_mapping(state_dict: Dict[str, torch.Tensor], key_map: Dict[str, str]) -> Dict[str, str]:
    """
    Build mapping from normalized alpha keys to original alpha keys.

    For each weight key pair (norm→orig), generates the corresponding
    alpha key pair using the ``.lora_A.weight`` → ``.alpha`` convention.

    This is a **map-building** operation (not a value lookup). It returns
    a dict mapping ``normalized_alpha_key -> original_alpha_key`` for use
    in key-mapping pipelines such as :func:`identity_normalize`.

    Returns:
        Dict mapping normalized_alpha_key -> original_alpha_key
    """
    alpha_map: Dict[str, str] = {}
    for norm_key, orig_key in key_map.items():
        if "lora_A.weight" in norm_key or "lora_down.weight" in norm_key:
            base = norm_key.replace(".lora_A.weight", "").replace(".lora_down.weight", "")
            alpha_key = f"{base}.alpha"
            orig_base = orig_key.replace(".lora_A.weight", "").replace(".lora_down.weight", "")
            target_alpha_key = f"{orig_base}.alpha"
            if target_alpha_key in state_dict:
                alpha_map[alpha_key] = target_alpha_key
    return alpha_map
