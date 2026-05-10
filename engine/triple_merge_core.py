"""
Shared Triple Merge Core — Unified 3-way merge functions for both LoRA and checkpoint domains.

This module contains the 15 core triple merge functions plus shared helpers,
extracted from the duplicated implementations in triple_methods.py and
checkpoint_methods.py.  No domain-specific logic (LoRA alpha/rank scaling,
checkpoint component scaling, corrupt tensor detection) lives here — that
stays in the domain wrappers.

Each merge function accepts:
    tensors: List[torch.Tensor]  — 3 tensors to merge
    weights: List[float]         — 3 corresponding weights
    blend_mode: str              — "active" or "dense"
    **kwargs                     — method-specific parameters

And returns a single merged torch.Tensor.
"""

import hashlib
import torch
from typing import List

from ._merge_kernels import (
    compute_per_channel_alignment,
    apply_svd_reduction,
    apply_slerp,
    make_dare_mask,
)

# Import MergeMethodRegistry — supports both package-relative and standalone execution
try:
    from ..config import MergeMethodRegistry
except ImportError:
    from config import MergeMethodRegistry


# ========================================================================
# SHARED HELPERS (best-of-both pattern)
# ========================================================================


def _compute_active_masks(tensors, active_threshold=1e-8):
    """Compute 7 active-region masks for 3 tensors.

    From checkpoint_methods — cleaner, single responsibility.

    Returns (only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three).
    """
    a_active = torch.abs(tensors[0]) > active_threshold
    b_active = torch.abs(tensors[1]) > active_threshold
    c_active = torch.abs(tensors[2]) > active_threshold
    only_a = a_active & ~b_active & ~c_active
    only_b = ~a_active & b_active & ~c_active
    only_c = ~a_active & ~b_active & c_active
    both_ab = a_active & b_active & ~c_active
    both_ac = a_active & ~b_active & c_active
    both_bc = ~a_active & b_active & c_active
    all_three = a_active & b_active & c_active
    return only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three


def _weight_tensors(tensors, weights):
    """Return weighted versions of 3 tensors.

    From checkpoint_methods — reusable, single responsibility.
    """
    return (tensors[0] * weights[0],
            tensors[1] * weights[1],
            tensors[2] * weights[2])


def _assign_single_active(result, weighted, only_a, only_b, only_c):
    """Assign single-active regions directly into result.

    From checkpoint_methods — reusable across all active-blend methods.
    """
    if only_a.any():
        result[only_a] = weighted[0][only_a]
    if only_b.any():
        result[only_b] = weighted[1][only_b]
    if only_c.any():
        result[only_c] = weighted[2][only_c]


def _dense_sequential_merge(method_name: str, tensors, weights, **kwargs):
    """Sequential binary merging for dense mode.

    From triple_methods — DRY, delegates to binary MergeMethodRegistry.
    Folds N tensors one at a time, tracking combined weight for correctness.

    NOTE: For non-linear methods (ties_strict, magnitude, cross, etc.) with
    3+ tensors, sequential binary merging is semantically imperfect even with
    correct combined-weight tracking.  True multi-way merging is a future
    enhancement.  Linear (weighted sum) is always correct with this approach.
    """
    merge_fn = MergeMethodRegistry.get_method(method_name)
    tmp = merge_fn(tensors[0], tensors[1], weights[0], weights[1],
                   blend_mode="dense", **kwargs)
    combined_weight = weights[0] + weights[1]
    for i in range(2, len(tensors)):
        tmp = merge_fn(tmp, tensors[i], combined_weight, weights[i],
                       blend_mode="dense", **kwargs)
        combined_weight += weights[i]
    return tmp


def _apply_feature_mix_to_region(active_weighted_list, uniqueness=0.7):
    """Apply feature_mix logic to a list of (already-sliced) weighted tensors.

    From triple_methods — module-level, testable, reusable.
    """
    n = len(active_weighted_list)
    if n == 1:
        return active_weighted_list[0]
    # Compute shares
    magnitudes = torch.stack([w.abs() for w in active_weighted_list])
    total_mag = magnitudes.sum(dim=0) + 1e-8
    shares = magnitudes / total_mag
    max_share, dominant_local = shares.max(dim=0)
    is_dominant = max_share > uniqueness
    # Dominant regions: use the dominant tensor
    region_result = torch.zeros_like(active_weighted_list[0])
    for local_idx, w_tensor in enumerate(active_weighted_list):
        mask = (dominant_local == local_idx) & is_dominant
        region_result = region_result + w_tensor * mask.to(active_weighted_list[0].dtype)
    # Shared regions: average of all active tensors
    shared_mask = (~is_dominant).to(active_weighted_list[0].dtype)
    avg_active = sum(active_weighted_list) / n
    region_result = region_result + avg_active * shared_mask
    return region_result


def _cos_sim_ckpt(a, b):
    """Compute cosine similarity between two flattened tensors.

    From checkpoint_methods — used by ties_gentle.
    """
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    norm_prod = (torch.norm(a_f) * torch.norm(b_f) + 1e-8)
    return torch.dot(a_f, b_f) / norm_prod


def ensure_energy_preservation_triple(
    merged: torch.Tensor,
    tensors: List[torch.Tensor],
    weights: List[float],
    threshold: float = 0.8,
    gain_min: float = 1.0,
    target: str = 'avg',
) -> torch.Tensor:
    """Ensure merged tensor's RMS energy is not below a threshold relative to inputs.

    Identical in both triple_methods and checkpoint_methods.
    """
    energy_merged = torch.sqrt(torch.mean(merged ** 2)).item()
    energies = []
    for t, w in zip(tensors, weights):
        weighted = t * w
        energy = torch.sqrt(torch.mean(weighted ** 2)).item()
        energies.append(energy)
    if target == 'avg':
        target_energy = sum(energies) / len(energies)
    elif target == 'max':
        target_energy = max(energies)
    else:
        target_energy = sum(energies) / len(energies)
    if energy_merged < threshold * target_energy:
        gain = gain_min * target_energy / (energy_merged + 1e-8)
        gain = min(gain, 10.0)
        merged = merged * gain
        print(f"   🔋 Energy safety applied gain {gain:.2f}x")
    return merged


# ========================================================================
# 15 UNIFIED MERGE FUNCTIONS
# ========================================================================


# ── 1. merge_linear (winner: checkpoint_methods) ───────────────────────

def merge_linear(tensors, weights, blend_mode="auto", **kwargs):
    """Linear weighted sum of three tensors.

    Winner: checkpoint_methods — cleaner decomposed helpers.
    """
    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            result[both_ab] = a_w[both_ab] + b_w[both_ab]
        if both_ac.any():
            result[both_ac] = a_w[both_ac] + c_w[both_ac]
        if both_bc.any():
            result[both_bc] = b_w[both_bc] + c_w[both_bc]
        if all_three.any():
            result[all_three] = a_w[all_three] + b_w[all_three] + c_w[all_three]
        return result
    else:
        return _dense_sequential_merge("linear", tensors, weights, **kwargs)


# ── 2. merge_ties_strict (winner: checkpoint_methods) ─────────────────

def merge_ties_strict(tensors, weights, blend_mode="auto", **kwargs):
    """TIES strict — only keep weights where signs agree.

    Winner: checkpoint_methods — same logic, cleaner helpers.
    """
    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            signs_agree = (torch.sign(a_w[both_ab]) == torch.sign(b_w[both_ab])).to(a_w.dtype)
            result[both_ab] = (a_w[both_ab] + b_w[both_ab]) * signs_agree
        if both_ac.any():
            signs_agree = (torch.sign(a_w[both_ac]) == torch.sign(c_w[both_ac])).to(a_w.dtype)
            result[both_ac] = (a_w[both_ac] + c_w[both_ac]) * signs_agree
        if both_bc.any():
            signs_agree = (torch.sign(b_w[both_bc]) == torch.sign(c_w[both_bc])).to(b_w.dtype)
            result[both_bc] = (b_w[both_bc] + c_w[both_bc]) * signs_agree

        if all_three.any():
            signs = [torch.sign(w) for w in [a_w[all_three], b_w[all_three], c_w[all_three]]]
            stacked_signs = torch.stack(signs).float()
            agreement = stacked_signs.mean(dim=0).abs() > 0.5
            merged = a_w[all_three] + b_w[all_three] + c_w[all_three]
            result[all_three] = merged * agreement.to(merged.dtype)
        return result
    else:
        signs = [torch.sign(t * w) for t, w in zip(tensors, weights)]
        agreement = torch.stack(signs).float().mean(dim=0).abs() > 0.5
        return sum(t * w for t, w in zip(tensors, weights)) * agreement


# ── 3. merge_ties_gentle (winner: checkpoint_methods — more sophisticated)

def merge_ties_gentle(tensors, weights, blend_mode="auto", **kwargs):
    """Gentle TIES — cosine-similarity gating per region.

    Winner: checkpoint_methods — has agreement_threshold parameter,
    cosine-similarity gating on dual-active regions, and proper
    all-three handling.
    """
    agreement_threshold = kwargs.get('agreement_threshold', 0.3)

    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            # Per-element sign agreement (scalar cos_sim gating was buggy —
            # a single scalar cannot correctly gate per-element behavior).
            signs_agree = (torch.sign(a_w[both_ab]) == torch.sign(b_w[both_ab])).to(a_w.dtype)
            result[both_ab] = (a_w[both_ab] + b_w[both_ab]) * signs_agree
        if both_ac.any():
            signs_agree = (torch.sign(a_w[both_ac]) == torch.sign(c_w[both_ac])).to(a_w.dtype)
            result[both_ac] = (a_w[both_ac] + c_w[both_ac]) * signs_agree
        if both_bc.any():
            signs_agree = (torch.sign(b_w[both_bc]) == torch.sign(c_w[both_bc])).to(b_w.dtype)
            result[both_bc] = (b_w[both_bc] + c_w[both_bc]) * signs_agree

        if all_three.any():
            signs = [torch.sign(w) for w in [a_w[all_three], b_w[all_three], c_w[all_three]]]
            stacked_signs = torch.stack(signs).float()
            agreement = stacked_signs.mean(dim=0).abs() > agreement_threshold
            merged = a_w[all_three] + b_w[all_three] + c_w[all_three]
            result[all_three] = merged * agreement.to(merged.dtype)
        return result
    else:
        signs = [torch.sign(t * w) for t, w in zip(tensors, weights)]
        agreement = torch.stack(signs).float().mean(dim=0).abs() > agreement_threshold
        return sum(t * w for t, w in zip(tensors, weights)) * agreement


# ── 4. merge_feature_mix (winner: triple_methods — module-level helper)

def merge_feature_mix(tensors, weights, uniqueness=0.7, blend_mode="auto", **kwargs):
    """Feature-mix merging — dominant tensor per position or equal average.

    Winner: triple_methods — uses module-level _apply_feature_mix_to_region
    which is testable and reusable.
    """
    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        weighted = [t * w for t, w in zip(tensors, weights)]
        _assign_single_active(result, weighted, only_a, only_b, only_c)

        if both_ab.any():
            result[both_ab] = _apply_feature_mix_to_region(
                [weighted[0][both_ab], weighted[1][both_ab]], uniqueness=uniqueness)
        if both_ac.any():
            result[both_ac] = _apply_feature_mix_to_region(
                [weighted[0][both_ac], weighted[2][both_ac]], uniqueness=uniqueness)
        if both_bc.any():
            result[both_bc] = _apply_feature_mix_to_region(
                [weighted[1][both_bc], weighted[2][both_bc]], uniqueness=uniqueness)
        if all_three.any():
            result[all_three] = _apply_feature_mix_to_region(
                [weighted[0][all_three], weighted[1][all_three], weighted[2][all_three]],
                uniqueness=uniqueness)
        return result
    else:
        n = len(tensors)
        weighted = [t * w for t, w in zip(tensors, weights)]
        magnitudes = torch.stack([w.abs() for w in weighted])
        total_mag = magnitudes.sum(dim=0) + 1e-8
        shares = magnitudes / total_mag
        max_share, dominant_idx = shares.max(dim=0)
        is_dominant = max_share > uniqueness
        result = torch.zeros_like(weighted[0])
        for i in range(n):
            mask = (dominant_idx == i) & is_dominant
            result = result + weighted[i] * mask.to(weighted[0].dtype)
        shared_mask = (~is_dominant).to(weighted[0].dtype)
        avg_all = sum(weighted) / n
        result = result + avg_all * shared_mask
        return result


# ── 5. merge_magnitude (winner: checkpoint_methods) ───────────────────

def merge_magnitude(tensors, weights, blend=0.5, blend_mode="auto", **kwargs):
    """Magnitude-based merging — picks max-magnitude tensor per element.

    Winner: checkpoint_methods — same logic, cleaner decomposed helpers.
    """
    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            stacked = torch.stack([a_w[both_ab], b_w[both_ab]])
            winner = stacked.abs().argmax(dim=0)
            selected = torch.gather(stacked, 0, winner.unsqueeze(0)).squeeze(0)
            avg = stacked.mean(dim=0)
            result[both_ab] = selected * blend + avg * (1 - blend)
        if both_ac.any():
            stacked = torch.stack([a_w[both_ac], c_w[both_ac]])
            winner = stacked.abs().argmax(dim=0)
            selected = torch.gather(stacked, 0, winner.unsqueeze(0)).squeeze(0)
            avg = stacked.mean(dim=0)
            result[both_ac] = selected * blend + avg * (1 - blend)
        if both_bc.any():
            stacked = torch.stack([b_w[both_bc], c_w[both_bc]])
            winner = stacked.abs().argmax(dim=0)
            selected = torch.gather(stacked, 0, winner.unsqueeze(0)).squeeze(0)
            avg = stacked.mean(dim=0)
            result[both_bc] = selected * blend + avg * (1 - blend)
        if all_three.any():
            stacked = torch.stack([a_w[all_three], b_w[all_three], c_w[all_three]])
            winner = stacked.abs().argmax(dim=0)
            selected = torch.gather(stacked, 0, winner.unsqueeze(0)).squeeze(0)
            avg = stacked.mean(dim=0)
            result[all_three] = selected * blend + avg * (1 - blend)
        return result
    else:
        stacked = torch.stack([t * w for t, w in zip(tensors, weights)])
        magnitudes = stacked.abs()
        winner = magnitudes.argmax(dim=0)
        selected = torch.gather(stacked, 0, winner.unsqueeze(0)).squeeze(0)
        averaged = stacked.mean(dim=0)
        return selected * blend + averaged * (1 - blend)


# ── 6. merge_subtract (winner: checkpoint_methods) ────────────────────

def merge_subtract(tensors, weights, threshold=0.2, blend_mode="auto", **kwargs):
    """Subtract merging (A - B - C) with threshold gating.

    Winner: checkpoint_methods — same logic, cleaner decomposed helpers.
    """
    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)

        # Single active: A positive, B/C negative (subtractees)
        if only_a.any():
            result[only_a] = a_w[only_a]
        if only_b.any():
            result[only_b] = -b_w[only_b]
        if only_c.any():
            result[only_c] = -c_w[only_c]

        # Two active: subtract with threshold gating
        if both_ab.any():
            a_slice = a_w[both_ab]
            b_slice = b_w[both_ab]
            mask = (b_slice.abs() > (threshold * a_slice.abs() + 1e-8)).to(a_slice.dtype)
            result[both_ab] = (a_slice - b_slice) * mask + a_slice * (1 - mask)
        if both_ac.any():
            a_slice = a_w[both_ac]
            c_slice = c_w[both_ac]
            mask = (c_slice.abs() > (threshold * a_slice.abs() + 1e-8)).to(a_slice.dtype)
            result[both_ac] = (a_slice - c_slice) * mask + a_slice * (1 - mask)
        if both_bc.any():
            # A absent: result = -(B + C) with gating on C relative to B
            b_slice = b_w[both_bc]
            c_slice = c_w[both_bc]
            mask = (c_slice.abs() > (threshold * b_slice.abs() + 1e-8)).to(b_slice.dtype)
            result[both_bc] = (-b_slice - c_slice) * mask + (-b_slice) * (1 - mask)

        # All three active: sequential subtract with gating
        # NOTE: mask_c is computed against a_slice (not intermediate) to avoid
        # order-dependent behavior — the gating decision for C should depend
        # on the original A, not on whether B was subtracted first.
        if all_three.any():
            a_slice = a_w[all_three]
            b_slice = b_w[all_three]
            c_slice = c_w[all_three]
            mask_b = (b_slice.abs() > (threshold * a_slice.abs() + 1e-8)).to(a_slice.dtype)
            intermediate = a_slice - b_slice * mask_b
            mask_c = (c_slice.abs() > (threshold * a_slice.abs() + 1e-8)).to(a_slice.dtype)
            result[all_three] = intermediate - c_slice * mask_c

        return result
    else:
        # dense mode: sequential subtract with threshold gating
        result = tensors[0] * weights[0]
        for i in range(1, len(tensors)):
            current = tensors[i] * weights[i]
            if threshold > 1e-8:
                result_magnitude = result.abs()
                current_magnitude = current.abs()
                significant = current_magnitude > (threshold * result_magnitude + 1e-8)
                result = result - current * significant.to(result.dtype)
            else:
                result = result - current
        return result


# ── 7. merge_dare_rescale (winner: checkpoint_methods) ────────────────

def merge_dare_rescale(tensors, weights, density=1.0, seed=None, blend_mode="auto", **kwargs):
    """DARE rescale — random dropout with rescaling.

    If seed is provided, uses a seeded torch.Generator for reproducible results.

    Winner: checkpoint_methods — same logic, cleaner decomposed helpers.
    """
    device = tensors[0].device if tensors else 'cpu'
    rng = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        drop_rate = 1.0 - density
        rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            mask_a = make_dare_mask(a_w[both_ab], drop_rate, dtype=a_w.dtype, rng=rng)
            mask_b = make_dare_mask(b_w[both_ab], drop_rate, dtype=a_w.dtype, rng=rng)
            result[both_ab] = a_w[both_ab] * mask_a * rescale + b_w[both_ab] * mask_b * rescale
        if both_ac.any():
            mask_a = make_dare_mask(a_w[both_ac], drop_rate, dtype=a_w.dtype, rng=rng)
            mask_c = make_dare_mask(c_w[both_ac], drop_rate, dtype=a_w.dtype, rng=rng)
            result[both_ac] = a_w[both_ac] * mask_a * rescale + c_w[both_ac] * mask_c * rescale
        if both_bc.any():
            mask_b = make_dare_mask(b_w[both_bc], drop_rate, dtype=b_w.dtype, rng=rng)
            mask_c = make_dare_mask(c_w[both_bc], drop_rate, dtype=b_w.dtype, rng=rng)
            result[both_bc] = b_w[both_bc] * mask_b * rescale + c_w[both_bc] * mask_c * rescale
        if all_three.any():
            weighted = [a_w[all_three], b_w[all_three], c_w[all_three]]
            dare_results = []
            for w in weighted:
                mask = make_dare_mask(w, drop_rate, dtype=w.dtype, rng=rng)
                dare_results.append(w * mask * rescale)
            result[all_three] = sum(dare_results)
        return result
    else:
        drop_rate = 1.0 - density
        rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
        results = []
        for t, w in zip(tensors, weights):
            mask = make_dare_mask(t, drop_rate, dtype=t.dtype, rng=rng)
            results.append(t * mask * rescale * w)
        return sum(results)


# ── 8. merge_dare_lite (winner: triple_methods — uses _dense_sequential_merge)

def merge_dare_lite(tensors, weights, seed=None, blend_mode="auto", **kwargs):
    """DARE lite — random dropout without rescaling.

    If seed is provided, uses a seeded torch.Generator for reproducible results.

    Winner: triple_methods — uses shared _dense_sequential_merge for dense mode.

    NOTE: Applies dropout per-tensor before summing (matching dare_rescale's
    per-tensor strategy), not to the post-sum merged result.  This ensures
    consistent behavior between the two DARE variants.
    """
    density = kwargs.get('density', 1.0)
    drop_rate = 1.0 - density
    device = tensors[0].device if tensors else 'cpu'
    rng = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            mask_a = make_dare_mask(a_w[both_ab], drop_rate, dtype=a_w.dtype, rng=rng)
            mask_b = make_dare_mask(b_w[both_ab], drop_rate, dtype=a_w.dtype, rng=rng)
            result[both_ab] = a_w[both_ab] * mask_a + b_w[both_ab] * mask_b
        if both_ac.any():
            mask_a = make_dare_mask(a_w[both_ac], drop_rate, dtype=a_w.dtype, rng=rng)
            mask_c = make_dare_mask(c_w[both_ac], drop_rate, dtype=a_w.dtype, rng=rng)
            result[both_ac] = a_w[both_ac] * mask_a + c_w[both_ac] * mask_c
        if both_bc.any():
            mask_b = make_dare_mask(b_w[both_bc], drop_rate, dtype=b_w.dtype, rng=rng)
            mask_c = make_dare_mask(c_w[both_bc], drop_rate, dtype=b_w.dtype, rng=rng)
            result[both_bc] = b_w[both_bc] * mask_b + c_w[both_bc] * mask_c
        if all_three.any():
            mask_a = make_dare_mask(a_w[all_three], drop_rate, dtype=a_w.dtype, rng=rng)
            mask_b = make_dare_mask(b_w[all_three], drop_rate, dtype=b_w.dtype, rng=rng)
            mask_c = make_dare_mask(c_w[all_three], drop_rate, dtype=c_w.dtype, rng=rng)
            result[all_three] = (a_w[all_three] * mask_a +
                                 b_w[all_three] * mask_b +
                                 c_w[all_three] * mask_c)
        return result
    else:
        return _dense_sequential_merge("dare_lite", tensors, weights, seed=seed, **kwargs)


# ── 9. merge_svd_preserve (winner: checkpoint_methods) ────────────────

def merge_svd_preserve(tensors, weights, blend_mode="auto", **kwargs):
    """SVD-based merge with rank reduction.

    Reshapes 4D conv weights to 2D [C_out, C_in*H*W] before SVD for a
    meaningful per-output-channel low-rank decomposition (instead of
    PyTorch's default batched SVD over spatial dims).  Reshapes back
    to the original 4D shape after decomposition.

    1D tensors (biases) and 1x1 matrices skip SVD and fall back to
    linear blend, since their structure has no meaningful low-rank
    approximation.

    Checkpoint compatibility:
        ⚠️ 4D conv weights are now properly flattened for meaningful SVD.
        ⚠️ 1D bias tensors use linear blend (logged).
        ✅ 2D linear weights work optimally.

    Winner: checkpoint_methods — same logic, cleaner decomposed helpers.
    """
    preserve_ratio = kwargs.get('preserve_ratio', 0.8)

    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            result[both_ab] = a_w[both_ab] + b_w[both_ab]
        if both_ac.any():
            result[both_ac] = a_w[both_ac] + c_w[both_ac]
        if both_bc.any():
            result[both_bc] = b_w[both_bc] + c_w[both_bc]

        if all_three.any():
            region = a_w[all_three].float() + b_w[all_three].float() + c_w[all_three].float()
            result[all_three] = apply_svd_reduction(region, preserve_ratio).to(tensors[0].dtype)
        return result
    else:
        combined = tensors[0].float() * weights[0] + tensors[1].float() * weights[1] + tensors[2].float() * weights[2]
        result = apply_svd_reduction(combined, preserve_ratio)
        return result.to(tensors[0].dtype)


# ── 10. merge_noise_aware (winner: checkpoint_methods) ────────────────

def merge_noise_aware(tensors, weights, blend_mode="auto", **kwargs):
    """Noise-aware merging — reduces low-magnitude noise before merging.

    Winner: checkpoint_methods — same logic, cleaner decomposed helpers.
    """
    noise_threshold = kwargs.get('noise_threshold', 0.01)

    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)

        def clean_slice(tensor_slice, threshold_factor=noise_threshold):
            threshold = threshold_factor * tensor_slice.abs().max()
            return torch.where(tensor_slice.abs() < threshold, tensor_slice * 0.1, tensor_slice)

        if only_a.any():
            result[only_a] = clean_slice(a_w[only_a])
        if only_b.any():
            result[only_b] = clean_slice(b_w[only_b])
        if only_c.any():
            result[only_c] = clean_slice(c_w[only_c])

        if both_ab.any():
            result[both_ab] = clean_slice(a_w[both_ab]) + clean_slice(b_w[both_ab])
        if both_ac.any():
            result[both_ac] = clean_slice(a_w[both_ac]) + clean_slice(c_w[both_ac])
        if both_bc.any():
            result[both_bc] = clean_slice(b_w[both_bc]) + clean_slice(c_w[both_bc])
        if all_three.any():
            result[all_three] = clean_slice(a_w[all_three]) + clean_slice(b_w[all_three]) + clean_slice(c_w[all_three])

        return result
    else:
        threshold_a = noise_threshold * tensors[0].abs().max()
        threshold_b = noise_threshold * tensors[1].abs().max()
        threshold_c = noise_threshold * tensors[2].abs().max()
        a_clean = torch.where(tensors[0].abs() < threshold_a, tensors[0] * 0.1, tensors[0])
        b_clean = torch.where(tensors[1].abs() < threshold_b, tensors[1] * 0.1, tensors[1])
        c_clean = torch.where(tensors[2].abs() < threshold_c, tensors[2] * 0.1, tensors[2])
        return a_clean * weights[0] + b_clean * weights[1] + c_clean * weights[2]


# ── 11. merge_gradient_alignment (winner: checkpoint_methods) ─────────

def merge_gradient_alignment(tensors, weights, blend_mode="auto", **kwargs):
    """Gradient-alignment merging — scales by directional similarity.

    NOTE: The active_threshold parameter (default 1e-8) may discard directional
    information from low-magnitude gradient elements.  Users working with very
    small gradient values can override with active_threshold=0 to include all
    elements in alignment computation.

    Winner: checkpoint_methods — uses _assign_single_active, cleaner pattern.
    """
    dtype = tensors[0].dtype

    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            alignment = compute_per_channel_alignment(a_w[both_ab], b_w[both_ab])
            result[both_ab] = a_w[both_ab] * alignment + b_w[both_ab] * alignment
        if both_ac.any():
            alignment = compute_per_channel_alignment(a_w[both_ac], c_w[both_ac])
            result[both_ac] = a_w[both_ac] * alignment + c_w[both_ac] * alignment
        if both_bc.any():
            alignment = compute_per_channel_alignment(b_w[both_bc], c_w[both_bc])
            result[both_bc] = b_w[both_bc] * alignment + c_w[both_bc] * alignment

        if all_three.any():
            a_slice = a_w[all_three]
            b_slice = b_w[all_three]
            c_slice = c_w[all_three]
            align_ab = compute_per_channel_alignment(a_slice, b_slice)
            align_ac = compute_per_channel_alignment(a_slice, c_slice)
            align_bc = compute_per_channel_alignment(b_slice, c_slice)
            alignment = (align_ab + align_ac + align_bc) / 3
            result[all_three] = a_slice * alignment + b_slice * alignment + c_slice * alignment

        return result
    else:
        a_fp32 = tensors[0].float() * weights[0]
        b_fp32 = tensors[1].float() * weights[1]
        c_fp32 = tensors[2].float() * weights[2]
        align_ab = compute_per_channel_alignment(a_fp32, b_fp32)
        align_ac = compute_per_channel_alignment(a_fp32, c_fp32)
        align_bc = compute_per_channel_alignment(b_fp32, c_fp32)
        alignment = (align_ab + align_ac + align_bc) / 3
        merged = a_fp32 * alignment + b_fp32 * alignment + c_fp32 * alignment
        return merged.to(dtype)


# ── 12. merge_slerp (winner: checkpoint_methods — single function) ────

def merge_slerp(tensors, weights, blend_mode="auto", **kwargs):
    """Spherical Linear Interpolation (SLERP) — single function.

    Winner: checkpoint_methods — clean single function with active filtering
    built in, no separate wrapper needed.
    """
    # Filter to active tensors only (|weight| > 0)
    active = [(t, w) for t, w in zip(tensors, weights) if abs(w) > 0]

    if len(active) == 0:
        return torch.zeros_like(tensors[0]) if tensors else torch.tensor(0.0)

    if len(active) == 1:
        return active[0][0] * active[0][1]

    if len(active) > 2:
        print(f"   ⚠️ SLERP requires exactly 2 active sources with |weight| > 0, "
              f"but {len(active)} active detected. Falling back to dense weighted sum.")
        return sum(t * w for t, w in zip(tensors, weights))

    # Exactly 2 active — perform SLERP
    a, b = active[0][0], active[1][0]
    wa, wb = active[0][1], active[1][1]

    total_abs = abs(wa) + abs(wb)
    if total_abs == 0:
        return torch.zeros_like(a)
    t = abs(wa) / total_abs

    orig_shape = a.shape

    # 1D tensors (e.g., bias): early return with linear interpolation
    # Per-channel SLERP is meaningless for single-element channels.
    if a.ndim <= 1:
        return a * (1 - t) + b * t

    result = apply_slerp(a, b, t)
    return result.to(a.dtype)


# ── 13. merge_cross (winner: triple_methods — uses _dense_sequential_merge)

def merge_cross(tensors, weights, blend_mode="auto", **kwargs):
    """Cross-magnitude merge — linear blend plus pairwise cross terms.

    Winner: triple_methods — uses _dense_sequential_merge for dense mode (DRY).
    """
    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            base = a_w[both_ab] + b_w[both_ab]
            cross = (a_w[both_ab].abs() * b_w[both_ab].abs()).sqrt() * 0.3
            result[both_ab] = base + cross * torch.sign(base)
        if both_ac.any():
            base = a_w[both_ac] + c_w[both_ac]
            cross = (a_w[both_ac].abs() * c_w[both_ac].abs()).sqrt() * 0.3
            result[both_ac] = base + cross * torch.sign(base)
        if both_bc.any():
            base = b_w[both_bc] + c_w[both_bc]
            cross = (b_w[both_bc].abs() * c_w[both_bc].abs()).sqrt() * 0.3
            result[both_bc] = base + cross * torch.sign(base)
        if all_three.any():
            base = a_w[all_three] + b_w[all_three] + c_w[all_three]
            cross_sum = (
                (a_w[all_three].abs() * b_w[all_three].abs()).sqrt()
                + (a_w[all_three].abs() * c_w[all_three].abs()).sqrt()
                + (b_w[all_three].abs() * c_w[all_three].abs()).sqrt()
            )
            result[all_three] = base + cross_sum * 0.3 * torch.sign(base)
        return result
    else:
        return _dense_sequential_merge("cross", tensors, weights, **kwargs)


# ── 14. merge_ties_contrast (winner: checkpoint_methods — more sophisticated)

def merge_ties_contrast(tensors, weights, agreement_threshold=0.3,
                        blend_mode="auto", **kwargs):
    """TIES contrast — amplifies disagreements, mutes agreements.

    Winner: checkpoint_methods — more sophisticated all_three logic with
    majority voting and amplification/muting.
    """
    dtype = tensors[0].dtype

    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            a_s = torch.sign(a_w[both_ab]).float()
            b_s = torch.sign(b_w[both_ab]).float()
            disagreement = (a_s != b_s).to(dtype)
            result[both_ab] = (a_w[both_ab] + b_w[both_ab]) * (1.0 + disagreement) * 0.5
        if both_ac.any():
            a_s = torch.sign(a_w[both_ac]).float()
            c_s = torch.sign(c_w[both_ac]).float()
            disagreement = (a_s != c_s).to(dtype)
            result[both_ac] = (a_w[both_ac] + c_w[both_ac]) * (1.0 + disagreement) * 0.5
        if both_bc.any():
            b_s = torch.sign(b_w[both_bc]).float()
            c_s = torch.sign(c_w[both_bc]).float()
            disagreement = (b_s != c_s).to(dtype)
            result[both_bc] = (b_w[both_bc] + c_w[both_bc]) * (1.0 + disagreement) * 0.5

        if all_three.any():
            weighted = [a_w[all_three], b_w[all_three], c_w[all_three]]
            signs = [torch.sign(w).float() for w in weighted]
            base = sum(weighted)
            # Per-element sign counting (tensor-safe, not Python bool on multi-element tensors)
            pos_count = sum((s > 0).float() for s in signs)
            neg_count = sum((s < 0).float() for s in signs)
            majority = (pos_count >= 2).float() - (neg_count >= 2).float()
            no_majority = ((pos_count < 2) & (neg_count < 2)).to(dtype)
            scale = 1.0 + no_majority * 1.0 - (1 - no_majority) * 0.5
            result[all_three] = base * scale
        return result
    else:
        return _dense_sequential_merge("ties_contrast", tensors, weights, **kwargs)


# ── 15. merge_block_swap (winner: triple_methods — uses _dense_sequential_merge)

def _block_swap_2d(tensor_a, tensor_b, block_size, rng):
    """Block-level random selection between 2 tensors (2D only).

    Divides a 2D tensor into block_size x block_size blocks.
    Each block is randomly chosen from A or B using the seeded RNG.
    """
    h, w = tensor_a.shape
    block_h = max(1, h // block_size)
    block_w = max(1, w // block_size)
    result = tensor_a.clone()
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            rand_val = torch.rand(1, generator=rng, device=tensor_a.device).item()
            if rand_val > 0.5:
                i_end = min(i + block_h, h)
                j_end = min(j + block_w, w)
                result[i:i_end, j:j_end] = tensor_b[i:i_end, j:j_end]
    return result


def _block_swap_2d_three(tensor_a, tensor_b, tensor_c, block_size, rng):
    """Block-level random selection among 3 tensors (2D only)."""
    h, w = tensor_a.shape
    block_h = max(1, h // block_size)
    block_w = max(1, w // block_size)
    tensors = [tensor_a, tensor_b, tensor_c]
    result = tensor_a.clone()
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            i_end = min(i + block_h, h)
            j_end = min(j + block_w, w)
            choice = int(torch.randint(0, 3, (1,), generator=rng,
                                       device=tensor_a.device).item())
            result[i:i_end, j:j_end] = tensors[choice][i:i_end, j:j_end]
    return result


def _deterministic_seed_offset(shape, base_offset=0):
    """Compute a deterministic, non-negative seed offset from a tensor shape.

    Uses hashlib.md5 instead of Python's built-in hash(), which is randomized
    per process when PYTHONHASHSEED is enabled (default on Python 3.6+).
    """
    shape_bytes = str(shape).encode()
    digest = int(hashlib.md5(shape_bytes).hexdigest()[:8], 16)
    return (digest + base_offset) % 10000


def merge_block_swap(tensors, weights, block_size=8, seed=42,
                     blend_mode="auto", **kwargs):
    """Block-swapping merge — each block randomly chosen from one source.

    Winner: triple_methods — uses _dense_sequential_merge for dense mode (DRY).

    For 2D tensors in active mode, divides into block_size x block_size blocks
    and randomly assigns each block to one source (matching the binary version
    in methods.py).  For 4D conv weights, flattens to [C_out, C_in*H*W], performs
    block swapping, then reshapes back.  For other dims, falls back to
    element-wise selection.

    Checkpoint compatibility:
        ⚠️ 4D conv weights now properly block-swapped via 2D flatten.
        ⚠️ 1D bias tensors use element-wise random selection.
        ✅ 2D linear weights work optimally.
    """
    if blend_mode == "active":
        active_threshold = kwargs.get('active_threshold', 1e-8)
        only_a, only_b, only_c, both_ab, both_ac, both_bc, all_three = \
            _compute_active_masks(tensors, active_threshold)

        result = torch.zeros_like(tensors[0])
        a_w, b_w, c_w = _weight_tensors(tensors, weights)
        _assign_single_active(result, (a_w, b_w, c_w), only_a, only_b, only_c)

        if both_ab.any():
            region_a = a_w[both_ab]
            region_b = b_w[both_ab]
            rng = torch.Generator(device=a_w.device).manual_seed(seed + 10000)
            if region_a.dim() == 2:
                result[both_ab] = _block_swap_2d(region_a, region_b, block_size, rng)
            elif region_a.dim() == 4:
                # 4D conv: flatten to 2D [C_out, C_in*H*W], block swap, reshape back
                orig_shape = region_a.shape
                result[both_ab] = _block_swap_2d(
                    region_a.reshape(orig_shape[0], -1),
                    region_b.reshape(orig_shape[0], -1),
                    block_size, rng
                ).reshape(orig_shape)
            else:
                # Non-2D/non-4D: element-wise random selection with seeded RNG
                mask = torch.rand(region_a.shape, generator=rng,
                                  device=a_w.device) > 0.5
                result[both_ab] = torch.where(mask, region_a, region_b)

        if both_ac.any():
            region_a = a_w[both_ac]
            region_c = c_w[both_ac]
            rng = torch.Generator(device=a_w.device).manual_seed(seed + 20000)
            if region_a.dim() == 2:
                result[both_ac] = _block_swap_2d(region_a, region_c, block_size, rng)
            elif region_a.dim() == 4:
                orig_shape = region_a.shape
                result[both_ac] = _block_swap_2d(
                    region_a.reshape(orig_shape[0], -1),
                    region_c.reshape(orig_shape[0], -1),
                    block_size, rng
                ).reshape(orig_shape)
            else:
                mask = torch.rand(region_a.shape, generator=rng,
                                  device=a_w.device) > 0.5
                result[both_ac] = torch.where(mask, region_a, region_c)

        if both_bc.any():
            region_b = b_w[both_bc]
            region_c = c_w[both_bc]
            rng = torch.Generator(device=a_w.device).manual_seed(seed + 30000)
            if region_b.dim() == 2:
                result[both_bc] = _block_swap_2d(region_b, region_c, block_size, rng)
            elif region_b.dim() == 4:
                orig_shape = region_b.shape
                result[both_bc] = _block_swap_2d(
                    region_b.reshape(orig_shape[0], -1),
                    region_c.reshape(orig_shape[0], -1),
                    block_size, rng
                ).reshape(orig_shape)
            else:
                mask = torch.rand(region_b.shape, generator=rng,
                                  device=a_w.device) > 0.5
                result[both_bc] = torch.where(mask, region_b, region_c)

        if all_three.any():
            region_a = a_w[all_three]
            region_b = b_w[all_three]
            region_c = c_w[all_three]
            # Deterministic seed offset using hashlib (not Python hash())
            offset = _deterministic_seed_offset(region_a.shape, base_offset=40000)
            rng = torch.Generator(device=a_w.device).manual_seed(seed + offset)
            if region_a.dim() == 2:
                result[all_three] = _block_swap_2d_three(
                    region_a, region_b, region_c, block_size, rng)
            elif region_a.dim() == 4:
                # 4D conv: flatten all 3 to 2D, block swap 3-way, reshape back
                orig_shape = region_a.shape
                a_2d = region_a.reshape(orig_shape[0], -1)
                b_2d = region_b.reshape(orig_shape[0], -1)
                c_2d = region_c.reshape(orig_shape[0], -1)
                result[all_three] = _block_swap_2d_three(
                    a_2d, b_2d, c_2d, block_size, rng
                ).reshape(orig_shape)
            else:
                # Non-2D/non-4D: element-wise random selection with seeded RNG
                choice = torch.randint(0, 3, region_a.shape, generator=rng,
                                       device=a_w.device)
                stacked = torch.stack([region_a, region_b, region_c], dim=-1)
                result[all_three] = stacked.gather(-1, choice.unsqueeze(-1)).squeeze(-1)

        return result
    else:
        return _dense_sequential_merge("block_swap", tensors, weights, **kwargs)


# ========================================================================
# UNIFIED REGISTRY
# ========================================================================

MERGE_METHOD_REGISTRY = {
    "linear": merge_linear,
    "cross": merge_cross,
    "ties_strict": merge_ties_strict,
    "ties_gentle": merge_ties_gentle,
    "ties_contrast": merge_ties_contrast,
    "slerp": merge_slerp,
    "feature_mix": merge_feature_mix,
    "magnitude": merge_magnitude,
    "subtract": merge_subtract,
    "dare_rescale": merge_dare_rescale,
    "dare_lite": merge_dare_lite,
    "svd_preserve": merge_svd_preserve,
    "block_swap": merge_block_swap,
    "noise_aware": merge_noise_aware,
    "gradient_alignment": merge_gradient_alignment,
}
