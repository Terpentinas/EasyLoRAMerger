"""
Easy LoRA Extractor — Extract a LoRA from the delta between two checkpoints.

Architecture-aware key matching using checkpoint_normalizer, float32-safe delta
computation, and SVD-based compression into standard ComfyUI-native LoRA format.

Inputs:
    checkpoint_base  — The reference (unedited) checkpoint.
    checkpoint_tuned — The fine-tuned checkpoint whose concept to extract.
    rank             — Target rank for the extracted LoRA.
    alpha            — LoRA alpha scaling factor.
    svd_mode         — 'auto_energy': energy-threshold auto rank selection;
                       'manual': use the specified rank.
                       'full': full SVD preserving all singular values.

Outputs:
    model      — Modified model (MODEL) with extracted LoRA applied (if model+clip provided).
    clip       — Modified CLIP (CLIP) with extracted LoRA applied (if model+clip provided).
    forensics  — Human-readable extraction forensic report (STRING).
    lora_path  — Path to the saved .safetensors LoRA file (STRING).
"""

from __future__ import annotations

import math
import time
import statistics
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import folder_paths
import comfy.sd
import comfy.utils

from .checkpoint_normalizer import (
    detect_checkpoint_architecture,
    normalize_checkpoint_key,
)
from .key_mapper import checkpoint_key_to_lora_key, checkpoint_key_to_diffusers_key
from ..config import DEVICE_OPTIONS
from ..utils import (
    categorize_checkpoint_key,
    DeviceManager,
    load_checkpoint_with_metadata,
    ProgressTracker,
    get_combined_model_list,
    resolve_model_path,
)
from .metadata_factory import finalize_metadata


# ── Shared constants ──────────────────────────────────────────────────────
_COMPONENT_ORDER = ["unet", "te", "clip", "vae", "other"]
_COMPONENT_LABELS = {
    "unet": "UNET",
    "te": "TE",
    "clip": "CLIP",
    "vae": "VAE",
    "other": "Other",
}


# ============================================================================
# SVD DECOMPOSITION FOR DELTA → LORA
# ============================================================================


# ── Noise floor estimation constants ──────────────────────────────────────
# Gap threshold for elbow detection: if a drop between consecutive normalized
# SVs exceeds this, it marks the signal/noise boundary.
_GAP_THRESHOLD = 0.10


# ── Architecture-aware rank suggestions ────────────────────────────────────
# These are suggested max ranks for each architecture when rank_mode='auto'.
# The actual optimal rank is determined from SVD spectrum analysis and capped
# by this value.
_ARCH_RANK_SUGGESTIONS: Dict[str, int] = {
    "sd15": 64,
    "sdxl": 128,
    "flux": 128,
    "flux2": 128,
    "lumina2": 256,
    "anima": 64,
}


def _aggregate_optimal_rank(
    per_layer_ranks: List[int],
    strategy: str = "p90",
) -> int:
    """
    Aggregate per-layer optimal rank estimates into a single global rank.

    Strategies:
        - 'median':  median of per-layer ranks (conservative, ignores outliers)
        - 'p90':     90th percentile (most layers get full rank)
        - 'max':     max of per-layer ranks (ensure all layers get full rank)
        - 'mean':    mean rounded to nearest int
    """
    if not per_layer_ranks:
        return 64  # fallback
    if strategy == "median":
        return int(statistics.median(per_layer_ranks))
    elif strategy == "p90":
        sorted_ranks = sorted(per_layer_ranks)
        idx = int(len(sorted_ranks) * 0.9)
        return sorted_ranks[max(idx, 0)]
    elif strategy == "max":
        return max(per_layer_ranks)
    else:  # mean
        return int(round(statistics.mean(per_layer_ranks)))


def _find_signal_boundary(S: torch.Tensor) -> int:
    """
    Find the signal/noise boundary index in singular values using gap detection.

    Returns the index of the first SV in the noise tail, or ``len(S)``
    if no clear boundary is detected.  Same gap-detection logic as
    ``_estimate_sv_noise_threshold`` but returns an index instead of a
    threshold value.
    """
    q = len(S)
    if q < 4:
        return q

    S_norm = S / S[0]
    gaps = S_norm[:-1] - S_norm[1:]

    max_gap, max_idx = torch.max(gaps, dim=0)
    max_gap_val = max_gap.item()
    max_idx_val = max_idx.item()

    if (max_gap_val > _GAP_THRESHOLD
            and max_idx_val >= 1
            and max_idx_val < q // 2
            and max_idx_val + 1 < q
            and S[max_idx_val + 1] / S[0] < 0.05):
        return max_idx_val + 1  # first SV in noise tail

    return q  # no clear boundary


def _estimate_sv_noise_threshold(
    S: torch.Tensor,
    M: int,
    N: int,
    noise_factor: float = 2.0,
) -> float:
    """
    Estimate a noise floor threshold for singular values using gap/elbow detection.

    Looks for the largest drop between consecutive normalized singular values.
    If a clear elbow is found (gap > 10%, not at index 0, in first half of SVs,
    with a truly small noise tail <5% of S[0]), the noise floor is computed as
    the mean of the noise tail SVs × ``noise_factor``.

    If no clear elbow is found, returns ``0.0`` (no filtering) — all SVs are
    treated as signal. The energy-based truncation (``k_energy``) already handles
    rank selection in ``auto_energy`` mode; there is no need for a tail-mean
    fallback that would spuriously classify legitimate signal as noise.

    This follows **P4 (Resist the "Just Add a Guard" Reflex)** from
    working_principles.md: rather than adding more guards for edge cases,
    we made the function simpler so the edge cases don't arise.

    Args:
        S: Singular values tensor, sorted descending.
        M, N: Matrix dimensions (unused — kept for API compatibility).
        noise_factor: Multiplier for the noise tail mean. Default: 2.0.

    Returns:
        Threshold value — only singular values above this are treated as signal.
        Returns ``0.0`` (no filtering) if:
          - fewer than 4 SVs available
          - no clear gap/elbow is detected
          - the noise tail is degenerate (empty, non-finite, or zero mean)
    """
    q = len(S)
    if q < 4:
        return 0.0  # too few SVs to estimate noise reliably

    # ── Gap-based (elbow) detection ─────────────────────────────────────
    # Find the largest drop between consecutive normalized SVs.
    # A sharp drop indicates the transition from signal to noise.
    S_norm = S / S[0]  # normalize to [0, 1]
    gaps = S_norm[:-1] - S_norm[1:]

    max_gap, max_idx = torch.max(gaps, dim=0)
    max_gap_val = max_gap.item()
    max_idx_val = max_idx.item()

    # Accept gap only if:
    #   - Exceeds _GAP_THRESHOLD (significant elbow)
    #   - NOT at index 0 (the SV[0]→SV[1] drop is natural for any SVD,
    #     NOT a signal→noise boundary — misidentifying it would collapse rank)
    #   - In the first half of SVs (signal→noise transition)
    #   - At least 1 SV remains after the gap (noise tail exists)
    #   - First SV in the noise tail is <5% of S[0] (truly noise-like,
    #     not a gap between two signal-dominated component groups)
    if (max_gap_val > _GAP_THRESHOLD
            and max_idx_val >= 1
            and max_idx_val < q // 2
            and max_idx_val + 1 < q
            and S[max_idx_val + 1] / S[0] < 0.05):
        noise_tail = S[max_idx_val + 1:]
        if len(noise_tail) >= 2:
            noise_floor = noise_tail.mean().item()
            if noise_floor > 0.0 and math.isfinite(noise_floor):
                return noise_floor * noise_factor

    # ── No clear elbow — all SVs are signal ────────────────────────────
    # The energy-based truncation (k_energy) handles rank selection in
    # auto_energy mode. For manual/full modes, keeping all SVs is correct.
    # The old tail-mean fallback (see git history) spuriously classified
    # legitimate signal components as noise for attenuated/baked deltas
    # (e.g. Flux baked LoRA), collapsing the effective rank from ~64 to ~7.
    # Removing it fixes the root cause rather than adding yet another guard.
    return 0.0


def _apply_noise_thresholding(
    S: torch.Tensor,
    k: int,
    key: str,
    mode_label: str,
    M: int,
    N: int,
    reference_k: Optional[int] = None,
) -> int:
    """
    Apply Marchenko-Pastur inspired noise floor thresholding.

    If a noise floor is detected, reduces ``k`` to exclude noise-level
    singular values.  The ``reference_k`` (optional) is shown in the
    diagnostic print (e.g. ``energy_k`` for auto_energy mode).

    Args:
        S: Singular values tensor, sorted descending.
        k: Current rank estimate (will be reduced if noise detected).
        key: Layer key for logging.
        mode_label: Label for the diagnostic print (``energy_k``, ``rank_k``, ``full_k``).
        M, N: Matrix dimensions (passed to noise threshold estimator).
        reference_k: Optional reference K value for logging (e.g. energy-based K).

    Returns:
        Potentially reduced rank after noise filtering.
    """
    if k <= 1:
        return k
    threshold = _estimate_sv_noise_threshold(S, M, N)
    if threshold > 0.0:
        # ── Sanity guard: if threshold exceeds S[0], noise model is invalid ──
        # This happens when the delta has no clear noise tail (e.g., cross-model
        # deltas like Turbo vs. Base, where all SVs are of similar magnitude).
        # The noise model would classify ALL singular values as noise, collapsing
        # the effective rank to 1 — which destroys all signal information.
        if threshold > S[0]:
            ref_k = reference_k if reference_k is not None else k
            print(f"   ℹ️  [{key[:40]:>40s}] noise model invalid: "
                  f"threshold={threshold:.4e} > S[0]={S[0]:.4e} "
                  f"({mode_label}={ref_k}, skipping filter)")
            return k  # unchanged — noise model does not apply
        k_noise = (S > threshold).sum().item()
        k_noise = max(k_noise, 1)
        if k_noise < k:
            ref_k = reference_k if reference_k is not None else k
            print(f"   🔇 [{key[:40]:>40s}] noise-filtered: "
                  f"{mode_label}={ref_k}, noise_k={k_noise}, "
                  f"threshold={threshold:.4e}, "
                  f"sv=[{S[-1]:.4e}..{S[0]:.4e}]")
        k = min(k, k_noise)
    return k


def _decompose_delta_to_lora(
    delta: torch.Tensor,
    key: str,
    rank: int,
    svd_mode: str,
    energy_threshold: float = 0.95,
    noise_thresholding: bool = True,
    device: str = "auto",
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Decompose a 2-D delta matrix into lora_A / lora_B via SVD.

    All math is performed in **float32** for numerical cleanliness, then
    downcast to the delta's original dtype.

    When ``device`` is not ``"cpu"`` and sufficient VRAM is available, the
    float32 working matrix is moved to GPU for accelerated SVD (10-120×
    speedup for large weight matrices). Results are moved back to CPU before
    returning.

    Args:
        delta:  The weight difference ``W_tuned - W_base``, shape ``(M, N)``.
        key:    Human-readable key for logging.
        rank:   Target rank (used in ``manual`` mode; also the maximum rank
                in ``auto_energy`` mode).
        svd_mode: ``"auto_energy"``, ``"manual"``, or ``"full"``.
        energy_threshold: Fraction of cumulative squared singular-value energy
                          to preserve (ignored unless ``svd_mode='auto_energy'``).
        noise_thresholding: When True, apply Marchenko-Pastur inspired noise floor
                            estimation to discard singular values below the noise
                            level. Only active in ``auto_energy`` and ``manual`` modes.
        device:  ``"auto"``, ``"cuda"``, or ``"cpu"``.  When ``"auto"``, uses
                 DeviceManager.get_device() which picks GPU if available and
                 enough VRAM is detected.

    Returns:
        ``(lora_A, lora_B, effective_rank)`` where:
            * ``lora_A`` — shape ``(effective_rank, N)``
            * ``lora_B`` — shape ``(M, effective_rank)``
            * ``effective_rank`` — the rank actually used after truncation.
        Returns ``None`` if the delta is all-zeros, 1-D, or contains NaN/Inf.
    """
    # Handle 4D Conv2d delta: reshape to 2D [C_out, C_in * kH * kW] for SVD.
    # lora_A / lora_B stay 2D — ComfyUI's LoRA loader applies them via
    # (lora_B @ lora_A).reshape(w.shape), which correctly handles 4D weights.
    was_4d = (delta.dim() == 4)
    if was_4d:
        delta = delta.view(delta.shape[0], -1)  # [C_out, C_in*kH*kW]
    elif delta.dim() != 2:
        return None
    M, N = delta.shape
    if M == 0 or N == 0:
        return None

    # NaN / Inf guard
    if not torch.isfinite(delta).all():
        print(f"   ⚠️  Non-finite values in delta [{key}]; clamping to zero")
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero delta → skip
    if torch.abs(delta).max().item() == 0.0:
        return None

    # ── Float32 SVD ──────────────────────────────────────────────────────
    orig_dtype = delta.dtype
    W = delta.float()  # always decompose in float32

    # ── GPU acceleration with VRAM guard ─────────────────────────────────
    moved_to_gpu = False
    if device != "cpu":
        target_device = DeviceManager.get_device(device)
        if target_device.type != "cpu":
            free_vram = DeviceManager.get_free_vram(target_device)
            w_bytes = W.numel() * 4  # float32 = 4 bytes per element
            # SVD needs ~3× the tensor size for working memory (U, S, Vh)
            needed_bytes = w_bytes * 3
            if free_vram is not None and free_vram >= needed_bytes:
                if W.device != target_device:
                    W = W.to(device=target_device, non_blocking=True)
                    moved_to_gpu = True
                    # Log GPU acceleration (truncate key for readability)
                    short_key = key[:48]
                    print(f"      ⚡ GPU SVD [{short_key}] "
                          f"{M}×{N} | "
                          f"{w_bytes / (1024**2):.1f} MB → "
                          f"free VRAM {free_vram / (1024**2):.0f} MB")
            elif free_vram is not None:
                # Not enough VRAM for GPU SVD — log reason
                short_key = key[:48]
                need_str = f"need ≥{needed_bytes / (1024**2):.0f} MB"
                warn_icon = "⚠️" if free_vram < w_bytes * 2 else "ℹ️"
                print(f"      {warn_icon} CPU SVD [{short_key}] "
                      f"({M}×{N}, {need_str}, "
                      f"free VRAM {free_vram / (1024**2):.0f} MB)")

    max_sv = min(M, N)
    q = min(rank, max_sv)

    if svd_mode == "auto_energy":
        # Use randomized SVD up to rank for speed, then compute energy
        q = max(q, 1)
        U, S, V = torch.svd_lowrank(W, q=q, niter=4)
        Vh = V.T  # (q, N) format
        # Compute cumulative energy
        total_energy = torch.sum(S ** 2)
        if total_energy > 0:
            cumulative = torch.cumsum(S ** 2, dim=0)
            k_energy = torch.searchsorted(cumulative, energy_threshold * total_energy).item() + 1
        else:
            k_energy = 1
        k_energy = min(k_energy, q)
        k_energy = max(k_energy, 1)

        # ── Noise-aware thresholding ─────────────────────────────────────
        k_noise = q
        if noise_thresholding and k_energy < q:
            k_noise = _apply_noise_thresholding(S, k_noise, key, "energy_k", M, N, reference_k=k_energy)
        k = min(k_energy, k_noise)
        k = min(k, q)
        k = max(k, 1)

    elif svd_mode == "manual":
        q = max(q, 1)
        U, S, V = torch.svd_lowrank(W, q=q, niter=4)
        Vh = V.T
        k = min(rank, q)
        k = max(k, 1)

        # ── Noise-aware thresholding for manual mode too ─────────────────
        if noise_thresholding:
            k = _apply_noise_thresholding(S, k, key, "rank_k", M, N)

    else:  # "full"
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = max_sv  # keep everything

        # ── Noise-aware thresholding for full mode too ───────────────────
        if noise_thresholding:
            k = _apply_noise_thresholding(S, k, key, "full_k", M, N)

    # Truncate and factorise into lora_A (k, N) and lora_B (M, k)
    #   W ≈ U[:,:k] @ diag(S[:k]) @ Vh[:k,:]
    # Let:
    #   lora_B = U[:,:k] * sqrt(S[:k])   (M, k)
    #   lora_A = sqrt(S[:k]) * Vh[:k,:]  (k, N)
    sqrt_S_k = torch.sqrt(S[:k])
    lora_B = (U[:, :k] * sqrt_S_k).contiguous()   # (M, k)
    lora_A = (sqrt_S_k[:, None] * Vh[:k, :]).contiguous()  # (k, N)

    # Downcast to original dtype
    if lora_A.dtype != orig_dtype:
        lora_A = lora_A.to(dtype=orig_dtype)
    if lora_B.dtype != orig_dtype:
        lora_B = lora_B.to(dtype=orig_dtype)

    # Move results back to CPU if we accelerated on GPU
    if moved_to_gpu:
        lora_A = lora_A.cpu()
        lora_B = lora_B.cpu()

    return lora_A, lora_B, k

# ============================================================================
# FORENSIC REPORT BUILDER
# ============================================================================

def _build_extraction_report(
    base_name: str,
    tuned_name: str,
    base_arch: str,
    tuned_arch: str,
    matched_keys: int,
    skipped_shape: int,
    skipped_non_2d: int,
    skipped_zero_delta: int,
    total_base_keys: int,
    total_tuned_keys: int,
    extracted_layers: int,
    component_breakdown: Dict[str, Dict[str, Any]],
    rank: int,
    svd_mode: str,
    effective_rank_avg: float,
    alpha: float,
    total_energy_extracted: float,
    output_path: str,
    strength_multiplier: Union[str, float] = "auto",
    detection_mode: str = "auto_fast",
    attenuation_factor: float = 1.0,
    attenuation_layers_analyzed: int = 0,
    skipped_insignificant: int = 0,
    component_matching_counts: Optional[Dict[str, int]] = None,
) -> str:
    """Build a human-readable forensic report for the extraction."""
    lines: List[str] = []
    lines.append("--- EASY LORA EXTRACTOR: FORENSIC REPORT ---")
    lines.append(f"BASE CHECKPOINT:     {base_name}")
    lines.append(f"TUNED CHECKPOINT:    {tuned_name}")
    lines.append(f"BASE ARCHITECTURE:   {base_arch}")
    lines.append(f"TUNED ARCHITECTURE:  {tuned_arch}")
    lines.append("")
    lines.append("EXTRACTION PARAMETERS:")
    lines.append(f"   SVD Mode:            {svd_mode}")
    lines.append(f"   Target Rank:         {rank}")
    lines.append(f"   Effective Rank (avg): {effective_rank_avg:.1f}")
    lines.append(f"   Alpha:               {alpha}")
    lines.append(f"   Strength Multiplier: {strength_multiplier}")
    if detection_mode in ("auto_fast", "auto_precise"):
        lines.append(f"   Detection Mode:      {detection_mode}"
                      f"{f'  ({attenuation_layers_analyzed} layers)' if attenuation_layers_analyzed > 0 else ''}")
    else:
        lines.append(f"   Detection Mode:      {detection_mode}")
    if (isinstance(strength_multiplier, str) and strength_multiplier == "auto"
            and attenuation_factor < 0.7):
        applied_mult = 1.0 / max(attenuation_factor, 0.01)
        lines.append(f"   Attenuation: {attenuation_factor:.2f}x -> "
                     f"auto-applied multiplier: {applied_mult:.1f}x")
    lines.append("")
    lines.append("KEY MATCHING STATISTICS:")
    lines.append(f"   Base keys:      {total_base_keys}")
    lines.append(f"   Tuned keys:     {total_tuned_keys}")
    lines.append(f"   Matched keys:   {matched_keys}")
    lines.append(f"   Shape mismatch: {skipped_shape}")
    lines.append(f"   Non-2D skipped: {skipped_non_2d}")
    lines.append(f"   Zero-delta skipped: {skipped_zero_delta}")
    lines.append(f"   Insignificant delta skipped: {skipped_insignificant}")
    lines.append(f"   Extracted layers: {extracted_layers}")
    lines.append("")

    # ── Matching breakdown (by component, before zero-delta filtering) ────
    if component_matching_counts:
        lines.append("KEY MATCHING BY COMPONENT:")
        for comp in _COMPONENT_ORDER:
            count = component_matching_counts.get(comp, 0)
            label = _COMPONENT_LABELS.get(comp, comp.upper())
            lines.append(f"   {label}: {count} keys")
        total_m = sum(component_matching_counts.values())
        lines.append(f"   Total: {total_m} keys")
        lines.append("")

    lines.append("EXTRACTION BREAKDOWN:")
    for comp in _COMPONENT_ORDER:
        data = component_breakdown.get(comp, {})
        count = data.get("count", 0)
        if count == 0:
            continue
        energy = data.get("energy", 0.0)
        avg_rank = data.get("avg_rank", 0.0)
        lines.append(
            f"   {comp.upper()}: {count} layers, "
            f"energy={energy:.4f}, avg_rank={avg_rank:.1f}"
        )
    lines.append("")
    lines.append(f"TOTAL EXTRACTION ENERGY: {total_energy_extracted:.4f}")
    lines.append(f"OUTPUT LORA: {output_path}")
    lines.append(f"EXTRACTION DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("--- END FORENSIC REPORT ---")
    return "\n".join(lines)


# ============================================================================
# MAIN NODE CLASS
# ============================================================================

class EasyLoRAExtractor:
    """
    Extract a LoRA from the delta between two checkpoints.

    Given a **base** (unedited) checkpoint and a **tuned** (fine-tuned) checkpoint,
    the node:

    1. Normalizes both state dicts to a shared canonical key-space via
       :mod:`checkpoint_normalizer`.
    2. Matches keys that appear in **both** checkpoints with compatible shapes.
    3. Computes ``Δ = W_tuned.float() - W_base.float()``.
    4. Decomposes each Δ via SVD into ``lora_A`` / ``lora_B`` pairs.
    5. Assembles a standard ComfyUI-native LoRA `.safetensors` file.
    6. If optional model+clip inputs are connected, applies the extracted LoRA
       for immediate downstream preview.
    7. Generates a forensic report with component-level extraction statistics.
    """

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_combined_model_list()
        default_folder = ""
        ckpt_folders = folder_paths.get_folder_paths("checkpoints")
        if ckpt_folders:
            default_folder = str(ckpt_folders[0])

        return {
            "required": {
                # ── SECTION 1: SOURCES ────────────────────────────────────
                "checkpoint_base": (["None"] + checkpoints, {
                    "tooltip": "Reference (unedited) checkpoint — the 'before' state.",
                }),
                "checkpoint_tuned": (["None"] + checkpoints, {
                    "tooltip": "Fine-tuned checkpoint — the 'after' state whose concept to extract.",
                }),

                # ── SECTION 2: LORA PARAMETERS ────────────────────────────
                "rank_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: automatically determine optimal rank from SVD spectrum analysis. "
                               "manual: use the specified rank value below.",
                }),
                "rank": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 320,
                    "step": 1,
                    "tooltip": "Target rank (when rank_mode='manual') or upper bound (when rank_mode='auto').",
                }),
                "alpha_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: alpha = effective_rank × component_scale (e.g., TE=0.5×). "
                               "manual: use the specified alpha value below.",
                }),
                "alpha": ("FLOAT", {
                    "default": 64.0,
                    "min": 1.0,
                    "max": 512.0,
                    "step": 1.0,
                    "tooltip": "Alpha value (only used when alpha_mode='manual').",
                }),
                "svd_mode": (["auto_energy", "manual", "full"], {
                    "default": "auto_energy",
                    "tooltip": "auto_energy: automatic rank selection via energy threshold. manual: use specified rank. full: keep all singular values (no compression).",
                }),
                "energy_threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.50,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Only used in 'auto_energy' mode. Energy retention threshold (0.50–1.0). Higher = more precision (higher effective rank). Lower = more compression.",
                }),
                "noise_thresholding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Marchenko-Pastur noise-aware SVD thresholding. "
                               "Disable for cross-model deltas (e.g., Turbo vs. Base) "
                               "where the noise model may incorrectly reject signal.",
                }),

                # ── SECTION 3: STRENGTH ────────────────────────────────────
                "strength_multiplier": (["auto", "1.5", "2.0", "3.0"], {
                    "default": "auto",
                    "tooltip": "auto: automatically detect baking attenuation and compensate. "
                               "1.5/2.0/3.0: manual multiplier applied before SVD.",
                }),
                "detection_mode": (["auto_fast", "auto_precise"], {
                    "default": "auto_fast",
                    "tooltip": "auto_fast: sample 30 layers for quick attenuation estimate (default). "
                               "auto_precise: analyze ALL layers for exact estimate.",
                }),

                # ── SECTION 4: OUTPUT ─────────────────────────────────────
                "save_trigger": ("BOOLEAN", {"default": False}),
                "filename": ("STRING", {
                    "default": "extracted_lora",
                    "multiline": False,
                    "tooltip": "Filename for the output LoRA (.safetensors added automatically).",
                }),
                "lora_format": (["native", "diffusers"], {
                    "default": "native",
                    "tooltip": "native: standard CivitAI-compatible format (diffusion_model.* keys, merged QKV). "
                               "diffusers: HuggingFace Diffusers format (transformer.* keys, QKV split).",
                }),

                # ── SECTION 5: HARDWARE ────────────────────────────────────
                "device": (DEVICE_OPTIONS, {
                    "default": "auto",
                    "tooltip": "auto: pick best available (CUDA if enough VRAM). cuda: force GPU. cpu: force CPU.",
                }),
            },
            "optional": {
                # ── SECTION 1: SOURCES (continued) ─────────────────────────
                "model": ("MODEL", {
                    "tooltip": "Optional: connect a model to preview the extracted LoRA applied to it.",
                }),
                "clip": ("CLIP", {
                    "tooltip": "Optional: connect a CLIP to preview the extracted LoRA applied to it.",
                }),

                # ── SECTION 3: STRENGTH (continued) ───────────────────────
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Strength of the extracted LoRA when applied to the model (0.0–10.0). Ignored if model/clip not connected.",
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Strength of the extracted LoRA when applied to the CLIP (0.0–10.0). Ignored if model/clip not connected.",
                }),

                # ── SECTION 4: OUTPUT (continued) ─────────────────────────
                "save_folder": ("STRING", {
                    "default": default_folder,
                    "multiline": False,
                    "tooltip": "Output folder. Leave blank to use ComfyUI's default loras folder.",
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import hashlib
        cache_key = (
            kwargs.get("checkpoint_base", "None"),
            kwargs.get("checkpoint_tuned", "None"),
            kwargs.get("rank_mode", "auto"),
            kwargs.get("rank", 64),
            kwargs.get("alpha_mode", "auto"),
            kwargs.get("alpha", 64.0),
            kwargs.get("svd_mode", "auto_energy"),
            kwargs.get("energy_threshold", 0.95),
            kwargs.get("noise_thresholding", True),
            kwargs.get("strength_multiplier", "auto"),
            kwargs.get("detection_mode", "auto_fast"),
            kwargs.get("save_trigger", False),
            kwargs.get("strength_model", 1.0),
            kwargs.get("strength_clip", 1.0),
            kwargs.get("device", "auto"),
        )
        key_str = str(cache_key).encode("utf-8")
        return hashlib.md5(key_str).hexdigest()

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "forensics", "lora_path")
    FUNCTION = "extract"
    CATEGORY = "EasyLoRAMerger/extraction"

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_checkpoint_path(name: str) -> Optional[Path]:
        """Resolve a checkpoint filename to its full path."""
        if not name or name == "None":
            return None
        full = resolve_model_path(name)
        if full:
            return Path(full)
        # Fallback: try as absolute / relative path
        p = Path(name)
        if p.exists():
            return p
        return None

    @staticmethod
    def _match_checkpoint_keys(
        base_sd: Dict[str, torch.Tensor],
        tuned_sd: Dict[str, torch.Tensor],
        base_arch: str,
        tuned_arch: str,
    ) -> Tuple[
        List[Tuple[str, torch.Tensor, torch.Tensor]],
        int,
        int,
        int,
    ]:
        """
        Match keys between base and tuned checkpoints.

        Uses canonical key normalization (via :mod:`checkpoint_normalizer`) to
        bring both state dicts into a shared key-space, then intersects on
        matching shapes.

        Returns:
            ``(matched_pairs, matched_count, skipped_shape, skipped_non_2d)``

            where ``matched_pairs`` is a list of
            ``(lora_key, base_tensor, tuned_tensor)`` tuples.
        """
        matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        seen_lora_keys: Set[str] = set()

        # Build normalized → original key maps for both SDs
        def _normalize_dict(
            sd: Dict[str, torch.Tensor],
            arch: str,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
            norm_sd = {}
            norm_to_orig = {}
            for orig_key, tensor in sd.items():
                norm_key = normalize_checkpoint_key(orig_key, arch)
                if norm_key not in norm_sd:
                    norm_sd[norm_key] = tensor
                    norm_to_orig[norm_key] = orig_key
            return norm_sd, norm_to_orig

        base_norm, base_map = _normalize_dict(base_sd, base_arch)
        tuned_norm, tuned_map = _normalize_dict(tuned_sd, tuned_arch)

        # Find common normalized keys
        common_keys = set(base_norm.keys()) & set(tuned_norm.keys())

        matched_count = 0
        skipped_shape = 0
        skipped_non_2d = 0

        for norm_key in sorted(common_keys):
            base_t = base_norm[norm_key]
            tuned_t = tuned_norm[norm_key]

            # Shape check
            if base_t.shape != tuned_t.shape:
                skipped_shape += 1
                continue

            # Dimensionality check: accept 2D (linear) and 4D (Conv2d) tensors.
            # 4D Conv2d weights are reshaped to 2D [C_out, C_in*kH*kW] before SVD.
            # 1D/3D tensors (biases, norm weights) are skipped.
            if base_t.dim() not in (2, 4):
                orig_key = base_map.get(norm_key, norm_key)
                print(f"   ⏭️  Skipping non-2D/4D key: {orig_key}  (dim={base_t.dim()}, shape={list(base_t.shape)})")
                skipped_non_2d += 1
                continue

            # Convert to LoRA key format (uses shared key_mapper)
            lora_key = checkpoint_key_to_lora_key(norm_key, base_arch)

            # Deduplication guard: skip if this lora_key already processed
            if lora_key in seen_lora_keys:
                continue
            seen_lora_keys.add(lora_key)

            matched_pairs.append((lora_key, base_t, tuned_t))
            matched_count += 1

        return matched_pairs, matched_count, skipped_shape, skipped_non_2d

    @staticmethod
    def _report_baked_metadata(
        tuned_meta: Dict[str, Any],
        strength_multiplier: Union[str, float],
    ) -> None:
        """Log baked LoRA metadata detected in the tuned checkpoint."""
        is_baking_method = 'baking_method' in tuned_meta
        method_or_tool = tuned_meta.get('baking_method') or tuned_meta.get('baking_tool', 'unknown')
        label = "Baked LoRA detected" if is_baking_method else "Checkpoint appears to be baked"
        tool_line = (f"      Method:      {tuned_meta['baking_method']}"
                     if is_baking_method
                     else f"      Tool:        {method_or_tool}")

        baking_source = tuned_meta.get('baked_lora_source', 'unknown')
        baking_strength = tuned_meta.get('baking_strength', 'unknown')
        weight_unet = tuned_meta.get('weight_unet', 'unknown')
        weight_te = tuned_meta.get('weight_te', 'unknown')
        weight_clip = tuned_meta.get('weight_clip', 'unknown')
        weight_vae = tuned_meta.get('weight_vae', 'unknown')

        print(f"\n   🔍 {label} in tuned checkpoint!")
        print(f"      {tool_line}")
        if is_baking_method:
            print(f"      Source:      {baking_source}")
            print(f"      Strength:    {baking_strength}")
        print(f"      Weights:     UNet={weight_unet}, TE={weight_te}, CLIP={weight_clip}, VAE={weight_vae}")
        if weight_te == '0.0' or weight_te == 0.0 or weight_te == '0':
            print(f"      ⚠️  weight_te=0.0 — TE was NOT baked! Character may depend on TE.")
        if weight_unet == '0.0' or weight_unet == 0.0 or weight_unet == '0':
            print(f"      ⚠️  weight_unet=0.0 — UNet was NOT baked!")
        if isinstance(strength_multiplier, str) and strength_multiplier == "auto":
            print(f"      ℹ️  Auto strength detection will compensate for baking attenuation.")
        else:
            print(f"      ⚠️  Baked LoRAs have attenuated deltas — consider increasing")
            print(f"         'strength_multiplier' (e.g., 1.5–3.0) to compensate for baking attenuation.")

    @staticmethod
    def _estimate_attenuation(
        matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
        sample_size: Optional[int] = 30,
    ) -> Tuple[float, float, int, float, float]:
        """
        Estimate baking attenuation factor and compute compensating multiplier.

        Samples ``sample_size`` pairs (or all if ``None``), computes the
        ``||delta||_F / ||W_base||_F`` ratio for each non-zero delta, then
        compares the median against a **dynamic** reference magnitude computed
        from the top decile of observed deltas.

        This avoids over-boosting naturally small LoRAs (low rank, weak effect)
        while still compensating for genuine baking attenuation.

        Args:
            matched_pairs: List of ``(lora_key, base_tensor, tuned_tensor)``.
            sample_size:   Number of layers to sample. ``None`` = all layers.

        Returns:
            ``(auto_multiplier, attenuation_factor, layers_analyzed, target_ratio, median_ratio)``
            where:

            * ``auto_multiplier`` — Legacy global multiplier (1.0 if per-layer
              mode active, >1.0 fallback for manual/global mode).
            * ``attenuation_factor`` — Ratio of detected-to-expected magnitude
              (1.0 = no attenuation, 0.5 = signal at 50% of expected).
            * ``layers_analyzed`` — How many non-zero deltas were analyzed.
            * ``target_ratio`` — Target ``||delta||_F / ||W_base||_F`` ratio for
              per-layer adaptive boosting. Each layer's delta is boosted
              proportionally to reach this target. Equal to the dynamic reference
              when attenuation is detected, or the median when healthy.
            * ``median_ratio`` — Median delta/base norm ratio across sampled
              layers; used by ``_compute_boost_targets`` for significance checks.
        """
        # ── Constants ────────────────────────────────────────────────────
        # Fallback reference if we can't compute dynamic (too few layers).
        FALLBACK_REFERENCE_RATIO = 0.015  # 1.5%
        ATTENUATION_THRESHOLD = 0.7  # Only correct when signal < 70% of expected
        MAX_AUTO_MULTIPLIER = 5.0  # Safety clamp for legacy global multiplier
        # Minimum number of layers needed to compute dynamic reference.
        MIN_LAYERS_FOR_DYNAMIC = 5

        # Determine effective sample
        total_available = len(matched_pairs)
        if sample_size is None:
            sample_size = total_available
        else:
            sample_size = min(sample_size, total_available)

        pairs_to_analyze = matched_pairs[:sample_size]
        ratios: List[float] = []

        mode_tag = "all" if sample_size == total_available else f"first {sample_size}"
        print(f"\n   🔍 Auto strength multiplier analysis [{mode_tag} layers]...")

        # ── GPU acceleration for attenuation estimation ─────────────────
        # Attenuation compares norms of small deltas — GPU moves are cheap
        # and norms are much faster on GPU (~10-50×). Only accelerate if
        # sufficient VRAM available (512 MB minimum threshold).
        atten_target = DeviceManager.get_device("auto")
        atten_on_gpu = (atten_target.type != "cpu")
        if atten_on_gpu:
            free_vram = DeviceManager.get_free_vram(atten_target)
            if free_vram is not None and free_vram < 512 * (1024**2):  # < 512 MB
                atten_on_gpu = False

        for lora_key, base_t, tuned_t in pairs_to_analyze:
            # Optional GPU move for faster float conversion and norm
            if atten_on_gpu:
                base_t = base_t.to(atten_target, non_blocking=True)
                tuned_t = tuned_t.to(atten_target, non_blocking=True)

            delta = tuned_t.float() - base_t.float()

            # Handle 4D Conv2d: flatten to 2D for norm computation
            if delta.dim() == 4:
                delta = delta.view(delta.shape[0], -1)
            elif delta.dim() != 2:
                continue

            # Skip zero deltas
            if torch.abs(delta).max().item() == 0.0:
                continue

            base_norm = torch.norm(base_t.float()).item()
            if base_norm < 1e-8:
                continue

            delta_norm = torch.norm(delta).item()
            ratio = delta_norm / base_norm
            ratios.append(ratio)

        layers_analyzed = len(ratios)

        if layers_analyzed == 0:
            print(f"      ⚠️  All sampled deltas are zero — cannot estimate attenuation. "
                  f"No multiplier applied.")
            return 1.0, 1.0, 0, 0.0, 0.0

        # ── Compute dynamic reference magnitude from top decile ──────────
        # For a clean LoRA, the strong layers define the "true" delta magnitude.
        # Using the top decile avoids dilution by weak/noisy layers.
        ratios_sorted = sorted(ratios)
        if len(ratios_sorted) >= MIN_LAYERS_FOR_DYNAMIC:
            top_decile_idx = max(int(len(ratios_sorted) * 0.9), 1)
            top_decile = ratios_sorted[top_decile_idx:]
            dynamic_reference = statistics.mean(top_decile)
            # Ensure a minimum floor to avoid degenerate cases
            dynamic_reference = max(dynamic_reference, 0.001)  # at least 0.1%
        else:
            # Too few layers — fall back to hardcoded reference
            dynamic_reference = FALLBACK_REFERENCE_RATIO
            print(f"      ⚠️  Too few layers ({layers_analyzed}) for dynamic reference — "
                  f"using fallback {FALLBACK_REFERENCE_RATIO * 100:.1f}%")

        # Use median for robustness against outliers
        ratios_tensor = torch.tensor(ratios)
        median_ratio = ratios_tensor.median().item()

        # Compute attenuation factor vs DYNAMIC reference
        attenuation_factor = median_ratio / dynamic_reference

        print(f"      Analyzed {layers_analyzed} non-zero deltas out of {sample_size} sampled")
        print(f"      Median relative delta magnitude: {median_ratio * 100:.2f}%")
        print(f"      Dynamic reference (top decile):  {dynamic_reference * 100:.2f}%")

        if attenuation_factor >= ATTENUATION_THRESHOLD:
            print(f"      ✅ Deltas appear healthy (ratio {attenuation_factor:.2f}x) — "
                  f"no multiplier needed")
            # Target = median (no boost needed); per-layer mode will leave most alone.
            return 1.0, attenuation_factor, layers_analyzed, median_ratio, median_ratio

        # Attenuation detected — compute compensating multiplier
        auto_multiplier = 1.0 / max(attenuation_factor, 0.01)  # avoid div-by-zero
        if auto_multiplier > MAX_AUTO_MULTIPLIER:
            print(f"      ⚠️  Extreme attenuation detected — multiplier clamped to "
                  f"{MAX_AUTO_MULTIPLIER}x")
            auto_multiplier = MAX_AUTO_MULTIPLIER
        elif auto_multiplier < 1.0:
            auto_multiplier = 1.0

        # Target ratio for per-layer boosting = the dynamic reference
        # (i.e., the magnitude the strongest layers naturally have)
        target_ratio = dynamic_reference

        print(f"      Detected attenuation factor: {attenuation_factor:.2f}x "
              f"(signal at {attenuation_factor * 100:.0f}% of expected)")
        print(f"      Per-layer target ratio: {target_ratio * 100:.2f}%")

        return auto_multiplier, attenuation_factor, layers_analyzed, target_ratio, median_ratio

    # ── Delta type classification ─────────────────────────────────────────

    @staticmethod
    def _classify_delta_type(
        matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
        strength_multiplier: Union[str, float],
        sample_size: Optional[int] = 30,
        cross_model_threshold: float = 0.10,
    ) -> str:
        """
        Classify the delta type as ``"cross_model"``, ``"baked_lora"``, or ``"normal_lora"``.

        This is a **single gatekeeper decision** made before any SVD or noise
        filtering, replacing the previously entangled logic where cross-model
        detection was a side-effect of attenuation estimation.

        Logic:
        1. If ``strength_multiplier`` is a numeric string (not ``"auto"``),
           classification is bypassed — returns the user's explicit choice.
        2. Samples ``sample_size`` matched pairs, computes delta/base norm ratios.
        3. If median ratio > ``cross_model_threshold`` (10%) → ``"cross_model"``.
        4. Else if attenuation is detected (median < 70% of top-decile reference)
           → ``"baked_lora"``.
        5. Else → ``"normal_lora"``.

        Returns:
            One of ``"cross_model"``, ``"baked_lora"``, ``"normal_lora"``.
        """
        # If user provided an explicit numeric strength, skip auto-classification
        if isinstance(strength_multiplier, str) and strength_multiplier != "auto":
            return "manual"

        # Sample pairs and compute delta/base norm ratios
        sample = matched_pairs[:sample_size] if sample_size is not None else matched_pairs[:]
        ratios: List[float] = []

        for lora_key, base_t, tuned_t in sample:
            delta = tuned_t.float() - base_t.float()
            if delta.dim() == 4:
                delta = delta.view(delta.shape[0], -1)
            elif delta.dim() != 2:
                continue
            if torch.abs(delta).max().item() == 0.0:
                continue
            base_norm = torch.norm(base_t.float()).item()
            if base_norm < 1e-8:
                continue
            ratios.append(torch.norm(delta).item() / base_norm)

        if not ratios:
            return "normal_lora"  # can't classify — assume normal

        median_ratio = torch.tensor(ratios).median().item()

        # ── Cross-model check ─────────────────────────────────────────────
        if median_ratio > cross_model_threshold:
            print(f"   🧠 Delta type: CROSS-MODEL (median ratio={median_ratio * 100:.2f}%)")
            return "cross_model"

        # ── Attenuation check (baked LoRA vs normal) ──────────────────────
        # If median ratio is below 70% of the top-decile reference, signal
        # is attenuated — typical of baked LoRAs.
        if len(ratios) >= 5:
            ratios_sorted = sorted(ratios)
            top_decile_idx = max(int(len(ratios_sorted) * 0.9), 1)
            top_decile = ratios_sorted[top_decile_idx:]
            dynamic_reference = max(statistics.mean(top_decile), 0.001)
            if median_ratio / dynamic_reference < 0.7:
                print(f"   🔍 Delta type: BAKED LoRA (median ratio={median_ratio * 100:.2f}%, "
                      f"ref={dynamic_reference * 100:.2f}%)")
                return "baked_lora"

        print(f"   ✅ Delta type: NORMAL LoRA (median ratio={median_ratio * 100:.2f}%)")
        return "normal_lora"

    # ── Compute boost targets ──────────────────────────────────────────────

    @staticmethod
    def _compute_boost_targets(
        matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
        delta_type: str,
        detection_mode: str = "auto_fast",
    ) -> Tuple[float, Dict[str, float], float, float, int, int]:
        """
        Compute per-layer boost targets for baked LoRA deltas.

        Delegates to :meth:`_estimate_attenuation` for the actual ratio analysis,
        then translates the result into per-layer target ratios (for adaptive
        boosting) or a global multiplier (legacy fallback).

        Returns:
            ``(per_layer_target_ratio, per_layer_targets, effective_multiplier,
              attenuation_factor, layers_analyzed, skipped_insignificant)``

            where ``per_layer_target_ratio`` is 0.0 when per-layer boosting is
            disabled, and ``per_layer_targets`` maps each lora key to its target
            delta/base norm ratio (empty dict when disabled).
        """
        SIGNIFICANCE_THRESHOLD = 0.0001  # 0.01% of base norm

        if delta_type != "baked_lora":
            return 0.0, {}, 1.0, 1.0, 0, 0

        sample_size = None if detection_mode == "auto_precise" else 30
        auto_mult, atten, analyzed, target_ratio, median_ratio = (
            EasyLoRAExtractor._estimate_attenuation(
                matched_pairs, sample_size=sample_size
            )
        )

        if target_ratio > 0.0 and auto_mult > 1.0:
            # Pre-compute per-layer targets
            per_layer_targets: Dict[str, float] = {}
            for lora_key, base_t, tuned_t in matched_pairs:
                delta = tuned_t.float() - base_t.float()
                if delta.dim() == 4:
                    delta = delta.view(delta.shape[0], -1)
                elif delta.dim() != 2:
                    continue
                if torch.abs(delta).max().item() == 0.0:
                    continue
                base_norm = torch.norm(base_t.float()).item()
                if base_norm < 1e-8:
                    continue
                ratio = torch.norm(delta).item() / base_norm
                if ratio >= SIGNIFICANCE_THRESHOLD and ratio < target_ratio:
                    per_layer_targets[lora_key] = target_ratio

            print(f"   🔧 Per-layer adaptive boosting enabled "
                  f"(target ratio={target_ratio * 100:.2f}%, "
                  f"{len(per_layer_targets)} layers targeted)")
            return target_ratio, per_layer_targets, 1.0, atten, analyzed, 0

        if auto_mult != 1.0:
            print(f"   🔧 Using global strength_multiplier={auto_mult:.1f}x")
            return 0.0, {}, auto_mult, atten, analyzed, 0

        return 0.0, {}, 1.0, atten, analyzed, 0

    # ── Noise filtering decision ──────────────────────────────────────────

    @staticmethod
    def _should_apply_noise_filtering(
        delta_type: str,
        user_noise_thresholding: bool,
    ) -> bool:
        """
        Decide whether noise thresholding should be applied based on delta type.

        Cross-model deltas always disable noise filtering (the MP model
        incorrectly classifies genuine model differences as noise).
        NORMAL (Flux fine-tune) deltas also disable it — the MP noise floor
        estimate is too aggressive for two close fine-tunes and strips real signal.
        For baked LoRAs, respects the user's ``noise_thresholding`` choice.
        """
        if delta_type in ("cross_model", "normal"):
            if user_noise_thresholding:
                print(f"   🧠 {delta_type} delta — noise thresholding auto-disabled")
            return False
        return user_noise_thresholding

    # ── Auto rank detection ────────────────────────────────────────────────

    def _estimate_optimal_rank_from_spectrum(
        self,
        matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
        max_rank: int,
        sample_size: int = 5,
        energy_target: float = 0.99,
    ) -> int:
        """
        Estimate optimal rank by analyzing SVD spectrum of sampled layers.

        Samples ``sample_size`` layers, computes SVD on each, and uses the
        combined energy-based + gap-based criteria to find the optimal rank.
        Results are aggregated via 90th percentile.

        Args:
            matched_pairs: List of ``(lora_key, base_tensor, tuned_tensor)``.
            max_rank: Maximum rank to consider (upper bound).
            sample_size: Number of layers to sample for analysis.
            energy_target: Energy retention threshold for energy-based method.

        Returns:
            Estimated optimal rank for this LoRA extraction.
        """
        total_available = len(matched_pairs)
        if total_available == 0:
            return min(64, max_rank)

        sample_sz = min(sample_size, total_available)
        pairs_to_analyze = matched_pairs[:sample_sz]

        print(f"\n   📊 Auto rank estimation (analyzing {sample_sz} sampled layers)...")
        per_layer_ranks: List[int] = []

        for lora_key, base_t, tuned_t in pairs_to_analyze:
            delta = tuned_t.float() - base_t.float()

            if delta.dim() == 4:
                delta = delta.view(delta.shape[0], -1)
            elif delta.dim() != 2:
                continue

            if torch.abs(delta).max().item() == 0.0:
                continue

            M, N = delta.shape
            k = min(max_rank, M, N)
            if k <= 1:
                per_layer_ranks.append(k)
                continue

            # Randomized SVD for speed
            try:
                U, S, V = torch.svd_lowrank(delta.float(), q=k, niter=4)
            except RuntimeError:
                continue

            # 1. Energy-based: find rank that captures energy_target%
            total_energy = torch.sum(S ** 2)
            if total_energy > 0:
                cumulative = torch.cumsum(S ** 2, dim=0)
                k_energy = torch.searchsorted(cumulative, energy_target * total_energy).item() + 1
            else:
                k_energy = 1
            k_energy = min(k_energy, k)
            k_energy = max(k_energy, 1)

            # 2. Gap-based: find signal/noise boundary
            k_gap = _find_signal_boundary(S)
            k_gap = min(k_gap, k)

            # 3. Pick the more conservative (higher rank = more signal)
            optimal = max(k_energy, k_gap)
            optimal = min(optimal, k)

            per_layer_ranks.append(optimal)

            short_key = lora_key[:48]
            print(f"      Layer {short_key:>48s}: optimal rank={optimal} "
                  f"(energy_k={k_energy}, gap_k={k_gap})")

        if not per_layer_ranks:
            print(f"      ⚠️  Could not analyze any layer — using fallback rank=64")
            return min(64, max_rank)

        # Aggregate results
        aggregated = _aggregate_optimal_rank(per_layer_ranks, strategy="p90")
        aggregated = min(aggregated, max_rank)
        aggregated = max(aggregated, 1)

        print(f"      Aggregated (p90): optimal rank = {aggregated}")
        print(f"   ✅ Using rank={aggregated} (capped by user max={max_rank})")
        return aggregated

    def _resolve_rank(
        self,
        rank_mode: str,
        user_rank: int,
        matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
        base_arch: str,
    ) -> int:
        """
        Resolve the effective rank based on ``rank_mode``.

        - ``"auto"``: run SVD spectrum analysis, capped by architecture suggestion.
        - ``"manual"``: return user rank directly.
        """
        if rank_mode == "manual":
            return user_rank

        # Auto mode: suggest max rank from architecture, cap by user rank
        arch_max = _ARCH_RANK_SUGGESTIONS.get(base_arch, 64)
        max_rank = min(user_rank, arch_max)
        optimal = self._estimate_optimal_rank_from_spectrum(
            matched_pairs, max_rank=max_rank
        )
        # Hint when every sampled layer hit the ceiling — true optimal may be higher
        if optimal >= max_rank:
            print(f"   ⚠️  Auto rank hit ceiling at {max_rank} — true optimal may be higher. "
                  f"Consider manual rank >{max_rank} for this architecture.")
        return optimal

    # ── Smart alpha computation ────────────────────────────────────────────

    @staticmethod
    def _compute_layer_alpha(
        effective_rank: int,
        svd_mode: str,
        user_alpha: float,
        component: str,
    ) -> float:
        """
        Compute alpha for a layer based on component type and SVD mode.

        Called when ``alpha_mode='auto'`` from ``_extract_layers()``.

        Per-component scaling:
            - UNet / VAE:  alpha = effective_rank (1.0×)
            - TE / CLIP:   alpha = effective_rank × 0.5 (lower to prevent over-amplification)

        In ``manual`` svd_mode, falls back to ``user_alpha`` (no scaling).
        """
        if svd_mode == "manual":
            return user_alpha

        base_alpha = max(1.0, float(effective_rank))

        component_scale = {
            "unet": 1.0,
            "te": 0.5,
            "clip": 0.5,
            "vae": 1.0,
            "other": 1.0,
        }.get(component, 1.0)

        return base_alpha * component_scale

    # ── Layer extraction (SVD loop) ───────────────────────────────────────

    def _extract_layers(
        self,
        matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
        rank: int,
        alpha: float,
        alpha_mode: str,
        svd_mode: str,
        energy_threshold: float,
        noise_thresholding: bool,
        per_layer_target_ratio: float,
        per_layer_targets: Dict[str, float],
        effective_multiplier: float,
        device: str,
        base_arch: str = "",
        lora_format: str = "native",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]], float,
               int, int, int, int, float]:
        """
        Compute deltas, apply per-layer boosting (if active), SVD decompose,
        and assemble the LoRA state dict.

        When ``alpha_mode='auto'``, uses ``_compute_layer_alpha()`` for
        per-component scaling.  When ``'manual'``, uses the user-provided
        ``alpha`` value directly.

        Returns:
            ``(lora_sd, component_breakdown, total_extraction_energy,
              extracted_layers, actual_zero_delta, per_layer_boosted,
              skipped_insignificant, effective_rank_avg)``
        """
        MAX_LAYER_MULTIPLIER = 20.0
        SIGNIFICANCE_THRESHOLD = 0.0001

        lora_sd: Dict[str, torch.Tensor] = {}
        component_breakdown: Dict[str, Dict[str, Any]] = {
            comp: {"count": 0, "energy": 0.0, "avg_rank": 0.0}
            for comp in _COMPONENT_ORDER
        }
        total_extraction_energy = 0.0
        rank_sum = 0
        extracted_layers = 0
        actual_zero_delta = 0
        per_layer_boosted: int = 0
        skipped_insignificant: int = 0

        total_pairs = len(matched_pairs)
        with ProgressTracker(total=total_pairs, desc="[Extractor] Processing layers") as progress:

            for lora_key, base_t, tuned_t in matched_pairs:
                # ── Delta in float32 ─────────────────────────────────────────
                delta = tuned_t.float() - base_t.float()

                # Zero-delta check
                if torch.abs(delta).max().item() == 0.0:
                    actual_zero_delta += 1
                    progress += 1
                    continue

                # ── Significance threshold ───────────────────────────────────
                delta_norm = torch.norm(delta).item()
                base_norm = torch.norm(base_t.float()).item()
                ratio = delta_norm / max(base_norm, 1e-12)

                if per_layer_target_ratio > 0.0 and ratio < SIGNIFICANCE_THRESHOLD:
                    skipped_insignificant += 1
                    progress += 1
                    continue

                # ── DIAGNOSTIC: log first 5 deltas ──────────────────────────
                if extracted_layers < 5:
                    print(f"   [DIAG] #{extracted_layers + 1} {lora_key[:48]:>48s} | "
                          f"delta_norm={delta_norm:.6e} "
                          f"base_norm={base_norm:.6e} "
                          f"ratio={ratio*100:.4f}% "
                          f"dtype={delta.dtype} "
                          f"shape={list(delta.shape)}")

                # ── Per-layer adaptive multiplier ────────────────────────────
                if lora_key in per_layer_targets and ratio < per_layer_target_ratio:
                    layer_mult = min(per_layer_target_ratio / max(ratio, 1e-12), MAX_LAYER_MULTIPLIER)
                    if layer_mult > 1.0:
                        delta = delta * layer_mult
                        per_layer_boosted += 1
                elif effective_multiplier != 1.0 and not per_layer_targets:
                    # Legacy global multiplier (only when per-layer is disabled)
                    delta = delta * effective_multiplier

                # ── Resolve output key names (native vs diffusers format) ──────
                if lora_format == "native":
                    # ── Native format ──────────────────────────────────────
                    # Block keys (double_blocks.*, single_blocks.*):
                    #   Use checkpoint-style naming with diffusion_model. prefix
                    #   (e.g. diffusion_model.double_blocks.0.img_attn.qkv).
                    #   This matches CivitAI reference LoRA format — merged QKV,
                    #   no block_map renaming, no qkv_splits.
                    #
                    # Non-block keys (img_in, txt_in, time_in, final_layer.*, etc.):
                    #   Use checkpoint_key_to_diffusers_key() to apply the basic_map
                    #   (e.g. img_in → transformer.x_embedder), keeping the
                    #   transformer. prefix so ComfyUI can map them correctly.
                    native_key = lora_key
                    if (native_key.startswith("transformer.double_blocks.") or
                        native_key.startswith("transformer.single_blocks.")):
                        # Flux block keys: diffusion_model. prefix, keep merged QKV
                        native_key = "diffusion_model." + native_key[len("transformer."):]
                        diff_keys = [(native_key, None)]
                        has_qkv_split = False
                    else:
                        # Non-Flux keys: check if diffusers conversion would QKV-split
                        # (e.g. Lumina2/Z-Image layers.*.attention.qkv → to_q/to_k/to_v)
                        diff_keys = checkpoint_key_to_diffusers_key(lora_key, base_arch)
                        has_qkv_split = any(
                            offset_info is not None for _, offset_info in diff_keys
                        )
                        if not has_qkv_split:
                            # Non-block, non-split key: use basic_map rename
                            native_key = diff_keys[0][0] if diff_keys else lora_key
                            diff_keys = [(native_key, None)]
                else:
                    # Diffusers: apply block_map, qkv_splits, and simple_map
                    # (e.g. transformer.transformer_blocks.0.attn.to_q).
                    diff_keys = checkpoint_key_to_diffusers_key(lora_key, base_arch)
                    has_qkv_split = any(
                        offset_info is not None for _, offset_info in diff_keys
                    )

                if has_qkv_split:
                    # ── SVD on the FULL merged tensor (QKV / linear1) ─────────
                    merge_label = diff_keys[0][0][:48] if diff_keys else lora_key
                    result = _decompose_delta_to_lora(
                        delta, f"{merge_label}_merged", rank, svd_mode,
                        energy_threshold=energy_threshold,
                        noise_thresholding=noise_thresholding,
                        device=device,
                    )
                    if result is None:
                        progress += 1
                        continue

                    lora_A_full, lora_B_full, effective_rank = result

                    # Compute hidden_size and total split width from diff_keys.
                    # Use delta.shape[1] (the input/hidden dimension) instead of
                    # delta.shape[0] // total_width, because some architectures
                    # (e.g. Flux 2 single_blocks.linear1) have extra rows beyond
                    # what the qkv_splits config accounts for (9×h vs 7×h).
                    # delta.shape[1] is always the true hidden_size.
                    total_width = sum(w for _, (_, w) in diff_keys)
                    hidden_size = delta.shape[1]
                    # Log when the merged tensor has extra rows beyond what the
                    # qkv_splits config accounts for (these rows are discarded).
                    expected_rows = hidden_size * total_width
                    if delta.shape[0] > expected_rows:
                        extra_rows = delta.shape[0] - expected_rows
                        print(f"  [INFO] {lora_key[:56]:>56s}: merged tensor has "
                              f"{delta.shape[0]} rows but qkv_splits account for "
                              f"{expected_rows} — discarding {extra_rows} trailing rows")
                    lora_A = lora_A_full   # [k, N] shared across slices
                    lora_B = lora_B_full   # [M, k] — will be sliced per key
                else:
                    # ── Simple 1:1 SVD on individual tensor ─────────────────
                    output_key = diff_keys[0][0] if diff_keys else lora_key
                    result = _decompose_delta_to_lora(
                        delta, output_key, rank, svd_mode,
                        energy_threshold=energy_threshold,
                        noise_thresholding=noise_thresholding,
                        device=device,
                    )
                    if result is None:
                        progress += 1
                        continue

                    lora_A, lora_B, effective_rank = result

                # ── Alpha key (with per-component scaling when auto) ─────────
                if alpha_mode == "auto":
                    comp = categorize_checkpoint_key(lora_key)
                    layer_alpha = self._compute_layer_alpha(
                        effective_rank, svd_mode, alpha, component=comp
                    )
                else:
                    layer_alpha = alpha

                # ── DIAGNOSTIC: SVD reconstruction quality check ────────────
                if extracted_layers < 3 and effective_rank > 0:
                    recon = (lora_B.float() @ lora_A.float()).to(delta.device)
                    recon_err = torch.norm(recon - delta.float()).item()
                    delta_norm_after = torch.norm(delta.float()).item()
                    if delta_norm_after > 1e-12:
                        rel_err = recon_err / delta_norm_after
                        print(f"   [DIAG] SVD recon err: {lora_key[:40]:>40s} | "
                              f"rel_err={rel_err:.6e} "
                              f"abs_err={recon_err:.6e} "
                              f"eff_rank={effective_rank} "
                              f"alpha_used={layer_alpha}")
                    else:
                        print(f"   [DIAG] SVD recon err: {lora_key[:40]:>40s} | "
                              f"delta_near_zero={delta_norm_after:.6e}")

                # ── Assemble LoRA state dict ─────────────────────────────────
                for diff_key, offset_info in diff_keys:
                    if offset_info is not None:
                        # Slice lora_B along output dimension for each split
                        offset_h, width_h = offset_info
                        start = offset_h * hidden_size
                        end = (offset_h + width_h) * hidden_size
                        B_slice = lora_B[start:end, :].contiguous()
                        # Clone lora_A to avoid safetensors shared-memory error
                        # when the same lora_A is referenced by multiple splits
                        # (q, k, v all share one SVD-decomposed lora_A).
                        A_slice = lora_A.clone()
                    else:
                        A_slice = lora_A
                        B_slice = lora_B

                    lora_sd[f"{diff_key}.lora_A.weight"] = A_slice
                    lora_sd[f"{diff_key}.lora_B.weight"] = B_slice
                    lora_sd[f"{diff_key}.alpha"] = torch.tensor(
                        [layer_alpha], dtype=A_slice.dtype
                    )

                    # ── Component tracking per diff_key ─────────────────────
                    comp = categorize_checkpoint_key(diff_key)
                    if comp not in component_breakdown:
                        component_breakdown[comp] = {"count": 0, "energy": 0.0, "avg_rank": 0.0}
                    component_breakdown[comp]["count"] += 1
                    # For QKV splits, energy is shared across slices
                    split_energy = torch.sum(delta ** 2).item() / len(diff_keys)
                    component_breakdown[comp]["energy"] += split_energy
                    component_breakdown[comp]["avg_rank"] = (
                        (component_breakdown[comp]["avg_rank"] * (component_breakdown[comp]["count"] - 1)
                         + effective_rank)
                        / component_breakdown[comp]["count"]
                    )

                total_extraction_energy += torch.sum(delta ** 2).item()
                rank_sum += effective_rank
                extracted_layers += 1

                progress += 1

        effective_rank_avg = rank_sum / max(extracted_layers, 1)

        return (lora_sd, component_breakdown, total_extraction_energy,
                extracted_layers, actual_zero_delta, per_layer_boosted,
                skipped_insignificant, effective_rank_avg)

    # ── Diagnostics helper (P5) ───────────────────────────────────────────

    @staticmethod
    def _print_extraction_diagnostics(
        extracted_layers: int,
        effective_rank_avg: float,
        actual_zero_delta: int,
        skipped_insignificant: int,
        per_layer_boosted: int,
        rank: int,
        base_arch: str,
        lora_sd: Dict[str, torch.Tensor],
    ) -> None:
        """Print post-extraction diagnostics (extracted from _finalize for P5)."""
        print(f"   ✅ Extracted {extracted_layers} LoRA layers "
              f"(avg effective rank={effective_rank_avg:.1f})")
        if actual_zero_delta > 0:
            print(f"   ℹ️  Skipped {actual_zero_delta} layers with zero delta")
        if skipped_insignificant > 0:
            print(f"   🔇 Skipped {skipped_insignificant} layers below significance threshold")
        if per_layer_boosted > 0:
            print(f"   📈 Per-layer boosted {per_layer_boosted} layers adaptively")

        # ⚠️  Low effective rank warning
        if extracted_layers > 0 and effective_rank_avg < max(rank * 0.1, 0.5):
            print(f"   ⚠️  Low effective rank ({effective_rank_avg:.1f}) relative to target rank ({rank}).")
            print(f"       Consider increasing 'strength_multiplier' (1.5-3.0) to amplify the signal")
            print(f"       before SVD, or lower 'svd_mode' from auto_energy to manual with a")
            print(f"       higher rank.")

        # 💡 Large-matrix rank suggestion
        if (extracted_layers > 0
                and effective_rank_avg >= rank * 0.9
                and rank < 128
                and base_arch in ("lumina2", "z_image")):
            print(f"   💡 Large-matrix architecture detected ({base_arch}). "
                  f"Effective rank ({effective_rank_avg:.0f}) capped by target "
                  f"rank ({rank}). Consider increasing rank to 128+ "
                  f"for better reconstruction quality.")

        # ── CLIP Key Debug Output ────────────────────────────────────────
        if lora_sd:
            clip_keys = [k for k in lora_sd if 'text_model' in k or 'text_encoders' in k]
            if clip_keys:
                clip_bases = sorted(set(
                    k.replace('.lora_A.weight', '').replace('.lora_B.weight', '').replace('.alpha', '')
                    for k in clip_keys
                ))
                print(f"\n   🔍 CLIP LoRA keys generated ({len(clip_bases)} unique bases):")
                for base in clip_bases[:5]:
                    print(f"      - {base}")
                if len(clip_bases) > 5:
                    print(f"      - ... and {len(clip_bases) - 5} more")
                print(f"   ℹ️  If 0 CLIP patches attach, the key prefix may not match "
                      f"ComfyUI's expected format.")
            else:
                print(f"\n   ℹ️  No CLIP/text encoder keys in extracted LoRA")

        # ── Flux modulation key diagnostic ────────────────────────────────
        # Flux 2 modulation layers (double_stream_modulation_img.lin,
        # double_stream_modulation_txt.lin, single_stream_modulation.lin)
        # are extracted into the LoRA but ComfyUI's flux_to_diffusers()
        # MAP_BASIC may not have entries for them, resulting in 0 patches.
        if lora_sd and base_arch in ("flux",):
            mod_keys = [k for k in lora_sd
                        if 'double_stream_modulation' in k or 'single_stream_modulation' in k]
            if mod_keys:
                mod_bases = sorted(set(
                    k.replace('.lora_A.weight', '').replace('.lora_B.weight', '').replace('.alpha', '')
                    for k in mod_keys
                ))
                print(f"\n   🔍 Flux modulation LoRA keys generated ({len(mod_bases)} unique bases):")
                for base in mod_bases:
                    print(f"      - {base}")
                print(f"   ℹ️  Modulation keys may show 0 patches attached with current ComfyUI;\n"
                      f"       this is expected until ComfyUI adds Flux 2 modulation support.")

    # ── Finalize: save + preview + forensics ──────────────────────────────

    def _finalize(
        self,
        lora_sd: Dict[str, torch.Tensor],
        component_breakdown: Dict[str, Dict[str, Any]],
        extracted_layers: int,
        actual_zero_delta: int,
        per_layer_boosted: int,
        skipped_insignificant: int,
        effective_rank_avg: float,
        rank: int,
        alpha: float,
        svd_mode: str,
        base_name: str,
        tuned_name: str,
        base_arch: str,
        tuned_arch: str,
        matched_count: int,
        skipped_shape: int,
        skipped_non_2d: int,
        total_base_keys: int,
        total_tuned_keys: int,
        total_extraction_energy: float,
        strength_multiplier: Union[str, float],
        detection_mode: str,
        attenuation_factor: float,
        attenuation_layers_analyzed: int,
        component_matching_counts: Dict[str, int],
        base_meta: Optional[Dict[str, Any]],
        save_trigger: bool,
        filename: str,
        save_folder: str,
        model: Any,
        clip: Any,
        strength_model: float,
        strength_clip: float,
    ) -> Tuple[Any, Any, str, str]:
        """
        Save the extracted LoRA, apply preview, and build forensic report.

        Returns:
            ``(preview_model, preview_clip, forensic_report, output_path_str)``.
        """
        # ── Post-extraction diagnostics (delegated to helper for P5) ─────
        self._print_extraction_diagnostics(
            extracted_layers, effective_rank_avg,
            actual_zero_delta, skipped_insignificant, per_layer_boosted,
            rank, base_arch, lora_sd,
        )

        if extracted_layers == 0:
            return (
                None, None,
                "❌ ERROR: All matched keys produced zero delta or non-decomposable tensors. "
                "The two checkpoints appear identical.",
                "",
            )

        # ── Save to disk ─────────────────────────────────────────────────
        output_path_str = ""
        if save_trigger:
            try:
                if not filename or filename.strip() == "":
                    filename = "extracted_lora"
                if not filename.endswith(".safetensors"):
                    filename += ".safetensors"
                safe_filename = "".join(
                    c for c in filename if c.isalnum() or c in "._- "
                ).rstrip()

                if save_folder and isinstance(save_folder, str) and save_folder.strip():
                    user_path = save_folder.strip()
                    output_dir = Path(user_path)
                    if not output_dir.is_absolute():
                        lora_folders = folder_paths.get_folder_paths("loras")
                        base_dir = Path(lora_folders[0]) if lora_folders else Path.cwd()
                        output_dir = base_dir / user_path
                else:
                    lora_folders = folder_paths.get_folder_paths("loras")
                    output_dir = Path(lora_folders[0]) if lora_folders else Path.cwd()

                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / safe_filename
                counter = 1
                while output_path.exists():
                    stem = output_path.stem
                    parts = stem.split("_")
                    if len(parts) > 1 and parts[-1].isdigit():
                        stem = "_".join(parts[:-1])
                    output_path = output_dir / f"{stem}_{counter}{output_path.suffix}"
                    counter += 1

                metadata = finalize_metadata(
                    metadata=base_meta,
                    mode="preserve_a",
                    component="extractor",
                    extra_fields={
                        "base_checkpoint": base_name,
                        "tuned_checkpoint": tuned_name,
                        "base_architecture": base_arch,
                        "tuned_architecture": tuned_arch,
                        "svd_mode": svd_mode,
                        "target_rank": str(rank),
                        "effective_rank_avg": f"{effective_rank_avg:.2f}",
                        "alpha": str(alpha),
                        "extracted_layers": str(extracted_layers),
                        "matched_keys": str(matched_count),
                        "total_extraction_energy": f"{total_extraction_energy:.4f}",
                    }
                )
                if base_meta:
                    metadata["base_metadata"] = json.dumps(base_meta, indent=2)

                from safetensors.torch import save_file as safe_save
                safe_save(lora_sd, str(output_path), metadata)
                output_path_str = str(output_path)
                print(f"   💾 LoRA saved to: {output_path_str}")

            except Exception as e:
                print(f"   ❌ Failed to save LoRA: {e}")

        # ── Small-rank key diagnostic ────────────────────────────────────
        if lora_sd:
            base_keys = sorted(set(
                k.replace(".lora_A.weight", "").replace(".lora_B.weight", "").replace(".alpha", "")
                for k in lora_sd
            ))
            expected_patches = sum(
                1 for bk in base_keys
                if f"{bk}.lora_A.weight" in lora_sd and f"{bk}.lora_B.weight" in lora_sd
            )
            small_rank_keys = []
            for bk in sorted(base_keys):
                a_key = f"{bk}.lora_A.weight"
                if a_key in lora_sd:
                    r = lora_sd[a_key].shape[0]
                    if r <= 4:
                        a_shape = list(lora_sd[a_key].shape)
                        b_key = f"{bk}.lora_B.weight"
                        b_shape = list(lora_sd[b_key].shape) if b_key in lora_sd else "?"
                        small_rank_keys.append((bk, r, a_shape, b_shape))
            if small_rank_keys:
                print(f"\n   🔍 LoRA key diagnostic ({expected_patches} expected patches):")
                for bk, r, a_shape, b_shape in small_rank_keys:
                    print(f"      ⚠️  {bk}: rank={r}, "
                          f"lora_A={a_shape}, lora_B={b_shape}")
                print(f"      ℹ️  Rank-1/2 keys (like pad tokens) may fail to patch;\n"
                      f"         compare '{expected_patches} expected' with ComfyUI's\n"
                      f"         'N patches attached' message.")

        # ── Apply preview ────────────────────────────────────────────────
        preview_model = None
        preview_clip = None
        if model is not None and clip is not None and lora_sd:
            try:
                if save_trigger and output_path_str:
                    # Save mode: reload from saved file for pixel-identical
                    # preview (same pattern as baker_node.py lines 512-523).
                    reloaded_sd = comfy.utils.load_torch_file(
                        output_path_str, safe_load=True
                    )
                    # ── DEBUG: verify tensor identity after save→reload ──
                    _mismatches = 0
                    _max_diff = 0.0
                    _all_keys = set(lora_sd) | set(reloaded_sd)
                    for k in sorted(_all_keys):
                        in_orig = k in lora_sd
                        in_reload = k in reloaded_sd
                        if in_orig and in_reload:
                            t_orig = lora_sd[k]
                            t_reload = reloaded_sd[k]
                            if isinstance(t_orig, torch.Tensor) and isinstance(t_reload, torch.Tensor):
                                if not torch.equal(t_orig, t_reload):
                                    _mismatches += 1
                                    d = (t_orig - t_reload).abs().max().item()
                                    if d > _max_diff:
                                        _max_diff = d
                        elif in_orig and not in_reload:
                            print(f"   🔬 KEY MISSING in reloaded_sd: {k}")
                            _mismatches += 1
                        elif not in_orig and in_reload:
                            print(f"   🔬 KEY EXTRA in reloaded_sd: {k}")
                            _mismatches += 1
                    if _mismatches:
                        print(f"   🔬 TENSOR COMPARISON: {_mismatches} key(s) differ, "
                              f"max abs diff = {_max_diff:.2e}")
                    else:
                        print(f"   ✅ TENSOR COMPARISON: all {len(lora_sd)} tensors "
                              f"bit-identical after save→reload")
                    # ── End debug ──────────────────────────────────────────
                    preview_model, preview_clip = comfy.sd.load_lora_for_models(
                        model, clip, reloaded_sd, strength_model, strength_clip
                    )
                else:
                    # Preview-only mode: use in-memory dict directly.
                    preview_model, preview_clip = comfy.sd.load_lora_for_models(
                        model, clip, lora_sd, strength_model, strength_clip
                    )
                print(f"   🖼️ LoRA preview applied to model+clip "
                      f"(strength_model={strength_model}, strength_clip={strength_clip})")
            except Exception as e:
                print(f"   ⚠️ Failed to apply LoRA preview: {e}")
                preview_model = model
                preview_clip = clip

        # ── Build forensic report ────────────────────────────────────────
        forensics = _build_extraction_report(
            base_name=base_name,
            tuned_name=tuned_name,
            base_arch=base_arch,
            tuned_arch=tuned_arch,
            matched_keys=matched_count,
            skipped_shape=skipped_shape,
            skipped_non_2d=skipped_non_2d,
            skipped_zero_delta=actual_zero_delta,
            total_base_keys=total_base_keys,
            total_tuned_keys=total_tuned_keys,
            extracted_layers=extracted_layers,
            component_breakdown=component_breakdown,
            rank=rank,
            svd_mode=svd_mode,
            effective_rank_avg=effective_rank_avg,
            alpha=alpha,
            total_energy_extracted=total_extraction_energy,
            output_path=output_path_str or "(not saved)",
            strength_multiplier=strength_multiplier,
            detection_mode=detection_mode,
            attenuation_factor=attenuation_factor,
            attenuation_layers_analyzed=attenuation_layers_analyzed,
            skipped_insignificant=skipped_insignificant,
            component_matching_counts=component_matching_counts,
        )

        print(f"   ✅ Extraction complete")
        return (preview_model, preview_clip, forensics, output_path_str)

    # ── Public entry point (P5 dispatch) ──────────────────────────────────

    def extract(
        self,
        checkpoint_base: str,
        checkpoint_tuned: str,
        rank_mode: str = "auto",
        rank: int = 64,
        alpha_mode: str = "auto",
        alpha: float = 64.0,
        svd_mode: str = "auto_energy",
        energy_threshold: float = 0.95,
        noise_thresholding: bool = True,
        strength_multiplier: Union[str, float] = "auto",
        detection_mode: str = "auto_fast",
        device: str = "auto",
        save_trigger: bool = False,
        filename: str = "extracted_lora",
        lora_format: str = "native",
        save_folder: str = "",
        model=None,
        clip=None,
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
        node_id: str = "",
    ) -> Tuple[Any, Any, str, str]:
        """
        Execute the extraction — **dispatch-only** (P5 compliance).

        Each phase is delegated to a focused helper method. This method is a
        readable decision tree, NOT a 500-line procedural block.

        Args:
            checkpoint_base:     Filename or path to the base checkpoint.
            checkpoint_tuned:    Filename or path to the tuned checkpoint.
            rank_mode:           ``"auto"`` (SVD spectrum analysis) or ``"manual"``.
            rank:                Target rank (manual) or upper bound (auto).
            alpha_mode:          ``"auto"`` (per-component scaling) or ``"manual"``.
            alpha:               Alpha value (only used when alpha_mode='manual').
            svd_mode:            ``"auto_energy"``, ``"manual"``, or ``"full"``.
            energy_threshold:    Energy retention for auto_energy mode (0.50–0.999).
            noise_thresholding:  When True (default), apply noise floor filtering.
            strength_multiplier: ``"auto"`` or a numeric string like ``"1.0"``–``"3.0"``.
            detection_mode:      ``"auto_fast"`` (default), ``"auto_precise"``, or ``"manual"``.
            device:              ``"auto"`` (default), ``"cuda"``, or ``"cpu"``.
            save_trigger:        When ``True``, save the LoRA to disk.
            filename:            Output filename (without extension).
            save_folder:         Optional custom output directory.
            model:               Optional MODEL for preview.
            clip:                Optional CLIP for preview.
            strength_model:      LoRA strength for model preview (0.0–10.0).
            strength_clip:       LoRA strength for CLIP preview (0.0–10.0).
            node_id:             ComfyUI node ID (for cache isolation).

        Returns:
            ``(model, clip, forensic_report_string, lora_path_or_empty_string)``.
        """
        # ── 1. Resolve paths & load checkpoints ──────────────────────────
        base_path = self._resolve_checkpoint_path(checkpoint_base)
        tuned_path = self._resolve_checkpoint_path(checkpoint_tuned)
        if base_path is None:
            return (None, None,
                    "❌ ERROR: Base checkpoint not found.", "")
        if tuned_path is None:
            return (None, None,
                    "❌ ERROR: Tuned checkpoint not found.", "")

        base_name, tuned_name = base_path.name, tuned_path.name
        print(f"\n🛡️ Easy LoRA Extractor — Loading checkpoints...")
        print(f"   Base:  {base_name}")
        print(f"   Tuned: {tuned_name}")

        base_sd, base_meta = load_checkpoint_with_metadata(base_path)
        tuned_sd, tuned_meta = load_checkpoint_with_metadata(tuned_path)
        total_base_keys, total_tuned_keys = len(base_sd), len(tuned_sd)
        print(f"   Loaded {total_base_keys} base keys, {total_tuned_keys} tuned keys")

        # ── 2. Detect architectures ──────────────────────────────────────
        base_arch = detect_checkpoint_architecture(list(base_sd.keys()))
        tuned_arch = detect_checkpoint_architecture(list(tuned_sd.keys()))
        print(f"   Base arch:  {base_arch}")
        print(f"   Tuned arch: {tuned_arch}")
        if base_arch != tuned_arch:
            print(f"   ⚠️  Architectures differ! Matching may be limited.")

        # ── 3. Detect baked metadata ─────────────────────────────────────
        if tuned_meta and ('baking_method' in tuned_meta or 'baking_tool' in tuned_meta):
            self._report_baked_metadata(tuned_meta, strength_multiplier)

        # ── 4. Match keys ────────────────────────────────────────────────
        print(f"   Device: {DeviceManager.get_device(device)} (user choice: {device})")
        print(f"   Matching keys...")
        matched_pairs, matched_count, skipped_shape, skipped_non_2d = (
            self._match_checkpoint_keys(base_sd, tuned_sd, base_arch, tuned_arch)
        )
        print(f"   Matched {matched_count} keys, {skipped_shape} shape mismatches, "
              f"{skipped_non_2d} non-2D skipped")

        component_matching_counts: Dict[str, int] = {}
        for lora_key, _, _ in matched_pairs:
            comp = categorize_checkpoint_key(lora_key)
            component_matching_counts[comp] = component_matching_counts.get(comp, 0) + 1
        total_matched = sum(component_matching_counts.values())
        print(f"   Key matching by component:")
        for comp in _COMPONENT_ORDER:
            count = component_matching_counts.get(comp, 0)
            label = _COMPONENT_LABELS.get(comp, comp.upper())
            print(f"      {label}: {count} keys")
        print(f"   Total: {total_matched} keys matched")

        if matched_count == 0:
            return (None, None,
                    "❌ ERROR: No compatible keys found between base and tuned checkpoints.", "")

        # ── 5. Resolve rank (auto or manual) ─────────────────────────────
        effective_rank = self._resolve_rank(
            rank_mode, rank, matched_pairs, base_arch
        )
        print(f"   Rank: mode={rank_mode}, user_rank={rank}, effective_rank={effective_rank}")

        # ── 6. Classify delta type ──────────────────────────────────────
        # Classification uses a fixed representative sample (30 layers)
        # regardless of detection_mode.  This prevents false "baked LoRA"
        # detection when auto_precise samples all layers (which can inflate
        # the top-decile reference via outliers).
        #
        # The detection_mode only affects _compute_boost_targets below,
        # where analyzing all layers for precise attenuation estimation is
        # genuinely beneficial for quantifying "how much to boost" when
        # the delta IS actually baked.
        delta_type = self._classify_delta_type(
            matched_pairs, strength_multiplier, sample_size=30
        )

        # ── 7. Compute boost targets ─────────────────────────────────────
        per_layer_target_ratio, per_layer_targets, effective_multiplier, \
            attenuation_factor, attenuation_layers_analyzed, skipped_insignificant = (
            self._compute_boost_targets(matched_pairs, delta_type, detection_mode)
        )

        # ── 8. Determine noise filtering strategy ────────────────────────
        noise_thresholding = self._should_apply_noise_filtering(
            delta_type, noise_thresholding
        )

        # ── 9. Extract layers (SVD loop) ────────────────────────────────
        print(f"   Computing deltas and applying SVD (mode={svd_mode}, "
              f"rank={effective_rank}, alpha_mode={alpha_mode})...")
        lora_sd, component_breakdown, total_extraction_energy, \
            extracted_layers, actual_zero_delta, per_layer_boosted, \
            skipped_insig, effective_rank_avg = self._extract_layers(
            matched_pairs, effective_rank, alpha, alpha_mode, svd_mode,
            energy_threshold, noise_thresholding, per_layer_target_ratio,
            per_layer_targets, effective_multiplier, device,
            base_arch=base_arch,
            lora_format=lora_format,
        )
        # Merge skipped_insignificant from both passes
        skipped_insignificant += skipped_insig

        # ── 10. Finalize: save + preview + forensics ─────────────────────
        return self._finalize(
            lora_sd, component_breakdown,
            extracted_layers, actual_zero_delta, per_layer_boosted,
            skipped_insignificant, effective_rank_avg,
            effective_rank, alpha, svd_mode,
            base_name, tuned_name, base_arch, tuned_arch,
            matched_count, skipped_shape, skipped_non_2d,
            total_base_keys, total_tuned_keys, total_extraction_energy,
            strength_multiplier, detection_mode,
            attenuation_factor, attenuation_layers_analyzed,
            component_matching_counts, base_meta,
            save_trigger, filename, save_folder,
            model, clip, strength_model, strength_clip,
        )
