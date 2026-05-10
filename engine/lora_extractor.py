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

from .checkpoint_normalizer import (
    detect_checkpoint_architecture,
    normalize_checkpoint_key,
)
from ..config import DEVICE_OPTIONS
from ..utils import (
    categorize_checkpoint_key,
    DeviceManager,
    load_checkpoint_with_metadata,
    ProgressTracker,
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
# CHECKPOINT KEY → LORA KEY CONVERSION
# ============================================================================

# Known prefix mappings: checkpoint_prefix → lora_prefix
# Order matters — more specific prefixes must be checked first.
_CHECKPOINT_TO_LORA_PREFIXES: List[Tuple[str, str, str]] = [
    # ── Flux (un-normalized — full model.diffusion_model.transformer. prefix) ──
    ("model.diffusion_model.transformer.double_blocks.", "double_blocks.", "flux"),
    ("model.diffusion_model.transformer.single_blocks.", "single_blocks.", "flux"),
    # ── Flux (normalized — after normalizer strips model.diffusion_model.) ──
    # The normalizer strips model.diffusion_model. for Flux, so the key
    # arrives as transformer.double_blocks.X.Y — must strip transformer. too.
    ("transformer.double_blocks.", "double_blocks.", "flux"),
    ("transformer.single_blocks.", "single_blocks.", "flux"),
    # SDXL TE2: model.conditioner.embedders.0.transformer.text_model.xxx → text_model.xxx
    # SDXL / SD1.5 TE1: model.text_model.xxx → text_model.xxx
    ("model.text_model.", "text_model.", "te"),
    # ── SD1.5 CLIP: cond_stage_model.transformer.text_model.xxx ──
    #   → text_encoders.transformer.text_model.xxx
    # Standard SD1.5 checkpoints use cond_stage_model. prefix for the CLIP text encoder.
    # ComfyUI's model_lora_keys_clip() maps LoRA keys with a "text_encoders." prefix
    # (see comfy/lora.py line 101), so the LoRA key must use text_encoders.transformer.text_model.xxx
    # to match the key_map entries built from clip.cond_stage_model.state_dict().
    ("cond_stage_model.transformer.text_model.", "text_encoders.clip_l.transformer.text_model.", "te"),
    # ── SDXL TE1/TE2 conditioner ──────────────────────────────────────────
    # SDXL checkpoints store CLIP text encoder weights under
    # model.conditioner.embedders.N.transformer.text_model.xxx where N=0
    # (OpenCLIP-G, TE2) or N=1 (CLIP-L, TE1). The baker already handles this
    # via _try_te_conditioner_prefix() (handles both te1. and te2. prefixes)
    # but the extractor needs explicit entries to produce correct LoRA key names.
    ("model.conditioner.embedders.0.transformer.text_model.", "text_encoders.transformer.text_model.", "te"),
    ("model.conditioner.embedders.1.transformer.text_model.", "text_encoders.transformer.text_model.", "te"),
    # Catch-all for other conditioner/embedder keys (pooled embeddings, etc.)
    ("model.conditioner.", "conditioner.", "te"),
    # SDXL UNet: model.diffusion_model.xxx → diffusion_model.xxx
    ("model.diffusion_model.", "diffusion_model.", "sdxl_unet"),
    # Anima: net.blocks.xxx → diffusion_model.blocks.xxx
    ("net.blocks.", "diffusion_model.blocks.", "anima"),
    # SD1.5 UNet: diffusion_model.xxx → diffusion_model.xxx (already LoRA format)
    # Z-Image: diffusion_model.layers.xxx → diffusion_model.layers.xxx (already LoRA format)
    # VAE: first_stage_model.xxx → first_stage_model.xxx (pass-through, already LoRA format)
    #
    # The following prefixes are already in LoRA master format and pass through
    # as-is via the fallback at the end of _checkpoint_key_to_lora_key():
    #   clip_l.xxx / clip_g.xxx / t5.xxx / text_encoder.xxx
    #   text_encoders.xxx / double_blocks.xxx / single_blocks.xxx
    # These are used by Flux, SD3, and other architectures where the checkpoint
    # key format already matches the LoRA key format.
]


def _checkpoint_key_to_lora_key(ckpt_key: str) -> str:
    """
    Convert a normalized checkpoint key to the equivalent LoRA master-format key.

    After :func:`normalize_checkpoint_key`, the key reflects the canonical form of
    the *checkpoint* naming (e.g. ``model.diffusion_model.xxx`` for SDXL,
    ``double_blocks.xxx`` for Flux).  This function maps it back to the *LoRA*
    master format used by :func:`identity_normalize` and the rest of the merger
    suite.

    Args:
        ckpt_key: A checkpoint key normalized by :func:`normalize_checkpoint_key`.

    Returns:
        LoRA master-format key.  If no known prefix conversion applies, the key
        is returned unchanged.
    """
    # ── Strip tensor-type suffix ──────────────────────────────────────────
    # Checkpoint keys end in .weight/.bias/.alpha, but LoRA base keys must
    # NOT include these — they are tensor metadata, not part of the layer name.
    # ComfyUI's native LoRA loader and the baker's reverse key map both expect
    # clean base keys (e.g. diffusion_model.input_blocks.0.0, NOT
    # diffusion_model.input_blocks.0.0.weight).
    for suffix in ('.weight', '.bias', '.alpha'):
        if ckpt_key.endswith(suffix):
            ckpt_key = ckpt_key[:-len(suffix)]
            break

    for ckpt_prefix, lora_prefix, _arch in _CHECKPOINT_TO_LORA_PREFIXES:
        if ckpt_key.startswith(ckpt_prefix):
            return lora_prefix + ckpt_key[len(ckpt_prefix):]
    # Pass-through: already in LoRA format (Flux bare, Z-Image, SD1.5)
    return ckpt_key


# ============================================================================
# SVD DECOMPOSITION FOR DELTA → LORA
# ============================================================================


# ── Noise floor estimation constants ──────────────────────────────────────
# Multiplier above the estimated noise floor for Marchenko-Pastur thresholding.
# Singular values below noise_factor × noise_floor are treated as noise.
# The factor is now ADAPTIVE based on the number of singular values:
#   ≤8 SVs  → 1.5 (gentle — few SVs likely all signal)
#   ≤16 SVs → 2.0 (moderate)
#   >16 SVs → 3.0 (standard — clear noise tail expected)
# Minimum fraction of singular values to use for noise floor estimation.
_NOISE_TAIL_FRACTION = 0.25

# Gap threshold for elbow detection: if a drop between consecutive normalized
# SVs exceeds this, it marks the signal/noise boundary.
_GAP_THRESHOLD = 0.10


def _estimate_sv_noise_threshold(
    S: torch.Tensor,
    M: int,
    N: int,
    noise_factor: Optional[float] = None,
) -> float:
    """
    Estimate a noise floor threshold for singular values.

    Uses a two-stage approach:
    1. **Gap (elbow) detection:** Look for the largest drop between consecutive
       normalized singular values. If the gap exceeds ``_GAP_THRESHOLD`` (10%)
       and occurs in the first half of SVs, the SVs after the gap are treated
       as the noise tail.
    2. **Fallback (tail mean):** If no clear elbow is found, use the smallest
       ``_NOISE_TAIL_FRACTION`` of SVs as the noise population (same as before).

    The ``noise_factor`` is **adaptive** based on the number of SVs:
    - ≤8 SVs  → ``1.5``  (gentle — few SVs likely all signal)
    - ≤16 SVs → ``2.0``  (moderate — slight noise may be present)
    - >16 SVs → ``3.0``  (standard — clear noise tail expected)

    Args:
        S: Singular values tensor, sorted descending.
        M, N: Dimensions of the original matrix (M rows, N cols).
        noise_factor: Override for the adaptive noise factor. If ``None``,
                      the factor is chosen adaptively based on ``len(S)``.

    Returns:
        Threshold value — only singular values above this are treated as signal.
        Returns ``0.0`` (no filtering) if there are too few SVs to estimate.
    """
    q = len(S)
    if q < 4:
        return 0.0  # too few SVs to estimate noise reliably

    # ── Determine adaptive noise factor ─────────────────────────────────
    if noise_factor is None:
        if q <= 8:
            noise_factor = 1.5   # gentle — few SVs, likely all signal
        elif q <= 16:
            noise_factor = 2.0   # moderate
        else:
            noise_factor = 3.0   # standard — expected noise tail

    # ── Stage 1: Gap-based (elbow) detection ────────────────────────────
    # Find the largest drop between consecutive normalized SVs.
    # A sharp drop indicates the transition from signal to noise.
    S_norm = S / S[0]  # normalize to [0, 1]
    gaps = S_norm[:-1] - S_norm[1:]

    max_gap, max_idx = torch.max(gaps, dim=0)
    max_gap_val = max_gap.item()
    max_idx_val = max_idx.item()

    # Only use gap detection if:
    #   - The gap exceeds _GAP_THRESHOLD (significant elbow)
    #   - The gap occurs in the first half of SVs (signal→noise transition)
    #   - At least 1 SV remains after the gap (noise tail exists)
    #   - The first SV in the noise tail is <5% of S[0] (truly noise-like,
    #     not a gap between two groups of signal components)
    # The gap must NOT be at index 0 (between SV[0] and SV[1]).
    # For low-rank signals (q ≤ 8), the drop from the first to the second
    # singular value is always the largest by far — it is a natural
    # characteristic of any SVD decomposition, NOT a signal→noise transition.
    # Misidentifying this drop as a signal-noise boundary would discard
    # legitimate signal components (SV[1] through SV[q-1]) and collapse
    # the effective rank, destroying subtle character information.
    if max_gap_val > _GAP_THRESHOLD and max_idx_val >= 1 and max_idx_val < q // 2 and max_idx_val + 1 < q:
        # ── Safeguard: only accept gap if noise tail SVs are truly small ─
        # Prevents misidentifying a gap between signal-dominated components
        # (e.g. SVs [1.0, 0.5, 0.3, 0.2]) as a signal-noise boundary.
        if S[max_idx_val + 1] / S[0] < 0.05:
            noise_tail = S[max_idx_val + 1:]
            if len(noise_tail) >= 2:
                noise_floor = noise_tail.mean().item()
                if noise_floor > 0.0 and math.isfinite(noise_floor):
                    return noise_floor * noise_factor

    # ── Stage 2: Fallback — tail mean with adaptive factor ──────────────
    # For small q (≤8), if gap detection found no clear signal-noise elbow,
    # ALL SVs are treated as signal — skip noise filtering.
    # The energy-based truncation (k_energy) already handles rank selection.
    # The tail-based method was designed for high-rank matrices (rank≥64)
    # where a clear noise tail exists; for low-rank signals, the bottom SVs
    # are legitimate signal components, not noise.
    if q <= 8:
        return 0.0

    tail_size = max(int(q * _NOISE_TAIL_FRACTION), 2)
    tail = S[-tail_size:]

    noise_floor = tail.mean().item()

    # Guard against degenerate cases
    if noise_floor <= 0.0 or not math.isfinite(noise_floor):
        return 0.0

    return noise_floor * noise_factor


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
        checkpoints = folder_paths.get_filename_list("checkpoints")
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
                "rank": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 320,
                    "step": 1,
                    "tooltip": "Target LoRA rank (1–320). In 'auto_energy' mode this is the upper bound.",
                }),
                "alpha": ("FLOAT", {
                    "default": 64.0,
                    "min": 1.0,
                    "max": 512.0,
                    "step": 1.0,
                    "tooltip": "LoRA alpha scaling factor. In 'manual' mode: used as-is. In 'auto_energy'/'full' modes: auto-computed (alpha ≈ effective_rank).",
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
            kwargs.get("rank", 64),
            kwargs.get("alpha", 64.0),
            kwargs.get("svd_mode", "auto_energy"),
            kwargs.get("energy_threshold", 0.95),
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
        full = folder_paths.get_full_path("checkpoints", name)
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
                skipped_non_2d += 1
                continue

            # Convert to LoRA key format
            lora_key = _checkpoint_key_to_lora_key(norm_key)

            # Deduplication guard: skip if this lora_key already processed
            if lora_key in seen_lora_keys:
                continue
            seen_lora_keys.add(lora_key)

            matched_pairs.append((lora_key, base_t, tuned_t))
            matched_count += 1

        return matched_pairs, matched_count, skipped_shape, skipped_non_2d

    @staticmethod
    def _estimate_attenuation(
        matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
        sample_size: Optional[int] = 30,
    ) -> Tuple[float, float, int, float]:
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
            ``(auto_multiplier, attenuation_factor, layers_analyzed, target_ratio)``
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
            return 1.0, 1.0, 0, 0.0

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
            return 1.0, attenuation_factor, layers_analyzed, median_ratio

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

        return auto_multiplier, attenuation_factor, layers_analyzed, target_ratio

    # ── Public entry point ───────────────────────────────────────────────

    def extract(
        self,
        checkpoint_base: str,
        checkpoint_tuned: str,
        rank: int,
        alpha: float,
        svd_mode: str,
        energy_threshold: float = 0.95,
        strength_multiplier: Union[str, float] = "auto",
        detection_mode: str = "auto_fast",
        device: str = "auto",
        save_trigger: bool = False,
        filename: str = "extracted_lora",
        save_folder: str = "",
        model=None,
        clip=None,
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
        node_id: str = "",
    ) -> Tuple[Any, Any, str, str]:
        """
        Execute the extraction.

        Args:
            checkpoint_base:     Filename or path to the base checkpoint.
            checkpoint_tuned:    Filename or path to the tuned checkpoint.
            rank:                Target LoRA rank.
            alpha:               LoRA alpha scaling factor.
            svd_mode:            ``"auto_energy"``, ``"manual"``, or ``"full"``.
            energy_threshold:    Energy retention for auto_energy mode (0.50–0.999).
            strength_multiplier: ``"auto"`` (auto-detect attenuation) or a numeric string
                                 like ``"1.0"``, ``"1.5"``, ``"2.0"``, ``"3.0"``.
            detection_mode:      ``"auto_fast"`` (sample 30 layers), ``"auto_precise"``
                                 (analyze all layers), or ``"manual"`` (use multiplier directly).
            save_trigger:        When ``True``, save the LoRA to disk.
            filename:            Output filename (without extension).
            save_folder:         Optional custom output directory.
            model:               Optional MODEL to apply the extracted LoRA to for preview.
            clip:                Optional CLIP to apply the extracted LoRA to for preview.
            strength_model:      LoRA strength applied to the model (0.0–10.0).
            strength_clip:       LoRA strength applied to the CLIP (0.0–10.0).
            node_id:             ComfyUI node ID (for cache isolation).

        Returns:
            ``(model, clip, forensic_report_string, lora_path_or_empty_string)``.
        """
        # ── 1. Resolve paths & load checkpoints ──────────────────────────
        base_path = self._resolve_checkpoint_path(checkpoint_base)
        tuned_path = self._resolve_checkpoint_path(checkpoint_tuned)

        if base_path is None:
            return (
                None, None,
                "❌ ERROR: Base checkpoint not found. Ensure the file exists "
                "in ComfyUI's 'checkpoints' folder or provide an absolute path.",
                "",
            )
        if tuned_path is None:
            return (
                None, None,
                "❌ ERROR: Tuned checkpoint not found. Ensure the file exists "
                "in ComfyUI's 'checkpoints' folder or provide an absolute path.",
                "",
            )

        base_name = base_path.name
        tuned_name = tuned_path.name
        print(f"\n🛡️ Easy LoRA Extractor — Loading checkpoints...")
        print(f"   Base:  {base_name}")
        print(f"   Tuned: {tuned_name}")

        base_sd, base_meta = load_checkpoint_with_metadata(base_path)
        tuned_sd, tuned_meta = load_checkpoint_with_metadata(tuned_path)

        total_base_keys = len(base_sd)
        total_tuned_keys = len(tuned_sd)
        print(f"   Loaded {total_base_keys} base keys, {total_tuned_keys} tuned keys")

        # ── 2. Detect architectures ──────────────────────────────────────
        base_arch = detect_checkpoint_architecture(list(base_sd.keys()))
        tuned_arch = detect_checkpoint_architecture(list(tuned_sd.keys()))
        print(f"   Base arch:  {base_arch}")
        print(f"   Tuned arch: {tuned_arch}")

        # ⚠️  Cross-architecture warning
        if base_arch != tuned_arch:
            print(f"   ⚠️  Architectures differ! Matching may be limited.")

        # ── 3a. Baked LoRA detection ─────────────────────────────────────
        if tuned_meta and ('baking_method' in tuned_meta or 'baking_tool' in tuned_meta):
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

        # ── 3b. Resolve device for SVD acceleration ──────────────────────
        target_device = DeviceManager.get_device(device)
        print(f"   Device: {target_device} (user choice: {device})")

        # ── 4. Match keys ────────────────────────────────────────────────
        print(f"   Matching keys...")
        matched_pairs, matched_count, skipped_shape, skipped_non_2d = (
            self._match_checkpoint_keys(base_sd, tuned_sd, base_arch, tuned_arch)
        )
        print(f"   Matched {matched_count} keys, {skipped_shape} shape mismatches, "
              f"{skipped_non_2d} non-2D skipped")

        # Compute per-component counts from the matched pairs BEFORE zero-delta
        # filtering. This gives visibility into which component types were
        # matched, even if they produce zero delta (e.g. TE keys in a baked
        # LoRA that only modified UNet weights).
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
            return (
                None, None,
                "❌ ERROR: No compatible keys found between base and tuned checkpoints. "
                "They may be incompatible architectures or have no overlapping layers.",
                "",
            )

        # ── 4a. Auto strength multiplier analysis ─────────────────────────
        # ── Constants for per-layer boosting ──────────────────────────────
        # Significance threshold: layers with delta/base norm ratio below this
        # are treated as pure quantization noise and skipped entirely.
        SIGNIFICANCE_THRESHOLD = 0.0001  # 0.01% of base norm
        # Maximum per-layer multiplier to prevent extreme noise amplification.
        MAX_LAYER_MULTIPLIER = 20.0

        effective_multiplier: float = 1.0
        per_layer_target_ratio: float = 0.0  # 0.0 = disabled
        attenuation_factor: float = 1.0
        attenuation_layers_analyzed: int = 0
        skipped_insignificant: int = 0

        if isinstance(strength_multiplier, str) and strength_multiplier == "auto":
            # Determine sampling strategy from detection_mode
            if detection_mode == "auto_precise":
                sample_size = None  # analyze ALL layers
            else:  # auto_fast (default)
                sample_size = 30

            auto_mult, atten, analyzed, target_ratio = self._estimate_attenuation(
                matched_pairs, sample_size=sample_size
            )
            effective_multiplier = auto_mult
            attenuation_factor = atten
            attenuation_layers_analyzed = analyzed

            # ── Activate per-layer adaptive boosting ─────────────────────
            # When target_ratio > 0 and auto_mult > 1.0, use per-layer
            # instead of global multiplier. Each layer is boosted only enough
            # to reach target_ratio, preventing noise amplification in
            # layers the original LoRA didn't modify.
            if target_ratio > 0.0 and auto_mult > 1.0:
                per_layer_target_ratio = target_ratio
                print(f"   🔧 Per-layer adaptive boosting enabled "
                      f"(target ratio={target_ratio * 100:.2f}%)")
            elif auto_mult != 1.0:
                print(f"   🔧 Auto-applied strength_multiplier={auto_mult:.1f}x "
                      f"to compensate for baking attenuation")
            else:
                print(f"   ✅ Deltas appear healthy — no multiplier needed")
        elif isinstance(strength_multiplier, str):
            effective_multiplier = float(strength_multiplier)
            print(f"   Using manual strength_multiplier={effective_multiplier}")

        # ── 4b. Compute deltas & apply SVD ──────────────────────────────
        print(f"   Computing deltas and applying SVD (mode={svd_mode}, rank={rank})...")
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
                # Compute delta/base norm ratio for significance check and
                # diagnostic logging. Skip layers where delta is below the
                # noise floor — these are pure quantization noise.
                delta_norm = torch.norm(delta).item()
                base_norm = torch.norm(base_t.float()).item()
                ratio = delta_norm / max(base_norm, 1e-12)

                if per_layer_target_ratio > 0.0 and ratio < SIGNIFICANCE_THRESHOLD:
                    skipped_insignificant += 1
                    progress += 1
                    continue

                # ── DIAGNOSTIC: log first 5 deltas ──────────────────────────
                # Print actual delta stats to verify baked delta is present.
                # Expected: delta_norm ≈ 0.25 × original_delta_norm for a
                # rank-4 alpha-1 baked LoRA.  After multiplier, ≈ 0.75×.
                if extracted_layers < 5:
                    print(f"   [DIAG] #{extracted_layers + 1} {lora_key[:48]:>48s} | "
                          f"delta_norm={delta_norm:.6e} "
                          f"base_norm={base_norm:.6e} "
                          f"ratio={ratio*100:.4f}% "
                          f"dtype={delta.dtype} "
                          f"shape={list(delta.shape)}")

                # ── Per-layer adaptive multiplier ────────────────────────────
                # Boost only enough to reach target_ratio, capped to prevent
                # extreme noise amplification. Layers already at or above
                # target_ratio get no boost.
                if per_layer_target_ratio > 0.0 and ratio < per_layer_target_ratio:
                    layer_mult = min(per_layer_target_ratio / max(ratio, 1e-12), MAX_LAYER_MULTIPLIER)
                    if layer_mult > 1.0:
                        delta = delta * layer_mult
                        per_layer_boosted += 1
                elif effective_multiplier != 1.0 and per_layer_target_ratio == 0.0:
                    # Legacy global multiplier (only when per-layer is disabled)
                    delta = delta * effective_multiplier

                # ── SVD decomposition (always in float32) ────────────────────
                result = _decompose_delta_to_lora(
                    delta, lora_key, rank, svd_mode,
                    energy_threshold=energy_threshold,
                    device=device,
                )
                if result is None:
                    progress += 1
                    continue

                lora_A, lora_B, effective_rank = result

                # ── Alpha key ─────────────────────────────────────────────────
                # Compute alpha BEFORE the diagnostic section so it can be used
                # in SVD reconstruction quality check printout.  Use effective
                # rank for auto_energy mode to ensure alpha/rank ≈ 1.0 for
                # correct delta reconstruction.  For full mode, also use
                # effective_rank so that alpha/effective_rank ≈ 1.0 — otherwise
                # the full delta would be attenuated by alpha/rank_factor where
                # rank_factor is the number of SVs retained.
                if svd_mode == "auto_energy":
                    layer_alpha = max(1.0, float(effective_rank))
                elif svd_mode == "full":
                    # Full mode keeps ALL singular values, so the LoRA's
                    # effective rank = number of SVs kept = effective_rank.
                    # Set alpha = effective_rank so that ComfyUI's
                    # scale = alpha / effective_rank = 1.0, applying the
                    # full reconstructed delta.
                    layer_alpha = max(alpha, float(effective_rank))
                else:
                    layer_alpha = alpha

                # ── DIAGNOSTIC: SVD reconstruction quality check ────────────
                # Verify lora_B @ lora_A ≈ delta (should be exact for manual
                # mode with rank=4 and 0 🔇 entries).
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

                # ── Assemble LoRA state dict (ComfyUI-native naming) ─────────
                lora_sd[f"{lora_key}.lora_A.weight"] = lora_A
                lora_sd[f"{lora_key}.lora_B.weight"] = lora_B
                lora_sd[f"{lora_key}.alpha"] = torch.tensor(
                    [layer_alpha], dtype=lora_A.dtype
                )

                # ── Component tracking ───────────────────────────────────────
                comp = categorize_checkpoint_key(lora_key)
                if comp not in component_breakdown:
                    component_breakdown[comp] = {"count": 0, "energy": 0.0, "avg_rank": 0.0}
                component_breakdown[comp]["count"] += 1
                # Energy = sum of squared singular values of the delta
                delta_energy = torch.sum(delta ** 2).item()
                component_breakdown[comp]["energy"] += delta_energy
                component_breakdown[comp]["avg_rank"] = (
                    (component_breakdown[comp]["avg_rank"] * (component_breakdown[comp]["count"] - 1) + effective_rank)
                    / component_breakdown[comp]["count"]
                )
                total_extraction_energy += delta_energy
                rank_sum += effective_rank
                extracted_layers += 1

                progress += 1


        effective_rank_avg = rank_sum / max(extracted_layers, 1)

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
            print(f"       This often indicates attenuated deltas from baked LoRAs.")
            print(f"       Try increasing 'strength_multiplier' (1.5–3.0) to amplify the signal")
            print(f"       before SVD, or lower 'svd_mode' from auto_energy to manual with a")
            print(f"       higher rank.")

        if extracted_layers == 0:
            return (
                None, None,
                "❌ ERROR: All matched keys produced zero delta or non-decomposable tensors. "
                "The two checkpoints appear identical.",
                "",
            )

        # ── 4c. CLIP Key Debug Output ────────────────────────────────────
        # Log a sample of CLIP-related LoRA keys to help diagnose key mapping issues.
        # ComfyUI's SD1.5 SD1ClipModel stores parameters with keys like
        # "transformer.text_model.encoder.layers.X.self_attn.k_proj.weight"
        # (stripping the "cond_stage_model." prefix from checkpoint keys).
        # The LoRA keys must match these parameter names for patches to attach.
        if lora_sd:
            clip_keys = [k for k in lora_sd if 'text_model' in k or 'text_encoders' in k]
            if clip_keys:
                # Show first 3 unique base keys for CLIP
                clip_bases = sorted(set(
                    k.replace('.lora_A.weight', '').replace('.lora_B.weight', '').replace('.alpha', '')
                    for k in clip_keys
                ))
                print(f"\n   🔍 CLIP LoRA keys generated ({len(clip_bases)} unique bases):")
                for base in clip_bases[:5]:
                    print(f"      • {base}")
                if len(clip_bases) > 5:
                    print(f"      • ... and {len(clip_bases) - 5} more")
                print(f"   ℹ️  If 0 CLIP patches attach, the key prefix may not match")
                print(f"      ComfyUI's expected format. Expected format for SD1.5:")
                print(f"      text_encoders.clip_l.transformer.text_model.encoder.layers.X.self_attn.k_proj")
            else:
                print(f"\n   ℹ️  No CLIP/text encoder keys in extracted LoRA")

        # ── 5. Save to disk ──────────────────────────────────────────────
        output_path_str = ""
        if save_trigger:
            try:
                # Resolve output path
                if not filename or filename.strip() == "":
                    filename = "extracted_lora"
                if not filename.endswith(".safetensors"):
                    filename += ".safetensors"
                safe_filename = "".join(
                    c for c in filename if c.isalnum() or c in "._- "
                ).rstrip()

                # Determine output directory
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

                # Auto-increment filename
                output_path = output_dir / safe_filename
                counter = 1
                while output_path.exists():
                    stem = output_path.stem
                    parts = stem.split("_")
                    if len(parts) > 1 and parts[-1].isdigit():
                        stem = "_".join(parts[:-1])
                    output_path = output_dir / f"{stem}_{counter}{output_path.suffix}"
                    counter += 1

                # Build metadata using unified factory
                metadata = finalize_metadata(
                    metadata=base_meta,  # preserve base checkpoint metadata
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

                # Write safetensors
                from safetensors.torch import save_file as safe_save
                safe_save(lora_sd, str(output_path), metadata)
                output_path_str = str(output_path)
                print(f"   💾 LoRA saved to: {output_path_str}")

            except Exception as e:
                error_msg = f"❌ Failed to save LoRA: {e}"
                print(f"   {error_msg}")
                # Continue — still return forensics without save path

        # ── 6. Apply extracted LoRA to model + clip for preview ──────────
        preview_model = None
        preview_clip = None
        if model is not None and clip is not None and lora_sd:
            try:
                preview_model, preview_clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_sd, strength_model, strength_clip
                )
                print(f"   🖼️ LoRA preview applied to model+clip "
                      f"(strength_model={strength_model}, strength_clip={strength_clip})")
            except Exception as e:
                print(f"   ⚠️ Failed to apply LoRA preview: {e}")
                # Fall back to original model/clip on error
                preview_model = model
                preview_clip = clip

        # ── 7. Build forensic report ─────────────────────────────────────
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
