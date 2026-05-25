"""
GGUF Writer — Export quantized checkpoints to GGUF format for ComfyUI-GGUF.

Provides:
  - GGUFSaveWriter       — Wraps ``gguf.GGUFWriter`` for diffusion model checkpoints
  - gguf_arch_from_arch  — Maps internal architecture names to GGUF arch strings
  - GGUF_SUPPORTED_ARCHS — Set of architectures that ComfyUI-GGUF can load
  - dequant_gguf_tensor  — Dequantize a GGUF-quantized tensor back to FP32

Design principles (from working_principles.md):
  - P3 (One Pattern Per Concept): single module, single writer class
  - P2 (No Temp Files): writes directly to final output path
  - File is self-contained; musubi_checkpoint_studio.py delegates to it.

Usage (within Checkpoint Studio save path)::

    from .gguf_writer import GGUFSaveWriter, gguf_arch_from_arch, GGUF_SUPPORTED_ARCHS

    arch_raw = detect_checkpoint_architecture(keys)  # e.g. "flux"
    if arch_raw not in GGUF_SUPPORTED_ARCHS:
        print(f"  ⚠️ GGUF export not supported for architecture '{arch_raw}'")
        return

    writer = GGUFSaveWriter(output_path, arch_raw, gguf_qtype)
    for key, tensor in processed.items():
        writer.add_tensor(key, tensor)
    writer.finalize()
"""

from typing import Optional, Set

import numpy as np
import torch

try:
    import gguf
except ImportError:
    gguf = None  # graceful fallback — check in save branch


# ── Architecture support ──────────────────────────────────────────────────────
# Architectures that ComfyUI-GGUF's loader recognises.
# Source: ComfyUI-GGUF/loader.py IMG_ARCH_LIST
GGUF_SUPPORTED_ARCHS: Set[str] = {
    "flux", "sd1", "sdxl", "sd3", "aura", "hidream",
    "cosmos", "ltxv", "hyvid", "wan", "lumina2",
}

# Internal architecture name → GGUF arch string mapping.
# Key = raw arch from detect_checkpoint_architecture() (lowercase),
# Value = GGUF general.architecture value that ComfyUI-GGUF expects.
_ARCH_MAP = {
    "flux":     "flux",
    "sdxl":     "sdxl",
    "sd15":     "sd1",
    "sd1.5":    "sd1",
    "lumina2":  "lumina2",
    # The following are NOT in GGUF_SUPPORTED_ARCHS — will be rejected.
    "anima":    None,
    "z_image":  None,
}


def gguf_arch_from_arch(arch_raw: str) -> Optional[str]:
    """Map internal architecture string to GGUF ``general.architecture`` value.

    Args:
        arch_raw: Raw architecture string from ``detect_checkpoint_architecture()``
                  (lowercase, e.g. ``"flux"``, ``"sd15"``, ``"sdxl"``).

    Returns:
        GGUF arch string (e.g. ``"flux"``, ``"sd1"``) or ``None`` if unsupported.
    """
    key = arch_raw.lower().replace(" ", "_")
    gguf_arch = _ARCH_MAP.get(key)
    if gguf_arch is not None and gguf_arch not in GGUF_SUPPORTED_ARCHS:
        return None
    return gguf_arch


# ── GGUF quant type selection ────────────────────────────────────────────────
# Map our config-level names (e.g. "gguf_q8_0") to GGMLQuantizationType enums.
GGUF_QTYPE_MAP = {
    "gguf_q8_0": gguf.GGMLQuantizationType.Q8_0 if gguf else "Q8_0",
    "gguf_q5_0": gguf.GGMLQuantizationType.Q5_0 if gguf else "Q5_0",
    "gguf_q4_0": gguf.GGMLQuantizationType.Q4_0 if gguf else "Q4_0",
}

# Human-readable labels for the UI dropdown.
GGUF_QTYPE_LABELS = {
    "gguf_q8_0": "gguf (Q8_0)",
    "gguf_q5_0": "gguf (Q5_0)",
    "gguf_q4_0": "gguf (Q4_0)",
}

# File type for LlamaFileType metadata
_GGUF_QTYPE_TO_FILE_TYPE = {} if gguf is None else {
    gguf.GGMLQuantizationType.Q8_0: gguf.LlamaFileType.MOSTLY_Q8_0,
    gguf.GGMLQuantizationType.Q5_0: gguf.LlamaFileType.MOSTLY_Q5_0,
    gguf.GGMLQuantizationType.Q4_0: gguf.LlamaFileType.MOSTLY_Q4_0,
}

# Thresholds for shape rearrangement (SD1.5/SDXL only)
_REARRANGE_THRESHOLD = 512
_QUANTIZATION_THRESHOLD = 1024

# ── High-precision key patterns ──────────────────────────────────────────────
# Patterns for tensor keys that MUST remain as F32 regardless of size.
# Block quantization (Q8_0/Q5_0/Q4_0) produces GGMLTensor (non-float), which
# crashes torch.nn.Module.load_state_dict() when the model expects float params.
#
# This mirrors ComfyUI-GGUF's ``ModelTemplate.keys_hiprec`` pattern
# (tools/convert.py:20, :283-285).
_KEYS_HIPREC: tuple = (
    # Lumina2 / NextDiT — embedding/padding tokens
    "x_pad_token",
    "cap_pad_token",
    # HiDream (from ComfyUI-GGUF)
    ".ff_i.gate.weight",
    "img_emb.emb_pos",
    # Cosmos (from ComfyUI-GGUF)
    "pos_embedder",
    # LTXV (from ComfyUI-GGUF)
    "scale_shift_table",
    # Wan (from ComfyUI-GGUF)
    ".modulation",
)


class GGUFSaveWriter:
    """Write a state dict to a GGUF file with block-wise quantization.

    Usage::

        writer = GGUFSaveWriter("/path/to/output.gguf", "flux", gguf.GGMLQuantizationType.Q8_0)
        for key, tensor in state_dict.items():
            writer.add_tensor(key, tensor)
        writer.finalize()
    """

    def __init__(
        self,
        output_path: str,
        arch_raw: str,
        quant_type: "gguf.GGMLQuantizationType",
        shape_fix: bool = False,
    ):
        """Initialise the GGUF writer.

        Args:
            output_path: Absolute or relative path for the output ``.gguf`` file.
            arch_raw: Architecture string from the checkpoint analyser
                      (e.g. ``"flux"``, ``"sd15"``, ``"sdxl"``).
            quant_type: Target GGML quantisation type (Q8_0, Q5_0, Q4_0, etc.).
            shape_fix: Enable shape rearrangement for CNN models (SD1.5/SDXL).
                       Flux/transformer models should set this to ``False``.
        """
        if gguf is None:
            raise ImportError("The 'gguf' Python package is required for GGUF export. "
                              "Install it with: pip install gguf")

        self._output_path = output_path
        self._quant_type = quant_type
        self._shape_fix = shape_fix

        # Resolve GGUF arch string
        self._gguf_arch = gguf_arch_from_arch(arch_raw)
        if self._gguf_arch is None:
            raise ValueError(
                f"Architecture '{arch_raw}' is not supported for GGUF export. "
                f"Supported: {sorted(GGUF_SUPPORTED_ARCHS)}"
            )

        # Create the underlying writer
        self._writer = gguf.GGUFWriter(path=None, arch=self._gguf_arch)
        self._writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        # Add file type metadata (maps quant type → LlamaFileType)
        file_type = _GGUF_QTYPE_TO_FILE_TYPE.get(quant_type)
        if file_type is not None:
            self._writer.add_file_type(file_type)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Quantise and add a single tensor to the GGUF file.

        Handles:
          - dtype conversion (BF16/FP16 → FP32 → quantise)
          - 1D tensor preservation (F32, no quantisation)
          - FP8 tensors (found in re-saved checkpoints) → FP32 → quantise
          - Shape rearrangement for ``shape_fix`` models (SD1.5/SDXL)
          - ``comfy.gguf.orig_shape.*`` metadata for rearranged tensors

        Args:
            key: Tensor key (e.g. ``"double_blocks.0.img_attn.qkv.weight"``).
            tensor: PyTorch tensor to quantise and add.
        """
        # ── Skip non-weight tensors ──────────────────────────────────────
        # Companion scales (.weight_scale, .input_scale) are embedded in
        # GGUF quantisation and must NOT be written as separate tensors.
        # comfy_quant metadata (INT8) is also safetensors-specific.
        if key.endswith(('.weight_scale', '.input_scale', '.comfy_quant')):
            return

        # ── Convert to numpy ─────────────────────────────────────────────
        # GGUF quantise operates on float32 numpy arrays.
        data = tensor.detach().cpu()
        old_dtype = data.dtype

        # BF16 → FP32 (gguf.quants.quantize needs float32 input)
        if old_dtype == torch.bfloat16:
            data = data.float()
        # FP8 → FP32
        elif old_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            data = data.float()
        # FP16 → FP32 for quantisation (F16 tensors are stored directly)
        elif old_dtype == torch.float16 and self._quant_type != gguf.GGMLQuantizationType.F16:
            data = data.float()

        data_np = data.numpy()
        n_dims = len(data_np.shape)
        n_params = data_np.size

        # Determine the effective GGML quant type for this tensor
        qtype = self._select_qtype(key, data_np, n_dims, n_params, old_dtype)

        # ── Shape rearrangement (CNN models only) ────────────────────────
        orig_shape = None
        if (
            self._shape_fix
            and n_dims > 1
            and n_params >= _REARRANGE_THRESHOLD
            and (n_params / 256).is_integer()
            and not (data_np.shape[-1] / 256).is_integer()
        ):
            orig_shape = data_np.shape
            data_np = data_np.reshape(n_params // 256, 256)
            self._writer.add_array(
                f"comfy.gguf.orig_shape.{key}",
                tuple(int(d) for d in orig_shape),
            )

        # ── Quantise ─────────────────────────────────────────────────────
        try:
            quantised = gguf.quants.quantize(data_np, qtype)
        except (AttributeError, gguf.QuantError) as exc:
            # Fallback to F16 if quantisation fails
            print(f"   ⚠️ GGUF quantisation fallback to F16 for '{key}': {exc}")
            qtype = gguf.GGMLQuantizationType.F16
            quantised = gguf.quants.quantize(data_np, qtype)

        self._writer.add_tensor(key, quantised, raw_dtype=qtype)

    def finalize(self) -> str:
        """Write all metadata and tensors to the GGUF file and close.

        Returns:
            The output file path.
        """
        self._writer.write_header_to_file(path=self._output_path)
        self._writer.write_kv_data_to_file()
        self._writer.write_tensors_to_file(progress=False)
        self._writer.close()
        return self._output_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select_qtype(
        self,
        key: str,
        data: np.ndarray,
        n_dims: int,
        n_params: int,
        old_dtype: torch.dtype,
    ) -> "gguf.GGMLQuantizationType":
        """Select the appropriate GGML quant type for a single tensor.

        The goal is block-wise quantisation (Q8_0/Q5_0/Q4_0) — unlike
        ComfyUI-GGUF's ``convert.py`` which defaults all tensors to F16
        ("generate F16 GGUF"), we apply the user's selected quant type to
        *all* non-trivial tensors regardless of original dtype.

        Rules:
          - 1D tensors → ``F32`` (norms, biases — tiny, avoid dequant overhead)
          - Very small tensors (< 1024 params) → ``F32``
          - Matching ``_KEYS_HIPREC`` patterns → ``F32`` (must stay float for
            ``torch.nn.Module.load_state_dict()`` compatibility)
          - Already BF16 → ``BF16`` (no block-quant path in ComfyUI-GGUF loader)
          - Otherwise → user-selected quant type (Q8_0/Q5_0/Q4_0)
        """
        # 1D tensors: keep in F32 (no block quantisation benefit)
        if n_dims == 1:
            return gguf.GGMLQuantizationType.F32

        # Very small tensors: F32 is cheaper than quantise + dequant
        if n_params <= _QUANTIZATION_THRESHOLD:
            return gguf.GGMLQuantizationType.F32

        # BF16 — preserve as BF16 (ComfyUI-GGUF's loader can handle it,
        # but there's no block-quant path for BF16 in the gguf library)
        if old_dtype == torch.bfloat16:
            return gguf.GGMLQuantizationType.BF16

        # High-precision keys — must remain F32 (embedding/padding tokens
        # like x_pad_token, cap_pad_token, etc.). Block quantization produces
        # GGMLTensor (non-float), which crashes load_state_dict().
        if any(pattern in key for pattern in _KEYS_HIPREC):
            return gguf.GGMLQuantizationType.F32

        # User-selected quant type — applies to FP32, FP16, FP8, etc.
        return self._quant_type


# ── Dequantisation for preview mode ──────────────────────────────────────────

def dequant_gguf_tensor(
    data: np.ndarray,
    qtype: "gguf.GGMLQuantizationType",
    orig_shape: Optional[tuple] = None,
) -> torch.Tensor:
    """Dequantize a GGUF-quantized numpy array back to a FP32 torch tensor.

    Args:
        data: Quantised tensor data (uint8 numpy array from GGUF file).
        qtype: The GGML quantisation type that was used.
        orig_shape: Original tensor shape (for shape-rearranged tensors).

    Returns:
        Dequantised FP32 torch tensor.
    """
    dq = gguf.quants.dequantize(data, qtype)
    if orig_shape is not None:
        dq = dq.reshape(orig_shape)
    return torch.from_numpy(dq)


# ── Convenience: map precision string → GGUF quant type ──────────────────────

def gguf_qtype_from_precision(precision: str) -> Optional["gguf.GGMLQuantizationType"]:
    """Map a precision string (from the Checkpoint Studio UI) to a GGML quant type.

    Args:
        precision: E.g. ``"gguf_q8_0"``, ``"gguf_q5_0"``, ``"gguf_q4_0"``.

    Returns:
        The corresponding ``GGMLQuantizationType`` or ``None`` if not a GGUF type.
    """
    entry = GGUF_QTYPE_MAP.get(precision)
    if entry is None:
        return None
    if gguf and isinstance(entry, gguf.GGMLQuantizationType):
        return entry
    return None


def is_gguf_precision(precision: str) -> bool:
    """Return ``True`` if *precision* is a GGUF output format."""
    return precision in GGUF_QTYPE_MAP
