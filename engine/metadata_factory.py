"""
Metadata Factory — Unified metadata cleansing and signing for Easy LoRA Merger.

Provides three functions used by all nodes/engines:

  scrub_metadata(metadata, mode)
      → Discard / preserve / merge based on mode.

  sign_metadata(metadata, component, extra_fields)
      → Add project signature per component type.

  finalize_metadata(metadata, mode, component, extra_fields)
      → Shorthand: scrub() + sign() in one call.
"""

import time
from typing import Dict, Optional, Any

from ..config import EASY_LORA_MERGER_VERSION, EASY_LORA_MERGER_DATE

# ── Helpers ────────────────────────────────────────────────────────────────


def _ensure_dict(metadata: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Coerce None to empty dict."""
    return dict(metadata) if metadata else {}


def _now_str() -> str:
    """Current timestamp string used in multiple signatures."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ── Signature presets per component ───────────────────────────────────────


def _build_signature(component: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return the base signature dict for *component*, merged with *extra*."""
    sig: Dict[str, str] = {}

    if component == "merger":
        sig["lora_merge_tool"] = "EasyLoRAMerger-Ariadne"
        sig["merged_date"] = _now_str()
        # merged_key_count expected in extra

    elif component == "baker":
        sig["baking_tool"] = "EasyLoRAMerger-SmartBaker"
        sig["easy_lora_merger_version"] = EASY_LORA_MERGER_VERSION
        sig["baking_date"] = _now_str()
        # baking_method, baking_strength, baked_lora_source, source_checkpoint,
        # baked_key_count, preserved_key_count, weight_unet/te/clip/vae,
        # metadata_mode expected in extra

    elif component == "converter":
        sig["conversion_tool"] = "EasyLoRAMerger-LoraStudio"
        sig["easy_lora_merger_version"] = EASY_LORA_MERGER_VERSION
        sig["conversion_date"] = _now_str()
        # target_format, converted_key_count, original_format expected in extra

    elif component == "extractor":
        sig["extraction_tool"] = "EasyLoRAMerger-Extractor"
        sig["easy_lora_merger_version"] = EASY_LORA_MERGER_VERSION
        sig["extraction_date"] = _now_str()
        # base_checkpoint, tuned_checkpoint, svd_mode, target_rank, etc.
        # expected in extra

    elif component == "checkpoint_studio":
        sig["modelspec.sai_model_spec"] = "1.0.0"
        sig["modelspec.implementation"] = "musubi_checkpoint_studio"
        sig["modelspec.date"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        # modelspec.architecture expected in extra

    else:
        raise ValueError(f"Unknown metadata component: {component}")

    if extra:
        sig.update(extra)

    return sig


# ── Public API ─────────────────────────────────────────────────────────────


def scrub_metadata(
    metadata: Optional[Dict[str, str]],
    mode: str = "none",
) -> Dict[str, str]:
    """Cleanse *metadata* according to *mode*.

    Modes
    -----
    ``"none"``
        Discard all original metadata; return empty dict.
    ``"preserve_a"`` / ``"preserve_b"``
        Keep original metadata as-is (both modes are symmetrical at
        the single-source level — the a/b distinction only matters
        for two-source merge using MetadataMerger.merge directly).
    ``"merge_basic"``
        Return a shallow copy (no conflict resolution needed for a
        single source).  When two sources need merging, use
        MetadataMerger.merge directly before calling this function.
    """
    if metadata is None or mode == "none":
        return {}

    if mode in ("preserve_a", "preserve_b", "merge_basic"):
        return dict(metadata)  # shallow copy

    raise ValueError(f"Unknown metadata_mode: {mode}")


def sign_metadata(
    metadata: Dict[str, str],
    component: str = "merger",
    extra_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Add project signature to *metadata* for *component*.

    Parameters
    ----------
    metadata
        Base metadata dict (already scrubbed).
    component
        One of ``"merger"``, ``"baker"``, ``"converter"``,
        ``"extractor"``, ``"checkpoint_studio"``.
    extra_fields
        Optional additional key/value pairs appended to the signature
        (e.g. component-specific fields like ``baked_key_count``).

    Returns
    -------
    A new dict with the signature merged on top of *metadata*.
    The signature fields overwrite any colliding keys in *metadata*.
    """
    base = dict(metadata)  # shallow copy
    sig = _build_signature(component, extra=extra_fields)
    # Signature fields take priority (the project's own stamp)
    base.update(sig)
    return base


def finalize_metadata(
    metadata: Optional[Dict[str, str]],
    mode: str = "none",
    component: str = "merger",
    extra_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Convenience: scrub → sign in one call.

    Equivalent to::

        scrubbed = scrub_metadata(metadata, mode)
        return sign_metadata(scrubbed, component, extra_fields)
    """
    scrubbed = scrub_metadata(metadata, mode)
    return sign_metadata(scrubbed, component, extra_fields)
