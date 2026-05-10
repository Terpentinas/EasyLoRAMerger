"""
Shared forensic report builder for merge/bake operations.

Consolidates three implementations:
  - baker:  _build_forensic_report()    in baking_processor_baking.py
  - triple: build_triple_forensic_report() in triple_forensics.py
  - ckpt:   generate_forensic_report()  in checkpoint_weaver.py

Design: Each caller passes an ordered list of ``(section_title_or_None, lines)``
tuples, giving full control over section ordering.  Shared helper functions
(``_build_component_breakdown``, ``_build_effect_bar``, ``_section``, …)
generate the content for standard section types.

Usage::

    from engine.forensics import build_forensic_report, _build_component_breakdown

    report = build_forensic_report(
        report_type="EASY LoRA BAKER",
        title_data=OrderedDict([("📦 BAKED LORA", loRA_name), …]),
        sections=[
            ("IMPACT PROFILE", impact_lines),
            (None, comp_lines),         # raw – no section wrapper
            ("DELTA ANALYSIS", delta_lines),
        ],
    )
"""

from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# Shared building blocks
# ═══════════════════════════════════════════════════════════════════


def _build_header(report_type: str) -> str:
    """Return the shield-surrounded header line."""
    return f"🛡️ --- {report_type}: FORENSIC REPORT --- 🛡️"


def _build_footer(width: int = 50) -> str:
    """Return the footer separator line."""
    return "=" * width


def _build_effect_bar(mean_pct: float, max_pct: float, width: int = 20) -> str:
    """Build a compact visual bar representing the effect magnitude."""
    filled = max(0, min(width, int((mean_pct / 5.0) * width)))
    bar = '█' * filled + '▁' * (width - filled)
    return f"{bar} mean={mean_pct:.2f}% max={max_pct:.2f}%"


def _section(title: str, lines: List[str], sep: str = '-') -> List[str]:
    """Wrap *lines* with a separator and a section title."""
    result: List[str] = []
    result.append(sep * 50)
    result.append(title)
    result.extend(lines)
    return result


def _build_component_breakdown(
    components: Dict[str, Any],
    icons: Optional[Dict[str, str]] = None,
    comp_order: Optional[List[str]] = None,
) -> List[str]:
    """
    Build component-breakdown lines.

    *components* is a ``{name: data}`` dict.  *data* may be:

    * ``int`` – rendered as ``{icon} {LABEL}: {count} keys``
    * ``dict`` with one of the following shapes:

      * Baker-style keys ``lora_keys``, ``matched``, ``baked``, ``reason``
      * Checkpoint-style keys ``count``, ``params``
      * Plain ``count``

    *icons* overrides the default emoji mapping per component.
    *comp_order* controls the display order (default: ``unet → model → te → clip → vae → other``).
    """
    if not components:
        return []

    default_icons = {
        'unet': '🔷',
        'model': '🔷',
        'te': '📝',
        'clip': '📷',
        'vae': '🎬',
        'other': '❓',
    }
    effective_icons = {**default_icons, **(icons or {})}

    if comp_order is None:
        comp_order = ['unet', 'model', 'te', 'clip', 'vae', 'other']

    lines: List[str] = []
    for comp in comp_order:
        data = components.get(comp)
        if data is None:
            continue

        icon = effective_icons.get(comp, '▫️')

        if isinstance(data, dict):
            # Detect section style by present keys
            has_lora_keys = 'lora_keys' in data
            has_params = 'params' in data
            count = data.get('count', 0)
            lora_keys = data.get('lora_keys', 0)
            matched = data.get('matched', 0)
            baked = data.get('baked', 0)
            params = data.get('params', 0)
            reason = data.get('reason', '')

            if has_lora_keys:
                # Baker-style
                label = comp.upper()
                if reason:
                    lines.append(
                        f"   {icon} {label}: {lora_keys} lora keys → SKIPPED ({reason})"
                    )
                else:
                    lines.append(
                        f"   {icon} {label}: {lora_keys} lora keys → {matched} matched → {baked} baked"
                    )
            elif has_params and count:
                # Checkpoint-style
                lines.append(
                    f"   {icon} {comp.upper():<6} {count:>4} tensors, {params:>12,} parameters"
                )
            else:
                lines.append(f"   {icon} {comp.upper()}: {count} keys")
        elif isinstance(data, int):
            lines.append(f"   {icon} {comp.upper()}: {data} keys")
        elif isinstance(data, str):
            lines.append(f"   {data}")

    return lines


# ═══════════════════════════════════════════════════════════════════
# Main builder
# ═══════════════════════════════════════════════════════════════════


def build_forensic_report(
    report_type: str,
    title_data: Dict[str, str],
    sections: Optional[List[Tuple[Optional[str], List[str]]]] = None,
    footer_width: int = 50,
) -> str:
    """
    Build a human-readable forensic report string.

    Renders in order:

    1. **Header** from *report_type*
    2. **Title data** – one ``{label}: {value}`` line per entry
    3. **Ordered sections** – each ``(title, lines)`` tuple is rendered with
       ``_section()`` if *title* is not ``None``, otherwise *lines* are added
       verbatim (useful for blank separators or raw paragraphs).
    4. **Footer** – ``=`` repeated *footer_width* times

    Parameters
    ----------
    report_type:
        Short name for the header – e.g. ``"EASY LoRA BAKER"``,
        ``"EASY LoRA MERGER"``, ``"EASY CHECKPOINT MERGER"``.
    title_data:
        Ordered dict whose items become ``{label}: {value}`` lines right
        after the header.  Use ``collections.OrderedDict`` if insertion
        order must be explicit.
    sections:
        Ordered list of ``(section_title_or_None, lines_for_section)``.
        Pass ``None`` as title to emit raw lines without a section wrapper.
    footer_width:
        Character width of the footer separator (default 50).

    Returns
    -------
    The formatted report as a single string.
    """
    lines: List[str] = []

    # ── 1. Header ──
    lines.append(_build_header(report_type))

    # ── 2. Title data ──
    for label, value in title_data.items():
        lines.append(f"{label}: {value}")

    # ── 3. Ordered sections ──
    if sections:
        for section_title, section_lines in sections:
            if section_title is not None:
                lines.extend(_section(f"📊 {section_title}:", section_lines))
            else:
                lines.extend(section_lines)

    # ── 4. Footer ──
    lines.append(_build_footer(footer_width))

    return '\n'.join(lines)
