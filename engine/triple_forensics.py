"""
Easy LoRA Merger forensic report — extracted from triple_merger_node.py
"""

import time
import statistics
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple

import torch

from ..utils import categorize_key
from .forensics import build_forensic_report, _build_effect_bar


def build_triple_forensic_report(
    method: str,
    density: float,
    blend_mode: str,
    resolved_blend_mode: str,
    balancing_mode: str,
    weights: List[float],
    lora_names: List[str],
    two_way_mode: bool,
    active_lora_count: int,
    merged_dict: Dict[str, torch.Tensor],
    original_sds: List[Dict[str, torch.Tensor]],
    energy_by_component: Optional[Dict] = None,
    adjusted_weights: Optional[List[float]] = None,
    warnings: Optional[List[str]] = None,
) -> str:
    """Build a human-readable forensic report using the shared builder."""

    # ── Summary string (needed for both title_data and active-lora line) ──
    w_str = ", ".join(f"{w:.2f}" for w in weights if abs(w) > 1e-6)

    title_data = OrderedDict([
        ("📦 METHOD", method),
        ("📋 SUMMARY",
         f"{method} | {active_lora_count} LoRAs [{w_str}] "
         f"| blend={resolved_blend_mode} | balance={balancing_mode} "
         f"| {len(merged_dict)} keys"),
    ])

    sections: List[Tuple[Optional[str], List[str]]] = []
    sec: List[str] = []  # accumulator for current raw section

    # ── Blend mode ──
    blend_line = f"🔀 BLEND MODE: {blend_mode}"
    if resolved_blend_mode != blend_mode:
        blend_line += f" → {resolved_blend_mode}"
        if blend_mode == "auto":
            if resolved_blend_mode == "active":
                blend_line += " (trainer mismatch detected)"
            elif resolved_blend_mode == "dense":
                blend_line += " (all trainers match)"
    sec.append(blend_line)

    # ── Weight balancing ──
    if balancing_mode != "disabled":
        sec.append(f"⚖️ WEIGHT BALANCING: {balancing_mode} (active)")
        if energy_by_component and adjusted_weights:
            energy_per_lora = [0.0, 0.0, 0.0]
            for component, energies in energy_by_component.items():
                for i in range(3):
                    energy_per_lora[i] += energies[i]
            avg_energy = sum(energy_per_lora) / 3.0 if energy_per_lora else 1.0
            eps = 1e-12
            for i in range(3):
                label = lora_names[i] if i < len(lora_names) else chr(65 + i)
                orig_w = weights[i] if i < len(weights) else 1.0
                adj_w = adjusted_weights[i] if i < len(adjusted_weights) else orig_w
                ratio_vs_avg = energy_per_lora[i] / max(avg_energy, eps)
                if abs(adj_w - orig_w) > 0.001:
                    direction = "reduced" if adj_w < orig_w else "boosted"
                    sec.append(
                        f"   LoRA {label}: {orig_w:.2f} → {adj_w:.2f} "
                        f"(energy {energy_per_lora[i]:.2e}, {ratio_vs_avg:.2f}x avg, {direction})"
                    )
                else:
                    sec.append(
                        f"   LoRA {label}: {orig_w:.2f} (no change, "
                        f"energy {energy_per_lora[i]:.2e}, {ratio_vs_avg:.2f}x avg)"
                    )
    else:
        sec.append(f"⚖️ WEIGHT BALANCING: disabled")

    # ── Active LoRAs ──
    active_labels = [lora_names[i] for i in range(active_lora_count)]
    w_str_active = ", ".join(f"{weights[i]:.2f}" for i in range(active_lora_count))
    sec.append(f"📊 ACTIVE LoRAs: {active_lora_count} [{', '.join(active_labels)}]")
    sec.append(f"   Weights: [{w_str_active}]")

    # ── Key statistics ──
    total_merged = len(merged_dict)
    sec.append("📊 KEY STATISTICS:")
    sec.append(f"   Total keys merged: {total_merged}")

    # Unique keys per LoRA
    if original_sds and len(original_sds) >= 2:
        unique_counts = []
        for i, sd in enumerate(original_sds):
            other_keys = set()
            for j, other_sd in enumerate(original_sds):
                if j != i:
                    other_keys.update(other_sd.keys())
            unique_count = len(set(sd.keys()) - other_keys)
            label = lora_names[i] if i < len(lora_names) else chr(65 + i)
            unique_counts.append(f"Unique to {label}: {unique_count}")
        sec.append(f"   {' | '.join(unique_counts)}")

    # Shared by ≥2
    if original_sds and len(original_sds) >= 2:
        all_keys_set = set()
        for sd in original_sds:
            all_keys_set.update(sd.keys())
        shared_by_2 = sum(
            1 for k in all_keys_set
            if sum(1 for sd in original_sds if k in sd) >= 2
        )
        sec.append(f"   Shared by ≥2: {shared_by_2}")

    # ── Component breakdown ──
    comp_counts: Dict[str, int] = {}
    for key in merged_dict:
        cat = categorize_key(key)
        comp_counts[cat] = comp_counts.get(cat, 0) + 1
    if comp_counts:
        sec.append("🧱 COMPONENT BREAKDOWN:")
        comp_icons = {
            'model': '🔷',
            'te': '📝',
            'unet': '🔷',
            'clip': '📷',
            'vae': '🎬',
            'other': '❓',
        }
        for comp in ('model', 'te', 'unet', 'clip', 'vae', 'other'):
            if comp in comp_counts:
                icon = comp_icons.get(comp, '▫️')
                label = 'MODEL (UNet)' if comp == 'model' else 'TE (Text Encoder)' if comp == 'te' else comp.upper()
                sec.append(f"   {icon} {label}: {comp_counts[comp]} keys")

    # ── Effect analysis ──
    shared_keys = []
    if original_sds and len(original_sds) >= 2:
        all_keys_set = set()
        for sd in original_sds:
            all_keys_set.update(sd.keys())
        shared_keys = [
            k for k in all_keys_set
            if k in merged_dict and sum(1 for sd in original_sds if k in sd) >= 2
        ]

    if shared_keys:
        merged_norms = []
        ratio_vs_individual = []
        for k in shared_keys:
            mt = merged_dict[k]
            if isinstance(mt, torch.Tensor):
                mn = torch.norm(mt.float()).item()
                merged_norms.append(mn)
                individual_norms = []
                for sd in original_sds:
                    if k in sd and isinstance(sd[k], torch.Tensor):
                        individual_norms.append(torch.norm(sd[k].float()).item())
                if individual_norms:
                    mean_ind = sum(individual_norms) / len(individual_norms)
                    ratio_vs_individual.append(mn / max(mean_ind, 1e-12))

        if merged_norms:
            mean_mn = statistics.mean(merged_norms)
            median_mn = statistics.median(merged_norms)
            max_mn = max(merged_norms)
            sec.append("📊 EFFECT ANALYSIS:")
            sec.append(f"   Shared layers analyzed: {len(merged_norms)}")
            sec.append(f"   Mean merged norm: {mean_mn:.4f}")
            sec.append(f"   Median merged norm: {median_mn:.4f}")
            sec.append(f"   Max merged norm: {max_mn:.4f}")
            if ratio_vs_individual:
                mean_ratio = statistics.mean(ratio_vs_individual)
                median_ratio = statistics.median(ratio_vs_individual)
                max_ratio = max(ratio_vs_individual)
                mean_pct = mean_ratio * 100
                median_pct = median_ratio * 100
                max_pct = max_ratio * 100
                sec.append(f"   Mean ratio vs individual: {mean_pct:.2f}%")
                sec.append(f"   Median ratio: {median_pct:.2f}%")
                sec.append(f"   Max ratio: {max_pct:.2f}%")
                sec.append(f"   {_build_effect_bar(mean_pct, max_pct, width=20)}")
    else:
        sec.append("📊 EFFECT ANALYSIS:")
        sec.append("   No shared keys to analyze")

    # ── Warnings ──
    if warnings:
        sec.append("⚠️ WARNINGS:")
        for w in warnings:
            sec.append(f"   ⚠️ {w}")

    # ── Date (must be last before footer in original output) ──
    sec.append(f"📅 DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    sections.append((None, sec))

    return build_forensic_report(
        report_type="EASY LoRA MERGER",
        title_data=title_data,
        sections=sections,
        footer_width=50,
    )
