"""
Easy LoRA Merger — ComfyUI custom node pack.
Re-exports all node class/display-name mappings and registers the SmartModelBaker.
"""

from .easy_lora_merger import NODE_CLASS_MAPPINGS as ELM_CLASS_MAPPINGS
from .easy_lora_merger import NODE_DISPLAY_NAME_MAPPINGS as ELM_DISPLAY_MAPPINGS

# This line is the bridge that tells ComfyUI to load your JS files
WEB_DIRECTORY = "./js"

# ── SmartModelBaker (guarded) ──────────────────────────────────────────────
try:
    from .baker_node import SmartModelBaker
    _BAKER_AVAILABLE = True
except ImportError as _e:
    SmartModelBaker = None  # type: ignore[assignment]
    _BAKER_AVAILABLE = False
    print(f"⚠️ Easy LoRA Merger: SmartModelBaker not available ({_e})")

# Merge mappings: start with existing, add SmartModelBaker if available
NODE_CLASS_MAPPINGS = {
    **ELM_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **ELM_DISPLAY_MAPPINGS,
}

if _BAKER_AVAILABLE:
    NODE_CLASS_MAPPINGS["SmartModelBaker"] = SmartModelBaker
    NODE_DISPLAY_NAME_MAPPINGS["SmartModelBaker"] = "🔥 Easy LoRA Baker"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
