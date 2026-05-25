"""
Easy LoRA Merger - Node class mappings and display name mappings.
This module acts as a re-export hub for all remaining nodes.

Each node import is individually guarded so that a failure in one
module (e.g. missing optional dependency) does not cascade and
unregister all nodes.
"""

# ── Individual guarded imports ────────────────────────────────────────────
# Each import is wrapped in try/except so one failure doesn't cascade.
# Only successfully imported nodes are registered below.

_node_registry = {}
_node_display_names = {}

# Mapping of node IDs to their display names (used if import succeeds)
_DISPLAY_NAMES = {
    "EasyLoRATripleMerger": "🎨 Easy LoRA Merger",
    "MusubiLoraConverter": "🛡️ Easy LoRA Studio",
    "MusubiCheckpointStudio": "🛡️ Easy Checkpoint Studio",
    "EasyCheckpointMerger": "🎨 Easy Checkpoint Merger",
    "EasyLoRAExtractor": "🔬 Easy LoRA Extractor",
    "EasyComponentExtractor": "🔧 Easy Component Extractor",
    "EasyComponentCombiner": "🧩 Easy Component Combiner",
    "EasyComponentMerger": "🧩 Easy Component Merger",
    "EasyTextDisplay": "📋 Easy Text Display",
}

# ── EasyLoRATripleMerger ─────────────────────────────────────────────────
try:
    from .triple_merger_node import EasyLoRATripleMerger
    _node_registry["EasyLoRATripleMerger"] = EasyLoRATripleMerger
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: EasyLoRATripleMerger not available ({_e})")

# ── MusubiLoraConverter ──────────────────────────────────────────────────
try:
    from .engine.lora_studio_converter import MusubiLoraConverter
    _node_registry["MusubiLoraConverter"] = MusubiLoraConverter
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: MusubiLoraConverter not available ({_e})")

# ── MusubiCheckpointStudio ───────────────────────────────────────────────
try:
    from .engine.musubi_checkpoint_studio import MusubiCheckpointStudio
    _node_registry["MusubiCheckpointStudio"] = MusubiCheckpointStudio
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: MusubiCheckpointStudio not available ({_e})")

# ── EasyCheckpointMerger ─────────────────────────────────────────────────
try:
    from .checkpoint_merger_node import EasyCheckpointMerger
    _node_registry["EasyCheckpointMerger"] = EasyCheckpointMerger
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: EasyCheckpointMerger not available ({_e})")

# ── EasyLoRAExtractor ────────────────────────────────────────────────────
try:
    from .engine.lora_extractor import EasyLoRAExtractor
    _node_registry["EasyLoRAExtractor"] = EasyLoRAExtractor
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: EasyLoRAExtractor not available ({_e})")

# ── EasyComponentExtractor ───────────────────────────────────────────────
try:
    from .engine.component_extractor import EasyComponentExtractor
    _node_registry["EasyComponentExtractor"] = EasyComponentExtractor
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: EasyComponentExtractor not available ({_e})")

# ── EasyComponentCombiner ────────────────────────────────────────────────
try:
    from .engine.component_combiner import EasyComponentCombiner
    _node_registry["EasyComponentCombiner"] = EasyComponentCombiner
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: EasyComponentCombiner not available ({_e})")

# ── EasyComponentMerger ──────────────────────────────────────────────────
try:
    from .engine.component_merger import EasyComponentMerger
    _node_registry["EasyComponentMerger"] = EasyComponentMerger
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: EasyComponentMerger not available ({_e})")

# ── EasyTextDisplay ─────────────────────────────────────────────────────
try:
    from .easy_text_display import EasyTextDisplay
    _node_registry["EasyTextDisplay"] = EasyTextDisplay
except ImportError as _e:
    print(f"⚠️ EasyLoRAMerger: EasyTextDisplay not available ({_e})")


# ==================== REGISTRATION ====================

NODE_CLASS_MAPPINGS = dict(_node_registry)

NODE_DISPLAY_NAME_MAPPINGS = {
    node_id: _DISPLAY_NAMES[node_id]
    for node_id in _node_registry
    if node_id in _DISPLAY_NAMES
}

# Startup summary — users can see how many nodes loaded in the ComfyUI console
_loaded = len(_node_registry)
_total = len(_DISPLAY_NAMES)
if _loaded == _total:
    print(f"✅ Easy LoRA Merger — all {_total} nodes loaded successfully.")
else:
    _missing = _total - _loaded
    print(f"⚠️ Easy LoRA Merger — {_loaded}/{_total} nodes loaded ({_missing} unavailable).")
