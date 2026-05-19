"""
Easy LoRA Merger - Node class mappings and display name mappings.
This module acts as a re-export hub for all remaining nodes.
"""

from .triple_merger_node import EasyLoRATripleMerger
from .engine.lora_studio_converter import MusubiLoraConverter
from .engine.musubi_checkpoint_studio import MusubiCheckpointStudio
from .checkpoint_merger_node import EasyCheckpointMerger
from .engine.lora_extractor import EasyLoRAExtractor
from .engine.component_extractor import EasyComponentExtractor
from .engine.component_combiner import EasyComponentCombiner
from .engine.component_merger import EasyComponentMerger

# ==================== REGISTRATION ====================

NODE_CLASS_MAPPINGS = {
    "EasyLoRATripleMerger": EasyLoRATripleMerger,
    "MusubiLoraConverter": MusubiLoraConverter,
    "MusubiCheckpointStudio": MusubiCheckpointStudio,
    "EasyCheckpointMerger": EasyCheckpointMerger,
    "EasyLoRAExtractor": EasyLoRAExtractor,
    "EasyComponentExtractor": EasyComponentExtractor,
    "EasyComponentCombiner": EasyComponentCombiner,
    "EasyComponentMerger": EasyComponentMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyLoRATripleMerger": "🎨 Easy LoRA Merger",
    "MusubiLoraConverter": "🛡️ Easy LoRA Studio",
    "MusubiCheckpointStudio": "🛡️ Easy Checkpoint Studio",
    "EasyCheckpointMerger": "🎨 Easy Checkpoint Merger",
    "EasyLoRAExtractor": "🔬 Easy LoRA Extractor",
    "EasyComponentExtractor": "🔧 Easy Component Extractor",
    "EasyComponentCombiner": "🧩 Easy Component Combiner",
    "EasyComponentMerger": "🧩 Easy Component Merger",
}
