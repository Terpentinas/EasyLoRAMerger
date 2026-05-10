from .easy_lora_merger import NODE_CLASS_MAPPINGS as ELM_CLASS_MAPPINGS
from .easy_lora_merger import NODE_DISPLAY_NAME_MAPPINGS as ELM_DISPLAY_MAPPINGS
from .baker_node import SmartModelBaker

# This line is the bridge that tells ComfyUI to load your JS files
WEB_DIRECTORY = "./js"

# Merge mappings: start with existing, add SmartModelBaker
NODE_CLASS_MAPPINGS = {
    **ELM_CLASS_MAPPINGS,
    "SmartModelBaker": SmartModelBaker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **ELM_DISPLAY_MAPPINGS,
    "SmartModelBaker": "🔥 Easy LoRA Baker",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
