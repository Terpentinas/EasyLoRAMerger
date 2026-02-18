from .easy_lora_merger import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# This line is the bridge that tells ComfyUI to load your JS files
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']