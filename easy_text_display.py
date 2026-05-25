"""
Easy Text Display — ComfyUI Node Definition

Generic text display node that renders any STRING input as visible text on the
node body in the graph UI.  Uses the same ``OUTPUT_NODE + {"ui": {"text": …}}``
pattern as other ComfyUI text‑display nodes, so no external dependencies are
required.

Use cases
---------
- View forensic reports from any EasyLoRAMerger node (merger, baker, studio, …)
- Inspect output_path strings without opening the console
- Debug any STRING wire in your workflow
"""

import json


class EasyTextDisplay:
    """
    Display a STRING value as visible text on the node body.

    Inputs
    ------
    text : STRING
        Any text content to display.  Accepts multi‑line strings such as
        forensic reports, file paths, JSON dumps, etc.

    Outputs
    -------
    text : STRING
        Passthrough of the input — allows chaining to other nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": "Any text to display — forensic reports, paths, debug output…",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "display"
    OUTPUT_NODE = True
    CATEGORY = "LoRA/Utils"

    def display(self, text: str, unique_id=None, extra_pnginfo=None) -> tuple:
        """
        Forward the text to the UI and pass it through the output socket.

        The ``{"ui": {"text": …}}`` payload is picked up by the JavaScript
        extension which renders the content as a read‑only multiline widget
        on the node.
        """
        # ── Persist the text into the workflow JSON so it survives saves ──
        if unique_id is not None and extra_pnginfo is not None:
            try:
                # ComfyUI may pass extra_pnginfo as a list  [{"workflow": …}]
                # or as a bare dict  {"workflow": …}  depending on the version.
                workflow_container = (
                    extra_pnginfo[0]
                    if isinstance(extra_pnginfo, list)
                    else extra_pnginfo
                )
                workflow = (
                    workflow_container.get("workflow")
                    if isinstance(workflow_container, dict)
                    else None
                )
                if workflow is None:
                    print(
                        "⚠️ EasyTextDisplay: extra_pnginfo has no 'workflow' key "
                        "— cannot persist text"
                    )
                else:
                    node = next(
                        (
                            x
                            for x in workflow["nodes"]
                            if str(x["id"]) == str(unique_id)
                        ),
                        None,
                    )
                    if node:
                        node["widgets_values"] = [text]
            except Exception as exc:
                print(f"⚠️ EasyTextDisplay: failed to persist text ({exc})")

        return {"ui": {"text": [text]}, "result": (text,)}


# =====================================================================
# Registry helpers (used by easy_lora_merger.py)
# =====================================================================
NODE_CLASS_MAPPINGS = {
    "EasyTextDisplay": EasyTextDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyTextDisplay": "📋 Easy Text Display",
}
