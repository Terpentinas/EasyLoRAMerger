"""
SD1.5-specific safety for SmartBakingProcessor.

NOTE: The Diffusers-Bridge Pipeline (engine/diffusers_bridge.py) now handles
the primary SD1.5 baking path. The auto-scale safety check was inlined into
baking_processor.py using detect_native_sd15_checkpoint().

This file is retained as a placeholder for any future SD1.5-specific logic.
"""
