"""
Easy LoRA Merger - Engine Package
Contains the core merging algorithms, normalization, and serialization logic.

Module structure (baking_processor split):
  baking_processor.py              — Orchestration core (SmartBakingProcessor class)
  baking_processor_constants.py     — Registry data (prefix mappings, patterns)
  baking_processor_delta.py         — Delta reconstruction, alpha/rank scaling
  baking_processor_baking.py        — Bake methods, shape alignment, assembly
  baking_processor_matching.py      — Key matching, reverse map, strategy cascade
  baking_processor_sd15.py          — SD1.5 auto-scale channel safety only
                                      (old SD1.5 matching code removed — bridge handles it)
"""
