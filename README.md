# Easy LoRA Merger for ComfyUI

*A simple tool for experimenting with LoRA merging - created mainly for Flux Klein 4B/9B*

[![GitHub](https://img.shields.io/github/license/Terpentinas/EasyLoRAMerger)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-green)](https://github.com/comfyanonymous/ComfyUI)

> ⚠️ **Experimental Tool - Testing Invited!** ⚠️
> 
> This started as a personal project for merging Flux Klein models. It *might* work with other formats, but I need your help testing!

## 🎯 What This Is

A simple ComfyUI node that lets you **experiment with merging two LoRAs** using different methods. It was born from my need to merge Flux Klein 4B models, but I've tried to make it work with other formats too.

## 🤔 What This Is NOT

- **Not** a production-ready, perfect solution
- **Not** guaranteed to work with all LoRA formats
- **Not** extensively tested with SD1.5/SDXL (but seems to work)
- **Not** created by an AI expert (just a hobbyist!)

## ✨ What Works (Probably)

| Format | Status | Notes |
|--------|---------|-------|
| **Flux Klein 4B** | ✅ Tested | The main reason this exists! |
| **Flux Klein 9B** | ✅ Seems OK | Tested with a few models |
| **Z-Image Turbo (AI-Toolkit)** | ✅ Seems OK | Tested with a few models |
| **Pony Diffusion** | ✅ Should work | Recent fix applied |
| **SD1.5/SDXL Kohya** | ⚠️ Maybe | Not heavily tested |
| **Z-Image Base** | ❓ Unknown | Different architecture |
| **LyCORIS/LoCon** | ❓ Untested | Might need adjustments |

## 🚀 Quick Start

1. **Download** the ZIP or clone:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Terpentinas/EasyLoRAMerger

- [Quick Start Guide](Quick_Start_Guide.md)

![Easy LoRA Merger](https://raw.githubusercontent.com/Terpentinas/EasyLoRAMerger/refs/heads/main/images/node.jpg)
