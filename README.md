## 🛠️ Easy LoRA Merger: Complexity Made Simple

Models are evolving fast—between Z-Image, Klein 4B/9B, Flux 2 Dev, and SDXL, it's a lot to keep track of. I built this suite of nodes because I wanted a way to merge these different architectures without worrying about tensor dimensions or scaling math.

The goal isn't to be the most "complex" tool, but the most helpful. It does the heavy lifting in the background so you can just focus on the results.

---

## 🧪 Proof in the Results: Identity Blending

I tested this by merging characters with very different training backgrounds. By balancing the weights, you can achieve perfect hybrids that keep the identity of both. (workflow included)

✅ Z-Image (Turbo + Base Mix)  
✅ Flux Klein 9B Character Blends  
✅ SDXL Style + Character Mixes  

---

## 📦 The Toolbox (Node List)

- **Easy LoRA Merger**: The main engine. Connect your LoRAs, hit merge, and preview instantly.
- **Easy LoRA-Only Merger**: Perfect for chaining multiple merges or keeping your workspace clean.
- **🎨 Triple Merger (Experimental)**: Mix a character, an outfit, and an art style in one node.
- **🔄 Musubi LoRA Converter**: A specialized bridge for LoRAs trained in the Musubi tuner environment.
- **🔄 Z-Image Normalizer**: Fixes weight balancing when mixing Z-Image Turbo and Base models.
- **🔥 Base Model Baker (Experimental)**: Bakes your merge directly into the model for instant testing.

---

## 🚧 Status: Beta - Help Wanted!

This project works great for my workflows, but I need your help to push it further:

- **Flux 2 Dev Testing**: Feedback on newest Flux models
- **High-VRAM Users (24GB+)**: Try baking 9B models and report performance
- **SDXL Users**: Test style + character mixes

---

## 📝 Smart Diagnostics

The console status window shows:

✅ Green checkmarks when math is aligned  
📊 Layer-by-layer statistics  
⚠️ Clear warnings if something needs attention  

---

## 🚀 Get Started

- **Download**: https://github.com/Terpentinas/EasyLoRAMerger
- **Workflow**: Drag the attached image into ComfyUI to see the Identity Hybrid workflow
- **Questions?**: Open an issue on GitHub or comment below!

---

*Built because merging different LoRA formats shouldn't require a PhD in tensor math* 💪
