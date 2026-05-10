# 🛠️ Easy LoRA Merger Suite
*Complexity Made Simple — Merge Anything*

The Easy LoRA Merger Suite gives you a complete toolkit to merge, convert, extract, bake, and
audit LoRAs and checkpoints — without worrying about tensor dimensions, sparsity mismatches,
or arcane scaling math.

Every node is built with the same philosophy: **powerful defaults when you need simplicity,
full control when you need precision.**

---

# 🔍 All Nodes at a Glance

![All Nodes Overview](assets/nodes.png)

> The complete node suite — drag, connect, and merge.

---

# 📦 Node Catalog

| Node | Display Name | Purpose |
|------|-------------|---------|
| 🎨 | **Easy LoRA Merger** | Merge 2–3 LoRAs with 15+ methods, auto‑format detection, smart defaults, and forensic reports. The core workhorse. |
| 🛡️ | **Easy LoRA Studio** | Universal LoRA analyzer and converter. Transforms between Standard WebUI, Comfy Native, and Forge‑Optimized formats. SVD compression, TE scaling, weight adjustment. |
| 🛡️ | **Easy Checkpoint Studio** | Precision surgery for full checkpoints — precision casting FP8/BF16/FP32, component stripping VAE/TE/CLIP, SVD compression, and structural remapping. |
| 🎨 | **Easy Checkpoint Merger** | Merge 2–3 full model checkpoints with Weight Block Map component‑wise scaling. Streaming engine for low RAM usage. |
| 🔬 | **Easy LoRA Extractor** | Extract a LoRA from the delta between a base and fine‑tuned checkpoint. SVD decomposition with auto‑energy rank selection. |
| 🔥 | **Easy LoRA Baker** | Bake a LoRA (or merged LoRA) directly into a full checkpoint at the tensor level. Produces MODEL+CLIP+VAE with RAM Guard fallback. |

---

# 🎯 Merge Methods

| Method | Best for | Works on Checkpoints? |
|--------|----------|:---------------------:|
| **linear** | Simple weighted average — the safe starting point for any merge | ✅ Yes |
| **cross** | Blending with a cross‑magnitude interaction term for richer mixes | ✅ Yes |
| **ties_strict** | Conflicting styles — only keeps weights where both sources agree | ⚠️ Limited (all-positive weights reduce effect) |
| **ties_gentle** | Softer version of TIES — applies only to strong disagreements | ⚠️ Limited (all-positive weights reduce effect) |
| **ties_contrast** | Amplify differences between two sources, mute agreements | ⚠️ Limited (best on LoRA deltas) |
| **dare_lite** | Random dropout — creates sparse, stochastic blends | ❌ LoRA only (risky on checkpoint weights) |
| **dare_rescale** | Random dropout with rescaling — preserves magnitude distribution | ❌ LoRA only (risky on checkpoint weights) |
| **magnitude** | Keep the stronger signal per-element from either source | ✅ Yes |
| **subtract** | Remove unwanted features by subtracting one source from another | ✅ Yes |
| **feature_mix** | Preserve unique features from each source — great for style + character | ✅ Yes |
| **slerp** | Smooth spherical interpolation between two vectors | ✅ Yes (2‑way only) |
| **svd_preserve** | SVD‑based rank reduction — keeps core structure while reducing noise | ✅ Yes |
| **block_swap** | Swap blocks between sources using a seeded random pattern | ✅ Yes |
| **noise_aware** | Suppress small noise values before merging for cleaner results | ✅ Yes |
| **gradient_alignment** | Weight contributions by directional similarity between sources | ✅ Yes |

> 💡 **Tip:** When in doubt, start with `linear` or `magnitude` — they're the most versatile and work well across LoRAs and checkpoints alike.

---

# 🧠 Smart Diagnostics

Every merge node outputs a detailed forensic report with:

- ✅ **Alignment verification** — cross‑architecture key matching stats
- 📊 **Layer‑by‑layer statistics** — energy distribution, sparsity, component breakdown
- ⚠️ **Clear warnings** — mismatched trainers, conflicting trigger words, density risks
- 🔍 **Sparsity + scaling reports** — magnitude distribution per layer

Console output keeps you informed at every step without overwhelming.

---

# 🏗️ Supported Architectures (aspirational — not all fully tested)

| Architecture | Status |
|-------------|--------|
| Flux.1‑Dev / Flux.1‑S | Tested |
| Flux Klein 4B / 9B | Tested |
| Z‑Image Turbo / Base | Tested |
| SDXL | Tested |
| SD1.5 | Tested |
| Anima | Early support |
| Lumina 2 | Early support |
| SD3 | Experimental |

---

# 🚀 Get Started

- **Download:** [github.com/Terpentinas/EasyLoRAMerger](https://github.com/Terpentinas/EasyLoRAMerger)
- **Install:** Drop into `ComfyUI/custom_nodes/` and restart ComfyUI.
- **Explore:** Drag [`assets/nodes.png`](assets/nodes.png) into the workflow area to see the suite in action.
- **Experiment:** Connect a *Easy LoRA Merger* → *Easy LoRA Baker* pipeline, or use *Easy Checkpoint Studio* to shrink a 12GB checkpoint to FP8.
- **Feedback:** Open an issue on GitHub — contributions and ideas welcome.

---

*Built because merging different model formats shouldn't require a PhD in tensor math.* 💪
