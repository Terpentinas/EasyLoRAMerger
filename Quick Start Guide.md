🚀 Easy LoRA Merger: Quick Start Guide
This guide will help you get up and running with the Easy LoRA Merger for Flux Klein 4B/9B and other formats.

1. Basic Setup
Select LoRA A & B: You can choose .safetensors files from your LoRA folder or plug in existing LoRA data from other nodes.

Set Weights:

Weight A: The strength of the first LoRA (default 1.0).

Weight B: The strength of the second LoRA (default 1.0).

Choose Method:

linear: Standard blending (best for most cases).

ties_gentle: Reduces "ghosting" when models conflict.

dare_rescale: High-energy merging that keeps details sharp.

2. ⚠️ Important Weight Warning
The merged LoRA may require weight adjustment depending on your results. Since you're combining two LoRAs, the effective strength can vary significantly.

**If results look over-saturated or "burned":** The merge is too strong - try lower weights like 0.5-0.7 for each LoRA during merging.

**If results look too subtle or weak:** The merge is too soft - try higher weights like 1.2-1.5 for each LoRA during merging.

Recommendation: Start with both weights at 1.0, then adjust based on your results. Experiment with the 0.5-1.5 range to find the sweet spot for your specific combination.

3. Save vs. Experiment
Save Trigger (OFF): The merged LoRA is stored in a temp folder. It works perfectly for your current session but won't clutter your hard drive permanently.

Save Trigger (ON): The file is saved permanently to your LoRA folder using the name you provided in the filename field.

4. Performance Settings
Device: Leave on auto to use your GPU for speed, or switch to cpu if you are running low on VRAM (8GB or less).

Precision: Use bfloat16 for the best balance of speed and quality on modern NVIDIA cards.

5. Output Types
Easy LoRA Merger: Outputs a MODEL and CLIP that you can plug directly into your Sampler.

Easy LoRA-Only Merger: Outputs a LORA tuple, which is perfect for "chaining" multiple merges together before finally applying them to a model.
