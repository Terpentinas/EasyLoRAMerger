## Two Simple Nodes

- **Easy LoRA Merger**: Merge and immediately use
- **Easy LoRA-Only Merger**: Merge and save as LoRA file

## Quick Start

1. Load Easy LoRA Merger
2. Choose two LoRAs
3. Try:
   - Method: `linear`
   - Weights: `1.0` and `1.0`
   - Density: `1.0`
4. Click Queue Prompt!

## Choose Method:

   - linear: Standard blending (best for most cases).
   - ties_gentle: Reduces "ghosting" when models conflict.
   - dare_rescale: High-energy merging that keeps details sharp.
   - ...

## ⚠️ Important Weight Warning
   - The merged LoRA may require weight adjustment depending on your results. Since you're combining two LoRAs, the effective strength can vary significantly.
   - **If results look over-saturated or "burned":** The merge is too strong - try lower weights like 0.5-0.7 for each LoRA during merging.
   - **If results look too subtle or weak:** The merge is too soft - try higher weights like 1.2-1.5 for each LoRA during merging.
   - Recommendation: Start with both weights at 1.0, then adjust based on your results. Experiment with the 0.5-1.5 range to find the sweet spot for your specific combination.
