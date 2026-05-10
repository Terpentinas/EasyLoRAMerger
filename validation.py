"""
Model validation and metadata merging for Easy LoRA Merger.
"""

import re
from typing import Dict, List, Set
import torch

try:
    from .utils import safe_get_rank
except ImportError:
    from utils import safe_get_rank


class ModelValidator:
    """Validates LoRA model compatibility and integrity."""
    
    @staticmethod
    def detect_model_type(state_dict: Dict[str, torch.Tensor]) -> str:
        """Detect LoRA model architecture type."""
        keys = list(state_dict.keys())
        if not keys:
            return "unknown"
        
        # Check for specific patterns
        if any("diffusion_model.double_blocks" in k for k in keys):
            return "flux_klein"
        elif any("diffusion_model.single_blocks" in k for k in keys):
            return "flux_klein_4b"
        elif any("input_blocks" in k or "output_blocks" in k for k in keys):
            return "sd15_or_sdxl"
        elif any("transformer" in k for k in keys):
            return "transformer_based"
        elif any("lora_unet_" in k for k in keys):
            return "musubi_klein"
        
        return "unknown"
    
    @staticmethod
    def extract_layer_name(key: str) -> str:
        """Extract base layer name from LoRA key."""
        patterns = [
            r'(.*)\.lora_A\.weight$',
            r'(.*)\.lora_B\.weight$',
            r'(.*)\.lora\.down\.weight$',
            r'(.*)\.lora\.up\.weight$',
            r'(.*)\.alpha$',
            r'(.*)\.lora_down\.weight$',
            r'(.*)\.lora_up\.weight$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, key)
            if match:
                return match.group(1)
        return key
    
    @staticmethod
    def validate_compatibility(sd_a: Dict[str, torch.Tensor], 
                              sd_b: Dict[str, torch.Tensor]) -> bool:
        """
        Validate if two LoRAs are compatible for merging.
        
        Returns:
            True if compatible, raises ValueError if not
        """
        # Extract layer names (without lora_A/lora_B suffix)
        layers_a = {ModelValidator.extract_layer_name(k) for k in sd_a.keys()}
        layers_b = {ModelValidator.extract_layer_name(k) for k in sd_b.keys()}
        
        # Check model types
        type_a = ModelValidator.detect_model_type(sd_a)
        type_b = ModelValidator.detect_model_type(sd_b)
        
        if type_a != type_b:
            print(f"⚠️ Different model architectures: {type_a} vs {type_b}")
            print("   Attempting merge anyway, but results may be unstable")
        
        # Check layer overlap
        overlap = layers_a.intersection(layers_b)
        total_layers = min(len(layers_a), len(layers_b))
        
        if total_layers == 0:
            raise ValueError("No layers found in one or both models")
        
        overlap_ratio = len(overlap) / total_layers
        
        if overlap_ratio < 0.3:
            raise ValueError(
                f"Models have only {overlap_ratio:.1%} layer overlap "
                f"({len(overlap)}/{total_layers}) - likely incompatible"
            )
        
        if overlap_ratio < 0.7:
            print(f"⚠️ Low layer overlap: {overlap_ratio:.1%}")
            print("   Some layers may not be merged")
        
        # Check rank consistency
        ranks_a = ModelValidator.get_model_ranks(sd_a)
        ranks_b = ModelValidator.get_model_ranks(sd_b)
        
        if ranks_a and ranks_b:
            avg_rank_a = sum(ranks_a) / len(ranks_a)
            avg_rank_b = sum(ranks_b) / len(ranks_b)

            if abs(avg_rank_a - avg_rank_b) > max(avg_rank_a, avg_rank_b) * 0.5:
                print(f"⚠️ Significant rank difference: {avg_rank_a:.1f} vs {avg_rank_b:.1f}")

        # Check Text Encoder (TE) key compatibility
        te_keys_a = [k for k in sd_a if "lora_te" in k]
        te_keys_b = [k for k in sd_b if "lora_te" in k]

        if te_keys_a and not te_keys_b:
            print("⚠️ LoRA A has Text Encoder keys but LoRA B doesn't")
            print("   TE keys will be preserved from A only")
        elif te_keys_b and not te_keys_a:
            print("⚠️ LoRA B has Text Encoder keys but LoRA A doesn't")
            print("   TE keys will be preserved from B only")
        elif te_keys_a and te_keys_b:
            # Check if TE formats match (te1 vs te)
            te_format_a = "te1" if any("te1" in k for k in te_keys_a) else "te"
            te_format_b = "te1" if any("te1" in k for k in te_keys_b) else "te"
            if te_format_a != te_format_b:
                print(f"⚠️ Different TE formats: {te_format_a} vs {te_format_b}")
                print("   Attempting merge but results may be unstable")

        return True
    
    @staticmethod
    def get_model_ranks(state_dict: Dict[str, torch.Tensor]) -> List[int]:
        """Get all unique ranks in a model."""
        ranks = set()
        for key, tensor in state_dict.items():
            if len(tensor.shape) >= 2:
                rank = safe_get_rank(tensor, key)
                ranks.add(rank)
        return list(ranks)


class MetadataMerger:
    """Handles merging of metadata from multiple LoRA files."""
    
    @staticmethod
    def merge(meta_a: Dict[str, str], meta_b: Dict[str, str], 
              mode: str = "merge_basic") -> Dict[str, str]:
        """
        Merge metadata from two LoRA files with conflict resolution.
        
        Args:
            meta_a: Metadata from first LoRA
            meta_b: Metadata from second LoRA
            mode: Merge strategy
        
        Returns:
            Merged metadata dictionary
        """
        if mode == "none":
            return {}
        elif mode == "preserve_a":
            merged = {f"lora_a_{k}": v for k, v in meta_a.items()}
            # Also include modelspec fields from meta_b (architecture may differ)
            for k, v in meta_b.items():
                if k.startswith('modelspec.'):
                    merged[k] = v
            return merged
        elif mode == "preserve_b":
            merged = {f"lora_b_{k}": v for k, v in meta_b.items()}
            for k, v in meta_a.items():
                if k.startswith('modelspec.'):
                    merged[k] = v
            return merged
        
        # merge_basic mode
        merged = {}
        conflicts = []
        
        # Important fields to preserve separately
        important_fields = [
            "ss_base_model", "ss_sd_model_name", "ss_network_module",
            "ss_network_dim", "ss_network_alpha", "ss_training_started_at"
        ]
        
        # Merge with conflict detection
        all_keys = set(meta_a.keys()) | set(meta_b.keys())
        
        # Preserve architecture‑critical modelspec fields (keep original key)
        modelspec_keys = {k for k in all_keys if k.startswith('modelspec.')}
        for key in modelspec_keys:
            # Prefer LoRA A (Avatar) for architecture identification
            if key in meta_a:
                merged[key] = meta_a[key]
            elif key in meta_b:
                merged[key] = meta_b[key]
            # If both present and different, we already picked A; log if they differ
            if key in meta_a and key in meta_b and meta_a[key] != meta_b[key]:
                conflicts.append((key, meta_a[key], meta_b[key]))
        # Remove processed keys from all_keys to avoid double handling
        all_keys -= modelspec_keys
        
        for key in all_keys:
            val_a = meta_a.get(key)
            val_b = meta_b.get(key)
            
            if val_a is not None and val_b is not None:
                if val_a != val_b:
                    conflicts.append((key, val_a, val_b))
                    
                    # Handle important fields specially
                    if key in important_fields:
                        merged[f"lora_a_{key}"] = val_a
                        merged[f"lora_b_{key}"] = val_b
                    else:
                        # Try to merge values
                        try:
                            # Numeric values: average
                            num_a = float(val_a)
                            num_b = float(val_b)
                            merged[f"{key}_avg"] = str((num_a + num_b) / 2)
                            merged[f"{key}_a"] = val_a
                            merged[f"{key}_b"] = val_b
                        except ValueError:
                            # String values: concatenate
                            merged[f"{key}_a"] = val_a
                            merged[f"{key}_b"] = val_b
                else:
                    merged[key] = val_a
            elif val_a is not None:
                if key in important_fields:
                    merged[f"lora_a_{key}"] = val_a
                else:
                    merged[f"{key}_a"] = val_a
            elif val_b is not None:
                if key in important_fields:
                    merged[f"lora_b_{key}"] = val_b
                else:
                    merged[f"{key}_b"] = val_b
        
        if conflicts:
            print(f"📝 Resolved {len(conflicts)} metadata conflicts")
        
        return merged