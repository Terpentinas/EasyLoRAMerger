"""
Diffusers-Bridge for SD1.5: match/bake in Diffusers key-space, then convert back.

The existing pipeline normalizes LoRA keys from Diffusers → native format, losing
~175 of 396 modules during key conversion. This bridge eliminates that loss by:

1. Keeping LoRA keys in Diffusers format (no block-structure conversion)
2. Extending the reverse_key_map to map Diffusers-format keys → native ckpt keys
3. All 396 LoRA modules match 1:1 → all 258 weight keys get baked

Non-SD1.5 / non-Diffusers-LoRA cases are detected and EXCLUDED (fall through to
the existing pipeline unchanged).

Reference: Kohya's convert_unet_state_dict_to_sd() at
  model_util.py:674-767 (sd-scripts/library/model_util.py)
"""

import re
from typing import Dict, List, Tuple, Optional, Set

import torch

from ..utils import ProgressTracker


# =====================================================================
# SHARED HELPERS (for deduplication)
# =====================================================================

def _cleanup_dots(key: str) -> str:
    """Remove double dots and '._.' artifacts from key paths."""
    while '..' in key:
        key = key.replace('..', '.')
    return key.replace('._.', '.')


def _apply_resnet_rename_native_to_diff(inner: str) -> str:
    """Apply native→Diffusers ResNet submodule renaming."""
    for native_sub, diff_sub in _NATIVE_RESNET_TO_DIFF:
        inner = inner.replace(native_sub, diff_sub)
    return inner


# =====================================================================
# DETECTION GATES
# =====================================================================

# Known SD1.5 native key signatures (must all be present for detection)
_SD15_NATIVE_SIGNATURES: List[str] = [
    'model.diffusion_model.input_blocks.0.0.weight',
    'model.diffusion_model.time_embed.0.weight',
    'model.diffusion_model.out.0.weight',
    'model.diffusion_model.out.2.weight',
]

# Keys that indicate non-SD1.5 (SDXL, Flux, etc.)
_NON_SD15_INDICATORS: List[str] = [
    'conditioner.embedders.',           # SDXL
    'model.diffusion_model.transformer.double_blocks.',  # Flux
    'model.diffusion_model.transformer.single_blocks.',   # Flux
    'model.diffusion_model.layers.',     # Z-Image/Musubi
]


def detect_native_sd15_checkpoint(ckpt_sd: Dict[str, torch.Tensor]) -> bool:
    """
    Check if checkpoint is a native SD1.5 format.

    Returns True when ALL SD1.5 signature keys are present AND
    no non-SD1.5 indicators are found.
    """
    # Must have all SD1.5 signature keys
    has_sigs = all(
        any(k.startswith(sig) for k in ckpt_sd)
        for sig in _SD15_NATIVE_SIGNATURES
    )
    if not has_sigs:
        return False

    # Must NOT have any non-SD1.5 indicators
    has_non_sd15 = any(
        any(indicator in k for k in ckpt_sd)
        for indicator in _NON_SD15_INDICATORS
    )
    if has_non_sd15:
        return False

    return True


def detect_diffusers_sd15_lora(lora_sd: Dict[str, torch.Tensor]) -> bool:
    """
    Check if LoRA uses SD1.5 Diffusers-style keys.

    Returns True when:
    - 'lora_unet_down_blocks_' is found in at least one key
    - 'lora_unet_up_blocks_' is found in at least one key
    - AND the majority of UNet-prefixed keys use Diffusers format
      (not e.g. lora_unet_input_blocks_ which is native format)
    """
    has_down_blocks = any('lora_unet_down_blocks_' in k for k in lora_sd)
    has_up_blocks = any('lora_unet_up_blocks_' in k for k in lora_sd)

    if not (has_down_blocks and has_up_blocks):
        return False

    # Count Diffusers-style vs native-style UNet keys
    diffusers_count = sum(1 for k in lora_sd
                          if k.startswith('lora_unet_')
                          and ('down_blocks_' in k or 'up_blocks_' in k or 'mid_block_' in k))
    native_count = sum(1 for k in lora_sd
                       if k.startswith('lora_unet_')
                       and ('input_blocks_' in k or 'output_blocks_' in k or 'middle_block_' in k))

    # Must be majority Diffusers-style (or exclusively Diffusers)
    total_unet = diffusers_count + native_count
    if total_unet == 0:
        return False

    return diffusers_count >= native_count


# =====================================================================
# LoRA NORMALIZATION (Preserving Diffusers Block Structure)
# =====================================================================

def normalize_diffusers_preserving(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Normalize LoRA keys preserving Diffusers block structure.

    Unlike identity_normalize() which calls universal_normalize() → _convert_sd15_diffusers_key()
    (converting down_blocks→input_blocks, attentions→1, etc.), this function:

    - Strips 'lora_unet_' / 'lora_te_' prefixes
    - Converts underscore-separated paths to dotted paths
    - Leaves Diffusers block names intact (down_blocks, up_blocks, mid_block, attentions)
    - Preserves lora_down/lora_up suffixes for delta reconstruction

    Example:
        Input:  lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
        Output: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight

    Returns:
        (normalized_dict, key_map)
    """
    normalized: Dict[str, torch.Tensor] = {}
    key_map: Dict[str, str] = {}

    for orig_key, tensor in state_dict.items():
        new_key = orig_key

        # ---- Step 1: Strip known prefixes ----
        if new_key.startswith('lora_unet_'):
            new_key = new_key[len('lora_unet_'):]
        elif new_key.startswith('lora_te_'):
            new_key = new_key[len('lora_te_'):]
            # Convert text_model_encoder_layers_N_ → text_model.encoder.layers.N.
            new_key = re.sub(
                r'text_model_encoder_layers_(\d+)_',
                r'text_model.encoder.layers.\1.',
                new_key
            )
            new_key = re.sub(
                r'self_attn_(q|k|v|out)_proj',
                r'self_attn.\1_proj',
                new_key
            )
            new_key = re.sub(r'mlp_fc(\d)', r'mlp.fc\1', new_key)
        else:
            # Pass-through for unknown formats (alpha keys, etc.)
            normalized[orig_key] = tensor
            key_map[orig_key] = orig_key
            continue

        # ---- Step 2: Convert underscore numeric separators to dots ----
        # _N_ → .N.  (handles down_blocks_0_attentions → down_blocks.0.attentions)
        new_key = re.sub(r'_(\d+)_', r'.\1.', new_key)

        # ---- Step 2b: Convert mid_block_attentions → mid_block.attentions ----
        # The regex in Step 2 only handles _N_ patterns (underscore-digit-underscore).
        # mid_block_attentions has an underscore between words with NO digits between
        # segments, so Step 2 leaves it as-is. This conversion is needed because
        # build_diffusers_reverse_map() generates 'mid_block.attentions.{idx}.{subpath}'
        # keys, and the normalized LoRA key must match exactly.
        new_key = new_key.replace('mid_block_attentions', 'mid_block.attentions')

        # ---- Step 3: Handle trailing numeric segments ----
        # E.g., to_out_0 → to_out.0
        new_key = re.sub(r'_(\d+)(\.)', r'.\1\2', new_key)

        # ---- Step 4: Convert attention projection names ----
        # attn1_to_q → attn1.to_q
        new_key = re.sub(r'attn(\d)_to_q\b', r'attn\1.to_q', new_key)
        new_key = re.sub(r'attn(\d)_to_k\b', r'attn\1.to_k', new_key)
        new_key = re.sub(r'attn(\d)_to_v\b', r'attn\1.to_v', new_key)
        new_key = re.sub(r'attn(\d)_to_out_(\d+)', r'attn\1.to_out.\2', new_key)
        new_key = re.sub(r'attn(\d)_to_out\b', r'attn\1.to_out', new_key)
        # Catch any remaining to_out_N
        new_key = re.sub(r'to_out_(\d+)', r'to_out.\1', new_key)

        # ---- Step 5: Convert feed-forward network names ----
        # ff_net → ff.net (with numeric variant)
        new_key = re.sub(r'\bff_net_(\d+)\b', r'ff.net.\1', new_key)
        new_key = re.sub(r'\bff_net\b', r'ff.net', new_key)

        # ---- Step 6: Cleanup ----
        new_key = _cleanup_dots(new_key)

        normalized[new_key] = tensor
        key_map[new_key] = orig_key

    return normalized, key_map


# =====================================================================
# CHECKPOINT CONVERSION: Native SD1.5 ↔ Diffusers
# =====================================================================

# Direct 1:1 mappings (SD name → Diffusers name)
# From Kohya's unet_conversion_map
NATIVE_TO_DIFFUSERS_DIRECT: List[Tuple[str, str]] = [
    # (native, diffusers)
    ('input_blocks.0.0.weight', 'conv_in.weight'),
    ('input_blocks.0.0.bias',   'conv_in.bias'),
    ('time_embed.0.weight',     'time_embedding.linear_1.weight'),
    ('time_embed.0.bias',       'time_embedding.linear_1.bias'),
    ('time_embed.2.weight',     'time_embedding.linear_2.weight'),
    ('time_embed.2.bias',       'time_embedding.linear_2.bias'),
    ('out.0.weight',            'conv_norm_out.weight'),
    ('out.0.bias',              'conv_norm_out.bias'),
    ('out.2.weight',            'conv_out.weight'),
    ('out.2.bias',              'conv_out.bias'),
]

# Reverse: Diffusers name → SD name (for converting back)
DIFFUSERS_TO_NATIVE_DIRECT: Dict[str, str] = {
    diff: native for native, diff in NATIVE_TO_DIFFUSERS_DIRECT
}

# Block prefix mapping (SD prefix → Diffusers prefix)
# Generated same way as Kohya's unet_conversion_map_layer but with
# (sd_part, hf_part) ordering
def _build_block_prefix_maps() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Build prefix maps for native↔diffusers conversion.

    Returns:
        native_to_diff: list of (native_prefix, diffusers_prefix)
        diff_to_native: list of (diffusers_prefix, native_prefix)
    """
    native_to_diff: List[Tuple[str, str]] = []
    diff_to_native: List[Tuple[str, str]] = []

    for i in range(4):  # down/up block index
        # --- Down blocks (resnets + attentions) ---
        for j in range(2):  # resnet/attention per block
            # ResNet
            native_res = f'input_blocks.{3 * i + j + 1}.0.'
            diff_res = f'down_blocks.{i}.resnets.{j}.'
            native_to_diff.append((native_res, diff_res))
            diff_to_native.append((diff_res, native_res))

            # Attention (only for i < 3)
            if i < 3:
                native_atn = f'input_blocks.{3 * i + j + 1}.1.'
                diff_atn = f'down_blocks.{i}.attentions.{j}.'
                native_to_diff.append((native_atn, diff_atn))
                diff_to_native.append((diff_atn, native_atn))

        # --- Up blocks (resnets + attentions) ---
        for j in range(3):  # 3 resnets per up block
            # ResNet
            native_res = f'output_blocks.{3 * i + j}.0.'
            diff_res = f'up_blocks.{i}.resnets.{j}.'
            native_to_diff.append((native_res, diff_res))
            diff_to_native.append((diff_res, native_res))

            # Attention (only for i > 0)
            if i > 0:
                native_atn = f'output_blocks.{3 * i + j}.1.'
                diff_atn = f'up_blocks.{i}.attentions.{j}.'
                native_to_diff.append((native_atn, diff_atn))
                diff_to_native.append((diff_atn, native_atn))

        # --- Downsamplers (only for i < 3) ---
        if i < 3:
            native_down = f'input_blocks.{3 * (i + 1)}.0.op.'
            diff_down = f'down_blocks.{i}.downsamplers.0.conv.'
            native_to_diff.append((native_down, diff_down))
            diff_to_native.append((diff_down, native_down))

            # --- Upsamplers (only for i < 3) ---
            native_up = f'output_blocks.{3 * i + 2}.{1 if i == 0 else 2}.'
            diff_up = f'up_blocks.{i}.upsamplers.0.'
            native_to_diff.append((native_up, diff_up))
            diff_to_native.append((diff_up, native_up))

    # --- Mid block ---
    # Attention
    native_to_diff.append(('middle_block.1.', 'mid_block.attentions.0.'))
    diff_to_native.append(('mid_block.attentions.0.', 'middle_block.1.'))
    # ResNets
    for j in range(2):
        native_res = f'middle_block.{2 * j}.'
        diff_res = f'mid_block.resnets.{j}.'
        native_to_diff.append((native_res, diff_res))
        diff_to_native.append((diff_res, native_res))

    return native_to_diff, diff_to_native


# Build prefix maps once at module load
_NATIVE_TO_DIFF_PREFIXES, _DIFF_TO_NATIVE_PREFIXES = _build_block_prefix_maps()

# ResNet submodule renaming (native SD in_layers/out_layers → Diffusers norm/conv)
# From Kohya's unet_conversion_map_resnet (but inverse direction)
_NATIVE_RESNET_TO_DIFF: List[Tuple[str, str]] = [
    # (native, diffusers)
    ('in_layers.0', 'norm1'),
    ('in_layers.2', 'conv1'),
    ('out_layers.0', 'norm2'),
    ('out_layers.3', 'conv2'),
    ('emb_layers.1', 'time_emb_proj'),
    ('skip_connection', 'conv_shortcut'),
]

# Reverse: Diffusers resnet names → native resnet names
_DIFF_RESNET_TO_NATIVE: List[Tuple[str, str]] = [
    (diff, native) for native, diff in _NATIVE_RESNET_TO_DIFF
]


def native_to_diffusers_sd15(ckpt_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert native SD1.5 UNet state dict to Diffusers format.

    Only converts 'model.diffusion_model.*' keys. All other keys
    (TE, VAE, etc.) pass through unchanged.

    This is the INVERSE of Kohya's convert_unet_state_dict_to_sd().
    """
    converted: Dict[str, torch.Tensor] = {}

    for key, tensor in ckpt_sd.items():
        if not key.startswith('model.diffusion_model.'):
            # Non-UNet keys pass through unchanged
            converted[key] = tensor
            continue

        # Strip the model.diffusion_model. prefix
        inner = key[len('model.diffusion_model.'):]

        # ---- Step 1: Check direct 1:1 mappings ----
        for native_name, diff_name in NATIVE_TO_DIFFUSERS_DIRECT:
            if inner == native_name:
                converted[diff_name] = tensor
                break
            if inner == native_name.replace('.weight', '.bias'):
                # Some edge cases
                pass
        else:
            # ---- Step 2: Apply block prefix replacements (native→diff) ----
            for native_prefix, diff_prefix in _NATIVE_TO_DIFF_PREFIXES:
                if inner.startswith(native_prefix):
                    inner = diff_prefix + inner[len(native_prefix):]
                    break

            # ---- Step 3: Apply resnet submodule renaming (native→diff) ----
            inner = _apply_resnet_rename_native_to_diff(inner)

            # Clean up any double dots
            inner = _cleanup_dots(inner)

            converted[inner] = tensor

    return converted


def _convert_single_diffusers_to_native(diff_key: str) -> str:
    """
    Convert a single Diffusers-format UNet key to native format.

    IMPORTANT: ResNet submodule renaming (norm1->in_layers.0, etc.) is scoped
    to keys with 'resnets.' in the ORIGINAL diff_key path. This is because the
    prefix replacement (Step 2) strips 'resnets.' from the key, making it
    impossible to detect resnet membership afterwards. This mirrors Kohya's
    approach in model_util.py::convert_unet_state_dict_to_sd() which only calls
    renew_resnet_paths() for resnet-block keys.
    """
    inner = diff_key

    # Determine BEFORE prefix replacement whether this is a resnet key.
    # Diffusers resnet paths contain 'resnets.' (e.g., down_blocks.0.resnets.0.norm1).
    # Attention paths contain 'attentions.' (e.g., down_blocks.0.attentions.0.transformer_blocks.0.norm1).
    # The prefix replacement (Step 2) removes 'resnets.' from the path, so we
    # must check before that step.
    is_resnet_key = 'resnets.' in diff_key

    # Step 1: Check direct 1:1 reverse mappings
    if inner in DIFFUSERS_TO_NATIVE_DIRECT:
        return DIFFUSERS_TO_NATIVE_DIRECT[inner]

    # Step 2: Apply block prefix replacements (diff→native)
    for diff_prefix, native_prefix in _DIFF_TO_NATIVE_PREFIXES:
        if inner.startswith(diff_prefix):
            inner = native_prefix + inner[len(diff_prefix):]
            break

    # Step 3: Apply resnet submodule renaming (diff→native)
    # Only applies to resnet-block keys (checked against original diff_key).
    if is_resnet_key:
        for diff_sub, native_sub in _DIFF_RESNET_TO_NATIVE:
            inner = inner.replace(diff_sub, native_sub)

    # Cleanup
    inner = _cleanup_dots(inner)

    return inner


def diffusers_to_native_sd15(ckpt_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert Diffusers-format UNet state dict back to native SD1.5 format.

    This implements the SAME conversion as Kohya's convert_unet_state_dict_to_sd()
    (model_util.py:674-767), ensuring the output matches Kohya exactly.

    Non-UNet keys pass through unchanged.
    """
    converted: Dict[str, torch.Tensor] = {}

    for key, tensor in ckpt_sd.items():
        # Check if this is a Diffusers UNet key (no model.diffusion_model. prefix)
        # Note: 'out.' prefix does NOT match conv_norm_out / conv_out (they start with conv_)
        is_diffusers_unet = any(
            key.startswith(prefix)
            for prefix in [
                'conv_in.', 'time_embedding.',
                'conv_norm_out.', 'conv_out.',
                'down_blocks.', 'up_blocks.', 'mid_block.',
            ]
        )
        if not is_diffusers_unet:
            # Non-UNet keys pass through
            converted[key] = tensor
            continue

        # Convert to native format with model prefix
        native_inner = _convert_single_diffusers_to_native(key)
        native_key = f'model.diffusion_model.{native_inner}'
        converted[native_key] = tensor

    return converted


# =====================================================================
# KEY MATCHING (1:1 in Diffusers space)
# =====================================================================

def build_diffusers_reverse_map(
    ckpt_sd: Dict[str, torch.Tensor],
) -> Dict[str, str]:
    """
    Build a reverse map from Diffusers-format keys to native checkpoint keys.

    For each native SD1.5 UNet key, registers Diffusers-format variants:
    - down_blocks.{i}.resnets.{j}.xxx → model.diffusion_model.input_blocks.{3i+j+1}.0.xxx
    - down_blocks.{i}.attentions.{j}.xxx → model.diffusion_model.input_blocks.{3i+j+1}.1.xxx
    - etc.

    Also handles TE keys (text_model. → cond_stage_model.transformer.text_model.)
    """
    reverse_map: Dict[str, str] = {}

    # Build base keys (without .weight/.bias/.alpha)
    ckpt_bases: List[str] = []
    for k in ckpt_sd:
        base = k
        for suffix in ('.weight', '.bias', '.alpha'):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        if base not in ckpt_bases:
            ckpt_bases.append(base)

    # ---- Section 1: UNet native keys → Diffusers variants ----
    for base_key in ckpt_bases:
        if not base_key.startswith('model.diffusion_model.'):
            continue

        inner = base_key[len('model.diffusion_model.'):]  # input_blocks.1.0.xxx

        # Direct native variant (diffusion_model. prefix)
        reverse_map[f'diffusion_model.{inner}'] = base_key

        # Pure path variant
        reverse_map[inner] = base_key

        # ---- Generate Diffusers variants ----
        for native_prefix, diff_prefix in _NATIVE_TO_DIFF_PREFIXES:
            if inner.startswith(native_prefix):
                diff_key = diff_prefix + inner[len(native_prefix):]
                # Apply resnet submodule renaming (native→diff)
                diff_key = _apply_resnet_rename_native_to_diff(diff_key)
                diff_key = _cleanup_dots(diff_key)
                reverse_map[diff_key] = base_key
                break

    # ---- Section 2: TE native keys → text_model. variants ----
    for base_key in ckpt_bases:
        if not base_key.startswith('cond_stage_model.transformer.'):
            continue

        pure_te = base_key[len('cond_stage_model.transformer.'):]  # text_model.encoder.xxx
        reverse_map[pure_te] = base_key

        # Also register lora_te_ variants (Kohya-style)
        reverse_map[f'lora_te_{pure_te}'] = base_key
        reverse_map[f'lora_te1_{pure_te}'] = base_key

        # te. prefix variants
        reverse_map[f'te.{pure_te}'] = base_key
        reverse_map[f'te1.{pure_te}'] = base_key

        # Underscore variants
        underscorized = pure_te.replace('.', '_')
        if underscorized != pure_te:
            reverse_map[underscorized] = base_key
            reverse_map[f'lora_te_{underscorized}'] = base_key

    # ---- Section 3: Diffusers direct keys (conv_in, time_embedding, etc.) ----
    for base_key in ckpt_bases:
        if not base_key.startswith('model.diffusion_model.'):
            continue
        inner = base_key[len('model.diffusion_model.'):]

        # Check if this is a direct-mapped key
        for native_name, diff_name in NATIVE_TO_DIFFUSERS_DIRECT:
            if inner == native_name:
                reverse_map[diff_name] = base_key
                break

    return reverse_map


def match_diffusers_deltas(
    ckpt_sd: Dict[str, torch.Tensor],
    lora_deltas: Dict[str, torch.Tensor],
    diffusers_reverse_map: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """
    Match LoRA delta bases to checkpoint keys using Diffusers-aware reverse map.

    Args:
        ckpt_sd: Native format checkpoint state dict
        lora_deltas: Dict of {lora_delta_base: tensor} from _reconstruct_lora_delta()
                     Keys are in Diffusers-preserving format (normalize_diffusers_preserving)
        diffusers_reverse_map: Pre-built map from build_diffusers_reverse_map()

    Returns:
        Dict of {checkpoint_key: tensor} (keys are native format)
    """
    matched: Dict[str, torch.Tensor] = {}
    unmatched: List[str] = []

    total_deltas = len(lora_deltas)
    with ProgressTracker(total=total_deltas, desc="Matching diffusers deltas") as match_progress:
        for lora_base, delta in lora_deltas.items():
            # The lora_base is in Diffusers-preserving format (e.g., down_blocks.0.attentions.0.xxx)
            # The diffusers_reverse_map has entries like:
            #   down_blocks.0.attentions.0.proj_in → model.diffusion_model.input_blocks.1.1.proj_in

            if lora_base in diffusers_reverse_map:
                ckpt_base = diffusers_reverse_map[lora_base]
                # Find the full checkpoint key (with .weight suffix)
                ckpt_weight_key = f'{ckpt_base}.weight'
                if ckpt_weight_key in ckpt_sd:
                    matched[ckpt_weight_key] = delta
                elif ckpt_base in ckpt_sd:
                    # Key without .weight suffix (e.g., bare bias or alpha)
                    matched[ckpt_base] = delta
                else:
                    # Try .bias suffix
                    ckpt_bias_key = f'{ckpt_base}.bias'
                    if ckpt_bias_key in ckpt_sd:
                        matched[ckpt_bias_key] = delta
                    else:
                        unmatched.append(lora_base)
            else:
                unmatched.append(lora_base)

            match_progress += 1

    if unmatched:
        print(f'   [WARN] {len(unmatched)} delta bases unmatched in diffusers_reverse_map')
        for b in unmatched[:5]:
            print(f'      - {b}')

    print(f'   [OK] Matched {len(matched)} deltas via Diffusers-bridge reverse map')
    return matched


# =====================================================================
# CONVENIENCE: Single-key native↔diffusers conversion
# =====================================================================

def native_to_diffusers_key(native_key: str) -> str:
    """Convert a single native SD1.5 UNet key to Diffusers format."""
    if not native_key.startswith('model.diffusion_model.'):
        # Try TE conversion (native → text_model. variant)
        if native_key.startswith('cond_stage_model.transformer.'):
            return native_key[len('cond_stage_model.transformer.'):]
        return native_key

    inner = native_key[len('model.diffusion_model.'):]

    # Direct mappings
    for native_name, diff_name in NATIVE_TO_DIFFUSERS_DIRECT:
        if inner == native_name:
            return diff_name
        # Also check for inner without suffix (e.g., just weight/bias stripped)
        native_base = native_name.rsplit('.', 1)[0]
        inner_base = inner.rsplit('.', 1)[0]
        if inner_base == native_base:
            suffix = inner[len(native_base):]
            diff_base = diff_name.rsplit('.', 1)[0]
            return f'{diff_base}{suffix}'

    # Block prefix replacements
    for native_prefix, diff_prefix in _NATIVE_TO_DIFF_PREFIXES:
        if inner.startswith(native_prefix):
            inner = diff_prefix + inner[len(native_prefix):]
            break

    # Resnet submodule renaming
    inner = _apply_resnet_rename_native_to_diff(inner)

    inner = _cleanup_dots(inner)

    return inner


def diffusers_to_native_key(diff_key: str) -> str:
    """Convert a single Diffusers-format UNet key to native SD1.5 format."""
    # Check if it's already a native key
    if diff_key.startswith('model.diffusion_model.'):
        return diff_key
    if diff_key.startswith('cond_stage_model.'):
        return diff_key

    return f'model.diffusion_model.{_convert_single_diffusers_to_native(diff_key)}'
