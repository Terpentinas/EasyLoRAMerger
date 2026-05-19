"""
Delta reconstruction, alpha/rank scaling, and shape alignment utilities.

Extracted from baking_processor.py to reduce file size and isolate
tensor-level operations from key matching logic.
"""

import torch
from typing import (
    Dict, Iterator, Iterable, Mapping, MutableMapping,
    Optional, Tuple, Union,
)

from ..utils import ProgressTracker
from .scale_utils import apply_alpha_correction


# ===================================================================
# LoRA Pair Grouping
# ===================================================================

def _group_lora_pairs(
    lora_sd: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Group LoRA state dict entries by their base key and component type.

    For each key in lora_sd, determines if it's a component of a decomposed
    LoRA pair (``lora_A``, ``lora_B``, ``lora_up``, ``lora_down``, ``.alpha``,
    ``.diff``) or a standalone tensor (pre-baked delta, bias).

    Returns:
        pairs: dict of ``base_key → {component_name: tensor, ...}``
        standalone: dict of ``base_key → tensor`` (non-decomposed entries)
    """
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    standalone: Dict[str, torch.Tensor] = {}

    for key, tensor in lora_sd.items():
        base_key = None
        component = None

        if key.endswith('.lora_A.weight'):
            base_key = key[:-14]
            component = 'A'
        elif key.endswith('.lora_B.weight'):
            base_key = key[:-14]
            component = 'B'
        elif key.endswith('.lora_up.weight'):
            base_key = key[:-15]
            component = 'A'
        elif key.endswith('.lora_down.weight'):
            base_key = key[:-17]
            component = 'B'
        elif key.endswith('.lora_A'):
            base_key = key[:-7]
            component = 'A'
        elif key.endswith('.lora_B'):
            base_key = key[:-7]
            component = 'B'
        elif key.endswith('.lora_up'):
            base_key = key[:-8]
            component = 'A'
        elif key.endswith('.lora_down'):
            base_key = key[:-10]
            component = 'B'
        elif key.endswith('.alpha'):
            base_key = key[:-6]
            component = 'alpha'
        elif key.endswith('.diff'):
            base_key = key[:-5]
            component = 'diff'
        else:
            standalone[key] = tensor
            continue

        if base_key not in pairs:
            pairs[base_key] = {}
        pairs[base_key][component] = tensor

    return pairs, standalone


# ===================================================================
# Single Delta Reconstruction
# ===================================================================

def _reconstruct_one(
    base_key: str,
    components: Dict[str, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[torch.Tensor]:
    """
    Reconstruct a single LoRA weight delta from its A/B components.

    ``delta = B @ A`` (Kohya convention) or ``delta = A @ B`` (transposed),
    followed by ``(alpha / rank)`` scaling.

    Handles 4D Conv2d squeeze, GPU acceleration, and convention-agnostic
    rank detection.

    Returns:
        The reconstructed delta on CPU, or **None** if the components are
        incompatible (non-2D after squeeze, ambiguous rank).
    """
    if 'A' in components and 'B' in components:
        B = components['B']
        A = components['A']

        # Handle 4D Conv2d tensors (proj_in/proj_out) — squeeze spatial dims
        if B.dim() == 4 and A.dim() == 4:
            if B.shape[2] == 1 and B.shape[3] == 1 and A.shape[2] == 1 and A.shape[3] == 1:
                B = B.squeeze(-1).squeeze(-1)
                A = A.squeeze(-1).squeeze(-1)
            else:
                return None

        if B.dim() != 2 or A.dim() != 2:
            return None

        # GPU acceleration for matmul
        if device is not None:
            A = A.to(device=device)
            B = B.to(device=device)

        # Convention-agnostic delta reconstruction
        all_dims = [A.shape[0], A.shape[1], B.shape[0], B.shape[1]]
        rank = min(all_dims)

        if A.shape[1] == rank:
            delta = A @ B  # Kohya convention
        elif B.shape[1] == rank:
            delta = B @ A  # transposed convention
        else:
            return None

        # Alpha scaling
        alpha = components.get('alpha')
        if alpha is not None:
            alpha_val = alpha.item() if hasattr(alpha, 'item') else float(alpha)
            delta = apply_alpha_correction(delta, alpha_val, rank, mode="linear")

        return delta.cpu()

    elif 'diff' in components:
        # Pre-baked delta — pass through unchanged
        return components['diff']

    return None


# ===================================================================
# Shape Inference (no matmul needed)
# ===================================================================

def _infer_delta_shape(
    base_key: str,
    components: Dict[str, torch.Tensor],
) -> Optional[Tuple[int, ...]]:
    """
    Infer the shape of a delta tensor from its A/B component shapes.

    This avoids the cost of a full matmul (O(n³)) just to determine
    shape compatibility during key matching.  Only the dimensions are
    needed, which can be derived from the component tensor metadata.

    For Kohya convention: ``A=(out_dim,rank), B=(rank,in_dim)``
      → ``delta = A @ B = (out_dim, in_dim)``

    For transposed convention: ``A=(rank,in_dim), B=(out_dim,rank)``
      → ``delta = B @ A = (out_dim, in_dim)``
    """
    if 'A' not in components or 'B' not in components:
        return None

    B, A = components['B'], components['A']

    # Handle 4D Conv2d tensors — extract spatial dims from metadata
    if B.dim() == 4 and A.dim() == 4:
        if B.shape[2] == 1 and B.shape[3] == 1 and A.shape[2] == 1 and A.shape[3] == 1:
            B_dims = (B.shape[0], B.shape[1])
            A_dims = (A.shape[0], A.shape[1])
        else:
            return None
    elif B.dim() == 2 and A.dim() == 2:
        B_dims = (B.shape[0], B.shape[1])
        A_dims = (A.shape[0], A.shape[1])
    else:
        return None

    all_dims = [A_dims[0], A_dims[1], B_dims[0], B_dims[1]]
    rank = min(all_dims)

    if A_dims[1] == rank:
        return (A_dims[0], B_dims[1])  # Kohya: delta = A @ B
    elif B_dims[1] == rank:
        return (B_dims[0], A_dims[1])  # Transposed: delta = B @ A
    else:
        return None


# ===================================================================
# Lightweight Shape Proxy
# ===================================================================

class _ShapeInfo:
    """
    Lightweight shape metadata — quacks like a ``torch.Tensor`` for shape checks.

    Exposes ``.shape``, ``.dim()``, and ``.numel()`` so that
    :func:`_check_shape_compatible` and related functions can operate without
    a materialised tensor.  Used by :class:`_LazyDeltaDict.items()` to avoid
    reconstructing deltas during key matching.
    """
    __slots__ = ('shape',)

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __repr__(self):
        return f"_ShapeInfo({self.shape})"


# ===================================================================
# Lazy Delta Dictionary
# ===================================================================

class _LazyDeltaDict(Mapping[str, torch.Tensor]):
    """
    Dict-like wrapper storing LoRA A/B components for lazy delta reconstruction.

    Stores only the tiny component tensors (A, B, alpha — ≈14 MB total for
    112 LoRA pairs) instead of reconstructing all full deltas (≈9.6 GB).
    Each delta is reconstructed on demand via :meth:`pop` or :meth:`__getitem__`.

    During matching, :meth:`items` yields :class:`_ShapeInfo` objects (shape
    metadata only) to avoid unnecessary matmul computation — the matcher only
    needs shapes for ``_is_shape_compatible`` checks.

    Typical workflow::

        pairs, standalone = _group_lora_pairs(lora_sd)
        lazy = _LazyDeltaDict(pairs, standalone, device="cuda:0")

        # Matching — shapes only, no matmul
        for key, shape_info in lazy.items():
            if is_shape_compatible(ckpt_key, shape_info):
                matched[ckpt_key] = (lora_base, shape_info)

        # Baking — reconstruct one delta at a time
        for ckpt_key in bake_order:
            delta = lazy.reconstruct(lora_base)  # actual matmul
            bake(delta)
            # delta freed after loop iteration
    """

    def __init__(
        self,
        pairs: Dict[str, Dict[str, torch.Tensor]],
        standalone: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ):
        self._pairs = pairs
        self._standalone = standalone
        self._device = device
        self._shape_cache: Dict[str, Optional[Tuple[int, ...]]] = {}

    # ── Shape Inference ───────────────────────────────────────────────

    def _infer_shape(self, base_key: str) -> Optional[Tuple[int, ...]]:
        """Infer delta shape from component shapes (cached)."""
        if base_key in self._shape_cache:
            return self._shape_cache[base_key]
        components = self._pairs.get(base_key)
        if components is not None:
            shape = _infer_delta_shape(base_key, components)
        else:
            shape = None
        self._shape_cache[base_key] = shape
        return shape

    def get_shape_info(
        self, base_key: str
    ) -> Union['_ShapeInfo', torch.Tensor, None]:
        """
        Get shape metadata without reconstructing the delta.

        Returns:
            ``_ShapeInfo`` for A/B pairs (shape only),
            the actual tensor for standalone entries (already materialised),
            or **None** if the key doesn't exist or shape can't be inferred.
        """
        if base_key in self._standalone:
            return self._standalone[base_key]
        shape = self._infer_shape(base_key)
        if shape is not None:
            return _ShapeInfo(shape)
        return None

    # ── Reconstruction ────────────────────────────────────────────────

    def reconstruct(self, base_key: str) -> torch.Tensor:
        """
        Fully reconstruct a single delta (matmul + alpha scaling).

        Raises ``KeyError`` if *base_key* is not found.
        """
        components = self._pairs.get(base_key)
        if components is not None:
            result = _reconstruct_one(base_key, components, self._device)
            if result is not None:
                return result
        if base_key in self._standalone:
            return self._standalone[base_key]
        raise KeyError(f"Key {base_key!r} not found in _LazyDeltaDict")

    # ── Dict Protocol ────────────────────────────────────────────────

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.reconstruct(key)

    def __iter__(self) -> Iterator[str]:
        seen: set = set()
        for k in self._pairs:
            if k not in seen:
                yield k
                seen.add(k)
        for k in self._standalone:
            if k not in seen:
                yield k
                seen.add(k)

    def __len__(self) -> int:
        return len(self._pairs) + len(self._standalone)

    def __contains__(self, key: object) -> bool:
        return key in self._pairs or key in self._standalone

    def keys(self):
        return self._pairs.keys() | self._standalone.keys()

    def items(self):
        """
        Yield ``(key, value)`` pairs.

        For A/B pairs, yields ``_ShapeInfo`` instead of the full delta tensor
        to avoid matmul during key matching.  Standalone tensors are yielded
        as-is since they are already materialised.

        Use :meth:`pop` or :meth:`__getitem__` to get the actual reconstructed
        tensor.
        """
        for key in self._pairs:
            shape_info = self.get_shape_info(key)
            if shape_info is not None:
                yield key, shape_info
        for key, tensor in self._standalone.items():
            yield key, tensor

    def values(self):
        for key in self:
            yield self[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: str, *args):
        """
        Pop and reconstruct a delta on demand.

        Accepts an optional default as the second argument, matching
        ``dict.pop(key, default)``.
        """
        if len(args) > 1:
            raise TypeError(
                f"pop expected at most 2 arguments, got {1 + len(args)}"
            )
        if key in self._pairs:
            delta = self.reconstruct(key)
            del self._pairs[key]
            if key in self._shape_cache:
                del self._shape_cache[key]
            return delta
        if key in self._standalone:
            return self._standalone.pop(key)
        if args:
            return args[0]
        raise KeyError(key)

    def clear(self):
        """Clear all stored data and shape caches."""
        self._pairs.clear()
        self._standalone.clear()
        self._shape_cache.clear()


# ===================================================================
# Materialised Delta Wrapper for Baking
# ===================================================================

class _MatchedDeltas(MutableMapping[str, torch.Tensor]):
    """
    Dict-like wrapper that reconstructs LoRA deltas on demand during baking.

    Wraps a :class:`_LazyDeltaDict` together with a ``ckpt_key → lora_base``
    mapping produced by :func:`_find_matching_keys`.  Each :meth:`pop` or
    :meth:`__getitem__` call triggers a single delta reconstruction — no
    materialised tensor is held between calls.

    This eliminates the ≈9.6 GB ``matched_deltas`` dict that previously
    persisted during the entire bake loop.  Only one delta (≈86 MB for
    768×768 float32) is in memory at any time.

    Supports the full ``MutableMapping`` protocol needed by the three bake
    methods (``bake_linear``, ``bake_impact_weighted``, ``bake_orthogonal``),
    including :meth:`items` (which reconstructs incrementally during energy
    analysis), :meth:`pop`, :meth:`clear`, :meth:`keys`, and :meth:`values`.

    Some matched keys (fused QKV, ``to_out.0`` fallback) produce tensors
    that cannot be reconstructed from a single ``_LazyDeltaDict`` entry
    (they are concatenations or already-materialised tensors).  These are
    stored in *precomputed* and returned directly.
    """

    def __init__(
        self,
        lazy_deltas: '_LazyDeltaDict',
        ckpt_to_lora: Dict[str, str],
        precomputed: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            lazy_deltas: Component-based lazy delta store.
            ckpt_to_lora: Mapping from checkpoint key → LoRA base key,
                produced during matching.
            precomputed: Optional dict of checkpoint keys whose tensors
                are already materialised (fused QKV, to_out.0 fallback)
                and cannot be reconstructed from ``_LazyDeltaDict`` alone.
        """
        self._lazy = lazy_deltas
        self._ckpt_to_lora = ckpt_to_lora
        self._precomputed = precomputed if precomputed is not None else {}

    def _get_lora_key(self, ckpt_key: str) -> str:
        return self._ckpt_to_lora[ckpt_key]

    def pop(self, key: str, *args):
        """
        Pop a checkpoint key and reconstruct its delta on demand.

        Checks *precomputed* dict first (fused QKV, ``to_out.0``), then
        falls back to :meth:`_LazyDeltaDict.reconstruct` via the key map.

        Accepts optional default, matching ``dict.pop(key, default)``.
        """
        if len(args) > 1:
            raise TypeError(
                f"pop expected at most 2 arguments, got {1 + len(args)}"
            )
        # 1. Check precomputed dict (fused QKV, to_out.0 fallback)
        if key in self._precomputed:
            return self._precomputed.pop(key)
        # 2. Reconstruct from lazy components on demand
        lora_base = self._ckpt_to_lora.pop(key, None)
        if lora_base is not None:
            return self._lazy.reconstruct(lora_base)
        if args:
            return args[0]
        raise KeyError(key)

    def __getitem__(self, key: str) -> torch.Tensor:
        # Check precomputed first
        if key in self._precomputed:
            return self._precomputed[key]
        return self._lazy.reconstruct(self._get_lora_key(key))

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        # Not used during baking — raise for safety if it ever is.
        raise NotImplementedError("_MatchedDeltas does not support __setitem__")

    def __delitem__(self, key: str) -> None:
        if key in self._precomputed:
            del self._precomputed[key]
        else:
            del self._ckpt_to_lora[key]

    def __iter__(self) -> Iterator[str]:
        seen: set = set()
        for k in self._precomputed:
            if k not in seen:
                yield k
                seen.add(k)
        for k in self._ckpt_to_lora:
            if k not in seen:
                yield k
                seen.add(k)

    def __len__(self) -> int:
        return len(self._ckpt_to_lora) + len(self._precomputed)

    def __contains__(self, key: object) -> bool:
        return key in self._ckpt_to_lora or key in self._precomputed

    def keys(self):
        seen: set = set()
        for k in self._precomputed:
            if k not in seen:
                yield k
                seen.add(k)
        for k in self._ckpt_to_lora:
            if k not in seen:
                yield k
                seen.add(k)

    def items(self):
        """Yield ``(ckpt_key, delta)`` pairs, reconstructing each in turn."""
        # Yield precomputed entries first (no reconstruction needed)
        for ckpt_key, tensor in self._precomputed.items():
            yield ckpt_key, tensor
        for ckpt_key, lora_base in self._ckpt_to_lora.items():
            yield ckpt_key, self._lazy.reconstruct(lora_base)

    def values(self):
        for ckpt_key in list(self._precomputed):
            yield self._precomputed[ckpt_key]
        for ckpt_key, lora_base in self._ckpt_to_lora.items():
            yield self._lazy.reconstruct(lora_base)

    def clear(self):
        self._ckpt_to_lora.clear()
        self._precomputed.clear()


# ===================================================================
# Full Delta Reconstruction (Original API — backward compatible)
# ===================================================================

def _reconstruct_lora_delta(
    lora_sd: Dict[str, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct LoRA weight deltas from decomposed A/B pairs.

    For each pair ``(lora_B, lora_A)`` sharing a base key::

        delta = B @ A

    Then applies ``(alpha / rank)`` scaling to produce the final delta
    that represents the weight change for this LoRA key.

    Also handles standalone tensors (pre-baked deltas, biases) — passed
    through unchanged.

    This is the **original eager API** — use :class:`_LazyDeltaDict` for
    on-demand reconstruction.

    Args:
        lora_sd: LoRA state dict with A/B decomposed tensors.
        device: If provided (e.g. ``"cuda:0"``), moves A/B tensors to this
            device before the matmul for GPU-accelerated reconstruction.

    Returns:
        ``dict`` of ``base_key → delta tensor`` (on CPU).
    """
    pairs, standalone = _group_lora_pairs(lora_sd)
    deltas: Dict[str, torch.Tensor] = {}

    with ProgressTracker(total=len(pairs), desc="Reconstructing deltas") as delta_progress:
        for base_key, components in pairs.items():
            delta = _reconstruct_one(base_key, components, device)
            if delta is not None:
                deltas[base_key] = delta
            delta_progress += 1

    # Add standalone tensors (pre-baked deltas, biases)
    deltas.update(standalone)

    return deltas
