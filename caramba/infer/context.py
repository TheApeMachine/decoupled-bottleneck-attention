"""Inference context: KV caches and position tracking.

During generation, each attention layer needs its own KV-cache to store
past keys and values. The InferContext holds all these caches and tracks
the current position offset so layers know where they are in the sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.cache.layer import LayerKVCache

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class InferContext:
    """Carries KV caches and position metadata through the model.

    Each forward pass calls begin() to reset the cache index, then each
    attention layer calls next_cache() to get its cache. After the pass,
    ensure_consumed() validates all caches were used.
    """

    caches: list[LayerKVCache | DecoupledLayerKVCache]
    pos_offset: int = 0
    attn_mask: "Tensor | None" = None

    _index: int = 0

    def begin(self, *, pos_offset: int, attn_mask: "Tensor | None" = None) -> None:
        """Reset for a new forward pass.

        Called before each forward to set the position offset and reset
        the cache traversal index.
        """
        self._index = 0
        self.pos_offset = int(pos_offset)
        self.attn_mask = attn_mask

    def next_cache(self) -> LayerKVCache | DecoupledLayerKVCache:
        """Get the next cache in traversal order.

        Each attention layer calls this to get its cache. The order must
        match the order layers appear in the model.
        """
        if self._index >= len(self.caches):
            raise ValueError(
                "InferContext cache underflow: more Attention calls than caches. "
                f"index={self._index}, caches={len(self.caches)}"
            )
        c = self.caches[self._index]
        self._index += 1
        return c

    def ensure_consumed(self) -> None:
        """Validate all caches were used in the forward pass.

        Called after forward to detect model/cache mismatch early.
        """
        if self._index != len(self.caches):
            raise ValueError(
                "InferContext cache mismatch: not all caches were consumed. "
                f"used={self._index}, total={len(self.caches)}"
            )


def causal_mask(*, t_q: int, t_k: int, device: torch.device) -> torch.Tensor:
    """Build an explicit causal attention mask.

    Returns a boolean mask where True means "don't attend". Used when
    is_causal=True isn't available or when the mask shape is unusual.
    """
    if t_q <= 0 or t_k <= 0:
        raise ValueError(f"Expected t_q,t_k > 0, got {t_q},{t_k}")
    return torch.triu(
        torch.ones((t_q, t_k), device=device, dtype=torch.bool),
        diagonal=1,
    )
