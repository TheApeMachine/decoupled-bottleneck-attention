"""
context provides inference-time context (KV caches, pos offset, masks).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from caramba.cache.layer import LayerKVCache
from caramba.cache.decoupled import DecoupledLayerKVCache

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class InferContext:
    """
    InferContext carries KV caches and sequence position metadata for decoding.
    """

    caches: list[LayerKVCache | DecoupledLayerKVCache]
    pos_offset: int = 0
    attn_mask: "Tensor | None" = None

    _index: int = 0

    def begin(self, *, pos_offset: int, attn_mask: "Tensor | None" = None) -> None:
        """
        begin resets per-forward counters and sets the current position offset.
        """
        self._index = 0
        self.pos_offset = int(pos_offset)
        self.attn_mask = attn_mask

    def next_cache(self) -> LayerKVCache | DecoupledLayerKVCache:
        """
        next_cache returns the next cache object in traversal order.
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
        """
        ensure_consumed validates that all caches were used in a forward pass.
        """
        if self._index != len(self.caches):
            raise ValueError(
                "InferContext cache mismatch: not all caches were consumed. "
                f"used={self._index}, total={len(self.caches)}"
            )


def causal_mask(*, t_q: int, t_k: int, device: torch.device) -> torch.Tensor:
    """
    causal_mask builds an explicit causal mask for (t_q, t_k).
    """
    if t_q <= 0 or t_k <= 0:
        raise ValueError(f"Expected t_q,t_k > 0, got {t_q},{t_k}")
    return torch.triu(
        torch.ones((t_q, t_k), device=device, dtype=torch.bool),
        diagonal=1,
    )


