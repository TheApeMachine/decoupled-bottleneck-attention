from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast
from typing_extensions import override

import torch
import torch.nn as nn

from production.attention import DecoupledBottleneckAttention
from production.kvcache_backend import DecoupledLayerKVCache, LayerKVCache
from production.model.ff import FeedForward, FeedForwardCfg

if TYPE_CHECKING:
    from production.model.config import ModelConfig


class Block(nn.Module):
    """Transformer block: LN → attention → LN → MLP."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.ln1: nn.LayerNorm = nn.LayerNorm(cfg.d_model)
        self.attn: Callable[..., object] = cast(Callable[..., object], DecoupledBottleneckAttention(cfg))
        self.ln2: nn.LayerNorm = nn.LayerNorm(cfg.d_model)
        self.ff: FeedForward = FeedForward(cast(FeedForwardCfg, cast(object, cfg)))

    @override
    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None,
        cache: DecoupledLayerKVCache | LayerKVCache | None,
        pos_offset: int
    ) -> tuple[torch.Tensor, DecoupledLayerKVCache | LayerKVCache | None]:
        """Forward pass for a single transformer block."""
        a, new_cache = cast(
            tuple[torch.Tensor, DecoupledLayerKVCache | LayerKVCache | None],
            self.attn(self.ln1(x), attn_mask=attn_mask, cache=cache, pos_offset=pos_offset),
        )
        x = x + a
        x = x + cast(torch.Tensor, self.ff(self.ln2(x)))
        return x, new_cache
