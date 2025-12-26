"""
normalize provides the normalize layer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import LayerNormLayerConfig


class LayerNormLayer(nn.Module):
    """
    LayerNorm provides the layer norm layer.
    """
    def __init__(self, config: LayerNormLayerConfig) -> None:
        super().__init__()
        self.config: LayerNormLayerConfig = config
        self.norm = nn.LayerNorm(
            config.d_model,
            eps=float(config.eps),
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """forward pass for the normalize layer."""
        return self.norm(x)
