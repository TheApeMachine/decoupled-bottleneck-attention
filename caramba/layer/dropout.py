"""
dropout provides the dropout layer.
"""
from __future__ import annotations


from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import DropoutLayerConfig


class DropoutLayer(nn.Module):
    """
    Dropout provides the dropout layer.
    """
    def __init__(self, config: DropoutLayerConfig) -> None:
        super().__init__()
        self.config: DropoutLayerConfig = config
        self.dropout = nn.Dropout(config.p)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """forward pass for the dropout layer."""
        return self.dropout(x)
