"""
stacked provides the stacked network.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import StackedTopologyConfig


class Stacked(nn.Module):
    """
    Stacked provides the stacked network.
    """
    def __init__(
        self,
        config: StackedTopologyConfig,
        layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.config: StackedTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the stacked network.
        """
        for layer in self.layers:
            x = layer.forward(x)

        return x
