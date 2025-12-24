"""
cyclic provides the cyclic topology.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import CyclicTopologyConfig


class Cyclic(nn.Module):
    """
    Cyclic provides a cyclic topology.
    """
    def __init__(
        self,
        config: CyclicTopologyConfig,
        layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.config: CyclicTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the cyclic topology.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x