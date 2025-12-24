"""
sequential provides the sequential topology.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import SequentialTopologyConfig


class Sequential(nn.Module):
    """
    Sequential provides a sequential topology.
    """
    def __init__(
        self,
        config: SequentialTopologyConfig,
        layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.config: SequentialTopologyConfig = config
        if not layers or len(layers) == 0:
            raise ValueError("SequentialTopology requires at least one layer")
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the sequential topology.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x