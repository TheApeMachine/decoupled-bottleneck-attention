"""
nested allows a Topology to be used as a Layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import NestedTopologyConfig


class Nested(nn.Module):
    """
    Nested provides a nested topology.
    """
    def __init__(
        self,
        config: NestedTopologyConfig,
        layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.config: NestedTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for nested topology.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x