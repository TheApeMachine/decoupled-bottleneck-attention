"""
residual provides the residual topology.
"""

from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import ResidualTopologyConfig


class Residual(nn.Module):
    """
    Residual provides a residual topology.
    """
    def __init__(
        self,
        config: ResidualTopologyConfig,
        layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.config: ResidualTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(layers)


    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the residual topology.
        """
        residual = x
        for layer in self.layers:
            x = layer(x)
        return residual + x
