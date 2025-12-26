"""
residual provides the residual topology.
"""

from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import ResidualTopologyConfig
from caramba.topology.utils import unwrap_output


class ResidualTopology(nn.Module):
    """
    Residual provides a residual topology.
    """
    def __init__(self, config: ResidualTopologyConfig) -> None:
        super().__init__()
        self.config: ResidualTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """
        forward pass for the residual topology.
        """
        residual = x
        for layer in self.layers:
            x = unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
        return residual + x
