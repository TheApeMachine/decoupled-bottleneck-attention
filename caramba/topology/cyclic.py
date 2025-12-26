"""
cyclic provides the cyclic topology.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import CyclicTopologyConfig


class CyclicTopology(nn.Module):
    """
    Cyclic provides a cyclic topology.
    """
    def __init__(self, config: CyclicTopologyConfig) -> None:
        super().__init__()
        self.config: CyclicTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """
        forward pass for the cyclic topology.
        """
        for layer in self.layers:
            x = layer.forward(x, ctx=ctx)  # type: ignore[call-arg]
        return x