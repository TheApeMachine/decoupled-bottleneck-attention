"""
nested allows a Topology to be used as a Layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import NestedTopologyConfig
from caramba.topology.utils import unwrap_output


class NestedTopology(nn.Module):
    """
    Nested provides a nested topology.
    """
    def __init__(self, config: NestedTopologyConfig) -> None:
        super().__init__()
        self.config: NestedTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """
        forward pass for nested topology.
        """
        for layer in self.layers:
            x = unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
        return x