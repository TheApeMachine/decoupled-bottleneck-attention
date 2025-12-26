"""Nested topology: topologies as layers.

Allows a topology to be used as a layer within another topology.
This enables hierarchical architectures where blocks of blocks
form larger structures.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import NestedTopologyConfig
from caramba.topology.utils import unwrap_output


class NestedTopology(nn.Module):
    """Wrap a topology to use as a layer.

    This is how you build hierarchical architectures: define a block
    as a topology, then nest it inside another topology.
    """

    def __init__(self, config: NestedTopologyConfig) -> None:
        """Build all nested layers from config."""
        super().__init__()
        self.config: NestedTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Forward through all nested layers."""
        for layer in self.layers:
            x = unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
        return x
