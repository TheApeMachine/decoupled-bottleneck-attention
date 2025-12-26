"""Network topology building blocks.

A topology defines how layers are connected. Unlike a flat nn.Sequential,
topologies can express residual connections, parallel branches, cycles,
and nesting. Each topology type builds from a config that specifies its
layers and how many times to repeat the pattern.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from caramba.config.topology import TopologyConfig


class Topology(nn.Module):
    """Base topology that delegates to its built layers.

    This is the simplest topology: just build whatever the config
    specifies and forward through it.
    """

    def __init__(self, config: TopologyConfig) -> None:
        """Build the topology from config."""
        super().__init__()
        self.layers: nn.Module = config.build()

    def forward(self, x: Tensor) -> Tensor:
        """Forward through the built layers."""
        return self.layers(x)
