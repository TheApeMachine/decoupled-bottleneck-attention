"""Base topology module."""

from __future__ import annotations

from torch import nn

from caramba.config.topology import TopologyConfig

class Topology(nn.Module):
    """Base topology class."""
    def __init__(self, config: TopologyConfig) -> None:
        super().__init__()
        self.layers: nn.Module = config.build()