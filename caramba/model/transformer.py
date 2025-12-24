"""
transformer provides the transformer model.
"""
from __future__ import annotations

from torch import nn, Tensor
from caramba.config.topology import TopologyConfig
from caramba.builder.topology import TopologyBuilder
from caramba.compiler.lower import lower_topology
from caramba.compiler.validate import validate_topology



class Transformer(nn.Module):
    """
    Transformer provides the transformer model.
    """
    def __init__(self, config: TopologyConfig) -> None:
        super().__init__()
        lowered = lower_topology(config)
        validate_topology(lowered)
        self.config: TopologyConfig = lowered
        builder = TopologyBuilder()
        self.topology: nn.Module = builder.build(lowered)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the transformer model.
        """
        return self.topology.forward(x)