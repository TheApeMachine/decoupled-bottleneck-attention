"""
transformer provides the transformer model.
"""
from __future__ import annotations

from dataclasses import dataclass
from torch import nn, Tensor
from caramba.config.topology import TopologyConfig
from caramba.topology.stacked import Stacked


@dataclass(frozen=True, slots=True)
class _TransformerBlockSpec:
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    causal: bool


class Transformer(nn.Module):
    """
    Transformer provides the transformer model.
    """
    def __init__(self, config: TopologyConfig) -> None:
        super().__init__()
        self.config: TopologyConfig = config
        self.network: nn.Module = Stacked(config)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the transformer model.
        """
        return self.network.forward(x)