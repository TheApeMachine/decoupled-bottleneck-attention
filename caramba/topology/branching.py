"""
branching provides the branching topology.
"""
from __future__ import annotations

from torch import nn, Tensor
import torch
from typing_extensions import override

from caramba.config.topology import BranchingTopologyConfig


class Branching(nn.Module):
    """
    Branching provides a branching topology.
    """
    def __init__(
        self,
        config: BranchingTopologyConfig,
        layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.config: BranchingTopologyConfig = config
        if not layers or len(layers) == 0:
            raise ValueError("layers must contain at least one nn.Module")
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the branching topology.
        """
        return torch.cat(
            [layer.forward(x) for layer in self.layers],
            dim=0,
        )