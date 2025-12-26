"""
branching provides the branching topology.
"""
from __future__ import annotations

from torch import nn, Tensor
import torch
from typing_extensions import override

from caramba.config.topology import BranchingTopologyConfig
from caramba.topology.utils import unwrap_output


class BranchingTopology(nn.Module):
    """
    BranchingTopology provides a branching topology.
    """
    def __init__(self, config: BranchingTopologyConfig) -> None:
        super().__init__()
        self.config: BranchingTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """
        forward pass for the branching topology.
        """
        outputs = [
            unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
            for layer in self.layers
        ]
        return torch.cat(outputs, dim=0)