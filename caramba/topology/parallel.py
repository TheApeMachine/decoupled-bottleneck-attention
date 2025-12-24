"""
parallel provides the parallel topology.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import ParallelTopologyConfig


class Parallel(nn.Module):
    """
    Parallel provides a parallel topology.
    """
    def __init__(
        self,
        config: ParallelTopologyConfig,
        layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.config: ParallelTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the parallel topology.
        """
        return torch.stack(
            [layer.forward(x) for layer in self.layers],
            dim=0,
        )