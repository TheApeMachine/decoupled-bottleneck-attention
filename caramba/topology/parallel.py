"""
parallel provides the parallel topology.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import ParallelTopologyConfig


class ParallelTopology(nn.Module):
    """
    Parallel provides a parallel topology.
    """
    def __init__(self, config: ParallelTopologyConfig) -> None:
        super().__init__()
        self.config: ParallelTopologyConfig = config
        built = [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        if not built:
            raise ValueError("ParallelTopology requires at least one layer")
        self.layers: nn.ModuleList = nn.ModuleList(built)

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """
        forward pass for the parallel topology.
        """
        return torch.stack(
            [layer.forward(x, ctx=ctx) for layer in self.layers],  # type: ignore[call-arg]
            dim=0,
        )