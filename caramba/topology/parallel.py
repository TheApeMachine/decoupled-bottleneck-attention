"""Parallel topology: layers applied independently.

All layers receive the same input and their outputs are stacked along
a new dimension. Useful for mixture-of-experts style architectures or
when you want to run multiple heads in parallel.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import ParallelTopologyConfig
from caramba.topology.utils import unwrap_output


class ParallelTopology(nn.Module):
    """Apply all layers to the same input, stack outputs.

    Unlike branching (which concatenates), parallel stacks outputs
    along dimension 0, preserving separation between branch outputs.
    """

    def __init__(self, config: ParallelTopologyConfig) -> None:
        """Build all layers from config."""
        super().__init__()
        self.config: ParallelTopologyConfig = config
        built = [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        if not built:
            raise ValueError("ParallelTopology requires at least one layer")
        self.layers: nn.ModuleList = nn.ModuleList(built)

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Apply all layers and stack their outputs."""
        outputs = [
            unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
            for layer in self.layers
        ]
        return torch.stack(outputs, dim=0)
