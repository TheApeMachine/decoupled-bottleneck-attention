"""Branching topology: layers applied in parallel, outputs concatenated.

All layers receive the same input. Their outputs are concatenated along
dimension 0. Use this when you want to merge multiple processing paths
into a single wider representation.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import BranchingTopologyConfig
from caramba.topology.utils import unwrap_output


class BranchingTopology(nn.Module):
    """Apply all layers to the same input, concatenate outputs.

    Unlike parallel (which stacks), branching concatenates, producing
    a wider output tensor.
    """

    def __init__(self, config: BranchingTopologyConfig) -> None:
        """Build all layers from config."""
        super().__init__()
        self.config: BranchingTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Apply all layers and concatenate their outputs."""
        outputs = [
            unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
            for layer in self.layers
        ]
        return torch.cat(outputs, dim=0)
