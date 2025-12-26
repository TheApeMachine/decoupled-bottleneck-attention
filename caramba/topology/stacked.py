"""
stacked provides the stacked network.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import StackedTopologyConfig


class StackedTopology(nn.Module):
    """
    Stacked provides the stacked network.
    """
    def __init__(self, config: StackedTopologyConfig) -> None:
        super().__init__()
        self.config: StackedTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """
        forward pass for the stacked network.
        """
        for layer in self.layers:
            out = layer(x, ctx=ctx)  # type: ignore[call-arg]
            # Handle layers that return (output, cache) tuples (e.g., AttentionLayer)
            x = out[0] if isinstance(out, tuple) else out

        return x
