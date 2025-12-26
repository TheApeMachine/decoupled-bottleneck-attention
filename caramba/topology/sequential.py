"""
sequential provides the sequential topology.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import SequentialTopologyConfig


class SequentialTopology(nn.Module):
    """
    Sequential provides a sequential topology.
    """
    def __init__(self, config: SequentialTopologyConfig) -> None:
        super().__init__()
        self.config: SequentialTopologyConfig = config
        built = [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        if not built:
            raise ValueError("SequentialTopology requires at least one layer")
        self.layers: nn.ModuleList = nn.ModuleList(built)

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """
        forward pass for the sequential topology.
        """
        for layer in self.layers:
            out = layer(x, ctx=ctx)  # type: ignore[call-arg]
            # Handle layers that return (output, cache) tuples (e.g., AttentionLayer)
            x = out[0] if isinstance(out, tuple) else out
        return x