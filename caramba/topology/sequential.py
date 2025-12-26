"""Sequential topology: simple layer chain.

The most basic topology: each layer's output feeds into the next.
Handles layers that return (output, cache) tuples by extracting
just the output tensor.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import SequentialTopologyConfig


class SequentialTopology(nn.Module):
    """Apply layers in sequence.

    Similar to nn.Sequential but handles ctx argument and (output, cache)
    tuple returns from layers like AttentionLayer.
    """

    def __init__(self, config: SequentialTopologyConfig) -> None:
        """Build all layers from config."""
        super().__init__()
        self.config: SequentialTopologyConfig = config
        built = [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        if not built:
            raise ValueError("SequentialTopology requires at least one layer")
        self.layers: nn.ModuleList = nn.ModuleList(built)

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Forward through all layers, extracting outputs from tuples."""
        for layer in self.layers:
            out = layer(x, ctx=ctx)  # type: ignore[call-arg]
            x = out[0] if isinstance(out, tuple) else out
        return x
