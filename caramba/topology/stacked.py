"""Stacked topology: layers applied in sequence.

The stacked topology is the standard transformer pattern: each layer's
output feeds into the next. Unlike sequential, stacked supports repeating
the layer pattern multiple times (e.g., 32 identical transformer blocks).
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import StackedTopologyConfig
from caramba.topology.utils import unwrap_output


class StackedTopology(nn.Module):
    """Apply layers sequentially, optionally repeated.

    This is how you build a 32-layer transformer: define one attention+FFN
    block, then set repeat=32. Weight sharing is not applied—each repeat
    gets its own parameters.
    """

    def __init__(self, config: StackedTopologyConfig) -> None:
        """Build all layers from config."""
        super().__init__()
        self.config: StackedTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Forward through all layers sequentially."""
        for layer in self.layers:
            x = unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
        return x
