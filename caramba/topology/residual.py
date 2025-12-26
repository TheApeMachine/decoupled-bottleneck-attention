"""Residual topology: layers with skip connection.

Adds the input to the output of all layers, implementing a residual
connection. This is the fundamental building block of modern transformers—
without residuals, deep networks are very hard to train.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import ResidualTopologyConfig
from caramba.topology.utils import unwrap_output


class ResidualTopology(nn.Module):
    """Apply layers then add the original input (residual connection).

    The residual is taken before any layers and added after all layers.
    For per-layer residuals, use a stacked topology with individual
    residual blocks.
    """

    def __init__(self, config: ResidualTopologyConfig) -> None:
        """Build all layers from config."""
        super().__init__()
        self.config: ResidualTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Forward through layers, then add the input residual."""
        residual = x
        for layer in self.layers:
            x = unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
        return residual + x
