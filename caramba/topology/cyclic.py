"""Cyclic topology: layers applied in a loop pattern.

Currently identical to stacked, but semantically represents architectures
where information flows in cycles. Reserved for future cycle-aware
optimizations or analysis.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import CyclicTopologyConfig
from caramba.topology.utils import unwrap_output


class CyclicTopology(nn.Module):
    """Apply layers sequentially (cyclic pattern).

    The "cyclic" name hints at architectures with feedback loops,
    though currently this is implemented as simple sequential flow.
    """

    def __init__(self, config: CyclicTopologyConfig) -> None:
        """Build all layers from config."""
        super().__init__()
        self.config: CyclicTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(
            [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        )

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Forward through all layers sequentially."""
        for layer in self.layers:
            x = unwrap_output(layer(x, ctx=ctx))  # type: ignore[call-arg]
        return x
