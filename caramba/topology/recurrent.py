"""
recurrent provides the recurrent topology.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import RecurrentTopologyConfig


class RecurrentTopology(nn.Module):
    """
    Recurrent provides a recurrent topology.
    """
    def __init__(self, config: RecurrentTopologyConfig) -> None:
        super().__init__()
        self.config: RecurrentTopologyConfig = config
        built = [cfg.build() for _ in range(config.repeat) for cfg in config.layers]
        if not built:
            raise ValueError("RecurrentTopology requires at least one layer")
        self.layers: nn.ModuleList = nn.ModuleList(built)

    @override
    def forward(
        self, x: Tensor, *, ctx: object | None = None
    ) -> Tensor | tuple[Tensor, list[object]]:
        """
        forward pass for the recurrent topology.

        Returns x or (x, caches) depending on whether layers return tuples.
        """
        caches: list[object] = []
        for layer in self.layers:
            out = layer(x, ctx=ctx)  # type: ignore[call-arg]
            # Handle layers that return (output, cache) tuples
            if isinstance(out, tuple):
                x, cache = out
                caches.append(cache)
            else:
                x = out
        # Return consistent with whether any caches were collected
        if caches:
            return x, caches
        return x