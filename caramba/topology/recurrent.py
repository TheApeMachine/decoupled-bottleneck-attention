"""Recurrent topology: layers that produce state.

Like sequential, but collects any cache/state returned by layers.
Returns either just the output (if no caches) or (output, caches).
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.topology import RecurrentTopologyConfig


class RecurrentTopology(nn.Module):
    """Apply layers in sequence, collecting any state they return.

    Layers that return (output, cache) tuples have their caches collected
    into a list. Useful for RNNs or transformers where you want to
    access per-layer state.
    """

    def __init__(self, config: RecurrentTopologyConfig) -> None:
        """Build all layers from config."""
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
        """Forward through layers, collecting any caches.

        Returns x if no caches, or (x, caches) if any layer returned state.
        """
        caches: list[object] = []
        for layer in self.layers:
            out = layer(x, ctx=ctx)  # type: ignore[call-arg]
            if isinstance(out, tuple):
                x, cache = out
                caches.append(cache)
            else:
                x = out
        if caches:
            return x, caches
        return x
