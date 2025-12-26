"""Shared utilities for topology implementations.

Common helpers that multiple topologies need, like extracting the
tensor output from layers that return (output, cache) tuples.
"""
from __future__ import annotations

from torch import Tensor


def unwrap_output(out: Tensor | tuple[Tensor, object]) -> Tensor:
    """Extract the tensor from a layer output.

    Many layers (especially AttentionLayer) return (output, cache) tuples.
    Topologies that don't track caches use this to get just the tensor.
    """
    if isinstance(out, tuple):
        return out[0]
    return out
