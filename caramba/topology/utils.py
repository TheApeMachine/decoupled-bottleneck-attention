"""
utils provides shared utilities for topology implementations.
"""
from __future__ import annotations

from torch import Tensor


def unwrap_output(out: Tensor | tuple[Tensor, object]) -> Tensor:
    """
    Unwrap a layer output, handling both plain tensors and (output, cache) tuples.

    Many layers (especially AttentionLayer) return (output, cache) tuples.
    This helper extracts just the tensor output for topologies that don't
    need to track caches.

    Args:
        out: Either a Tensor or a (Tensor, cache) tuple.

    Returns:
        The tensor output.
    """
    if isinstance(out, tuple):
        return out[0]
    return out
