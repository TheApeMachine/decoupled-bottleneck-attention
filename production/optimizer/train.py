"""Training optimizer placeholder.

This module exists as an extension point; training orchestration currently lives
in runner code.
"""

from __future__ import annotations

from production.optimizer.base import BaseOptimizer
from typing_extensions import override


class TrainOptimizer(BaseOptimizer):
    """No-op placeholder optimizer for training processes."""

    @override
    def optimize(self, process: object) -> None:
        """Optimize a training process (not implemented)."""
        _ = process
        raise NotImplementedError