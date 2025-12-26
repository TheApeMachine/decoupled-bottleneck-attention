"""Dropout layer for regularization during training.

Dropout randomly zeroes elements during training to prevent overfitting.
This wrapper provides our standard layer interface around nn.Dropout.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import DropoutLayerConfig


class DropoutLayer(nn.Module):
    """Dropout with our standard layer interface.

    Only active during training (model.train()); passes through unchanged
    during evaluation (model.eval()).
    """

    def __init__(self, config: DropoutLayerConfig) -> None:
        """Initialize dropout with the given probability.

        Args:
            config: Specifies dropout probability p (0 = no dropout, 1 = all zeros).
        """
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.p)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply dropout (only during training)."""
        return self.dropout(x)
