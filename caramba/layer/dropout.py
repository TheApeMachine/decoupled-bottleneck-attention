"""
dropout provides the dropout layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override
from caramba.config.layer import DropoutLayerConfig
from caramba.operation.build import build_dropout_operation
from caramba.operation.dropout import Drop


class Dropout(nn.Module):
    """
    Dropout provides the dropout layer.
    """
    def __init__(self, config: DropoutLayerConfig) -> None:
        super().__init__()
        self.config: DropoutLayerConfig = config
        self.operation: Drop = build_dropout_operation(config.operation)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the dropout layer.
        """
        return self.operation.forward(x)
