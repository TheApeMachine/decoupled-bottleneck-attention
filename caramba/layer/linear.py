"""
linear provides the linear layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import LinearLayerConfig
from caramba.operation.build import build_matmul_operation
from caramba.operation.matmul import Matmul
from caramba.weight.build import build_dense_weight
from caramba.weight.dense import DenseWeight


class Linear(nn.Module):
    """
    Linear provides the linear layer.
    """

    def __init__(self, config: LinearLayerConfig) -> None:
        super().__init__()
        self.config: LinearLayerConfig = config
        self.operation: Matmul = build_matmul_operation(config.operation)
        self.weight: DenseWeight = build_dense_weight(config.weight)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the linear layer.
        """
        return self.operation.forward(
            x,
            weight=self.weight.weight,
            bias=self.weight.bias,
        )

