"""
swiglu provides the SwiGLU MLP layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import SwiGLULayerConfig
from caramba.operation.build import build_swiglu_operation
from caramba.operation.matmul import Matmul
from caramba.operation.swiglu import SwiGLUOp
from caramba.weight.build import build_swiglu_weight
from caramba.weight.swiglu import SwiGLUWeight


class SwiGLU(nn.Module):
    """
    SwiGLU provides a SwiGLU MLP layer.
    """
    def __init__(self, config: SwiGLULayerConfig) -> None:
        super().__init__()
        self.config: SwiGLULayerConfig = config
        self.matmul: Matmul = Matmul()
        self.operation: SwiGLUOp = build_swiglu_operation(config.operation)
        self.weight: SwiGLUWeight = build_swiglu_weight(config.weight)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for SwiGLU.
        """
        gate = self.matmul.forward(
            x,
            weight=self.weight.w_gate.weight,
            bias=self.weight.w_gate.bias,
        )
        up = self.matmul.forward(
            x,
            weight=self.weight.w_up.weight,
            bias=self.weight.w_up.bias,
        )
        y = self.operation.forward(gate, up)
        return self.matmul.forward(
            y,
            weight=self.weight.w_down.weight,
            bias=self.weight.w_down.bias,
        )
