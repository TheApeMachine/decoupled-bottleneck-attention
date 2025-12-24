"""
swiglu provides SwiGLU MLP weight containers.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.weight.dense import DenseWeight
from caramba.weight.guard import require_bool, require_int


class SwiGLUWeight(nn.Module):
    """
    SwiGLUWeight stores the gate/up/down projection weights.
    """
    def __init__(self, *, d_model: int, d_ff: int, bias: bool) -> None:
        super().__init__()
        self.d_model: int = require_int("d_model", d_model, ge=1)
        self.d_ff: int = require_int("d_ff", d_ff, ge=1)
        self.bias: bool = require_bool("bias", bias)

        self.w_gate: DenseWeight = DenseWeight(self.d_model, self.d_ff, bias=self.bias)
        self.w_up: DenseWeight = DenseWeight(self.d_model, self.d_ff, bias=self.bias)
        self.w_down: DenseWeight = DenseWeight(self.d_ff, self.d_model, bias=self.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = x
        raise RuntimeError(
            "SwiGLUWeight is a weight container; call SwiGLU.forward."
        )


