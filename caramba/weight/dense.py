"""
dense provides dense weight containers.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.init as init
from typing_extensions import override

from caramba.weight.guard import require_bool, require_int


class DenseWeight(nn.Module):
    """
    DenseWeight stores a dense matrix and optional bias.
    """
    def __init__(self, d_in: int, d_out: int, *, bias: bool) -> None:
        super().__init__()
        self.d_in: int = require_int("d_in", d_in, ge=1)
        self.d_out: int = require_int("d_out", d_out, ge=1)
        self.has_bias: bool = require_bool("bias", bias)

        self.weight: nn.Parameter = nn.Parameter(
            torch.empty((self.d_out, self.d_in)),
        )
        self.bias: nn.Parameter | None = (
            nn.Parameter(torch.empty((self.d_out,))) if self.has_bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        reset weight parameters.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # fan_in is the input dimension for a dense [d_out, d_in] weight.
            fan_in = int(self.d_in)
            bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = x
        raise RuntimeError("DenseWeight is a weight container; call Matmul.forward.")


