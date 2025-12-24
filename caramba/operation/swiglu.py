"""
swiglu provides the SwiGLU activation operation.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override


class SwiGLUOp(nn.Module):
    """
    SwiGLUOp applies the SwiGLU gating function.
    """
    def __init__(self) -> None:
        """
        __init__ initializes the SwiGLU operation.
        """
        super().__init__()

    @override
    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        """
        forward applies silu(gate) * up.
        """
        if gate.shape != up.shape:
            raise ValueError(
                f"Expected gate/up to have same shape, got "
                f"gate={gate.shape}, up={up.shape}"
            )
        return F.silu(gate) * up
