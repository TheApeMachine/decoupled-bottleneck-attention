"""
swiglu provides the SwiGLU MLP layer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import SwiGLULayerConfig


class SwiGLULayer(nn.Module):
    """
    SwiGLU provides a SwiGLU MLP layer.
    """
    def __init__(self, config: SwiGLULayerConfig) -> None:
        super().__init__()
        self.config: SwiGLULayerConfig = config
        self.d_model: int = int(config.d_model)
        self.d_ff: int = int(config.d_ff)
        self.bias: bool = bool(getattr(config, "bias", True))
        self.w_gate = nn.Linear(
            self.d_model,
            self.d_ff,
            bias=self.bias,
        )
        self.w_up = nn.Linear(
            self.d_model,
            self.d_ff,
            bias=self.bias,
        )
        self.silu = nn.SiLU()
        self.w_down = nn.Linear(
            self.d_ff,
            self.d_model,
            bias=self.bias,
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """
        forward pass for SwiGLU.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {x.shape}")
        if int(x.shape[-1]) != int(self.d_model):
            raise ValueError(
                f"Expected last dim {int(self.d_model)}, got {x.shape}"
            )
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(self.silu(gate) * up)
