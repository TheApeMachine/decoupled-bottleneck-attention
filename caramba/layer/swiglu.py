"""SwiGLU: the MLP variant used in Llama and modern transformers.

SwiGLU combines a gated linear unit with the SiLU (Swish) activation.
The gate and up projections are computed in parallel, then combined:
output = down(silu(gate(x)) * up(x)). This gives better performance
than ReLU or GELU MLPs at the same parameter count.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import SwiGLULayerConfig


class SwiGLULayer(nn.Module):
    """SwiGLU MLP layer with gate, up, and down projections.

    The hidden dimension (d_ff) is typically 8/3 × d_model, though the
    exact ratio is configurable. The three linear layers (gate, up, down)
    are where most of a transformer's parameters live.
    """

    def __init__(self, config: SwiGLULayerConfig) -> None:
        """Initialize the three linear projections.

        Args:
            config: Specifies d_model, d_ff (hidden dim), and bias settings.
        """
        super().__init__()
        self.config = config
        self.d_model = int(config.d_model)
        self.d_ff = int(config.d_ff)
        self.bias = bool(config.bias)
        self.w_gate = nn.Linear(self.d_model, self.d_ff, bias=self.bias)
        self.w_up = nn.Linear(self.d_model, self.d_ff, bias=self.bias)
        self.silu = nn.SiLU()
        self.w_down = nn.Linear(self.d_ff, self.d_model, bias=self.bias)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply SwiGLU: silu(gate(x)) * up(x), then down-project.

        Args:
            x: Input tensor (B, T, d_model)

        Returns:
            Output tensor (B, T, d_model)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {x.shape}")
        if int(x.shape[-1]) != int(self.d_model):
            raise ValueError(f"Expected last dim {int(self.d_model)}, got {x.shape}")
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(self.silu(gate) * up)
