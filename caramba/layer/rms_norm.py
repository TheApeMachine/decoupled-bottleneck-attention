"""RMSNorm: a simpler alternative to LayerNorm.

RMSNorm normalizes by the root mean square instead of mean and variance.
It's computationally cheaper and empirically works just as well for LLMs.
Llama and many modern models use RMSNorm instead of LayerNorm.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import RMSNormLayerConfig


class RMSNormLayer(nn.Module):
    """Root mean square normalization layer.

    Unlike LayerNorm, RMSNorm doesn't center the activations (no mean
    subtraction), only scales them. This reduces computation and the
    learnable bias term while maintaining training stability.
    """

    def __init__(self, config: RMSNormLayerConfig) -> None:
        """Initialize RMSNorm with optional learnable scale.

        Args:
            config: Specifies d_model, epsilon, and whether to use
                   a learnable elementwise scale (weight).
        """
        super().__init__()
        self.config = config
        self.d_model = int(config.d_model)
        self.eps = float(config.eps)
        self.elementwise_affine = bool(config.elementwise_affine)
        self.weight: nn.Parameter | None = (
            nn.Parameter(torch.ones(self.d_model))
            if self.elementwise_affine
            else None
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Normalize by root mean square.

        Computes x / sqrt(mean(x^2) + eps), then optionally scales by
        a learned per-dimension weight.
        """
        if x.ndim < 1:
            raise ValueError(f"Expected x.ndim >= 1, got {x.shape}")
        if int(x.shape[-1]) != int(self.d_model):
            raise ValueError(f"Expected x last dim {int(self.d_model)}, got {x.shape}")

        x_f = x.float()
        inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + float(self.eps))
        y = (x_f * inv_rms).to(dtype=x.dtype)
        if self.weight is not None:
            y = y * self.weight
        return y
