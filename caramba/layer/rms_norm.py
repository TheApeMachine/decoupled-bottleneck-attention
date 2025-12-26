"""rms_norm provides the RMSNorm layer."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import RMSNormLayerConfig

class RMSNormLayer(nn.Module):
    """RMSNormLayer provides the RMSNorm layer."""
    def __init__(self, config: RMSNormLayerConfig) -> None:
        super().__init__()
        self.config: RMSNormLayerConfig = config
        self.d_model: int = int(config.d_model)
        self.eps: float = float(config.eps)
        self.elementwise_affine: bool = bool(config.elementwise_affine)
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
        """
        forward pass for RMSNorm.
        """
        if x.ndim < 1:
            raise ValueError(f"Expected x.ndim >= 1, got {x.shape}")
        if int(x.shape[-1]) != int(self.d_model):
            raise ValueError(
                f"Expected x last dim {int(self.d_model)}, got {x.shape}"
            )

        x_f = x.float()
        inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + float(self.eps))
        y = (x_f * inv_rms).to(dtype=x.dtype)
        if self.weight is not None:
            y = y * self.weight
        return y

