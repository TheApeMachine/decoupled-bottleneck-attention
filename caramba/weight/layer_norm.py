"""
layer_norm provides layer norm weight containers.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.init as init
from typing_extensions import override

from caramba.weight.guard import require_bool, require_int


class LayerNormWeight(nn.Module):
    """
    LayerNormWeight stores scale and bias for layer norm.
    """

    def __init__(
        self,
        d_model: int,
        *,
        elementwise_affine: bool,
    ) -> None:
        super().__init__()
        self.d_model: int = require_int("d_model", d_model, ge=1)
        self.elementwise_affine: bool = require_bool(
            "elementwise_affine",
            elementwise_affine,
        )

        self.weight: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((self.d_model,)))
            self.bias = nn.Parameter(torch.empty((self.d_model,)))
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        reset normalization parameters.
        """
        if self.weight is not None:
            init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = x
        raise RuntimeError(
            "LayerNormWeight is a weight container; call Normalize.forward."
        )


