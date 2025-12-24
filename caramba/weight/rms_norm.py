"""
rms_norm provides RMSNorm weight containers.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.init as init
from typing_extensions import override

from caramba.weight.guard import require_int


class RMSNormWeight(nn.Module):
    """
    RMSNormWeight stores the RMSNorm scale.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model: int = require_int("d_model", d_model, ge=1)

        self.weight: nn.Parameter = nn.Parameter(torch.empty((self.d_model,)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        reset RMSNorm parameters.
        """
        init.ones_(self.weight)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = x
        raise RuntimeError(
            "RMSNormWeight is a weight container; call RMSNorm.forward."
        )


