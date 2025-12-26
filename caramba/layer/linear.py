"""
linear provides the linear layer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import LinearLayerConfig

try:
    from tensordict import TensorDictBase as _TensorDictBase  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _TensorDictBase = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from tensordict import TensorDictBase  # type: ignore[import-not-found]


class LinearLayer(nn.Module):
    """
    Linear provides the linear layer.
    """

    def __init__(self, config: LinearLayerConfig) -> None:
        super().__init__()
        self.config: LinearLayerConfig = config
        self.linear = nn.Linear(
            config.d_in,
            config.d_out,
            bias=bool(config.bias),
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """
        forward pass for the linear layer.
        """
        return self.linear(x)
