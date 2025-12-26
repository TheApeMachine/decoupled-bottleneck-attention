"""Simple linear projection layer.

This wraps nn.Linear with our standard layer interface (config-based
construction, optional ctx parameter). Used for the LM head and other
projections that don't need special handling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import LinearLayerConfig

try:
    from tensordict import TensorDictBase as _TensorDictBase  # type: ignore[import-not-found]
except ImportError:
    _TensorDictBase = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from tensordict import TensorDictBase  # type: ignore[import-not-found]


class LinearLayer(nn.Module):
    """A linear projection with our standard layer interface.

    Wraps nn.Linear so it can be constructed from config and used
    consistently with other layer types.
    """

    def __init__(self, config: LinearLayerConfig) -> None:
        """Create a linear projection.

        Args:
            config: Specifies input dim (d_in), output dim (d_out), and bias.
        """
        super().__init__()
        self.config = config
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
        """Apply the linear projection."""
        return self.linear(x)
