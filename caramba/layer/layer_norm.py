"""Standard LayerNorm layer.

LayerNorm normalizes each sample independently across the feature dimension,
centering (subtracting mean) and scaling (dividing by std). While RMSNorm
is more common in modern LLMs, LayerNorm is still used in some architectures.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import LayerNormLayerConfig


class LayerNormLayer(nn.Module):
    """Standard layer normalization wrapping nn.LayerNorm.

    Normalizes each sample to zero mean and unit variance across the
    last dimension, then applies a learnable affine transform.
    """

    def __init__(self, config: LayerNormLayerConfig) -> None:
        """Initialize LayerNorm with the given dimension and epsilon.

        Args:
            config: Specifies d_model (normalized dimension) and eps for
                   numerical stability.
        """
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(
            config.d_model,
            eps=float(config.eps),
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply layer normalization."""
        return self.norm(x)
