"""
multihead provides multihead attention weight containers.
"""

from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.weight.guard import require_float, require_int


class MultiheadWeight(nn.Module):
    """
    MultiheadWeight stores the MultiheadAttention parameters.
    """

    def __init__(self, *, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        embed_dim = require_int("d_model", d_model, ge=1)
        num_heads = require_int("n_heads", n_heads, ge=1)
        p = require_float("dropout", dropout)
        if p < 0.0:
            raise ValueError(f"dropout must be >= 0, got {p}")

        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=p,
            batch_first=True,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = x
        raise RuntimeError(
            "MultiheadWeight is a weight container; call Multihead.forward."
        )


