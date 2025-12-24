"""
attention provides an attention layer (operation + weight strategy).
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import AttentionLayerConfig
from caramba.operation.attention import AttentionOp
from caramba.operation.build import build_attention_operation
from caramba.operation.matmul import Matmul
from caramba.weight.build import build_attention_weight
from caramba.weight.dense import DenseWeight


class _AttentionWeight(Protocol):
    """
    _AttentionWeight defines the minimum contract Attention requires from weights.
    """

    o_proj: DenseWeight

    def project_qkv(
        self,
        x: Tensor,
        *,
        matmul: Matmul,
        pos_offset: int,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class Attention(nn.Module):
    """
    Attention provides a causal self-attention layer with pluggable weights.

    Distillation hooks can capture per-layer attention outputs by registering
    a forward hook on this module (post output-projection), or on `operation`
    (raw head outputs).
    """

    def __init__(self, config: AttentionLayerConfig) -> None:
        super().__init__()
        self.config: AttentionLayerConfig = config
        self.matmul: Matmul = Matmul()
        self.operation: AttentionOp = build_attention_operation(config.operation)
        self.is_causal: bool = bool(config.operation.is_causal)
        self.dropout_p: float = float(config.operation.dropout_p)
        self.weight: _AttentionWeight = build_attention_weight(config.weight)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        pos_offset: int = 0,
    ) -> Tensor:
        """
        forward pass for attention.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {x.shape}")

        dropout_p = float(self.dropout_p) if self.training else 0.0
        q, k, v = self.weight.project_qkv(
            x,
            matmul=self.matmul,
            pos_offset=int(pos_offset),
        )
        out_h = self.operation.forward(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=bool(self.is_causal),
        )
        out = self._merge(out_h)
        return self.matmul.forward(
            out,
            weight=self.weight.o_proj.weight,
            bias=self.weight.o_proj.bias,
        )

    def _merge(self, x: Tensor) -> Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, int(h) * int(d))