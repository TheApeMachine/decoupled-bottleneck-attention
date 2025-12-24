"""
attention_llama provides Llama-compatible attention weights (GQA + RoPE).
"""

from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.operation.attention_math import shape_heads
from caramba.operation.matmul import Matmul
from caramba.operation.rope import RotaryEmbedding
from caramba.weight.dense import DenseWeight
from caramba.weight.guard import require_bool, require_float, require_int


class LlamaAttentionWeight(nn.Module):
    """
    LlamaAttentionWeight stores Q/K/V/O projection weights for
    Llama-style GQA attention with RoPE.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        rope_base: float,
        rope_dim: int,
        bias: bool,
    ) -> None:
        super().__init__()

        d_model_ = require_int("d_model", d_model, ge=1)
        n_heads_ = require_int("n_heads", n_heads, ge=1)
        n_kv_heads_ = require_int("n_kv_heads", n_kv_heads, ge=1)
        rope_dim_ = require_int("rope_dim", rope_dim, ge=1)
        rope_base_ = require_float("rope_base", rope_base)
        bias_ = require_bool("bias", bias)

        self._validate(d_model_, n_heads_, n_kv_heads_, rope_dim_)

        self.d_model: int = d_model_
        self.n_heads: int = n_heads_
        self.n_kv_heads: int = n_kv_heads_
        self.head_dim: int = d_model_ // n_heads_
        self.rope_dim: int = rope_dim_

        def dense(d_in: int, d_out: int) -> DenseWeight:
            return DenseWeight(d_in, d_out, bias=bias_)

        q_out = n_heads * self.head_dim
        kv_out = n_kv_heads * self.head_dim

        self.q_proj: DenseWeight = dense(d_model_, q_out)
        self.k_proj: DenseWeight = dense(d_model_, kv_out)
        self.v_proj: DenseWeight = dense(d_model_, kv_out)
        self.o_proj: DenseWeight = dense(q_out, d_model_)

        self.rope: RotaryEmbedding = RotaryEmbedding(rope_dim_, base=rope_base_)

    @staticmethod
    def _validate(d_model: int, n_heads: int, n_kv_heads: int, rope_dim: int) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0, got {n_kv_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        head_dim = d_model // n_heads
        if rope_dim <= 0:
            raise ValueError(f"rope_dim must be > 0, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim ({rope_dim}) must be <= head_dim ({head_dim})")
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the llama attention weight.
        """
        _ = x
        raise RuntimeError(
            "LlamaAttentionWeight is a weight container; call Attention.forward."
        )

    def project_qkv(
        self,
        x: Tensor,
        *,
        matmul: Matmul,
        pos_offset: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        project_qkv projects x into attention heads and applies RoPE.

        Returns (q, k, v) in shaped form:
          - q: (B, H, T, D_head)
          - k: (B, H_kv, T, D_head)
          - v: (B, H_kv, T, D_head)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {x.shape}")

        q = matmul.forward(x, weight=self.q_proj.weight, bias=self.q_proj.bias)
        k = matmul.forward(x, weight=self.k_proj.weight, bias=self.k_proj.bias)
        v = matmul.forward(x, weight=self.v_proj.weight, bias=self.v_proj.bias)

        qh = shape_heads(q, n_heads=int(self.n_heads), head_dim=int(self.head_dim))
        kh = shape_heads(k, n_heads=int(self.n_kv_heads), head_dim=int(self.head_dim))
        vh = shape_heads(v, n_heads=int(self.n_kv_heads), head_dim=int(self.head_dim))

        qh = self.rope.forward(qh, pos_offset=int(pos_offset))
        kh = self.rope.forward(kh, pos_offset=int(pos_offset))
        return qh, kh, vh
