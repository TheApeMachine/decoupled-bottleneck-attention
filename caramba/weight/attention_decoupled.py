"""
attention_decoupled provides decoupled bottleneck attention (DBA) weights.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.operation.attention_math import decoupled_qk_cat, shape_heads
from caramba.operation.matmul import Matmul
from caramba.operation.rope import RotaryEmbedding
from caramba.weight.dense import DenseWeight
from caramba.weight.guard import require_bool, require_float, require_int


class DecoupledAttentionWeight(nn.Module):
    """
    DecoupledAttentionWeight stores semantic/geometric QK projections plus V/O.

    RoPE is intended to be applied only to the geometric path.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        sem_dim: int,
        geo_dim: int,
        rope_base: float,
        rope_dim: int,
        bias: bool,
        gate: bool,
    ) -> None:
        super().__init__()
        self.d_model: int = require_int("d_model", d_model, ge=1)
        self.n_heads: int = require_int("n_heads", n_heads, ge=1)
        self.n_kv_heads: int = require_int("n_kv_heads", n_kv_heads, ge=1)
        self.sem_dim: int = require_int("sem_dim", sem_dim, ge=1)
        self.geo_dim: int = require_int("geo_dim", geo_dim, ge=1)
        rope_base_ = require_float("rope_base", rope_base)
        rope_dim_ = require_int("rope_dim", rope_dim, ge=1)
        bias_ = require_bool("bias", bias)
        gate_ = require_bool("gate", gate)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads % n_kv_heads must be 0, got {self.n_heads} % {self.n_kv_heads}"
            )
        if self.sem_dim % self.n_heads != 0:
            raise ValueError(
                f"sem_dim must be divisible by n_heads, got sem_dim={self.sem_dim}, "
                f"n_heads={self.n_heads}"
            )
        if self.geo_dim % self.n_heads != 0:
            raise ValueError(
                f"geo_dim must be divisible by n_heads, got geo_dim={self.geo_dim}, "
                f"n_heads={self.n_heads}"
            )

        self.head_dim: int = self.d_model // self.n_heads
        self.sem_head_dim: int = self.sem_dim // self.n_heads
        self.geo_head_dim: int = self.geo_dim // self.n_heads

        if rope_dim_ > self.geo_head_dim:
            raise ValueError(
                f"rope_dim ({rope_dim_}) must be <= geo_head_dim ({self.geo_head_dim})"
            )
        if rope_dim_ % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim_}")

        k_sem_dim = self.n_kv_heads * self.sem_head_dim
        k_geo_dim = self.n_kv_heads * self.geo_head_dim

        self.q_sem: DenseWeight = DenseWeight(
            self.d_model,
            self.n_heads * self.sem_head_dim,
            bias=bias_,
        )
        self.k_sem: DenseWeight = DenseWeight(
            self.d_model,
            k_sem_dim,
            bias=bias_,
        )
        self.q_geo: DenseWeight = DenseWeight(
            self.d_model,
            self.n_heads * self.geo_head_dim,
            bias=bias_,
        )
        self.k_geo: DenseWeight = DenseWeight(
            self.d_model,
            k_geo_dim,
            bias=bias_,
        )

        self.v_proj: DenseWeight = DenseWeight(
            self.d_model,
            self.n_kv_heads * self.head_dim,
            bias=bias_,
        )
        self.o_proj: DenseWeight = DenseWeight(
            self.n_heads * self.head_dim,
            self.d_model,
            bias=bias_,
        )

        self.rope: RotaryEmbedding = RotaryEmbedding(
            rope_dim_,
            base=rope_base_,
        )

        self.gate_enabled: bool = gate_
        self.gate_logit: nn.Parameter | None = (
            nn.Parameter(torch.zeros(self.n_heads))
            if self.gate_enabled
            else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = x
        raise RuntimeError(
            "DecoupledAttentionWeight is a weight container; call Attention.forward."
        )

    def project_qkv(
        self,
        x: Tensor,
        *,
        matmul: Matmul,
        pos_offset: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        project_qkv projects x into semantic/geometric Q/K plus V, then builds a
        composite (q, k) suitable for attention.

        Returns (q, k, v) in shaped form:
          - q: (B, H, T, sem_head_dim + geo_head_dim)
          - k: (B, H_kv, T, sem_head_dim + geo_head_dim)
          - v: (B, H_kv, T, head_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {x.shape}")

        q_sem = matmul.forward(x, weight=self.q_sem.weight, bias=self.q_sem.bias)
        k_sem = matmul.forward(x, weight=self.k_sem.weight, bias=self.k_sem.bias)
        q_geo = matmul.forward(x, weight=self.q_geo.weight, bias=self.q_geo.bias)
        k_geo = matmul.forward(x, weight=self.k_geo.weight, bias=self.k_geo.bias)
        v = matmul.forward(x, weight=self.v_proj.weight, bias=self.v_proj.bias)

        qsh = shape_heads(
            q_sem,
            n_heads=int(self.n_heads),
            head_dim=int(self.sem_head_dim),
        )
        ksh = shape_heads(
            k_sem,
            n_heads=int(self.n_kv_heads),
            head_dim=int(self.sem_head_dim),
        )
        qgh = shape_heads(
            q_geo,
            n_heads=int(self.n_heads),
            head_dim=int(self.geo_head_dim),
        )
        kgh = shape_heads(
            k_geo,
            n_heads=int(self.n_kv_heads),
            head_dim=int(self.geo_head_dim),
        )
        vh = shape_heads(v, n_heads=int(self.n_kv_heads), head_dim=int(self.head_dim))

        qgh = self.rope.forward(qgh, pos_offset=int(pos_offset))
        kgh = self.rope.forward(kgh, pos_offset=int(pos_offset))

        if self.gate_logit is not None:
            gate = torch.sigmoid(self.gate_logit).view(1, -1, 1, 1).to(dtype=x.dtype)
            qsh = qsh * (2.0 * gate)
            qgh = qgh * (2.0 - 2.0 * gate)

        sem_scale = 1.0 / math.sqrt(float(self.sem_head_dim))
        geo_scale = 1.0 / math.sqrt(float(self.geo_head_dim))
        q_cat, k_cat = decoupled_qk_cat(
            q_sem=qsh,
            q_geo=qgh,
            k_sem=ksh,
            k_geo=kgh,
            sem_scale=float(sem_scale),
            geo_scale=float(geo_scale),
        )
        return q_cat, k_cat, vh


