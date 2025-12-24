"""
operation provides parameter-free compute primitives.
"""

from __future__ import annotations

from caramba.operation.attention import AttentionOp
from caramba.operation.attention_math import decoupled_qk_cat
from caramba.operation.build import build_attention_operation, build_operation
from caramba.operation.dropout import Drop
from caramba.operation.layer_norm import LayerNormOp
from caramba.operation.matmul import Matmul
from caramba.operation.multihead import MultiheadOp
from caramba.operation.rope import RotaryEmbedding

__all__ = [
    "AttentionOp",
    "RotaryEmbedding",
    "decoupled_qk_cat",
    "build_attention_operation",
    "build_operation",
    "Drop",
    "LayerNormOp",
    "Matmul",
    "MultiheadOp",
]


