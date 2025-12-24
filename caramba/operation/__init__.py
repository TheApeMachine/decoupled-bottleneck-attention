"""
operation provides parameter-free compute primitives.
"""

from __future__ import annotations

from caramba.operation.attention import AttentionOp
from caramba.operation.attention_math import decoupled_qk_cat
from caramba.operation.build import (
    build_attention_operation,
    build_dropout_operation,
    build_layer_norm_operation,
    build_matmul_operation,
    build_multihead_operation,
    build_operation,
    build_rms_norm_operation,
    build_swiglu_operation,
)
from caramba.operation.dropout import Drop
from caramba.operation.layer_norm import LayerNormOp
from caramba.operation.matmul import Matmul
from caramba.operation.multihead import MultiheadOp
from caramba.operation.rope import RotaryEmbedding
from caramba.operation.swiglu import SwiGLUOp

__all__ = [
    "AttentionOp",
    "RotaryEmbedding",
    "decoupled_qk_cat",
    "build_attention_operation",
    "build_dropout_operation",
    "build_layer_norm_operation",
    "build_matmul_operation",
    "build_multihead_operation",
    "build_operation",
    "build_rms_norm_operation",
    "build_swiglu_operation",
    "Drop",
    "LayerNormOp",
    "Matmul",
    "MultiheadOp",
    "SwiGLUOp",
]
