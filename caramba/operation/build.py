"""
build provides factories for operation modules from config.
"""

from __future__ import annotations

from torch import nn

from caramba.config.operation import (
    AttentionOperationConfig,
    DropoutOperationConfig,
    LayerNormOperationConfig,
    MatmulOperationConfig,
    MultiheadOperationConfig,
    OperationConfig,
    RMSNormOperationConfig,
    SwiGLUOperationConfig,
)
from caramba.operation.attention import AttentionOp
from caramba.operation.dropout import Drop
from caramba.operation.layer_norm import LayerNormOp
from caramba.operation.matmul import Matmul
from caramba.operation.multihead import MultiheadOp
from caramba.operation.rms_norm import RMSNormOp
from caramba.operation.swiglu import SwiGLUOp


def build_operation(config: OperationConfig) -> nn.Module:
    """
    build_operation builds an operation module from an OperationConfig.
    """
    match config:
        case MatmulOperationConfig():
            return Matmul()
        case LayerNormOperationConfig():
            return LayerNormOp()
        case RMSNormOperationConfig():
            return RMSNormOp()
        case DropoutOperationConfig() as c:
            return Drop(float(c.p))
        case MultiheadOperationConfig():
            return MultiheadOp()
        case AttentionOperationConfig():
            return AttentionOp()
        case SwiGLUOperationConfig():
            return SwiGLUOp()
        case _:
            raise ValueError(f"Unsupported operation config: {type(config)!r}")


def build_attention_operation(config: AttentionOperationConfig) -> AttentionOp:
    """
    build_attention_operation builds the AttentionOp module.
    """
    op = build_operation(config)
    if not isinstance(op, AttentionOp):
        raise RuntimeError(f"Expected AttentionOp, got {type(op)!r}")
    return op


def build_matmul_operation(config: MatmulOperationConfig) -> Matmul:
    """
    build_matmul_operation builds a Matmul operation.
    """
    op = build_operation(config)
    if not isinstance(op, Matmul):
        raise RuntimeError(f"Expected Matmul, got {type(op)!r}")
    return op


def build_layer_norm_operation(
    config: LayerNormOperationConfig,
) -> LayerNormOp:
    """
    build_layer_norm_operation builds a LayerNorm operation.
    """
    op = build_operation(config)
    if not isinstance(op, LayerNormOp):
        raise RuntimeError(f"Expected LayerNormOp, got {type(op)!r}")
    return op


def build_rms_norm_operation(config: RMSNormOperationConfig) -> RMSNormOp:
    """
    build_rms_norm_operation builds an RMSNorm operation.
    """
    op = build_operation(config)
    if not isinstance(op, RMSNormOp):
        raise RuntimeError(f"Expected RMSNormOp, got {type(op)!r}")
    return op


def build_dropout_operation(config: DropoutOperationConfig) -> Drop:
    """
    build_dropout_operation builds a Dropout operation.
    """
    op = build_operation(config)
    if not isinstance(op, Drop):
        raise RuntimeError(f"Expected Drop, got {type(op)!r}")
    return op


def build_multihead_operation(config: MultiheadOperationConfig) -> MultiheadOp:
    """
    build_multihead_operation builds a Multihead operation.
    """
    op = build_operation(config)
    if not isinstance(op, MultiheadOp):
        raise RuntimeError(f"Expected MultiheadOp, got {type(op)!r}")
    return op


def build_swiglu_operation(config: SwiGLUOperationConfig) -> SwiGLUOp:
    """
    build_swiglu_operation builds a SwiGLU operation.
    """
    op = build_operation(config)
    if not isinstance(op, SwiGLUOp):
        raise RuntimeError(f"Expected SwiGLUOp, got {type(op)!r}")
    return op
