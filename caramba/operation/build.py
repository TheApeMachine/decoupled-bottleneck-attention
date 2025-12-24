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
            raise ValueError(
                "SwiGLU is a composite op implemented by the SwiGLU layer; "
                "there is no standalone SwiGLU operation module."
            )
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


