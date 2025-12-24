"""Layer configuration with discriminated union."""
from __future__ import annotations

import enum
from typing import Annotated

from pydantic import BaseModel, Field, BeforeValidator

from caramba.config.operation import (
    AttentionOperationConfig, DropoutOperationConfig, LayerNormOperationConfig,
    MatmulOperationConfig, MultiheadOperationConfig, RMSNormOperationConfig,
    SwiGLUOperationConfig,
)
from caramba.config.weight import (
    DecoupledAttentionWeightConfig, DenseWeightConfig, LlamaAttentionWeightConfig,
    MultiheadWeightConfig, NormWeightConfig, RMSNormWeightConfig, SwiGLUWeightConfig,
)

class LayerType(str, enum.Enum):
    """
    LayerType enumerates the layer types

    This prevents having to deal with magic strings, giving
    us compile-time safety, and better error messages.
    """
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    LINEAR = "linear"
    MULTIHEAD = "multihead"
    DROPOUT = "dropout"
    ATTENTION = "attention"
    SWIGLU = "swiglu"

    @classmethod
    def from_str(cls, s: str) -> LayerType:
        """
        from_str converts a string to a LayerType.
        """
        return cls(s)


def _normalize_layer(v: dict) -> dict:
    if isinstance(v, dict) and isinstance(v.get("type"), str):
        v["type"] = LayerType.from_str(v["type"])
    return v


class LinearLayerConfig(BaseModel):
    """
    LinearLayerConfig provides the linear layer configuration.
    """
    type: LayerType = LayerType.LINEAR
    operation: MatmulOperationConfig
    weight: DenseWeightConfig


class LayerNormLayerConfig(BaseModel):
    """
    LayerNormLayerConfig provides the layer norm layer configuration.
    """
    type: LayerType = LayerType.LAYER_NORM
    operation: LayerNormOperationConfig
    weight: NormWeightConfig


class RMSNormLayerConfig(BaseModel):
    """
    RMSNormLayerConfig provides the RMS norm layer configuration.
    """
    type: LayerType = LayerType.RMS_NORM
    operation: RMSNormOperationConfig
    weight: RMSNormWeightConfig


class MultiheadLayerConfig(BaseModel):
    """
    MultiheadLayerConfig provides the multihead layer configuration.
    """
    type: LayerType = LayerType.MULTIHEAD
    operation: MultiheadOperationConfig
    weight: MultiheadWeightConfig


class DropoutLayerConfig(BaseModel):
    """
    DropoutLayerConfig provides the dropout layer configuration.
    """
    type: LayerType = LayerType.DROPOUT
    operation: DropoutOperationConfig


class AttentionLayerConfig(BaseModel):
    """
    AttentionLayerConfig provides the attention layer configuration.
    """
    type: LayerType = LayerType.ATTENTION
    operation: AttentionOperationConfig
    weight: LlamaAttentionWeightConfig | DecoupledAttentionWeightConfig


class SwiGLULayerConfig(BaseModel):
    """
    SwiGLULayerConfig provides the SwiGLU layer configuration.
    """
    type: LayerType = LayerType.SWIGLU
    operation: SwiGLUOperationConfig
    weight: SwiGLUWeightConfig


LayerConfig = Annotated[
    LinearLayerConfig | LayerNormLayerConfig | RMSNormLayerConfig |
    MultiheadLayerConfig | DropoutLayerConfig | AttentionLayerConfig |
    SwiGLULayerConfig,
    BeforeValidator(_normalize_layer),
    Field(discriminator="type"),
]