"""Layer configuration with discriminated union."""
from __future__ import annotations

import enum
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from caramba.config.operation import (
    AttentionOperationConfig,
    DropoutOperationConfig,
    LayerNormOperationConfig,
    MatmulOperationConfig,
    MultiheadOperationConfig,
    RMSNormOperationConfig,
    SwiGLUOperationConfig,
)
from caramba.config.weight import (
    DecoupledAttentionWeightConfig,
    DenseWeightConfig,
    LlamaAttentionWeightConfig,
    MultiheadWeightConfig,
    NormWeightConfig,
    RMSNormWeightConfig,
    SwiGLUWeightConfig,
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


def _normalize_layer(v: object) -> object:
    """
    _normalize_layer normalizes the discriminated union "type" field.
    """
    if isinstance(v, BaseModel):
        return v

    if isinstance(v, dict):
        layer_type = v.get("type")
        if isinstance(layer_type, str):
            out = dict(v)
            out["type"] = LayerType.from_str(layer_type)
            return out
        if isinstance(layer_type, LayerType):
            return v
        raise TypeError(
            "LayerConfig must be a dict with a 'type' of str or LayerType, "
            "or a pydantic BaseModel; got "
            f"{v!r}."
        )

    raise TypeError(
        "LayerConfig must be a dict with a 'type' of str or LayerType, "
        "or a pydantic BaseModel; got "
        f"{v!r}."
    )


class _LayerConfigBase(BaseModel):
    """
    _LayerConfigBase provides the base type for layer configs.
    """


class LinearLayerConfig(_LayerConfigBase):
    """
    LinearLayerConfig provides the linear layer configuration.
    """
    type: Literal[LayerType.LINEAR] = LayerType.LINEAR
    operation: MatmulOperationConfig
    weight: DenseWeightConfig


class LayerNormLayerConfig(_LayerConfigBase):
    """
    LayerNormLayerConfig provides the layer norm layer configuration.
    """
    type: Literal[LayerType.LAYER_NORM] = LayerType.LAYER_NORM
    operation: LayerNormOperationConfig
    weight: NormWeightConfig


class RMSNormLayerConfig(_LayerConfigBase):
    """
    RMSNormLayerConfig provides the RMS norm layer configuration.
    """
    type: Literal[LayerType.RMS_NORM] = LayerType.RMS_NORM
    operation: RMSNormOperationConfig
    weight: RMSNormWeightConfig


class MultiheadLayerConfig(_LayerConfigBase):
    """
    MultiheadLayerConfig provides the multihead layer configuration.
    """
    type: Literal[LayerType.MULTIHEAD] = LayerType.MULTIHEAD
    operation: MultiheadOperationConfig
    weight: MultiheadWeightConfig


class DropoutLayerConfig(_LayerConfigBase):
    """
    DropoutLayerConfig provides the dropout layer configuration.
    """
    type: Literal[LayerType.DROPOUT] = LayerType.DROPOUT
    operation: DropoutOperationConfig


class AttentionLayerConfig(_LayerConfigBase):
    """
    AttentionLayerConfig provides the attention layer configuration.
    """
    type: Literal[LayerType.ATTENTION] = LayerType.ATTENTION
    operation: AttentionOperationConfig
    weight: LlamaAttentionWeightConfig | DecoupledAttentionWeightConfig


class SwiGLULayerConfig(_LayerConfigBase):
    """
    SwiGLULayerConfig provides the SwiGLU layer configuration.
    """
    type: Literal[LayerType.SWIGLU] = LayerType.SWIGLU
    operation: SwiGLUOperationConfig
    weight: SwiGLUWeightConfig


LayerConfig = Annotated[
    LinearLayerConfig
    | LayerNormLayerConfig
    | RMSNormLayerConfig
    | MultiheadLayerConfig
    | DropoutLayerConfig
    | AttentionLayerConfig
    | SwiGLULayerConfig,
    BeforeValidator(_normalize_layer),
    Field(discriminator="type"),
]
