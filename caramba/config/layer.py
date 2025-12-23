"""
layer provides the layer configuration.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias
from pydantic import BaseModel, Field


class LayerType(str, enum.Enum):
    """
    LayerType provides the layer type.
    """
    LAYER_NORM = "layer_norm"
    LINEAR = "linear"
    SEQUENTIAL = "sequential"
    CONVOLUTIONAL = "convolutional"
    MULTIHEAD = "multihead"
    POOLING = "pooling"
    NORMALIZATION = "normalization"
    DROPOUT = "dropout"


class _LayerConfigBase(BaseModel):
    pass


class LinearLayerConfig(_LayerConfigBase):
    """
    LinearLayerConfig provides the linear layer configuration.
    """
    type: Literal[LayerType.LINEAR] = LayerType.LINEAR
    d_in: int
    d_out: int
    bias: bool = True


class SequentialLayerConfig(_LayerConfigBase):
    """
    SequentialLayerConfig provides the sequential layer configuration.
    """
    type: Literal[LayerType.SEQUENTIAL] = LayerType.SEQUENTIAL
    layers: list[LayerConfig]


class LayerNormLayerConfig(_LayerConfigBase):
    """
    LayerNormLayerConfig provides the layer normalization layer configuration.
    """
    type: Literal[LayerType.LAYER_NORM] = LayerType.LAYER_NORM
    d_model: int
    eps: float = 1e-5
    elementwise_affine: bool = True


class MultiheadLayerConfig(_LayerConfigBase):
    """
    MultiheadLayerConfig provides the multihead layer configuration.
    """
    type: Literal[LayerType.MULTIHEAD] = LayerType.MULTIHEAD
    d_model: int
    n_heads: int
    dropout: float = 0.0


class DropoutLayerConfig(_LayerConfigBase):
    """
    DropoutLayerConfig provides the dropout layer configuration.
    """
    type: Literal[LayerType.DROPOUT] = LayerType.DROPOUT
    p: float = 0.0


LayerConfig: TypeAlias = Annotated[
    LinearLayerConfig
    | LayerNormLayerConfig
    | MultiheadLayerConfig
    | SequentialLayerConfig
    | DropoutLayerConfig,
    Field(discriminator="type"),
]
