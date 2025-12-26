"""Layer configuration with discriminated union."""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias
from pydantic import Field

from caramba.config import (
    Config, PositiveFloat, PositiveInt, Probability
)


class AttentionMode(str, enum.Enum):
    """AttentionMode selects the attention variant."""
    STANDARD = "standard"
    GQA = "gqa"
    DECOUPLED = "decoupled"


class LayerType(str, enum.Enum):
    """LayerType enumerates the layer types

    This prevents having to deal with magic strings, giving
    us compile-time safety, and better error messages.
    """
    LAYER_NORM = "LayerNormLayer"
    RMS_NORM = "RMSNormLayer"
    LINEAR = "LinearLayer"
    DROPOUT = "DropoutLayer"
    ATTENTION = "AttentionLayer"
    SWIGLU = "SwiGLULayer"

    @classmethod
    def from_str(cls, s: str) -> LayerType:
        """Converts a string to a LayerType."""
        return cls(s)

    @classmethod
    def module_name(cls) -> str:
        """Returns the module name for the layer type."""
        return "caramba.layer"


class LinearLayerConfig(Config):
    """
    LinearLayerConfig provides the linear layer configuration.
    """
    type: Literal[LayerType.LINEAR] = LayerType.LINEAR
    d_in: PositiveInt
    d_out: PositiveInt
    bias: bool = True


class LayerNormLayerConfig(Config):
    """
    LayerNormLayerConfig provides the layer norm layer configuration.
    """
    type: Literal[LayerType.LAYER_NORM] = LayerType.LAYER_NORM
    d_model: PositiveInt
    eps: PositiveFloat = 1e-5


class RMSNormLayerConfig(Config):
    """
    RMSNormLayerConfig provides the RMS norm layer configuration.
    """
    type: Literal[LayerType.RMS_NORM] = LayerType.RMS_NORM
    d_model: PositiveInt
    eps: PositiveFloat = 1e-5
    elementwise_affine: bool = True


class DropoutLayerConfig(Config):
    """
    DropoutLayerConfig provides the dropout layer configuration.
    """
    type: Literal[LayerType.DROPOUT] = LayerType.DROPOUT
    p: Probability = 0.0


class AttentionLayerConfig(Config):
    """
    AttentionLayerConfig provides unified attention configuration.

    Modes:
    - standard: Full multi-head attention (d_model -> d_model)
    - gqa: Grouped-query attention (fewer KV heads than Q heads)
    - decoupled: DBA with separate semantic/geometric key paths

    For decoupled mode, set sem_dim and geo_dim. RoPE is applied
    only to the geometric path; semantic path is position-free.
    """
    type: Literal[LayerType.ATTENTION] = LayerType.ATTENTION

    # Core dimensions
    d_model: PositiveInt
    n_heads: PositiveInt
    n_kv_heads: PositiveInt | None = None  # None = same as n_heads

    # Attention mode
    mode: AttentionMode = AttentionMode.STANDARD

    # Bottleneck dimension (optional, defaults to d_model)
    attn_dim: PositiveInt | None = None

    # Decoupled mode dimensions
    sem_dim: PositiveInt | None = None  # Semantic key dimension
    geo_dim: PositiveInt | None = None  # Geometric key dimension (RoPE applied here)

    # RoPE settings
    rope_enabled: bool = True
    rope_base: float = 10000.0

    # Decoupled gate (learned per-head sem/geo mixing)
    decoupled_gate: bool = False
    decoupled_gate_dynamic: bool = False  # Query-dependent gate

    # Optional features
    is_causal: bool = True
    dropout_p: Probability = 0.0
    bias: bool = False

    # Learned temperature per head
    learned_temp: bool = False

    @property
    def head_dim(self) -> int:
        """Compute head dimension from attn_dim or d_model."""
        dim = self.attn_dim if self.attn_dim is not None else self.d_model
        return dim // self.n_heads

    @property
    def kv_heads(self) -> int:
        """Number of KV heads (for GQA)."""
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def sem_head_dim(self) -> int | None:
        """Semantic head dimension for decoupled mode."""
        if self.sem_dim is None:
            return None
        return self.sem_dim // self.n_heads

    @property
    def geo_head_dim(self) -> int | None:
        """Geometric head dimension for decoupled mode."""
        if self.geo_dim is None:
            return None
        return self.geo_dim // self.n_heads

    @property
    def v_dim(self) -> int:
        """Value projection dimension."""
        return self.attn_dim if self.attn_dim is not None else self.d_model


class SwiGLULayerConfig(Config):
    """
    SwiGLULayerConfig provides the SwiGLU layer configuration.
    """
    type: Literal[LayerType.SWIGLU] = LayerType.SWIGLU
    d_model: PositiveInt
    d_ff: PositiveInt
    bias: bool = True


LayerConfig: TypeAlias = Annotated[
    LinearLayerConfig
    | LayerNormLayerConfig
    | RMSNormLayerConfig
    | DropoutLayerConfig
    | AttentionLayerConfig
    | SwiGLULayerConfig,
    Field(discriminator="type"),
]
