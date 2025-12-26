"""Layer configuration with discriminated unions.

Each layer type (attention, MLP, normalization) has its own config class.
Pydantic's discriminated unions allow YAML like `type: AttentionLayer` to
automatically deserialize into the correct config class.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config, PositiveFloat, PositiveInt, Probability


class AttentionMode(str, enum.Enum):
    """Which attention variant to use.

    STANDARD: Full multi-head attention
    GQA: Grouped-query attention (shared KV heads)
    DECOUPLED: DBA with separate semantic/geometric paths
    """

    STANDARD = "standard"
    GQA = "gqa"
    DECOUPLED = "decoupled"


class LayerType(str, enum.Enum):
    """Enumeration of layer types for type-safe config parsing.

    Using an enum prevents magic strings and gives better error messages
    when an unknown layer type is specified in YAML.
    """

    LAYER_NORM = "LayerNormLayer"
    RMS_NORM = "RMSNormLayer"
    LINEAR = "LinearLayer"
    DROPOUT = "DropoutLayer"
    ATTENTION = "AttentionLayer"
    SWIGLU = "SwiGLULayer"

    @classmethod
    def from_str(cls, s: str) -> "LayerType":
        """Convert a string to a LayerType."""
        return cls(s)

    @staticmethod
    def module_name() -> str:
        """Return the Python module containing layer implementations."""
        return "caramba.layer"


class LinearLayerConfig(Config):
    """Configuration for a simple linear projection."""

    type: Literal[LayerType.LINEAR] = LayerType.LINEAR
    d_in: PositiveInt
    d_out: PositiveInt
    bias: bool = True


class LayerNormLayerConfig(Config):
    """Configuration for standard LayerNorm."""

    type: Literal[LayerType.LAYER_NORM] = LayerType.LAYER_NORM
    d_model: PositiveInt
    eps: PositiveFloat = 1e-5


class RMSNormLayerConfig(Config):
    """Configuration for RMSNorm (used in Llama and modern LLMs)."""

    type: Literal[LayerType.RMS_NORM] = LayerType.RMS_NORM
    d_model: PositiveInt
    eps: PositiveFloat = 1e-5
    elementwise_affine: bool = True


class DropoutLayerConfig(Config):
    """Configuration for dropout regularization."""

    type: Literal[LayerType.DROPOUT] = LayerType.DROPOUT
    p: Probability = 0.0


class AttentionLayerConfig(Config):
    """Configuration for unified attention (standard/GQA/DBA).

    The mode field selects the attention variant:
    - standard: Each head has its own Q/K/V projections
    - gqa: Multiple Q heads share K/V heads
    - decoupled: Separate semantic (content) and geometric (position) paths

    For DBA (decoupled), set sem_dim and geo_dim. RoPE is only applied
    to the geometric path; the semantic path is position-invariant.
    """

    type: Literal[LayerType.ATTENTION] = LayerType.ATTENTION

    # Core dimensions
    d_model: PositiveInt
    n_heads: PositiveInt
    n_kv_heads: PositiveInt | None = None

    # Attention mode
    mode: AttentionMode = AttentionMode.STANDARD

    # Optional attention bottleneck dimension
    attn_dim: PositiveInt | None = None

    # DBA dimensions (only used when mode=decoupled)
    sem_dim: PositiveInt | None = None
    geo_dim: PositiveInt | None = None

    # RoPE settings
    rope_enabled: bool = True
    rope_base: float = 10000.0

    # DBA gating (learned per-head semantic/geometric mixing)
    decoupled_gate: bool = False
    decoupled_gate_dynamic: bool = False

    # Standard attention settings
    is_causal: bool = True
    dropout_p: Probability = 0.0
    bias: bool = False
    learned_temp: bool = False

    @property
    def head_dim(self) -> int:
        """Compute head dimension from total attention dimension."""
        dim = self.attn_dim if self.attn_dim is not None else self.d_model
        return dim // self.n_heads

    @property
    def kv_heads(self) -> int:
        """Number of KV heads (for GQA)."""
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def sem_head_dim(self) -> int | None:
        """Per-head semantic dimension for DBA."""
        if self.sem_dim is None:
            return None
        return self.sem_dim // self.n_heads

    @property
    def geo_head_dim(self) -> int | None:
        """Per-head geometric dimension for DBA."""
        if self.geo_dim is None:
            return None
        return self.geo_dim // self.n_heads

    @property
    def v_dim(self) -> int:
        """Value projection dimension."""
        return self.attn_dim if self.attn_dim is not None else self.d_model


class SwiGLULayerConfig(Config):
    """Configuration for SwiGLU MLP (gate/up/down projections)."""

    type: Literal[LayerType.SWIGLU] = LayerType.SWIGLU
    d_model: PositiveInt
    d_ff: PositiveInt
    bias: bool = True


# Union type for any layer config, with automatic deserialization
LayerConfig: TypeAlias = Annotated[
    LinearLayerConfig
    | LayerNormLayerConfig
    | RMSNormLayerConfig
    | DropoutLayerConfig
    | AttentionLayerConfig
    | SwiGLULayerConfig,
    Field(discriminator="type"),
]
