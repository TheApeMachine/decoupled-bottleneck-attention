"""KV-cache configuration for inference optimization.

The KV-cache stores key and value tensors from previous tokens during
generation, avoiding recomputation. Different precision formats trade
off memory vs. quality, and DBA models have separate semantic/geometric
caches for further compression.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config, NonNegativeInt, PositiveInt


class KVCacheKind(str, enum.Enum):
    """Storage format for cached tensors.

    FP16/FP32: Full precision (highest quality, most memory)
    Q8_0: 8-bit quantized (good balance)
    Q4_0/NF4: 4-bit quantized (most compression, some quality loss)
    """

    FP16 = "fp16"
    FP32 = "fp32"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    NF4 = "nf4"


CACHE_MODULE_NAME = "caramba.cache"


class KVCacheType(str, enum.Enum):
    """Type of KV-cache for different attention modes.

    STANDARD: Single K/V tensors (standard/GQA attention)
    DECOUPLED: Separate k_sem/k_geo/v tensors (DBA attention)
    """

    STANDARD = "LayerKVCache"
    DECOUPLED = "DecoupledLayerKVCache"

    @staticmethod
    def module_name() -> str:
        """Return the Python module containing cache implementations."""
        return CACHE_MODULE_NAME


class KVCacheTensorConfig(Config):
    """Configuration for how a single cache tensor is stored.

    Can specify quantization format, block size, and residual storage
    for hybrid quantization schemes.
    """

    kind: KVCacheKind = KVCacheKind.FP16
    qblock: PositiveInt = 32
    residual_len: NonNegativeInt = 0


class KVCachePolicyConfig(Config):
    """Cache policy for standard attention (single K/V per layer)."""

    type: Literal[KVCacheType.STANDARD] = KVCacheType.STANDARD
    k: KVCacheTensorConfig = KVCacheTensorConfig()
    v: KVCacheTensorConfig = KVCacheTensorConfig()


class KVCachePolicyDecoupledConfig(Config):
    """Cache policy for DBA attention (separate semantic/geometric keys).

    DBA uses three cached tensors: k_sem (semantic keys), k_geo (geometric
    keys), and v (values). This enables aggressive compression since
    semantic and geometric keys can use different formats.
    """

    type: Literal[KVCacheType.DECOUPLED] = KVCacheType.DECOUPLED
    k_sem: KVCacheTensorConfig = KVCacheTensorConfig()
    k_geo: KVCacheTensorConfig = KVCacheTensorConfig()
    v: KVCacheTensorConfig = KVCacheTensorConfig()


# Union type for any cache config
KVCacheConfig: TypeAlias = Annotated[
    KVCachePolicyConfig | KVCachePolicyDecoupledConfig,
    Field(discriminator="type"),
]
