"""kvcache provides config models for KV-cache policies."""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config, NonNegativeInt, PositiveInt


class KVCacheKind(str, enum.Enum):
    """KVCacheKind enumerates the KV-cache storage formats."""
    FP16 = "fp16"
    FP32 = "fp32"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    NF4 = "nf4"


CACHE_MODULE_NAME = "caramba.cache"


class KVCacheType(str, enum.Enum):
    """KVCacheType enumerates the KV-cache policy types."""
    STANDARD = "LayerKVCache"
    DECOUPLED = "DecoupledLayerKVCache"

    @staticmethod
    def module_name() -> str:
        """Returns the module name for the cache type."""
        return CACHE_MODULE_NAME


class KVCacheTensorConfig(Config):
    """KVCacheTensorConfig defines how a cache tensor is stored."""
    kind: KVCacheKind = KVCacheKind.FP16
    qblock: PositiveInt = 32
    residual_len: NonNegativeInt = 0


class KVCachePolicyConfig(Config):
    """KVCachePolicyConfig defines storage for standard (single K/V) caches."""
    type: Literal[KVCacheType.STANDARD] = KVCacheType.STANDARD
    k: KVCacheTensorConfig = KVCacheTensorConfig()
    v: KVCacheTensorConfig = KVCacheTensorConfig()


class KVCachePolicyDecoupledConfig(Config):
    """KVCachePolicyDecoupledConfig defines storage for decoupled (k_sem/k_geo/v) caches."""
    type: Literal[KVCacheType.DECOUPLED] = KVCacheType.DECOUPLED
    k_sem: KVCacheTensorConfig = KVCacheTensorConfig()
    k_geo: KVCacheTensorConfig = KVCacheTensorConfig()
    v: KVCacheTensorConfig = KVCacheTensorConfig()


KVCacheConfig: TypeAlias = Annotated[
    KVCachePolicyConfig | KVCachePolicyDecoupledConfig,
    Field(discriminator="type"),
]
