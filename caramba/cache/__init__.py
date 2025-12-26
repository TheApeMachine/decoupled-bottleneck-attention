"""KV cache construction and management.

During inference, attention layers need access to all past keys and values.
Recomputing them each step would be O(n²)—instead we cache them. This
package provides:
- LayerKVCache: Standard K/V caches for standard/GQA attention
- DecoupledLayerKVCache: Separate k_sem/k_geo/v caches for DBA
- Quantized storage (q8_0, q4_0, nf4) for memory efficiency
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from caramba.config.kvcache import KVCacheTensorConfig


@dataclass(frozen=True)
class QuantSpec:
    """Quantization storage geometry for a cache tensor.

    Describes how a dimension is padded and blocked for quantization.
    """

    kind: str
    dim: int
    qblock: int
    pad_dim: int
    n_blocks: int


def _scale_min_fp16() -> float:
    """Minimum safe positive fp16 scale to avoid underflow."""
    return float(torch.finfo(torch.float16).tiny)


def _qblock_eff(kind: str, dim: int, qblock: int) -> int:
    """Compute effective quantization block size.

    For 4-bit formats, ensures block size is even (required for packing).
    """
    qb = min(qblock if qblock > 0 else 32, dim)
    if kind in ("q4_0", "nf4"):
        if dim < 2:
            raise ValueError(f"{kind} cache requires dim >= 2")
        max_even = dim if (dim % 2 == 0) else (dim - 1)
        qb = min(qb, max_even)
        if qb < 2:
            qb = 2
        if qb % 2 != 0:
            qb -= 1
    return max(1, qb)


def make_quantspec(kind: str, dim: int, qblock: int) -> QuantSpec:
    """Create a QuantSpec for the given kind, dimension, and block size."""
    qb = _qblock_eff(kind, int(dim), int(qblock))
    pad_dim = int(math.ceil(int(dim) / qb) * qb)
    if kind in ("q4_0", "nf4") and (pad_dim % 2 != 0):
        pad_dim += qb
    n_blocks = pad_dim // qb
    return QuantSpec(
        kind=kind, dim=int(dim), qblock=qb, pad_dim=pad_dim, n_blocks=n_blocks
    )


# Import after types are defined to avoid circular imports
from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.cache.layer import LayerKVCache


class Cache:
    """Factory for building lists of KV caches.

    Provides convenience methods to create one cache per layer, either
    standard or decoupled depending on the attention type.
    """

    @staticmethod
    def build(
        *,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        k_dim: int,
        v_dim: int,
        k_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ) -> list[LayerKVCache]:
        """Build standard KV caches for all layers."""
        return [
            LayerKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                k_dim=k_dim,
                v_dim=v_dim,
                k_cfg=k_cfg,
                v_cfg=v_cfg,
                device=device,
            )
            for _ in range(n_layers)
        ]

    @staticmethod
    def build_decoupled(
        *,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        k_sem_dim: int,
        k_geo_dim: int,
        v_dim: int,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ) -> list[DecoupledLayerKVCache]:
        """Build decoupled KV caches for all layers."""
        return [
            DecoupledLayerKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                k_sem_dim=k_sem_dim,
                k_geo_dim=k_geo_dim,
                v_dim=v_dim,
                k_sem_cfg=k_sem_cfg,
                k_geo_cfg=k_geo_cfg,
                v_cfg=v_cfg,
                device=device,
            )
            for _ in range(n_layers)
        ]
