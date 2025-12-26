"""Cache provides KV cache construction and management."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from caramba.config.kvcache import KVCacheTensorConfig


@dataclass(frozen=True)
class QuantSpec:
    """QuantSpec defines quantization storage geometry for a cache tensor."""
    kind: str
    dim: int
    qblock: int
    pad_dim: int
    n_blocks: int


def _scale_min_fp16() -> float:
    """Minimum safe positive fp16 scale to avoid underflow to 0."""
    return float(torch.finfo(torch.float16).tiny)


def _qblock_eff(kind: str, dim: int, qblock: int) -> int:
    """Compute effective qblock size."""
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
    """Compute pad_dim and n_blocks for a given dim and qblock."""
    qb = _qblock_eff(kind, int(dim), int(qblock))
    pad_dim = int(math.ceil(int(dim) / qb) * qb)
    if kind in ("q4_0", "nf4") and (pad_dim % 2 != 0):
        pad_dim += qb
    n_blocks = pad_dim // qb
    return QuantSpec(kind=kind, dim=int(dim), qblock=qb, pad_dim=pad_dim, n_blocks=n_blocks)


# Import after types are defined to avoid circular imports
from caramba.cache.layer import LayerKVCache
from caramba.cache.decoupled import DecoupledLayerKVCache


class Cache:
    """Factory for KV cache construction."""

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
        """Build a list of standard KV caches (one per layer)."""
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
        """Build a list of decoupled KV caches (one per layer)."""
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
