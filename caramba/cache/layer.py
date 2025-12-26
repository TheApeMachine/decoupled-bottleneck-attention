"""Standard KV cache for one attention layer.

In standard or GQA attention, we cache K and V tensors of shape
(batch, seq, kv_heads × head_dim). This class manages a pair of
SeqCacheTensors and keeps them synchronized.
"""
from __future__ import annotations

import torch

from caramba.cache.tensor import SeqCacheTensor
from caramba.config.kvcache import KVCacheTensorConfig


class LayerKVCache:
    """Stores K and V caches for one standard attention layer.

    Provides append, get, and truncate operations that keep K and V
    synchronized. Supports both fp16/fp32 storage and quantized formats.
    """

    k: SeqCacheTensor
    v: SeqCacheTensor

    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        k_dim: int,
        v_dim: int,
        k_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ) -> None:
        """Allocate K and V storage."""
        self.k = SeqCacheTensor(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dim=int(k_dim),
            cfg=k_cfg,
            device=device,
        )
        self.v = SeqCacheTensor(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dim=int(v_dim),
            cfg=v_cfg,
            device=device,
        )

    @property
    def pos(self) -> int:
        """Current sequence position (tokens cached so far)."""
        return int(self.k.pos)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        """Append new K and V tensors.

        Returns the position before append (where new tokens start).
        """
        k_pos = self.k.append(k_new)
        v_pos = self.v.append(v_new)
        if int(k_pos) != int(v_pos):
            raise RuntimeError("K/V append position mismatch")
        return int(k_pos)

    def get(
        self, *, dtype: torch.dtype = torch.float16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve all cached K and V tensors."""
        return self.k.get(dtype=dtype), self.v.get(dtype=dtype)

    def truncate(self, new_pos: int) -> None:
        """Rollback both caches to a previous position."""
        self.k.truncate(new_pos)
        self.v.truncate(new_pos)
        if self.k.pos != self.v.pos:
            raise RuntimeError("K/V cache desync after truncate")
