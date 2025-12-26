"""LayerKVCache stores baseline K/V caches (non-decoupled)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from caramba.cache.tensor import SeqCacheTensor
from caramba.config.kvcache import KVCacheTensorConfig


class LayerKVCache:
    """LayerKVCache stores baseline K/V caches (non-decoupled)."""
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
        return int(self.k.pos)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        k_pos = self.k.append(k_new)
        v_pos = self.v.append(v_new)
        if int(k_pos) != int(v_pos):
            raise RuntimeError("K/V append position mismatch")
        return int(k_pos)

    def get(self, *, dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k.get(dtype=dtype), self.v.get(dtype=dtype)

    def truncate(self, new_pos: int) -> None:
        self.k.truncate(new_pos)
        self.v.truncate(new_pos)
        if self.k.pos != self.v.pos:
            raise RuntimeError("K/V cache desync after truncate")
