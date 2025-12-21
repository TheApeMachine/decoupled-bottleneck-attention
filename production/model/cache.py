"""
cache handles the construction of KV caches.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from production.kvcache_backend import DecoupledLayerKVCache, LayerKVCache, KVCacheTensorConfig
from .config import Mode

if TYPE_CHECKING:
    from .config import ModelConfig

class Cache:
    """Factory for KV cache construction."""
    @staticmethod
    def build(
        cfg: ModelConfig,
        batch_size: int,
        max_seq: int,
        device: torch.device,
        **tensor_cfgs: KVCacheTensorConfig
    ) -> list[DecoupledLayerKVCache | LayerKVCache]:
        """Build a list of KV caches (one per layer)."""
        return [
            Cache.build_layer(cfg, batch_size, max_seq, device, **tensor_cfgs)
            for _ in range(cfg.n_layer)
        ]

    @staticmethod
    def build_layer(
        cfg: ModelConfig,
        batch_size: int,
        max_seq: int,
        device: torch.device,
        **tensor_cfgs: KVCacheTensorConfig
    ) -> DecoupledLayerKVCache | LayerKVCache:
        """Build a single KV cache layer."""
        if cfg.attn_mode == Mode.DECOUPLED:
            return DecoupledLayerKVCache(
                batch_size=batch_size, max_seq_len=max_seq,
                k_sem_dim=cfg.sem_dim, k_geo_dim=cfg.geo_dim, v_dim=cfg.attn_dim,
                k_sem_cfg=tensor_cfgs.get("k_sem", KVCacheTensorConfig(kind="fp16", qblock=32)),
                k_geo_cfg=tensor_cfgs.get("k_geo", KVCacheTensorConfig(kind="fp16", qblock=32)),
                v_cfg=tensor_cfgs.get("v", KVCacheTensorConfig(kind="fp16", qblock=32)),
                device=device
            )

        # Standard or GQA
        kdim = (
            cfg.d_model if cfg.attn_mode == Mode.BASELINE
            else int((cfg.kv_head or cfg.n_head) * (cfg.attn_dim // cfg.n_head))
        )
        kv_cfg = tensor_cfgs.get("v", KVCacheTensorConfig(kind="fp16", qblock=32))
        return LayerKVCache(
            batch_size=batch_size, max_seq_len=max_seq,
            k_dim=kdim, v_dim=kdim,
            k_cfg=tensor_cfgs.get("k", kv_cfg), v_cfg=kv_cfg,
            device=device
        )
