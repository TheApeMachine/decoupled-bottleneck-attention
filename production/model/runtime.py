"""
runtime manages the runtime environment for model execution.
"""
from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable
import torch

from production.kvcache_backend import (
    DecoupledLayerKVCache,
    KVCacheTensorConfig,
    KVCacheKind,
    LayerKVCache,
)
from production.runtime_tuning import KVDecodeSelfOptimizer

from .cache import Cache
from .policy import Policy, Model as PolicyModel
from .config import Mode

if TYPE_CHECKING:
    from production.runtime_tuning import KVSelfOptConfig

class KVRuntime:
    """Runtime environment for KV cache management."""
    def __init__(
        self,
        caches: Sequence[DecoupledLayerKVCache | LayerKVCache],
        layerwise_promote_layers: int | None,
        decode_tuner: KVDecodeSelfOptimizer | None,
        k_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_dec_cfg: KVCacheTensorConfig,
        kv_residual: int,
    ) -> None:
        self.caches: list[DecoupledLayerKVCache | LayerKVCache] = list(caches)
        self.layerwise_promote_layers: int | None = layerwise_promote_layers
        self.decode_tuner: KVDecodeSelfOptimizer | None = decode_tuner
        self.k_cfg: KVCacheTensorConfig = k_cfg
        self.v_cfg: KVCacheTensorConfig = v_cfg
        self.k_sem_cfg: KVCacheTensorConfig = k_sem_cfg
        self.k_geo_cfg: KVCacheTensorConfig = k_geo_cfg
        self.v_dec_cfg: KVCacheTensorConfig = v_dec_cfg
        self.kv_residual: int = kv_residual

    def __getitem__(self, index: int) -> DecoupledLayerKVCache | LayerKVCache:
        return self.caches[index]

    def __len__(self) -> int:
        return len(self.caches)

class Runtime:
    """Manages model runtime state and self-optimization."""
    def __init__(self, model: PolicyModel):
        self.model: PolicyModel = model
        self.policy: Policy = Policy(model)

    def build_kv(
        self,
        prompt: torch.Tensor,
        max_new: int,
        self_opt: KVSelfOptConfig | None = None,
        *,
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",
        kv_cache_k: KVCacheKind | None = None,
        kv_cache_v: KVCacheKind | None = None,
        kv_cache_k_sem: KVCacheKind | None = None,
        kv_cache_k_geo: KVCacheKind | None = None,
        log_callback: Callable[[dict[str, object]], None] | None = None,
        **_kwargs: object
    ) -> KVRuntime:
        """Build the complete KV runtime environment."""
        batch_size, prompt_len = prompt.shape
        max_seq = prompt_len + max_new

        # 1. Base Tensors (with hetero defaults)
        def _cfg(k: KVCacheKind | None = None, q: int | None = None) -> KVCacheTensorConfig:
            return KVCacheTensorConfig(
                kind=k or kv_cache,
                qblock=q or kv_qblock,
                residual_len=kv_residual
            )

        k_sem = _cfg(kv_cache_k_sem)
        geo_kind: KVCacheKind | None = kv_cache_k_geo or ("q8_0" if kv_cache == "q4_0" else None)
        k_geo = _cfg(geo_kind)
        v = _cfg(kv_cache_v)

        # 2. Selection
        if (self_opt and self_opt.mode != "none" and self_opt.scope in ("cache", "all")
                and self.model.cfg.attn_mode == Mode.DECOUPLED):
            k_sem, k_geo, v, promote, residual = self.policy.select(
                prompt, self_opt, k_sem=k_sem, k_geo=k_geo, v=v,
                residual=kv_residual,
                decode_block=kv_decode_block, fused=kv_fused, max_new_tokens=max_new
            )
        else:
            promote, residual = None, kv_residual

        # 3. Construction
        caches = Cache.build(
            self.model.cfg, batch_size, max_seq, prompt.device,
            k_sem=k_sem, k_geo=k_geo, v=v, k=_cfg(kv_cache_k)
        )
        for c in caches:
            # These are runtime tuning hints; set dynamically to avoid tight coupling to cache impls.
            setattr(c, "decode_block", kv_decode_block)
            setattr(c, "fused", kv_fused)

        # 4. Tuning
        tuner = None
        if self_opt and self_opt.mode != "none" and self_opt.scope in ("decode", "all"):
            tuner = KVDecodeSelfOptimizer(
                self_opt, device=prompt.device, base_fused=kv_fused,
                base_decode_block=kv_decode_block, log_callback=log_callback
            )

        k_cfg = _cfg(kv_cache_k)
        return KVRuntime(
            caches=caches, layerwise_promote_layers=promote, decode_tuner=tuner,
            k_cfg=k_cfg, v_cfg=v, k_sem_cfg=k_sem, k_geo_cfg=k_geo, v_dec_cfg=v,
            kv_residual=residual
        )
