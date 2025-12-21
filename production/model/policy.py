"""
policy manages KV cache format selection and quality gates.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, cast
import torch
import torch.nn as nn

from production.kvcache_backend import (
    KVCacheTensorConfig,
    DecoupledLayerKVCache,
    LayerKVCache
)
from production.runtime_tuning import (
    KVCachePolicy,
    KVCachePolicySelfOptimizer,
    policy_quality_reject_reasons
)

from .block import Block
from .config import ModelConfig
from .metrics import Metrics
from .cache import Cache

if TYPE_CHECKING:
    from production.runtime_tuning import KVSelfOptConfig

class Model(Protocol):
    """Minimal protocol for models Policy can manage."""
    cfg: ModelConfig
    blocks: nn.ModuleList
    def forward(
        self,
        idx: torch.Tensor,
        *,
        caches: list[DecoupledLayerKVCache | LayerKVCache] | None = None,
        pos_offset: int = 0,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache] | None]:
        """Model forward used by Policy for calibration/gating."""
        raise NotImplementedError

class Policy:
    """Selection engine for KV cache configurations."""
    def __init__(self, model: Model):
        self.model: Model = model
        self.cfg: ModelConfig = model.cfg

    def select(
        self,
        prompt: torch.Tensor,
        self_opt: KVSelfOptConfig,
        *,
        k_sem: KVCacheTensorConfig,
        k_geo: KVCacheTensorConfig,
        v: KVCacheTensorConfig,
        residual: int,
        decode_block: int,
        fused: str,
        max_new_tokens: int = 0,
    ) -> tuple[KVCacheTensorConfig, KVCacheTensorConfig, KVCacheTensorConfig, int | None, int]:
        """Choose the optimal policy, falling back to layerwise promotion if needed."""
        batch_size, prompt_len = prompt.shape
        max_seq = prompt_len + max_new_tokens

        base = KVCachePolicy(
            k_sem.kind, k_geo.kind, v.kind, k_sem.qblock, k_geo.qblock, v.qblock, residual
        )

        tuner = KVCachePolicySelfOptimizer(
            self_opt, device=prompt.device, attn=cast(Block, self.model.blocks[0]).attn,
            model_cfg=self.cfg, batch_size=batch_size, max_seq_len=max_seq,
            base_policy=base, base_decode_block=decode_block,
            base_fused=fused
        )

        if not getattr(self_opt, "policy_quality", False):
            chosen = tuner.choose_policy(prompt_len=prompt_len)
            ks, kg, vv = chosen.to_tensor_cfgs()
            return ks, kg, vv, None, chosen.residual_len

        # Gated Selection with Layerwise Fallback
        candidates = tuner.shortlist_policies(prompt_len=prompt_len, max_candidates=8)
        for cand in candidates:
            if self._gate(prompt, cand, self_opt):
                ks, kg, vv = cand.to_tensor_cfgs()
                return ks, kg, vv, None, cand.residual_len

            # If global failed, try layerwise promotion
            if getattr(self_opt, "layerwise_cache", False):
                for n in [1, 2, 4, 8, self.cfg.n_layer]:
                    if n > self.cfg.n_layer:
                        break
                    if self._gate(prompt, cand, self_opt, promote=n):
                        ks, kg, vv = cand.to_tensor_cfgs()
                        return ks, kg, vv, n, cand.residual_len

        ks, kg, vv = base.to_tensor_cfgs()
        return ks, kg, vv, None, base.residual_len

    def _gate(
        self,
        tokens: torch.Tensor,
        cand: KVCachePolicy,
        self_opt: KVSelfOptConfig,
        promote: int | None = None
    ) -> bool:
        """Judge a candidate policy over a calibration window."""
        pre = int(getattr(self_opt, "calib_prefill", 128))
        dec = int(getattr(self_opt, "calib_decode_steps", 32))
        ks, kg, vv = cand.to_tensor_cfgs()
        fp16 = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)

        def _build(cfg: KVCacheTensorConfig, l_idx: int) -> KVCacheTensorConfig:
            return fp16 if promote and l_idx < promote else cfg

        base_c = [
            Cache.build_layer(self.cfg, 1, pre+dec, tokens.device,
                              k_sem_cfg=fp16, k_geo_cfg=fp16, v_cfg=fp16)
            for _ in range(self.cfg.n_layer)
        ]
        test_c = [
            Cache.build_layer(self.cfg, 1, pre+dec, tokens.device,
                              k_sem_cfg=_build(ks, i), k_geo_cfg=_build(kg, i),
                              v_cfg=_build(vv, i))
            for i in range(self.cfg.n_layer)
        ]

        history: list[dict[str, float]] = []
        # Return of forward is tuple[Tensor, list[Any] | None]
        res_b = cast(
            tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache]],
            self.model.forward(tokens[:, :pre], caches=base_c)
        )
        lb = res_b[0]
        res_t = cast(
            tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache]],
            self.model.forward(tokens[:, :pre], caches=test_c)
        )
        lt = res_t[0]

        for i in range(pre, pre + dec):
            x = tokens[:, i:i+1]
            res_b = cast(
                tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache]],
                self.model.forward(x, caches=base_c, pos_offset=i)
            )
            lb, base_c = res_b[0], res_b[1]
            res_t = cast(
                tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache]],
                self.model.forward(x, caches=test_c, pos_offset=i)
            )
            lt, test_c = res_t[0], res_t[1]
            history.append(Metrics.compare(lb, lt, tokens[:, i+1]))

        agg = {
            "delta_nll": sum(h["delta_nll"] for h in history) / len(history),
            "max_abs_logit": max(h["max_abs_logit"] for h in history)
        }
        return not bool(policy_quality_reject_reasons(
            agg,
            max_abs_logit_tol=getattr(self_opt, "quality_tol", None),
            delta_nll_tol=getattr(self_opt, "quality_delta_nll_tol", None),
            ppl_ratio_tol=None,
            kl_tol=None
        ))
