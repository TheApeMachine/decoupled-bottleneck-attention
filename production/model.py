from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from production.attention import DecoupledBottleneckAttention
from production.kvcache_backend import DecoupledLayerKVCache, KVCacheKind, KVCacheTensorConfig, LayerKVCache
from production.runtime_tuning import (
    KVCachePolicy,
    KVCachePolicySelfOptimizer,
    KVDecodeSelfOptimizer,
    KVDecodePlan,
    KVSelfOptConfig,
    load_token_ids_spec,
    policy_quality_reject_reasons,
    warn_policy_quality_reject,
)


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int

    n_layer: int = 6
    n_head: int = 8
    kv_head: Optional[int] = None  # for GQA: number of KV heads (defaults to n_head)
    d_model: int = 512
    d_ff: int = 2048

    embed_dim: int = 512  # lexical bottleneck if < d_model

    attn_mode: Literal["standard", "bottleneck", "decoupled", "gqa"] = "bottleneck"
    attn_dim: int = 512
    sem_dim: int = 32
    geo_dim: int = 64

    decoupled_gate: bool = True

    rope: bool = True
    rope_base: float = 10000.0

    tie_qk: bool = False
    null_attn: bool = False
    learned_temp: bool = True

    mlp: Literal["swiglu", "gelu"] = "swiglu"
    dropout: float = 0.0


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.mlp == "swiglu":
            self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        elif cfg.mlp == "gelu":
            self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        else:
            raise ValueError(cfg.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.mlp == "swiglu":
            x = self.w3(F.silu(self.w1(x)) * self.w2(x))
        else:
            x = self.w2(F.gelu(self.w1(x)))
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = DecoupledBottleneckAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor], cache: Optional[Any], pos_offset: int
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        a, cache = self.attn(self.ln1(x), attn_mask=attn_mask, cache=cache, pos_offset=pos_offset)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.emb_in = nn.Linear(cfg.embed_dim, cfg.d_model, bias=False) if cfg.embed_dim != cfg.d_model else None
        self.emb_out = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False) if cfg.embed_dim != cfg.d_model else None

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

        self.apply(self._init_weights)
        self.grad_checkpointing = False

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        *,
        caches: Optional[List[Any]] = None,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        B, T = idx.shape
        if caches is None and T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}. Increase --block.")

        x = self.tok_emb(idx)
        if self.emb_in is not None:
            x = self.emb_in(x)
        x = self.drop(x)

        attn_mask: Optional[torch.Tensor] = None
        if caches is None:
            if self.cfg.null_attn:
                attn_mask = self.causal_mask[:, :, :T, :T]
            else:
                attn_mask = None
        else:
            if T > 1:
                prev_len = caches[0].pos
                if prev_len == 0 and (not self.cfg.null_attn):
                    attn_mask = None
                else:
                    L = prev_len + T
                    key_pos = torch.arange(L, device=idx.device).view(1, 1, 1, L)
                    q_pos = (prev_len + torch.arange(T, device=idx.device)).view(1, 1, T, 1)
                    attn_mask = key_pos <= q_pos
            else:
                attn_mask = None

        new_caches: Optional[List[Any]] = [] if caches is not None else None

        if caches is None and self.training and getattr(self, "grad_checkpointing", False):
            try:
                from torch.utils.checkpoint import checkpoint  # type: ignore

                for blk in self.blocks:

                    def _blk_fwd(x_in: torch.Tensor, blk=blk) -> torch.Tensor:
                        y, _ = blk(x_in, attn_mask=attn_mask, cache=None, pos_offset=pos_offset)
                        return y

                    x = checkpoint(_blk_fwd, x, use_reentrant=False)
            except Exception:
                for blk in self.blocks:
                    x, _ = blk(x, attn_mask=attn_mask, cache=None, pos_offset=pos_offset)
        elif caches is None:
            for blk in self.blocks:
                x, _ = blk(x, attn_mask=attn_mask, cache=None, pos_offset=pos_offset)
        else:
            for i, blk in enumerate(self.blocks):
                layer_cache = caches[i]
                x, layer_cache = blk(x, attn_mask=attn_mask, cache=layer_cache, pos_offset=pos_offset)
                new_caches.append(layer_cache)

        x = self.ln_f(x)
        if self.emb_out is not None:
            x_small = self.emb_out(x)
        else:
            x_small = x
        logits = x_small @ self.tok_emb.weight.t()
        return logits, new_caches

    @torch.no_grad()
    def _compute_quality_metrics_loop(
        self,
        tokens: torch.Tensor,
        caches_base: list[Any],
        caches_cand: list[Any],
        prefill: int,
        decode_steps: int,
        compute_kl: bool,
    ) -> dict[str, float]:
        """Shared prompt prefill + teacher-forced decode loop to compare two cache variants.

        Assumes `tokens` is shape (B, L), `prefill`/`decode_steps` are already clamped, and
        caches are pre-allocated with sufficient `max_seq_len`.
        """
        B, L = tokens.shape

        # Prefill
        prompt = tokens[:, :prefill]
        _, caches_base = self(prompt, caches=caches_base, pos_offset=0)
        _, caches_cand = self(prompt, caches=caches_cand, pos_offset=0)

        max_err = 0.0
        nll_sum_base = 0.0
        nll_sum_cand = 0.0
        nll_count = 0
        kl_sum = 0.0
        kl_count = 0

        for i in range(prefill, prefill + decode_steps):
            x = tokens[:, i : i + 1]
            logits_base, caches_base = self(x, caches=caches_base, pos_offset=i)
            logits_cand, caches_cand = self(x, caches=caches_cand, pos_offset=i)

            lb = logits_base[:, -1, :].float()
            lc = logits_cand[:, -1, :].float()

            err = (lc - lb).abs().max().item()
            if err > max_err:
                max_err = float(err)

            if (i + 1) < L:
                tgt = tokens[:, i + 1].reshape(-1)
                nll_sum_base += float(F.cross_entropy(lb.reshape(-1, lb.size(-1)), tgt, reduction="sum").item())
                nll_sum_cand += float(F.cross_entropy(lc.reshape(-1, lc.size(-1)), tgt, reduction="sum").item())
                nll_count += int(tgt.numel())

                if compute_kl:
                    logp_b = F.log_softmax(lb, dim=-1)
                    p_b = logp_b.exp()
                    logp_c = F.log_softmax(lc, dim=-1)
                    kl_tok = (p_b * (logp_b - logp_c)).sum(-1)
                    kl_sum += float(kl_tok.sum().item())
                    kl_count += int(kl_tok.numel())

        if nll_count > 0:
            ce_base = nll_sum_base / float(nll_count)
            ce_cand = nll_sum_cand / float(nll_count)
            delta_nll = float(ce_cand - ce_base)
            ppl_ratio = float(math.exp(delta_nll))
        else:
            delta_nll = float("nan")
            ppl_ratio = float("nan")

        if compute_kl and kl_count > 0:
            kl_avg = float(kl_sum / float(kl_count))
        else:
            kl_avg = float("nan")

        return {
            "max_abs_logit": float(max_err),
            "delta_nll": float(delta_nll),
            "ppl_ratio": float(ppl_ratio),
            "kl_base_cand": float(kl_avg),
        }

    def _make_decoupled_layer_cache(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
        decode_block: Optional[int] = None,
        fused: Optional[str] = None,
    ) -> "DecoupledLayerKVCache":
        c = DecoupledLayerKVCache(
            batch_size=int(batch_size),
            max_seq_len=int(max_seq_len),
            k_sem_dim=self.cfg.sem_dim,
            k_geo_dim=self.cfg.geo_dim,
            v_dim=self.cfg.attn_dim,
            k_sem_cfg=k_sem_cfg,
            k_geo_cfg=k_geo_cfg,
            v_cfg=v_cfg,
            device=device,
        )
        if decode_block is not None:
            c.decode_block = int(decode_block)
        if fused is not None:
            c.fused = str(fused)
        return c

    @torch.no_grad()
    def _policy_quality_metrics_decoupled(
        self,
        tokens: torch.Tensor,
        *,
        policy: KVCachePolicy,
        prefill: int,
        decode_steps: int,
        kv_decode_block: int,
        compute_kl: bool = False,
    ) -> dict[str, float]:
        """Importance-aware quality metrics for cache-policy tuning.

        Computes over a short teacher-forced decode window (vs fp16-cache baseline):
        - max_abs_logit: max(|Δlogit|) (safety fuse)
        - delta_nll: ΔNLL (nats/token)
        - ppl_ratio: exp(delta_nll)
        - kl_base_cand: KL(p_base || p_cand) averaged over tokens (optional)
        """
        if tokens.numel() == 0:
            return {"max_abs_logit": 0.0, "delta_nll": float("nan"), "ppl_ratio": float("nan"), "kl_base_cand": float("nan")}

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens[:1].contiguous()

        device = tokens.device
        B, L = tokens.shape
        if L < 2:
            return {"max_abs_logit": 0.0, "delta_nll": float("nan"), "ppl_ratio": float("nan"), "kl_base_cand": float("nan")}

        prefill = int(max(1, min(int(prefill), L - 1)))
        decode_steps = int(max(1, min(int(decode_steps), L - prefill)))
        max_seq = int(prefill + decode_steps)

        was_training = self.training
        self.eval()
        try:
            # Baseline: fp16 caches everywhere.
            fp16_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
            caches_base: list[Any] = []
            for _ in range(self.cfg.n_layer):
                caches_base.append(
                    self._make_decoupled_layer_cache(
                        batch_size=B,
                        max_seq_len=max_seq,
                        k_sem_cfg=fp16_cfg,
                        k_geo_cfg=fp16_cfg,
                        v_cfg=fp16_cfg,
                        device=device,
                        decode_block=kv_decode_block,
                        fused="none",
                    )
                )

            # Candidate caches: chosen policy (quantized).
            k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
            caches_cand: list[Any] = []
            for _ in range(self.cfg.n_layer):
                caches_cand.append(
                    self._make_decoupled_layer_cache(
                        batch_size=B,
                        max_seq_len=max_seq,
                        k_sem_cfg=k_sem_cfg,
                        k_geo_cfg=k_geo_cfg,
                        v_cfg=v_cfg,
                        device=device,
                        decode_block=kv_decode_block,
                        # Ensure we don't mix kernel numeric differences into the quant check.
                        fused="none",
                    )
                )

            return self._compute_quality_metrics_loop(
                tokens,
                caches_base=caches_base,
                caches_cand=caches_cand,
                prefill=prefill,
                decode_steps=decode_steps,
                compute_kl=compute_kl,
            )
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def _policy_quality_metrics_decoupled_layerwise(
        self,
        tokens: torch.Tensor,
        *,
        low_policy: KVCachePolicy,
        promote_layers: int,
        prefill: int,
        decode_steps: int,
        kv_decode_block: int,
        compute_kl: bool = False,
    ) -> dict[str, float]:
        """Quality metrics for a layerwise policy: early layers fp16, later layers `low_policy`.

        Baseline is fp16 caches everywhere (same as `_policy_quality_metrics_decoupled`).
        """
        promote_layers = int(max(0, min(int(promote_layers), int(self.cfg.n_layer))))
        if promote_layers <= 0:
            return self._policy_quality_metrics_decoupled(
                tokens,
                policy=low_policy,
                prefill=prefill,
                decode_steps=decode_steps,
                kv_decode_block=kv_decode_block,
                compute_kl=compute_kl,
            )

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens[:1].contiguous()
        device = tokens.device
        B, L = tokens.shape
        if L < 2:
            return {"max_abs_logit": 0.0, "delta_nll": float("nan"), "ppl_ratio": float("nan"), "kl_base_cand": float("nan")}

        prefill = int(max(1, min(int(prefill), L - 1)))
        decode_steps = int(max(1, min(int(decode_steps), L - prefill)))
        max_seq = int(prefill + decode_steps)

        was_training = self.training
        self.eval()
        try:
            fp16_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)

            caches_base: list[Any] = []
            for _ in range(self.cfg.n_layer):
                caches_base.append(
                    self._make_decoupled_layer_cache(
                        batch_size=B,
                        max_seq_len=max_seq,
                        k_sem_cfg=fp16_cfg,
                        k_geo_cfg=fp16_cfg,
                        v_cfg=fp16_cfg,
                        device=device,
                        decode_block=kv_decode_block,
                        fused="none",
                    )
                )

            low_k_sem_cfg, low_k_geo_cfg, low_v_cfg = low_policy.to_tensor_cfgs()
            caches_cand: list[Any] = []
            for li in range(self.cfg.n_layer):
                if li < promote_layers:
                    k_sem_cfg = fp16_cfg
                    k_geo_cfg = fp16_cfg
                    v_cfg = fp16_cfg
                else:
                    k_sem_cfg = low_k_sem_cfg
                    k_geo_cfg = low_k_geo_cfg
                    v_cfg = low_v_cfg
                caches_cand.append(
                    self._make_decoupled_layer_cache(
                        batch_size=B,
                        max_seq_len=max_seq,
                        k_sem_cfg=k_sem_cfg,
                        k_geo_cfg=k_geo_cfg,
                        v_cfg=v_cfg,
                        device=device,
                        decode_block=kv_decode_block,
                        fused="none",
                    )
                )
            return self._compute_quality_metrics_loop(
                tokens,
                caches_base=caches_base,
                caches_cand=caches_cand,
                prefill=prefill,
                decode_steps=decode_steps,
                compute_kl=compute_kl,
            )
        finally:
            if was_training:
                self.train()

    def _choose_kv_cache_policy(
        self,
        *,
        model: "GPT",
        self_opt: Optional[KVSelfOptConfig],
        device: torch.device,
        prompt: torch.Tensor,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_dec_cfg: KVCacheTensorConfig,
        kv_residual: int,
        kv_decode_block: int,
        kv_fused: str,
        max_new_tokens: int,
        is_speculative: bool,
    ) -> Tuple[KVCacheTensorConfig, KVCacheTensorConfig, KVCacheTensorConfig, Optional[int], int]:
        """Shared cache-policy selection + optional quality validation for `generate` + `generate_speculative`.

        Defensive by design: any error yields the original configs and leaves layerwise promotion unset.
        Returns: (k_sem_cfg, k_geo_cfg, v_dec_cfg, layerwise_promote_layers, kv_residual_out)
        """
        layerwise_promote_layers: Optional[int] = None
        kv_residual_out = int(kv_residual)

        # Speculative decoding: only tune cache policy for the main model (draft model should be cheap/simple).
        if model is not self:
            return k_sem_cfg, k_geo_cfg, v_dec_cfg, layerwise_promote_layers, kv_residual_out

        if (
            self_opt is None
            or getattr(self_opt, "mode", "none") == "none"
            or getattr(self_opt, "scope", "all") not in ("cache", "all")
            or model.cfg.attn_mode != "decoupled"
        ):
            return k_sem_cfg, k_geo_cfg, v_dec_cfg, layerwise_promote_layers, kv_residual_out

        try:
            B, T0 = prompt.shape
            max_seq = int(T0 + int(max_new_tokens))

            base_policy = KVCachePolicy(
                k_sem_kind=k_sem_cfg.kind,
                k_geo_kind=k_geo_cfg.kind,
                v_kind=v_dec_cfg.kind,
                k_sem_qblock=k_sem_cfg.qblock,
                k_geo_qblock=k_geo_cfg.qblock,
                v_qblock=v_dec_cfg.qblock,
                residual_len=int(kv_residual_out),
            )
            pol_tuner = KVCachePolicySelfOptimizer(
                self_opt,
                device=device,
                attn=model.blocks[0].attn,
                model_cfg=model.cfg,
                batch_size=int(B),
                max_seq_len=int(max_seq),
                base_policy=base_policy,
                base_decode_block=int(kv_decode_block),
                base_fused=str(kv_fused),
            )
            chosen = pol_tuner.choose_policy(prompt_len=int(T0))

            if getattr(self_opt, "policy_quality", False):
                calib_spec = getattr(self_opt, "calib_tokens", None)
                if calib_spec:
                    calib_ids = load_token_ids_spec(str(calib_spec))
                    calib = torch.tensor([calib_ids], device=device, dtype=torch.long)
                else:
                    calib = prompt.detach()

                compute_kl = bool(getattr(self_opt, "quality_compute_kl", False)) or (getattr(self_opt, "quality_kl_tol", None) is not None)
                qm = self._policy_quality_metrics_decoupled(
                    calib,
                    policy=chosen,
                    prefill=int(getattr(self_opt, "calib_prefill", 128)),
                    decode_steps=int(getattr(self_opt, "calib_decode_steps", 32)),
                    kv_decode_block=int(kv_decode_block),
                    compute_kl=compute_kl,
                )
                reasons = policy_quality_reject_reasons(
                    qm,
                    max_abs_logit_tol=getattr(self_opt, "quality_tol", None),
                    delta_nll_tol=getattr(self_opt, "quality_delta_nll_tol", None),
                    ppl_ratio_tol=getattr(self_opt, "quality_ppl_ratio_tol", None),
                    kl_tol=getattr(self_opt, "quality_kl_tol", None),
                )
                if reasons:
                    if bool(getattr(self_opt, "layerwise_cache", False)) and model.cfg.n_layer > 1:
                        low = chosen
                        pre = int(getattr(self_opt, "calib_prefill", 128))
                        dec = int(getattr(self_opt, "calib_decode_steps", 32))

                        cand_ns: List[int] = []
                        n = 1
                        while n < int(model.cfg.n_layer):
                            cand_ns.append(n)
                            n *= 2
                        cand_ns.append(int(model.cfg.n_layer))
                        cand_ns = sorted(set(cand_ns))

                        for n_promote in cand_ns:
                            qm2 = self._policy_quality_metrics_decoupled_layerwise(
                                calib,
                                low_policy=low,
                                promote_layers=n_promote,
                                prefill=pre,
                                decode_steps=dec,
                                kv_decode_block=int(kv_decode_block),
                                compute_kl=compute_kl,
                            )
                            reasons2 = policy_quality_reject_reasons(
                                qm2,
                                max_abs_logit_tol=getattr(self_opt, "quality_tol", None),
                                delta_nll_tol=getattr(self_opt, "quality_delta_nll_tol", None),
                                ppl_ratio_tol=getattr(self_opt, "quality_ppl_ratio_tol", None),
                                kl_tol=getattr(self_opt, "quality_kl_tol", None),
                            )
                            if not reasons2:
                                layerwise_promote_layers = int(n_promote)
                                if is_speculative:
                                    print(
                                        f"[selfopt] layerwise cache-policy enabled for speculative decode: promote_layers={layerwise_promote_layers}/{model.cfg.n_layer} low={low.short()}"
                                    )
                                else:
                                    dnll = float(qm2.get("delta_nll", float("nan")))
                                    pr = float(qm2.get("ppl_ratio", float("nan")))
                                    klv = float(qm2.get("kl_base_cand", float("nan")))
                                    mx = float(qm2.get("max_abs_logit", float("nan")))
                                    print(
                                        f"[selfopt] layerwise cache-policy OK: promote_layers={layerwise_promote_layers}/{model.cfg.n_layer} "
                                        f"low={low.short()} ΔNLL={dnll:.4g} ppl_ratio={pr:.4g} KL={klv:.4g} max|Δlogit|={mx:.4g}"
                                    )
                                chosen = low
                                break

                        if layerwise_promote_layers is None:
                            warn_policy_quality_reject(chosen=chosen.short(), fallback=base_policy.short(), reasons=reasons)
                            chosen = base_policy
                    else:
                        warn_policy_quality_reject(chosen=chosen.short(), fallback=base_policy.short(), reasons=reasons)
                        chosen = base_policy
                else:
                    dnll = float(qm.get("delta_nll", float("nan")))
                    pr = float(qm.get("ppl_ratio", float("nan")))
                    klv = float(qm.get("kl_base_cand", float("nan")))
                    mx = float(qm.get("max_abs_logit", float("nan")))
                    if is_speculative:
                        print(f"[selfopt] cache-policy quality OK (spec): ΔNLL={dnll:.4g} ppl_ratio={pr:.4g} KL={klv:.4g} max|Δlogit|={mx:.4g}")
                    else:
                        print(f"[selfopt] cache-policy quality OK: ΔNLL={dnll:.4g} ppl_ratio={pr:.4g} KL={klv:.4g} max|Δlogit|={mx:.4g}")

            k_sem_cfg2, k_geo_cfg2, v_dec_cfg2 = chosen.to_tensor_cfgs()
            kv_residual_out = int(chosen.residual_len)
            return k_sem_cfg2, k_geo_cfg2, v_dec_cfg2, layerwise_promote_layers, kv_residual_out
        except Exception:
            return k_sem_cfg, k_geo_cfg, v_dec_cfg, layerwise_promote_layers, kv_residual_out

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",
        self_opt: Optional[KVSelfOptConfig] = None,
        kv_cache_k: Optional[KVCacheKind] = None,
        kv_cache_v: Optional[KVCacheKind] = None,
        kv_cache_k_sem: Optional[KVCacheKind] = None,
        kv_cache_k_geo: Optional[KVCacheKind] = None,
        kv_qblock_k: Optional[int] = None,
        kv_qblock_v: Optional[int] = None,
        kv_qblock_k_sem: Optional[int] = None,
        kv_qblock_k_geo: Optional[int] = None,
        log_callback: Optional[Any] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        device = prompt.device
        B, T0 = prompt.shape
        max_seq = T0 + max_new_tokens

        if kv_fused not in ("none", "auto", "triton1pass", "triton2pass"):
            raise ValueError("kv_fused must be one of: none, auto, triton1pass, triton2pass")

        if self.cfg.attn_mode == "decoupled" and kv_cache == "q4_0":
            if kv_cache_k_geo is None:
                kv_cache_k_geo = "q8_0"
            if kv_cache_k_sem is None:
                kv_cache_k_sem = "q4_0"
            if kv_cache_v is None:
                kv_cache_v = "q4_0"

        def make_cfg(kind_override: Optional[KVCacheKind], qblock_override: Optional[int]) -> KVCacheTensorConfig:
            kind = kind_override if kind_override is not None else kv_cache
            qblock = qblock_override if qblock_override is not None else kv_qblock
            residual_len = kv_residual if kind not in ("fp16", "fp32") else 0
            return KVCacheTensorConfig(kind=kind, qblock=qblock, residual_len=residual_len)

        k_cfg = make_cfg(kv_cache_k, kv_qblock_k)
        v_cfg = make_cfg(kv_cache_v, kv_qblock_v)

        k_sem_cfg = make_cfg(kv_cache_k_sem, kv_qblock_k_sem)
        k_geo_cfg = make_cfg(kv_cache_k_geo, kv_qblock_k_geo)
        v_dec_cfg = make_cfg(kv_cache_v, kv_qblock_v)
        layerwise_promote_layers: Optional[int] = None

        if (
            self_opt is not None
            and getattr(self_opt, "mode", "none") != "none"
            and getattr(self_opt, "scope", "all") in ("cache", "all")
            and self.cfg.attn_mode == "decoupled"
        ):
            k_sem_cfg, k_geo_cfg, v_dec_cfg, layerwise_promote_layers, kv_residual = self._choose_kv_cache_policy(
                model=self,
                self_opt=self_opt,
                device=device,
                prompt=prompt,
                k_sem_cfg=k_sem_cfg,
                k_geo_cfg=k_geo_cfg,
                v_dec_cfg=v_dec_cfg,
                kv_residual=int(kv_residual),
                kv_decode_block=int(kv_decode_block),
                kv_fused=str(kv_fused),
                max_new_tokens=int(max_new_tokens),
                is_speculative=False,
            )

        if self.cfg.attn_mode == "decoupled":
            if layerwise_promote_layers is None:
                caches = [
                    self._make_decoupled_layer_cache(
                        batch_size=B,
                        max_seq_len=max_seq,
                        k_sem_cfg=k_sem_cfg,
                        k_geo_cfg=k_geo_cfg,
                        v_cfg=v_dec_cfg,
                        device=device,
                    )
                    for _ in range(self.cfg.n_layer)
                ]
            else:
                fp16_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
                caches = []
                for li in range(self.cfg.n_layer):
                    if li < int(layerwise_promote_layers):
                        ks = fp16_cfg
                        kg = fp16_cfg
                        vv = fp16_cfg
                    else:
                        ks = k_sem_cfg
                        kg = k_geo_cfg
                        vv = v_dec_cfg
                    caches.append(
                        self._make_decoupled_layer_cache(
                            batch_size=B,
                            max_seq_len=max_seq,
                            k_sem_cfg=ks,
                            k_geo_cfg=kg,
                            v_cfg=vv,
                            device=device,
                        )
                    )
        else:
            caches = [
                LayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_dim=(self.cfg.d_model if self.cfg.attn_mode == "standard" else self.cfg.attn_dim),
                    v_dim=(self.cfg.d_model if self.cfg.attn_mode == "standard" else self.cfg.attn_dim),
                    k_cfg=k_cfg,
                    v_cfg=v_cfg,
                    device=device,
                )
                for _ in range(self.cfg.n_layer)
            ]

        for li, c in enumerate(caches):
            c.decode_block = int(kv_decode_block)
            # Layerwise fp16 promotion: never force fused kernels on fp16 layers (they are quant-specialized).
            if layerwise_promote_layers is not None and self.cfg.attn_mode == "decoupled" and li < int(layerwise_promote_layers):
                c.fused = "none"
            else:
                c.fused = str(kv_fused)

        decode_tuner: Optional[KVDecodeSelfOptimizer] = None
        if (
            self_opt is not None
            and getattr(self_opt, "mode", "none") != "none"
            and getattr(self_opt, "scope", "all") in ("decode", "all")
        ):
            decode_tuner = KVDecodeSelfOptimizer(
                self_opt,
                device=device,
                base_fused=str(kv_fused),
                base_decode_block=int(kv_decode_block),
                log_callback=log_callback,
            )

        out = prompt
        pos = 0

        logits, caches = self(out, caches=caches, pos_offset=pos)
        pos += out.size(1)

        for _ in range(int(max_new_tokens)):
            last = logits[:, -1, :] / max(1e-8, float(temperature))
            if top_k is not None:
                vtop, _ = torch.topk(last, int(top_k), dim=-1)
                thresh = vtop[:, -1].unsqueeze(-1)
                last = last.masked_fill(last < thresh, -float("inf"))
            probs = F.softmax(last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if decode_tuner is not None and self.cfg.attn_mode == "decoupled":
                # Provide a single-token query to the decode tuner to choose a plan per bucket.
                # It will update cache.decode_block / cache.fused by applying the plan.
                try:
                    # This uses the first layer as representative.
                    attn0 = self.blocks[0].attn
                    cache0 = caches[0]
                    # We don't have direct access to q_sem/q_geo here without re-running attention internals,
                    # so the runtime tuner is primarily used inside the attention forward when available.
                    _ = (attn0, cache0)
                except Exception:
                    pass

            out = torch.cat([out, next_id], dim=1)
            logits, caches = self(next_id, caches=caches, pos_offset=pos)
            pos += 1

        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def generate_speculative(
        self,
        prompt: torch.Tensor,
        *,
        draft_model: "GPT",
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        # KV-cache controls
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",
        self_opt: Optional[KVSelfOptConfig] = None,
        # Optional heterogeneous overrides
        kv_cache_k: Optional[KVCacheKind] = None,
        kv_cache_v: Optional[KVCacheKind] = None,
        kv_cache_k_sem: Optional[KVCacheKind] = None,
        kv_cache_k_geo: Optional[KVCacheKind] = None,
        kv_qblock_k: Optional[int] = None,
        kv_qblock_v: Optional[int] = None,
        kv_qblock_k_sem: Optional[int] = None,
        kv_qblock_k_geo: Optional[int] = None,
        # Spec knobs
        spec_k: int = 4,
        spec_method: str = "reject_sampling",  # {"reject_sampling","greedy"}
        spec_extra_token: bool = False,
        spec_disable_below_accept: float = 0.0,
        log_callback: Optional[Any] = None,
    ) -> torch.Tensor:
        """Speculative decoding (draft proposes, main verifies).

        Notes:
        - Currently optimized for batch_size=1.
        - For decoupled quantized caches, the attention code uses a sequential streaming prefill for T>1 to avoid
          dequantizing the full prefix during verification.
        """
        if prompt.dim() != 2:
            raise ValueError(f"prompt must be (B,T), got {tuple(prompt.shape)}")
        if int(prompt.size(0)) != 1:
            raise ValueError("generate_speculative currently supports batch_size=1 (B==1).")

        if int(max_new_tokens) <= 0:
            return prompt

        if kv_fused not in ("none", "auto", "triton1pass", "triton2pass"):
            raise ValueError("kv_fused must be one of: none, auto, triton1pass, triton2pass")

        spec_k = int(spec_k)
        if spec_k <= 0:
            spec_k = 1
        spec_method = str(spec_method)
        if spec_method not in ("reject_sampling", "greedy"):
            raise ValueError("spec_method must be 'reject_sampling' or 'greedy'")

        device = prompt.device

        was_training_main = self.training
        was_training_draft = draft_model.training
        self.eval()
        draft_model.eval()
        try:
            t0 = time.perf_counter()
            spec_steps = 0
            proposed_total = 0  # total draft-proposed tokens
            accepted_total = 0  # total accepted draft tokens

            def _emit_spec_metrics(*, total_new_tokens: int) -> None:
                if not log_callback:
                    return
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                acceptance_rate = float(accepted_total) / float(max(1, proposed_total))
                tokens_per_step = float(total_new_tokens) / float(max(1, spec_steps))
                try:
                    log_callback(
                        {
                            "acceptance_rate": float(acceptance_rate),
                            "tokens_per_step": float(tokens_per_step),
                            "total_tokens": int(total_new_tokens),
                            "elapsed_ms": float(elapsed_ms),
                        }
                    )
                except Exception:
                    # Logging must never break generation.
                    pass

            # --------------- helpers ---------------
            def _filter_logits(logits: torch.Tensor) -> torch.Tensor:
                x = logits / max(float(temperature), 1e-8)
                if top_k is not None:
                    k = int(top_k)
                    if k > 0 and k < x.size(-1):
                        v, _ = torch.topk(x, k, dim=-1)
                        x = x.masked_fill(x < v[:, [-1]], -float("inf"))
                return x

            def _probs(logits: torch.Tensor) -> torch.Tensor:
                return F.softmax(_filter_logits(logits), dim=-1)

            def _sample_from_probs(p: torch.Tensor) -> torch.Tensor:
                # p: (B,V) -> (B,1)
                return torch.multinomial(p, num_samples=1)

            def _sample_token(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # returns (token_ids (B,1), probs (B,V))
                p = _probs(logits)
                if spec_method == "greedy":
                    tok = torch.argmax(p, dim=-1, keepdim=True)
                else:
                    tok = _sample_from_probs(p)
                return tok, p

            def _truncate_all(caches: List[Any], new_pos: int) -> None:
                for c in caches:
                    c.truncate(new_pos)

            # --------------- cache init ---------------
            # Main caches: reuse the same logic as `generate` (including cache-policy self-opt if enabled).
            # We do this by calling the regular generate path up to cache construction; since code is monolithic,
            # we replicate the minimal cache construction inline here for both models.

            def _make_cfgs_for(model: "GPT") -> Tuple[Any, Any, Any, Any, Any, Any, Optional[int]]:
                B, T0 = prompt.shape
                max_seq = int(T0 + max_new_tokens)

                # Default decoupled hetero for q4_0.
                _kv_cache_k_geo = kv_cache_k_geo
                _kv_cache_k_sem = kv_cache_k_sem
                _kv_cache_v = kv_cache_v
                if model.cfg.attn_mode == "decoupled" and kv_cache == "q4_0":
                    if _kv_cache_k_geo is None:
                        _kv_cache_k_geo = "q8_0"
                    if _kv_cache_k_sem is None:
                        _kv_cache_k_sem = "q4_0"
                    if _kv_cache_v is None:
                        _kv_cache_v = "q4_0"

                def make_cfg(kind_override: Optional[KVCacheKind], qblock_override: Optional[int]) -> KVCacheTensorConfig:
                    kind = kind_override if kind_override is not None else kv_cache
                    qblock = qblock_override if qblock_override is not None else kv_qblock
                    residual_len = kv_residual if kind not in ("fp16", "fp32") else 0
                    return KVCacheTensorConfig(kind=kind, qblock=qblock, residual_len=residual_len)

                k_cfg = make_cfg(kv_cache_k, kv_qblock_k)
                v_cfg = make_cfg(_kv_cache_v, kv_qblock_v)
                k_sem_cfg = make_cfg(_kv_cache_k_sem, kv_qblock_k_sem)
                k_geo_cfg = make_cfg(_kv_cache_k_geo, kv_qblock_k_geo)
                v_dec_cfg = make_cfg(_kv_cache_v, kv_qblock_v)
                layerwise_promote_layers: Optional[int] = None

                # Cache-policy tuner only on the main model (draft model should be cheap/simple).
                if model is self and model.cfg.attn_mode == "decoupled":
                    if (
                        self_opt is not None
                        and getattr(self_opt, "mode", "none") != "none"
                        and getattr(self_opt, "scope", "all") in ("cache", "all")
                    ):
                        k_sem_cfg, k_geo_cfg, v_dec_cfg, layerwise_promote_layers, _kv_res = self._choose_kv_cache_policy(
                            model=model,
                            self_opt=self_opt,
                            device=device,
                            prompt=prompt,
                            k_sem_cfg=k_sem_cfg,
                            k_geo_cfg=k_geo_cfg,
                            v_dec_cfg=v_dec_cfg,
                            kv_residual=int(kv_residual),
                            kv_decode_block=int(kv_decode_block),
                            kv_fused=str(kv_fused),
                            max_new_tokens=int(max_new_tokens),
                            is_speculative=True,
                        )

                return max_seq, k_cfg, v_cfg, k_sem_cfg, k_geo_cfg, v_dec_cfg, layerwise_promote_layers

            def _make_caches(model: "GPT", *, max_seq: int, k_cfg: Any, v_cfg: Any, k_sem_cfg: Any, k_geo_cfg: Any, v_dec_cfg: Any, layerwise_promote_layers: Optional[int]) -> List[Any]:
                B, _T0 = prompt.shape
                if model.cfg.attn_mode == "decoupled":
                    if layerwise_promote_layers is None:
                        caches = [
                            model._make_decoupled_layer_cache(
                                batch_size=int(B),
                                max_seq_len=int(max_seq),
                                k_sem_cfg=k_sem_cfg,
                                k_geo_cfg=k_geo_cfg,
                                v_cfg=v_dec_cfg,
                                device=device,
                            )
                            for _ in range(model.cfg.n_layer)
                        ]
                    else:
                        fp16_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
                        caches = []
                        for li in range(model.cfg.n_layer):
                            if li < int(layerwise_promote_layers):
                                ks = fp16_cfg
                                kg = fp16_cfg
                                vv = fp16_cfg
                            else:
                                ks = k_sem_cfg
                                kg = k_geo_cfg
                                vv = v_dec_cfg
                            caches.append(
                                model._make_decoupled_layer_cache(
                                    batch_size=int(B),
                                    max_seq_len=int(max_seq),
                                    k_sem_cfg=ks,
                                    k_geo_cfg=kg,
                                    v_cfg=vv,
                                    device=device,
                                )
                            )
                else:
                    caches = [
                        LayerKVCache(
                            batch_size=int(B),
                            max_seq_len=int(max_seq),
                            k_dim=(model.cfg.d_model if model.cfg.attn_mode == "standard" else model.cfg.attn_dim),
                            v_dim=(model.cfg.d_model if model.cfg.attn_mode == "standard" else model.cfg.attn_dim),
                            k_cfg=k_cfg,
                            v_cfg=v_cfg,
                            device=device,
                        )
                        for _ in range(model.cfg.n_layer)
                    ]

                for li, c in enumerate(caches):
                    c.decode_block = int(kv_decode_block)
                    if layerwise_promote_layers is not None and model.cfg.attn_mode == "decoupled" and li < int(layerwise_promote_layers):
                        c.fused = "none"
                    else:
                        c.fused = str(kv_fused)
                return caches

            max_seq_m, k_cfg_m, v_cfg_m, k_sem_cfg_m, k_geo_cfg_m, v_dec_cfg_m, layerwise_m = _make_cfgs_for(self)
            caches_main = _make_caches(self, max_seq=max_seq_m, k_cfg=k_cfg_m, v_cfg=v_cfg_m, k_sem_cfg=k_sem_cfg_m, k_geo_cfg=k_geo_cfg_m, v_dec_cfg=v_dec_cfg_m, layerwise_promote_layers=layerwise_m)

            max_seq_d, k_cfg_d, v_cfg_d, k_sem_cfg_d, k_geo_cfg_d, v_dec_cfg_d, layerwise_d = _make_cfgs_for(draft_model)
            caches_draft = _make_caches(draft_model, max_seq=max_seq_d, k_cfg=k_cfg_d, v_cfg=v_cfg_d, k_sem_cfg=k_sem_cfg_d, k_geo_cfg=k_geo_cfg_d, v_dec_cfg=v_dec_cfg_d, layerwise_promote_layers=layerwise_d)

            # Prefill both models
            logits_main, caches_main = self(prompt, caches=caches_main, pos_offset=0)
            logits_draft, caches_draft = draft_model(prompt, caches=caches_draft, pos_offset=0)
            main_next_logits = logits_main[:, -1, :]
            draft_next_logits = logits_draft[:, -1, :]

            out = prompt
            pos_cur = int(out.size(1))
            generated = 0

            ema_accept = 1.0
            ema_decay = 0.9

            def _finish_without_spec() -> torch.Tensor:
                nonlocal out, pos_cur, generated, main_next_logits, caches_main
                remaining = int(max_new_tokens) - int(generated)
                for _ in range(remaining):
                    tok, _p = _sample_token(main_next_logits)
                    out = torch.cat([out, tok], dim=1)
                    logits_main2, caches_main2 = self(tok, caches=caches_main, pos_offset=pos_cur)
                    caches_main = caches_main2
                    main_next_logits = logits_main2[:, -1, :]
                    pos_cur += 1
                    generated += 1
                return out

            while generated < int(max_new_tokens):
                spec_steps += 1
                remaining = int(max_new_tokens) - int(generated)

                # Optional online disable if acceptance drops.
                if float(spec_disable_below_accept) > 0.0 and ema_accept < float(spec_disable_below_accept):
                    print(f"[spec] disabling speculative decode (ema_accept={ema_accept:.3f} < {spec_disable_below_accept})")
                    out2 = _finish_without_spec()
                    _emit_spec_metrics(total_new_tokens=int(generated))
                    return out2

                k = min(int(spec_k), remaining)
                if k <= 0:
                    break
                proposed_total += int(k)

                def _truncate_all(caches: List[Any], new_pos: int, *, label: str) -> None:
                    """Truncate all layer caches to `new_pos` (used to resync after speculative rollback)."""
                    for li, c in enumerate(caches):
                        if not hasattr(c, "truncate"):
                            raise RuntimeError(f"[spec] cache {label}[{li}] missing truncate(); cannot resync")
                        c.truncate(int(new_pos))

                # Cache position must match the current token position.
                # If caches got ahead (e.g. partial speculative step), truncate them back.
                # If caches are behind, that's a hard bug (can't magically reconstruct KV).
                pos0 = int(caches_main[0].pos)
                if pos0 != pos_cur:
                    if pos0 > pos_cur:
                        print(
                            f"[spec] warning: main cache pos {pos0} != expected {pos_cur}; truncating caches to resync",
                            file=sys.stderr,
                        )
                        _truncate_all(caches_main, pos_cur, label="main")
                        # Draft cache should also be at the same logical position.
                        pos_d = int(caches_draft[0].pos)
                        if pos_d != pos_cur:
                            if pos_d > pos_cur:
                                print(
                                    f"[spec] warning: draft cache pos {pos_d} != expected {pos_cur}; truncating caches to resync",
                                    file=sys.stderr,
                                )
                                _truncate_all(caches_draft, pos_cur, label="draft")
                            else:
                                raise RuntimeError(f"[spec] draft cache behind expected pos: cache={pos_d} expected={pos_cur}")
                        pos0 = pos_cur
                    else:
                        raise RuntimeError(f"[spec] main cache behind expected pos: cache={pos0} expected={pos_cur}")

                # ---- Draft proposes k tokens ----
                proposed: List[torch.Tensor] = []
                q_probs: List[torch.Tensor] = []

                for j in range(k):
                    tok_j, qj = _sample_token(draft_next_logits)
                    proposed.append(tok_j)
                    q_probs.append(qj)
                    # advance draft
                    logits_draft, caches_draft = draft_model(tok_j, caches=caches_draft, pos_offset=pos_cur + j)
                    draft_next_logits = logits_draft[:, -1, :]

                proposed_blk = torch.cat(proposed, dim=1)  # (1,k)

                # ---- Main verifies the block (one forward over k tokens) ----
                logits_block, caches_main = self(proposed_blk, caches=caches_main, pos_offset=pos0)

                accepted = k
                rejected = False
                for i in range(k):
                    yi = proposed_blk[:, i : i + 1]
                    if i == 0:
                        p_logits_i = main_next_logits
                    else:
                        p_logits_i = logits_block[:, i - 1, :]

                    p_i = _probs(p_logits_i)
                    q_i = q_probs[i]

                    if spec_method == "greedy":
                        yi_hat = torch.argmax(p_i, dim=-1, keepdim=True)
                        if int(yi_hat.item()) != int(yi.item()):
                            accepted = i
                            rejected = True
                            # fallback token: sample from p
                            z = yi_hat
                            break
                        continue

                    # reject_sampling
                    p_tok = p_i.gather(-1, yi).clamp(min=1e-12)
                    q_tok = q_i.gather(-1, yi).clamp(min=1e-12)
                    ratio = (p_tok / q_tok).clamp(max=1.0)
                    u = torch.rand_like(ratio)
                    if bool((u <= ratio).item()):
                        continue

                    accepted = i
                    rejected = True
                    # Sample from r ∝ [p - q]+
                    r = torch.relu(p_i - q_i)
                    z_probs = r
                    z_sum = float(z_probs.sum().item())
                    if not (z_sum > 0.0) or math.isnan(z_sum):
                        z_probs = p_i
                    z = _sample_from_probs(z_probs)
                    break

                if rejected:
                    # Roll back caches to accepted prefix.
                    new_pos = pos0 + int(accepted)
                    _truncate_all(caches_main, new_pos)
                    _truncate_all(caches_draft, new_pos)

                    # Append accepted draft tokens (if any) to output.
                    if accepted > 0:
                        out = torch.cat([out, proposed_blk[:, :accepted]], dim=1)
                        pos_cur += int(accepted)
                        generated += int(accepted)
                        accepted_total += int(accepted)

                    # Append replacement token z, advance both models.
                    out = torch.cat([out, z], dim=1)
                    logits_main, caches_main = self(z, caches=caches_main, pos_offset=pos_cur)
                    logits_draft, caches_draft = draft_model(z, caches=caches_draft, pos_offset=pos_cur)
                    main_next_logits = logits_main[:, -1, :]
                    draft_next_logits = logits_draft[:, -1, :]
                    pos_cur += 1
                    generated += 1

                    # Update accept EMA (accepted/(proposed))
                    step_accept = float(accepted) / float(max(1, k))
                    ema_accept = ema_decay * ema_accept + (1.0 - ema_decay) * step_accept
                    continue

                # All accepted.
                out = torch.cat([out, proposed_blk], dim=1)
                pos_cur += int(k)
                generated += int(k)
                accepted_total += int(k)

                # Next-token logits after the last accepted token.
                main_next_logits = logits_block[:, -1, :]
                # draft_next_logits already updated during proposal loop.

                # Optionally sample one extra verifier token (paper-style).
                if spec_extra_token and generated < int(max_new_tokens):
                    z_tok, _ = _sample_token(main_next_logits)
                    out = torch.cat([out, z_tok], dim=1)
                    logits_main, caches_main = self(z_tok, caches=caches_main, pos_offset=pos_cur)
                    logits_draft, caches_draft = draft_model(z_tok, caches=caches_draft, pos_offset=pos_cur)
                    main_next_logits = logits_main[:, -1, :]
                    draft_next_logits = logits_draft[:, -1, :]
                    pos_cur += 1
                    generated += 1

                ema_accept = ema_decay * ema_accept + (1.0 - ema_decay) * 1.0

            _emit_spec_metrics(total_new_tokens=int(generated))
            return out
        finally:
            if was_training_main:
                self.train()
            if was_training_draft:
                draft_model.train()


