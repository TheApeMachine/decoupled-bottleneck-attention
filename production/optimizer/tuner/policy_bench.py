"""Benchmark helpers for cache-policy tuning."""

from __future__ import annotations

import math

import torch

from production.kvcache_backend import DecoupledLayerKVCache

from production.optimizer.tuner.cache_policy import KVCachePolicy
from production.optimizer.tuner.config import KVSelfOptConfig
from production.optimizer.tuner.decode_optimizer import KVDecodeSelfOptimizer
from production.optimizer.tuner.decode_plan import KVDecodePlan
from production.optimizer.tuner.triton_availability import TRITON_AVAILABLE


def supports_fused_q4q8q4(*, device: torch.device, attn: object, policy: KVCachePolicy) -> bool:
    """Whether fused kernels are *eligible* for this policy/device."""
    if not TRITON_AVAILABLE:
        return False
    null_attn = bool(getattr(getattr(attn, "cfg", object()), "null_attn", False))
    return (
        str(policy.k_sem_kind) == "q4_0"
        and str(policy.k_geo_kind) == "q8_0"
        and str(policy.v_kind) == "q4_0"
        and int(policy.k_sem_qblock) == 32
        and int(policy.k_geo_qblock) == 32
        and int(policy.v_qblock) == 32
        and (not null_attn)
        and (device.type == "cuda")
    )


def bench_policy_ms(
    cfg: KVSelfOptConfig,
    *,
    device: torch.device,
    attn: object,
    model_cfg: object,
    batch_size: int,
    base_decode_block: int,
    base_fused: str,
    policy: KVCachePolicy,
    prefix_len: int,
) -> float:
    """Benchmark decode latency for a given cache `policy` at `prefix_len`."""
    batch_size = int(batch_size)
    head_count = int(getattr(attn, "H", int(getattr(model_cfg, "n_head"))))

    sem_dim = int(getattr(model_cfg, "sem_dim"))
    geo_dim = int(getattr(model_cfg, "geo_dim"))
    attn_dim = int(getattr(model_cfg, "attn_dim"))

    sem_hd = int(getattr(attn, "sem_head_dim", sem_dim // max(1, head_count)))
    geo_hd = int(getattr(attn, "geo_head_dim", geo_dim // max(1, head_count)))
    v_hd = int(getattr(attn, "v_head_dim", attn_dim // max(1, head_count)))
    _ = v_hd

    k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
    cache = DecoupledLayerKVCache(
        batch_size=batch_size,
        max_seq_len=int(prefix_len) + 1,
        k_sem_dim=sem_dim,
        k_geo_dim=geo_dim,
        v_dim=attn_dim,
        k_sem_cfg=k_sem_cfg,
        k_geo_cfg=k_geo_cfg,
        v_cfg=v_cfg,
        device=device,
    )

    # Runtime hints (set dynamically to avoid tight coupling to cache impl types).
    setattr(cache, "decode_block", int(base_decode_block))
    setattr(cache, "fused", "none")

    with torch.no_grad():
        k_sem = torch.randn((batch_size, prefix_len, sem_dim), device=device, dtype=torch.float16)
        k_geo = torch.randn((batch_size, prefix_len, geo_dim), device=device, dtype=torch.float16)
        v = torch.randn((batch_size, prefix_len, attn_dim), device=device, dtype=torch.float16)
        cache.append(k_sem, k_geo, v)

    q_sem = torch.randn((batch_size, head_count, 1, sem_hd), device=device, dtype=torch.float16)
    q_geo = torch.randn((batch_size, head_count, 1, geo_hd), device=device, dtype=torch.float16)
    sem_scale = 1.0 / math.sqrt(float(sem_hd))
    geo_scale = 1.0 / math.sqrt(float(geo_hd))

    fused_menu: list[str]
    if str(base_fused) in ("triton1pass", "triton2pass"):
        fused_menu = [str(base_fused)]
    elif str(base_fused) == "none":
        fused_menu = ["none"]
    else:
        fused_menu = ["none"]
        if supports_fused_q4q8q4(device=device, attn=attn, policy=policy):
            fused_menu += ["triton1pass", "triton2pass"]

    decode_blocks = cfg.decode_blocks if cfg.scope in ("decode", "all") else (int(base_decode_block),)
    decode_blocks = tuple(int(x) for x in decode_blocks if int(x) > 0)

    best = float("inf")
    tuner = KVDecodeSelfOptimizer(
        KVSelfOptConfig(
            mode="startup",
            scope="decode",
            decode_blocks=decode_blocks,
            block_ns=cfg.block_ns,
            warps=cfg.warps,
            stages=cfg.stages,
            warmup=max(0, int(cfg.policy_warmup)),
            iters=max(1, int(cfg.policy_iters)),
            interval=cfg.interval,
            hysteresis=cfg.hysteresis,
            cache_path=None,
            verbose=False,
            verify=False,
            verify_tol=cfg.verify_tol,
        ),
        device=device,
        base_fused="auto",
        base_decode_block=int(base_decode_block),
    )

    for fused in fused_menu:
        for db in decode_blocks:
            plan = KVDecodePlan(fused=str(fused), decode_block=int(db))
            try:
                plan.apply_to_cache(cache)
                ms = tuner.bench_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=plan,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                    baseline_out=None,
                )
            except (RuntimeError, ValueError, TypeError, AttributeError):
                ms = float("inf")
            if ms < best:
                best = ms
    return float(best)


