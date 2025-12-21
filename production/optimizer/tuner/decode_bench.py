"""Benchmark helpers for decode-plan tuning."""

from __future__ import annotations

import time
from typing import Callable, cast

import torch

from production.optimizer.tuner.config import KVSelfOptConfig
from production.optimizer.tuner.decode_plan import KVDecodePlan


def time_ms(*, device: torch.device, fn: Callable[[], None]) -> float:
    """Time a callable, using CUDA events when available."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(cast(object, start).elapsed_time(cast(object, end)))
    t0 = time.perf_counter()
    fn()
    return float((time.perf_counter() - t0) * 1000.0)


def run_plan(
    *,
    attn: object,
    cache: object,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    plan: KVDecodePlan,
    sem_scale: float,
    geo_scale: float,
) -> torch.Tensor:
    """Execute one decode plan (best-effort dynamic dispatch)."""
    plan.apply_to_cache(cache)

    fused = str(plan.fused)
    match fused:
        case "none":
            return cast(
                torch.Tensor,
                getattr(attn, "_streaming_decode_attn_decoupled")(
                    q_sem=q_sem,
                    q_geo=q_geo,
                    k_sem_cache=getattr(cache, "k_sem"),
                    k_geo_cache=getattr(cache, "k_geo"),
                    v_cache=getattr(cache, "v"),
                    sem_head_dim=getattr(attn, "sem_head_dim"),
                    geo_head_dim=getattr(attn, "geo_head_dim"),
                    v_head_dim=getattr(attn, "v_head_dim"),
                    decode_block=int(plan.decode_block),
                    sem_scale=float(sem_scale),
                    geo_scale=float(geo_scale),
                    k_sem_null=None,
                    k_geo_null=None,
                    v_null=None,
                ),
            )
        case "triton1pass":
            return cast(
                torch.Tensor,
                getattr(attn, "_fused_decode_attn_decoupled_q4q8q4")(
                    q_sem=q_sem,
                    q_geo=q_geo,
                    cache=cache,
                    sem_head_dim=getattr(attn, "sem_head_dim"),
                    geo_head_dim=getattr(attn, "geo_head_dim"),
                    v_head_dim=getattr(attn, "v_head_dim"),
                    decode_block=int(plan.decode_block),
                    sem_scale=float(sem_scale),
                    geo_scale=float(geo_scale),
                ),
            )
        case "triton2pass":
            return cast(
                torch.Tensor,
                getattr(attn, "_fused_decode_attn_decoupled_q4q8q4_2pass")(
                    q_sem=q_sem,
                    q_geo=q_geo,
                    cache=cache,
                    sem_head_dim=getattr(attn, "sem_head_dim"),
                    geo_head_dim=getattr(attn, "geo_head_dim"),
                    v_head_dim=getattr(attn, "v_head_dim"),
                    decode_block=int(plan.decode_block),
                    sem_scale=float(sem_scale),
                    geo_scale=float(geo_scale),
                ),
            )
        case _:
            raise ValueError(fused)


def bench_plan(
    cfg: KVSelfOptConfig,
    *,
    device: torch.device,
    attn: object,
    cache: object,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    plan: KVDecodePlan,
    sem_scale: float,
    geo_scale: float,
    baseline_out: torch.Tensor | None,
) -> float:
    """Benchmark a plan, optionally verifying against a baseline output."""

    def fn() -> None:
        out = run_plan(
            attn=attn,
            cache=cache,
            q_sem=q_sem,
            q_geo=q_geo,
            plan=plan,
            sem_scale=sem_scale,
            geo_scale=geo_scale,
        )
        if cfg.verify and baseline_out is not None:
            err = (out.float() - baseline_out.float()).abs().max().item()
            if err > float(cfg.verify_tol):
                raise RuntimeError(f"verify failed: max_abs_err={err} > tol={cfg.verify_tol}")

    for _ in range(max(0, int(cfg.warmup))):
        fn()
    ms = 0.0
    iters = max(1, int(cfg.iters))
    for _ in range(iters):
        ms += time_ms(device=device, fn=fn)
    return float(ms) / float(iters)


