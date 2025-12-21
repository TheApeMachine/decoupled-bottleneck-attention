"""Decode-plan self-optimizer.

This file stays intentionally small: it orchestrates plan selection and delegates
keying, persistence, search-space construction, and benchmarking to helper
modules in `production.optimizer.tuner.*`.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from production.optimizer.tuner.config import KVSelfOptConfig
from production.optimizer.tuner.decode_bench import bench_plan as _bench_plan
from production.optimizer.tuner.decode_bench import run_plan as _run_plan
from production.optimizer.tuner.decode_candidates import candidate_plans
from production.optimizer.tuner.decode_keys import decode_plan_key
from production.optimizer.tuner.decode_plan import KVDecodePlan
from production.optimizer.tuner.decode_store import DecodePlanStore


class KVDecodeSelfOptimizer:
    """Self-optimizes decode performance knobs per prefix-length bucket."""

    def __init__(
        self,
        cfg: KVSelfOptConfig,
        *,
        device: torch.device,
        base_fused: str,
        base_decode_block: int,
        log_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.cfg: KVSelfOptConfig = cfg
        self.device: torch.device = device
        self.base_fused: str = str(base_fused)
        self.base_decode_block: int = int(base_decode_block)
        self.log_callback: Callable[[dict[str, object]], None] | None = log_callback

        self._store = DecodePlanStore(cfg.cache_path, verbose=bool(cfg.verbose))
        self._plans: dict[str, KVDecodePlan] = self._store.load()
        self._last_probe_step: dict[str, int] = {}
        self._step_counter: int = 0

    def bench_plan(
        self,
        *,
        attn: object,
        cache: object,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        plan: KVDecodePlan,
        sem_scale: float,
        geo_scale: float,
        baseline_out: torch.Tensor | None,
    ) -> float:
        """Benchmark a plan, optionally verifying vs. `baseline_out`."""
        return _bench_plan(
            self.cfg,
            device=self.device,
            attn=attn,
            cache=cache,
            q_sem=q_sem,
            q_geo=q_geo,
            plan=plan,
            sem_scale=sem_scale,
            geo_scale=geo_scale,
            baseline_out=baseline_out,
        )

    def maybe_get_plan(self, *, attn: object, cache: object, prefix_len: int) -> KVDecodePlan | None:
        """Get the best plan for the given attention/cache signature."""
        if self.cfg.mode == "none":
            return None

        self._step_counter += 1
        k = decode_plan_key(device=self.device, attn=attn, cache=cache, prefix_len=int(prefix_len))

        if k in self._plans and self.cfg.mode == "startup":
            return self._plans[k]

        if k in self._plans and self.cfg.mode == "online":
            last = self._last_probe_step.get(k, -10**9)
            if (self._step_counter - last) < int(self.cfg.interval):
                return self._plans[k]

        plans = candidate_plans(
            self.cfg,
            device=self.device,
            base_decode_block=self.base_decode_block,
            base_fused=self.base_fused,
            cache=cache,
        )
        if not plans:
            return None

        batch_size = 1
        try:
            ks = getattr(cache, "k_sem", None)
            if ks is not None:
                if getattr(ks, "buf", None) is not None:
                    batch_size = int(getattr(ks, "buf").shape[0])
                elif getattr(ks, "q", None) is not None:
                    batch_size = int(getattr(ks, "q").shape[0])
        except (AttributeError, TypeError, ValueError):
            batch_size = 1

        head_count = int(getattr(attn, "H", 1))

        q_sem = torch.randn(
            (batch_size, head_count, 1, int(getattr(attn, "sem_head_dim"))),
            device=self.device,
            dtype=torch.float16,
        )
        q_geo = torch.randn(
            (batch_size, head_count, 1, int(getattr(attn, "geo_head_dim"))),
            device=self.device,
            dtype=torch.float16,
        )

        sem_scale = 1.0 / math.sqrt(float(getattr(attn, "sem_head_dim")))
        geo_scale = 1.0 / math.sqrt(float(getattr(attn, "geo_head_dim")))

        baseline_plan = self._plans.get(k, KVDecodePlan(fused="none", decode_block=self.base_decode_block))
        baseline_out = None
        if self.cfg.verify:
            try:
                baseline_out = _run_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=baseline_plan,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                ).detach()
            except (RuntimeError, ValueError, TypeError, AttributeError):
                baseline_out = None

        best_plan: KVDecodePlan | None = None
        best_ms: float = float("inf")

        for p in plans:
            try:
                ms = self.bench_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=p,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                    baseline_out=baseline_out,
                )
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                if self.cfg.verbose:
                    print(f"[selfopt] plan failed {p}: {e}")
                ms = float("inf")
            if ms < best_ms:
                best_ms = ms
                best_plan = p

        if best_plan is not None and k in self._plans and self.cfg.mode == "online":
            old = self._plans[k]
            try:
                old_ms = self.bench_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=old,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                    baseline_out=baseline_out,
                )
            except (RuntimeError, ValueError, TypeError, AttributeError):
                old_ms = float("inf")
            if not best_ms < (old_ms * 1.0 - float(self.cfg.hysteresis)):
                best_plan = old
                best_ms = old_ms
            self._last_probe_step[k] = self._step_counter

        if best_plan is not None:
            self._plans[k] = best_plan
            self._store.save(self._plans)
            if self.cfg.verbose:
                print(f"[selfopt] bucket_key={k} -> {best_plan} ({best_ms:.3f} ms)")
            if self.log_callback:
                self.log_callback(
                    {
                        "type": "analysis",
                        "subtype": "selfopt_decode",
                        "bucket_key": k,
                        "decode_block": int(best_plan.decode_block),
                        "fused": str(best_plan.fused),
                        "block_n": int(best_plan.block_n),
                        "best_ms": float(best_ms),
                    }
                )
        return best_plan

    def choose_plan(
        self,
        *,
        attn: object,
        cache: object,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        sem_scale: float,
        geo_scale: float,
    ) -> KVDecodePlan | None:
        """Backward-compatible alias for `maybe_get_plan` (query values do not affect timing)."""
        _ = (q_sem, q_geo, sem_scale, geo_scale)
        prefix_len = int(getattr(cache, "pos", 0))
        return self.maybe_get_plan(attn=attn, cache=cache, prefix_len=prefix_len)


