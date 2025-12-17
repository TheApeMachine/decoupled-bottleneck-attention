#!/usr/bin/env python3
""" 
v28_transformer_decoupled_bottleneck_instrumented.py

One-file research Transformer that implements, in a runnable way:

1) RoPE (rotary positional embeddings).
2) KV-cache quantization (q8_0 / q4_0 / nf4) for generation/inference.
3) Bottleneck and Decoupled Bottleneck attention:
      score = (Q_sem · K_sem^T) + (Q_geo · K_geo^T)
   with RoPE applied only on the geometric path.

"Survive scale" upgrades in this v28 variant:

  D) Training: gradient accumulation, gradient checkpointing, vectorized batch sampler,
     mixed precision on CUDA/MPS/CPU (torch.amp), memory/timing dashboards (rich),
     and lean optimizers (Lion) to push bigger models on modest hardware.

  A) Streaming ("online softmax") decode that dequantizes only small sequence blocks.
  B) Optional fused GPU kernels (Triton, if installed).
     - v23: 1-pass fused decode update (dequant + online-softmax update per block).
     - v24: 2-pass "FlashAttention-style" split-K decode:
            Pass 1: parallel partitions compute (m_i, l_i, O_i) per partition.
            Pass 2: reduce partitions into a single (m, l, O) for the row.
            This gives sequence-length parallelism (useful when you have 1 query token, huge KV).

  C) v25: Self-optimizing runtime tuning (optional).
     - Online hill-climb / bucketed microbench to pick decode_block + fused mode.
     - Tunable kernel launch params (BLOCK_N / num_warps / num_stages) via cache attributes.
     - JSON persistence so the model 'remembers' what was fastest on a given GPU + shape.

Data format: whitespace-separated integer token IDs in a single file.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import json
import sys
import platform
import datetime
import traceback
import contextlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tiktoken
except ImportError:
    tiktoken = None


# Optional: fused kernels via Triton.
# This file runs without Triton; if you install it and run on CUDA, decode can use fused dequant+attn updates.
TRITON_AVAILABLE = False
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    TRITON_AVAILABLE = True
except Exception:
    triton = None  # type: ignore
    tl = None  # type: ignore


# -----------------------------
# Utils
# -----------------------------

def pick_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




# -----------------------------
# v25: Self-Optimizing Runtime Tuner (decode performance)
# -----------------------------
#
# Motivation:
#   - KV-cache bandwidth dominates long-context decode.
#   - The "best" decode_block / fused kernel choice / launch params are *hardware-dependent*.
#   - We treat these as runtime knobs and auto-pick them with a small, guardrailed microbench.
#
# Design:
#   - Bucketed by prefix length (power-of-two) so we don't retune every token.
#   - Optional "online" mode: occasionally probe a neighbor setting and keep it if faster.
#   - JSON persistence (optional) so the script remembers the best plan per GPU + shape.

@dataclass
class KVSelfOptConfig:
    mode: Literal["none", "startup", "online"] = "none"

    # v26: which knobs to optimize.
    #   - decode: decode_block, fused kernel choice, launch params (v25 behavior)
    #   - cache : kv_residual length, quantization kinds, qblock sizes (startup only)
    #   - all   : both
    scope: Literal["decode", "cache", "all"] = "all"

    # ---------------------------
    # Decode-plan tuning (v25)
    # ---------------------------
    # Candidate decode partition sizes (affects streaming decode slice size and split-K partition size).
    decode_blocks: Tuple[int, ...] = (256, 512, 1024, 2048)
    # Candidate Triton tile sizes (BLOCK_N). Only used for fused kernels.
    block_ns: Tuple[int, ...] = (128,)
    # Candidate launch params (only used for fused kernels).
    warps: Tuple[int, ...] = (4, 8)
    stages: Tuple[int, ...] = (2, 3)

    warmup: int = 1
    iters: int = 3

    # online mode: at most once every N decode steps per bucket, try a neighbor and keep it if faster.
    interval: int = 256
    hysteresis: float = 0.03  # require >=3% improvement to switch

    cache_path: Optional[str] = None
    verbose: bool = False

    # Optional correctness guardrail when comparing decode candidates (slow, but useful while bringing kernels up).
    verify: bool = False
    verify_tol: float = 5e-3

    # ---------------------------
    # Cache-policy tuning (v26)
    # ---------------------------
    # Candidate kv_residual hot-window lengths (fp16 ring) for quantized caches.
    residuals: Tuple[int, ...] = (0, 32, 64, 128)
    # Candidate quantization block sizes (qblock). For q4/nf4 we enforce evenness in make_quantspec().
    qblocks: Tuple[int, ...] = (16, 32, 64)

    # Candidate quantization kinds (decoupled only).
    # These are ordered roughly from "more compressed" -> "less compressed".
    k_sem_kinds: Tuple["KVCacheKind", ...] = ("q4_0", "nf4", "q8_0", "fp16")
    k_geo_kinds: Tuple["KVCacheKind", ...] = ("q8_0", "q4_0", "fp16")
    v_kinds: Tuple["KVCacheKind", ...] = ("q4_0", "q8_0", "fp16")

    # Memory budget for cache-policy tuning.
    # If mem_budget_mb is None, we set a budget as: baseline(residual=0) * (1 + mem_overhead_frac).
    mem_budget_mb: Optional[float] = None
    mem_overhead_frac: float = 0.10  # 10% headroom by default (keeps memory-reduction objective intact)

    # Policy tuning benchmark details
    policy_prefix_len: Optional[int] = None  # if None, derived from prompt/max_seq buckets
    policy_warmup: int = 1
    policy_iters: int = 3
    policy_hysteresis: float = 0.02          # accept >=2% speed improvement when adjusting policy
    prefer_lower_mem_within: float = 0.02    # tie-break: if speed within 2%, prefer lower mem

    # Optional quality guard for policy tuning (slow).
    # Compares teacher-forced logits vs an fp16-cache baseline on a short calibration sequence.
    policy_quality: bool = False
    calib_tokens: Optional[str] = None   # whitespace-separated ints OR a path to a token file
    calib_prefill: int = 64
    calib_decode_steps: int = 8
    quality_tol: float = 0.5             # max abs logit error vs fp16 baseline allowed
@dataclass
class KVDecodePlan:
    fused: str                    # "none" | "triton1pass" | "triton2pass"
    decode_block: int

    # Fused-kernel tunables (ignored for fused="none")
    block_n: int = 128
    num_warps_1pass: int = 4
    num_stages_1pass: int = 2

    num_warps_part: int = 4
    num_stages_part: int = 2

    num_warps_reduce: int = 1
    num_stages_reduce: int = 1

    def apply_to_cache(self, cache: Any) -> None:
        # Core knobs used by attention forward path
        cache.decode_block = int(self.decode_block)
        cache.fused = str(self.fused)

        # Fused-kernel internal knobs (consumed inside the Triton wrappers).
        cache.block_n = int(self.block_n)
        cache.num_warps_1pass = int(self.num_warps_1pass)
        cache.num_stages_1pass = int(self.num_stages_1pass)
        cache.num_warps_part = int(self.num_warps_part)
        cache.num_stages_part = int(self.num_stages_part)
        cache.num_warps_reduce = int(self.num_warps_reduce)
        cache.num_stages_reduce = int(self.num_stages_reduce)


def _pow2_bucket(n: int) -> int:
    # bucket 1,2,4,8,...; clamp n<=0 to 0
    if n <= 0:
        return 0
    return 1 << (int(n - 1).bit_length())


def _device_sig(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "cuda"
        props = torch.cuda.get_device_properties(idx)
        return f"cuda:{idx}:{name}:cc{props.major}{props.minor}"
    return device.type


class KVDecodeSelfOptimizer:
    """Self-optimizes decode performance knobs per prefix-length bucket.

    This is intentionally conservative:
      - It only targets decode-time knobs (no model weight changes).
      - It only activates for decoupled q4/q8/q4 KV caches when Triton is available.
      - It can optionally verify candidate outputs against a baseline (slow).
    """
    def __init__(
        self,
        cfg: KVSelfOptConfig,
        *,
        device: torch.device,
        base_fused: str,
        base_decode_block: int,
        log_callback: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.device = device
        self.base_fused = base_fused
        self.base_decode_block = int(base_decode_block)
        self.log_callback = log_callback

        self._plans: Dict[str, KVDecodePlan] = {}
        self._last_probe_step: Dict[str, int] = {}
        self._step_counter: int = 0

        # Persistence
        self._cache_path = cfg.cache_path
        if self._cache_path:
            try:
                if os.path.exists(self._cache_path):
                    with open(self._cache_path, "r") as f:
                        raw0 = json.load(f)
                    # Back-compat:
                    #   v25 wrote {key -> KVDecodePlan}
                    #   v26 writes {"version": 26, "decode_plans": {key -> KVDecodePlan}, ...}
                    raw = raw0.get("decode_plans", raw0) if isinstance(raw0, dict) else {}
                    if isinstance(raw, dict):
                        for k, v in raw.items():
                            self._plans[k] = KVDecodePlan(**v)
            except Exception as e:
                if cfg.verbose:
                    print(f"[selfopt] Failed to load cache '{self._cache_path}': {e}")

    def _save(self) -> None:
        if not self._cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
            decode_plans = {k: asdict(v) for k, v in self._plans.items()}

            # Preserve other sections (e.g., cache policies) if the file already exists.
            root: Dict[str, Any] = {"version": 26, "decode_plans": decode_plans}
            if os.path.exists(self._cache_path):
                try:
                    with open(self._cache_path, "r") as f:
                        prev = json.load(f)
                    if isinstance(prev, dict):
                        if "decode_plans" in prev:
                            root = dict(prev)
                            root["version"] = 26
                            root["decode_plans"] = decode_plans
                        else:
                            # Older format: prev itself was the decode_plans mapping.
                            root = {"version": 26, "decode_plans": decode_plans}
                except Exception:
                    pass

            with open(self._cache_path, "w") as f:
                json.dump(root, f, indent=2, sort_keys=True)
        except Exception as e:
            if self.cfg.verbose:
                print(f"[selfopt] Failed to save cache '{self._cache_path}': {e}")

    def _key(self, *, attn: Any, cache: Any, L_prefix: int) -> str:
        # Key should be stable across runs on the same GPU/model.
        bucket = _pow2_bucket(L_prefix)
        # Capture the quantization policy in the signature (we only tune for q4/q8/q4 anyway).
        try:
            ksig = f"ksem={cache.k_sem.kind},kgeo={cache.k_geo.kind},v={cache.v.kind}"
        except Exception:
            ksig = "kv=unknown"
        try:
            dims = f"H={attn.H},hd_sem={attn.sem_head_dim},hd_geo={attn.geo_head_dim},hd_v={attn.v_head_dim}"
        except Exception:
            dims = "dims=unknown"
        return f"{_device_sig(self.device)}|{bucket}|{dims}|{ksig}"

    def _allowed_fused_modes(self, *, cache: Any) -> List[str]:
        # Honor base_fused, but allow "auto" to explore.
        if self.base_fused == "none":
            return ["none"]
        # If Triton isn't available, only streaming is possible.
        if not _triton_decoupled_q4q8q4_available():
            return ["none"]
        # Only meaningful for the decoupled q4/q8/q4 policy.
        ok = True
        try:
            ok = (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0")
        except Exception:
            ok = False
        if not ok:
            return ["none"]

        if self.base_fused in ("triton1pass", "triton2pass"):
            return [self.base_fused]
        # auto: allow exploring all, including streaming (sometimes wins for short prefixes).
        return ["none", "triton1pass", "triton2pass"]

    def _candidate_plans(self, *, cache: Any) -> List[KVDecodePlan]:
        cfg = self.cfg
        fused_modes = self._allowed_fused_modes(cache=cache)
        decode_blocks = list(dict.fromkeys([self.base_decode_block, *cfg.decode_blocks]))
        decode_blocks = [int(x) for x in decode_blocks if int(x) > 0]
        decode_blocks.sort()

        block_ns = [int(x) for x in cfg.block_ns if int(x) > 0]
        if not block_ns:
            block_ns = [128]

        warps = [int(x) for x in cfg.warps if int(x) > 0]
        if not warps:
            warps = [4]

        stages = [int(x) for x in cfg.stages if int(x) > 0]
        if not stages:
            stages = [2]

        plans: List[KVDecodePlan] = []
        for fused in fused_modes:
            for db in decode_blocks:
                if fused == "none":
                    plans.append(KVDecodePlan(fused="none", decode_block=db))
                    continue
                # Fused: explore a small menu of launch params.
                for bn in block_ns:
                    if db < bn:
                        continue
                    for w in warps:
                        for st in stages:
                            if fused == "triton1pass":
                                plans.append(KVDecodePlan(
                                    fused=fused,
                                    decode_block=db,
                                    block_n=bn,
                                    num_warps_1pass=w,
                                    num_stages_1pass=st,
                                ))
                            else:
                                # 2-pass: tune partition pass mostly; keep reduce simple.
                                plans.append(KVDecodePlan(
                                    fused=fused,
                                    decode_block=db,
                                    block_n=bn,
                                    num_warps_part=w,
                                    num_stages_part=st,
                                    num_warps_reduce=1,
                                    num_stages_reduce=1,
                                ))
        return plans

    def _time_ms(self, fn) -> float:
        # Accurate GPU timing via CUDA events; CPU fallback to perf_counter.
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(self.device)
            return float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            return float((time.perf_counter() - t0) * 1000.0)

    def _run_plan(
        self,
        *,
        attn: Any,
        cache: Any,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        plan: KVDecodePlan,
        sem_scale: float,
        geo_scale: float,
    ) -> torch.Tensor:
        # Apply knobs for fused wrappers (block_n/warps/etc).
        plan.apply_to_cache(cache)

        if plan.fused == "none":
            return attn._streaming_decode_attn_decoupled(
                q_sem=q_sem,
                q_geo=q_geo,
                k_sem_cache=cache.k_sem,
                k_geo_cache=cache.k_geo,
                v_cache=cache.v,
                sem_head_dim=attn.sem_head_dim,
                geo_head_dim=attn.geo_head_dim,
                v_head_dim=attn.v_head_dim,
                decode_block=plan.decode_block,
                sem_scale=sem_scale,
                geo_scale=geo_scale,
                k_sem_null=None,
                k_geo_null=None,
                v_null=None,
            )
        elif plan.fused == "triton1pass":
            return attn._fused_decode_attn_decoupled_q4q8q4(
                q_sem=q_sem,
                q_geo=q_geo,
                cache=cache,
                sem_head_dim=attn.sem_head_dim,
                geo_head_dim=attn.geo_head_dim,
                v_head_dim=attn.v_head_dim,
                decode_block=plan.decode_block,
                sem_scale=sem_scale,
                geo_scale=geo_scale,
            )
        elif plan.fused == "triton2pass":
            return attn._fused_decode_attn_decoupled_q4q8q4_2pass(
                q_sem=q_sem,
                q_geo=q_geo,
                cache=cache,
                sem_head_dim=attn.sem_head_dim,
                geo_head_dim=attn.geo_head_dim,
                v_head_dim=attn.v_head_dim,
                decode_block=plan.decode_block,
                sem_scale=sem_scale,
                geo_scale=geo_scale,
            )
        else:
            raise ValueError(plan.fused)

    def _bench_plan(
        self,
        *,
        attn: Any,
        cache: Any,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        plan: KVDecodePlan,
        sem_scale: float,
        geo_scale: float,
        baseline_out: Optional[torch.Tensor] = None,
    ) -> float:
        # Warmup for compilation + caches.
        with torch.no_grad():
            for _ in range(max(0, int(self.cfg.warmup))):
                _ = self._run_plan(attn=attn, cache=cache, q_sem=q_sem, q_geo=q_geo, plan=plan, sem_scale=sem_scale, geo_scale=geo_scale)

            if self.cfg.verify and baseline_out is not None:
                out = self._run_plan(attn=attn, cache=cache, q_sem=q_sem, q_geo=q_geo, plan=plan, sem_scale=sem_scale, geo_scale=geo_scale)
                err = (out.to(torch.float32) - baseline_out.to(torch.float32)).abs().max().item()
                if err > float(self.cfg.verify_tol):
                    # Treat as invalid plan by returning inf.
                    if self.cfg.verbose:
                        print(f"[selfopt] reject plan {plan} (max_abs_err={err:.3e} > {self.cfg.verify_tol})")
                    return float("inf")

            # Timed runs.
            times: List[float] = []
            for _ in range(max(1, int(self.cfg.iters))):
                def _call():
                    _ = self._run_plan(attn=attn, cache=cache, q_sem=q_sem, q_geo=q_geo, plan=plan, sem_scale=sem_scale, geo_scale=geo_scale)
                times.append(self._time_ms(_call))
            return float(sum(times) / len(times))

    def maybe_get_plan(self, *, attn: Any, cache: Any, L_prefix: int) -> Optional[KVDecodePlan]:
        if self.cfg.mode == "none":
            return None

        self._step_counter += 1
        k = self._key(attn=attn, cache=cache, L_prefix=L_prefix)

        # If we already have a plan and we're not doing online probes, just return it.
        if k in self._plans and self.cfg.mode == "startup":
            return self._plans[k]

        # Online mode: probe at most once per interval.
        if k in self._plans and self.cfg.mode == "online":
            last = self._last_probe_step.get(k, -10**9)
            if (self._step_counter - last) < int(self.cfg.interval):
                return self._plans[k]

        # No plan yet, or time to probe/tune.
        plans = self._candidate_plans(cache=cache)
        if not plans:
            return None

        # Build a stable synthetic query (performance doesn't depend on values, but shape matters).
        # Infer batch size from cache tensors (SeqCacheTensor doesn't store batch_size explicitly).
        B = 1
        try:
            if getattr(cache, "k_sem", None) is not None:
                ks = cache.k_sem
                if getattr(ks, "buf", None) is not None:
                    B = int(ks.buf.shape[0])
                elif getattr(ks, "q", None) is not None:
                    B = int(ks.q.shape[0])
        except Exception:
            B = 1
        H = attn.H if hasattr(attn, "H") else 1
        q_sem = torch.randn((B, H, 1, attn.sem_head_dim), device=self.device, dtype=torch.float16)
        q_geo = torch.randn((B, H, 1, attn.geo_head_dim), device=self.device, dtype=torch.float16)
        sem_scale = 1.0 / math.sqrt(float(attn.sem_head_dim))
        geo_scale = 1.0 / math.sqrt(float(attn.geo_head_dim))

        # Baseline for optional correctness check.
        baseline_plan = self._plans.get(k, KVDecodePlan(fused="none", decode_block=self.base_decode_block))
        baseline_out = None
        if self.cfg.verify:
            try:
                baseline_out = self._run_plan(attn=attn, cache=cache, q_sem=q_sem, q_geo=q_geo, plan=baseline_plan, sem_scale=sem_scale, geo_scale=geo_scale).detach()
            except Exception:
                baseline_out = None

        # Choose the best plan by timing (and optional verify constraint).
        best_plan: Optional[KVDecodePlan] = None
        best_ms: float = float("inf")

        for p in plans:
            try:
                ms = self._bench_plan(attn=attn, cache=cache, q_sem=q_sem, q_geo=q_geo, plan=p, sem_scale=sem_scale, geo_scale=geo_scale, baseline_out=baseline_out)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[selfopt] plan failed {p}: {e}")
                ms = float("inf")
            if ms < best_ms:
                best_ms = ms
                best_plan = p

        # Online: require hysteresis if we already had a plan.
        if best_plan is not None and k in self._plans and self.cfg.mode == "online":
            old = self._plans[k]
            # Benchmark old quickly for fair comparison (1 iter).
            try:
                old_ms = self._bench_plan(attn=attn, cache=cache, q_sem=q_sem, q_geo=q_geo, plan=old, sem_scale=sem_scale, geo_scale=geo_scale, baseline_out=baseline_out)
            except Exception:
                old_ms = float("inf")
            if not (best_ms < old_ms * (1.0 - float(self.cfg.hysteresis))):
                best_plan = old
                best_ms = old_ms
            self._last_probe_step[k] = self._step_counter

        if best_plan is not None:
            self._plans[k] = best_plan
            self._save()
            if self.cfg.verbose:
                print(f"[selfopt] bucket_key={k} -> {best_plan} ({best_ms:.3f} ms)")
            
            if self.log_callback:
                self.log_callback({
                    "type": "analysis",
                    "subtype": "selfopt_decode",
                    "bucket_key": k,
                    "decode_block": int(best_plan.decode_block),
                    "fused": str(best_plan.fused),
                    "block_n": int(best_plan.block_n),
                    "best_ms": float(best_ms)
                })

        return best_plan



# -----------------------------
# v26: Cache-policy self-optimizer
# -----------------------------
#
# This targets the "static" knobs that materially affect *memory bandwidth* and *quantization error*:
#   1) kv_residual hot-window length (fp16 tail in the cache)
#   2) quantization kind itself (q4_0 vs nf4 vs q8_0 vs fp16)
#   3) qblock sizes
# and it does so while keeping the core project constraint front-and-center:
#   - memory reduction is the primary objective (we constrain memory with a strict budget)
#
# Philosophy:
#   - Hill-climb locally instead of brute-forcing a combinatorial explosion.
#   - Each accepted move must *earn its keep* (>= policy_hysteresis improvement).
#   - Hard memory budget; optional quality guard (teacher-forced logits).

@dataclass(frozen=True)
class KVCachePolicy:
    # Decoupled cache policy: K_sem, K_geo, V
    k_sem_kind: "KVCacheKind"
    k_geo_kind: "KVCacheKind"
    v_kind: "KVCacheKind"

    k_sem_qblock: int
    k_geo_qblock: int
    v_qblock: int

    residual_len: int

    def _residual_for(self, kind: "KVCacheKind") -> int:
        return int(self.residual_len) if kind not in ("fp16", "fp32") else 0

    def to_tensor_cfgs(self) -> Tuple["KVCacheTensorConfig", "KVCacheTensorConfig", "KVCacheTensorConfig"]:
        k_sem = KVCacheTensorConfig(kind=self.k_sem_kind, qblock=int(self.k_sem_qblock), residual_len=self._residual_for(self.k_sem_kind))
        k_geo = KVCacheTensorConfig(kind=self.k_geo_kind, qblock=int(self.k_geo_qblock), residual_len=self._residual_for(self.k_geo_kind))
        v = KVCacheTensorConfig(kind=self.v_kind, qblock=int(self.v_qblock), residual_len=self._residual_for(self.v_kind))
        return k_sem, k_geo, v

    def short(self) -> str:
        return (
            f"ksem={self.k_sem_kind}@{self.k_sem_qblock},"
            f"kgeo={self.k_geo_kind}@{self.k_geo_qblock},"
            f"v={self.v_kind}@{self.v_qblock},"
            f"resid={self.residual_len}"
        )


def _is_quant(kind: "KVCacheKind") -> bool:
    return kind not in ("fp16", "fp32")


def estimate_seq_cache_bytes(*, batch_size: int, max_seq_len: int, dim: int, cfg: "KVCacheTensorConfig") -> int:
    """Rough-but-useful memory estimator for a SeqCacheTensor."""
    B = int(batch_size)
    L = int(max_seq_len)
    D = int(dim)
    kind = str(cfg.kind)
    if kind == "fp16":
        return B * L * D * 2
    if kind == "fp32":
        return B * L * D * 4

    spec = make_quantspec(cfg.kind, dim, cfg.qblock)

    # Quantized buffers
    if kind == "q8_0":
        q_bytes = B * L * spec.pad_dim * 1
        s_bytes = B * L * spec.n_blocks * 2
    elif kind in ("q4_0", "nf4"):
        q_bytes = B * L * (spec.pad_dim // 2) * 1
        s_bytes = B * L * spec.n_blocks * 2
    else:
        raise ValueError(kind)

    # fp16 residual tail (only allocated for quantized kinds)
    rlen = int(max(0, cfg.residual_len))
    r_eff = min(rlen, L)
    r_bytes = B * r_eff * D * 2

    return int(q_bytes + s_bytes + r_bytes)


def estimate_decoupled_kvcache_bytes(
    *,
    n_layer: int,
    batch_size: int,
    max_seq_len: int,
    sem_dim: int,
    geo_dim: int,
    v_dim: int,
    policy: KVCachePolicy,
) -> int:
    k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
    per_layer = (
        estimate_seq_cache_bytes(batch_size=batch_size, max_seq_len=max_seq_len, dim=sem_dim, cfg=k_sem_cfg)
        + estimate_seq_cache_bytes(batch_size=batch_size, max_seq_len=max_seq_len, dim=geo_dim, cfg=k_geo_cfg)
        + estimate_seq_cache_bytes(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, cfg=v_cfg)
    )
    return int(n_layer) * int(per_layer)


def _as_mb(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


class KVCachePolicySelfOptimizer:
    """Pick a cache policy that fits a strict memory budget and improves decode throughput.

    This is used once at generation startup (because changing cache layouts mid-generation is expensive).
    """
    def __init__(
        self,
        cfg: KVSelfOptConfig,
        *,
        device: torch.device,
        attn: Any,
        model_cfg: Any,
        batch_size: int,
        max_seq_len: int,
        base_policy: KVCachePolicy,
        base_decode_block: int,
        base_fused: str,
    ):
        self.cfg = cfg
        self.device = device
        self.attn = attn
        self.model_cfg = model_cfg
        self.batch_size = int(batch_size)
        self.max_seq_len = int(max_seq_len)
        self.base_policy = base_policy
        self.base_decode_block = int(base_decode_block)
        self.base_fused = str(base_fused)

        # Persistence piggy-backs on cfg.cache_path under "cache_policies" (optional).
        self._cache_path = cfg.cache_path
        self._policy_cache: Dict[str, Dict[str, Any]] = {}
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r") as f:
                    root = json.load(f)
                if isinstance(root, dict):
                    self._policy_cache = dict(root.get("cache_policies", {})) if isinstance(root.get("cache_policies", {}), dict) else {}
            except Exception:
                self._policy_cache = {}

    def _save_policy_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
            root: Dict[str, Any] = {"version": 26, "cache_policies": self._policy_cache}
            # Preserve decode plans (and any other keys) if present.
            if os.path.exists(self._cache_path):
                try:
                    with open(self._cache_path, "r") as f:
                        prev = json.load(f)
                    if isinstance(prev, dict):
                        root = dict(prev)
                        root["version"] = 26
                        root["cache_policies"] = self._policy_cache
                except Exception:
                    pass
            with open(self._cache_path, "w") as f:
                json.dump(root, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _policy_key(self) -> str:
        # Bucket max_seq_len to avoid exploding cache keys.
        max_bucket = _pow2_bucket(self.max_seq_len)
        dims = f"sem={self.model_cfg.sem_dim},geo={self.model_cfg.geo_dim},v={self.model_cfg.attn_dim},H={self.model_cfg.n_head}"
        return f"{_device_sig(self.device)}|decoupled|max={max_bucket}|B={self.batch_size}|{dims}"

    def _budget_bytes(self) -> int:
        if self.cfg.mem_budget_mb is not None:
            return int(float(self.cfg.mem_budget_mb) * 1024.0 * 1024.0)
        # Baseline = same kinds/qblocks as base_policy but with residual=0
        base0 = KVCachePolicy(
            k_sem_kind=self.base_policy.k_sem_kind,
            k_geo_kind=self.base_policy.k_geo_kind,
            v_kind=self.base_policy.v_kind,
            k_sem_qblock=self.base_policy.k_sem_qblock,
            k_geo_qblock=self.base_policy.k_geo_qblock,
            v_qblock=self.base_policy.v_qblock,
            residual_len=0,
        )
        base_bytes = estimate_decoupled_kvcache_bytes(
            n_layer=self.model_cfg.n_layer,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            sem_dim=self.model_cfg.sem_dim,
            geo_dim=self.model_cfg.geo_dim,
            v_dim=self.model_cfg.attn_dim,
            policy=base0,
        )
        return int(base_bytes * (1.0 + float(self.cfg.mem_overhead_frac)))

    def _supports_fused_q4q8q4(self, policy: KVCachePolicy) -> bool:
        if not _triton_decoupled_q4q8q4_available():
            return False
        # Only the q4/q8/q4 + qblock=32 specialization is implemented in Triton in this file.
        return (
            str(policy.k_sem_kind) == "q4_0"
            and str(policy.k_geo_kind) == "q8_0"
            and str(policy.v_kind) == "q4_0"
            and int(policy.k_sem_qblock) == 32
            and int(policy.k_geo_qblock) == 32
            and int(policy.v_qblock) == 32
            and (not getattr(self.attn.cfg, "null_attn", False))
            and (self.device.type == "cuda")
        )

    def _bench_policy_ms(self, policy: KVCachePolicy, *, L_prefix: int) -> float:
        """Benchmark 1 decode attention op for the first layer (proxy for full model)."""
        B = self.batch_size
        H = int(getattr(self.attn, "H", self.model_cfg.n_head))
        sem_hd = int(getattr(self.attn, "sem_head_dim", self.model_cfg.sem_dim // H))
        geo_hd = int(getattr(self.attn, "geo_head_dim", self.model_cfg.geo_dim // H))
        v_hd = int(getattr(self.attn, "v_head_dim", self.model_cfg.attn_dim // H))

        # Build dummy cache (one layer).
        k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
        cache = DecoupledLayerKVCache(
            batch_size=B,
            max_seq_len=L_prefix + 1,
            k_sem_dim=self.model_cfg.sem_dim,
            k_geo_dim=self.model_cfg.geo_dim,
            v_dim=self.model_cfg.attn_dim,
            k_sem_cfg=k_sem_cfg,
            k_geo_cfg=k_geo_cfg,
            v_cfg=v_cfg,
            device=self.device,
        )
        cache.decode_block = self.base_decode_block
        cache.fused = "none"

        # Fill with random KV so the decode path actually reads cache memory.
        with torch.no_grad():
            k_sem = torch.randn((B, L_prefix, self.model_cfg.sem_dim), device=self.device, dtype=torch.float16)
            k_geo = torch.randn((B, L_prefix, self.model_cfg.geo_dim), device=self.device, dtype=torch.float16)
            v = torch.randn((B, L_prefix, self.model_cfg.attn_dim), device=self.device, dtype=torch.float16)
            cache.append(k_sem, k_geo, v)

        q_sem = torch.randn((B, H, 1, sem_hd), device=self.device, dtype=torch.float16)
        q_geo = torch.randn((B, H, 1, geo_hd), device=self.device, dtype=torch.float16)
        sem_scale = 1.0 / math.sqrt(float(sem_hd))
        geo_scale = 1.0 / math.sqrt(float(geo_hd))

        # Decide which decode kernel(s) to consider for this policy.
        # We only do a tiny selection here; the full decode-tuner still runs later.
        fused_menu: List[str]
        if self.base_fused in ("triton1pass", "triton2pass"):
            fused_menu = [self.base_fused]
        elif self.base_fused == "none":
            fused_menu = ["none"]
        else:
            # auto
            fused_menu = ["none"]
            if self._supports_fused_q4q8q4(policy):
                fused_menu += ["triton1pass", "triton2pass"]

        # Use decode_block candidates only when the user enabled decode tuning; else just base.
        decode_blocks = (self.cfg.decode_blocks if self.cfg.scope in ("decode", "all") else (self.base_decode_block,))
        decode_blocks = tuple(int(x) for x in decode_blocks if int(x) > 0)

        # Time all candidates and take the best.
        best = float("inf")
        # We can reuse the decode tuner timing helper.
        tuner = KVDecodeSelfOptimizer(
            KVSelfOptConfig(
                mode="startup",
                scope="decode",
                decode_blocks=decode_blocks,
                block_ns=self.cfg.block_ns,
                warps=self.cfg.warps,
                stages=self.cfg.stages,
                warmup=max(0, int(self.cfg.policy_warmup)),
                iters=max(1, int(self.cfg.policy_iters)),
                interval=self.cfg.interval,
                hysteresis=self.cfg.hysteresis,
                cache_path=None,
                verbose=False,
                verify=False,
                verify_tol=self.cfg.verify_tol,
            ),
            device=self.device,
            base_fused="auto",  # allow selection among fused_menu below via plan filtering
            base_decode_block=self.base_decode_block,
        )

        # Hack: restrict fused modes by temporarily overriding base_fused.
        tuner.base_fused = "auto"

        for fused in fused_menu:
            for db in decode_blocks:
                plan = KVDecodePlan(fused=fused, decode_block=db)
                try:
                    # Apply plan knobs to the cache
                    plan.apply_to_cache(cache)
                    ms = tuner._bench_plan(
                        attn=self.attn,
                        cache=cache,
                        q_sem=q_sem,
                        q_geo=q_geo,
                        plan=plan,
                        sem_scale=sem_scale,
                        geo_scale=geo_scale,
                        baseline_out=None,
                    )
                except Exception:
                    ms = float("inf")
                if ms < best:
                    best = ms

        return float(best)

    def _neighbors(self, p: KVCachePolicy) -> List[KVCachePolicy]:
        cfg = self.cfg

        # Candidate lists (dedup + sorted for numeric fields).
        resid_cands = sorted({int(x) for x in cfg.residuals if int(x) >= 0})
        qb_cands = sorted({int(x) for x in cfg.qblocks if int(x) > 0})

        def neigh_num(cur: int, cands: List[int]) -> List[int]:
            if cur not in cands:
                cands = sorted(set(cands + [cur]))
            i = cands.index(cur)
            out = []
            if i - 1 >= 0:
                out.append(cands[i - 1])
            if i + 1 < len(cands):
                out.append(cands[i + 1])
            return out

        # Kinds
        def neigh_kind(cur: str, cands: Tuple[str, ...]) -> List[str]:
            c = [str(x) for x in cands]
            if cur not in c:
                c = list(dict.fromkeys([cur] + c))
            i = c.index(cur)
            out = []
            if i - 1 >= 0:
                out.append(c[i - 1])
            if i + 1 < len(c):
                out.append(c[i + 1])
            return out

        out: List[KVCachePolicy] = []

        # Residual
        for r in neigh_num(int(p.residual_len), resid_cands):
            out.append(KVCachePolicy(
                k_sem_kind=p.k_sem_kind, k_geo_kind=p.k_geo_kind, v_kind=p.v_kind,
                k_sem_qblock=p.k_sem_qblock, k_geo_qblock=p.k_geo_qblock, v_qblock=p.v_qblock,
                residual_len=r,
            ))

        # qblock: keep them tied for now (simplifies fused compatibility)
        for qb in neigh_num(int(p.k_sem_qblock), qb_cands):
            out.append(KVCachePolicy(
                k_sem_kind=p.k_sem_kind, k_geo_kind=p.k_geo_kind, v_kind=p.v_kind,
                k_sem_qblock=qb, k_geo_qblock=qb, v_qblock=qb,
                residual_len=p.residual_len,
            ))

        # kind tweaks (one tensor at a time)
        for k in neigh_kind(str(p.k_sem_kind), cfg.k_sem_kinds):
            out.append(KVCachePolicy(
                k_sem_kind=k, k_geo_kind=p.k_geo_kind, v_kind=p.v_kind,
                k_sem_qblock=p.k_sem_qblock, k_geo_qblock=p.k_geo_qblock, v_qblock=p.v_qblock,
                residual_len=p.residual_len,
            ))
        for k in neigh_kind(str(p.k_geo_kind), cfg.k_geo_kinds):
            out.append(KVCachePolicy(
                k_sem_kind=p.k_sem_kind, k_geo_kind=k, v_kind=p.v_kind,
                k_sem_qblock=p.k_sem_qblock, k_geo_qblock=p.k_geo_qblock, v_qblock=p.v_qblock,
                residual_len=p.residual_len,
            ))
        for k in neigh_kind(str(p.v_kind), cfg.v_kinds):
            out.append(KVCachePolicy(
                k_sem_kind=p.k_sem_kind, k_geo_kind=p.k_geo_kind, v_kind=k,
                k_sem_qblock=p.k_sem_qblock, k_geo_qblock=p.k_geo_qblock, v_qblock=p.v_qblock,
                residual_len=p.residual_len,
            ))

        # Dedup
        uniq = []
        seen = set()
        for cand in out:
            key = cand.short()
            if key not in seen:
                seen.add(key)
                uniq.append(cand)
        return uniq

    def choose_policy(self, *, prompt_len: int) -> KVCachePolicy:
        """Return the chosen policy (may be base_policy)."""
        if self.cfg.mode == "none":
            return self.base_policy

        # Only meaningful for decoupled attention in this file.
        if getattr(self.model_cfg, "attn_mode", "standard") != "decoupled":
            return self.base_policy

        key = self._policy_key()
        if key in self._policy_cache:
            try:
                p = self._policy_cache[key]
                policy = KVCachePolicy(**p)
                if self.cfg.verbose:
                    print(f"[selfopt] cache-policy hit {key} -> {policy.short()}")
                return policy
            except Exception:
                pass

        # Pick a representative prefix length for benchmarking.
        if self.cfg.policy_prefix_len is not None:
            L = int(self.cfg.policy_prefix_len)
        else:
            # Favor "interesting" long contexts without going beyond max_seq_len.
            # If prompt is short, still benchmark at 1024 so we tune for the regime where KV dominates.
            L = int(min(self.max_seq_len - 1, max(1024, _pow2_bucket(prompt_len))))
        L = max(1, min(L, self.max_seq_len - 1))

        budget = self._budget_bytes()
        if self.cfg.verbose:
            base_mem = estimate_decoupled_kvcache_bytes(
                n_layer=self.model_cfg.n_layer, batch_size=self.batch_size, max_seq_len=self.max_seq_len,
                sem_dim=self.model_cfg.sem_dim, geo_dim=self.model_cfg.geo_dim, v_dim=self.model_cfg.attn_dim,
                policy=self.base_policy,
            )
            print(f"[selfopt] cache-policy budget: { _as_mb(budget):.1f} MB "
                  f"(base={_as_mb(base_mem):.1f} MB, max_seq={self.max_seq_len}, B={self.batch_size})")

        def mem_bytes(pol: KVCachePolicy) -> int:
            return estimate_decoupled_kvcache_bytes(
                n_layer=self.model_cfg.n_layer,
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
                sem_dim=self.model_cfg.sem_dim,
                geo_dim=self.model_cfg.geo_dim,
                v_dim=self.model_cfg.attn_dim,
                policy=pol,
            )

        def ok_mem(pol: KVCachePolicy) -> bool:
            return mem_bytes(pol) <= budget

        # Start from base_policy but clamp to candidate sets (if base is outside).
        cur = self.base_policy
        best = cur
        best_ms = float("inf")

        # If base violates budget, forcibly drop residual first (never hurts memory objective).
        if not ok_mem(cur):
            cur = KVCachePolicy(
                k_sem_kind=cur.k_sem_kind, k_geo_kind=cur.k_geo_kind, v_kind=cur.v_kind,
                k_sem_qblock=cur.k_sem_qblock, k_geo_qblock=cur.k_geo_qblock, v_qblock=cur.v_qblock,
                residual_len=0,
            )

        # Evaluate base.
        try:
            best_ms = self._bench_policy_ms(cur, L_prefix=L)
            best = cur
        except Exception:
            best_ms = float("inf")
            best = cur

        improved = True
        while improved:
            improved = False
            candidates = [p for p in self._neighbors(best) if ok_mem(p)]
            if not candidates:
                break

            # Evaluate all neighbors and pick the best improvement.
            scored: List[Tuple[float, int, KVCachePolicy]] = []
            for p in candidates:
                try:
                    ms = self._bench_policy_ms(p, L_prefix=L)
                except Exception:
                    ms = float("inf")
                scored.append((ms, mem_bytes(p), p))

            scored.sort(key=lambda x: (x[0], x[1]))

            for ms, mb, p in scored:
                # Accept only real improvements (hysteresis).
                if ms < best_ms * (1.0 - float(self.cfg.policy_hysteresis)):
                    if self.cfg.verbose:
                        print(f"[selfopt] cache-policy step: {best.short()} -> {p.short()} "
                              f"({best_ms:.3f}ms -> {ms:.3f}ms, mem={_as_mb(mb):.1f}MB)")
                    best = p
                    best_ms = ms
                    improved = True
                    break

        # Tie-break: if within prefer_lower_mem_within, choose lower-memory policy.
        # This makes the tuner "memory-first" when speed differences are small.
        if self.cfg.prefer_lower_mem_within > 0 and best_ms < float("inf"):
            cur_ms = best_ms
            cur_mem = mem_bytes(best)
            # Consider immediate neighbors only to keep it cheap.
            for p in [p for p in self._neighbors(best) if ok_mem(p)]:
                try:
                    ms = self._bench_policy_ms(p, L_prefix=L)
                except Exception:
                    continue
                if ms <= cur_ms * (1.0 + float(self.cfg.prefer_lower_mem_within)):
                    m = mem_bytes(p)
                    if m < cur_mem:
                        if self.cfg.verbose:
                            print(f"[selfopt] cache-policy tie-break: {best.short()} -> {p.short()} "
                                  f"(ms={ms:.3f} within {self.cfg.prefer_lower_mem_within*100:.1f}%, "
                                  f"mem {_as_mb(cur_mem):.1f}MB -> {_as_mb(m):.1f}MB)")
                        best = p
                        cur_mem = m

        # Persist.
        self._policy_cache[key] = asdict(best)
        self._save_policy_cache()

        if self.cfg.verbose:
            print(f"[selfopt] cache-policy chosen: {best.short()} @ L={L} (best_ms={best_ms:.3f}ms)")

        return best
# -----------------------------
# Tokenizer (fallback for raw text)
# -----------------------------

class WordTokenizer:
    """
    Minimal word-level tokenizer from v20:
      - splits on whitespace
      - inserts <eos> at end of each non-empty line (preserves boundaries)
      - uses <unk> for OOV
    """
    def __init__(self, stoi: Dict[str, int], itos: List[str], unk: str = "<unk>", eos: str = "<eos>"):
        self.stoi = stoi
        self.itos = itos
        self.unk = unk
        self.eos = eos
        self.unk_id = int(stoi[unk])
        self.eos_id = int(stoi[eos])

    @staticmethod
    def _line_to_tokens(line: str) -> List[str]:
        # WikiText-2 uses a lot of formatting; whitespace tokenization keeps it simple and reproducible.
        toks = line.strip().split()
        return toks

    @classmethod
    def build_from_train_text(cls, train_text: str, extra_specials: Optional[List[str]] = None) -> "WordTokenizer":
        extra_specials = extra_specials or []
        vocab = {}
        def add(tok: str):
            if tok not in vocab:
                vocab[tok] = 1
            else:
                vocab[tok] += 1

        # special tokens first
        specials = ["<unk>", "<eos>"] + [t for t in extra_specials if t not in ("<unk>", "<eos>")]
        for s in specials:
            add(s)

        for line in train_text.splitlines():
            toks = cls._line_to_tokens(line)
            if not toks:
                continue
            for t in toks:
                add(t)
            add("<eos>")

        # deterministic ordering: specials first, then alpha by token
        vocab_tokens = [t for t in vocab.keys() if t not in specials]
        vocab_tokens.sort()

        itos = specials + vocab_tokens
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode_text(self, text: str) -> List[int]:
        ids: List[int] = []
        for line in text.splitlines():
            toks = self._line_to_tokens(line)
            if not toks:
                continue
            for t in toks:
                ids.append(self.stoi.get(t, self.unk_id))
            ids.append(self.eos_id)
        return ids

    def decode_ids(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            tok = self.itos[int(i)] if 0 <= int(i) < len(self.itos) else self.unk
            if tok == self.eos:
                out.append("\n")
            else:
                out.append(tok)
        return " ".join(out).replace(" \n ", "\n")

    def vocab_size(self) -> int:
        return len(self.itos)


def read_tokens(path: str, tokenizer_mode: str = "word") -> Tuple[torch.Tensor, int]:
    """
    Returns (tokens_tensor, vocab_size).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    # 1) If user specifically asked for tiktoken
    if tokenizer_mode == "tiktoken":
        if tiktoken is None:
            raise ImportError("Please `pip install tiktoken` to use --tokenizer tiktoken")
        print(f"Loading {path} with tiktoken (gpt2)...")
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode_ordinary(text) 
        return torch.tensor(ids, dtype=torch.long), enc.n_vocab

    # 2) Try reading as space-separated integers (legacy/pre-processed format)
    # We do this manually to avoid numpy.fromfile
    try:
        # Check if file starts with a number? 
        # Actually, let's just try to read it as text and split into ints if it looks like numbers
        # Reading the whole file into memory as string might be heavy but consistent with v20
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        # heuristic: if the first few tokens are digits, assume integer format
        first_tokens = text.split(maxsplit=5)
        if first_tokens and all(t.isdigit() for t in first_tokens):
             ids = [int(t) for t in text.split()]
             if ids:
                 return torch.tensor(ids, dtype=torch.long), max(ids) + 1
    except ValueError:
        pass
        
    # 3) Fallback: WordTokenizer
    print(f"File {path} could not be read as space-separated integers. Tokenizing as words...")
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    tokenizer = WordTokenizer.build_from_train_text(text)
    ids = tokenizer.encode_text(text)
    print(f"Tokenized {len(ids)} tokens. Vocab size: {tokenizer.vocab_size()}")
    return torch.tensor(ids, dtype=torch.long), tokenizer.vocab_size()



def get_batch(tokens_cpu: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tokens_cpu: 1D CPU tensor
    Returns x,y on `device`.
    """
    if tokens_cpu.device.type != "cpu":
        tokens_cpu = tokens_cpu.cpu()
    n = tokens_cpu.size(0)
    if n <= block_size + 1:
        raise ValueError(f"Need > block_size+1 tokens, got {n} with block_size={block_size}")
    ix = torch.randint(0, n - block_size - 1, (batch_size,), device="cpu")
    x = torch.stack([tokens_cpu[i:i + block_size] for i in ix], dim=0).to(device)
    y = torch.stack([tokens_cpu[i + 1:i + block_size + 1] for i in ix], dim=0).to(device)
    return x, y


def neg_inf(dtype: torch.dtype) -> float:
    # Safer than -inf on some backends.
    return float(torch.finfo(dtype).min)


# -----------------------------
# RoPE
# -----------------------------

class RotaryEmbedding(nn.Module):
    """
    RoPE with cached cos/sin.
    """
    def __init__(self, rot_dim: int, base: float = 10000.0):
        super().__init__()
        if rot_dim % 2 != 0:
            raise ValueError(f"rot_dim must be even, got {rot_dim}")
        self.rot_dim = rot_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), str(dtype), int(seq_len))
        if key in self._cache:
            return self._cache[key]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        self._cache[key] = (cos, sin)
        return cos, sin

    def rotate(self, x: torch.Tensor, pos_offset: int) -> torch.Tensor:
        """
        x: (B,H,T,D)
        applies to first rot_dim of D
        """
        B, H, T, D = x.shape
        rot = self.rot_dim
        if rot > D:
            raise ValueError(f"rot_dim {rot} > head_dim {D}")
        cos, sin = self._cos_sin(pos_offset + T, x.device, x.dtype)
        cos = cos[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,rot/2)
        sin = sin[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0)

        x_rot = x[..., :rot]
        x_pass = x[..., rot:]

        x1 = x_rot[..., :rot // 2]
        x2 = x_rot[..., rot // 2:rot]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2, x_pass], dim=-1)


# -----------------------------
# KV cache quantization
# -----------------------------
# -----------------------------
# KV cache quantization
# -----------------------------

# NOTE: This research file keeps the quantization formats deliberately simple and kernel-friendly:
#   - q8_0: per-block symmetric int8 with fp16 scale (absmax / 127)
#   - q4_0: per-block symmetric int4 (packed into uint8) with fp16 scale (absmax / 7)
#   - nf4 : per-block "NormalFloat4" codebook (packed into uint8) with fp16 scale (absmax)
#
# For production-scale inference, you'd normally use fused kernels to avoid dequantizing the whole cache.
# This file adds a streaming ("online softmax") decode path that dequantizes in small sequence blocks.

KVCacheKind = Literal["fp16", "fp32", "q8_0", "q4_0", "nf4"]


@dataclass(frozen=True)
class QuantSpec:
    kind: KVCacheKind
    dim: int
    qblock: int
    pad_dim: int
    n_blocks: int


@dataclass(frozen=True)
class KVCacheTensorConfig:
    kind: KVCacheKind = "fp16"
    qblock: int = 32
    residual_len: int = 0   # keep a small fp16 "hot" window for the newest tokens (ring-buffer)


def _qblock_eff(kind: KVCacheKind, dim: int, qblock: int) -> int:
    qb = min(qblock if qblock > 0 else 32, dim)
    if kind in ("q4_0", "nf4"):
        if dim < 2:
            raise ValueError(f"{kind} cache requires dim >= 2")
        # ensure even qb <= dim (or <= dim-1 if dim is odd)
        max_even = dim if (dim % 2 == 0) else (dim - 1)
        qb = min(qb, max_even)
        if qb < 2:
            qb = 2
        if qb % 2 != 0:
            qb -= 1
    return max(1, qb)


def make_quantspec(kind: KVCacheKind, dim: int, qblock: int) -> QuantSpec:
    qb = _qblock_eff(kind, dim, qblock)
    pad_dim = int(math.ceil(dim / qb) * qb)
    if kind in ("q4_0", "nf4") and (pad_dim % 2 != 0):
        pad_dim += qb
    n_blocks = pad_dim // qb
    return QuantSpec(kind=kind, dim=dim, qblock=qb, pad_dim=pad_dim, n_blocks=n_blocks)


def quantize_q8_0(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (..., dim) float
    returns (q int8 (..., pad_dim), scale fp16 (..., n_blocks))
    """
    if spec.kind != "q8_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)
    amax = x2.abs().amax(dim=-1)  # (N, nb)
    scale = (amax / 127.0).clamp(min=1e-8)
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    q = q.reshape(*orig, pad_dim)
    return q, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_q8_0(q: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "q8_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if q.size(-1) != pad_dim:
        raise ValueError(f"Expected q pad_dim {pad_dim}, got {q.size(-1)}")
    if scale.size(-1) != nb:
        raise ValueError(f"Expected scale n_blocks {nb}, got {scale.size(-1)}")
    orig = q.shape[:-1]
    q2 = q.reshape(-1, pad_dim).reshape(-1, nb, qb).to(torch.float32)
    s2 = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q2 * s2
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


def quantize_q4_0(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Q4_0-like: int4 packed into uint8, with fp16 scale per block.
    returns (packed uint8 (..., pad_dim//2), scale fp16 (..., n_blocks))
    """
    if spec.kind != "q4_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)
    amax = x2.abs().amax(dim=-1)
    scale = (amax / 7.0).clamp(min=1e-8)
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int16)  # int16 for packing
    u = (q + 8).clamp(0, 15).to(torch.uint8)  # 0..15

    # pack two nibbles per byte via arithmetic (works on MPS too)
    u_even = u[..., 0::2]
    u_odd = u[..., 1::2]
    packed = (u_even * 16) + u_odd  # uint8

    packed = packed.reshape(*orig, pad_dim // 2)
    return packed, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_q4_0(packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "q4_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if packed.size(-1) != pad_dim // 2:
        raise ValueError(f"Expected packed last dim {pad_dim//2}, got {packed.size(-1)}")
    if scale.size(-1) != nb:
        raise ValueError(f"Expected scale n_blocks {nb}, got {scale.size(-1)}")

    orig = packed.shape[:-1]
    p2 = packed.reshape(-1, pad_dim // 2).to(torch.int16)
    hi = p2 // 16
    lo = p2 % 16
    u = torch.stack([hi, lo], dim=-1).reshape(-1, pad_dim)  # 0..15
    q = (u - 8).clamp(-8, 7).to(torch.float32)

    q = q.reshape(-1, nb, qb)
    s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q * s
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


# NF4 codebook from QLoRA Appendix E / bitsandbytes (normalized to [-1, 1]).
# (We keep it explicit here so the file stays self-contained.)
NF4_LEVELS = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)


def quantize_nf4(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NF4: non-uniform 4-bit codebook quantization.

    x: (..., dim) float
    returns (packed uint8 (..., pad_dim//2), scale fp16 (..., n_blocks))

    NOTE: This is implemented in pure PyTorch for research (not fast vs kernel implementations).
    """
    if spec.kind != "nf4":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)  # (N, nb, qb)
    amax = x2.abs().amax(dim=-1)  # (N, nb)
    scale = amax.clamp(min=1e-8)  # map into [-1, 1] via absmax

    y = (x2 / scale.unsqueeze(-1)).clamp(-1.0, 1.0)  # (N, nb, qb)
    levels = NF4_LEVELS.to(device=y.device, dtype=torch.float32)

    # nearest-neighbor assignment into 16-level LUT
    # (N, nb, qb, 1) - (16,) -> (N, nb, qb, 16)
    diff = (y.to(torch.float32).unsqueeze(-1) - levels).abs()
    idx = diff.argmin(dim=-1).to(torch.uint8)  # 0..15

    # pack two 4-bit indices per byte (same packing layout as q4_0)
    idx_even = idx[..., 0::2]
    idx_odd = idx[..., 1::2]
    packed = (idx_even * 16) + idx_odd  # uint8

    packed = packed.reshape(*orig, pad_dim // 2)
    return packed, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_nf4(packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "nf4":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if packed.size(-1) != pad_dim // 2:
        raise ValueError(f"Expected packed last dim {pad_dim//2}, got {packed.size(-1)}")
    if scale.size(-1) != nb:
        raise ValueError(f"Expected scale n_blocks {nb}, got {scale.size(-1)}")

    orig = packed.shape[:-1]
    p2 = packed.reshape(-1, pad_dim // 2).to(torch.int16)
    hi = p2 // 16
    lo = p2 % 16
    idx = torch.stack([hi, lo], dim=-1).reshape(-1, pad_dim).to(torch.long)  # 0..15

    levels = NF4_LEVELS.to(device=packed.device, dtype=torch.float32)
    q = levels[idx]  # (N, pad_dim)

    q = q.reshape(-1, nb, qb)
    s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q * s
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


# -----------------------------
# Optional Triton fused kernels (decode only)
# -----------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _kv_decode_update_decoupled_q4q8q4(
        q_sem_ptr, q_geo_ptr,
        k_sem_q_ptr, k_sem_s_ptr,
        k_geo_q_ptr, k_geo_s_ptr,
        v_q_ptr, v_s_ptr,
        m_ptr, d_ptr, o_ptr,
        start: tl.int32,
        # runtime lengths
        L_prefix: tl.int32,
        # meta
        H: tl.constexpr,
        HD_SEM: tl.constexpr,
        HD_GEO: tl.constexpr,
        HD_V: tl.constexpr,
        QBLOCK_SEM: tl.constexpr,
        QBLOCK_GEO: tl.constexpr,
        QBLOCK_V: tl.constexpr,
        SEM_SCALE: tl.constexpr,
        GEO_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SUBBLOCKS: tl.constexpr,
        # strides
        stride_qsem_b: tl.constexpr, stride_qsem_h: tl.constexpr,
        stride_qgeo_b: tl.constexpr, stride_qgeo_h: tl.constexpr,
        stride_ksq_b: tl.constexpr, stride_ksq_t: tl.constexpr,
        stride_kss_b: tl.constexpr, stride_kss_t: tl.constexpr, stride_kss_c: tl.constexpr,
        stride_kgq_b: tl.constexpr, stride_kgq_t: tl.constexpr,
        stride_kgs_b: tl.constexpr, stride_kgs_t: tl.constexpr, stride_kgs_c: tl.constexpr,
        stride_vq_b: tl.constexpr, stride_vq_t: tl.constexpr,
        stride_vs_b: tl.constexpr, stride_vs_t: tl.constexpr, stride_vs_c: tl.constexpr,
        stride_o: tl.constexpr,
    ):
        """One streaming-update kernel: updates (m,d,o) for a block-range of tokens.

        This is intended for **decode (T==1)**, where we run online softmax.
        We fuse:
          - dequant (K_sem q4_0, K_geo q8_0, V q4_0)
          - logits computation
          - exp / sums
          - weighted value accumulation
        """
        pid = tl.program_id(0)  # 0 .. B*H-1
        b = pid // H
        h = pid - b * H

        # Load running state.
        m = tl.load(m_ptr + pid).to(tl.float32)
        d = tl.load(d_ptr + pid).to(tl.float32)
        dv = tl.arange(0, HD_V)
        o = tl.load(o_ptr + pid * stride_o + dv, mask=dv < HD_V, other=0.0).to(tl.float32)

        # Load query vectors.
        ds = tl.arange(0, HD_SEM)
        dg = tl.arange(0, HD_GEO)
        q_sem = tl.load(q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds, mask=ds < HD_SEM, other=0.0).to(tl.float32)
        q_geo = tl.load(q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg, mask=dg < HD_GEO, other=0.0).to(tl.float32)

        # Static loop: process NUM_SUBBLOCKS contiguous blocks of BLOCK_N tokens.
        for sb in tl.static_range(NUM_SUBBLOCKS):
            t = start + sb * BLOCK_N + tl.arange(0, BLOCK_N)
            tm = t < L_prefix

            # ---- semantic keys: q4_0 ----
            ksd = h * HD_SEM + ds  # global dim indices in [0, sem_dim)
            ks_byte = ksd // 2
            ks_nib = ksd % 2
            ks_ptr = k_sem_q_ptr + b * stride_ksq_b + t[:, None] * stride_ksq_t + ks_byte[None, :]
            ks_p = tl.load(ks_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0).to(tl.int32)
            ks_hi = ks_p >> 4
            ks_lo = ks_p & 0xF
            ks_u = tl.where(ks_nib[None, :] == 0, ks_hi, ks_lo)
            ks_q = ks_u.to(tl.int32) - 8
            ks_sb = ksd // QBLOCK_SEM
            ks_s_ptr = k_sem_s_ptr + b * stride_kss_b + t[:, None] * stride_kss_t + ks_sb[None, :] * stride_kss_c
            ks_s = tl.load(ks_s_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0.0).to(tl.float32)
            k_sem = ks_q.to(tl.float32) * ks_s
            logit_sem = tl.sum(k_sem * q_sem[None, :], axis=1)

            # ---- geometric keys: q8_0 ----
            kgd = h * HD_GEO + dg  # global dim indices in [0, geo_dim)
            kg_ptr = k_geo_q_ptr + b * stride_kgq_b + t[:, None] * stride_kgq_t + kgd[None, :]
            kg_q = tl.load(kg_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0).to(tl.int32)
            kg_sb = kgd // QBLOCK_GEO
            kg_s_ptr = k_geo_s_ptr + b * stride_kgs_b + t[:, None] * stride_kgs_t + kg_sb[None, :] * stride_kgs_c
            kg_s = tl.load(kg_s_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0.0).to(tl.float32)
            k_geo = kg_q.to(tl.float32) * kg_s
            logit_geo = tl.sum(k_geo * q_geo[None, :], axis=1)

            logits = logit_sem * SEM_SCALE + logit_geo * GEO_SCALE
            logits = tl.where(tm, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m, block_max)
            exp_m = tl.exp(m - m_new)
            exp_logits = tl.exp(logits - m_new)

            # ---- values: q4_0 ----
            vd = h * HD_V + dv
            v_byte = vd // 2
            v_nib = vd % 2
            v_ptr = v_q_ptr + b * stride_vq_b + t[:, None] * stride_vq_t + v_byte[None, :]
            v_p = tl.load(v_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0).to(tl.int32)
            v_hi = v_p >> 4
            v_lo = v_p & 0xF
            v_u = tl.where(v_nib[None, :] == 0, v_hi, v_lo)
            v_q = v_u.to(tl.int32) - 8
            v_sb = vd // QBLOCK_V
            vs_ptr = v_s_ptr + b * stride_vs_b + t[:, None] * stride_vs_t + v_sb[None, :] * stride_vs_c
            v_s = tl.load(vs_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0.0).to(tl.float32)
            v_val = v_q.to(tl.float32) * v_s

            d = d * exp_m + tl.sum(exp_logits, axis=0)
            # weighted sum over tokens -> (HD_V,)
            wv = tl.sum(exp_logits[:, None] * v_val, axis=0)
            o = o * exp_m + wv
            m = m_new

        tl.store(m_ptr + pid, m)
        tl.store(d_ptr + pid, d)
        tl.store(o_ptr + pid * stride_o + dv, o, mask=dv < HD_V)



    # -----------------------------
    # v24: 2-pass "FlashAttention-style" split-K decode kernels
    # -----------------------------
    #
    # Motivation:
    # - v23's fused kernel is "one program per (batch, head)" and loops over the sequence.
    #   Great for moderate context, but it can't parallelize over sequence length when T==1.
    # - This split-K design parallelizes across the KV sequence dimension by slicing it into partitions.
    #   Each partition computes local (m, d, o) using online-softmax, then we reduce partitions.
    #
    # This is decode-only (T==1), forward-only, and currently specialized for:
    #   K_sem: q4_0, K_geo: q8_0, V: q4_0  with qblock=32.

    @triton.jit
    def _kv_decode_partition_stats_decoupled_q4q8q4(
        q_sem_ptr, q_geo_ptr,
        k_sem_q_ptr, k_sem_s_ptr,
        k_geo_q_ptr, k_geo_s_ptr,
        v_q_ptr, v_s_ptr,
        m_part_ptr, d_part_ptr, o_part_ptr,
        L_prefix: tl.int32,
        P: tl.int32,
        PARTITION_SIZE: tl.constexpr,
        H: tl.constexpr,
        HD_SEM: tl.constexpr,
        HD_GEO: tl.constexpr,
        HD_V: tl.constexpr,
        QBLOCK_SEM: tl.constexpr,
        QBLOCK_GEO: tl.constexpr,
        QBLOCK_V: tl.constexpr,
        SEM_SCALE: tl.constexpr,
        GEO_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SUBBLOCKS: tl.constexpr,
        stride_qsem_b: tl.constexpr,
        stride_qsem_h: tl.constexpr,
        stride_qgeo_b: tl.constexpr,
        stride_qgeo_h: tl.constexpr,
        stride_ksq_b: tl.constexpr,
        stride_ksq_t: tl.constexpr,
        stride_kss_b: tl.constexpr,
        stride_kss_t: tl.constexpr,
        stride_kss_c: tl.constexpr,
        stride_kgq_b: tl.constexpr,
        stride_kgq_t: tl.constexpr,
        stride_kgs_b: tl.constexpr,
        stride_kgs_t: tl.constexpr,
        stride_kgs_c: tl.constexpr,
        stride_vq_b: tl.constexpr,
        stride_vq_t: tl.constexpr,
        stride_vs_b: tl.constexpr,
        stride_vs_t: tl.constexpr,
        stride_vs_c: tl.constexpr,
        stride_mp_row: tl.constexpr,
        stride_mp_part: tl.constexpr,
        stride_dp_row: tl.constexpr,
        stride_dp_part: tl.constexpr,
        stride_op_row: tl.constexpr,
        stride_op_part: tl.constexpr,
    ):
        pid_row = tl.program_id(0)  # 0 .. BH-1
        pid_part = tl.program_id(1) # 0 .. P-1 (grid)
        b = pid_row // H
        h = pid_row - b * H

        # Load query vectors (fp16) -> fp32.
        ds = tl.arange(0, HD_SEM)
        dg = tl.arange(0, HD_GEO)
        dv = tl.arange(0, HD_V)

        qsem = tl.load(q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds, mask=ds < HD_SEM, other=0.0).to(tl.float32)
        qgeo = tl.load(q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg, mask=dg < HD_GEO, other=0.0).to(tl.float32)

        # Local online-softmax state for this partition.
        m = -float("inf")
        d = 0.0
        o = tl.zeros([HD_V], dtype=tl.float32)

        start = pid_part * PARTITION_SIZE
        # Process PARTITION_SIZE tokens in NUM_SUBBLOCKS * BLOCK_N tiles.
        for sb in tl.static_range(0, NUM_SUBBLOCKS):
            t = start + sb * BLOCK_N + tl.arange(0, BLOCK_N)
            tm = t < L_prefix

            # --- Semantic K (q4_0) dot ---
            # Global dim index within merged (H*HD_SEM)
            ksd = h * HD_SEM + ds                       # (HD_SEM,)
            ks_byte = ksd // 2
            ks_nib = ksd % 2
            ks_ptr = k_sem_q_ptr + b * stride_ksq_b + t[:, None] * stride_ksq_t + ks_byte[None, :]
            ks_p = tl.load(ks_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0).to(tl.uint8)
            ks_hi = ks_p >> 4
            ks_lo = ks_p & 0xF
            ks_u = tl.where(ks_nib[None, :] == 0, ks_hi, ks_lo)
            ks_q = ks_u.to(tl.int32) - 8

            ks_sb = ksd // QBLOCK_SEM
            ks_s_ptr = k_sem_s_ptr + b * stride_kss_b + t[:, None] * stride_kss_t + ks_sb[None, :] * stride_kss_c
            ks_s = tl.load(ks_s_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0.0).to(tl.float32)
            ksem = ks_q.to(tl.float32) * ks_s
            sem = tl.sum(ksem * qsem[None, :], axis=1)

            # --- Geometric K (q8_0) dot ---
            kgd = h * HD_GEO + dg
            kg_ptr = k_geo_q_ptr + b * stride_kgq_b + t[:, None] * stride_kgq_t + kgd[None, :]
            kg_q = tl.load(kg_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0).to(tl.int8).to(tl.float32)

            kg_sb = kgd // QBLOCK_GEO
            kg_s_ptr = k_geo_s_ptr + b * stride_kgs_b + t[:, None] * stride_kgs_t + kg_sb[None, :] * stride_kgs_c
            kg_s = tl.load(kg_s_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0.0).to(tl.float32)
            kgeo = kg_q * kg_s
            geo = tl.sum(kgeo * qgeo[None, :], axis=1)

            logits = sem * SEM_SCALE + geo * GEO_SCALE
            logits = tl.where(tm, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m, block_max)
            exp_m = tl.exp(m - m_new)

            exp_logits = tl.exp(logits - m_new)

            # --- V (q4_0) weighted sum ---
            vd = h * HD_V + dv
            v_byte = vd // 2
            v_nib = vd % 2
            v_ptr = v_q_ptr + b * stride_vq_b + t[:, None] * stride_vq_t + v_byte[None, :]
            v_p = tl.load(v_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0).to(tl.uint8)
            v_hi = v_p >> 4
            v_lo = v_p & 0xF
            v_u = tl.where(v_nib[None, :] == 0, v_hi, v_lo)
            v_q = v_u.to(tl.int32) - 8

            v_sb = vd // QBLOCK_V
            vs_ptr = v_s_ptr + b * stride_vs_b + t[:, None] * stride_vs_t + v_sb[None, :] * stride_vs_c
            v_s = tl.load(vs_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0.0).to(tl.float32)
            v_val = v_q.to(tl.float32) * v_s

            d = d * exp_m + tl.sum(exp_logits, axis=0)
            wv = tl.sum(exp_logits[:, None] * v_val, axis=0)
            o = o * exp_m + wv
            m = m_new

        # Write partition stats
        tl.store(m_part_ptr + pid_row * stride_mp_row + pid_part * stride_mp_part, m)
        tl.store(d_part_ptr + pid_row * stride_dp_row + pid_part * stride_dp_part, d)
        tl.store(o_part_ptr + pid_row * stride_op_row + pid_part * stride_op_part + dv, o, mask=dv < HD_V)

    @triton.jit
    def _kv_decode_reduce_partitions(
        m_part_ptr, d_part_ptr, o_part_ptr,
        m_ptr, d_ptr, o_ptr,
        P: tl.int32,
        NUM_PARTS: tl.constexpr,
        HD_V: tl.constexpr,
        stride_mp_row: tl.constexpr,
        stride_mp_part: tl.constexpr,
        stride_dp_row: tl.constexpr,
        stride_dp_part: tl.constexpr,
        stride_op_row: tl.constexpr,
        stride_op_part: tl.constexpr,
        stride_o: tl.constexpr,
    ):
        pid_row = tl.program_id(0)  # 0..BH-1
        dv = tl.arange(0, HD_V)

        # First pass: global max over partitions.
        m = -float("inf")
        for p in tl.static_range(0, NUM_PARTS):
            p_i = tl.full([], p, tl.int32)
            pm = p_i < P
            mp = tl.load(m_part_ptr + pid_row * stride_mp_row + p_i * stride_mp_part, mask=pm, other=-float("inf"))
            m = tl.maximum(m, mp)

        # Second pass: combine denominators and outputs.
        d = 0.0
        o = tl.zeros([HD_V], dtype=tl.float32)
        for p in tl.static_range(0, NUM_PARTS):
            p_i = tl.full([], p, tl.int32)
            pm = p_i < P
            mp = tl.load(m_part_ptr + pid_row * stride_mp_row + p_i * stride_mp_part, mask=pm, other=-float("inf"))
            dp = tl.load(d_part_ptr + pid_row * stride_dp_row + p_i * stride_dp_part, mask=pm, other=0.0)
            op = tl.load(o_part_ptr + pid_row * stride_op_row + p_i * stride_op_part + dv, mask=pm & (dv < HD_V), other=0.0).to(tl.float32)

            scale = tl.where(pm, tl.exp(mp - m), 0.0)
            d += dp * scale
            o += op * scale

        tl.store(m_ptr + pid_row, m)
        tl.store(d_ptr + pid_row, d)
        tl.store(o_ptr + pid_row * stride_o + dv, o, mask=dv < HD_V)


def _triton_decoupled_q4q8q4_available() -> bool:
    return bool(TRITON_AVAILABLE)



class SeqCacheTensor:
    """
    A [B, max_seq_len, dim] sequence tensor stored in fp16/fp32/q8_0/q4_0/nf4.

    New in v22:
      - get_slice(start, end): dequantize only a slice (critical for long-context decode)
      - residual fp16 "hot" ring-buffer for the newest tokens (optional)
    """
    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        dim: int,
        cfg: KVCacheTensorConfig,
        device: torch.device,
    ):
        self.kind: KVCacheKind = cfg.kind
        self.device = device
        self.spec = make_quantspec(cfg.kind, dim, cfg.qblock)
        self.pos = 0
        self.max_seq_len = max_seq_len

        self.residual_len = int(max(0, cfg.residual_len))
        self._residual: Optional[torch.Tensor]
        self._residual_len_eff = min(self.residual_len, max_seq_len) if self.residual_len > 0 else 0

        if self.kind in ("fp16", "fp32"):
            dtype = torch.float16 if self.kind == "fp16" else torch.float32
            self.buf = torch.empty((batch_size, max_seq_len, dim), device=device, dtype=dtype)
            self.q = None
            self.s = None
            self._residual = None
        elif self.kind == "q8_0":
            self.buf = None
            self.q = torch.empty((batch_size, max_seq_len, self.spec.pad_dim), device=device, dtype=torch.int8)
            self.s = torch.empty((batch_size, max_seq_len, self.spec.n_blocks), device=device, dtype=torch.float16)
            self._residual = (
                torch.empty((batch_size, self._residual_len_eff, dim), device=device, dtype=torch.float16)
                if self._residual_len_eff > 0 else None
            )
        elif self.kind in ("q4_0", "nf4"):
            self.buf = None
            self.q = torch.empty((batch_size, max_seq_len, self.spec.pad_dim // 2), device=device, dtype=torch.uint8)
            self.s = torch.empty((batch_size, max_seq_len, self.spec.n_blocks), device=device, dtype=torch.float16)
            self._residual = (
                torch.empty((batch_size, self._residual_len_eff, dim), device=device, dtype=torch.float16)
                if self._residual_len_eff > 0 else None
            )
        else:
            raise ValueError(self.kind)

    @property
    def is_quantized(self) -> bool:
        return self.kind not in ("fp16", "fp32")

    def _residual_start(self) -> int:
        if self._residual is None:
            return self.pos  # empty range
        return max(0, self.pos - self._residual_len_eff)

    def _residual_gather(self, start: int, end: int) -> torch.Tensor:
        """
        Gather [start, end) from the fp16 residual ring buffer.
        Assumes the range is fully inside the residual window.
        """
        if self._residual is None:
            raise RuntimeError("No residual buffer allocated")
        if not (0 <= start <= end <= self.pos):
            raise ValueError(f"Invalid residual slice {start}:{end} for pos={self.pos}")
        rlen = self._residual_len_eff
        if rlen <= 0:
            raise RuntimeError("Residual length is 0")
        idx = (torch.arange(start, end, device=self.device, dtype=torch.long) % rlen)
        # (B, end-start, dim)
        return self._residual.index_select(1, idx)

    def append(self, x_new: torch.Tensor) -> int:
        """
        x_new: (B, T_new, dim) float
        returns old_pos (start index)
        """
        B, Tn, D = x_new.shape
        if D != self.spec.dim:
            raise ValueError(f"dim mismatch: expected {self.spec.dim}, got {D}")
        if self.pos + Tn > self.max_seq_len:
            raise ValueError(f"Cache overflow: pos {self.pos} + {Tn} > max {self.max_seq_len}")
        old = self.pos

        if self.kind in ("fp16", "fp32"):
            self.buf[:, old:old + Tn] = x_new.to(self.buf.dtype)
        elif self.kind == "q8_0":
            q, s = quantize_q8_0(x_new, self.spec)
            self.q[:, old:old + Tn] = q
            self.s[:, old:old + Tn] = s
        elif self.kind == "q4_0":
            q, s = quantize_q4_0(x_new, self.spec)
            self.q[:, old:old + Tn] = q
            self.s[:, old:old + Tn] = s
        elif self.kind == "nf4":
            q, s = quantize_nf4(x_new, self.spec)
            self.q[:, old:old + Tn] = q
            self.s[:, old:old + Tn] = s
        else:
            raise ValueError(self.kind)

        # maintain residual fp16 ring for hot tokens (helps decode; negligible memory).
        if self._residual is not None:
            rlen = self._residual_len_eff
            if rlen > 0:
                if Tn >= rlen:
                    # Only the newest rlen tokens matter for the ring.
                    x_tail = x_new[:, -rlen:].to(torch.float16)
                    idx = (torch.arange(old + Tn - rlen, old + Tn, device=self.device, dtype=torch.long) % rlen)
                    self._residual[:, idx] = x_tail
                else:
                    x_fp16 = x_new.to(torch.float16)
                    idx = (torch.arange(old, old + Tn, device=self.device, dtype=torch.long) % rlen)
                    self._residual[:, idx] = x_fp16

        self.pos += Tn
        return old

    def get_slice(self, start: int, end: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        returns (B, end-start, dim) in `dtype`, dequantizing only the requested slice.
        """
        start = int(start); end = int(end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid slice {start}:{end}")
        if end > self.pos:
            raise ValueError(f"Requested end {end} > cached length {self.pos}")
        if start == end:
            # preserve shape semantics
            B = (self.buf.size(0) if self.buf is not None else self.q.size(0))  # type: ignore[union-attr]
            return torch.empty((B, 0, self.spec.dim), device=self.device, dtype=dtype)

        if self.kind in ("fp16", "fp32"):
            return self.buf[:, start:end].to(dtype)  # type: ignore[index]

        # residual fast-path (newest tokens)
        r_start = self._residual_start()
        if self._residual is not None and start >= r_start:
            return self._residual_gather(start, end).to(dtype)

        # mixed slice: older part dequant + tail from residual
        if self._residual is not None and end > r_start and start < r_start:
            a = self.get_slice(start, r_start, dtype=dtype)
            b = self._residual_gather(r_start, end).to(dtype)
            return torch.cat([a, b], dim=1)

        # fully in the quantized region
        if self.kind == "q8_0":
            x = dequantize_q8_0(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        if self.kind == "q4_0":
            x = dequantize_q4_0(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        if self.kind == "nf4":
            x = dequantize_nf4(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        raise ValueError(self.kind)

    def get(self, length: Optional[int] = None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        returns (B, length, dim) in `dtype` (compatibility helper).
        """
        L = self.pos if length is None else int(length)
        return self.get_slice(0, L, dtype=dtype)


class LayerKVCache:
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
    ):
        self.k = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_dim, cfg=k_cfg, device=device)
        self.v = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, cfg=v_cfg, device=device)

    @property
    def pos(self) -> int:
        return self.k.pos

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        old = self.k.append(k_new)
        old2 = self.v.append(v_new)
        if old != old2:
            raise RuntimeError("K/V cache desync")
        return old

    def get(self, *, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k.get(dtype=dtype), self.v.get(dtype=dtype)


class DecoupledLayerKVCache:
    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        k_sem_dim: int,
        k_geo_dim: int,
        v_dim: int,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ):
        self.k_sem = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_sem_dim, cfg=k_sem_cfg, device=device)
        self.k_geo = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_geo_dim, cfg=k_geo_cfg, device=device)
        self.v = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, cfg=v_cfg, device=device)

    @property
    def pos(self) -> int:
        return self.k_sem.pos

    def append(self, k_sem_new: torch.Tensor, k_geo_new: torch.Tensor, v_new: torch.Tensor) -> int:
        old = self.k_sem.append(k_sem_new)
        old2 = self.k_geo.append(k_geo_new)
        old3 = self.v.append(v_new)
        if not (old == old2 == old3):
            raise RuntimeError("Decoupled cache desync")
        return old

    def get(self, *, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.k_sem.get(dtype=dtype), self.k_geo.get(dtype=dtype), self.v.get(dtype=dtype)


# -----------------------------
# Model
# -----------------------------

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
    attn_dim: int = 512    # total V dim (and Q/K dim for bottleneck)
    sem_dim: int = 32      # total semantic Q/K dim across heads (decoupled)
    geo_dim: int = 64      # total geometric Q/K dim across heads (decoupled)

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


class DecoupledBottleneckAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.n_head
        self.H = H
        self.H_kv = H
        self.group_size = 1
        self.drop = nn.Dropout(cfg.dropout)

        def must_div(name: str, total: int, denom: int) -> int:
            if total % denom != 0:
                raise ValueError(f"{name} ({total}) must be divisible by {denom}")
            return total // denom

        if cfg.attn_mode == "standard":
            qk_dim = cfg.d_model
            v_dim = cfg.d_model
            self.qk_head_dim = must_div("d_model", qk_dim, H)
            self.v_head_dim = must_div("d_model", v_dim, H)

            self.q_proj = nn.Linear(cfg.d_model, qk_dim, bias=False)
            self.k_proj = self.q_proj if cfg.tie_qk else nn.Linear(cfg.d_model, qk_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, v_dim, bias=False)
            self.out_proj = nn.Linear(v_dim, cfg.d_model, bias=False)

            self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
            self.sem_head_dim = self.geo_head_dim = None
            self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_null = nn.Parameter(torch.zeros(1, 1, qk_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, v_dim)) if cfg.null_attn else None
            self.k_sem_null = self.k_geo_null = None

        elif cfg.attn_mode == "bottleneck":
            qk_dim = cfg.attn_dim
            v_dim = cfg.attn_dim
            self.qk_head_dim = must_div("attn_dim", qk_dim, H)
            self.v_head_dim = must_div("attn_dim", v_dim, H)
            if cfg.rope and (self.qk_head_dim % 2 != 0):
                raise ValueError("RoPE needs even head dim; pick attn_dim divisible by 2*n_head")

            self.q_proj = nn.Linear(cfg.d_model, qk_dim, bias=False)
            self.k_proj = self.q_proj if cfg.tie_qk else nn.Linear(cfg.d_model, qk_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, v_dim, bias=False)
            self.out_proj = nn.Linear(v_dim, cfg.d_model, bias=False)

            self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
            self.sem_head_dim = self.geo_head_dim = None
            self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_null = nn.Parameter(torch.zeros(1, 1, qk_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, v_dim)) if cfg.null_attn else None
            self.k_sem_null = self.k_geo_null = None

        elif cfg.attn_mode == "gqa":
            # Grouped-Query Attention (GQA): Q has H heads, K/V has H_kv heads shared across groups.
            kv_head = cfg.kv_head if cfg.kv_head is not None else H
            if kv_head <= 0:
                raise ValueError("kv_head must be > 0")
            if H % kv_head != 0:
                raise ValueError(f"gqa requires n_head % kv_head == 0 (got n_head={H}, kv_head={kv_head})")
            self.H_kv = kv_head
            self.group_size = H // kv_head

            self.qk_head_dim = must_div("attn_dim", cfg.attn_dim, H)
            self.v_head_dim = self.qk_head_dim
            kv_dim = kv_head * self.qk_head_dim

            if cfg.rope and (self.qk_head_dim % 2 != 0):
                raise ValueError("RoPE requires an even head dim. Choose attn_dim divisible by 2*n_head.")

            if cfg.tie_qk:
                raise ValueError("tie_qk is not supported for gqa unless kv_head == n_head (use --attn-mode standard).")

            self.q_proj = nn.Linear(cfg.d_model, cfg.attn_dim, bias=False)
            self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
            self.out_proj = nn.Linear(cfg.attn_dim, cfg.d_model, bias=False)

            self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
            self.sem_head_dim = self.geo_head_dim = None
            self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_null = nn.Parameter(torch.zeros(1, 1, kv_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, kv_dim)) if cfg.null_attn else None

            self.k_sem_null = self.k_geo_null = None

        elif cfg.attn_mode == "decoupled":
            self.sem_head_dim = must_div("sem_dim", cfg.sem_dim, H)
            self.geo_head_dim = must_div("geo_dim", cfg.geo_dim, H)
            self.v_head_dim = must_div("attn_dim", cfg.attn_dim, H)
            if cfg.rope and (self.geo_head_dim % 2 != 0):
                raise ValueError("RoPE needs even geo_head_dim; pick geo_dim divisible by 2*n_head")

            self.q_sem = nn.Linear(cfg.d_model, cfg.sem_dim, bias=False)
            self.k_sem = self.q_sem if cfg.tie_qk else nn.Linear(cfg.d_model, cfg.sem_dim, bias=False)
            self.q_geo = nn.Linear(cfg.d_model, cfg.geo_dim, bias=False)
            self.k_geo = self.q_geo if cfg.tie_qk else nn.Linear(cfg.d_model, cfg.geo_dim, bias=False)

            self.v_proj = nn.Linear(cfg.d_model, cfg.attn_dim, bias=False)
            self.out_proj = nn.Linear(cfg.attn_dim, cfg.d_model, bias=False)

            self.q_proj = self.k_proj = None
            self.qk_head_dim = None
            self.rotary = RotaryEmbedding(self.geo_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_sem_null = nn.Parameter(torch.zeros(1, 1, cfg.sem_dim)) if cfg.null_attn else None
            self.k_geo_null = nn.Parameter(torch.zeros(1, 1, cfg.geo_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, cfg.attn_dim)) if cfg.null_attn else None
            self.k_null = None
        else:
            raise ValueError(cfg.attn_mode)

        # learned per-head temperature (applied as a multiplicative scale on logits)
        self.logit_scale = nn.Parameter(torch.zeros(H)) if cfg.learned_temp else None
        # Scratch buffers for Triton 2-pass (split-K) decode. Allocated lazily on CUDA.
        self._flash2_scratch = None  # type: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        self._flash2_scratch_cap = (0, 0, 0)  # (BH, parts_cap, hd_v)


    def _shape(self, x: torch.Tensor, head_dim: int, H: Optional[int] = None) -> torch.Tensor:
        # (B,T,H*hd)->(B,H,T,hd)
        B, T, D = x.shape
        H = self.H if H is None else H
        return x.view(B, T, H, head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,T,hd)->(B,T,H*hd)
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def _apply_logit_scale_to_q(self, q: torch.Tensor) -> torch.Tensor:
        # scores = (q @ k^T) * exp(logit_scale)
        if self.logit_scale is None:
            return q
        return q * torch.exp(self.logit_scale.view(1, -1, 1, 1))

    def _sdp(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Use PyTorch's fused SDPA when available (Flash / mem-efficient attention).
        # - If attn_mask is None, we can use is_causal=True and let the kernel handle causality.
        # - If attn_mask is provided (e.g., chunked prefill with KV-cache), we pass it explicitly.
        dropout_p = self.cfg.dropout if self.training else 0.0
        if attn_mask is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)

    def _streaming_decode_attn(
        self,
        *,
        q: torch.Tensor,          # (B,H,1,hd_qk)
        k_cache: SeqCacheTensor,  # stores (B,L,H*hd_qk) merged
        v_cache: SeqCacheTensor,  # stores (B,L,H*hd_v)  merged
        head_dim: int,
        decode_block: int,
        scale: float,
        v_head_dim: Optional[int] = None,
        k_null: Optional[torch.Tensor] = None,   # (B,H,1,hd_qk) or None
        v_null: Optional[torch.Tensor] = None,   # (B,H,1,hd_v)  or None
    ) -> torch.Tensor:
        """
        Streaming attention for decode (T==1): computes softmax(qK^T)V without materializing the full score vector.

        Returns: (B,H,1,hd_v)
        """
        B, H, Tq, hd = q.shape
        assert Tq == 1
        if v_head_dim is None:
            v_head_dim = head_dim

        L = k_cache.pos
        if L != v_cache.pos:
            raise RuntimeError("K/V cache desync in streaming decode")

        # We'll compute in fp32 for the running softmax state, but use fp16/bf16 matmuls where possible.
        compute_dtype = torch.float16 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

        m = torch.full((B, H, 1), -float("inf"), device=q.device, dtype=torch.float32)
        d = torch.zeros((B, H, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, 1, v_head_dim), device=q.device, dtype=torch.float32)

        qh = q.to(compute_dtype)

        def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
            nonlocal m, d, o
            # scores_f32: (B,H,1,Bl)
            block_max = scores_f32.amax(dim=-1)  # (B,H,1)
            m_new = torch.maximum(m, block_max)
            exp_m = torch.exp(m - m_new)  # (B,H,1)

            exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1))  # (B,H,1,Bl)
            exp_scores_f16 = exp_scores.to(compute_dtype)

            d = d * exp_m + exp_scores_f16.sum(dim=-1).to(torch.float32)  # (B,H,1)

            # (B,H,1,Bl) @ (B,H,Bl,hd) -> (B,H,1,hd)
            o = o * exp_m.unsqueeze(-1) + torch.matmul(exp_scores_f16, v_block_f16).to(torch.float32)
            m = m_new

        # Optional null token (one extra key/value)
        if k_null is not None and v_null is not None:
            # scores: (B,H,1,1)
            s = (qh * k_null.to(compute_dtype)).sum(dim=-1, keepdim=True).to(torch.float32) * scale
            update(s, v_null.to(compute_dtype))

        # Stream over cached sequence in blocks.
        blk = int(max(1, decode_block))
        for start in range(0, L, blk):
            end = min(L, start + blk)
            k_blk = k_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H*hd)
            v_blk = v_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H*hdv)
            kbh = self._shape(k_blk, head_dim)                          # (B,H,Bl,hd)
            vbh = self._shape(v_blk, v_head_dim)                        # (B,H,Bl,hdv)

            # (B,H,1,hd) @ (B,H,hd,Bl) -> (B,H,1,Bl)
            scores = torch.matmul(qh, kbh.transpose(-2, -1)) * scale
            update(scores.to(torch.float32), vbh)

        out = o / d.clamp(min=1e-9).unsqueeze(-1)
        return out.to(q.dtype)

    def _streaming_decode_attn_decoupled(
        self,
        *,
        q_sem: torch.Tensor,          # (B,H,1,hd_sem)
        q_geo: torch.Tensor,          # (B,H,1,hd_geo)
        k_sem_cache: SeqCacheTensor,  # stores (B,L,H*hd_sem) merged
        k_geo_cache: SeqCacheTensor,  # stores (B,L,H*hd_geo) merged
        v_cache: SeqCacheTensor,      # stores (B,L,H*hd_v) merged
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
        k_sem_null: Optional[torch.Tensor] = None,  # (B,H,1,hd_sem)
        k_geo_null: Optional[torch.Tensor] = None,  # (B,H,1,hd_geo)
        v_null: Optional[torch.Tensor] = None,      # (B,H,1,hd_v)
    ) -> torch.Tensor:
        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = k_sem_cache.pos
        if not (L == k_geo_cache.pos == v_cache.pos):
            raise RuntimeError("Decoupled cache desync in streaming decode")

        compute_dtype = torch.float16 if q_sem.dtype in (torch.float16, torch.bfloat16) else q_sem.dtype

        m = torch.full((B, H, 1), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((B, H, 1), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((B, H, 1, v_head_dim), device=q_sem.device, dtype=torch.float32)

        qsh = q_sem.to(compute_dtype)
        qgh = q_geo.to(compute_dtype)

        def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
            nonlocal m, d, o
            block_max = scores_f32.amax(dim=-1)
            m_new = torch.maximum(m, block_max)
            exp_m = torch.exp(m - m_new)

            exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1))
            exp_scores_f16 = exp_scores.to(compute_dtype)

            d = d * exp_m + exp_scores_f16.sum(dim=-1).to(torch.float32)
            o = o * exp_m.unsqueeze(-1) + torch.matmul(exp_scores_f16, v_block_f16).to(torch.float32)
            m = m_new

        # Optional null token.
        if k_sem_null is not None and k_geo_null is not None and v_null is not None:
            s = (
                (qsh * k_sem_null.to(compute_dtype)).sum(dim=-1, keepdim=True).to(torch.float32) * sem_scale
                + (qgh * k_geo_null.to(compute_dtype)).sum(dim=-1, keepdim=True).to(torch.float32) * geo_scale
            )
            update(s, v_null.to(compute_dtype))

        blk = int(max(1, decode_block))
        for start in range(0, L, blk):
            end = min(L, start + blk)
            k_sem_blk = k_sem_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H*hd_sem)
            k_geo_blk = k_geo_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H*hd_geo)
            v_blk = v_cache.get_slice(start, end, dtype=compute_dtype)          # (B,Bl,H*hd_v)

            ksh = self._shape(k_sem_blk, sem_head_dim)  # (B,H,Bl,hd_sem)
            kgh = self._shape(k_geo_blk, geo_head_dim)  # (B,H,Bl,hd_geo)
            vbh = self._shape(v_blk, v_head_dim)        # (B,H,Bl,hd_v)

            s = (
                torch.matmul(qsh, ksh.transpose(-2, -1)).to(torch.float32) * sem_scale
                + torch.matmul(qgh, kgh.transpose(-2, -1)).to(torch.float32) * geo_scale
            )
            update(s, vbh)

        out = o / d.clamp(min=1e-9).unsqueeze(-1)
        return out.to(q_sem.dtype)

    def _fused_decode_attn_decoupled_q4q8q4(
        self,
        *,
        q_sem: torch.Tensor,          # (B,H,1,hd_sem)
        q_geo: torch.Tensor,          # (B,H,1,hd_geo)
        cache: "DecoupledLayerKVCache",
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
    ) -> torch.Tensor:
        """Decode-only fused path (T==1) for the common decoupled quant policy:

          K_sem: q4_0  (merged dim = sem_dim)
          K_geo: q8_0  (merged dim = geo_dim)
          V:     q4_0  (merged dim = attn_dim)

        Uses a Triton kernel (if installed) to fuse dequant + online-softmax update per block.
        Falls back to Python streaming if Triton is unavailable.
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        if q_sem.device.type != "cuda":
            raise RuntimeError("Fused decode requires CUDA")
        if self.cfg.null_attn:
            raise RuntimeError("Fused decode currently assumes null_attn=False")

        # Enforce the expected cache layout.
        if not (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0"):
            raise RuntimeError("Fused decode q4q8q4 requires k_sem=q4_0, k_geo=q8_0, v=q4_0")
        if not (cache.k_sem.spec.qblock == cache.k_geo.spec.qblock == cache.v.spec.qblock == 32):
            raise RuntimeError("Fused decode currently assumes qblock=32")

        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = cache.pos

        # Split into a big quantized prefix and an fp16 residual tail (processed in PyTorch).
        rlen = cache.k_sem._residual_len_eff if cache.k_sem._residual is not None else 0
        r_start = max(0, L - rlen) if rlen > 0 else L
        L_prefix = int(r_start)

        # Triton expects contiguous (B,H,hd) query tensors.
        q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
        q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

        # Running state in fp32.
        BH = B * H
        m = torch.full((BH,), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((BH,), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

        if L_prefix > 0:
            # Kernel tiling + launch params. v25 lets these be tuned at runtime via cache attributes.
            block_n = int(getattr(cache, "block_n", 128))
            if block_n <= 0:
                block_n = 128
            num_sub = max(1, int(decode_block // block_n))
            step = block_n * num_sub
            num_warps = int(getattr(cache, "num_warps_1pass", 4))
            num_stages = int(getattr(cache, "num_stages_1pass", 2))

            # Shorthand tensors.
            ksq = cache.k_sem.q
            kss = cache.k_sem.s
            kgq = cache.k_geo.q
            kgs = cache.k_geo.s
            vq = cache.v.q
            vs = cache.v.s
            assert ksq is not None and kss is not None and kgq is not None and kgs is not None and vq is not None and vs is not None

            grid = (BH,)
            for start in range(0, L_prefix, step):
                _kv_decode_update_decoupled_q4q8q4[grid](
                    q_sem2,
                    q_geo2,
                    ksq,
                    kss,
                    kgq,
                    kgs,
                    vq,
                    vs,
                    m,
                    d,
                    o,
                    start,
                    L_prefix,
                    H=H,
                    HD_SEM=sem_head_dim,
                    HD_GEO=geo_head_dim,
                    HD_V=v_head_dim,
                    QBLOCK_SEM=32,
                    QBLOCK_GEO=32,
                    QBLOCK_V=32,
                    SEM_SCALE=sem_scale,
                    GEO_SCALE=geo_scale,
                    BLOCK_N=block_n,
                    NUM_SUBBLOCKS=num_sub,
                    stride_qsem_b=q_sem2.stride(0),
                    stride_qsem_h=q_sem2.stride(1),
                    stride_qgeo_b=q_geo2.stride(0),
                    stride_qgeo_h=q_geo2.stride(1),
                    stride_ksq_b=ksq.stride(0),
                    stride_ksq_t=ksq.stride(1),
                    stride_kss_b=kss.stride(0),
                    stride_kss_t=kss.stride(1),
                    stride_kss_c=kss.stride(2),
                    stride_kgq_b=kgq.stride(0),
                    stride_kgq_t=kgq.stride(1),
                    stride_kgs_b=kgs.stride(0),
                    stride_kgs_t=kgs.stride(1),
                    stride_kgs_c=kgs.stride(2),
                    stride_vq_b=vq.stride(0),
                    stride_vq_t=vq.stride(1),
                    stride_vs_b=vs.stride(0),
                    stride_vs_t=vs.stride(1),
                    stride_vs_c=vs.stride(2),
                    stride_o=o.stride(0),
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

        # Process residual tail in PyTorch (small; uses fp16 residual ring via get_slice).
        if L_prefix < L:
            qsh = q_sem.to(torch.float16)
            qgh = q_geo.to(torch.float16)

            # reshape state to (B,H,1) / (B,H,1,hd)
            m_t = m.view(B, H, 1)
            d_t = d.view(B, H, 1)
            o_t = o.view(B, H, 1, v_head_dim)

            def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
                nonlocal m_t, d_t, o_t
                block_max = scores_f32.amax(dim=-1)
                m_new = torch.maximum(m_t, block_max)
                exp_m = torch.exp(m_t - m_new)
                exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1)).to(torch.float16)
                d_t = d_t * exp_m + exp_scores.sum(dim=-1).to(torch.float32)
                o_t = o_t * exp_m.unsqueeze(-1) + torch.matmul(exp_scores, v_block_f16).to(torch.float32)
                m_t = m_new

            # one or a few blocks only
            k_sem_blk = cache.k_sem.get_slice(L_prefix, L, dtype=torch.float16)
            k_geo_blk = cache.k_geo.get_slice(L_prefix, L, dtype=torch.float16)
            v_blk = cache.v.get_slice(L_prefix, L, dtype=torch.float16)
            ksh = self._shape(k_sem_blk, sem_head_dim)
            kgh = self._shape(k_geo_blk, geo_head_dim)
            vbh = self._shape(v_blk, v_head_dim)
            s = (
                torch.matmul(qsh, ksh.transpose(-2, -1)).to(torch.float32) * sem_scale
                + torch.matmul(qgh, kgh.transpose(-2, -1)).to(torch.float32) * geo_scale
            )
            update(s, vbh)

            # flatten back
            m = m_t.view(BH)
            d = d_t.view(BH)
            o = o_t.view(BH, v_head_dim)

        out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
        return out.to(q_sem.dtype)


    def _fused_decode_attn_decoupled_q4q8q4_2pass(
        self,
        *,
        q_sem: torch.Tensor,          # (B,H,1,hd_sem)
        q_geo: torch.Tensor,          # (B,H,1,hd_geo)
        cache: "DecoupledLayerKVCache",
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
    ) -> torch.Tensor:
        """Decode-only fused path (T==1) using a 2-pass split-K ("FlashAttention-style") kernel.

        Pass 1 computes local (m, d, o) for each partition of the KV sequence in parallel.
        Pass 2 reduces partitions into a single (m, d, o) for the row.
        A tiny fp16 residual tail (hot window) is then folded in via the Python streaming updater.

        Specializes to the common decoupled heterogeneous quant policy:
          K_sem: q4_0  (merged dim = sem_dim)
          K_geo: q8_0  (merged dim = geo_dim)
          V:     q4_0  (merged dim = attn_dim)
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        if q_sem.device.type != "cuda":
            raise RuntimeError("Fused decode requires CUDA")
        if self.cfg.null_attn:
            raise RuntimeError("Fused decode currently assumes null_attn=False")

        # Enforce the expected cache layout.
        if not (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0"):
            raise RuntimeError("Fused decode q4q8q4 requires k_sem=q4_0, k_geo=q8_0, v=q4_0")
        if not (cache.k_sem.spec.qblock == cache.k_geo.spec.qblock == cache.v.spec.qblock == 32):
            raise RuntimeError("Fused decode currently assumes qblock=32")

        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = cache.pos

        # Split into a big quantized prefix and an fp16 residual tail (processed in PyTorch).
        rlen = cache.k_sem._residual_len_eff if cache.k_sem._residual is not None else 0
        r_start = max(0, L - rlen) if rlen > 0 else L
        L_prefix = int(r_start)

        # Queries: contiguous (B,H,hd)
        q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
        q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

        BH = B * H

        # Running state in fp32 (flattened).
        m = torch.full((BH,), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((BH,), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

        if L_prefix > 0:
            # Partition sizing:
            # - decode_block is the user-facing knob. We round it up to a multiple of BLOCK_N.
            # - v25 lets BLOCK_N + launch params be tuned at runtime via cache attributes.
            block_n = int(getattr(cache, "block_n", 128))
            if block_n <= 0:
                block_n = 128
            num_warps_part = int(getattr(cache, "num_warps_part", 4))
            num_stages_part = int(getattr(cache, "num_stages_part", 2))
            num_warps_reduce = int(getattr(cache, "num_warps_reduce", 1))
            num_stages_reduce = int(getattr(cache, "num_stages_reduce", 1))
            part_size = int(max(block_n, decode_block))
            if part_size % block_n != 0:
                part_size = ((part_size + block_n - 1) // block_n) * block_n
            num_sub = part_size // block_n

            P = int((L_prefix + part_size - 1) // part_size)

            # Allocate/reuse scratch buffers (BH, P_cap).
            # We grow capacity to the next power-of-two to reduce realloc+recompile churn.
            P_cap = 1 << (int(P - 1).bit_length())
            cap_BH, cap_P, cap_V = self._flash2_scratch_cap
            if (self._flash2_scratch is None) or (cap_BH < BH) or (cap_P < P_cap) or (cap_V != v_head_dim):
                m_part = torch.empty((BH, P_cap), device=q_sem.device, dtype=torch.float32)
                d_part = torch.empty((BH, P_cap), device=q_sem.device, dtype=torch.float32)
                o_part = torch.empty((BH, P_cap, v_head_dim), device=q_sem.device, dtype=torch.float32)
                self._flash2_scratch = (m_part, d_part, o_part)
                self._flash2_scratch_cap = (BH, P_cap, v_head_dim)
            else:
                m_part, d_part, o_part = self._flash2_scratch

            # Shorthand quant tensors.
            ksq = cache.k_sem.q
            kss = cache.k_sem.s
            kgq = cache.k_geo.q
            kgs = cache.k_geo.s
            vq = cache.v.q
            vs = cache.v.s
            assert ksq is not None and kss is not None and kgq is not None and kgs is not None and vq is not None and vs is not None

            # Pass 1: partition stats
            grid1 = (BH, P)
            _kv_decode_partition_stats_decoupled_q4q8q4[grid1](
                q_sem2,
                q_geo2,
                ksq,
                kss,
                kgq,
                kgs,
                vq,
                vs,
                m_part,
                d_part,
                o_part,
                L_prefix,
                P,
                PARTITION_SIZE=part_size,
                H=H,
                HD_SEM=sem_head_dim,
                HD_GEO=geo_head_dim,
                HD_V=v_head_dim,
                QBLOCK_SEM=32,
                QBLOCK_GEO=32,
                QBLOCK_V=32,
                SEM_SCALE=sem_scale,
                GEO_SCALE=geo_scale,
                BLOCK_N=block_n,
                NUM_SUBBLOCKS=num_sub,
                stride_qsem_b=q_sem2.stride(0),
                stride_qsem_h=q_sem2.stride(1),
                stride_qgeo_b=q_geo2.stride(0),
                stride_qgeo_h=q_geo2.stride(1),
                stride_ksq_b=ksq.stride(0),
                stride_ksq_t=ksq.stride(1),
                stride_kss_b=kss.stride(0),
                stride_kss_t=kss.stride(1),
                stride_kss_c=kss.stride(2),
                stride_kgq_b=kgq.stride(0),
                stride_kgq_t=kgq.stride(1),
                stride_kgs_b=kgs.stride(0),
                stride_kgs_t=kgs.stride(1),
                stride_kgs_c=kgs.stride(2),
                stride_vq_b=vq.stride(0),
                stride_vq_t=vq.stride(1),
                stride_vs_b=vs.stride(0),
                stride_vs_t=vs.stride(1),
                stride_vs_c=vs.stride(2),
                stride_mp_row=m_part.stride(0),
                stride_mp_part=m_part.stride(1),
                stride_dp_row=d_part.stride(0),
                stride_dp_part=d_part.stride(1),
                stride_op_row=o_part.stride(0),
                stride_op_part=o_part.stride(1),
                num_warps=num_warps_part,
                num_stages=num_stages_part,
            )

            # Pass 2: reduce partitions -> (m, d, o) for the prefix.
            grid2 = (BH,)
            _kv_decode_reduce_partitions[grid2](
                m_part,
                d_part,
                o_part,
                m,
                d,
                o,
                P,
                NUM_PARTS=P_cap,
                HD_V=v_head_dim,
                stride_mp_row=m_part.stride(0),
                stride_mp_part=m_part.stride(1),
                stride_dp_row=d_part.stride(0),
                stride_dp_part=d_part.stride(1),
                stride_op_row=o_part.stride(0),
                stride_op_part=o_part.stride(1),
                stride_o=o.stride(0),
                num_warps=num_warps_reduce,
                num_stages=num_stages_reduce,
            )

        # Process residual tail in PyTorch (small; uses fp16 residual ring via get_slice).
        if L_prefix < L:
            qsh = q_sem.to(torch.float16)
            qgh = q_geo.to(torch.float16)

            # reshape state to (B,H,1) / (B,H,1,hd)
            m_t = m.view(B, H, 1)
            d_t = d.view(B, H, 1)
            o_t = o.view(B, H, 1, v_head_dim)

            def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
                nonlocal m_t, d_t, o_t
                block_max = scores_f32.amax(dim=-1)
                m_new = torch.maximum(m_t, block_max)
                exp_m = torch.exp(m_t - m_new)
                exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1)).to(torch.float16)
                d_t = d_t * exp_m + exp_scores.sum(dim=-1).to(torch.float32)
                o_t = o_t * exp_m.unsqueeze(-1) + torch.matmul(exp_scores, v_block_f16).to(torch.float32)
                m_t = m_new

            k_sem_blk = cache.k_sem.get_slice(L_prefix, L, dtype=torch.float16)
            k_geo_blk = cache.k_geo.get_slice(L_prefix, L, dtype=torch.float16)
            v_blk = cache.v.get_slice(L_prefix, L, dtype=torch.float16)
            ksh = self._shape(k_sem_blk, sem_head_dim)
            kgh = self._shape(k_geo_blk, geo_head_dim)
            vbh = self._shape(v_blk, v_head_dim)
            s = (
                torch.matmul(qsh, ksh.transpose(-2, -1)).to(torch.float32) * sem_scale
                + torch.matmul(qgh, kgh.transpose(-2, -1)).to(torch.float32) * geo_scale
            )
            update(s, vbh)

            m = m_t.view(BH)
            d = d_t.view(BH)
            o = o_t.view(BH, v_head_dim)

        out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
        return out.to(q_sem.dtype)


    def _streaming_decode_attn_gqa(
        self,
        *,
        q: torch.Tensor,          # (B,H,1,hd)
        k_cache: SeqCacheTensor,  # stores (B,L,H_kv*hd)
        v_cache: SeqCacheTensor,  # stores (B,L,H_kv*hd)
        head_dim: int,
        decode_block: int,
        scale: float,
        k_null: Optional[torch.Tensor] = None,  # (B,H_kv,1,hd) or None
        v_null: Optional[torch.Tensor] = None,  # (B,H_kv,1,hd) or None
    ) -> torch.Tensor:
        """
        Streaming decode for GQA without expanding KV heads to Q heads.
        Returns: (B,H,1,hd)
        """
        B, H, Tq, hd = q.shape
        assert Tq == 1
        H_kv = self.H_kv
        g = self.group_size
        if H != H_kv * g:
            raise RuntimeError("Invalid GQA head geometry")

        L = k_cache.pos
        if L != v_cache.pos:
            raise RuntimeError("K/V cache desync in streaming decode (gqa)")

        compute_dtype = torch.float16 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

        m = torch.full((B, H, 1), -float("inf"), device=q.device, dtype=torch.float32)
        d = torch.zeros((B, H, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, 1, hd), device=q.device, dtype=torch.float32)

        qh = q.to(compute_dtype)
        # reshape Q into KV groups: (B,H_kv,g,1,hd)
        qg = qh.view(B, H_kv, g, 1, hd)

        def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
            nonlocal m, d, o
            # scores_f32: (B,H,1,Bl)
            block_max = scores_f32.amax(dim=-1)
            m_new = torch.maximum(m, block_max)
            exp_m = torch.exp(m - m_new)

            exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1))
            exp_scores_f16 = exp_scores.to(compute_dtype)

            d = d * exp_m + exp_scores_f16.sum(dim=-1).to(torch.float32)

            # matmul in groups:
            # exp_scores: (B,H,1,Bl) -> (B,H_kv,g,1,Bl)
            es = exp_scores_f16.view(B, H_kv, g, 1, -1)
            # v_block_f16: (B,H_kv,Bl,hd) -> (B,H_kv,1,Bl,hd)
            vb = v_block_f16.unsqueeze(2)  # (B,H_kv,1,Bl,hd)
            # (B,H_kv,g,1,Bl) @ (B,H_kv,1,Bl,hd) -> (B,H_kv,g,1,hd)
            out_blk = torch.matmul(es, vb).to(torch.float32)
            o = o * exp_m.unsqueeze(-1) + out_blk.view(B, H, 1, hd)
            m = m_new

        # Optional null token (KV-head count).
        if k_null is not None and v_null is not None:
            # Expand to query heads logically by grouping.
            # scores: (B,H_kv,g,1,1) -> view (B,H,1,1)
            s = (qg * k_null.to(compute_dtype).unsqueeze(2)).sum(dim=-1, keepdim=True).to(torch.float32) * scale
            update(s.view(B, H, 1, 1), v_null.to(compute_dtype).unsqueeze(2).view(B, H, 1, hd))

        blk = int(max(1, decode_block))
        for start in range(0, L, blk):
            end = min(L, start + blk)
            k_blk = k_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H_kv*hd)
            v_blk = v_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H_kv*hd)
            kbh = self._shape(k_blk, head_dim, H=H_kv)                   # (B,H_kv,Bl,hd)
            vbh = self._shape(v_blk, head_dim, H=H_kv)                   # (B,H_kv,Bl,hd)

            # scores per kv head/group: (B,H_kv,g,1,Bl)
            s = torch.matmul(qg, kbh.unsqueeze(2).transpose(-2, -1)) * scale
            update(s.view(B, H, 1, -1).to(torch.float32), vbh)

        out = o / d.clamp(min=1e-9).unsqueeze(-1)
        return out.to(q.dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        cache: Optional[Any],
        pos_offset: int,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        x: (B,T,d_model)
        attn_mask:
          - training: causal mask (B,1,T,T) bool, or None when using SDPA causal.
          - cached prefill: mask (B,1,T,L) bool to enforce causality inside the new chunk.
        cache:
          - None (training)
          - LayerKVCache (standard/bottleneck/gqa)
          - DecoupledLayerKVCache (decoupled)
        pos_offset: absolute position for RoPE for x[:,0]
        """
        cfg = self.cfg
        B, T, _ = x.shape
        ninfty = neg_inf(x.dtype)

        # -----------------------------
        # standard / bottleneck
        # -----------------------------
        if cfg.attn_mode in ("standard", "bottleneck"):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            qh = self._shape(q, self.qk_head_dim)
            kh = self._shape(k, self.qk_head_dim)
            vh = self._shape(v, self.v_head_dim)

            if self.rotary is not None:
                qh = self.rotary.rotate(qh, pos_offset)
                kh = self.rotary.rotate(kh, pos_offset)

            qh = self._apply_logit_scale_to_q(qh)

            # ---- training / full-attn ----
            if cache is None:
                if not cfg.null_attn:
                    # Prefer SDPA/Flash when possible.
                    out = self._sdp(qh, kh, vh, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    return y, None

                # Null-attn path (manual).
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                return y, None

            # ---- KV-cache path ----
            old_len = cache.pos

            # Fast prefill when the cache is empty: compute attention from local K/V, then append.
            if old_len == 0 and T > 1:
                if not cfg.null_attn:
                    out = self._sdp(qh, kh, vh, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    cache.append(self._merge(kh), self._merge(vh))
                    return y, cache

                # Null-attn prefill (manual).
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                cache.append(self._merge(kh), self._merge(vh))
                return y, cache

            # General cached path: append, then attend over the full cache.
            cache.append(self._merge(kh), self._merge(vh))
            L = cache.pos

            # Decode streaming for T==1 (critical for long context).
            if T == 1:
                decode_block = getattr(cache, "decode_block", 1024)
                if cfg.null_attn:
                    k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                    v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                else:
                    k_null = v_null = None

                if cache.k.is_quantized or cache.v.is_quantized or cfg.null_attn:
                    out = self._streaming_decode_attn(
                        q=qh,
                        k_cache=cache.k,
                        v_cache=cache.v,
                        head_dim=self.qk_head_dim,
                        v_head_dim=self.v_head_dim,
                        decode_block=decode_block,
                        scale=(1.0 / math.sqrt(self.qk_head_dim)),
                        k_null=k_null,
                        v_null=v_null,
                    )
                else:
                    # fp16 cache on GPU -> SDPA is usually faster than Python streaming
                    k_all = self._shape(cache.k.get(dtype=qh.dtype), self.qk_head_dim)
                    v_all = self._shape(cache.v.get(dtype=qh.dtype), self.v_head_dim)
                    out = F.scaled_dot_product_attention(qh, k_all, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)

                y = self.out_proj(self._merge(out))
                return y, cache

            # Prefill/chunked attention (T>1): fallback (materializes K/V). This is still O(T*L).
            k_all, v_all = cache.get(dtype=x.dtype)
            kh_all = self._shape(k_all, self.qk_head_dim)
            vh_all = self._shape(v_all, self.v_head_dim)

            scores = torch.matmul(qh, kh_all.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)  # (B,H,T,L)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh_all], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                out = torch.matmul(attn, vh_all)

            y = self.out_proj(self._merge(out))
            return y, cache

        # -----------------------------
        # gqa
        # -----------------------------
        if cfg.attn_mode == "gqa":
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            qh = self._shape(q, self.qk_head_dim, H=self.H)               # (B,H,T,hd)
            kh = self._shape(k, self.qk_head_dim, H=self.H_kv)            # (B,H_kv,T,hd)
            vh = self._shape(v, self.v_head_dim, H=self.H_kv)             # (B,H_kv,T,hd)

            if self.rotary is not None:
                qh = self.rotary.rotate(qh, pos_offset)
                kh = self.rotary.rotate(kh, pos_offset)

            qh = self._apply_logit_scale_to_q(qh)

            if cache is None:
                # For simplicity (and because this is a research file), we do a straightforward broadcast.
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)  # (B,H,T,hd)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)  # (B,H,T,hd)

                if not cfg.null_attn:
                    out = self._sdp(qh, kh_rep, vh_rep, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    return y, None

                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)

                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                return y, None

            old_len = cache.pos

            if old_len == 0 and T > 1:
                # prefill without cache readback
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)
                if not cfg.null_attn:
                    out = self._sdp(qh, kh_rep, vh_rep, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    cache.append(self._merge(kh), self._merge(vh))
                    return y, cache

                # null-attn prefill (manual)
                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)

                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                cache.append(self._merge(kh), self._merge(vh))
                return y, cache

            cache.append(self._merge(kh), self._merge(vh))
            L = cache.pos

            if T == 1:
                decode_block = getattr(cache, "decode_block", 1024)
                if cfg.null_attn:
                    k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                    v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                else:
                    k_null = v_null = None

                out = self._streaming_decode_attn_gqa(
                    q=qh,
                    k_cache=cache.k,
                    v_cache=cache.v,
                    head_dim=self.qk_head_dim,
                    decode_block=decode_block,
                    scale=(1.0 / math.sqrt(self.qk_head_dim)),
                    k_null=k_null,
                    v_null=v_null,
                )
                y = self.out_proj(self._merge(out))
                return y, cache

            # T>1 fallback
            k_all, v_all = cache.get(dtype=x.dtype)
            kh_all = self._shape(k_all, self.qk_head_dim, H=self.H_kv).repeat_interleave(self.group_size, dim=1)
            vh_all = self._shape(v_all, self.v_head_dim, H=self.H_kv).repeat_interleave(self.group_size, dim=1)

            scores = torch.matmul(qh, kh_all.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv).repeat_interleave(self.group_size, dim=1)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv).repeat_interleave(self.group_size, dim=1)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh_all], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                out = torch.matmul(attn, vh_all)

            y = self.out_proj(self._merge(out))
            return y, cache

        # -----------------------------
        # decoupled
        # -----------------------------
        q_sem = self.q_sem(x)
        k_sem = self.k_sem(x)
        q_geo = self.q_geo(x)
        k_geo = self.k_geo(x)
        v = self.v_proj(x)

        qsh = self._shape(q_sem, self.sem_head_dim)
        ksh = self._shape(k_sem, self.sem_head_dim)
        qgh = self._shape(q_geo, self.geo_head_dim)
        kgh = self._shape(k_geo, self.geo_head_dim)
        vh = self._shape(v, self.v_head_dim)

        if self.rotary is not None:
            qgh = self.rotary.rotate(qgh, pos_offset)
            kgh = self.rotary.rotate(kgh, pos_offset)

        # Apply per-head temperature to both paths by scaling Q.
        qsh = self._apply_logit_scale_to_q(qsh)
        qgh = self._apply_logit_scale_to_q(qgh)

        sem_scale = 1.0 / math.sqrt(self.sem_head_dim)
        geo_scale = 1.0 / math.sqrt(self.geo_head_dim)

        if cache is None:
            if not cfg.null_attn:
                # Combine into a single SDPA call by concatenating along head_dim.
                q_cat = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)
                k_cat = torch.cat([ksh, kgh], dim=-1)
                out = self._sdp(q_cat, k_cat, vh, attn_mask=None if attn_mask is None else attn_mask)
                y = self.out_proj(self._merge(out))
                return y, None

            # Null-attn manual path
            sem = torch.matmul(qsh, ksh.transpose(-2, -1)) * sem_scale
            geo = torch.matmul(qgh, kgh.transpose(-2, -1)) * geo_scale
            scores = sem + geo

            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)

            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)

            s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)

            if attn_mask is not None:
                extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                keep = torch.cat([extra, attn_mask], dim=-1)
                scores = scores.masked_fill(~keep, ninfty)

            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            vals = torch.cat([vn, vh], dim=-2)
            out = torch.matmul(attn, vals)

            y = self.out_proj(self._merge(out))
            return y, None

        old_len = cache.pos

        # Empty-cache prefill: compute locally, then append (no dequant).
        if old_len == 0 and T > 1:
            if not cfg.null_attn:
                q_cat = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)
                k_cat = torch.cat([ksh, kgh], dim=-1)
                out = self._sdp(q_cat, k_cat, vh, attn_mask=None if attn_mask is None else attn_mask)
                y = self.out_proj(self._merge(out))
                cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
                return y, cache

            # Null-attn prefill manual
            sem = torch.matmul(qsh, ksh.transpose(-2, -1)) * sem_scale
            geo = torch.matmul(qgh, kgh.transpose(-2, -1)) * geo_scale
            scores = sem + geo

            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)

            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)

            s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)

            if attn_mask is not None:
                extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                keep = torch.cat([extra, attn_mask], dim=-1)
                scores = scores.masked_fill(~keep, ninfty)

            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            vals = torch.cat([vn, vh], dim=-2)
            out = torch.matmul(attn, vals)

            y = self.out_proj(self._merge(out))
            cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
            return y, cache

        # General cached path.
        cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
        L = cache.pos

        # Streaming decode for T==1.
        if T == 1:
            decode_block = getattr(cache, "decode_block", 1024)
            if cfg.null_attn:
                ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
                kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
                vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
            else:
                ksn = kgn = vn = None

            if cache.k_sem.is_quantized or cache.k_geo.is_quantized or cache.v.is_quantized or cfg.null_attn:
                # Prefer fused kernels when available/allowed.
                use_fused = getattr(cache, "fused", "none")
                fused_ok = (
                    (not cfg.null_attn)
                    and use_fused in ("auto", "triton1pass", "triton2pass")
                    and _triton_decoupled_q4q8q4_available()
                    and cache.k_sem.kind == "q4_0"
                    and cache.k_geo.kind == "q8_0"
                    and cache.v.kind == "q4_0"
                )
                if fused_ok:
                    try:
                        if use_fused == "triton1pass":
                            out = self._fused_decode_attn_decoupled_q4q8q4(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                sem_head_dim=self.sem_head_dim,
                                geo_head_dim=self.geo_head_dim,
                                v_head_dim=self.v_head_dim,
                                decode_block=decode_block,
                                sem_scale=sem_scale,
                                geo_scale=geo_scale,
                            )
                        elif use_fused == "triton2pass":
                            out = self._fused_decode_attn_decoupled_q4q8q4_2pass(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                sem_head_dim=self.sem_head_dim,
                                geo_head_dim=self.geo_head_dim,
                                v_head_dim=self.v_head_dim,
                                decode_block=decode_block,
                                sem_scale=sem_scale,
                                geo_scale=geo_scale,
                            )
                        else:
                            # auto: 2-pass when the sequence is "long enough" that split-K parallelism helps.
                            if cache.pos >= 4 * int(decode_block):
                                out = self._fused_decode_attn_decoupled_q4q8q4_2pass(
                                    q_sem=qsh,
                                    q_geo=qgh,
                                    cache=cache,
                                    sem_head_dim=self.sem_head_dim,
                                    geo_head_dim=self.geo_head_dim,
                                    v_head_dim=self.v_head_dim,
                                    decode_block=decode_block,
                                    sem_scale=sem_scale,
                                    geo_scale=geo_scale,
                                )
                            else:
                                out = self._fused_decode_attn_decoupled_q4q8q4(
                                    q_sem=qsh,
                                    q_geo=qgh,
                                    cache=cache,
                                    sem_head_dim=self.sem_head_dim,
                                    geo_head_dim=self.geo_head_dim,
                                    v_head_dim=self.v_head_dim,
                                    decode_block=decode_block,
                                    sem_scale=sem_scale,
                                    geo_scale=geo_scale,
                                )
                    except Exception:
                        out = self._streaming_decode_attn_decoupled(
                            q_sem=qsh,
                            q_geo=qgh,
                            k_sem_cache=cache.k_sem,
                            k_geo_cache=cache.k_geo,
                            v_cache=cache.v,
                            sem_head_dim=self.sem_head_dim,
                            geo_head_dim=self.geo_head_dim,
                            v_head_dim=self.v_head_dim,
                            decode_block=decode_block,
                            sem_scale=sem_scale,
                            geo_scale=geo_scale,
                            k_sem_null=ksn,
                            k_geo_null=kgn,
                            v_null=vn,
                        )
                else:
                    out = self._streaming_decode_attn_decoupled(
                        q_sem=qsh,
                        q_geo=qgh,
                        k_sem_cache=cache.k_sem,
                        k_geo_cache=cache.k_geo,
                        v_cache=cache.v,
                        sem_head_dim=self.sem_head_dim,
                        geo_head_dim=self.geo_head_dim,
                        v_head_dim=self.v_head_dim,
                        decode_block=decode_block,
                        sem_scale=sem_scale,
                        geo_scale=geo_scale,
                        k_sem_null=ksn,
                        k_geo_null=kgn,
                        v_null=vn,
                    )
            else:
                # fp16 cache -> materialize and use SDPA
                k_sem_all = self._shape(cache.k_sem.get(dtype=qsh.dtype), self.sem_head_dim)
                k_geo_all = self._shape(cache.k_geo.get(dtype=qsh.dtype), self.geo_head_dim)
                v_all = self._shape(cache.v.get(dtype=qsh.dtype), self.v_head_dim)
                q_cat = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)
                k_cat = torch.cat([k_sem_all, k_geo_all], dim=-1)
                out = F.scaled_dot_product_attention(q_cat, k_cat, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)

            y = self.out_proj(self._merge(out))
            return y, cache

        # T>1 fallback (materialize).
        k_sem_all, k_geo_all, v_all = cache.get(dtype=x.dtype)
        ksh_all = self._shape(k_sem_all, self.sem_head_dim)
        kgh_all = self._shape(k_geo_all, self.geo_head_dim)
        vh_all = self._shape(v_all, self.v_head_dim)

        sem = torch.matmul(qsh, ksh_all.transpose(-2, -1)) * sem_scale
        geo = torch.matmul(qgh, kgh_all.transpose(-2, -1)) * geo_scale
        scores = sem + geo

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, ninfty)
        elif T > 1:
            key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
            q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
            keep = key_pos <= q_pos
            scores = scores.masked_fill(~keep, ninfty)

        if cfg.null_attn:
            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
            s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            vals = torch.cat([vn, vh_all], dim=-2)
            out = torch.matmul(attn, vals)
        else:
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            out = torch.matmul(attn, vh_all)

        y = self.out_proj(self._merge(out))
        return y, cache


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = DecoupledBottleneckAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor], cache: Optional[Any], pos_offset: int) -> Tuple[torch.Tensor, Optional[Any]]:
        a, cache = self.attn(self.ln1(x), attn_mask=attn_mask, cache=cache, pos_offset=pos_offset)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # lexical bottleneck
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

        # Training-only toggles (not part of ModelConfig)
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

        # Attention mask strategy:
        # - Training (no cache):
        #     * if null_attn is disabled, we pass attn_mask=None and let SDPA run with is_causal=True.
        #     * if null_attn is enabled (manual attention path), we provide a causal boolean mask.
        # - Cached prefill (T>1):
        #     * build a (1,1,T,L) boolean mask so the new chunk can attend to the prefix + causal within itself.
        #     * if cache is empty and null_attn is disabled, we again pass attn_mask=None so SDPA can use is_causal=True.
        # - Decode (T==1): no mask needed.
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

        # Training memory saver: gradient checkpointing (only when not using KV cache).
        if caches is None and self.training and getattr(self, "grad_checkpointing", False):
            try:
                from torch.utils.checkpoint import checkpoint  # type: ignore
                for blk in self.blocks:
                    def _blk_fwd(x_in: torch.Tensor, blk=blk) -> torch.Tensor:
                        y, _ = blk(x_in, attn_mask=attn_mask, cache=None, pos_offset=pos_offset)
                        return y
                    x = checkpoint(_blk_fwd, x, use_reentrant=False)
            except Exception:
                # Fallback: run without checkpointing
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
    def generate(
        self,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        # KV-cache controls (these matter *a lot* at long context)
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",  # {none, auto, triton1pass, triton2pass}
        # v25 self-optimizer (optional): picks decode knobs that are fastest on this GPU.
        self_opt: Optional["KVSelfOptConfig"] = None,
        # Optional heterogeneous overrides
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
        """Autoregressive generation with optional heterogeneous KV-cache quantization.

        The cache API supports fp16/fp32/q8_0/q4_0/nf4, plus an fp16 residual window for the newest tokens.
        """
        was_training = self.training
        self.eval()
        device = prompt.device
        B, T0 = prompt.shape
        max_seq = T0 + max_new_tokens

        if kv_fused not in ("none", "auto", "triton1pass", "triton2pass"):
            raise ValueError("kv_fused must be one of: none, auto, triton1pass, triton2pass")

        # Heterogeneous default for decoupled mode:
        # - Per the draft paper: keep the RoPE/geometric key path higher precision, compress the semantic path harder.
        #   (You can override this with --kv-cache-k-sem / --kv-cache-k-geo.)
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

        # default K/V configs (standard/bottleneck/gqa)
        k_cfg = make_cfg(kv_cache_k, kv_qblock_k)
        v_cfg = make_cfg(kv_cache_v, kv_qblock_v)

        # decoupled configs
        k_sem_cfg = make_cfg(kv_cache_k_sem, kv_qblock_k_sem)
        k_geo_cfg = make_cfg(kv_cache_k_geo, kv_qblock_k_geo)
        v_dec_cfg = make_cfg(kv_cache_v, kv_qblock_v)

        # v26: cache-policy self-opt (startup only).
        # This chooses: kv_residual hot-window length, quant kinds, qblocks (within a strict memory budget).
        if (
            self_opt is not None
            and getattr(self_opt, "mode", "none") != "none"
            and getattr(self_opt, "scope", "all") in ("cache", "all")
            and self.cfg.attn_mode == "decoupled"
        ):
            try:
                base_policy = KVCachePolicy(
                    k_sem_kind=k_sem_cfg.kind,
                    k_geo_kind=k_geo_cfg.kind,
                    v_kind=v_dec_cfg.kind,
                    k_sem_qblock=k_sem_cfg.qblock,
                    k_geo_qblock=k_geo_cfg.qblock,
                    v_qblock=v_dec_cfg.qblock,
                    residual_len=int(kv_residual),
                )
                pol_tuner = KVCachePolicySelfOptimizer(
                    self_opt,
                    device=device,
                    attn=self.blocks[0].attn,
                    model_cfg=self.cfg,
                    batch_size=B,
                    max_seq_len=max_seq,
                    base_policy=base_policy,
                    base_decode_block=kv_decode_block,
                    base_fused=kv_fused,
                )
                chosen = pol_tuner.choose_policy(prompt_len=T0)

                # Optional policy quality guard: teacher-forced logits vs an fp16-cache baseline.
                # This is intentionally tiny (few steps) so it stays usable in practice.
                if getattr(self_opt, "policy_quality", False):
                    try:
                        calib_spec = getattr(self_opt, "calib_tokens", None)
                        if calib_spec:
                            if os.path.exists(str(calib_spec)):
                                raw = Path(str(calib_spec)).read_text()
                            else:
                                raw = str(calib_spec)
                            calib_ids = [int(t) for t in raw.strip().split() if t.strip()]
                            calib = torch.tensor([calib_ids], device=device, dtype=torch.long)
                        else:
                            calib = prompt.detach()

                        maxerr = self._policy_logit_maxerr_decoupled(
                            calib,
                            policy=chosen,
                            prefill=int(getattr(self_opt, "calib_prefill", 64)),
                            decode_steps=int(getattr(self_opt, "calib_decode_steps", 8)),
                            kv_decode_block=int(kv_decode_block),
                        )
                        tol = float(getattr(self_opt, "quality_tol", 0.5))
                        if maxerr > tol:
                            if getattr(self_opt, "verbose", False):
                                print(f"[selfopt] cache-policy rejected by quality guard: max|Δlogit|={maxerr:.4g} > {tol:.4g} (falling back)")
                            chosen = base_policy
                        elif getattr(self_opt, "verbose", False):
                            print(f"[selfopt] cache-policy quality ok: max|Δlogit|={maxerr:.4g} <= {tol:.4g}")
                    except Exception as e:
                        if getattr(self_opt, "verbose", False):
                            print(f"[selfopt] cache-policy quality check skipped (error): {e}")

                k_sem_cfg, k_geo_cfg, v_dec_cfg = chosen.to_tensor_cfgs()
                if getattr(self_opt, "verbose", False):
                    print(f"[selfopt] cache-policy active: {chosen.short()}")
            except Exception as e:
                if getattr(self_opt, "verbose", False):
                    print(f"[selfopt] cache-policy tuning failed, falling back: {e}")

        caches: List[Any] = []
        for _ in range(self.cfg.n_layer):
            if self.cfg.attn_mode == "decoupled":
                c = DecoupledLayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_sem_dim=self.cfg.sem_dim,
                    k_geo_dim=self.cfg.geo_dim,
                    v_dim=self.cfg.attn_dim,
                    k_sem_cfg=k_sem_cfg,
                    k_geo_cfg=k_geo_cfg,
                    v_cfg=v_dec_cfg,
                    device=device,
                )
                c.decode_block = kv_decode_block
                c.fused = kv_fused
                caches.append(c)
            else:
                if self.cfg.attn_mode == "standard":
                    k_dim = v_dim = self.cfg.d_model
                elif self.cfg.attn_mode == "bottleneck":
                    k_dim = v_dim = self.cfg.attn_dim
                elif self.cfg.attn_mode == "gqa":
                    head_dim = self.cfg.attn_dim // self.cfg.n_head
                    kv_head = self.cfg.kv_head if self.cfg.kv_head is not None else self.cfg.n_head
                    k_dim = v_dim = kv_head * head_dim
                else:
                    raise ValueError(f"Unknown attn_mode for KV cache: {self.cfg.attn_mode}")

                c = LayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_dim=k_dim,
                    v_dim=v_dim,
                    k_cfg=k_cfg,
                    v_cfg=v_cfg,
                    device=device,
                )
                c.decode_block = kv_decode_block
                c.fused = kv_fused
                caches.append(c)

        # Prefill (fills caches). Thanks to the attention module, the "empty-cache prefill" path avoids cache readback.
        logits, caches = self(prompt, caches=caches, pos_offset=0)

        # v25: runtime self-optimizer for decode performance (optional).
        tuner: Optional[KVDecodeSelfOptimizer] = None
        if self_opt is not None and getattr(self_opt, "mode", "none") != "none":
            tuner = KVDecodeSelfOptimizer(
                self_opt,
                device=device,
                base_fused=kv_fused,
                base_decode_block=kv_decode_block,
                log_callback=log_callback,
            )

        out = prompt
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], neg_inf(next_logits.dtype))
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)

            # v25 self-opt: set per-layer decode knobs (decode_block, fused mode, kernel launch params).
            if tuner is not None and caches:
                try:
                    L_prefix = int(getattr(caches[0], "pos", out.size(1) - 1))
                    plan = tuner.maybe_get_plan(attn=self.blocks[0].attn, cache=caches[0], L_prefix=L_prefix)
                    if plan is not None:
                        for c in caches:
                            plan.apply_to_cache(c)
                except Exception:
                    pass

            # decode one token
            logits, caches = self(next_id, caches=caches, pos_offset=out.size(1) - 1)

        if was_training:
            self.train()
        return out






    @torch.no_grad()
    def _policy_logit_maxerr_decoupled(
        self,
        tokens: torch.Tensor,
        *,
        policy: KVCachePolicy,
        prefill: int,
        decode_steps: int,
        kv_decode_block: int,
    ) -> float:
        """Quality guard for cache-policy tuning.

        Returns max(abs(logits_candidate - logits_fp16_baseline)) over a short teacher-forced decode window.
        This is NOT a substitute for proper eval (perplexity), but it catches "oops, this quant combo is broken".
        """
        if tokens.numel() == 0:
            return 0.0

        # Keep it cheap: single batch item only.
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens[:1].contiguous()

        device = tokens.device
        B, L = tokens.shape
        if L < 2:
            return 0.0

        prefill = int(max(1, min(int(prefill), L - 1)))
        decode_steps = int(max(1, min(int(decode_steps), L - prefill)))

        # We'll run up to (prefill + decode_steps) tokens total.
        max_seq = prefill + decode_steps

        # Force eval for determinism (dropout off).
        was_training = self.training
        self.eval()

        try:
            # Baseline: fp16 caches everywhere.
            fp16_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
            caches_base: List[Any] = []
            for _ in range(self.cfg.n_layer):
                c = DecoupledLayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_sem_dim=self.cfg.sem_dim,
                    k_geo_dim=self.cfg.geo_dim,
                    v_dim=self.cfg.attn_dim,
                    k_sem_cfg=fp16_cfg,
                    k_geo_cfg=fp16_cfg,
                    v_cfg=fp16_cfg,
                    device=device,
                )
                c.decode_block = int(kv_decode_block)
                c.fused = "none"
                caches_base.append(c)

            # Candidate caches: chosen policy (quantized).
            k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
            caches_cand: List[Any] = []
            for _ in range(self.cfg.n_layer):
                c = DecoupledLayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_sem_dim=self.cfg.sem_dim,
                    k_geo_dim=self.cfg.geo_dim,
                    v_dim=self.cfg.attn_dim,
                    k_sem_cfg=k_sem_cfg,
                    k_geo_cfg=k_geo_cfg,
                    v_cfg=v_cfg,
                    device=device,
                )
                c.decode_block = int(kv_decode_block)
                c.fused = "none"  # ensure we don't mix kernel numeric differences into the quant check
                caches_cand.append(c)

            # Prefill
            prompt = tokens[:, :prefill]
            logits_base, caches_base = self(prompt, caches=caches_base, pos_offset=0)
            logits_cand, caches_cand = self(prompt, caches=caches_cand, pos_offset=0)

            # Teacher-forced decode window
            base_steps: List[torch.Tensor] = []
            for i in range(prefill, prefill + decode_steps):
                x = tokens[:, i : i + 1]
                logits_base, caches_base = self(x, caches=caches_base, pos_offset=i)
                base_steps.append(logits_base[:, -1, :].detach().float().cpu())

            max_err = 0.0
            for j, i in enumerate(range(prefill, prefill + decode_steps)):
                x = tokens[:, i : i + 1]
                logits_cand, caches_cand = self(x, caches=caches_cand, pos_offset=i)
                cand = logits_cand[:, -1, :].detach().float().cpu()
                err = (cand - base_steps[j]).abs().max().item()
                if err > max_err:
                    max_err = float(err)

            return float(max_err)
        finally:
            if was_training:
                self.train()

@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    *,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split, tok in [("train", train_tokens), ("val", val_tokens)]:
        losses: List[float] = []
        for _ in range(eval_iters):
            x, y = get_batch(tok, batch_size, block_size, device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out["train"], out["val"]


def save_ckpt(out_dir: str, name: str, model: GPT, cfg: ModelConfig, step: int, best_val: float) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(
        {
            "config": asdict(cfg),
            "model": model.state_dict(),
            "step": step,
            "best_val": best_val,
        },
        path,
    )
    return path


# -----------------------------
# Experiment presets + instrumentation (v27)
# -----------------------------

# These presets are designed to match the "Experiment Suite Summary" you shared.
# They only apply when you pass --size and/or --exp. Explicit CLI flags still win.

SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    # params, d_model, layers, heads, d_ff, context, batch, steps
    "tiny":   dict(d_model=512,  layers=6,  n_head=8,  d_ff=2048, block=1024, batch_size=16, steps=6000),
    "small":  dict(d_model=768,  layers=12, n_head=12, d_ff=3072, block=1024, batch_size=8,  steps=10000),
    "medium": dict(d_model=1024, layers=24, n_head=16, d_ff=4096, block=2048, batch_size=4,  steps=15000),
    "large":  dict(d_model=1536, layers=24, n_head=16, d_ff=6144, block=2048, batch_size=2,  steps=20000),
}

# Attention dims for the "paper_*" experiments (matching your table).
BOTTLENECK_ATTN_DIM: Dict[str, int] = {"tiny": 96, "small": 144, "medium": 192, "large": 288}
DECOUPLED_SEM_DIM:   Dict[str, int] = {"tiny": 32, "small": 48,  "medium": 64,  "large": 96}
DECOUPLED_GEO_DIM:   Dict[str, int] = {"tiny": 64, "small": 96,  "medium": 128, "large": 192}
DECOUPLED_ATTN_DIM:  Dict[str, int] = {"tiny": 96, "small": 144, "medium": 192, "large": 288}
GQA_KV_HEAD:         Dict[str, int] = {"tiny": 2,  "small": 3,   "medium": 4,   "large": 4}

EXP_PRESETS: Dict[str, Dict[str, Any]] = {
    "paper_baseline": dict(attn_mode="standard"),
    "paper_bottleneck": dict(attn_mode="bottleneck", null_attn=True),
    "paper_decoupled": dict(attn_mode="decoupled", tie_qk=True, null_attn=True, rope=True),
    "paper_gqa": dict(attn_mode="gqa"),
}

def _argv_has_flag(flag: str) -> bool:
    # Detect explicit user overrides (argparse defaults are otherwise indistinguishable).
    return flag in sys.argv

def apply_size_preset(args: argparse.Namespace) -> None:
    if not getattr(args, "size", None):
        return
    size = str(args.size)
    if size not in SIZE_PRESETS:
        raise ValueError(f"Unknown size preset: {size}")
    p = SIZE_PRESETS[size]
    # Only override if the user did not specify the corresponding CLI flag.
    if not _argv_has_flag("--d-model"):
        args.d_model = p["d_model"]
    if not _argv_has_flag("--layers"):
        args.layers = p["layers"]
    if not _argv_has_flag("--n-head"):
        args.n_head = p["n_head"]
    if not _argv_has_flag("--d-ff"):
        args.d_ff = p["d_ff"]
    if not _argv_has_flag("--block"):
        args.block = p["block"]
    if not _argv_has_flag("--batch-size"):
        args.batch_size = p["batch_size"]
    if not _argv_has_flag("--steps"):
        args.steps = p["steps"]
    # default embed_dim tracks d_model unless explicitly set
    if not _argv_has_flag("--embed-dim"):
        args.embed_dim = p["d_model"]

def apply_exp_preset(args: argparse.Namespace) -> None:
    if not getattr(args, "exp", None):
        return
    exp = str(args.exp)
    if exp not in EXP_PRESETS and exp != "paper_all":
        raise ValueError(f"Unknown experiment preset: {exp}")

    # For paper_all, we don't set mode here; the runner loops over EXP_PRESETS.
    if exp == "paper_all":
        return

    preset = EXP_PRESETS[exp]
    size = str(getattr(args, "size", "")) if getattr(args, "size", None) else None

    # attn_mode
    if not _argv_has_flag("--attn-mode") and "attn_mode" in preset:
        args.attn_mode = preset["attn_mode"]

    # Experiment-specific dims (size-dependent)
    if size is not None:
        if exp == "paper_bottleneck":
            if not _argv_has_flag("--attn-dim"):
                args.attn_dim = BOTTLENECK_ATTN_DIM[size]
        if exp == "paper_decoupled":
            if not _argv_has_flag("--sem-dim"):
                args.sem_dim = DECOUPLED_SEM_DIM[size]
            if not _argv_has_flag("--geo-dim"):
                args.geo_dim = DECOUPLED_GEO_DIM[size]
            if not _argv_has_flag("--attn-dim"):
                args.attn_dim = DECOUPLED_ATTN_DIM[size]
        if exp == "paper_gqa":
            if not _argv_has_flag("--kv-head"):
                args.kv_head = GQA_KV_HEAD[size]
            if not _argv_has_flag("--attn-dim"):
                # Keep head_dim identical to baseline by default.
                args.attn_dim = int(getattr(args, "d_model", SIZE_PRESETS[size]["d_model"]))

    # Bool toggles (only set if user didn't explicitly toggle)
    if "null_attn" in preset:
        if (not _argv_has_flag("--null-attn")) and (not _argv_has_flag("--no-null-attn")):
            args.null_attn = bool(preset["null_attn"])
    if "tie_qk" in preset:
        if (not _argv_has_flag("--tie-qk")) and (not _argv_has_flag("--no-tie-qk")):
            args.tie_qk = bool(preset["tie_qk"])
    if "rope" in preset:
        if (not _argv_has_flag("--no-rope")) and (not _argv_has_flag("--rope")):
            # v26 default is rope=True; keep explicit override logic anyway.
            if preset["rope"]:
                args.no_rope = False
            else:
                args.no_rope = True

def default_out_dir(args: argparse.Namespace) -> Optional[str]:
    """
    If the user didn't set --out-dir, build it as runs/{size}_{expSuffix}.
    Returns None if we cannot infer.
    """
    if getattr(args, "out_dir", None):
        return str(args.out_dir)
    size = getattr(args, "size", None)
    exp = getattr(args, "exp", None)
    run_root = getattr(args, "run_root", "runs")
    tag = getattr(args, "run_tag", None)
    if not size or not exp or exp == "paper_all":
        return None
    suffix = str(exp).replace("paper_", "")
    name = f"{size}_{suffix}"
    if tag:
        name = f"{name}_{tag}"
    return os.path.join(run_root, name)

def human_bytes(n: float) -> str:
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    while n >= 1024.0 and u < len(units) - 1:
        n /= 1024.0
        u += 1
    if u == 0:
        return f"{n:.0f}{units[u]}"
    if n >= 100:
        return f"{n:.1f}{units[u]}"
    return f"{n:.2f}{units[u]}"

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")

def _device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = "cuda"
        return f"cuda:{device.index or 0} ({name})"
    if device.type == "mps":
        return "mps"
    return str(device)

def _env_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["time"] = _now_iso()
    info["python"] = sys.version.replace("\n", " ")
    info["platform"] = platform.platform()
    info["torch"] = getattr(torch, "__version__", "unknown")
    info["device"] = _device_summary(device)
    info["triton_available"] = bool(TRITON_AVAILABLE)
    if device.type == "cuda":
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return info

# ---- KV cache memory estimator (architecture-aware, and quant-aware) ----

def _seqcache_bytes_for(kind: KVCacheKind, *, dim: int, qblock: int, residual_len: int, batch: int, seq_len: int) -> int:
    kind = str(kind)
    if kind == "fp16":
        return int(batch * seq_len * dim * 2)
    if kind == "fp32":
        return int(batch * seq_len * dim * 4)
    # Quantized kinds: use the same spec logic as the cache implementation (pad_dim + n_blocks).
    spec = make_quantspec(kind, dim, qblock)
    if kind == "q8_0":
        q_bytes = int(batch * seq_len * spec.pad_dim * 1)  # int8
    elif kind in ("q4_0", "nf4"):
        q_bytes = int(batch * seq_len * (spec.pad_dim // 2) * 1)  # packed uint8
    else:
        raise ValueError(kind)
    s_bytes = int(batch * seq_len * spec.n_blocks * 2)  # fp16 scales
    r_eff = int(max(0, residual_len))
    # Residual ring is allocated even if seq_len is smaller than residual_len; for estimates, assume max_seq_len == seq_len.
    r_alloc = int(min(r_eff, seq_len))
    r_bytes = int(batch * r_alloc * dim * 2)
    return q_bytes + s_bytes + r_bytes

def estimate_kv_cache_bytes(
    cfg: ModelConfig,
    *,
    seq_len: int,
    batch: int = 1,
    kv_cache: KVCacheKind = "fp16",
    kv_qblock: int = 32,
    kv_residual: int = 0,
    kv_cache_k: Optional[KVCacheKind] = None,
    kv_cache_v: Optional[KVCacheKind] = None,
    kv_cache_k_sem: Optional[KVCacheKind] = None,
    kv_cache_k_geo: Optional[KVCacheKind] = None,
    kv_qblock_k: Optional[int] = None,
    kv_qblock_v: Optional[int] = None,
    kv_qblock_k_sem: Optional[int] = None,
    kv_qblock_k_geo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Architecture-aware KV cache estimate for a *single* sequence length + batch.
    Returns per-layer and total sizes.
    """
    def kind_of(default_kind: KVCacheKind, override: Optional[KVCacheKind]) -> KVCacheKind:
        return override if override is not None else default_kind
    def qb_of(default_qb: int, override: Optional[int]) -> int:
        return int(override) if override is not None else int(default_qb)

    # v26 heterogeneous default for decoupled when kv_cache == q4_0
    k_sem_kind = kind_of(kv_cache, kv_cache_k_sem)
    k_geo_kind = kind_of(kv_cache, kv_cache_k_geo)
    v_kind = kind_of(kv_cache, kv_cache_v)
    k_kind = kind_of(kv_cache, kv_cache_k)

    k_sem_qb = qb_of(kv_qblock, kv_qblock_k_sem)
    k_geo_qb = qb_of(kv_qblock, kv_qblock_k_geo)
    k_qb = qb_of(kv_qblock, kv_qblock_k)
    v_qb = qb_of(kv_qblock, kv_qblock_v)

    # residual applies only to quant kinds (matches cache code)
    def resid_for(kind: KVCacheKind) -> int:
        return int(kv_residual) if str(kind) not in ("fp16", "fp32") else 0

    mode = cfg.attn_mode
    if mode in ("standard", "bottleneck"):
        k_dim = cfg.d_model if mode == "standard" else cfg.attn_dim
        v_dim = cfg.d_model if mode == "standard" else cfg.attn_dim
        k_bytes = _seqcache_bytes_for(k_kind, dim=k_dim, qblock=k_qb, residual_len=resid_for(k_kind), batch=batch, seq_len=seq_len)
        v_bytes = _seqcache_bytes_for(v_kind, dim=v_dim, qblock=v_qb, residual_len=resid_for(v_kind), batch=batch, seq_len=seq_len)
        per_layer = k_bytes + v_bytes
        total = int(cfg.n_layer * per_layer)
        return dict(mode=mode, seq_len=seq_len, batch=batch, per_layer_bytes=per_layer, total_bytes=total, details=dict(k=k_bytes, v=v_bytes))
    if mode == "gqa":
        H = cfg.n_head
        H_kv = cfg.kv_head if cfg.kv_head is not None else H
        head_dim = cfg.attn_dim // H
        kv_dim = int(H_kv * head_dim)
        k_bytes = _seqcache_bytes_for(k_kind, dim=kv_dim, qblock=k_qb, residual_len=resid_for(k_kind), batch=batch, seq_len=seq_len)
        v_bytes = _seqcache_bytes_for(v_kind, dim=kv_dim, qblock=v_qb, residual_len=resid_for(v_kind), batch=batch, seq_len=seq_len)
        per_layer = k_bytes + v_bytes
        total = int(cfg.n_layer * per_layer)
        return dict(mode=mode, seq_len=seq_len, batch=batch, per_layer_bytes=per_layer, total_bytes=total, details=dict(k=k_bytes, v=v_bytes, kv_dim=kv_dim))
    if mode == "decoupled":
        # heterogeneous default: q4 semantic, q8 geo, q4 V
        if str(kv_cache) == "q4_0":
            if kv_cache_k_geo is None:
                k_geo_kind = "q8_0"
            if kv_cache_k_sem is None:
                k_sem_kind = "q4_0"
            if kv_cache_v is None:
                v_kind = "q4_0"
        k_sem_bytes = _seqcache_bytes_for(k_sem_kind, dim=cfg.sem_dim, qblock=k_sem_qb, residual_len=resid_for(k_sem_kind), batch=batch, seq_len=seq_len)
        k_geo_bytes = _seqcache_bytes_for(k_geo_kind, dim=cfg.geo_dim, qblock=k_geo_qb, residual_len=resid_for(k_geo_kind), batch=batch, seq_len=seq_len)
        v_bytes = _seqcache_bytes_for(v_kind, dim=cfg.attn_dim, qblock=v_qb, residual_len=resid_for(v_kind), batch=batch, seq_len=seq_len)
        per_layer = k_sem_bytes + k_geo_bytes + v_bytes
        total = int(cfg.n_layer * per_layer)
        return dict(mode=mode, seq_len=seq_len, batch=batch, per_layer_bytes=per_layer, total_bytes=total,
                    details=dict(k_sem=k_sem_bytes, k_geo=k_geo_bytes, v=v_bytes))
    raise ValueError(mode)


# ---- Deep instrumentation: JSONL + HDF5 + plots ----

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore

try:
    import h5py as _h5py  # type: ignore
except Exception:
    _h5py = None  # type: ignore

class RunLogger:
    def __init__(
        self,
        out_dir: str,
        *,
        instrument: str,
        cfg: ModelConfig,
        args: argparse.Namespace,
        device: torch.device,
        live_plot: bool = False,
        tb: bool = False,
    ):
        self.out_dir = str(out_dir)
        self.instrument = str(instrument)
        self.cfg = cfg
        self.args = args
        self.device = device
        self.start_time = time.time()

        os.makedirs(self.out_dir, exist_ok=True)

        self.train_jsonl_path = os.path.join(self.out_dir, "train.jsonl")
        self.summary_path = os.path.join(self.out_dir, "summary.md")
        self.h5_path = os.path.join(self.out_dir, "analysis.h5")
        self.png_path = os.path.join(self.out_dir, "analysis.png")

        self._jsonl_f = None
        self._h5 = None

        self._live = None
        self._tb = None

        if self.instrument != "off":
            self._jsonl_f = open(self.train_jsonl_path, "w", encoding="utf-8")
            self.log({"type": "meta", "step": 0, "env": _env_info(device), "argv": sys.argv, "args": vars(args), "config": asdict(cfg)})

        # HDF5 only in "full"
        if self.instrument == "full" and _h5py is not None:
            try:
                self._h5 = _h5py.File(self.h5_path, "w")
                meta = self._h5.create_group("meta")
                meta.attrs["created"] = _now_iso()
                meta.attrs["argv"] = " ".join(sys.argv)
                meta.attrs["config_json"] = json.dumps(asdict(cfg), separators=(",", ":"), sort_keys=True)
            except Exception as e:
                print(f"[warn] Could not open HDF5 at {self.h5_path}: {e}")
                self._h5 = None

        # Optional live plot (matplotlib)
        if live_plot and self.instrument != "off":
            self._live = LivePlotter()
        # Optional tensorboard (requires tensorboard package)
        if tb and self.instrument != "off":
            self._tb = TensorBoardWriter(self.out_dir)

        # Write initial summary.md with config so runs are self-describing immediately.
        self._write_summary(initial_only=True)

    def close(self) -> None:
        try:
            if self._h5 is not None:
                self._h5.flush()
                self._h5.close()
        except Exception:
            pass
        try:
            if self._jsonl_f is not None:
                self._jsonl_f.flush()
                self._jsonl_f.close()
        except Exception:
            pass
        try:
            if self._tb is not None:
                self._tb.close()
        except Exception:
            pass
        try:
            if self._live is not None:
                self._live.close()
        except Exception:
            pass

    def log(self, event: Dict[str, Any]) -> None:
        if self._jsonl_f is None:
            return
        # Add time info for every line.
        event = dict(event)
        event.setdefault("wall_time", _now_iso())
        self._jsonl_f.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._jsonl_f.flush()

        if self._tb is not None:
            self._tb.maybe_log(event)

        if self._live is not None:
            self._live.maybe_update(event)

    def h5_write_step(self, step: int, *, group: str, tensors: Dict[str, torch.Tensor], attrs: Optional[Dict[str, Any]] = None) -> None:
        if self._h5 is None:
            return
        try:
            g = self._h5.require_group(f"{group}/step_{int(step)}")
            if attrs:
                for k, v in attrs.items():
                    try:
                        g.attrs[k] = v
                    except Exception:
                        g.attrs[k] = str(v)
            for name, t in tensors.items():
                if t is None:
                    continue
                arr = t.detach().to("cpu")
                if arr.dtype in (torch.bfloat16, torch.float16):
                    arr = arr.to(torch.float16)
                else:
                    arr = arr.to(torch.float32)
                g.create_dataset(name, data=arr.numpy(), compression="gzip", compression_opts=4)
            self._h5.flush()
        except Exception as e:
            print(f"[warn] HDF5 write failed at step {step}: {e}")

    def finalize(self, *, best_val: float, last_step: int) -> None:
        # Close live plot cleanly; generate final plots; update summary.md
        try:
            if self.instrument != "off":
                generate_analysis_png(self.train_jsonl_path, self.png_path)
        except Exception as e:
            print(f"[warn] analysis.png generation failed: {e}")
        self._write_summary(initial_only=False, best_val=best_val, last_step=last_step)

    def _write_summary(self, *, initial_only: bool, best_val: Optional[float] = None, last_step: Optional[int] = None) -> None:
        try:
            lines: List[str] = []
            lines.append(f"# Run Summary")
            lines.append("")
            lines.append(f"- Created: `{_now_iso()}`")
            lines.append(f"- Out dir: `{self.out_dir}`")
            lines.append(f"- Device: `{_device_summary(self.device)}`")
            lines.append(f"- Command: `{(' '.join(sys.argv))}`")
            if getattr(self.args, "size", None):
                lines.append(f"- Size preset: `{self.args.size}`")
            if getattr(self.args, "exp", None):
                lines.append(f"- Experiment: `{self.args.exp}`")
            lines.append("")

            lines.append("## Model Config")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(asdict(self.cfg), indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")

            lines.append("## Training Args")
            lines.append("")
            # args can include non-serializable values; make a best effort
            try:
                args_json = json.dumps(vars(self.args), indent=2, sort_keys=True, default=str)
            except Exception:
                args_json = str(vars(self.args))
            lines.append("```json")
            lines.append(args_json)
            lines.append("```")
            lines.append("")

            if not initial_only and best_val is not None and last_step is not None:
                ppl = math.exp(best_val) if best_val < 20 else float("inf")
                lines.append("## Results")
                lines.append("")
                lines.append(f"- Last step: `{last_step}`")
                lines.append(f"- Best val loss: `{best_val:.6f}` (ppl `{ppl:.2f}`)")
                lines.append(f"- Files: `train.jsonl`, `analysis.h5` (if enabled), `analysis.png`, `best.pt`, `last.pt`")
                lines.append("")

                # KV cache memory quick summary (fp16 baseline vs chosen kv policy)
                try:
                    ctx = int(self.cfg.block_size)
                    mem_base = estimate_kv_cache_bytes(ModelConfig(**{**asdict(self.cfg), "attn_mode": "standard"}), seq_len=ctx, batch=1, kv_cache="fp16", kv_qblock=32, kv_residual=0)
                    mem_this = estimate_kv_cache_bytes(self.cfg, seq_len=ctx, batch=1,
                                                       kv_cache=getattr(self.args, "kv_cache", "fp16"),
                                                       kv_qblock=int(getattr(self.args, "kv_qblock", 32)),
                                                       kv_residual=int(getattr(self.args, "kv_residual", 0)),
                                                       kv_cache_k=getattr(self.args, "kv_cache_k", None),
                                                       kv_cache_v=getattr(self.args, "kv_cache_v", None),
                                                       kv_cache_k_sem=getattr(self.args, "kv_cache_k_sem", None),
                                                       kv_cache_k_geo=getattr(self.args, "kv_cache_k_geo", None),
                                                       kv_qblock_k=getattr(self.args, "kv_qblock_k", None),
                                                       kv_qblock_v=getattr(self.args, "kv_qblock_v", None),
                                                       kv_qblock_k_sem=getattr(self.args, "kv_qblock_k_sem", None),
                                                       kv_qblock_k_geo=getattr(self.args, "kv_qblock_k_geo", None),
                                                       )
                    lines.append("## KV Cache Memory (batch=1)")
                    lines.append("")
                    lines.append(f"- Baseline fp16 (standard attn) @ ctx={ctx}: `{human_bytes(mem_base['total_bytes'])}`")
                    lines.append(f"- This run policy @ ctx={ctx}: `{human_bytes(mem_this['total_bytes'])}`")
                    if mem_this["total_bytes"] > 0:
                        lines.append(f"- Compression vs fp16 baseline: `{mem_base['total_bytes']/mem_this['total_bytes']:.2f}×`")
                    # Also probe at 128k
                    mem_this_128 = estimate_kv_cache_bytes(self.cfg, seq_len=128_000, batch=1,
                                                           kv_cache=getattr(self.args, "kv_cache", "fp16"),
                                                           kv_qblock=int(getattr(self.args, "kv_qblock", 32)),
                                                           kv_residual=int(getattr(self.args, "kv_residual", 0)),
                                                           kv_cache_k=getattr(self.args, "kv_cache_k", None),
                                                           kv_cache_v=getattr(self.args, "kv_cache_v", None),
                                                           kv_cache_k_sem=getattr(self.args, "kv_cache_k_sem", None),
                                                           kv_cache_k_geo=getattr(self.args, "kv_cache_k_geo", None),
                                                           kv_qblock_k=getattr(self.args, "kv_qblock_k", None),
                                                           kv_qblock_v=getattr(self.args, "kv_qblock_v", None),
                                                           kv_qblock_k_sem=getattr(self.args, "kv_qblock_k_sem", None),
                                                           kv_qblock_k_geo=getattr(self.args, "kv_qblock_k_geo", None),
                                                           )
                    lines.append(f"- This run policy @ 128k: `{human_bytes(mem_this_128['total_bytes'])}`")
                    lines.append("")
                except Exception as e:
                    lines.append(f"## KV Cache Memory")
                    lines.append("")
                    lines.append(f"- (Failed to compute memory estimate: {e})")
                    lines.append("")

            Path(self.summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[warn] Failed to write summary.md: {e}")


class TensorBoardWriter:
    """
    Optional: requires `pip install tensorboard`.
    We keep this extremely lightweight: only scalar logging.
    """
    def __init__(self, out_dir: str):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self.writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        except Exception as e:
            print(f"[warn] TensorBoard not available: {e}. Disable with --tb=0 or install tensorboard.")
            self.writer = None

    def maybe_log(self, event: Dict[str, Any]) -> None:
        if self.writer is None:
            return
        try:
            step = int(event.get("step", 0))
            etype = str(event.get("type", ""))
            if etype == "train":
                if "loss" in event:
                    self.writer.add_scalar("loss/train", float(event["loss"]), step)
                if "ppl" in event:
                    self.writer.add_scalar("ppl/train", float(event["ppl"]), step)
                if "tok_s" in event:
                    self.writer.add_scalar("perf/tok_s", float(event["tok_s"]), step)
            if etype == "eval":
                if "train_loss" in event:
                    self.writer.add_scalar("loss/train_eval", float(event["train_loss"]), step)
                if "val_loss" in event:
                    self.writer.add_scalar("loss/val", float(event["val_loss"]), step)
            if etype == "analysis":
                for k, v in event.items():
                    if k in ("type", "step", "wall_time"):
                        continue
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"analysis/{k}", float(v), step)
        except Exception:
            pass

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass


class LivePlotter:
    """
    Very simple realtime plots for "getting a feel" during dev runs.
    Uses matplotlib interactive mode; safe to disable for headless/prod.
    """
    def __init__(self):
        self.enabled = False
        self.steps: List[int] = []
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.entropy: List[float] = []
        
        # Self-opt history
        self.opt_steps: List[int] = []
        self.opt_block: List[int] = []
        self.opt_fused: List[int] = []  # 0=none, 1=1pass, 2=2pass

        try:
            import matplotlib.pyplot as plt  # type: ignore
            self.plt = plt
            self.plt.ion()
            # Layout: 2x3 grid
            # Row 0: Train Loss, Val Loss, Entropy
            # Row 1: Decode Block, Fused Mode, (Empty/Future)
            self.fig, self.ax = self.plt.subplots(2, 3, figsize=(12, 7))
            self.ax = self.ax.flatten()
            self.fig.tight_layout(pad=2.0)
            
            # 0: Train Loss
            self.l1, = self.ax[0].plot([], [], label='Train')
            self.ax[0].set_title("Train loss")
            
            # 1: Val Loss
            self.l2, = self.ax[1].plot([], [], label='Val', color='orange')
            self.ax[1].set_title("Val loss")
            
            # 2: Entropy
            self.l3, = self.ax[2].plot([], [], color='green')
            self.ax[2].set_title("Attn entropy")
            
            # 3: Decode Block
            self.l4, = self.ax[3].plot([], [], marker='o', linestyle='-', color='purple')
            self.ax[3].set_title("Decode Block Size")
            
            # 4: Fused Mode
            self.l5, = self.ax[4].plot([], [], marker='x', linestyle='None', color='red')
            self.ax[4].set_title("Fused Mode (0=none, 1=1p, 2=2p)")
            self.ax[4].set_ylim(-0.5, 2.5)
            self.ax[4].set_yticks([0, 1, 2])
            self.ax[4].set_yticklabels(["none", "1pass", "2pass"])

            self.fig.show()
            self.enabled = True
        except Exception as e:
            print(f"[warn] Live plot disabled: {e}")
            self.enabled = False

    def maybe_update(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        et = str(event.get("type", ""))
        step = int(event.get("step", 0))
        
        updated = False
        if et == "train" and "loss" in event:
            self.steps.append(step)
            self.train_loss.append(_safe_float(event["loss"]))
            updated = True
        elif et == "eval" and "val_loss" in event:
            # Align val points to same step index array by appending.
            self.val_loss.append(_safe_float(event["val_loss"]))
            updated = True
        elif et == "analysis":
            if "attn_entropy_mean" in event:
                self.entropy.append(_safe_float(event["attn_entropy_mean"]))
                updated = True
            
            if event.get("subtype") == "selfopt_decode":
                # Handle self-optimization events
                # bucket_key might be useful but for now just plot the latest decision
                self.opt_steps.append(len(self.opt_steps)) # Simple index for x-axis
                self.opt_block.append(int(event.get("decode_block", 0)))
                
                fstr = str(event.get("fused", "none"))
                fval = 0
                if "1pass" in fstr: fval = 1
                elif "2pass" in fstr: fval = 2
                self.opt_fused.append(fval)
                updated = True

        if not updated:
            return

        # Update every time we get something new.
        try:
            self.l1.set_data(self.steps, self.train_loss)
            self.ax[0].relim(); self.ax[0].autoscale_view()
            
            if self.val_loss:
                xs = self.steps[-len(self.val_loss):] if len(self.val_loss) <= len(self.steps) else list(range(len(self.val_loss)))
                self.l2.set_data(xs, self.val_loss)
                self.ax[1].relim(); self.ax[1].autoscale_view()
            
            if self.entropy:
                xs = self.steps[-len(self.entropy):] if len(self.entropy) <= len(self.steps) else list(range(len(self.entropy)))
                self.l3.set_data(xs, self.entropy)
                self.ax[2].relim(); self.ax[2].autoscale_view()
                
            if self.opt_block:
                self.l4.set_data(self.opt_steps, self.opt_block)
                self.ax[3].relim(); self.ax[3].autoscale_view()
                
            if self.opt_fused:
                self.l5.set_data(self.opt_steps, self.opt_fused)
                # Fixed Y limits for categorical, just scale X
                self.ax[4].set_xlim(0, max(len(self.opt_steps), 10))
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self.plt.ioff()
            self.plt.close(self.fig)
        except Exception:
            pass


# -----------------------------
# Training utilities (v28)
# -----------------------------

def _parse_two_floats(s: str, default: Tuple[float, float]) -> Tuple[float, float]:
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except Exception:
        return default


def _supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    """
    Cheap feature probe: try a tiny op on the target device.
    (Some backends claim support but will error at runtime for specific dtypes/ops.)
    """
    try:
        x = torch.ones(8, device=device, dtype=dtype)
        y = (x * 1.0001).sum()
        # Trigger execution
        _ = float(y.detach().to("cpu").item())
        return True
    except Exception:
        return False


def resolve_dtype(device: torch.device, spec: str, *, default: torch.dtype) -> torch.dtype:
    spec = str(spec).lower()
    if spec in ("fp32", "float32", "f32"):
        return torch.float32
    if spec in ("bf16", "bfloat16"):
        dt = torch.bfloat16
    elif spec in ("fp16", "float16", "f16"):
        dt = torch.float16
    else:
        dt = default

    # Backend reality check
    if device.type in ("cuda", "mps"):
        if dt in (torch.float16, torch.bfloat16) and not _supports_dtype(device, dt):
            # last resort: fp16 -> bf16 -> fp32
            if dt == torch.float16 and _supports_dtype(device, torch.bfloat16):
                return torch.bfloat16
            return torch.float32
    if device.type == "cpu" and dt == torch.float16:
        # CPU float16 is generally painful / unsupported for many ops.
        return torch.float32
    return dt


def device_synchronize(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except Exception:
        pass


def get_device_mem_stats(device: torch.device) -> Dict[str, float]:
    """
    Returns memory stats in bytes where possible.
    Keys are backend-specific to avoid implying false equivalence.
    """
    out: Dict[str, float] = {}
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            out["cuda_mem_alloc_bytes"] = float(torch.cuda.memory_allocated(device))
            out["cuda_mem_reserved_bytes"] = float(torch.cuda.memory_reserved(device))
            out["cuda_mem_peak_bytes"] = float(torch.cuda.max_memory_allocated(device))
        elif device.type == "mps":
            if hasattr(torch, "mps"):
                if hasattr(torch.mps, "current_allocated_memory"):
                    out["mps_mem_alloc_bytes"] = float(torch.mps.current_allocated_memory())
                if hasattr(torch.mps, "driver_allocated_memory"):
                    out["mps_mem_driver_bytes"] = float(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    return out


def get_process_rss_bytes() -> Optional[int]:
    """
    Best-effort RSS (resident set size).
    """
    # psutil is nicest if available.
    try:
        import psutil  # type: ignore
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    # Fallback: resource.getrusage
    try:
        import resource  # type: ignore
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB, macOS: bytes
        if sys.platform == "darwin":
            return int(rss)
        return int(rss) * 1024
    except Exception:
        return None


class Lion(torch.optim.Optimizer):
    """
    Lion optimizer (https://arxiv.org/abs/2302.06675).

    Memory win vs AdamW: ~1 momentum state instead of 2 (m, v).
    Often competitive for transformers, especially when you crank scale on a single device.

    This implementation uses decoupled weight decay.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                # First momentum update
                exp_avg.mul_(beta1).add_(g, alpha=(1.0 - beta1))
                # Sign update
                p.add_(exp_avg.sign(), alpha=-lr)
                # Second momentum update
                exp_avg.mul_(beta2).add_(g, alpha=(1.0 - beta2))

        return loss


def lr_for_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    schedule: str,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> float:
    schedule = str(schedule).lower()
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    if schedule == "constant":
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        return base_lr

    if schedule == "cosine":
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        # cosine decay from base_lr -> min_lr
        denom = max(total_steps - warmup_steps, 1)
        t = (step - warmup_steps) / denom
        t = min(max(t, 0.0), 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    # fallback
    return base_lr


def parse_seq_schedule(spec: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """
    "256@0,512@1000,1024@3000" -> [(0,256),(1000,512),(3000,1024)] sorted by step.
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec:
        return None
    pairs: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "@" not in part:
            continue
        a, b = part.split("@", 1)
        try:
            seq = int(a)
            st = int(b)
            pairs.append((st, seq))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs if pairs else None


def seq_len_for_step(
    step: int,
    *,
    default_seq_len: int,
    schedule: Optional[List[Tuple[int, int]]],
) -> int:
    if not schedule:
        return int(default_seq_len)
    cur = int(default_seq_len)
    for st, seq in schedule:
        if step >= st:
            cur = int(seq)
    return int(cur)


def optimizer_state_bytes(opt: torch.optim.Optimizer) -> int:
    total = 0
    try:
        for st in opt.state.values():
            if isinstance(st, dict):
                for v in st.values():
                    if torch.is_tensor(v):
                        total += v.numel() * v.element_size()
    except Exception:
        return -1
    return int(total)


class LiveDashboard:
    """
    Console "live view" for training: rich dashboard when available, with a basic fallback.

    Goal: see *everything that matters* without drowning in logs.
    """
    def __init__(
        self,
        mode: str,
        *,
        total_steps: int,
        out_dir: str,
        cfg: ModelConfig,
        args: argparse.Namespace,
        device: torch.device,
    ):
        self.mode = str(mode).lower()
        self.total_steps = int(total_steps)
        self.out_dir = str(out_dir)
        self.cfg = cfg
        self.args = args
        self.device = device

        self.last_train: Dict[str, Any] = {}
        self.last_eval: Dict[str, Any] = {}
        self.last_msg: str = ""
        self.enabled = (self.mode != "off")

        self._rich: Optional[Dict[str, Any]] = None
        self._console = None
        self._live = None
        self._progress = None
        self._task_id = None

        # Determine backend
        want_rich = self.mode in ("auto", "rich")
        use_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        if self.enabled and want_rich and use_tty:
            try:
                from rich.console import Console  # type: ignore
                from rich.live import Live  # type: ignore
                from rich.table import Table  # type: ignore
                from rich.panel import Panel  # type: ignore
                from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn  # type: ignore
                from rich.columns import Columns  # type: ignore
                from rich.layout import Layout  # type: ignore
                from rich.text import Text  # type: ignore
                from rich.align import Align  # type: ignore
                from rich import box  # type: ignore

                self._rich = {
                    "Table": Table,
                    "Panel": Panel,
                    "Progress": Progress,
                    "Columns": Columns,
                    "Layout": Layout,
                    "Text": Text,
                    "Align": Align,
                    "Live": Live,
                    "Console": Console,
                    "BarColumn": BarColumn,
                    "TextColumn": TextColumn,
                    "TimeElapsedColumn": TimeElapsedColumn,
                    "TimeRemainingColumn": TimeRemainingColumn,
                    "box": box,
                }
                self._console = Console()
                self._progress = Progress(
                    TextColumn("[bold]train[/bold]"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=False,
                )
                self._task_id = self._progress.add_task("steps", total=self.total_steps)
                self._live = Live(refresh_per_second=8, console=self._console)
                self._live.start()
            except Exception:
                # rich not available -> basic mode
                self._rich = None
                self._console = None
                self._live = None
                self._progress = None
                self._task_id = None

        if self.enabled and self._rich is None and self.mode == "rich":
            # user explicitly requested rich but it failed: keep basic fallback enabled
            self.enabled = True

    def close(self) -> None:
        try:
            if self._live is not None:
                self._live.stop()
        except Exception:
            pass

    def update_train(self, metrics: Dict[str, Any]) -> None:
        self.last_train = dict(metrics)
        self._refresh()

    def update_eval(self, metrics: Dict[str, Any]) -> None:
        self.last_eval = dict(metrics)
        self._refresh()

    def message(self, msg: str) -> None:
        self.last_msg = str(msg)
        if self._console is not None:
            try:
                self._console.log(self.last_msg)
            except Exception:
                pass
        else:
            print(self.last_msg)

    def _refresh(self) -> None:
        if not self.enabled:
            return
        if self._rich is None:
            # basic fallback
            if self.last_train and ("step" in self.last_train):
                s = int(self.last_train.get("step", 0))
                loss = self.last_train.get("loss", float("nan"))
                tok_s = self.last_train.get("tok_s", float("nan"))
                lr = self.last_train.get("lr", float("nan"))
                mem = ""
                if "cuda_mem_alloc_bytes" in self.last_train:
                    mem = f" cuda_alloc={human_bytes(int(self.last_train['cuda_mem_alloc_bytes']))}"
                if "mps_mem_alloc_bytes" in self.last_train:
                    mem = f" mps_alloc={human_bytes(int(self.last_train['mps_mem_alloc_bytes']))}"
                print(f"[step {s}] loss={loss:.4f} lr={lr:.2e} tok/s={tok_s:.0f}{mem}")
            return

        # rich render
        try:
            Table = self._rich["Table"]
            Panel = self._rich["Panel"]
            Columns = self._rich["Columns"]
            Text = self._rich["Text"]
            Align = self._rich["Align"]
            box = self._rich["box"]

            # progress
            if self._progress is not None and self._task_id is not None:
                try:
                    s = int(self.last_train.get("step", 0)) if self.last_train else 0
                    self._progress.update(self._task_id, completed=min(s, self.total_steps))
                except Exception:
                    pass

            def _kv(title: str, items: List[Tuple[str, Any]]) -> Any:
                t = Table(title=title, box=box.SIMPLE, show_header=False, pad_edge=False)
                t.add_column("k", style="bold")
                t.add_column("v")
                for k, v in items:
                    t.add_row(str(k), str(v))
                return t

            train = self.last_train
            ev = self.last_eval

            # Left: core train metrics
            items1: List[Tuple[str, Any]] = []
            if train:
                items1 += [
                    ("step", train.get("step", "")),
                    ("loss", f"{train.get('loss', float('nan')):.6f}" if "loss" in train else ""),
                    ("ppl", f"{train.get('ppl', float('nan')):.2f}" if "ppl" in train else ""),
                    ("lr", f"{train.get('lr', 0.0):.3e}" if "lr" in train else ""),
                    ("grad_norm", f"{train.get('grad_norm_total', float('nan')):.3f}" if "grad_norm_total" in train else ""),
                    ("scale", f"{train.get('amp_scale', '')}" if "amp_scale" in train else ""),
                ]

            # Middle: perf
            items2: List[Tuple[str, Any]] = []
            if train:
                items2 += [
                    ("tok/s", f"{train.get('tok_s', 0.0):.0f}" if "tok_s" in train else ""),
                    ("ms/step", f"{train.get('step_ms', 0.0):.2f}" if "step_ms" in train else ""),
                    ("data ms", f"{train.get('data_ms', 0.0):.2f}" if "data_ms" in train else ""),
                    ("fwd ms", f"{train.get('fwd_ms', 0.0):.2f}" if "fwd_ms" in train else ""),
                    ("bwd ms", f"{train.get('bwd_ms', 0.0):.2f}" if "bwd_ms" in train else ""),
                    ("opt ms", f"{train.get('opt_ms', 0.0):.2f}" if "opt_ms" in train else ""),
                    ("seq_len", f"{int(train.get('seq_len', 0))}" if "seq_len" in train else ""),
                    ("gbs", f"{int(train.get('global_batch', 0))}" if "global_batch" in train else ""),
                ]

            # Right: memory
            items3: List[Tuple[str, Any]] = []
            rss = train.get("cpu_rss_bytes", None) if train else None
            if rss is not None:
                items3.append(("cpu rss", human_bytes(int(rss))))
            if train:
                if "cuda_mem_alloc_bytes" in train:
                    items3 += [
                        ("cuda alloc", human_bytes(int(train["cuda_mem_alloc_bytes"]))),
                        ("cuda reserv", human_bytes(int(train.get("cuda_mem_reserved_bytes", 0)))),
                        ("cuda peak", human_bytes(int(train.get("cuda_mem_peak_bytes", 0)))),
                    ]
                if "mps_mem_alloc_bytes" in train:
                    items3 += [
                        ("mps alloc", human_bytes(int(train["mps_mem_alloc_bytes"]))),
                    ]
                if "mps_mem_driver_bytes" in train:
                    items3 += [
                        ("mps driver", human_bytes(int(train["mps_mem_driver_bytes"]))),
                    ]
                if "opt_state_bytes" in train and int(train.get("opt_state_bytes", -1)) >= 0:
                    items3 += [
                        ("opt state", human_bytes(int(train["opt_state_bytes"]))),
                    ]

            # Eval panel
            ev_items: List[Tuple[str, Any]] = []
            if ev:
                ev_items += [
                    ("val_loss", f"{ev.get('val_loss', float('nan')):.6f}" if "val_loss" in ev else ""),
                    ("val_ppl", f"{ev.get('val_ppl', float('nan')):.2f}" if "val_ppl" in ev else ""),
                    ("best_val", f"{ev.get('best_val', float('nan')):.6f}" if "best_val" in ev else ""),
                ]

            cols = Columns(
                [
                    Panel(_kv("train", items1), padding=(0, 1)),
                    Panel(_kv("perf", items2), padding=(0, 1)),
                    Panel(_kv("mem", items3), padding=(0, 1)),
                ],
                expand=True,
            )

            footer_text = self.last_msg or ""
            if ev_items:
                footer = Columns([Panel(_kv("eval", ev_items), padding=(0, 1)), Panel(Text(footer_text), padding=(0, 1))])
            else:
                footer = Panel(Text(footer_text), padding=(0, 1))

            # Vertical layout: progress (top), metrics (middle), footer (bottom)
            Layout = self._rich["Layout"]
            layout = Layout()
            layout.split_column(
                Layout(Panel(self._progress), name="progress", size=3),
                Layout(Panel(cols), name="main", ratio=1),
                Layout(footer, name="footer", size=7),
            )
            if self._live is not None:
                self._live.update(layout)
        except Exception:
            pass



def compute_grad_norms(model: nn.Module) -> Dict[str, float]:
    """
    Gradient norms grouped into {embed, attn, ffn} plus optional sem/geo for decoupled.
    """
    sums: Dict[str, float] = {"embed": 0.0, "attn": 0.0, "ffn": 0.0, "attn_sem": 0.0, "attn_geo": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        # Keep this cheap; avoid float64.
        g2 = float(g.float().pow(2).sum().item())
        if name.startswith("tok_emb") or name.startswith("emb_in") or name.startswith("emb_out"):
            sums["embed"] += g2
        elif ".attn." in name:
            sums["attn"] += g2
            if any(s in name for s in (".q_sem.", ".k_sem.")):
                sums["attn_sem"] += g2
            if any(s in name for s in (".q_geo.", ".k_geo.")):
                sums["attn_geo"] += g2
        elif ".ff." in name:
            sums["ffn"] += g2
    out: Dict[str, float] = {}
    total = 0.0
    for k, v in sums.items():
        out[f"grad_norm_{k}"] = math.sqrt(max(v, 0.0))
        total += float(v)
    out["grad_norm_total"] = math.sqrt(max(total, 0.0))
    return out


def _power_iter_spectral_norm(W: torch.Tensor, iters: int = 8) -> float:
    """
    Estimates spectral norm ||W||_2 by power iteration.
    Works on CPU for portability (MPS has gaps for some linalg ops).
    """
    try:
        A = W.detach().float().cpu()
        if A.ndim != 2:
            A = A.view(A.shape[0], -1)
        m, n = A.shape
        # start vector
        v = torch.randn(n)
        v = v / (v.norm() + 1e-9)
        for _ in range(max(1, iters)):
            u = A @ v
            u = u / (u.norm() + 1e-9)
            v = A.t() @ u
            v = v / (v.norm() + 1e-9)
        sigma = float((u @ (A @ v)).abs().item())
        return sigma
    except Exception:
        return float("nan")


def stable_rank(W: torch.Tensor) -> float:
    """
    Stable rank = ||W||_F^2 / ||W||_2^2 (robust proxy for effective rank).
    """
    try:
        A = W.detach().float().cpu()
        if A.ndim != 2:
            A = A.view(A.shape[0], -1)
        fro2 = float(A.pow(2).sum().item())
        spec = _power_iter_spectral_norm(A, iters=8)
        denom = max(spec * spec, 1e-12)
        return float(fro2 / denom)
    except Exception:
        return float("nan")


@torch.no_grad()
def analyze_attention(
    model: GPT,
    idx: torch.Tensor,
    *,
    layers: List[int],
    heads: List[int],
    max_tokens: int,
    topk: int,
    local_window: int,
    save_mats: bool,
    save_scores: bool,
    compute_svd: bool,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Runs a lightweight analysis pass that:
      - captures ln1(x) for selected layers
      - recomputes attention probabilities explicitly (so we can measure entropy / sparsity / rank)
      - optionally returns attention matrices + singular values for HDF5 persistence
    """
    was_training = model.training
    model.eval()

    # Trim to analysis window
    B, T = idx.shape
    T = min(int(T), int(max_tokens))
    idx = idx[:, :T]

    device = idx.device
    ninfty = neg_inf(torch.float32)

    # causal mask: (1,1,T,T)
    causal = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool)).view(1, 1, T, T)

    # capture ln1 outputs for requested layers
    acts: Dict[int, torch.Tensor] = {}
    hooks: List[Any] = []

    def _resolve_layer(i: int) -> int:
        return i if i >= 0 else (model.cfg.n_layer + i)

    layer_ids = sorted(set(_resolve_layer(i) for i in layers))
    for li in layer_ids:
        if li < 0 or li >= model.cfg.n_layer:
            continue
        def _make_hook(layer_index: int):
            def hook(module, inp, out):  # noqa: ANN001
                # out: (B,T,d_model)
                acts[layer_index] = out.detach()
            return hook
        hooks.append(model.blocks[li].ln1.register_forward_hook(_make_hook(li)))

    # Run a forward to populate acts (cheap; no grads).
    _ = model(idx)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # Stats accumulators
    entropies: List[float] = []
    topk_masses: List[float] = []
    null_masses: List[float] = []
    local_masses: List[float] = []
    sem_ratios: List[float] = []
    geo_ratios: List[float] = []
    eranks: List[float] = []

    tensors_out: Dict[str, torch.Tensor] = {}

    # For weight stable rank (projection matrices) on selected layers
    w_sr: List[float] = []

    for li, x_ln in acts.items():
        blk = model.blocks[li]
        attn = blk.attn
        cfg = attn.cfg

        # Estimate stable rank of the attention projection(s) (proxy for "effective rank of attention space")
        try:
            if cfg.attn_mode in ("standard", "bottleneck", "gqa"):
                w_sr.append(stable_rank(attn.q_proj.weight))
            else:
                w_sr.append(stable_rank(attn.q_sem.weight))
                w_sr.append(stable_rank(attn.q_geo.weight))
        except Exception:
            pass

        # Compute attention probs (explicit softmax) in fp32
        x_ln_f = x_ln.float()
        B2, T2, _ = x_ln_f.shape
        assert T2 == T

        if cfg.attn_mode in ("standard", "bottleneck"):
            q = attn.q_proj(x_ln_f)
            k = attn.k_proj(x_ln_f)
            v = attn.v_proj(x_ln_f)

            qh = attn._shape(q, attn.qk_head_dim)
            kh = attn._shape(k, attn.qk_head_dim)
            vh = attn._shape(v, attn.v_head_dim)

            if attn.rotary is not None:
                qh = attn.rotary.rotate(qh, 0)
                kh = attn.rotary.rotate(kh, 0)
            qh = attn._apply_logit_scale_to_q(qh)

            scale = 1.0 / math.sqrt(attn.qk_head_dim)
            scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale  # (B,H,T,T)
            scores = scores.masked_fill(~causal, ninfty)

            if cfg.null_attn:
                k_null = attn._shape(attn.k_null.expand(B2, 1, -1), attn.qk_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) * scale  # (B,H,T,1)
                scores = torch.cat([s_null, scores], dim=-1)
                keep = torch.cat([torch.ones((1, 1, T, 1), device=device, dtype=torch.bool), causal], dim=-1)
                scores = scores.masked_fill(~keep, ninfty)

            p = F.softmax(scores, dim=-1)  # (B,H,T,K)

            # local window mass (how much attention stays near the diagonal)
            if local_window > 0:
                w = int(local_window)
                # key positions for each row
                key_pos = torch.arange(p.size(-1), device=device)
                if cfg.null_attn:
                    # skip null key at index 0
                    key_pos = key_pos - 1
                q_pos = torch.arange(T, device=device).view(T, 1)
                # allow keys in [q-w, q] (causal + local)
                lo = (q_pos - w).clamp(min=0)
                hi = q_pos
                # mask: (T,K)
                m_local = (key_pos.view(1, -1) >= lo) & (key_pos.view(1, -1) <= hi)
                # broadcast to (B,H,T,K)
                m_local_b = m_local.view(1, 1, T, -1)
                local_mass = (p * m_local_b.float()).sum(dim=-1).mean().item()
                local_masses.append(float(local_mass))

            # entropy + topk mass
            p_cl = p.clamp(min=1e-9)
            ent = (-(p_cl * p_cl.log()).sum(dim=-1)).mean().item()
            entropies.append(float(ent))
            k = min(int(topk), p.size(-1))
            top_mass = p.topk(k, dim=-1).values.sum(dim=-1).mean().item()
            topk_masses.append(float(top_mass))
            if cfg.null_attn:
                null_masses.append(float(p[..., 0].mean().item()))

            # effective rank via SVD on selected heads (using real-token square matrix)
            if compute_svd and heads:
                for h in heads:
                    hh = int(h)
                    if hh < 0 or hh >= attn.H:
                        continue
                    A = p[0, hh, :, 1:] if cfg.null_attn else p[0, hh, :, :]
                    A_cpu = A.detach().float().cpu()
                    try:
                        s = torch.linalg.svdvals(A_cpu)
                        s_sum = float(s.sum().item()) + 1e-12
                        ps = (s / s_sum).clamp(min=1e-12)
                        er = float(torch.exp(-(ps * ps.log()).sum()).item())
                        eranks.append(er)
                        if save_mats:
                            tensors_out[f"attn/L{li}/H{hh}"] = A.detach().to(torch.float16)
                        if save_mats:
                            tensors_out[f"svd/L{li}/H{hh}"] = s.detach().to(torch.float32)
                    except Exception:
                        pass

        elif cfg.attn_mode == "gqa":
            q = attn.q_proj(x_ln_f)
            k = attn.k_proj(x_ln_f)
            v = attn.v_proj(x_ln_f)

            qh = attn._shape(q, attn.qk_head_dim, H=attn.H)
            kh = attn._shape(k, attn.qk_head_dim, H=attn.H_kv)
            vh = attn._shape(v, attn.v_head_dim, H=attn.H_kv)

            if attn.rotary is not None:
                qh = attn.rotary.rotate(qh, 0)
                kh = attn.rotary.rotate(kh, 0)
            qh = attn._apply_logit_scale_to_q(qh)

            kh_rep = kh.repeat_interleave(attn.group_size, dim=1)
            vh_rep = vh.repeat_interleave(attn.group_size, dim=1)

            scale = 1.0 / math.sqrt(attn.qk_head_dim)
            scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) * scale
            scores = scores.masked_fill(~causal, ninfty)

            if cfg.null_attn:
                k_null = attn._shape(attn.k_null.expand(B2, 1, -1), attn.qk_head_dim, H=attn.H_kv)
                k_null_rep = k_null.repeat_interleave(attn.group_size, dim=1)
                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) * scale
                scores = torch.cat([s_null, scores], dim=-1)
                keep = torch.cat([torch.ones((1, 1, T, 1), device=device, dtype=torch.bool), causal], dim=-1)
                scores = scores.masked_fill(~keep, ninfty)

            p = F.softmax(scores, dim=-1)

            p_cl = p.clamp(min=1e-9)
            entropies.append(float((-(p_cl * p_cl.log()).sum(dim=-1)).mean().item()))
            ktop = min(int(topk), p.size(-1))
            topk_masses.append(float(p.topk(ktop, dim=-1).values.sum(dim=-1).mean().item()))
            if cfg.null_attn:
                null_masses.append(float(p[..., 0].mean().item()))

            if compute_svd and heads:
                for h in heads:
                    hh = int(h)
                    if hh < 0 or hh >= attn.H:
                        continue
                    A = p[0, hh, :, 1:] if cfg.null_attn else p[0, hh, :, :]
                    A_cpu = A.detach().float().cpu()
                    try:
                        s = torch.linalg.svdvals(A_cpu)
                        s_sum = float(s.sum().item()) + 1e-12
                        ps = (s / s_sum).clamp(min=1e-12)
                        er = float(torch.exp(-(ps * ps.log()).sum()).item())
                        eranks.append(er)
                        if save_mats:
                            tensors_out[f"attn/L{li}/H{hh}"] = A.detach().to(torch.float16)
                            tensors_out[f"svd/L{li}/H{hh}"] = s.detach().to(torch.float32)
                    except Exception:
                        pass

        elif cfg.attn_mode == "decoupled":
            q_sem = attn.q_sem(x_ln_f)
            k_sem = attn.k_sem(x_ln_f)
            q_geo = attn.q_geo(x_ln_f)
            k_geo = attn.k_geo(x_ln_f)
            v = attn.v_proj(x_ln_f)

            qsh = attn._shape(q_sem, attn.sem_head_dim)
            ksh = attn._shape(k_sem, attn.sem_head_dim)
            qgh = attn._shape(q_geo, attn.geo_head_dim)
            kgh = attn._shape(k_geo, attn.geo_head_dim)
            vh = attn._shape(v, attn.v_head_dim)

            if attn.rotary is not None:
                qgh = attn.rotary.rotate(qgh, 0)
                kgh = attn.rotary.rotate(kgh, 0)

            qsh = attn._apply_logit_scale_to_q(qsh)
            qgh = attn._apply_logit_scale_to_q(qgh)

            sem_scale = 1.0 / math.sqrt(attn.sem_head_dim)
            geo_scale = 1.0 / math.sqrt(attn.geo_head_dim)

            sem_logits = torch.matmul(qsh, ksh.transpose(-2, -1)) * sem_scale
            geo_logits = torch.matmul(qgh, kgh.transpose(-2, -1)) * geo_scale
            scores = sem_logits + geo_logits
            scores = scores.masked_fill(~causal, ninfty)

            if cfg.null_attn:
                ksn = attn._shape(attn.k_sem_null.expand(B2, 1, -1), attn.sem_head_dim)
                kgn = attn._shape(attn.k_geo_null.expand(B2, 1, -1), attn.geo_head_dim)
                s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
                scores = torch.cat([s_null, scores], dim=-1)
                keep = torch.cat([torch.ones((1, 1, T, 1), device=device, dtype=torch.bool), causal], dim=-1)
                scores = scores.masked_fill(~keep, ninfty)

            p = F.softmax(scores, dim=-1)

            # sem/geo energy ratio
            try:
                sem_e = float(sem_logits.float().pow(2).mean().item())
                geo_e = float(geo_logits.float().pow(2).mean().item())
                denom = sem_e + geo_e + 1e-12
                sem_ratios.append(sem_e / denom)
                geo_ratios.append(geo_e / denom)
            except Exception:
                pass

            p_cl = p.clamp(min=1e-9)
            entropies.append(float((-(p_cl * p_cl.log()).sum(dim=-1)).mean().item()))
            ktop = min(int(topk), p.size(-1))
            topk_masses.append(float(p.topk(ktop, dim=-1).values.sum(dim=-1).mean().item()))
            if cfg.null_attn:
                null_masses.append(float(p[..., 0].mean().item()))

            # local mass: restrict to last local_window real tokens (skip null)
            if local_window > 0:
                w = int(local_window)
                key_pos = torch.arange(p.size(-1), device=device)
                if cfg.null_attn:
                    key_pos = key_pos - 1
                q_pos = torch.arange(T, device=device).view(T, 1)
                lo = (q_pos - w).clamp(min=0)
                hi = q_pos
                m_local = (key_pos.view(1, -1) >= lo) & (key_pos.view(1, -1) <= hi)
                m_local_b = m_local.view(1, 1, T, -1)
                local_mass = p.masked_select(m_local_b).view(B2, attn.H, T, -1).sum(dim=-1).mean().item()
                local_masses.append(float(local_mass))

            if compute_svd and heads:
                for h in heads:
                    hh = int(h)
                    if hh < 0 or hh >= attn.H:
                        continue
                    A = p[0, hh, :, 1:] if cfg.null_attn else p[0, hh, :, :]
                    A_cpu = A.detach().float().cpu()
                    try:
                        s = torch.linalg.svdvals(A_cpu)
                        s_sum = float(s.sum().item()) + 1e-12
                        ps = (s / s_sum).clamp(min=1e-12)
                        er = float(torch.exp(-(ps * ps.log()).sum()).item())
                        eranks.append(er)
                        if save_mats:
                            tensors_out[f"attn/L{li}/H{hh}"] = A.detach().to(torch.float16)
                            tensors_out[f"svd/L{li}/H{hh}"] = s.detach().to(torch.float32)
                        if save_scores:
                            # Store the raw component scores (real tokens only).
                            s_sem = sem_logits[0, hh, :, :].detach().to(torch.float16).cpu()
                            s_geo = geo_logits[0, hh, :, :].detach().to(torch.float16).cpu()
                            tensors_out[f"sem_scores/L{li}/H{hh}"] = s_sem
                            tensors_out[f"geo_scores/L{li}/H{hh}"] = s_geo
                    except Exception:
                        pass
        else:
            continue

    # summarize
    out: Dict[str, float] = {}
    if entropies:
        out["attn_entropy_mean"] = float(sum(entropies) / len(entropies))
    if topk_masses:
        out["attn_topk_mass_mean"] = float(sum(topk_masses) / len(topk_masses))
    if null_masses:
        out["attn_null_mass_mean"] = float(sum(null_masses) / len(null_masses))
    if local_masses:
        out["attn_local_mass_mean"] = float(sum(local_masses) / len(local_masses))
    if sem_ratios:
        out["sem_energy_ratio_mean"] = float(sum(sem_ratios) / len(sem_ratios))
    if geo_ratios:
        out["geo_energy_ratio_mean"] = float(sum(geo_ratios) / len(geo_ratios))
    if eranks:
        out["attn_erank_mean"] = float(sum(eranks) / len(eranks))
    if w_sr:
        out["proj_stable_rank_mean"] = float(sum(w_sr) / len(w_sr))

    if was_training:
        model.train()
    return out, tensors_out


def generate_analysis_png(jsonl_path: str, out_png: str) -> None:
    """
    Small, opinionated end-of-run plot pack -> analysis.png.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib not available: {e}")

    # Read metrics
    steps_train: List[int] = []
    loss_train: List[float] = []
    tok_s: List[float] = []

    steps_eval: List[int] = []
    loss_val: List[float] = []
    loss_train_eval: List[float] = []

    steps_an: List[int] = []
    ent: List[float] = []
    erank: List[float] = []
    topk_mass: List[float] = []
    sem_ratio: List[float] = []
    local_mass: List[float] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = str(e.get("type", ""))
            s = int(e.get("step", 0))
            if t == "train":
                if "loss" in e:
                    steps_train.append(s)
                    loss_train.append(_safe_float(e["loss"]))
                if "tok_s" in e:
                    tok_s.append(_safe_float(e["tok_s"]))
            elif t == "eval":
                steps_eval.append(s)
                if "val_loss" in e:
                    loss_val.append(_safe_float(e["val_loss"]))
                if "train_loss" in e:
                    loss_train_eval.append(_safe_float(e["train_loss"]))
            elif t == "analysis":
                steps_an.append(s)
                if "attn_entropy_mean" in e:
                    ent.append(_safe_float(e["attn_entropy_mean"]))
                if "attn_erank_mean" in e:
                    erank.append(_safe_float(e["attn_erank_mean"]))
                if "attn_topk_mass_mean" in e:
                    topk_mass.append(_safe_float(e["attn_topk_mass_mean"]))
                if "sem_energy_ratio_mean" in e:
                    sem_ratio.append(_safe_float(e["sem_energy_ratio_mean"]))
                if "attn_local_mass_mean" in e:
                    local_mass.append(_safe_float(e["attn_local_mass_mean"]))

    # Make plots
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    fig.tight_layout(pad=2.0)

    ax[0, 0].plot(steps_train, loss_train)
    ax[0, 0].set_title("Train loss")

    if steps_eval and loss_val:
        ax[0, 1].plot(steps_eval[:len(loss_val)], loss_val)
        ax[0, 1].set_title("Val loss")

    if tok_s:
        ax[1, 0].plot(range(len(tok_s)), tok_s)
        ax[1, 0].set_title("Throughput (tok/s) samples")

    if steps_an and ent:
        ax[1, 1].plot(steps_an[:len(ent)], ent)
        ax[1, 1].set_title("Attention entropy (mean)")

    if steps_an and erank:
        ax[2, 0].plot(steps_an[:len(erank)], erank)
        ax[2, 0].set_title("Attention effective rank (mean)")

    if steps_an and topk_mass:
        ax[2, 1].plot(steps_an[:len(topk_mass)], topk_mass, label="topk mass")
        if sem_ratio:
            ax[2, 1].plot(steps_an[:len(sem_ratio)], sem_ratio, label="semantic energy ratio")
        if local_mass:
            ax[2, 1].plot(steps_an[:len(local_mass)], local_mass, label="local mass")
        ax[2, 1].set_title("Attention diagnostics")
        ax[2, 1].legend()

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()

    # ---- Experiment suite controls (new in v27) ----
    ap.add_argument("--size", type=str, default=None, choices=list(SIZE_PRESETS.keys()),
                    help="Preset model+train size (tiny/small/medium/large). Applies only when provided.")
    ap.add_argument("--exp", type=str, default=None,
                    choices=["paper_baseline", "paper_bottleneck", "paper_decoupled", "paper_gqa", "paper_all"],
                    help="Preset experiment configuration matching the paper suite. Applies only when provided.")
    ap.add_argument("--run-root", type=str, default="runs", help="Root directory for auto run dirs when --out-dir is omitted.")
    ap.add_argument("--run-tag", type=str, default=None, help="Optional suffix for auto run dirs, e.g. 'seed2' -> runs/small_decoupled_seed2")
    ap.add_argument("--print-config", action="store_true", help="Print resolved config (after presets/overrides) and exit.")

    # ---- Core I/O ----
    ap.add_argument("--data", type=str, default=None, help="Token dataset path. For train mode only.")
    ap.add_argument("--data-format", type=str, default="auto", choices=["auto", "text", "npy", "bin", "pt"],
                    help="Dataset format. 'text' expects whitespace-separated ints. 'npy' uses np.load(mmap). 'bin' uses np.memmap. 'pt' loads a torch tensor.")
    ap.add_argument("--data-dtype", type=str, default="uint16",
                    help="For --data-format bin: numpy dtype (e.g. uint16, uint32, int32).")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="Explicit vocab size (recommended for binary/mmap datasets). If omitted and tokenizer=tiktoken, defaults to 50257.")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (tail split).")

    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)

    # ---- Compile / AMP ----
    ap.add_argument("--compile", action="store_true", help="Use torch.compile(...) for speed (experimental).")
    ap.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode (if --compile).")
    ap.add_argument("--amp", action="store_true", help="Enable torch.amp autocast (mixed precision) for training on CUDA/MPS/CPU (experimental).")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                    help="Autocast compute dtype. bf16 is usually safest; fp16 may require loss scaling.")
    ap.add_argument("--param-dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="Model parameter dtype. fp32 baseline; bf16/fp16 reduce memory ~2x (helpful for bigger models).")
    ap.add_argument("--matmul-precision", type=str, default="high", choices=["highest", "high", "medium"],
                    help="torch.set_float32_matmul_precision(...) hint for float32 matmuls (may improve speed).")

    # ---- Model ----
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=512)

    ap.add_argument("--attn-mode", type=str, default="bottleneck", choices=["standard", "bottleneck", "decoupled", "gqa"])
    ap.add_argument("--kv-head", type=int, default=None, help="For --attn-mode gqa: number of KV heads (must divide n_head). Default = n_head")
    ap.add_argument("--attn-dim", type=int, default=512)
    ap.add_argument("--sem-dim", type=int, default=32)
    ap.add_argument("--geo-dim", type=int, default=64)

    ap.add_argument("--no-rope", action="store_true")
    ap.add_argument("--rope", action="store_true", help="Force-enable RoPE even if a preset would disable it.")
    ap.add_argument("--rope-base", type=float, default=10000.0)

    ap.add_argument("--tie-qk", action="store_true")
    ap.add_argument("--no-tie-qk", action="store_true", help="Force-disable tie_qk (useful for overrides with presets).")

    ap.add_argument("--null-attn", action="store_true")
    ap.add_argument("--no-null-attn", action="store_true", help="Force-disable null_attn (useful for overrides with presets).")

    ap.add_argument("--no-learned-temp", action="store_true")

    ap.add_argument("--mlp", type=str, default="swiglu", choices=["swiglu", "gelu"])
    ap.add_argument("--dropout", type=float, default=0.0)

    # ---- Training ----
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=8, help="Micro-batch size (per optimizer step if --grad-accum=1).")
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (global batch = batch_size * grad_accum).")
    ap.add_argument("--train-seq-len", type=int, default=0, help="Effective sequence length for training batches (0 = use --block).")
    ap.add_argument("--seq-schedule", type=str, default=None, help="Optional seq-len curriculum: 'len@step,len@step,...' e.g. '256@0,512@1000,1024@3000'.")
    ap.add_argument("--eval-seq-len", type=int, default=0, help="Eval sequence length (0 = match training seq-len).")
    ap.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (recompute activations) to save memory.")
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lion"],
                    help="Optimizer. lion uses ~1 momentum state (lower memory) and can be faster on big models.")
    ap.add_argument("--adam-betas", type=str, default="0.9,0.95", help="AdamW betas as 'b1,b2'.")
    ap.add_argument("--adam-eps", type=float, default=1e-8, help="AdamW epsilon.")
    ap.add_argument("--lion-betas", type=str, default="0.9,0.99", help="Lion betas as 'b1,b2'.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"],
                    help="Learning rate schedule.")
    ap.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps for lr schedule.")
    ap.add_argument("--min-lr", type=float, default=0.0, help="Minimum lr for cosine schedule.")
    ap.add_argument("--opt-foreach", action="store_true", help="Use foreach optimizer implementation when available (can be faster).")
    ap.add_argument("--opt-fused", action="store_true", help="Use fused optimizer implementation when available (CUDA only).")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=0, help="Checkpoint interval (steps). 0 to disable.")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=100, help="Train-step logging interval (JSONL + console).")

    # ---- Instrumentation (new in v27) ----
    ap.add_argument("--instrument", type=str, default="full", choices=["off", "basic", "medium", "full"],
                    help="off: minimal. basic/medium: JSONL+summary+entropy. full: +HDF5 matrices/SVD + analysis.png")
    ap.add_argument("--analysis-every", type=int, default=100, help="Deep analysis interval. 0 disables.")
    ap.add_argument("--analysis-max-tokens", type=int, default=256, help="Max tokens for attention matrix analysis.")
    ap.add_argument("--analysis-layers", type=str, default="0,-1", help="Comma-separated layer indices to analyze (supports negatives).")
    ap.add_argument("--analysis-heads", type=str, default="0", help="Comma-separated head indices to analyze for SVD/matrix dumps.")
    ap.add_argument("--analysis-topk", type=int, default=8, help="Top-k mass metric for sparsity.")
    ap.add_argument("--analysis-local-window", type=int, default=32, help="Locality window for 'local mass' metric.")
    ap.add_argument("--analysis-save-scores", action="store_true", help="(Decoupled) Save sem/geo score matrices into analysis.h5 (bigger).")
    ap.add_argument("--live", type=str, default="auto", choices=["auto", "off", "basic", "rich"],
                    help="Console live dashboard. 'auto' uses rich if installed + TTY.")
    ap.add_argument("--live-update-every", type=int, default=1, help="Live dashboard refresh interval (optimizer steps).")
    ap.add_argument("--sync-timing", action="store_true", help="Synchronize device before timing/memory reads (more accurate, slightly slower).")
    ap.add_argument("--live-plot", action="store_true", help="Realtime matplotlib plots (dev only).")
    ap.add_argument("--tb", action="store_true", help="Write TensorBoard scalars (requires `tensorboard` package).")

    # ---- Mode ----
    ap.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    ap.add_argument("--ckpt", type=str, default=None)

    # ---- Sampling ----
    ap.add_argument("--prompt-tokens", type=str, default="0")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)

    # ---- KV cache / decode (generation) ----
    ap.add_argument("--kv-cache", type=str, default="fp16", choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Default KV-cache format (can be overridden per-tensor with the --kv-cache-* flags).")
    ap.add_argument("--kv-qblock", type=int, default=32, help="Quantization block size along the channel dimension.")
    ap.add_argument("--kv-residual", type=int, default=128,
                    help="Keep this many newest KV tokens in fp16 as a hot residual window (only for quantized caches).")
    ap.add_argument("--kv-decode-block", type=int, default=1024,
                    help="Sequence-block size for streaming decode attention (smaller = less memory, more Python overhead).")
    ap.add_argument("--kv-fused", type=str, default="auto", choices=["none", "auto", "triton1pass", "triton2pass"],
                    help="Use fused decode kernels when available. 'auto' picks a sensible kernel when Triton+CUDA are available.")

    # v25 self-optimizer (decode performance)
    ap.add_argument("--self-opt", type=str, default="none", choices=["none", "startup", "online"],
                    help="Self-optimize decode-time knobs (decode_block, fused kernel choice, launch params).")
    ap.add_argument("--self-opt-cache", type=str, default=None,
                    help="JSON path to persist tuned plans across runs (optional).")
    ap.add_argument("--self-opt-decode-blocks", type=str, default="256,512,1024,2048",
                    help="Comma-separated candidate kv_decode_block values for tuning.")
    ap.add_argument("--self-opt-block-n", type=str, default="128",
                    help="Comma-separated BLOCK_N candidates for fused kernels (e.g. '64,128').")
    ap.add_argument("--self-opt-warps", type=str, default="4,8",
                    help="Comma-separated num_warps candidates for fused kernels.")
    ap.add_argument("--self-opt-stages", type=str, default="2,3",
                    help="Comma-separated num_stages candidates for fused kernels.")
    ap.add_argument("--self-opt-warmup", type=int, default=1,
                    help="Warmup iterations per candidate during tuning.")
    ap.add_argument("--self-opt-iters", type=int, default=3,
                    help="Timed iterations per candidate during tuning.")
    ap.add_argument("--self-opt-interval", type=int, default=256,
                    help="Online mode: tune at most once every N decode steps per bucket.")
    ap.add_argument("--self-opt-hysteresis", type=float, default=0.03,
                    help="Online mode: require this relative improvement to switch plans.")
    ap.add_argument("--self-opt-verbose", action="store_true",
                    help="Print tuning decisions + chosen plans.")
    ap.add_argument("--self-opt-verify", action="store_true",
                    help="Verify candidate outputs vs baseline while tuning (slow; debugging).")
    ap.add_argument("--self-opt-verify-tol", type=float, default=5e-3,
                    help="Max abs error allowed for --self-opt-verify (fp32).")

    # v26 cache-policy self-optimizer (kv_residual, quant kind, qblock) — decoupled only
    ap.add_argument("--self-opt-scope", type=str, default="all", choices=["decode", "cache", "all"],
                    help="Which knobs to self-optimize. 'cache' tunes kv_residual/quant/qblock at startup; 'decode' tunes decode kernels; 'all' does both.")
    ap.add_argument("--self-opt-residuals", type=str, default="0,32,64,128",
                    help="Comma-separated kv_residual candidates for cache-policy tuning.")
    ap.add_argument("--self-opt-qblocks", type=str, default="16,32,64",
                    help="Comma-separated qblock candidates for cache-policy tuning (applied to all quantized decoupled tensors).")
    ap.add_argument("--self-opt-k-sem-kinds", type=str, default="q4_0,nf4,q8_0,fp16",
                    help="Comma-separated semantic-K quantization kinds to consider (decoupled).")
    ap.add_argument("--self-opt-k-geo-kinds", type=str, default="q8_0,q4_0,fp16",
                    help="Comma-separated geometric-K quantization kinds to consider (decoupled).")
    ap.add_argument("--self-opt-v-kinds", type=str, default="q4_0,q8_0,fp16",
                    help="Comma-separated V quantization kinds to consider (decoupled).")
    ap.add_argument("--self-opt-mem-budget-mb", type=float, default=None,
                    help="Absolute memory budget in MB for KV cache-policy tuning (decoupled). If unset, uses baseline*(1+--self-opt-mem-overhead-frac).")
    ap.add_argument("--self-opt-mem-overhead-frac", type=float, default=0.10,
                    help="If --self-opt-mem-budget-mb is unset, allow this fractional overhead over baseline (residual=0).")
    ap.add_argument("--self-opt-policy-prefix-len", type=int, default=None,
                    help="Prefix length to benchmark during cache-policy tuning. If unset, derives from prompt/max_seq.")
    ap.add_argument("--self-opt-policy-warmup", type=int, default=1,
                    help="Warmup iterations for cache-policy microbench.")
    ap.add_argument("--self-opt-policy-iters", type=int, default=3,
                    help="Timed iterations for cache-policy microbench.")
    ap.add_argument("--self-opt-policy-hysteresis", type=float, default=0.02,
                    help="Cache-policy hillclimb: require this relative improvement to accept a move.")
    ap.add_argument("--self-opt-prefer-low-mem-within", type=float, default=0.02,
                    help="Cache-policy tie-break: if speed is within this fraction, prefer lower memory.")
    ap.add_argument("--self-opt-policy-quality", action="store_true",
                    help="(Slow) After choosing a cache policy, run a small teacher-forced logits check vs fp16-cache baseline.")
    ap.add_argument("--self-opt-calib-tokens", type=str, default=None,
                    help="Calibration tokens for --self-opt-policy-quality (either a path or whitespace-separated ints). Defaults to --prompt-tokens.")
    ap.add_argument("--self-opt-calib-prefill", type=int, default=64,
                    help="Prefill length for policy quality check.")
    ap.add_argument("--self-opt-calib-decode", type=int, default=8,
                    help="Number of teacher-forced decode steps for policy quality check.")
    ap.add_argument("--self-opt-quality-tol", type=float, default=0.5,
                    help="Max abs logit error allowed for policy quality check.")
    ap.add_argument("--kv-cache-k", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Override K cache kind (standard/bottleneck/gqa).")
    ap.add_argument("--kv-cache-v", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Override V cache kind (standard/bottleneck/gqa and decoupled).")
    ap.add_argument("--kv-cache-k-sem", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Override semantic K cache kind (decoupled only).")
    ap.add_argument("--kv-cache-k-geo", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Override geometric K cache kind (decoupled only).")
    ap.add_argument("--kv-qblock-k", type=int, default=None, help="Override K qblock (standard/bottleneck/gqa).")
    ap.add_argument("--kv-qblock-v", type=int, default=None, help="Override V qblock.")
    ap.add_argument("--kv-qblock-k-sem", type=int, default=None, help="Override semantic K qblock.")
    ap.add_argument("--kv-qblock-k-geo", type=int, default=None, help="Override geometric K qblock.")

    # Tokenizer
    ap.add_argument("--tokenizer", type=str, default="word", choices=["word", "tiktoken"])

    args = ap.parse_args()

    # Apply paper suite presets (only when provided)
    apply_size_preset(args)
    apply_exp_preset(args)

    # Explicit "force disable" flags (useful because store_true args can't be negated otherwise)
    if args.no_null_attn:
        args.null_attn = False
    if args.no_tie_qk:
        args.tie_qk = False
    if args.rope:
        args.no_rope = False

    # Derive out_dir if omitted and size+exp provided
    inferred = default_out_dir(args)
    if args.out_dir is None and inferred is not None:
        args.out_dir = inferred

    device = pick_device(args.device)
    set_seed(args.seed)

    # Matmul precision hint (mostly impacts float32 matmuls).
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(args.matmul_precision))
    except Exception:
        pass


    # For paper_all, run each experiment sequentially (train mode only).
    if args.mode == "train" and args.exp == "paper_all":
        for exp in ["paper_baseline", "paper_bottleneck", "paper_decoupled", "paper_gqa"]:
            import copy
            a2 = copy.deepcopy(args)
            a2.exp = exp
            apply_exp_preset(a2)
            if a2.no_null_attn:
                a2.null_attn = False
            if a2.no_tie_qk:
                a2.tie_qk = False
            if a2.rope:
                a2.no_rope = False
            inferred2 = default_out_dir(a2)
            if inferred2 is not None:
                a2.out_dir = inferred2
            _run_single(a2, device)
        return

    _run_single(args, device)


def _run_single(args: argparse.Namespace, device: torch.device) -> None:
    # -------------------------
    # Sample mode (no dataset needed): build model from checkpoint config
    # -------------------------
    if args.mode == "sample":
        if not args.ckpt:
            raise ValueError("--ckpt is required for --mode sample")
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg_dict = ckpt.get("config", None)
        if cfg_dict is None:
            raise ValueError("Checkpoint missing 'config'. Can't reconstruct model safely.")
        cfg = ModelConfig(**cfg_dict)
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        # Prompt: either raw token IDs or text (tiktoken only)
        try:
            prompt_ids = [int(t) for t in args.prompt_tokens.strip().split()]
        except ValueError:
            if args.tokenizer != "tiktoken":
                raise ValueError("Text prompts require --tokenizer tiktoken")
            if tiktoken is None:
                raise ImportError("tiktoken needed for text prompts")
            enc = tiktoken.get_encoding("gpt2")
            prompt_ids = enc.encode_ordinary(args.prompt_tokens)

        prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)

        # Self-opt config (decode + cache policy)
        def _csv_ints(s: Optional[str]) -> Tuple[int, ...]:
            if s is None:
                return ()
            parts: List[int] = []
            for x in str(s).split(","):
                x = x.strip()
                if not x:
                    continue
                try:
                    parts.append(int(x))
                except Exception:
                    pass
            return tuple(parts)

        def _csv_strs(s: Optional[str]) -> Tuple[str, ...]:
            if s is None:
                return ()
            parts: List[str] = []
            for x in str(s).split(","):
                x = x.strip()
                if x:
                    parts.append(x)
            return tuple(parts)

        self_opt_cfg = None
        if getattr(args, "self_opt", "none") != "none":
            self_opt_cfg = KVSelfOptConfig(
                mode=args.self_opt,
                scope=getattr(args, "self_opt_scope", "all"),
                decode_blocks=_csv_ints(getattr(args, "self_opt_decode_blocks", "")) or (256, 512, 1024, 2048),
                block_ns=_csv_ints(getattr(args, "self_opt_block_n", "")) or (128,),
                warps=_csv_ints(getattr(args, "self_opt_warps", "")) or (4, 8),
                stages=_csv_ints(getattr(args, "self_opt_stages", "")) or (2, 3),
                warmup=int(getattr(args, "self_opt_warmup", 1)),
                iters=int(getattr(args, "self_opt_iters", 3)),
                interval=int(getattr(args, "self_opt_interval", 256)),
                hysteresis=float(getattr(args, "self_opt_hysteresis", 0.03)),
                cache_path=getattr(args, "self_opt_cache", None),
                verbose=bool(getattr(args, "self_opt_verbose", False)),
                verify=bool(getattr(args, "self_opt_verify", False)),
                verify_tol=float(getattr(args, "self_opt_verify_tol", 5e-3)),
                residuals=_csv_ints(getattr(args, "self_opt_residuals", "")) or (0, 32, 64, 128),
                qblocks=_csv_ints(getattr(args, "self_opt_qblocks", "")) or (16, 32, 64),
                k_sem_kinds=_csv_strs(getattr(args, "self_opt_k_sem_kinds", "")) or ("q4_0", "nf4", "q8_0", "fp16"),
                k_geo_kinds=_csv_strs(getattr(args, "self_opt_k_geo_kinds", "")) or ("q8_0", "q4_0", "fp16"),
                v_kinds=_csv_strs(getattr(args, "self_opt_v_kinds", "")) or ("q4_0", "q8_0", "fp16"),
                mem_budget_mb=getattr(args, "self_opt_mem_budget_mb", None),
                mem_overhead_frac=float(getattr(args, "self_opt_mem_overhead_frac", 0.10)),
                policy_prefix_len=getattr(args, "self_opt_policy_prefix_len", None),
                policy_warmup=int(getattr(args, "self_opt_policy_warmup", 1)),
                policy_iters=int(getattr(args, "self_opt_policy_iters", 3)),
                policy_hysteresis=float(getattr(args, "self_opt_policy_hysteresis", 0.02)),
                prefer_lower_mem_within=float(getattr(args, "self_opt_prefer_low_mem_within", 0.02)),
                policy_quality=bool(getattr(args, "self_opt_policy_quality", False)),
                calib_tokens=getattr(args, "self_opt_calib_tokens", None),
                calib_prefill=int(getattr(args, "self_opt_calib_prefill", 64)),
                calib_decode_steps=int(getattr(args, "self_opt_calib_decode", 8)),
                quality_tol=float(getattr(args, "self_opt_quality_tol", 0.5)),
            )

        # Logger for sampling (enable if --instrument/--live-plot/--tb used)
        logger = None
        if args.instrument != "off" or args.live_plot or args.tb:
            logger = RunLogger(
                args.out_dir, 
                instrument=args.instrument, 
                cfg=cfg, 
                args=args, 
                device=device,
                live_plot=bool(args.live_plot), 
                tb=bool(args.tb)
            )

        print(f"Generating {args.max_new_tokens} tokens...")
        t0 = time.time()
        try:
            out = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                kv_cache=args.kv_cache,
                kv_qblock=args.kv_qblock,
                kv_residual=args.kv_residual,
                kv_decode_block=args.kv_decode_block,
                kv_fused=args.kv_fused,
                self_opt=self_opt_cfg,
                kv_cache_k=args.kv_cache_k,
                kv_cache_v=args.kv_cache_v,
                kv_cache_k_sem=args.kv_cache_k_sem,
                kv_cache_k_geo=args.kv_cache_k_geo,
                kv_qblock_k=args.kv_qblock_k,
                kv_qblock_v=args.kv_qblock_v,
                kv_qblock_k_sem=args.kv_qblock_k_sem,
                kv_qblock_k_geo=args.kv_qblock_k_geo,
                log_callback=logger.log if logger else None,
            )
        finally:
            if logger:
                logger.close()

        dt = time.time() - t0
        print(f"Time: {dt:.2f}s | Tok/s: {args.max_new_tokens/max(dt,1e-9):.2f}")

        out_ids = out[0].tolist()
        if args.tokenizer == "tiktoken":
            enc = tiktoken.get_encoding("gpt2")
            print(enc.decode(out_ids))
        else:
            print(out_ids)
        return

    # -------------------------
    # Train mode
    # -------------------------
    if args.data is None:
        raise ValueError("--data is required for --mode train")
    if args.out_dir is None:
        raise ValueError("--out-dir is required for --mode train (or provide --size + --exp for auto dirs).")

    # ---- Load dataset (scale-aware) ----
    if _np is None:
        raise ImportError("numpy is required for training data loading in v27")

    data_path = Path(args.data)
    fmt = str(args.data_format)
    if fmt == "auto":
        suf = data_path.suffix.lower()
        if suf == ".npy":
            fmt = "npy"
        elif suf == ".bin":
            fmt = "bin"
        elif suf == ".pt":
            fmt = "pt"
        else:
            fmt = "text"

    tokens_any: Any
    if fmt == "text":
        # Fast-ish text parsing via numpy (still RAM heavy; use .npy/.bin for real scale)
        raw = data_path.read_text(encoding="utf-8")
        arr = _np.fromstring(raw.strip(), dtype=_np.int64, sep=" ")
        tokens_any = arr
    elif fmt == "npy":
        arr = _np.load(str(data_path), mmap_mode="r")
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        tokens_any = arr
    elif fmt == "bin":
        dt = _np.dtype(str(args.data_dtype))
        arr = _np.memmap(str(data_path), dtype=dt, mode="r")
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        tokens_any = arr
    elif fmt == "pt":
        t = torch.load(str(data_path), map_location="cpu")
        if isinstance(t, dict) and "tokens" in t:
            t = t["tokens"]
        if not isinstance(t, torch.Tensor):
            raise ValueError("pt data must be a 1D torch.Tensor or dict with 'tokens'")
        t = t.view(-1).to(torch.long)
        tokens_any = t
    else:
        raise ValueError(f"Unknown data format: {fmt}")

    n_total = int(tokens_any.numel()) if isinstance(tokens_any, torch.Tensor) else int(len(tokens_any))
    if n_total < int(args.block) + 2:
        raise ValueError(f"Dataset too small: n_tokens={n_total} block={args.block}")

    n_train = int((1.0 - float(args.val_frac)) * n_total)
    n_train = max(min(n_train, n_total - 2), 2)
    n_val = n_total - n_train

    class TokenView:
        def __init__(self, data: Any, start: int, end: int):
            self.data = data
            self.start = int(start)
            self.end = int(end)
        def __len__(self) -> int:
            return int(self.end - self.start)

    train_view = TokenView(tokens_any, 0, n_train)
    val_view = TokenView(tokens_any, n_train, n_total)

    # Determine vocab size
    if args.vocab_size is not None:
        vocab = int(args.vocab_size)
    elif args.tokenizer == "tiktoken":
        vocab = 50257
    else:
        # For text (or small tensors), we can compute max+1
        if isinstance(tokens_any, torch.Tensor):
            vocab = int(tokens_any.max().item()) + 1
        else:
            # Computing max over a huge memmap is expensive; warn loudly.
            print("[warn] --vocab-size not provided; scanning dataset for max token id (can be very slow on big memmaps).")
            vocab = int(_np.max(tokens_any)) + 1  # type: ignore

    # ---- Config ----
    cfg = ModelConfig(
        vocab_size=int(vocab),
        block_size=int(args.block),
        n_layer=int(args.layers),
        n_head=int(args.n_head),
        kv_head=args.kv_head,
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        embed_dim=int(args.embed_dim),
        attn_mode=str(args.attn_mode),
        attn_dim=int(args.attn_dim),
        sem_dim=int(args.sem_dim),
        geo_dim=int(args.geo_dim),
        rope=(not args.no_rope),
        rope_base=float(args.rope_base),
        tie_qk=bool(args.tie_qk),
        null_attn=bool(args.null_attn),
        learned_temp=(not args.no_learned_temp),
        mlp=str(args.mlp),
        dropout=float(args.dropout),
    )

    # Print config & exit (handy for make print_config)
    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
        try:
            mem_ctx = estimate_kv_cache_bytes(cfg, seq_len=cfg.block_size, batch=1,
                                             kv_cache=args.kv_cache, kv_qblock=args.kv_qblock, kv_residual=args.kv_residual,
                                             kv_cache_k=args.kv_cache_k, kv_cache_v=args.kv_cache_v,
                                             kv_cache_k_sem=args.kv_cache_k_sem, kv_cache_k_geo=args.kv_cache_k_geo,
                                             kv_qblock_k=args.kv_qblock_k, kv_qblock_v=args.kv_qblock_v,
                                             kv_qblock_k_sem=args.kv_qblock_k_sem, kv_qblock_k_geo=args.kv_qblock_k_geo)
            mem_128 = estimate_kv_cache_bytes(cfg, seq_len=128_000, batch=1,
                                             kv_cache=args.kv_cache, kv_qblock=args.kv_qblock, kv_residual=args.kv_residual,
                                             kv_cache_k=args.kv_cache_k, kv_cache_v=args.kv_cache_v,
                                             kv_cache_k_sem=args.kv_cache_k_sem, kv_cache_k_geo=args.kv_cache_k_geo,
                                             kv_qblock_k=args.kv_qblock_k, kv_qblock_v=args.kv_qblock_v,
                                             kv_qblock_k_sem=args.kv_qblock_k_sem, kv_qblock_k_geo=args.kv_qblock_k_geo)
            print(f"KV cache @ ctx={cfg.block_size}: {human_bytes(mem_ctx['total_bytes'])}")
            print(f"KV cache @ 128k: {human_bytes(mem_128['total_bytes'])}")
        except Exception as e:
            print(f"[warn] KV memory estimate failed: {e}")
        return

    model = GPT(cfg).to(device)

    # Parameter dtype (memory / speed). fp32 is default; bf16/fp16 cut memory ~2x.
    param_dtype = resolve_dtype(device, args.param_dtype, default=torch.float32)
    if param_dtype != torch.float32:
        model = model.to(dtype=param_dtype)

    # Training-only toggle: gradient checkpointing (activation recompute).
    try:
        model.grad_checkpointing = bool(args.grad_checkpoint)
    except Exception:
        pass

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled (mode={args.compile_mode}).")
        except Exception as e:
            print(f"torch.compile failed, continuing without it: {e}")

    # Ensure training-only flags survive compile wrappers.
    try:
        model.grad_checkpointing = bool(args.grad_checkpoint)
    except Exception:
        pass

    # ---- Batch sampling for both torch tensors and numpy memmaps ----
    _offs_cache_t: Dict[int, torch.Tensor] = {}
    _offs_cache_np: Dict[int, _np.ndarray] = {}

    def get_batch_any(view: TokenView, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = len(view) - block_size - 1
        if max_start <= 0:
            raise ValueError(f"Not enough tokens in split: len={len(view)} block={block_size}")

        # Offsets cache (avoid realloc every batch)
        offs_t = _offs_cache_t.get(block_size)
        if offs_t is None or offs_t.numel() != block_size:
            offs_t = torch.arange(block_size, dtype=torch.long)
            _offs_cache_t[block_size] = offs_t

        ix = torch.randint(0, max_start, (batch_size,), device="cpu", dtype=torch.long)

        if isinstance(view.data, torch.Tensor):
            # Vectorized gather from a 1D CPU tensor
            base = (view.start + ix).unsqueeze(1)  # (B,1)
            idx = base + offs_t.unsqueeze(0)       # (B,T)
            x = view.data[idx]
            y = view.data[idx + 1]
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # numpy array / memmap path (vectorized)
        offs_np = _offs_cache_np.get(block_size)
        if offs_np is None or offs_np.shape[0] != block_size:
            offs_np = _np.arange(block_size, dtype=_np.int64)
            _offs_cache_np[block_size] = offs_np

        ix_np = ix.numpy().astype(_np.int64, copy=False)
        idx_np = (view.start + ix_np[:, None] + offs_np[None, :]).astype(_np.int64, copy=False)

        x_np = _np.asarray(view.data[idx_np], dtype=_np.int64)
        y_np = _np.asarray(view.data[idx_np + 1], dtype=_np.int64)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    @torch.no_grad()
    def estimate_loss_any(eval_iters: int, block_size: int) -> Tuple[float, float]:
        """Return (train_loss, val_loss) using random batches."""
        model.eval()
        iters = int(eval_iters)
        bs = int(block_size)

        def _split_loss(view: TokenView) -> float:
            losses: List[float] = []
            for _ in range(iters):
                xb, yb = get_batch_any(view, batch_size=args.batch_size, block_size=bs, device=device)
                with autocast_ctx:
                    logits, _ = model(xb)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                losses.append(float(loss.detach().to("cpu").item()))
            return float(sum(losses) / max(1, len(losses)))

        tr = _split_loss(train_view)
        va = _split_loss(val_view)
        model.train()
        return tr, va



    # Logger (instrumentation)
    logger = RunLogger(args.out_dir, instrument=args.instrument, cfg=cfg, args=args, device=device,
                       live_plot=bool(args.live_plot), tb=bool(args.tb)) if args.instrument != "off" else None

    # Log KV cache memory probes up front
    if logger is not None:
        try:
            mem_ctx = estimate_kv_cache_bytes(cfg, seq_len=cfg.block_size, batch=1,
                                             kv_cache=args.kv_cache, kv_qblock=args.kv_qblock, kv_residual=args.kv_residual,
                                             kv_cache_k=args.kv_cache_k, kv_cache_v=args.kv_cache_v,
                                             kv_cache_k_sem=args.kv_cache_k_sem, kv_cache_k_geo=args.kv_cache_k_geo,
                                             kv_qblock_k=args.kv_qblock_k, kv_qblock_v=args.kv_qblock_v,
                                             kv_qblock_k_sem=args.kv_qblock_k_sem, kv_qblock_k_geo=args.kv_qblock_k_geo)
            mem_128 = estimate_kv_cache_bytes(cfg, seq_len=128_000, batch=1,
                                             kv_cache=args.kv_cache, kv_qblock=args.kv_qblock, kv_residual=args.kv_residual,
                                             kv_cache_k=args.kv_cache_k, kv_cache_v=args.kv_cache_v,
                                             kv_cache_k_sem=args.kv_cache_k_sem, kv_cache_k_geo=args.kv_cache_k_geo,
                                             kv_qblock_k=args.kv_qblock_k, kv_qblock_v=args.kv_qblock_v,
                                             kv_qblock_k_sem=args.kv_qblock_k_sem, kv_qblock_k_geo=args.kv_qblock_k_geo)
            logger.log({"type": "mem", "step": 0,
                        "kv_ctx_bytes": int(mem_ctx["total_bytes"]),
                        "kv_128k_bytes": int(mem_128["total_bytes"]),
                        "data_format": fmt,
                        "data_dtype": str(args.data_dtype),
                        "n_tokens_total": int(n_total),
                        "n_tokens_train": int(n_train),
                        "n_tokens_val": int(n_val),
                        "vocab_size": int(vocab)})
        except Exception as e:
            logger.log({"type": "mem", "step": 0, "error": str(e)})

    # -----------------------------
    # Optimizer / AMP / schedules
    # -----------------------------
    grad_accum = max(1, int(args.grad_accum))
    global_batch = int(args.batch_size) * grad_accum

    # Sequence length curriculum (optional)
    seq_schedule = parse_seq_schedule(args.seq_schedule)
    base_seq_len = int(args.train_seq_len) if int(args.train_seq_len) > 0 else int(cfg.block_size)
    base_seq_len = min(base_seq_len, int(cfg.block_size))
    fixed_eval_seq_len = int(args.eval_seq_len) if int(args.eval_seq_len) > 0 else 0

    # Optimizer
    if str(args.optimizer).lower() == "lion":
        lion_betas = _parse_two_floats(args.lion_betas, default=(0.9, 0.99))
        opt = Lion(model.parameters(), lr=args.lr, betas=lion_betas, weight_decay=args.weight_decay)
    else:
        adam_betas = _parse_two_floats(args.adam_betas, default=(0.9, 0.95))
        opt_kwargs: Dict[str, Any] = dict(lr=args.lr, weight_decay=args.weight_decay, betas=adam_betas, eps=float(args.adam_eps))
        if bool(args.opt_foreach):
            opt_kwargs["foreach"] = True
        if bool(args.opt_fused) and device.type == "cuda":
            opt_kwargs["fused"] = True
        try:
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        except TypeError:
            # Older torch: drop unsupported keys
            opt_kwargs.pop("foreach", None)
            opt_kwargs.pop("fused", None)
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    # AMP (torch.amp for CUDA/MPS/CPU)
    amp_enabled = bool(args.amp)
    try:
        if hasattr(torch, "amp") and hasattr(torch.amp, "is_autocast_available"):
            amp_enabled = amp_enabled and bool(torch.amp.is_autocast_available(device.type))
    except Exception:
        amp_enabled = False

    amp_dtype = resolve_dtype(device, args.amp_dtype, default=torch.bfloat16)
    autocast_ctx = contextlib.nullcontext()
    scaler = None
    try:
        if amp_enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler(device=device.type, enabled=(amp_enabled and amp_dtype == torch.float16))
    except Exception:
        autocast_ctx = contextlib.nullcontext()
        scaler = None

    # analysis layers/heads parsing
    def _parse_csv_int_list(s: str) -> List[int]:
        s = s.strip()
        if not s:
            return []
        out: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if part == "":
                continue
            try:
                out.append(int(part))
            except Exception:
                pass
        return out

    analysis_layers = _parse_csv_int_list(getattr(args, "analysis_layers", "")) or [0]
    analysis_heads = _parse_csv_int_list(getattr(args, "analysis_heads", "")) or [0]
    analysis_tokens = max(2, int(getattr(args, "analysis_tokens", 128)))

    # Console live dashboard
    dashboard = LiveDashboard(args.live, total_steps=args.steps, out_dir=args.out_dir, cfg=cfg, args=args, device=device)
    dash_every = max(1, int(getattr(args, "live_update_every", 1)))

    # One-time run summary for the live console.
    try:
        n_params = sum(p.numel() for p in model.parameters())
        dashboard.message(
            f"run: params={n_params/1e6:.2f}M | opt={args.optimizer} | gbs={global_batch} "
            f"| param_dtype={str(param_dtype).replace('torch.', '')} | amp={amp_enabled} "
            f"| amp_dtype={str(amp_dtype).replace('torch.', '')} | grad_ckpt={bool(args.grad_checkpoint)}"
        )
    except Exception:
        pass


    best_val = float("inf")

    # Interval accumulators for stable perf readouts
    tok_count = 0
    dt_acc = 0.0
    data_acc = 0.0
    fwd_acc = 0.0
    bwd_acc = 0.0
    opt_acc = 0.0
    steps_in_acc = 0

    # A one-time expensive measurement after optimizer state materializes
    opt_state_bytes_cached: Optional[int] = None

    def maybe_sync() -> None:
        if bool(getattr(args, "sync_timing", False)):
            device_synchronize(device)

    @torch.no_grad()
    def do_eval(step: int, *, seq_len_hint: int) -> Dict[str, float]:
        model.eval()
        eval_seq = fixed_eval_seq_len if fixed_eval_seq_len > 0 else seq_len_hint
        eval_seq = int(min(max(2, eval_seq), cfg.block_size))
        tr_loss, va_loss = estimate_loss_any(eval_iters=args.eval_iters, block_size=eval_seq)
        val_loss = float(va_loss)
        out = {
            "type": "eval",
            "step": int(step),
            "train_loss": float(tr_loss),
            "val_loss": val_loss,
            "val_ppl": float(math.exp(val_loss)),
        }
        return out

    try:
        # Baseline eval at step 0
        eval0 = do_eval(0, seq_len_hint=base_seq_len)
        best_val = min(best_val, eval0["val_loss"])
        eval0["best_val"] = best_val
        if logger is not None:
            logger.log(eval0)
        dashboard.update_eval(eval0)
        dashboard.message(f"eval@0: val_loss={eval0['val_loss']:.6f} val_ppl={eval0['val_ppl']:.2f}")

        for step in range(1, args.steps + 1):
            step_idx = step - 1

            # LR schedule
            lr = lr_for_step(
                step_idx,
                base_lr=float(args.lr),
                total_steps=int(args.steps),
                schedule=str(args.lr_schedule),
                warmup_steps=int(args.warmup_steps),
                min_lr=float(args.min_lr),
            )
            for pg in opt.param_groups:
                pg["lr"] = lr

            # Current training seq-len (curriculum)
            seq_len = seq_len_for_step(step_idx, default_seq_len=base_seq_len, schedule=seq_schedule)
            seq_len = int(min(max(2, seq_len), cfg.block_size))

            model.train()
            opt.zero_grad(set_to_none=True)

            # Timers (optionally synchronized)
            maybe_sync()
            t_step0 = time.perf_counter()

            loss_sum_t: Optional[torch.Tensor] = None
            data_t = 0.0
            fwd_t = 0.0
            bwd_t = 0.0

            for micro in range(grad_accum):
                t0 = time.perf_counter()
                xb, yb = get_batch_any(train_view, batch_size=args.batch_size, block_size=seq_len, device=device)
                maybe_sync()
                data_t += time.perf_counter() - t0

                with autocast_ctx:
                    t1 = time.perf_counter()
                    logits, _ = model(xb)
                    # mean loss per-token
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                    maybe_sync()
                    fwd_t += time.perf_counter() - t1

                if loss_sum_t is None:
                    loss_sum_t = loss.detach()
                else:
                    loss_sum_t = loss_sum_t + loss.detach()

                # Normalize for gradient accumulation
                loss_to_back = loss / grad_accum
                t2 = time.perf_counter()
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_to_back).backward()
                else:
                    loss_to_back.backward()
                maybe_sync()
                bwd_t += time.perf_counter() - t2

            # Backward done -> optimizer step
            t3 = time.perf_counter()
            grad_norm_clip = None
            if scaler is not None and scaler.is_enabled():
                # Unscale before clipping / measuring
                try:
                    scaler.unscale_(opt)
                except Exception:
                    pass

            if args.grad_clip and args.grad_clip > 0:
                grad_norm_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().to("cpu").item())

            # Light-weight grad norm for dashboard/logs
            grad_stats: Dict[str, float] = {}
            if (step % args.log_every) == 0 or step == 1:
                try:
                    grad_stats = compute_grad_norms(model)
                except Exception:
                    grad_stats = {}

            if scaler is not None and scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            maybe_sync()
            opt_t = time.perf_counter() - t3

            maybe_sync()
            step_dt = time.perf_counter() - t_step0

            # Update interval accumulators
            tok_count += int(args.batch_size) * seq_len * grad_accum
            dt_acc += step_dt
            data_acc += data_t
            fwd_acc += fwd_t
            bwd_acc += bwd_t
            opt_acc += opt_t
            steps_in_acc += 1

            # Eval / analysis checkpoints
            if (step % args.eval_every) == 0 or step == args.steps:
                ev = do_eval(step, seq_len_hint=seq_len)
                val_loss = ev["val_loss"]
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(args.out_dir, "best.pt", model, cfg, step, best_val)
                    dashboard.message(f"new best @ step {step}: val_loss={best_val:.6f}")
                ev["best_val"] = best_val
                if logger is not None:
                    logger.log(ev)
                dashboard.update_eval(ev)

            if args.analysis_every and (step % args.analysis_every) == 0:
                try:
                    # Use current seq_len (or smaller) to keep analysis cheap
                    ab = min(seq_len, analysis_tokens)
                    xb_a, yb_a = get_batch_any(train_view, batch_size=1, block_size=ab, device=device)
                    analysis_metrics, analysis_tensors = analyze_attention(
                        model,
                        xb_a,
                        layers=analysis_layers,
                        heads=analysis_heads,
                        max_tokens=ab,
                        topk=int(getattr(args, "analysis_topk", 8)),
                        local_window=int(getattr(args, "analysis_local_window", 32)),
                        save_mats=True,
                        save_scores=bool(getattr(args, "analysis_save_scores", False)),
                        compute_svd=True,
                    )
                    if logger is not None:
                        logger.log({
                            "type": "analysis",
                            "step": int(step),
                            **analysis_metrics,
                        })
                        if analysis_tensors:
                             logger.h5_write_step(step, group="analysis", tensors=analysis_tensors, attrs=analysis_metrics)
                except Exception as e:
                    dashboard.message(f"[warn] analysis failed @ step {step}: {e}")

            # Logging + live dashboard
            if (step % args.log_every) == 0 or step == 1:
                # interval averages
                tok_s = float(tok_count / max(dt_acc, 1e-9))
                if loss_sum_t is None:
                    loss_step = float("nan")
                else:
                    loss_step = float((loss_sum_t / grad_accum).detach().to("cpu").item())
                ppl_step = float(math.exp(loss_step))

                # Memory stats
                mem = get_device_mem_stats(device)
                rss = get_process_rss_bytes()
                if rss is not None:
                    mem["cpu_rss_bytes"] = float(rss)

                # Optimizer state bytes (once)
                nonlocal_opt_state = opt_state_bytes_cached
                if nonlocal_opt_state is None:
                    try:
                        nonlocal_opt_state = optimizer_state_bytes(opt)
                    except Exception:
                        nonlocal_opt_state = -1
                    opt_state_bytes_cached = nonlocal_opt_state

                # AMP scale
                amp_scale = None
                try:
                    if scaler is not None and scaler.is_enabled():
                        amp_scale = float(scaler.get_scale())
                except Exception:
                    amp_scale = None

                evt: Dict[str, Any] = {
                    "type": "train",
                    "step": int(step),
                    "loss": loss_step,
                    "ppl": ppl_step,
                    "lr": float(lr),
                    "tok_s": tok_s,
                    "seq_len": int(seq_len),
                    "grad_accum": int(grad_accum),
                    "global_batch": int(global_batch),
                    "step_ms": float((dt_acc / max(steps_in_acc, 1)) * 1000.0),
                    "data_ms": float((data_acc / max(steps_in_acc, 1)) * 1000.0),
                    "fwd_ms": float((fwd_acc / max(steps_in_acc, 1)) * 1000.0),
                    "bwd_ms": float((bwd_acc / max(steps_in_acc, 1)) * 1000.0),
                    "opt_ms": float((opt_acc / max(steps_in_acc, 1)) * 1000.0),
                    "grad_clip_norm": grad_norm_clip,
                    "opt_state_bytes": int(opt_state_bytes_cached) if opt_state_bytes_cached is not None else -1,
                }
                if amp_scale is not None:
                    evt["amp_scale"] = amp_scale
                evt.update(mem)
                evt.update(grad_stats)

                if logger is not None:
                    logger.log(evt)

                # Dashboard update (throttled)
                if (step % dash_every) == 0 or step == 1:
                    dashboard.update_train(evt)

                # reset interval accumulators
                tok_count = 0
                dt_acc = 0.0
                data_acc = 0.0
                fwd_acc = 0.0
                bwd_acc = 0.0
                opt_acc = 0.0
                steps_in_acc = 0

            # Periodic checkpoint
            if args.save_every and (step % args.save_every) == 0:
                save_ckpt(args.out_dir, f"step{step}.pt", model, cfg, step, best_val)

        # Save final checkpoint
        save_ckpt(args.out_dir, "last.pt", model, cfg, args.steps, best_val)

        if logger is not None:
            logger.finalize(best_val=best_val, cfg=cfg)

        dashboard.message("training complete")
    finally:
        _dash = locals().get("dashboard", None)
        if _dash is not None:
            try:
                _dash.close()
            except Exception:
                pass
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
