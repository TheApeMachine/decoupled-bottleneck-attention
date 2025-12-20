from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    TRITON_AVAILABLE = True
except Exception:
    triton = None  # type: ignore
    tl = None  # type: ignore
    TRITON_AVAILABLE = False


def _triton_decoupled_q4q8q4_available() -> bool:
    return bool(TRITON_AVAILABLE)


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
    decode_blocks: Tuple[int, ...] = (256, 512, 1024, 2048)
    block_ns: Tuple[int, ...] = (128,)
    warps: Tuple[int, ...] = (4, 8)
    stages: Tuple[int, ...] = (2, 3)

    # v32: hierarchical decode tuning.
    # By default we use a small set of internal kernel profiles (per device/fused/block) to avoid a
    # combinatorial explosion of low-level launch parameters. To retain legacy behavior, enable
    # expert launch space via CLI or set kernel_profiles="off".
    kernel_profiles: Literal["auto", "small", "off"] = "auto"
    expert_launch_space: bool = False

    warmup: int = 1
    iters: int = 3

    # online mode: at most once every N decode steps per bucket, try a neighbor and keep it if faster.
    interval: int = 256
    hysteresis: float = 0.03  # require >=3% improvement to switch

    cache_path: Optional[str] = None
    verbose: bool = False

    # Optional correctness guardrail when comparing decode candidates.
    verify: bool = False
    verify_tol: float = 5e-3

    # ---------------------------
    # Cache-policy tuning (v26)
    # ---------------------------
    residuals: Tuple[int, ...] = (0, 32, 64, 128)
    qblocks: Tuple[int, ...] = (16, 32, 64)

    k_sem_kinds: Tuple["KVCacheKind", ...] = ("q4_0", "nf4", "q8_0", "fp16")
    k_geo_kinds: Tuple["KVCacheKind", ...] = ("q8_0", "q4_0", "fp16")
    v_kinds: Tuple["KVCacheKind", ...] = ("q4_0", "q8_0", "fp16")

    mem_budget_mb: Optional[float] = None
    mem_overhead_frac: float = 0.10

    policy_prefix_len: Optional[int] = None
    policy_warmup: int = 1
    policy_iters: int = 3
    policy_hysteresis: float = 0.02
    prefer_lower_mem_within: float = 0.02

    # Optional quality guard for policy tuning (slow).
    policy_quality: bool = False
    calib_tokens: Optional[str] = None
    calib_prefill: int = 128
    calib_decode_steps: int = 32
    quality_tol: float = 0.5
    quality_delta_nll_tol: Optional[float] = 0.02
    quality_ppl_ratio_tol: Optional[float] = 1.02
    quality_kl_tol: Optional[float] = None
    quality_compute_kl: bool = False

    # Optional *long-horizon* quality gate (final accept check; much slower).
    # Motivation: short teacher-forced windows can miss accumulated quantization error that only appears
    # after thousands of cached steps.
    policy_quality_long: bool = False
    calib_long_tokens: Optional[str] = None
    calib_long_prefill: int = 4096
    calib_long_decode_steps: int = 128
    quality_long_tol: Optional[float] = None
    quality_long_delta_nll_tol: Optional[float] = None
    quality_long_ppl_ratio_tol: Optional[float] = None
    quality_long_kl_tol: Optional[float] = None
    quality_long_compute_kl: bool = False

    # v31: optional "repair" path for cache-policy tuning.
    # If a globally-chosen policy violates quality gates, allow promoting early layers to fp16 while
    # keeping later layers quantized (reduces memory vs full fp16 fallback).
    layerwise_cache: bool = False

    # v31: speculative decoding control loop (optional; used by generation code, not cache tuning itself).
    spec_enabled: bool = False
    spec_k: Tuple[int, ...] = (2, 4, 6, 8)
    spec_min_accept: float = 0.6
    spec_probe_every: int = 64


def load_token_ids_spec(spec: str) -> List[int]:
    """Load token IDs from either:
      - a path to a file containing whitespace-separated ints
      - a path to a .npy file (np.load)
      - an inline whitespace-separated string of ints
    """
    s = str(spec)
    p = Path(s)
    if os.path.exists(s):
        if p.suffix == ".npy":
            try:
                import numpy as _np  # type: ignore
            except Exception as e:
                raise RuntimeError(f"numpy required to load .npy token files: {e}")
            arr = _np.load(str(p), mmap_mode="r")
            arr = _np.asarray(arr).reshape(-1)
            if arr.dtype != _np.int64:
                arr = arr.astype(_np.int64, copy=False)
            return [int(x) for x in arr.tolist()]
        raw = p.read_text(encoding="utf-8", errors="ignore")
        return [int(t) for t in raw.strip().split() if t.strip()]
    return [int(t) for t in s.strip().split() if t.strip()]


@dataclass
class KVDecodePlan:
    fused: str  # "none" | "triton1pass" | "triton2pass"
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
        cache.decode_block = int(self.decode_block)
        cache.fused = str(self.fused)

        cache.block_n = int(self.block_n)
        cache.num_warps_1pass = int(self.num_warps_1pass)
        cache.num_stages_1pass = int(self.num_stages_1pass)
        cache.num_warps_part = int(self.num_warps_part)
        cache.num_stages_part = int(self.num_stages_part)
        cache.num_warps_reduce = int(self.num_warps_reduce)
        cache.num_stages_reduce = int(self.num_stages_reduce)


@dataclass(frozen=True)
class TritonKernelProfile:
    """A small, named set of launch parameters for fused decode kernels.

    The goal is not to encode every possible knob, but to provide a compact menu that yields
    near-optimal performance on a given GPU architecture for common block sizes.
    """
    name: str
    block_n: int
    # 1-pass params
    num_warps_1pass: int = 4
    num_stages_1pass: int = 2
    # 2-pass params
    num_warps_part: int = 4
    num_stages_part: int = 2
    num_warps_reduce: int = 1
    num_stages_reduce: int = 1


def _parse_cc_from_device_sig(device_sig: str) -> Optional[int]:
    """Parse compute capability from `_device_sig` string, returning e.g. 80 for cc80."""
    s = str(device_sig)
    if "cc" not in s:
        return None
    try:
        tail = s.split("cc", 1)[1]
        digs = ""
        for ch in tail:
            if ch.isdigit():
                digs += ch
            else:
                break
        if not digs:
            return None
        return int(digs)
    except Exception:
        return None


def get_triton_kernel_profiles(
    *,
    mode: str,
    device_sig: str,
    fused: str,
    decode_block: int,
) -> List[TritonKernelProfile]:
    """Return a small set of kernel profiles for fused decode."""
    mode = str(mode)
    fused = str(fused)
    db = int(decode_block)

    if mode == "off":
        return []

    cc = _parse_cc_from_device_sig(device_sig)
    # Conservative defaults: work reasonably across architectures.
    # For Hopper/Ada/Ampere we typically benefit from slightly higher warps/stages for throughput.
    is_modern = bool(cc is not None and cc >= 80)

    # Pick a BLOCK_N that divides the typical decode_block; keep it simple.
    if db < 128:
        bn = 64
    elif db < 512:
        bn = 128
    else:
        bn = 128

    # Profiles: keep count small.
    profs: List[TritonKernelProfile] = []
    if fused == "triton1pass":
        profs.append(TritonKernelProfile(name="latency", block_n=bn, num_warps_1pass=4, num_stages_1pass=2))
        if mode == "auto":
            profs.append(
                TritonKernelProfile(
                    name="throughput",
                    block_n=bn,
                    num_warps_1pass=(8 if is_modern else 4),
                    num_stages_1pass=(3 if is_modern else 2),
                )
            )
    elif fused == "triton2pass":
        profs.append(TritonKernelProfile(name="latency", block_n=bn, num_warps_part=4, num_stages_part=2, num_warps_reduce=1, num_stages_reduce=1))
        if mode == "auto":
            profs.append(
                TritonKernelProfile(
                    name="throughput",
                    block_n=bn,
                    num_warps_part=(8 if is_modern else 4),
                    num_stages_part=(3 if is_modern else 2),
                    num_warps_reduce=(2 if is_modern else 1),
                    num_stages_reduce=1,
                )
            )
    return profs


def _pow2_bucket(n: int) -> int:
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
    """Self-optimizes decode performance knobs per prefix-length bucket."""

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

        self._cache_path = cfg.cache_path
        if self._cache_path:
            try:
                if os.path.exists(self._cache_path):
                    with open(self._cache_path, "r") as f:
                        raw0 = json.load(f)
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
                            root = {"version": 26, "decode_plans": decode_plans}
                except Exception:
                    pass

            with open(self._cache_path, "w") as f:
                json.dump(root, f, indent=2, sort_keys=True)
        except Exception as e:
            if self.cfg.verbose:
                print(f"[selfopt] Failed to save cache '{self._cache_path}': {e}")

    def _key(self, *, attn: Any, cache: Any, L_prefix: int) -> str:
        bucket = _pow2_bucket(L_prefix)
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
        if self.base_fused == "none":
            return ["none"]
        if not _triton_decoupled_q4q8q4_available():
            return ["none"]
        ok = True
        try:
            ok = (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0")
        except Exception:
            ok = False
        if not ok:
            return ["none"]

        if self.base_fused in ("triton1pass", "triton2pass"):
            return [self.base_fused]
        return ["none", "triton1pass", "triton2pass"]

    def _candidate_plans(self, *, cache: Any) -> List[KVDecodePlan]:
        cfg = self.cfg
        fused_modes = self._allowed_fused_modes(cache=cache)
        decode_blocks = list(dict.fromkeys([self.base_decode_block, *cfg.decode_blocks]))
        decode_blocks = [int(x) for x in decode_blocks if int(x) > 0]
        decode_blocks.sort()

        # Candidate generation strategy:
        # - default: use a small internal set of kernel profiles (dramatically reduces search space)
        # - expert: use the legacy cross-product of block_n/warps/stages
        use_profiles = (not bool(getattr(cfg, "expert_launch_space", False))) and (str(getattr(cfg, "kernel_profiles", "auto")) != "off")

        block_ns = [int(x) for x in cfg.block_ns if int(x) > 0] or [128]
        warps = [int(x) for x in cfg.warps if int(x) > 0] or [4]
        stages = [int(x) for x in cfg.stages if int(x) > 0] or [2]

        plans: List[KVDecodePlan] = []
        for fused in fused_modes:
            for db in decode_blocks:
                if fused == "none":
                    plans.append(KVDecodePlan(fused="none", decode_block=db))
                    continue
                if use_profiles:
                    profs = get_triton_kernel_profiles(
                        mode=str(getattr(cfg, "kernel_profiles", "auto")),
                        device_sig=_device_sig(self.device),
                        fused=fused,
                        decode_block=int(db),
                    )
                    for pr in profs:
                        bn = int(pr.block_n)
                        if db < bn:
                            continue
                        if fused == "triton1pass":
                            plans.append(
                                KVDecodePlan(
                                    fused=fused,
                                    decode_block=db,
                                    block_n=bn,
                                    num_warps_1pass=int(pr.num_warps_1pass),
                                    num_stages_1pass=int(pr.num_stages_1pass),
                                )
                            )
                        else:
                            plans.append(
                                KVDecodePlan(
                                    fused=fused,
                                    decode_block=db,
                                    block_n=bn,
                                    num_warps_part=int(pr.num_warps_part),
                                    num_stages_part=int(pr.num_stages_part),
                                    num_warps_reduce=int(pr.num_warps_reduce),
                                    num_stages_reduce=int(pr.num_stages_reduce),
                                )
                            )
                else:
                    for bn in block_ns:
                        if db < bn:
                            continue
                        for w in warps:
                            for st in stages:
                                if fused == "triton1pass":
                                    plans.append(
                                        KVDecodePlan(
                                            fused=fused,
                                            decode_block=db,
                                            block_n=bn,
                                            num_warps_1pass=w,
                                            num_stages_1pass=st,
                                        )
                                    )
                                else:
                                    plans.append(
                                        KVDecodePlan(
                                            fused=fused,
                                            decode_block=db,
                                            block_n=bn,
                                            num_warps_part=w,
                                            num_stages_part=st,
                                            num_warps_reduce=1,
                                            num_stages_reduce=1,
                                        )
                                    )
        return plans

    def _time_ms(self, fn) -> float:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(self.device)
            return float(start.elapsed_time(end))
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
        if plan.fused == "triton1pass":
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
        if plan.fused == "triton2pass":
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
        baseline_out: Optional[torch.Tensor],
    ) -> float:
        def fn():
            out = self._run_plan(attn=attn, cache=cache, q_sem=q_sem, q_geo=q_geo, plan=plan, sem_scale=sem_scale, geo_scale=geo_scale)
            if self.cfg.verify and baseline_out is not None:
                err = (out.float() - baseline_out.float()).abs().max().item()
                if err > float(self.cfg.verify_tol):
                    raise RuntimeError(f"verify failed: max_abs_err={err} > tol={self.cfg.verify_tol}")

        # warmup
        for _ in range(max(0, int(self.cfg.warmup))):
            fn()
        ms = 0.0
        for _ in range(max(1, int(self.cfg.iters))):
            ms += self._time_ms(fn)
        return float(ms) / float(max(1, int(self.cfg.iters)))

    def maybe_get_plan(self, *, attn: Any, cache: Any, L_prefix: int) -> Optional[KVDecodePlan]:
        if self.cfg.mode == "none":
            return None

        self._step_counter += 1
        k = self._key(attn=attn, cache=cache, L_prefix=int(L_prefix))

        if k in self._plans and self.cfg.mode == "startup":
            return self._plans[k]

        if k in self._plans and self.cfg.mode == "online":
            last = self._last_probe_step.get(k, -10**9)
            if (self._step_counter - last) < int(self.cfg.interval):
                return self._plans[k]

        plans = self._candidate_plans(cache=cache)
        if not plans:
            return None

        # Build a stable synthetic query (timing depends on shapes, not values).
        B = 1
        try:
            ks = getattr(cache, "k_sem", None)
            if ks is not None:
                if getattr(ks, "buf", None) is not None:
                    B = int(ks.buf.shape[0])
                elif getattr(ks, "q", None) is not None:
                    B = int(ks.q.shape[0])
        except Exception:
            B = 1

        H = int(getattr(attn, "H", 1))
        q_sem = torch.randn((B, H, 1, int(attn.sem_head_dim)), device=self.device, dtype=torch.float16)
        q_geo = torch.randn((B, H, 1, int(attn.geo_head_dim)), device=self.device, dtype=torch.float16)
        sem_scale = 1.0 / math.sqrt(float(attn.sem_head_dim))
        geo_scale = 1.0 / math.sqrt(float(attn.geo_head_dim))

        baseline_plan = self._plans.get(k, KVDecodePlan(fused="none", decode_block=self.base_decode_block))
        baseline_out = None
        if self.cfg.verify:
            try:
                baseline_out = self._run_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=baseline_plan,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                ).detach()
            except Exception:
                baseline_out = None

        best_plan: Optional[KVDecodePlan] = None
        best_ms: float = float("inf")

        for p in plans:
            try:
                ms = self._bench_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=p,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                    baseline_out=baseline_out,
                )
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[selfopt] plan failed {p}: {e}")
                ms = float("inf")
            if ms < best_ms:
                best_ms = ms
                best_plan = p

        if best_plan is not None and k in self._plans and self.cfg.mode == "online":
            old = self._plans[k]
            try:
                old_ms = self._bench_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=old,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                    baseline_out=baseline_out,
                )
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
        attn: Any,
        cache: Any,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        sem_scale: float,
        geo_scale: float,
    ) -> Optional[KVDecodePlan]:
        """Backward-compatible alias for `maybe_get_plan` (query values do not affect timing)."""
        _ = (q_sem, q_geo, sem_scale, geo_scale)
        L_prefix = int(getattr(cache, "pos", 0))
        return self.maybe_get_plan(attn=attn, cache=cache, L_prefix=L_prefix)


@dataclass(frozen=True)
class KVCachePolicy:
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
        from production.kvcache_backend import KVCacheTensorConfig

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

    @classmethod
    def parse(cls, s: str) -> "KVCachePolicy":
        """Parse an atomic cache policy string.

        Canonical format (round-trips with `short()`):
          ksem=<kind>@<qblock>,kgeo=<kind>@<qblock>,v=<kind>@<qblock>,resid=<int>
        """
        raw = str(s).strip()
        if not raw:
            raise ValueError("Empty kv-policy string")

        def norm_key(k: str) -> str:
            return str(k).strip().lower().replace("_", "")

        def parse_kind_qblock(val: str) -> Tuple["KVCacheKind", int]:
            v = str(val).strip().lower()
            if "@" in v:
                kind_s, qb_s = v.split("@", 1)
                kind = str(kind_s).strip()
                qb = int(str(qb_s).strip())
            else:
                kind = str(v).strip()
                qb = 32
            if kind not in ("fp16", "fp32", "q8_0", "q4_0", "nf4"):
                raise ValueError(f"Unknown KV cache kind: {kind}")
            if qb <= 0:
                raise ValueError(f"qblock must be > 0 (got {qb})")
            return kind, int(qb)  # type: ignore[return-value]

        items: Dict[str, str] = {}
        for part in [p.strip() for p in raw.split(",") if p.strip()]:
            if "=" not in part:
                raise ValueError(f"Invalid kv-policy field (expected key=value): {part!r}")
            k, v = part.split("=", 1)
            nk = norm_key(k)
            items[nk] = str(v).strip()

        # Support a few aliases for convenience.
        ksem_s = items.get("ksem", None) or items.get("ksemkind", None)
        kgeo_s = items.get("kgeo", None) or items.get("kgeokind", None)
        v_s = items.get("v", None) or items.get("vkind", None)
        # Note: keys are normalized (lower + underscores removed), so `residual_len` becomes `residuallen`.
        resid_s = items.get("resid", None) or items.get("residual", None) or items.get("residuallen", None)

        missing: List[str] = []
        if ksem_s is None:
            missing.append("ksem")
        if kgeo_s is None:
            missing.append("kgeo")
        if v_s is None:
            missing.append("v")
        if resid_s is None:
            missing.append("resid")
        if missing:
            raise ValueError(f"Missing kv-policy fields: {', '.join(missing)}")

        ksem_kind, ksem_qb = parse_kind_qblock(ksem_s)
        kgeo_kind, kgeo_qb = parse_kind_qblock(kgeo_s)
        v_kind, v_qb = parse_kind_qblock(v_s)
        resid = int(str(resid_s).strip())
        if resid < 0:
            raise ValueError(f"resid must be >= 0 (got {resid})")

        return cls(
            k_sem_kind=ksem_kind,
            k_geo_kind=kgeo_kind,
            v_kind=v_kind,
            k_sem_qblock=int(ksem_qb),
            k_geo_qblock=int(kgeo_qb),
            v_qblock=int(v_qb),
            residual_len=int(resid),
        )


def estimate_seq_cache_bytes(*, batch_size: int, max_seq_len: int, dim: int, cfg: "KVCacheTensorConfig") -> int:
    from production.kvcache_backend import make_quantspec

    B = int(batch_size)
    L = int(max_seq_len)
    D = int(dim)
    kind = str(cfg.kind)
    if kind == "fp16":
        return B * L * D * 2
    if kind == "fp32":
        return B * L * D * 4

    spec = make_quantspec(cfg.kind, dim, cfg.qblock)

    if kind == "q8_0":
        q_bytes = B * L * spec.pad_dim * 1
        s_bytes = B * L * spec.n_blocks * 2
    elif kind in ("q4_0", "nf4"):
        q_bytes = B * L * (spec.pad_dim // 2) * 1
        s_bytes = B * L * spec.n_blocks * 2
    else:
        raise ValueError(kind)

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
    """Pick a cache policy that fits a strict memory budget and improves decode throughput."""

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

        self._cache_path = cfg.cache_path
        self._policy_cache: Dict[str, Dict[str, Any]] = {}
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r") as f:
                    root = json.load(f)
                if isinstance(root, dict):
                    cp = root.get("cache_policies", {})
                    self._policy_cache = dict(cp) if isinstance(cp, dict) else {}
            except Exception:
                self._policy_cache = {}

    def _save_policy_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
            root: Dict[str, Any] = {"version": 26, "cache_policies": self._policy_cache}
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

    def update_cached_policy(self, policy: "KVCachePolicy") -> None:
        """Overwrite the persisted cache policy for this hardware/model key.

        Used when a chosen policy is later rejected by model-level quality gates, to avoid repeatedly
        selecting the same known-bad cached policy on subsequent runs.
        """
        try:
            key = self._policy_key()
            self._policy_cache[key] = asdict(policy)
            self._save_policy_cache()
        except Exception:
            # Cache writes must never break inference.
            pass

    def _policy_key(self) -> str:
        max_bucket = _pow2_bucket(self.max_seq_len)
        dims = f"sem={self.model_cfg.sem_dim},geo={self.model_cfg.geo_dim},v={self.model_cfg.attn_dim},H={self.model_cfg.n_head}"
        return f"{_device_sig(self.device)}|decoupled|max={max_bucket}|B={self.batch_size}|{dims}"

    def _budget_bytes(self) -> int:
        if self.cfg.mem_budget_mb is not None:
            return int(float(self.cfg.mem_budget_mb) * 1024.0 * 1024.0)
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
        from production.kvcache_backend import DecoupledLayerKVCache

        B = self.batch_size
        H = int(getattr(self.attn, "H", self.model_cfg.n_head))
        sem_hd = int(getattr(self.attn, "sem_head_dim", self.model_cfg.sem_dim // H))
        geo_hd = int(getattr(self.attn, "geo_head_dim", self.model_cfg.geo_dim // H))
        v_hd = int(getattr(self.attn, "v_head_dim", self.model_cfg.attn_dim // H))
        _ = v_hd  # for parity with v30 locals

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

        with torch.no_grad():
            k_sem = torch.randn((B, L_prefix, self.model_cfg.sem_dim), device=self.device, dtype=torch.float16)
            k_geo = torch.randn((B, L_prefix, self.model_cfg.geo_dim), device=self.device, dtype=torch.float16)
            v = torch.randn((B, L_prefix, self.model_cfg.attn_dim), device=self.device, dtype=torch.float16)
            cache.append(k_sem, k_geo, v)

        q_sem = torch.randn((B, H, 1, sem_hd), device=self.device, dtype=torch.float16)
        q_geo = torch.randn((B, H, 1, geo_hd), device=self.device, dtype=torch.float16)
        sem_scale = 1.0 / math.sqrt(float(sem_hd))
        geo_scale = 1.0 / math.sqrt(float(geo_hd))

        fused_menu: List[str]
        if self.base_fused in ("triton1pass", "triton2pass"):
            fused_menu = [self.base_fused]
        elif self.base_fused == "none":
            fused_menu = ["none"]
        else:
            fused_menu = ["none"]
            if self._supports_fused_q4q8q4(policy):
                fused_menu += ["triton1pass", "triton2pass"]

        decode_blocks = (self.cfg.decode_blocks if self.cfg.scope in ("decode", "all") else (self.base_decode_block,))
        decode_blocks = tuple(int(x) for x in decode_blocks if int(x) > 0)

        best = float("inf")
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
            base_fused="auto",
            base_decode_block=self.base_decode_block,
        )

        for fused in fused_menu:
            for db in decode_blocks:
                plan = KVDecodePlan(fused=fused, decode_block=db)
                try:
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

        # Hypothesis-challenging neighbor: swap semantic/geometric kinds.
        # Motivation: RoPE (geometric path) is often more sensitive to quantization error; by explicitly
        # testing the swap, the tuner can empirically validate (or invalidate) this assumption rather
        # than only exploring monotonic kind tweaks.
        if str(p.k_sem_kind) != str(p.k_geo_kind):
            out.append(
                KVCachePolicy(
                    k_sem_kind=p.k_geo_kind,
                    k_geo_kind=p.k_sem_kind,
                    v_kind=p.v_kind,
                    k_sem_qblock=p.k_sem_qblock,
                    k_geo_qblock=p.k_geo_qblock,
                    v_qblock=p.v_qblock,
                    residual_len=p.residual_len,
                )
            )

        for r in neigh_num(int(p.residual_len), resid_cands):
            out.append(
                KVCachePolicy(
                    k_sem_kind=p.k_sem_kind,
                    k_geo_kind=p.k_geo_kind,
                    v_kind=p.v_kind,
                    k_sem_qblock=p.k_sem_qblock,
                    k_geo_qblock=p.k_geo_qblock,
                    v_qblock=p.v_qblock,
                    residual_len=r,
                )
            )

        for qb in neigh_num(int(p.k_sem_qblock), qb_cands):
            out.append(
                KVCachePolicy(
                    k_sem_kind=p.k_sem_kind,
                    k_geo_kind=p.k_geo_kind,
                    v_kind=p.v_kind,
                    k_sem_qblock=qb,
                    k_geo_qblock=qb,
                    v_qblock=qb,
                    residual_len=p.residual_len,
                )
            )

        for k in neigh_kind(str(p.k_sem_kind), cfg.k_sem_kinds):
            out.append(
                KVCachePolicy(
                    k_sem_kind=k,
                    k_geo_kind=p.k_geo_kind,
                    v_kind=p.v_kind,
                    k_sem_qblock=p.k_sem_qblock,
                    k_geo_qblock=p.k_geo_qblock,
                    v_qblock=p.v_qblock,
                    residual_len=p.residual_len,
                )
            )
        for k in neigh_kind(str(p.k_geo_kind), cfg.k_geo_kinds):
            out.append(
                KVCachePolicy(
                    k_sem_kind=p.k_sem_kind,
                    k_geo_kind=k,
                    v_kind=p.v_kind,
                    k_sem_qblock=p.k_sem_qblock,
                    k_geo_qblock=p.k_geo_qblock,
                    v_qblock=p.v_qblock,
                    residual_len=p.residual_len,
                )
            )
        for k in neigh_kind(str(p.v_kind), cfg.v_kinds):
            out.append(
                KVCachePolicy(
                    k_sem_kind=p.k_sem_kind,
                    k_geo_kind=p.k_geo_kind,
                    v_kind=k,
                    k_sem_qblock=p.k_sem_qblock,
                    k_geo_qblock=p.k_geo_qblock,
                    v_qblock=p.v_qblock,
                    residual_len=p.residual_len,
                )
            )

        uniq: List[KVCachePolicy] = []
        seen = set()
        for cand in out:
            key = cand.short()
            if key not in seen:
                seen.add(key)
                uniq.append(cand)
        return uniq

    def _print_policy_summary(
        self,
        *,
        policy: KVCachePolicy,
        L: int,
        best_ms: float,
        budget_bytes: int,
        policy_bytes: int,
        note: str,
    ) -> None:
        # Always surface the final decision in non-verbose mode too (high-level accountability).
        print(
            f"[selfopt] cache-policy {note}: {policy.short()} "
            f"(mem={_as_mb(policy_bytes):.1f}MB <= {_as_mb(budget_bytes):.1f}MB, "
            f"L={int(L)}, best_ms={float(best_ms):.3f})"
        )

    def choose_policy(self, *, prompt_len: int) -> KVCachePolicy:
        if self.cfg.mode == "none":
            return self.base_policy
        if getattr(self.model_cfg, "attn_mode", "standard") != "decoupled":
            return self.base_policy

        key = self._policy_key()
        budget = self._budget_bytes()

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

        if key in self._policy_cache:
            try:
                p = self._policy_cache[key]
                policy = KVCachePolicy(**p)
                ms = float("nan")
                pb = mem_bytes(policy)
                self._print_policy_summary(policy=policy, L=min(self.max_seq_len - 1, max(1, prompt_len)), best_ms=ms, budget_bytes=budget, policy_bytes=pb, note="cached")
                return policy
            except Exception:
                pass

        if self.cfg.policy_prefix_len is not None:
            L = int(self.cfg.policy_prefix_len)
        else:
            L = int(min(self.max_seq_len - 1, max(1024, _pow2_bucket(prompt_len))))
        L = max(1, min(L, self.max_seq_len - 1))

        cur = self.base_policy
        best = cur
        best_ms = float("inf")

        if not ok_mem(cur):
            cur = KVCachePolicy(
                k_sem_kind=cur.k_sem_kind,
                k_geo_kind=cur.k_geo_kind,
                v_kind=cur.v_kind,
                k_sem_qblock=cur.k_sem_qblock,
                k_geo_qblock=cur.k_geo_qblock,
                v_qblock=cur.v_qblock,
                residual_len=0,
            )

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

            scored: List[Tuple[float, int, KVCachePolicy]] = []
            for p in candidates:
                try:
                    ms = self._bench_policy_ms(p, L_prefix=L)
                except Exception:
                    ms = float("inf")
                scored.append((ms, mem_bytes(p), p))

            scored.sort(key=lambda x: (x[0], x[1]))

            for ms, _mb, p in scored:
                if ms < best_ms * (1.0 - float(self.cfg.policy_hysteresis)):
                    if self.cfg.verbose:
                        print(f"[selfopt] cache-policy step: {best.short()} -> {p.short()} ({best_ms:.3f}ms -> {ms:.3f}ms, mem={_as_mb(_mb):.1f}MB)")
                    best = p
                    best_ms = ms
                    improved = True
                    break

        if self.cfg.prefer_lower_mem_within > 0 and best_ms < float("inf"):
            cur_ms = best_ms
            cur_mem = mem_bytes(best)
            for p in [p for p in self._neighbors(best) if ok_mem(p)]:
                try:
                    ms = self._bench_policy_ms(p, L_prefix=L)
                except Exception:
                    continue
                if ms <= cur_ms * (1.0 + float(self.cfg.prefer_lower_mem_within)):
                    m = mem_bytes(p)
                    if m < cur_mem:
                        if self.cfg.verbose:
                            print(
                                f"[selfopt] cache-policy tie-break: {best.short()} -> {p.short()} "
                                f"(ms={ms:.3f} within {self.cfg.prefer_lower_mem_within*100:.1f}%, mem {_as_mb(cur_mem):.1f}MB -> {_as_mb(m):.1f}MB)"
                            )
                        best = p
                        cur_mem = m

        self._policy_cache[key] = asdict(best)
        self._save_policy_cache()

        self._print_policy_summary(policy=best, L=L, best_ms=best_ms, budget_bytes=budget, policy_bytes=mem_bytes(best), note="chosen")
        return best


def policy_quality_reject_reasons(
    metrics: Dict[str, float],
    *,
    max_abs_logit_tol: Optional[float],
    delta_nll_tol: Optional[float],
    ppl_ratio_tol: Optional[float],
    kl_tol: Optional[float],
) -> List[str]:
    """Convert quality metrics into human-readable reject reasons.

    This is intentionally pure: callers decide whether to compute metrics and whether to reject.
    """
    out: List[str] = []
    if max_abs_logit_tol is not None:
        mx = float(metrics.get("max_abs_logit", float("nan")))
        if not math.isnan(mx) and mx > float(max_abs_logit_tol):
            out.append(f"max_abs_logit={mx:.4g} > {float(max_abs_logit_tol):.4g}")
    if delta_nll_tol is not None:
        dnll = float(metrics.get("delta_nll", float("nan")))
        if not math.isnan(dnll) and dnll > float(delta_nll_tol):
            out.append(f"ΔNLL={dnll:.4g} > {float(delta_nll_tol):.4g} nats/tok")
    if ppl_ratio_tol is not None:
        pr = float(metrics.get("ppl_ratio", float("nan")))
        if not math.isnan(pr) and pr > float(ppl_ratio_tol):
            out.append(f"ppl_ratio={pr:.4g} > {float(ppl_ratio_tol):.4g}")
    if kl_tol is not None:
        klv = float(metrics.get("kl_base_cand", float("nan")))
        if not math.isnan(klv) and klv > float(kl_tol):
            out.append(f"KL={klv:.4g} > {float(kl_tol):.4g} nats/tok")
    return out


def warn_policy_quality_reject(*, chosen: str, fallback: str, reasons: List[str]) -> None:
    """Non-verbose warning when a candidate policy is rejected by quality guardrails."""
    if not reasons:
        return
    msg = "; ".join(reasons)
    print(f"[warn] selfopt cache-policy rejected: {chosen} -> fallback {fallback} ({msg})")
