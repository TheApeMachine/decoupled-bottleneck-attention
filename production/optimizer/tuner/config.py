"""Configuration for runtime tuning / self-optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from production.kvcache_backend import KVCacheKind


@dataclass
class KVSelfOptConfig:
    """Configuration for the KV self-optimizer."""

    mode: Literal["none", "startup", "online"] = "none"

    # which knobs to optimize:
    # - decode: decode_block, fused kernel choice, launch params
    # - cache : kv_residual length, quantization kinds, qblock sizes (startup only)
    # - all   : both
    scope: Literal["decode", "cache", "all"] = "all"

    # ---------------------------
    # Decode-plan tuning
    # ---------------------------
    decode_blocks: tuple[int, ...] = (256, 512, 1024, 2048)
    block_ns: tuple[int, ...] = (128,)
    warps: tuple[int, ...] = (4, 8)
    stages: tuple[int, ...] = (2, 3)

    # Hierarchical decode tuning: use internal kernel profiles to avoid combinatorial explosion.
    kernel_profiles: Literal["auto", "small", "off"] = "auto"
    expert_launch_space: bool = False

    warmup: int = 1
    iters: int = 3

    # online mode: at most once every N decode steps per bucket, try a neighbor and keep if faster.
    interval: int = 256
    hysteresis: float = 0.03

    cache_path: str | None = None
    verbose: bool = False

    # Optional correctness guardrail when comparing decode candidates.
    verify: bool = False
    verify_tol: float = 5e-3

    # ---------------------------
    # Cache-policy tuning
    # ---------------------------
    residuals: tuple[int, ...] = (0, 32, 64, 128)
    qblocks: tuple[int, ...] = (16, 32, 64)

    k_sem_kinds: tuple[KVCacheKind, ...] = ("q4_0", "nf4", "q8_0", "fp16")
    k_geo_kinds: tuple[KVCacheKind, ...] = ("q8_0", "q4_0", "fp16")
    v_kinds: tuple[KVCacheKind, ...] = ("q4_0", "q8_0", "fp16")

    mem_budget_mb: float | None = None
    mem_overhead_frac: float = 0.10

    policy_prefix_len: int | None = None
    policy_warmup: int = 1
    policy_iters: int = 3
    policy_hysteresis: float = 0.02
    prefer_lower_mem_within: float = 0.02

    # Optional quality guard for policy tuning (slow).
    policy_quality: bool = False
    calib_tokens: str | None = None
    calib_prefill: int = 128
    calib_decode_steps: int = 32
    quality_tol: float = 0.5
    quality_delta_nll_tol: float | None = 0.02
    quality_ppl_ratio_tol: float | None = 1.02
    quality_kl_tol: float | None = None
    quality_compute_kl: bool = False

    # Optional long-horizon quality gate (final accept check; much slower).
    policy_quality_long: bool = False
    calib_long_tokens: str | None = None
    calib_long_prefill: int = 4096
    calib_long_decode_steps: int = 128
    quality_long_tol: float | None = None
    quality_long_delta_nll_tol: float | None = None
    quality_long_ppl_ratio_tol: float | None = None
    quality_long_kl_tol: float | None = None
    quality_long_compute_kl: bool = False

    # Optional "repair" path for cache-policy tuning.
    layerwise_cache: bool = False

    # Speculative decoding control loop (optional; used by generation code).
    spec_enabled: bool = False
    spec_k: tuple[int, ...] = (2, 4, 6, 8)
    spec_min_accept: float = 0.6
    spec_probe_every: int = 64

    # Training-time long-seq attention approximation (decoupled mode).
    train_long_seq_enabled: bool = True
    train_long_seq_threshold: int | None = None
    train_long_seq_mem_block: int | None = None
    train_long_seq_local_window: int | None = None
    train_long_seq_q_chunk: int | None = None


