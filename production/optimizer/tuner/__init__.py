"""Runtime tuning / self-optimization (public API).

This package is the long-term home for KV-cache policy selection and decode-plan
tuning. It mirrors the previous `production.runtime_tuning_impl` surface while
we migrate implementation details into smaller, class-based modules.
"""

from __future__ import annotations

# Small, stable building blocks live in this package.
from production.optimizer.tuner.buckets import pow2_bucket
from production.optimizer.tuner.cache_estimates import (
    as_mb,
    estimate_decoupled_kvcache_bytes,
    estimate_seq_cache_bytes,
)
from production.optimizer.tuner.cache_policy import KVCachePolicy
from production.optimizer.tuner.config import KVSelfOptConfig
from production.optimizer.tuner.decode_plan import KVDecodePlan
from production.optimizer.tuner.decode_optimizer import KVDecodeSelfOptimizer
from production.optimizer.tuner.policy_optimizer import KVCachePolicySelfOptimizer
from production.optimizer.tuner.profiles import (
    TritonKernelProfile,
    parse_cc_from_device_sig,
    get_triton_kernel_profiles,
)
from production.optimizer.tuner.quality import (
    policy_quality_reject_reasons,
    warn_policy_quality_reject,
)
from production.optimizer.tuner.token_spec import load_token_ids_spec
from production.optimizer.tuner.triton_availability import (
    TRITON_AVAILABLE,
    triton_decoupled_q4q8q4_available,
)

# (migrated) cache-policy optimizer lives in this package.

__all__ = [
    "TRITON_AVAILABLE",
    "KVCachePolicy",
    "KVCachePolicySelfOptimizer",
    "KVDecodePlan",
    "KVDecodeSelfOptimizer",
    "KVSelfOptConfig",
    "TritonKernelProfile",
    "as_mb",
    "parse_cc_from_device_sig",
    "pow2_bucket",
    "triton_decoupled_q4q8q4_available",
    "estimate_decoupled_kvcache_bytes",
    "estimate_seq_cache_bytes",
    "get_triton_kernel_profiles",
    "load_token_ids_spec",
    "policy_quality_reject_reasons",
    "warn_policy_quality_reject",
]


