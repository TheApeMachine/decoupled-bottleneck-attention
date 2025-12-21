"""Public API for runtime KV-cache tuning.

Implementation is split into `production/runtime_tuning_impl/` to keep this module small and
decoupled.
"""

from __future__ import annotations

from production.optimizer.tuner import (  # preferred home for this API
    TRITON_AVAILABLE,
    KVCachePolicy,
    KVCachePolicySelfOptimizer,
    KVDecodePlan,
    KVDecodeSelfOptimizer,
    KVSelfOptConfig,
    TritonKernelProfile,
    as_mb,
    parse_cc_from_device_sig,
    pow2_bucket,
    triton_decoupled_q4q8q4_available,
    estimate_decoupled_kvcache_bytes,
    estimate_seq_cache_bytes,
    get_triton_kernel_profiles,
    load_token_ids_spec,
    policy_quality_reject_reasons,
    warn_policy_quality_reject,
)

# Back-compat aliases (avoid importing underscore names across modules).
_as_mb = as_mb
_parse_cc_from_device_sig = parse_cc_from_device_sig
_pow2_bucket = pow2_bucket
_triton_decoupled_q4q8q4_available = triton_decoupled_q4q8q4_available

__all__ = [
    "TRITON_AVAILABLE",
    "KVCachePolicy",
    "KVCachePolicySelfOptimizer",
    "KVDecodePlan",
    "KVDecodeSelfOptimizer",
    "KVSelfOptConfig",
    "TritonKernelProfile",
    "_as_mb",
    "_parse_cc_from_device_sig",
    "_pow2_bucket",
    "_triton_decoupled_q4q8q4_available",
    "estimate_decoupled_kvcache_bytes",
    "estimate_seq_cache_bytes",
    "get_triton_kernel_profiles",
    "load_token_ids_spec",
    "policy_quality_reject_reasons",
    "warn_policy_quality_reject",
]
