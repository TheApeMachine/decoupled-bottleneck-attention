"""Back-compat wrapper (migrated).

Prefer importing from `production.optimizer.tuner.cache_estimates`.
"""

from __future__ import annotations

from production.optimizer.tuner.cache_estimates import (
    as_mb as _as_mb,
    estimate_decoupled_kvcache_bytes,
    estimate_seq_cache_bytes,
)

__all__ = [
    "_as_mb",
    "estimate_decoupled_kvcache_bytes",
    "estimate_seq_cache_bytes",
]

