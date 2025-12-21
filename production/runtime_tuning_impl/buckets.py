"""Back-compat wrapper (migrated).

Prefer importing from `production.optimizer.tuner.buckets`.
"""

from __future__ import annotations

from production.optimizer.tuner.buckets import pow2_bucket as _pow2_bucket

__all__ = ["_pow2_bucket"]

