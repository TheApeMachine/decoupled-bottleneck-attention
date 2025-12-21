"""Back-compat wrapper (migrated).

Prefer importing from `production.optimizer.tuner.cache_policy`.
"""

from __future__ import annotations

from production.optimizer.tuner.cache_policy import KVCachePolicy

__all__ = ["KVCachePolicy"]

