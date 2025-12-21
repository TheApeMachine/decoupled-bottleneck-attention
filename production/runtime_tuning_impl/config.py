"""Back-compat wrapper (migrated).

Prefer importing from `production.optimizer.tuner.config`.
"""

from __future__ import annotations

from production.optimizer.tuner.config import KVSelfOptConfig

__all__ = ["KVSelfOptConfig"]

