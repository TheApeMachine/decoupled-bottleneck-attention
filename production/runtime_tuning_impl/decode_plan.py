"""Back-compat wrapper (migrated).

Prefer importing from `production.optimizer.tuner.decode_plan`.
"""

from __future__ import annotations

from production.optimizer.tuner.decode_plan import KVDecodePlan

__all__ = ["KVDecodePlan"]

