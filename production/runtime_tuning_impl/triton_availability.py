"""Back-compat wrapper (migrated).

Prefer importing from `production.optimizer.tuner.triton_availability`.
"""

from __future__ import annotations

from production.optimizer.tuner.triton_availability import (
    TRITON_AVAILABLE,
    triton_decoupled_q4q8q4_available as _triton_decoupled_q4q8q4_available,
)

__all__ = ["TRITON_AVAILABLE", "_triton_decoupled_q4q8q4_available"]

