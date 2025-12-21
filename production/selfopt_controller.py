"""Public API for always-on runtime self-optimization.

Implementation lives in `production/optimizer/selfopt/` to keep modules small.
"""

from __future__ import annotations

from production.optimizer.selfopt import RuntimePlan, SelfOptController

__all__ = [
    "RuntimePlan",
    "SelfOptController",
]


