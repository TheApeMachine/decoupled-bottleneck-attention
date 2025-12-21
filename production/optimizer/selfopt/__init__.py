"""Self-optimization controller (public API).

This package is the long-term home for always-on runtime planning (dtype/AMP,
batch sizing, torch.compile decisions). The legacy `production.selfopt_controller`
module remains as a thin wrapper for compatibility.
"""

from __future__ import annotations

from production.optimizer.selfopt.controller import SelfOptController
from production.optimizer.selfopt.types import RuntimePlan

__all__ = [
    "RuntimePlan",
    "SelfOptController",
]


