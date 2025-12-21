"""Public API for intent-first dynamic configuration.

Why this exists:
- Keeps the public import path stable (`production.optimizer.apply.DynamicConfigApplier`).
- The implementation lives in `production/optimizer/apply_impl/` so we can keep
  files small and responsibilities crisp.
"""

from __future__ import annotations

from production.optimizer.apply_impl.applier import DynamicConfigApplier

__all__ = ["DynamicConfigApplier"]


