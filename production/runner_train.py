"""Training loop entry point (public API).

Why this exists:
- Keep the public import path stable (`production.runner_train.run_train`).
- The implementation lives in `production/runner_train_impl/` so we can split
  it into small modules without churn for callers.
"""

from __future__ import annotations

from production.runner_train_impl import run_train

__all__ = ["run_train"]


