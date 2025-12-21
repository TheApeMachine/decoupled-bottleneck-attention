"""
Training runner implementation (internal).

`production/runner_train.py` is a stable public import path used by `production/runner.py`.
The training loop is large by nature; we move implementation here so the public
module can stay tiny while we carve internals into small, testable pieces.
"""

from __future__ import annotations

from production.runner_train_impl.run import run_train

__all__ = ["run_train"]
