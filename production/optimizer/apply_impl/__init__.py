"""Dynamic config application internals.

Why this exists:
- We want `production/optimizer/apply.py` to stay tiny (stable import path).
- The implementation evolves frequently, so splitting into focused modules keeps
  changes local and reviewable.
"""

from __future__ import annotations


