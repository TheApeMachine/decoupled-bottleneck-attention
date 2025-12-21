"""Back-compat wrapper (migrated).

Prefer importing from `production.optimizer.tuner.token_spec`.
"""

from __future__ import annotations

from production.optimizer.tuner.token_spec import load_token_ids_spec

__all__ = ["load_token_ids_spec"]

