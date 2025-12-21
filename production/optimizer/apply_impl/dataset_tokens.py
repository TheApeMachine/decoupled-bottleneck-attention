"""Dataset token-count derivation for intent-first config.

Why this exists:
- Many downstream decisions (model size, training schedule) need a dataset scale.
- We avoid scanning datasets; we infer from filename conventions or `.meta`.
"""

from __future__ import annotations

import argparse
from typing import cast

from production.optimizer.dataset_tokens import DatasetTokenCounter
from production.optimizer.watch import OptimizerLike


class DatasetTokensDeriver:
    """Derive dataset token-count (and its source) into the optimizer graph."""

    @staticmethod
    def apply(o: OptimizerLike, *, _args: argparse.Namespace) -> None:
        """Why: later stages need a token scale even when the user doesn't pass one."""
        data_path = cast(str | None, o.maybe("data", None))
        tokens, src = DatasetTokenCounter.infer_with_source(data_path)
        o.set("dataset_tokens", (int(tokens) if tokens is not None else None))
        o.set("dataset_tokens_source", str(src))


