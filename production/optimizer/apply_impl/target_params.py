"""Target parameter budget derivation.

Why this exists:
- Architecture derivation needs a single “capacity budget” scalar.
- Users may specify it directly (via `--size`) or implicitly via dataset tokens.
"""

from __future__ import annotations

import argparse
from typing import cast

from production.optimizer.counts import CountCodec
from production.optimizer.watch import OptimizerLike


class TargetParamsDeriver:
    """Derive `target_params` (and provenance) into the optimizer graph."""

    @staticmethod
    def apply(o: OptimizerLike, *, _args: argparse.Namespace) -> None:
        """Why: unify direct size overrides and dataset-derived sizing into one field."""
        size_raw = o.maybe("size", None)
        size = CountCodec.parse(size_raw)
        if size_raw is not None and size is None:
            raise ValueError(f"Unparseable --size {size_raw!r}. Use e.g. 100m, 1b, 1.5b, 2e9.")
        if size is not None:
            o.set("target_params", int(size))
            o.set("target_params_source", "size")
            return

        dtok_raw = o.get("dataset_tokens")
        if dtok_raw is None:
            raise ValueError(
                "Could not infer dataset token count from --data path. "
                + "Name it like `fineweb_20b.npy`, provide a sibling `.meta` with `tokens: ...`, "
                + "or pass `--size` explicitly."
            )
        dtok = int(cast(int, dtok_raw))

        # Chinchilla-ish heuristic: tokens ≈ 20 × params.
        tokens_per_param = 20.0
        o.set("tokens_per_param", float(tokens_per_param))
        o.set("target_params", int(max(1.0, float(dtok) / tokens_per_param)))
        o.set("target_params_source", f"dataset_tokens/{tokens_per_param:g}")


