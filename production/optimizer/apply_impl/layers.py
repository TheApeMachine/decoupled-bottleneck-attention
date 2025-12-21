"""Layer count derivation.

Why this exists:
- Many CLI flows omit `--layers` by design.
- We infer depth from existing run artifacts (out_dir), otherwise from capacity
  heuristics that keep head_dim sane across devices.
"""

from __future__ import annotations

import argparse
from typing import cast

from production.optimizer.heuristics import HeuristicPlanner
from production.optimizer.watch import OptimizerLike


class LayersDeriver:
    """Derive `layers` into the optimizer graph."""

    @staticmethod
    def apply(o: OptimizerLike, *, _args: argparse.Namespace) -> None:
        """Why: keep architecture inference working without exposing depth flags."""
        raw_layers = o.maybe("layers", None)
        try:
            if raw_layers is not None and int(cast(int | float | str, raw_layers)) > 0:
                o.set("layers", int(cast(int | float | str, raw_layers)))
                o.set("layers_source", "override")
                return
        except (ValueError, TypeError):
            pass

        out_dir = o.get("out_dir")
        if out_dir:
            try:
                from production.config import infer_layers_from_out_dir  # pylint: disable=import-outside-toplevel

                inferred = infer_layers_from_out_dir(str(out_dir))
                if inferred is not None and int(inferred) > 0:
                    o.set("layers", int(inferred))
                    o.set("layers_source", "out_dir")
                    return
            except (ValueError, TypeError, AttributeError, ImportError):
                pass

        target_params = int(cast(int, o.get("target_params")))
        L = HeuristicPlanner.choose_layers(  # pylint: disable=invalid-name
            target_params=target_params, device_type=str(o.get("device_type"))
        )
        o.set("layers", int(L))
        o.set("layers_source", "auto")


