"""Dynamic config applier (orchestrator).

Why this exists:
- The optimizer graph is a small dependency resolver; this class wires the
  derivation stages together in one place.
- Keeping orchestration separate from individual derivations makes it easy to
  add/remove stages without touching their logic.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Callable
from typing import cast

from production.optimizer.graph import Optimizer
from production.optimizer.result_types import ResultTypeMapper

from production.optimizer.apply_impl.args import arg
from production.optimizer.apply_impl.base_defaults import BaseDefaults
from production.optimizer.apply_impl.dataset_tokens import DatasetTokensDeriver
from production.optimizer.apply_impl.layers import LayersDeriver
from production.optimizer.apply_impl.summary import SelfOptSummary, TrainAssertions
from production.optimizer.apply_impl.target_params import TargetParamsDeriver
from production.optimizer.apply_impl.train_defaults import TrainingDefaultsDeriver
from production.optimizer.watch import OptimizerLike


class DynamicConfigApplier:
    """Populate runner-required fields from intent, without expanding CLI surface area."""

    @staticmethod
    def apply(args: argparse.Namespace, *, device: object) -> None:
        """Why: convert high-level intent into a fully-populated runtime config."""
        device_type = str(arg(device, "type", "cpu") or "cpu")
        opt = Optimizer()

        mode0 = str(arg(args, "mode", "train"))
        opt.set("mode", mode0)
        opt.set("data", arg(args, "data", None))
        opt.set("out_dir", arg(args, "out_dir", None))
        opt.set("seed", int(cast(int, arg(args, "seed", 1337))))
        opt.set("device_type", device_type)

        exp = arg(args, "exp", None)
        result = arg(args, "result", None)
        exp_source = "exp" if exp is not None else ("result" if result is not None else "")
        if exp is None and result is not None:
            exp = ResultTypeMapper.to_exp(str(result))
        elif exp is not None:
            exp = ResultTypeMapper.to_exp(str(exp))
        opt.set("exp", exp)
        opt.set("exp_source", exp_source)

        opt.set("size", arg(args, "size", None))
        opt.set("layers", arg(args, "layers", None))

        opt.when_ready(
            ["mode", "out_dir"],
            cast(Callable[[OptimizerLike], None], functools.partial(BaseDefaults.apply, args=args)),
            name="base_defaults",
        )

        if mode0 == "train":
            opt.when_ready(
                ["data"],
                cast(Callable[[OptimizerLike], None], functools.partial(DatasetTokensDeriver.apply, _args=args)),
                name="dataset_tokens_from_data",
            )
            opt.when_ready(
                ["size", "dataset_tokens"],
                cast(Callable[[OptimizerLike], None], functools.partial(TargetParamsDeriver.apply, _args=args)),
                name="target_params",
            )
            opt.when_ready(
                ["layers", "out_dir", "target_params", "device_type"],
                cast(Callable[[OptimizerLike], None], functools.partial(LayersDeriver.apply, _args=args)),
                name="layers",
            )
            opt.when_ready(
                ["exp", "dataset_tokens", "target_params", "layers", "device_type", "seed", "out_dir"],
                cast(Callable[[OptimizerLike], None], functools.partial(TrainingDefaultsDeriver.apply, args=args)),
                name="train_defaults",
            )

        opt.apply_to_args(args)

        if mode0 == "train":
            # Why: preserve existing “intent presets” behavior for experiments while keeping CLI small.
            from production.config import apply_exp_preset, apply_intent  # pylint: disable=import-outside-toplevel

            apply_intent(args)
            apply_exp_preset(args)
            TrainAssertions.assert_required(args)

        # Minimal defaults (safe even for non-train modes).
        if not hasattr(args, "tokenizer"):
            args.tokenizer = "tiktoken"
        if not hasattr(args, "instrument"):
            args.instrument = "off"
        if not hasattr(args, "live_plot"):
            args.live_plot = False
        if not hasattr(args, "tb"):
            args.tb = False

        if getattr(args, "exp", None):
            args.exp = ResultTypeMapper.to_exp(str(arg(args, "exp")))

        SelfOptSummary.populate(args=args, device_type=device_type)


