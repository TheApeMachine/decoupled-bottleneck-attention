"""Self-optimization summary injection.

Why this exists:
- We want runs to record *what was chosen and why* without increasing CLI flags.
- The summary travels with args and ends up in logs/artifacts for reproducibility.
"""

from __future__ import annotations

import argparse
from typing import cast


class SelfOptSummary:
    """Attach a small, structured summary of derived values onto args."""

    @staticmethod
    def populate(*, args: argparse.Namespace, device_type: str) -> None:
        """Why: make dynamic decisions observable for debugging and comparisons."""
        try:
            head_dim = None
            try:
                dm = int(getattr(args, "d_model", 0) or 0)
                nh = int(getattr(args, "n_head", 0) or 0)
                if dm > 0 and nh > 0:
                    head_dim = int(dm // nh)
            except (ValueError, TypeError, ZeroDivisionError):
                head_dim = None

            args.selfopt_summary = {
                "mode": str(getattr(args, "mode", "")),
                "device_type": str(device_type),
                "data": getattr(args, "data", None),
                "exp": getattr(args, "exp", None),
                "exp_source": getattr(args, "exp_source", None),
                "dataset_tokens": getattr(args, "dataset_tokens", None),
                "dataset_tokens_source": getattr(args, "dataset_tokens_source", None),
                "target_params": getattr(args, "target_params", None),
                "target_params_source": getattr(args, "target_params_source", None),
                "tokens_per_param": getattr(args, "tokens_per_param", None),
                "layers": getattr(args, "layers", None),
                "layers_source": getattr(args, "layers_source", None),
                "block": getattr(args, "block", None),
                "d_model": getattr(args, "d_model", None),
                "n_head": getattr(args, "n_head", None),
                "head_dim": head_dim,
                "d_ff": getattr(args, "d_ff", None),
                "embed_dim": getattr(args, "embed_dim", None),
                "attn_mode": getattr(args, "attn_mode", None),
                "attn_dim": getattr(args, "attn_dim", None),
                "sem_dim": getattr(args, "sem_dim", None),
                "geo_dim": getattr(args, "geo_dim", None),
                "rope": (not bool(getattr(args, "no_rope", False))),
                "tie_qk": bool(getattr(args, "tie_qk", False)),
                "null_attn": bool(getattr(args, "null_attn", False)),
                "optimizer": getattr(args, "optimizer", None),
                "lr": getattr(args, "lr", None),
                "weight_decay": getattr(args, "weight_decay", None),
                "lr_schedule": getattr(args, "lr_schedule", None),
                "warmup_steps": getattr(args, "warmup_steps", None),
                "min_lr": getattr(args, "min_lr", None),
                "steps": getattr(args, "steps", None),
                "out_dir": getattr(args, "out_dir", None),
            }
        except (AttributeError, TypeError, ValueError):
            pass


class TrainAssertions:
    """Invariant checks for derived training configs."""

    @staticmethod
    def assert_required(args: argparse.Namespace) -> None:
        """Why: fail fast if the derivation graph produced an unusable configuration."""
        for name in ("block", "layers", "d_model", "n_head", "d_ff", "embed_dim", "attn_dim"):
            if not hasattr(args, name):
                raise RuntimeError(f"dynamic config did not set required field: {name}")
            value_obj = cast(object, getattr(args, name))
            if int(cast(int | float | str, value_obj)) <= 0:
                raise RuntimeError(f"dynamic config produced non-positive {name}={value_obj!r}")


