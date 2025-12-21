"""Main entry point for training and sampling runs."""
from __future__ import annotations

import argparse
import os
import sys

from typing import Protocol, cast
import torch

from production.runner_sample import run_sample
from production.runner_train import run_train
from production.runtime_tuning import KVSelfOptConfig


class _ReconfigurableTextIO(Protocol):
    """Protocol for a text stream that can be reconfigured for line-buffering."""
    def reconfigure(self, *, line_buffering: bool) -> None: ...


def _configure_stdio_line_buffering() -> None:
    """Make stdout/stderr line-buffered even when piped (common in IDE consoles)."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            cast(_ReconfigurableTextIO, sys.stdout).reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    try:
        if hasattr(sys.stderr, "reconfigure"):
            cast(_ReconfigurableTextIO, sys.stderr).reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass


def _infer_selfopt_cache_path(args: argparse.Namespace) -> str | None:
    """Place selfopt cache next to out_dir (or its parent) to persist tuned plans across runs."""
    try:
        out_dir = str(getattr(args, "out_dir", "") or "")
        if not out_dir:
            return None
        parent = os.path.dirname(out_dir.rstrip(os.sep)) or "."
        return os.path.join(parent, "selfopt_cache.json")
    except (OSError, ValueError, TypeError):
        return None


def _default_selfopt_cfg(args: argparse.Namespace) -> KVSelfOptConfig:
    # Self-optimization is always enabled and non-configurable: the system chooses the policy.
    # NOTE: We still persist/cache tuned plans for reuse, but callers cannot override tuning knobs.
    return KVSelfOptConfig(mode="online", scope="all", cache_path=_infer_selfopt_cache_path(args))


def run_single(args: argparse.Namespace, device: torch.device) -> None:
    """
    Entry point invoked by `production/cli.py`.

    This file intentionally stays small; heavy logic lives in mode-specific modules.
    """
    _configure_stdio_line_buffering()
    self_opt_cfg = _default_selfopt_cfg(args)

    mode = str(getattr(args, "mode", "train"))
    if mode == "sample":
        run_sample(args=args, device=device, self_opt=self_opt_cfg)
        return

    run_train(args=args, device=device, self_opt=self_opt_cfg)
