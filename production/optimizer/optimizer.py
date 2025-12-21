"""Optimizer public surface.

This module provides the main dependency-driven `Optimizer` graph plus a single
entrypoint (`apply_dynamic_config`) used by the CLI to populate missing args
from intent and heuristics.
"""

from __future__ import annotations

import argparse

from production.optimizer.apply import DynamicConfigApplier
from production.optimizer.graph import Optimizer as _Optimizer

Optimizer = _Optimizer


def apply_dynamic_config(args: argparse.Namespace, *, device: object) -> None:
    """Populate missing runner-required args from intent-first heuristics."""
    DynamicConfigApplier.apply(args, device=device)


