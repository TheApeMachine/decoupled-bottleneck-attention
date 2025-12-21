"""
Training runner entry point (internal).

Keeps a stable internal import (`production.runner_train_impl.run.run_train`).
Delegates to `Trainer` so the entry point stays tiny and easy to lint/type-check.
"""

from __future__ import annotations

import argparse
import torch
from production.runtime_tuning import KVSelfOptConfig
from production.runner_train_impl.trainer import Trainer


def run_train(*, args: argparse.Namespace, device: torch.device, self_opt: KVSelfOptConfig | None) -> None:
    """public callers want a function, but the implementation is class-based."""
    Trainer(args=args, device=device, self_opt=self_opt).run()
