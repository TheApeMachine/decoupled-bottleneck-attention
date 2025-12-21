"""Types for self-optimized runtime planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from production.optimizer.tuner.train.types import TrainBatchPlan, TrainCompilePlan


@dataclass(frozen=True)
class RuntimePlan:
    """Self-optimized runtime execution plan (no user overrides)."""

    # Numeric/precision plan
    param_dtype: torch.dtype
    amp_enabled: bool
    amp_dtype: torch.dtype

    # Sequence plan (architectural max lives in cfg.block_size; these are runtime feasible)
    train_seq_len: int
    eval_seq_len: int

    # Batch/compile plans
    batch_plan: TrainBatchPlan
    compile_plan: TrainCompilePlan

    # Optional debug metrics
    metrics: dict[str, float]


