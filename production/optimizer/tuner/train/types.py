"""Dataclasses for training tuning outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainBatchPlan:
    """A batch/accum plan keyed by training seq_len."""

    by_seq: dict[int, tuple[int, int]]  # seq_len -> (batch_size, grad_accum)
    target_gbs: int
    warmup: int
    iters: int


@dataclass(frozen=True)
class TrainCompilePlan:
    """Decision about whether to use torch.compile for training."""

    enabled: bool
    mode: str
    warmup: int
    iters: int
    hysteresis: float


