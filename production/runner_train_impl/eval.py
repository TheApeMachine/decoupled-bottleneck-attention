"""Evaluation helpers for training.

Why this exists:
- Evaluation is a pure measurement step; isolating it keeps the train loop tight.
- It also makes it easy to change eval cadence/metrics without touching optimization code.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

import torch
import torch.nn.functional as F


class _Model(Protocol):
    def __call__(self, idx: torch.Tensor) -> tuple[torch.Tensor, object]: ...


def estimate_loss(
    *,
    model: _Model,
    get_batch_train: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    get_batch_val: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    eval_iters: int,
    batch_size: int,
    seq_len: int,
    autocast_ctx: Callable[[], AbstractContextManager[None]],
) -> tuple[float, float]:
    """Why: compute train/val loss using the same batching path as training."""
    iters = int(max(1, eval_iters))
    bs = int(max(1, batch_size))
    sl = int(max(1, seq_len))

    def _one(get_batch: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]]) -> float:
        loss_sum = 0.0
        with torch.no_grad():
            for _ in range(iters):
                xb, yb = get_batch(bs, sl)
                with autocast_ctx():
                    logits, _ = model(xb)
                    loss_t = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                loss_sum += float(loss_t.detach().float().cpu().item())
        return float(loss_sum / float(iters))

    train_loss = _one(get_batch_train)
    val_loss = _one(get_batch_val)
    return float(train_loss), float(val_loss)


