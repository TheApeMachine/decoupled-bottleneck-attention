from __future__ import annotations

import pytest
import torch

from caramba.trainer.scheduler import LRSchedulerConfig, build_lr_scheduler


def test_linear_scheduler_decays() -> None:
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = build_lr_scheduler(opt, LRSchedulerConfig(kind="linear", total_steps=10, warmup_steps=0, min_lr_ratio=0.1))
    assert sched is not None
    lrs = []
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] <= 1.0
    assert lrs[-1] >= 0.1


def test_cosine_scheduler_decays() -> None:
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = build_lr_scheduler(opt, LRSchedulerConfig(kind="cosine", total_steps=10, warmup_steps=0, min_lr_ratio=0.0))
    assert sched is not None
    for _ in range(10):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] >= 0.0


def test_constant_scheduler_keeps_lr() -> None:
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = build_lr_scheduler(
        opt,
        LRSchedulerConfig(kind="constant", total_steps=10, warmup_steps=0, min_lr_ratio=0.0),
    )
    assert sched is not None
    for _ in range(5):
        opt.step()
        sched.step()
    assert float(opt.param_groups[0]["lr"]) == pytest.approx(1.0)

