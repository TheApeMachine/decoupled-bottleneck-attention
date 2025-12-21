"""Learning-rate scheduling for training.

Why this exists:
- The training loop should be simple; the schedule math is isolated here.
- Keeping it pure makes schedules easy to test and reuse.
"""

from __future__ import annotations

import math


def lr_for_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    schedule: str,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> float:
    """Why: derive LR deterministically from a small set of knobs."""
    step = int(max(0, step))
    total_steps = int(max(1, total_steps))
    base_lr = float(base_lr)
    warmup_steps = int(max(0, warmup_steps))
    min_lr = float(min_lr)

    sched = str(schedule or "cosine").strip().lower()

    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * float(step + 1) / float(max(1, warmup_steps))

    t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    t = max(0.0, min(1.0, t))

    match sched:
        case "constant":
            return float(base_lr)
        case "linear":
            return float(min_lr + (base_lr - min_lr) * (1.0 - t))
        case "cosine":
            c = 0.5 * (1.0 + math.cos(math.pi * t))
            return float(min_lr + (base_lr - min_lr) * c)
        case _:
            return float(base_lr)


