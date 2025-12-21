"""Checkpoint save/load helpers.

Why this exists:
- Training state needs to survive process restarts (preemption, iteration, debugging).
- Keeping checkpoint I/O separate makes the training loop easier to read and safer to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

import torch


@dataclass(frozen=True)
class TrainCheckpoint:
    """Why: strongly-typed view of the checkpoint payload we care about."""

    opt_step: int
    model_state: dict[str, object]
    optim_state: dict[str, object]


def save_checkpoint(
    *,
    out_dir: str,
    opt_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: object,
    extra: dict[str, object] | None = None,
) -> str | None:
    """Why: persist enough state to resume training deterministically."""
    try:
        os.makedirs(str(out_dir), exist_ok=True)
        path = os.path.join(str(out_dir), f"ckpt_step{int(opt_step)}.pt")
        payload: dict[str, object] = {
            "opt_step": int(opt_step),
            "model": cast(object, model.state_dict()),
            "optimizer": cast(object, optimizer.state_dict()),
            "cfg": cast(object, getattr(cfg, "__dict__", {})),
        }
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)
        return str(path)
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def load_checkpoint(path: str) -> TrainCheckpoint | None:
    """Why: isolate deserialization hazards and keep training code clean."""
    try:
        raw = torch.load(str(path), map_location="cpu")
        if not isinstance(raw, dict):
            return None
        opt_step = int(raw.get("opt_step", 0) or 0)
        model_state = raw.get("model", {})
        optim_state = raw.get("optimizer", {})
        if not isinstance(model_state, dict) or not isinstance(optim_state, dict):
            return None
        return TrainCheckpoint(
            opt_step=int(opt_step),
            model_state=cast(dict[str, object], model_state),
            optim_state=cast(dict[str, object], optim_state),
        )
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


