"""Default settings that apply across all runs in a manifest.

These are convenience settings that would otherwise need to be repeated
in every run configuration. Things like tokenizer choice, wandb settings,
and checkpoint frequency.
"""
from __future__ import annotations

from pydantic import BaseModel

from caramba.config import PositiveInt, Probability


class Defaults(BaseModel):
    """Global defaults shared across all runs in a manifest.

    Override these per-run when needed, but having sensible defaults
    reduces boilerplate in experiment configs.
    """

    tokenizer: str = "tiktoken"
    val_frac: Probability = 0.1
    instrument: str = "rich"
    wandb: bool = True
    wandb_project: str
    wandb_entity: str
    wandb_mode: str = "online"
    eval_iters: PositiveInt = 50
    save_every: PositiveInt = 100
