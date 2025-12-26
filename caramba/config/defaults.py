"""
defaults provides the default values for the training loop.
"""
from __future__ import annotations

from pydantic import BaseModel

from caramba.config import PositiveInt, Probability


class Defaults(BaseModel):
    """
    Defaults provides the default values for the training loop.
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
