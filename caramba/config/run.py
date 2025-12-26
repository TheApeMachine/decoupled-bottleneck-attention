"""
run provides the run module for the training loop.
"""
from __future__ import annotations

from pydantic import BaseModel
from caramba.config.mode import Mode
from caramba.config.train import TrainConfig
from caramba.config.verify import VerifyConfig
from caramba.config import PositiveInt


class Run(BaseModel):
    """
    Run provides the run module for the training loop.
    """
    id: str
    mode: Mode
    exp: str
    seed: int
    steps: PositiveInt
    expected: dict[str, object]
    verify: VerifyConfig | None = None
    train: TrainConfig | None = None
