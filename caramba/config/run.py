"""Run configuration: a single training or evaluation run.

Each run specifies what to do (train, sample, chat), how many steps,
what seed to use, and what verification to perform afterward.
"""
from __future__ import annotations

from pydantic import BaseModel

from caramba.config import PositiveInt
from caramba.config.mode import Mode
from caramba.config.train import TrainConfig
from caramba.config.verify import VerifyConfig


class Run(BaseModel):
    """Configuration for a single training or evaluation run.

    A run is one execution of the training loop with specific settings.
    Multiple runs can be grouped together for comparison.
    """

    id: str
    mode: Mode
    exp: str
    seed: int
    steps: PositiveInt
    expected: dict[str, object]
    verify: VerifyConfig | None = None
    train: TrainConfig | None = None
