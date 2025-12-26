"""
train provides training configuration models.
"""
from __future__ import annotations

import enum
from pydantic import BaseModel

from caramba.config import PositiveFloat, PositiveInt


class TrainPhase(str, enum.Enum):
    """
    TrainPhase provides the training phase.
    """
    BLOCKWISE = "blockwise"
    GLOBAL = "global"


class TrainConfig(BaseModel):
    """
    TrainConfig provides training parameters for a run.
    """
    phase: TrainPhase
    batch_size: PositiveInt
    block_size: PositiveInt
    lr: PositiveFloat
    device: str = "cpu"
    dtype: str = "float32"
    teacher_ckpt: str | None = None
    teacher_rope_base: PositiveFloat | None = None
    teacher_rope_dim: PositiveInt | None = None
