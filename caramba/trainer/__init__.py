"""
trainer provides the modules needed for the training loop.
"""
from __future__ import annotations

from caramba.config.manifest import Manifest
from caramba.trainer.blockwise import BlockwiseTrainer
from caramba.trainer.distill import DistillLoss
from caramba.trainer.trainer import Trainer
from caramba.trainer.upcycle import Upcycle
from caramba.trainer.distributed import (
    DistributedStrategy,
    DistributedConfig,
    DistributedContext,
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
)

__all__ = [
    "BlockwiseTrainer",
    "DistillLoss",
    "Trainer",
    "Upcycle",
    "DistributedStrategy",
    "DistributedConfig",
    "DistributedContext",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
]
