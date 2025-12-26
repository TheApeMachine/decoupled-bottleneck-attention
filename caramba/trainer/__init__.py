"""Training loop components for model training and upcycling.

Training in caramba has two main modes:
1. Standard training: Train a model from scratch or fine-tune
2. Upcycling: Convert a pretrained model to a new architecture (like DBA)

This package provides:
- Trainer: Main training loop orchestrator
- Upcycle: Teacher-student distillation for architecture conversion
- BlockwiseTrainer: Layer-by-layer distillation for stable upcycling
- DistillLoss: L1 loss for knowledge transfer
- Distributed: DDP/FSDP support for multi-GPU training
"""
from __future__ import annotations

from caramba.trainer.blockwise import BlockwiseTrainer
from caramba.trainer.distill import DistillLoss
from caramba.trainer.distributed import (
    DistributedConfig,
    DistributedContext,
    DistributedStrategy,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)
from caramba.trainer.trainer import Trainer
from caramba.trainer.upcycle import Upcycle

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
