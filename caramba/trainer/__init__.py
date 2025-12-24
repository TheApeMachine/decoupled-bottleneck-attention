"""
trainer provides the modules needed for the training loop.
"""
from __future__ import annotations

from caramba.config.manifest import Manifest
from caramba.trainer.blockwise import BlockwiseTrainer
from caramba.trainer.distill import DistillLoss
from caramba.trainer.trainer import Trainer

__all__ = ["BlockwiseTrainer", "DistillLoss", "Trainer"]
