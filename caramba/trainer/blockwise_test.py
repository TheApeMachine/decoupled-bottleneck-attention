"""
blockwise_test provides tests for blockwise training.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn

from caramba.trainer.blockwise import BlockwiseTrainer
from caramba.trainer.distill import DistillLoss


class BlockwiseTrainerTest(unittest.TestCase):
    """
    BlockwiseTrainerTest provides tests for BlockwiseTrainer.
    """
    def test_step_returns_scalar(self) -> None:
        """
        test running a single blockwise step.
        """
        teacher = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )
        student = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )
        student.load_state_dict(teacher.state_dict())
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
        )
        self.assertEqual(trainer.block_count(), 2)

        x = torch.randn(2, 4)
        loss = trainer.step(x, block_index=1)
        self.assertEqual(loss.shape, ())

        trainable = [p for p in student.parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertLess(len(trainable), len(list(student.parameters())))
