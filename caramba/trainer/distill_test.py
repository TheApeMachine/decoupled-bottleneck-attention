"""
distill_test provides tests for the distillation loss module.
"""
from __future__ import annotations

import unittest

import torch

from caramba.trainer.distill import DistillLoss


class DistillLossTest(unittest.TestCase):
    """
    DistillLossTest provides tests for DistillLoss.
    """
    def test_forward_mean(self) -> None:
        """
        test mean reduction with matching outputs.
        """
        loss_fn = DistillLoss(reduction="mean")
        teacher = [torch.zeros(2, 3), torch.ones(2, 3)]
        student = [torch.ones(2, 3), torch.zeros(2, 3)]
        loss = loss_fn(teacher, student)
        self.assertEqual(loss.shape, ())

    def test_rejects_length_mismatch(self) -> None:
        """
        test rejecting a length mismatch.
        """
        loss_fn = DistillLoss()
        with self.assertRaises(ValueError):
            _ = loss_fn([torch.zeros(1)], [torch.zeros(1), torch.zeros(1)])

    def test_rejects_shape_mismatch(self) -> None:
        """
        test rejecting a shape mismatch.
        """
        loss_fn = DistillLoss()
        with self.assertRaises(ValueError):
            _ = loss_fn([torch.zeros(1, 2)], [torch.zeros(2, 1)])
