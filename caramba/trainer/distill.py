"""Distillation loss functions for knowledge transfer.

Knowledge distillation trains a student model to match a teacher's outputs.
We use L1 loss (mean absolute error) because it's robust to outliers and
produces stable gradients even when teacher and student outputs differ
significantly—which is common early in training.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override


class DistillLoss(nn.Module):
    """L1 loss between teacher and student layer outputs.

    During blockwise training, we compare the output tensors from matching
    layers in teacher and student. L1 loss measures the average absolute
    difference, pushing the student to produce outputs identical to the teacher.
    """

    def __init__(self, *, reduction: str = "mean") -> None:
        """Configure how losses from multiple layers are combined.

        Args:
            reduction: "mean" averages across layers, "sum" adds them.
                       Most training uses "mean" for stable loss magnitudes.
        """
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    @override
    def forward(
        self,
        teacher: Sequence[Tensor],
        student: Sequence[Tensor],
    ) -> Tensor:
        """Compute L1 loss between corresponding teacher and student outputs.

        Each list contains one tensor per layer. We compute L1 loss for each
        pair and combine them according to the reduction setting.

        Args:
            teacher: List of teacher layer outputs, shape (B, T, D) each
            student: List of student layer outputs, must match teacher shapes

        Returns:
            Scalar loss tensor
        """
        if len(teacher) != len(student):
            raise ValueError(
                f"Teacher/student output lists must match in length, got "
                f"{len(teacher)} and {len(student)}"
            )
        if not teacher:
            raise ValueError("Teacher/student outputs must be non-empty.")

        total: Tensor | None = None
        for t_out, s_out in zip(teacher, student):
            if t_out.shape != s_out.shape:
                raise ValueError(
                    f"Teacher/student output shapes must match, got "
                    f"{t_out.shape} and {s_out.shape}"
                )
            loss = F.l1_loss(s_out, t_out, reduction="mean")
            total = loss if total is None else total + loss

        if total is None:
            raise RuntimeError("No distillation outputs were processed.")

        if self.reduction == "sum":
            return total
        return total / float(len(teacher))
