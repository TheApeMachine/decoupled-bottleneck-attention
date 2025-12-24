"""
distill provides attention distillation losses.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override


class DistillLoss(nn.Module):
    """
    DistillLoss computes per-layer L1 loss between teacher and student outputs.
    """
    def __init__(self, *, reduction: str = "mean") -> None:
        """
        __init__ initializes the distillation loss module.
        """
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction: str = reduction

    @override
    def forward(
        self,
        teacher: Sequence[Tensor],
        student: Sequence[Tensor],
    ) -> Tensor:
        """
        forward computes an L1 loss across matching output sequences.
        """
        if len(teacher) != len(student):
            raise ValueError(
                "Teacher/student output lists must match in length, got "
                f"{len(teacher)} and {len(student)}"
            )
        if not teacher:
            raise ValueError("Teacher/student outputs must be non-empty.")

        total: Tensor | None = None
        for t_out, s_out in zip(teacher, student):
            if t_out.shape != s_out.shape:
                raise ValueError(
                    "Teacher/student output shapes must match, got "
                    f"{t_out.shape} and {s_out.shape}"
                )
            loss = F.l1_loss(s_out, t_out, reduction="mean")
            total = loss if total is None else total + loss

        if total is None:
            raise RuntimeError("No distillation outputs were processed.")
        if self.reduction == "sum":
            return total
        return total / float(len(teacher))
