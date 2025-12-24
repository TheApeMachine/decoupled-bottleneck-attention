"""
blockwise provides block-wise distillation training utilities.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from caramba.model.trace import Trace
from caramba.trainer.distill import DistillLoss


class BlockwiseTrainer:
    """
    BlockwiseTrainer performs block-wise distillation steps.
    """
    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: Optimizer,
        loss: DistillLoss,
        predicate: Callable[[str, nn.Module], bool],
    ) -> None:
        """
        __init__ initializes a block-wise trainer.
        """
        self.teacher: nn.Module = teacher
        self.student: nn.Module = student
        self.optimizer: Optimizer = optimizer
        self.loss: DistillLoss = loss
        self._predicate: Callable[[str, nn.Module], bool] = predicate
        self._teacher_blocks: list[nn.Module] = self._collect_blocks(teacher)
        self._student_blocks: list[nn.Module] = self._collect_blocks(student)
        if not self._teacher_blocks:
            raise ValueError("Teacher has no blocks matching predicate.")
        if len(self._teacher_blocks) != len(self._student_blocks):
            raise ValueError(
                "Teacher/student block counts must match, got "
                f"{len(self._teacher_blocks)} and {len(self._student_blocks)}"
            )
        self._teacher_trace: Trace = Trace(teacher, predicate=predicate)
        self._student_trace: Trace = Trace(student, predicate=predicate)

    def block_count(self) -> int:
        """
        block_count returns the number of blocks.
        """
        return len(self._student_blocks)

    def step(self, x: Tensor, *, block_index: int) -> Tensor:
        """
        step runs one distillation step for a single block.
        """
        self._set_block_trainable(block_index)
        self._teacher_trace.clear()
        self._student_trace.clear()

        with torch.no_grad():
            with self._teacher_trace:
                _ = self.teacher(x)

        with self._student_trace:
            _ = self.student(x)

        t_out = self._select_output(
            self._teacher_trace.outputs,
            block_index,
            kind="teacher",
        )
        s_out = self._select_output(
            self._student_trace.outputs,
            block_index,
            kind="student",
        )

        loss = self.loss([t_out], [s_out])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def _collect_blocks(self, model: nn.Module) -> list[nn.Module]:
        """
        _collect_blocks collects modules matching the predicate.
        """
        return [
            module
            for name, module in model.named_modules()
            if self._predicate(name, module)
        ]

    def _set_block_trainable(self, block_index: int) -> None:
        """
        _set_block_trainable freezes all blocks except the target block.
        """
        if block_index < 0 or block_index >= len(self._student_blocks):
            raise ValueError(
                f"Invalid block index {block_index}, expected "
                f"0..{len(self._student_blocks) - 1}"
            )

        for param in self.student.parameters():
            param.requires_grad = False

        block = self._student_blocks[block_index]
        for param in block.parameters():
            param.requires_grad = True

    def _select_output(
        self,
        outputs: list[Tensor],
        block_index: int,
        *,
        kind: str,
    ) -> Tensor:
        """
        _select_output selects a traced output by block index.
        """
        if block_index < 0 or block_index >= len(outputs):
            raise ValueError(
                f"{kind} outputs missing block {block_index}, "
                f"got {len(outputs)} outputs."
            )
        return outputs[block_index]
