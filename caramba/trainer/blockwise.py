"""Block-wise distillation training for model upcycling.

When converting a model to a new architecture (like standard attention → DBA),
we can't train the entire model at once—the student would diverge too far from
the teacher. Instead, we train one block at a time, freezing all other blocks.
This ensures each block learns to match its teacher counterpart before moving on.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from caramba.model.trace import Trace
from caramba.trainer.distill import DistillLoss


@dataclass
class BlockwiseConfig:
    """Settings that control blockwise training performance and behavior.

    These exist as a separate config to keep the BlockwiseTrainer constructor
    clean while still allowing fine-grained control over optimizations.
    """

    # Teacher output caching: The teacher is frozen, so its outputs are
    # deterministic for a given input. Caching avoids redundant forward passes.
    cache_teacher_outputs: bool = True
    max_cache_size: int = 100

    # Mixed precision: Use float16/bfloat16 for faster training on GPU.
    # Only applies to CUDA and MPS devices.
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16

    # Gradient accumulation: Simulate larger batch sizes by accumulating
    # gradients over multiple forward passes before stepping the optimizer.
    accumulation_steps: int = 1

    # Truncated forward: Stop the forward pass early for blocks near the
    # start of the model. Experimental—requires model architecture support.
    use_truncated_forward: bool = False


class TeacherOutputCache:
    """LRU cache for teacher model outputs.

    The teacher model never changes during distillation, so running it twice
    on the same input produces identical outputs. This cache stores those
    outputs, keyed by input tensor identity, to skip redundant computation.
    """

    def __init__(self, max_size: int = 100) -> None:
        """Create a cache with the given maximum entry count.

        When the cache is full, the least-recently-used entry is evicted.
        """
        self._cache: dict[int, list[Tensor]] = {}
        self._access_order: list[int] = []
        self._max_size = max_size

    def _compute_key(self, x: Tensor) -> int:
        """Generate a hash key from tensor metadata.

        We use data pointer, shape, dtype, and device—not the actual values.
        This is fast and collision-resistant enough for our use case.
        """
        key_parts = (
            x.data_ptr(),
            x.shape,
            x.dtype,
            x.device.type,
        )
        return hash(key_parts)

    def get(self, x: Tensor) -> list[Tensor] | None:
        """Retrieve cached outputs for an input tensor.

        Returns None on cache miss. On hit, updates the LRU order.
        """
        key = self._compute_key(x)
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, x: Tensor, outputs: list[Tensor]) -> None:
        """Store outputs in the cache, evicting old entries if needed.

        Outputs are cloned and detached so they don't hold onto the
        computation graph or get modified by later operations.
        """
        key = self._compute_key(x)

        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = [o.detach().clone() for o in outputs]
        if key not in self._access_order:
            self._access_order.append(key)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)


class BlockwiseTrainer:
    """Trains a student model one block at a time to match a frozen teacher.

    Blockwise training is essential for architecture changes like DBA upcycling.
    If we trained all blocks simultaneously, early blocks would produce bad
    inputs for later blocks, causing the whole model to diverge. By training
    block-by-block with all other blocks frozen, each block sees stable inputs.
    """

    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: Optimizer,
        loss: DistillLoss,
        predicate: Callable[[str, nn.Module], bool],
        config: BlockwiseConfig | None = None,
    ) -> None:
        """Set up blockwise training between a teacher and student model.

        The predicate function identifies which modules count as "blocks"—
        typically attention layers. Both models must have the same number
        of blocks, since we train them in corresponding pairs.
        """
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.loss = loss
        self._predicate = predicate
        self._teacher_blocks = self._collect_blocks(teacher)
        self._student_blocks = self._collect_blocks(student)

        if not self._teacher_blocks:
            raise ValueError("Teacher has no blocks matching predicate.")
        if len(self._teacher_blocks) != len(self._student_blocks):
            raise ValueError(
                f"Teacher/student block counts must match, got "
                f"{len(self._teacher_blocks)} and {len(self._student_blocks)}"
            )

        # Trace objects hook into the model to capture intermediate outputs
        self._teacher_trace = Trace(teacher, predicate=predicate)
        self._student_trace = Trace(student, predicate=predicate)

        self.config = config or BlockwiseConfig()

        # Set up teacher caching if enabled
        self._teacher_cache: TeacherOutputCache | None = None
        if self.config.cache_teacher_outputs:
            self._teacher_cache = TeacherOutputCache(
                max_size=self.config.max_cache_size
            )

        self._accumulation_count = 0
        self._device_type = self._detect_device_type()

    def _detect_device_type(self) -> str:
        """Determine which device the model is on for autocast compatibility.

        Autocast requires knowing the device type (cuda, mps, or cpu).
        """
        for param in self.student.parameters():
            if param.device.type == "cuda":
                return "cuda"
            elif param.device.type == "mps":
                return "mps"
        return "cpu"

    def block_count(self) -> int:
        """Return the number of trainable blocks in the model."""
        return len(self._student_blocks)

    def step(
        self,
        x: Tensor,
        *,
        block_index: int,
        accumulate: bool = False,
    ) -> Tensor:
        """Run one distillation step for a single block.

        This is the core training operation: freeze all blocks except the
        target, run both models, compute loss between their outputs at that
        block, and update the student's weights for that block only.

        Args:
            x: Input token batch
            block_index: Which block (0-indexed) to train
            accumulate: If True, don't step optimizer (for gradient accumulation)

        Returns:
            The loss value (detached, for logging)
        """
        self._set_block_trainable(block_index)

        t_outputs = self._get_teacher_outputs(x)
        s_outputs = self._get_student_outputs(x)

        t_out = self._select_output(t_outputs, block_index, kind="teacher")
        s_out = self._select_output(s_outputs, block_index, kind="student")

        # Compute loss, optionally with mixed precision
        if self.config.use_amp and self._device_type in ("cuda", "mps"):
            with torch.autocast(
                device_type=self._device_type,
                dtype=self.config.amp_dtype,
            ):
                loss = self.loss([t_out], [s_out])
        else:
            loss = self.loss([t_out], [s_out])

        # Scale loss when accumulating gradients
        if self.config.accumulation_steps > 1:
            loss = loss / self.config.accumulation_steps

        loss.backward()

        # Step optimizer only after accumulating enough gradients
        self._accumulation_count += 1
        should_step = (
            not accumulate and
            self._accumulation_count >= self.config.accumulation_steps
        )

        if should_step:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulation_count = 0

        return (loss * self.config.accumulation_steps).detach()

    def _get_teacher_outputs(self, x: Tensor) -> list[Tensor]:
        """Get teacher block outputs, using cache when possible.

        Since the teacher is frozen, caching avoids redundant forward passes
        when training multiple blocks on the same batch.
        """
        if self._teacher_cache is not None:
            cached = self._teacher_cache.get(x)
            if cached is not None:
                return cached

        self._teacher_trace.clear()
        with torch.no_grad():
            with self._teacher_trace:
                _ = self.teacher(x)

        outputs = list(self._teacher_trace.outputs)

        if self._teacher_cache is not None:
            self._teacher_cache.put(x, outputs)

        return outputs

    def _get_student_outputs(self, x: Tensor) -> list[Tensor]:
        """Get student block outputs with gradient tracking.

        Unlike teacher outputs, these need gradients for backpropagation.
        """
        self._student_trace.clear()

        if self.config.use_amp and self._device_type in ("cuda", "mps"):
            with torch.autocast(
                device_type=self._device_type,
                dtype=self.config.amp_dtype,
            ):
                with self._student_trace:
                    _ = self.student(x)
        else:
            with self._student_trace:
                _ = self.student(x)

        return list(self._student_trace.outputs)

    def _collect_blocks(self, model: nn.Module) -> list[nn.Module]:
        """Find all modules in the model that match the predicate.

        These are the "blocks" we'll train one at a time.
        """
        return [
            module
            for name, module in model.named_modules()
            if self._predicate(name, module)
        ]

    def _set_block_trainable(self, block_index: int) -> None:
        """Freeze the entire model except for one block.

        This ensures gradients only flow through the block we're training,
        keeping all other blocks stable.
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
        """Pick one block's output from the traced outputs list.

        The trace captures outputs from all blocks; we select just the one
        we're training.
        """
        if block_index < 0 or block_index >= len(outputs):
            raise ValueError(
                f"{kind} outputs missing block {block_index}, "
                f"got {len(outputs)} outputs."
            )
        return outputs[block_index]

    def clear_cache(self) -> None:
        """Empty the teacher output cache.

        Call this when switching to new data to avoid stale cache hits.
        """
        if self._teacher_cache is not None:
            self._teacher_cache.clear()

    def flush_gradients(self) -> None:
        """Force an optimizer step with any accumulated gradients.

        Use this at the end of training to ensure no gradients are lost.
        """
        if self._accumulation_count > 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulation_count = 0
