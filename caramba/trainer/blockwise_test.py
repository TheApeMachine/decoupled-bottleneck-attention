"""
blockwise_test provides tests for blockwise training.
"""
from __future__ import annotations

import unittest
from typing import cast

import torch
from torch import nn

from caramba.trainer.blockwise import (
    BlockwiseTrainer,
    BlockwiseConfig,
    TeacherOutputCache,
)
from caramba.trainer.distill import DistillLoss


def _make_simple_models() -> tuple[nn.Sequential, nn.Sequential]:
    """Create simple teacher/student model pair for testing."""
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
    return teacher, student


class TeacherOutputCacheTest(unittest.TestCase):
    """Tests for TeacherOutputCache."""

    def test_cache_miss_returns_none(self) -> None:
        """Test that cache miss returns None."""
        cache = TeacherOutputCache(max_size=10)
        x = torch.randn(2, 4)
        self.assertIsNone(cache.get(x))

    def test_cache_hit_returns_outputs(self) -> None:
        """Test that cache hit returns stored outputs."""
        cache = TeacherOutputCache(max_size=10)
        x = torch.randn(2, 4)
        outputs = [torch.randn(2, 4), torch.randn(2, 4)]
        cache.put(x, outputs)

        cached = cache.get(x)
        self.assertIsNotNone(cached)
        assert cached is not None  # Type narrowing for pyright
        self.assertEqual(len(cached), 2)
        # Values should be equal but not same object (cloned)
        for orig, cached_out in zip(outputs, cached):
            self.assertTrue(torch.allclose(orig, cached_out))

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = TeacherOutputCache(max_size=2)

        x1 = torch.randn(2, 4)
        x2 = torch.randn(2, 4)
        x3 = torch.randn(2, 4)

        cache.put(x1, [torch.randn(2, 4)])
        cache.put(x2, [torch.randn(2, 4)])
        # x1 should be evicted when we add x3
        cache.put(x3, [torch.randn(2, 4)])

        self.assertIsNone(cache.get(x1))  # Evicted
        self.assertIsNotNone(cache.get(x2))
        self.assertIsNotNone(cache.get(x3))

    def test_clear(self) -> None:
        """Test cache clearing."""
        cache = TeacherOutputCache(max_size=10)
        x = torch.randn(2, 4)
        cache.put(x, [torch.randn(2, 4)])
        self.assertEqual(len(cache), 1)

        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get(x))


class BlockwiseTrainerTest(unittest.TestCase):
    """
    BlockwiseTrainerTest provides tests for BlockwiseTrainer.
    """

    def test_step_returns_scalar(self) -> None:
        """
        test running a single blockwise step.
        """
        teacher, student = _make_simple_models()
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

    def test_teacher_caching_enabled_by_default(self) -> None:
        """Test that teacher output caching is enabled by default."""
        teacher, student = _make_simple_models()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
        )

        self.assertIsNotNone(trainer._teacher_cache)
        self.assertTrue(trainer.config.cache_teacher_outputs)

    def test_teacher_caching_can_be_disabled(self) -> None:
        """Test that teacher caching can be disabled."""
        teacher, student = _make_simple_models()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        config = BlockwiseConfig(cache_teacher_outputs=False)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
            config=config,
        )

        self.assertIsNone(trainer._teacher_cache)

    def test_same_batch_uses_cache(self) -> None:
        """Test that running the same batch twice uses cache."""
        teacher, student = _make_simple_models()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
        )

        x = torch.randn(2, 4)

        # First call should populate cache
        _ = trainer.step(x, block_index=0)
        self.assertEqual(len(trainer._teacher_cache), 1)  # type: ignore[arg-type]

        # Second call with same tensor should use cache
        _ = trainer.step(x, block_index=0)
        self.assertEqual(len(trainer._teacher_cache), 1)  # type: ignore[arg-type]

    def test_gradient_accumulation(self) -> None:
        """Test gradient accumulation mode."""
        teacher, student = _make_simple_models()
        # Use higher learning rate to ensure visible weight change
        optimizer = torch.optim.SGD(student.parameters(), lr=1.0)
        config = BlockwiseConfig(
            accumulation_steps=2,
            cache_teacher_outputs=False,  # Disable caching for this test
        )
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
            config=config,
        )

        # Use input that will produce non-zero gradients
        x = torch.randn(2, 4) * 10.0

        # Perturb student weights to create difference from teacher
        first_linear = cast(nn.Linear, student[0])
        with torch.no_grad():
            first_linear.weight.add_(torch.randn_like(first_linear.weight) * 0.5)

        # Get initial weights (block 0 = first linear layer)
        initial_weight = first_linear.weight.clone()

        # First step with accumulate=True - no optimizer step
        _ = trainer.step(x, block_index=0, accumulate=True)
        self.assertEqual(trainer._accumulation_count, 1)
        # Weight should not have changed (no optimizer step yet)
        self.assertTrue(torch.allclose(first_linear.weight, initial_weight))

        # Second step without accumulate - should trigger optimizer step
        _ = trainer.step(x, block_index=0)
        self.assertEqual(trainer._accumulation_count, 0)
        # Weight should now be different
        self.assertFalse(torch.allclose(first_linear.weight, initial_weight))

    def test_flush_gradients(self) -> None:
        """Test flushing accumulated gradients."""
        teacher, student = _make_simple_models()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        config = BlockwiseConfig(accumulation_steps=4)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
            config=config,
        )

        x = torch.randn(2, 4)

        # Accumulate some gradients
        _ = trainer.step(x, block_index=0, accumulate=True)
        _ = trainer.step(x, block_index=0, accumulate=True)
        self.assertEqual(trainer._accumulation_count, 2)

        # Flush should trigger optimizer step
        trainer.flush_gradients()
        self.assertEqual(trainer._accumulation_count, 0)

    def test_clear_cache(self) -> None:
        """Test clearing the teacher cache."""
        teacher, student = _make_simple_models()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
        )

        x = torch.randn(2, 4)
        _ = trainer.step(x, block_index=0)
        self.assertEqual(len(trainer._teacher_cache), 1)  # type: ignore[arg-type]

        trainer.clear_cache()
        self.assertEqual(len(trainer._teacher_cache), 0)  # type: ignore[arg-type]

    def test_loss_is_scaled_for_accumulation(self) -> None:
        """Test that loss is scaled when using gradient accumulation."""
        teacher, student = _make_simple_models()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

        # Without accumulation
        config_no_accum = BlockwiseConfig(
            accumulation_steps=1,
            cache_teacher_outputs=False,  # Disable cache for consistent loss
        )
        trainer_no_accum = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
            config=config_no_accum,
        )

        x = torch.randn(2, 4)
        loss_no_accum = trainer_no_accum.step(x, block_index=0)

        # Reset student
        student.load_state_dict(teacher.state_dict())
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

        # With accumulation (2 steps)
        config_accum = BlockwiseConfig(
            accumulation_steps=2,
            cache_teacher_outputs=False,
        )
        trainer_accum = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
            config=config_accum,
        )

        loss_accum = trainer_accum.step(x, block_index=0, accumulate=True)

        # The returned loss should be the same (unscaled for reporting)
        self.assertTrue(torch.allclose(loss_no_accum, loss_accum, atol=1e-5))

    def test_block_freezing(self) -> None:
        """Test that only the target block is trainable."""
        teacher, student = _make_simple_models()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
        )

        x = torch.randn(2, 4)

        # Train block 0
        _ = trainer.step(x, block_index=0)

        # Check that only block 0 (first Linear) is trainable
        linear_layers = [m for m in student.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(linear_layers), 2)

        # First linear should be trainable
        for param in linear_layers[0].parameters():
            self.assertTrue(param.requires_grad)

        # Second linear should be frozen (we just trained block 0)
        for param in linear_layers[1].parameters():
            self.assertFalse(param.requires_grad)

    def test_invalid_block_index_raises(self) -> None:
        """Test that invalid block index raises ValueError."""
        teacher, student = _make_simple_models()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, nn.Linear),
        )

        x = torch.randn(2, 4)

        with self.assertRaises(ValueError):
            _ = trainer.step(x, block_index=-1)

        with self.assertRaises(ValueError):
            _ = trainer.step(x, block_index=10)


class BlockwiseConfigTest(unittest.TestCase):
    """Tests for BlockwiseConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BlockwiseConfig()

        self.assertTrue(config.cache_teacher_outputs)
        self.assertEqual(config.max_cache_size, 100)
        self.assertFalse(config.use_amp)
        self.assertEqual(config.amp_dtype, torch.float16)
        self.assertEqual(config.accumulation_steps, 1)
        self.assertFalse(config.use_truncated_forward)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BlockwiseConfig(
            cache_teacher_outputs=False,
            max_cache_size=50,
            use_amp=True,
            amp_dtype=torch.bfloat16,
            accumulation_steps=4,
        )

        self.assertFalse(config.cache_teacher_outputs)
        self.assertEqual(config.max_cache_size, 50)
        self.assertTrue(config.use_amp)
        self.assertEqual(config.amp_dtype, torch.bfloat16)
        self.assertEqual(config.accumulation_steps, 4)


if __name__ == "__main__":
    unittest.main()
