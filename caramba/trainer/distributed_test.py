"""
Unit tests for the distributed training module.
"""
from __future__ import annotations

import io
import unittest
from unittest.mock import patch, MagicMock

import torch
from torch import nn

from caramba.trainer.distributed import (
    DistributedContext,
    DistributedConfig,
    DistributedStrategy,
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestDistributedStrategy(unittest.TestCase):
    """Tests for DistributedStrategy enum."""

    def test_strategies_exist(self) -> None:
        """All expected strategies are defined."""
        self.assertEqual(DistributedStrategy.NONE.value, "none")
        self.assertEqual(DistributedStrategy.DDP.value, "ddp")
        self.assertEqual(DistributedStrategy.FSDP.value, "fsdp")


class TestDistributedConfig(unittest.TestCase):
    """Tests for DistributedConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        config = DistributedConfig()
        self.assertEqual(config.strategy, DistributedStrategy.NONE)
        self.assertEqual(config.backend, "nccl")
        self.assertFalse(config.ddp_find_unused_parameters)

    def test_custom_config(self) -> None:
        """Custom config values are stored."""
        config = DistributedConfig(
            strategy=DistributedStrategy.DDP,
            backend="gloo",
            ddp_find_unused_parameters=True,
        )
        self.assertEqual(config.strategy, DistributedStrategy.DDP)
        self.assertEqual(config.backend, "gloo")
        self.assertTrue(config.ddp_find_unused_parameters)


class TestDistributedContextSingleProcess(unittest.TestCase):
    """Tests for DistributedContext in single-process mode."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Use init() with NONE strategy for single-process mode
        self.ctx = DistributedContext.init()

    def test_is_main_true(self) -> None:
        """Single process is always main."""
        self.assertTrue(self.ctx.is_main)

    def test_rank_is_zero(self) -> None:
        """Single process rank is 0."""
        self.assertEqual(self.ctx.rank, 0)

    def test_world_size_is_one(self) -> None:
        """Single process world size is 1."""
        self.assertEqual(self.ctx.world_size, 1)

    def test_local_rank_is_zero(self) -> None:
        """Single process local rank is 0."""
        self.assertEqual(self.ctx.local_rank, 0)

    @patch("caramba.trainer.distributed.console_logger")
    def test_log_prints_on_main(self, mock_logger: MagicMock) -> None:
        """log() prints on main process."""
        self.ctx.log("Test message")
        mock_logger.info.assert_called_once_with("Test message")

    @patch("caramba.trainer.distributed.console_logger")
    def test_print_calls_logger(self, mock_logger: MagicMock) -> None:
        """print() calls logger on main process."""
        self.ctx.print("Hello", "world")
        mock_logger.log.assert_called_once()


class TestDistributedContextNonMainProcess(unittest.TestCase):
    """Tests for DistributedContext when not main process."""

    def setUp(self) -> None:
        """Set up test fixtures with non-main context (simulated)."""
        # Create a context and manually override rank to simulate non-main
        self.ctx = DistributedContext.init()
        # Manually set rank to non-zero to simulate non-main process
        self.ctx._rank = 1

    def test_is_main_false(self) -> None:
        """Non-zero rank is not main."""
        self.assertFalse(self.ctx.is_main)

    @patch("caramba.trainer.distributed.console_logger")
    def test_log_silent_on_non_main(self, mock_logger: MagicMock) -> None:
        """log() is silent on non-main process."""
        self.ctx.log("Should not print")
        mock_logger.info.assert_not_called()

    @patch("caramba.trainer.distributed.console_logger")
    def test_print_silent_on_non_main(self, mock_logger: MagicMock) -> None:
        """print() is silent on non-main process."""
        self.ctx.print("Should not print")
        mock_logger.log.assert_not_called()


class TestDistributedDeviceSelection(unittest.TestCase):
    """Tests for device selection."""

    def test_device_is_valid(self) -> None:
        """Device is valid torch.device."""
        ctx = DistributedContext.init()
        # Device should be cpu or cuda depending on availability
        self.assertIsInstance(ctx.device, torch.device)


class TestGlobalFunctions(unittest.TestCase):
    """Tests for global convenience functions."""

    def test_is_distributed_default(self) -> None:
        """is_distributed returns False by default (no init)."""
        # When not initialized, should return False
        self.assertFalse(is_distributed())

    def test_get_rank_default(self) -> None:
        """get_rank returns 0 by default."""
        self.assertEqual(get_rank(), 0)

    def test_get_world_size_default(self) -> None:
        """get_world_size returns 1 by default."""
        self.assertEqual(get_world_size(), 1)

    def test_is_main_process_default(self) -> None:
        """is_main_process returns True by default."""
        self.assertTrue(is_main_process())


class TestDistributedContextModelWrapping(unittest.TestCase):
    """Tests for model wrapping (NONE strategy only, no actual DDP)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.ctx = DistributedContext.init()
        self.model = SimpleModel()

    def test_wrap_model_none_strategy(self) -> None:
        """Wrapping with NONE strategy returns model unchanged."""
        # With NONE strategy (default), wrap_model returns model as-is
        wrapped = self.ctx.wrap_model(self.model)
        self.assertIs(wrapped, self.model)


class TestDistributedContextBarrier(unittest.TestCase):
    """Tests for barrier synchronization."""

    def test_barrier_noop_single_process(self) -> None:
        """Barrier is a no-op for single process."""
        ctx = DistributedContext.init()
        # Should not raise
        ctx.barrier()


class TestDistributedContextAllReduce(unittest.TestCase):
    """Tests for all_reduce operations."""

    def test_all_reduce_single_process(self) -> None:
        """all_reduce returns tensor unchanged for single process."""
        ctx = DistributedContext.init()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = ctx.all_reduce(tensor)
        self.assertTrue(torch.allclose(result, tensor))


if __name__ == "__main__":
    unittest.main()
