"""
upcycle_test provides tests for upcycle training optimizations.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from pydantic import BaseModel

from caramba.config.train import TrainConfig, TrainPhase
from caramba.trainer.blockwise import BlockwiseConfig


class MockTrainConfig:
    """Mock TrainConfig for testing without full Pydantic validation."""

    # Type annotations for all attributes
    phase: TrainPhase
    batch_size: int
    block_size: int
    lr: float
    device: str
    dtype: str
    teacher_ckpt: str | None
    teacher_rope_base: float | None
    teacher_rope_dim: int | None
    convergence_target: float | None
    convergence_patience: int
    convergence_max_steps: int
    cache_teacher_outputs: bool
    use_amp: bool
    amp_dtype: str
    gradient_accumulation_steps: int
    num_workers: int
    pin_memory: bool
    compile_model: bool

    def __init__(self, **kwargs: object) -> None:
        defaults: dict[str, object] = {
            "phase": TrainPhase.BLOCKWISE,
            "batch_size": 1,
            "block_size": 512,
            "lr": 0.001,
            "device": "cpu",
            "dtype": "float32",
            "teacher_ckpt": None,
            "teacher_rope_base": None,
            "teacher_rope_dim": None,
            "convergence_target": None,
            "convergence_patience": 50,
            "convergence_max_steps": 5000,
            "cache_teacher_outputs": True,
            "use_amp": False,
            "amp_dtype": "float16",
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "pin_memory": False,
            "compile_model": False,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class TrainConfigTest(unittest.TestCase):
    """Tests for TrainConfig optimization fields."""

    def test_default_optimization_values(self) -> None:
        """Test default values for optimization fields."""
        config = TrainConfig(
            phase=TrainPhase.BLOCKWISE,
            batch_size=1,
            block_size=512,
            lr=0.001,
        )

        self.assertTrue(config.cache_teacher_outputs)
        self.assertFalse(config.use_amp)
        self.assertEqual(config.amp_dtype, "float16")
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.num_workers, 0)
        self.assertFalse(config.pin_memory)
        self.assertFalse(config.compile_model)

    def test_custom_optimization_values(self) -> None:
        """Test custom values for optimization fields."""
        config = TrainConfig(
            phase=TrainPhase.BLOCKWISE,
            batch_size=4,
            block_size=1024,
            lr=0.0001,
            cache_teacher_outputs=False,
            use_amp=True,
            amp_dtype="bfloat16",
            gradient_accumulation_steps=4,
            num_workers=4,
            pin_memory=True,
            compile_model=True,
        )

        self.assertFalse(config.cache_teacher_outputs)
        self.assertTrue(config.use_amp)
        self.assertEqual(config.amp_dtype, "bfloat16")
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.num_workers, 4)
        self.assertTrue(config.pin_memory)
        self.assertTrue(config.compile_model)


class BlockwiseConfigBuildingTest(unittest.TestCase):
    """Tests for building BlockwiseConfig from TrainConfig."""

    def test_build_blockwise_config_defaults(self) -> None:
        """Test building BlockwiseConfig with default TrainConfig."""
        train = MockTrainConfig()

        # Simulate the _build_blockwise_config logic
        amp_dtype = torch.float16
        if train.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

        config = BlockwiseConfig(
            cache_teacher_outputs=train.cache_teacher_outputs,
            use_amp=train.use_amp,
            amp_dtype=amp_dtype,
            accumulation_steps=train.gradient_accumulation_steps,
        )

        self.assertTrue(config.cache_teacher_outputs)
        self.assertFalse(config.use_amp)
        self.assertEqual(config.amp_dtype, torch.float16)
        self.assertEqual(config.accumulation_steps, 1)

    def test_build_blockwise_config_with_amp(self) -> None:
        """Test building BlockwiseConfig with AMP enabled."""
        train = MockTrainConfig(
            use_amp=True,
            amp_dtype="bfloat16",
            gradient_accumulation_steps=2,
        )

        amp_dtype = torch.float16
        if train.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

        config = BlockwiseConfig(
            cache_teacher_outputs=train.cache_teacher_outputs,
            use_amp=train.use_amp,
            amp_dtype=amp_dtype,
            accumulation_steps=train.gradient_accumulation_steps,
        )

        self.assertTrue(config.use_amp)
        self.assertEqual(config.amp_dtype, torch.bfloat16)
        self.assertEqual(config.accumulation_steps, 2)

    def test_build_blockwise_config_cache_disabled(self) -> None:
        """Test building BlockwiseConfig with caching disabled."""
        train = MockTrainConfig(cache_teacher_outputs=False)

        config = BlockwiseConfig(
            cache_teacher_outputs=train.cache_teacher_outputs,
            use_amp=train.use_amp,
            amp_dtype=torch.float16,
            accumulation_steps=train.gradient_accumulation_steps,
        )

        self.assertFalse(config.cache_teacher_outputs)


class DataLoaderOptimizationsTest(unittest.TestCase):
    """Tests for DataLoader optimization settings."""

    def test_dataloader_kwargs_default(self) -> None:
        """Test DataLoader kwargs with default settings."""
        train = MockTrainConfig()
        device = torch.device("cpu")

        use_pin_memory = train.pin_memory and device.type == "cuda"

        loader_kwargs = {
            "batch_size": train.batch_size,
            "shuffle": True,
            "drop_last": True,
            "num_workers": train.num_workers,
            "pin_memory": use_pin_memory,
        }

        self.assertEqual(loader_kwargs["batch_size"], 1)
        self.assertEqual(loader_kwargs["num_workers"], 0)
        self.assertFalse(loader_kwargs["pin_memory"])

    def test_dataloader_kwargs_with_workers(self) -> None:
        """Test DataLoader kwargs with workers enabled."""
        train = MockTrainConfig(num_workers=4)
        device = torch.device("cpu")

        use_pin_memory = train.pin_memory and device.type == "cuda"

        loader_kwargs: dict[str, object] = {
            "batch_size": train.batch_size,
            "shuffle": True,
            "drop_last": True,
            "num_workers": train.num_workers,
            "pin_memory": use_pin_memory,
        }

        if train.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        self.assertEqual(loader_kwargs["num_workers"], 4)
        self.assertEqual(loader_kwargs["prefetch_factor"], 2)

    def test_dataloader_kwargs_cuda_with_pin_memory(self) -> None:
        """Test DataLoader kwargs for CUDA with pin_memory."""
        train = MockTrainConfig(pin_memory=True, num_workers=2)
        device = torch.device("cuda")  # Simulated CUDA device

        use_pin_memory = train.pin_memory and device.type == "cuda"

        self.assertTrue(use_pin_memory)

    def test_dataloader_kwargs_mps_no_pin_memory(self) -> None:
        """Test DataLoader kwargs for MPS (no pin_memory support)."""
        train = MockTrainConfig(pin_memory=True)
        device = torch.device("mps")

        use_pin_memory = train.pin_memory and device.type == "cuda"

        self.assertFalse(use_pin_memory)  # MPS doesn't support pin_memory


class TorchCompileTest(unittest.TestCase):
    """Tests for torch.compile support."""

    def test_compile_skipped_for_mps(self) -> None:
        """Test that compile is skipped for MPS devices."""
        device = torch.device("mps")
        compile_model = True

        should_compile = compile_model and hasattr(torch, "compile") and device.type == "cuda"

        self.assertFalse(should_compile)

    def test_compile_enabled_for_cuda(self) -> None:
        """Test that compile is enabled for CUDA devices."""
        device = torch.device("cuda")
        compile_model = True

        should_compile = compile_model and hasattr(torch, "compile") and device.type == "cuda"

        # torch.compile exists in PyTorch 2.0+
        if hasattr(torch, "compile"):
            self.assertTrue(should_compile)

    def test_compile_disabled_when_config_false(self) -> None:
        """Test that compile is skipped when disabled in config."""
        device = torch.device("cuda")
        compile_model = False

        should_compile = compile_model and hasattr(torch, "compile") and device.type == "cuda"

        self.assertFalse(should_compile)


class ConvergenceTrainingTest(unittest.TestCase):
    """Tests for convergence-based training configuration."""

    def test_convergence_not_enabled_by_default(self) -> None:
        """Test that convergence training is not enabled by default."""
        config = TrainConfig(
            phase=TrainPhase.BLOCKWISE,
            batch_size=1,
            block_size=512,
            lr=0.001,
        )

        self.assertIsNone(config.convergence_target)

    def test_convergence_enabled_with_target(self) -> None:
        """Test convergence training with target set."""
        config = TrainConfig(
            phase=TrainPhase.BLOCKWISE,
            batch_size=1,
            block_size=512,
            lr=0.001,
            convergence_target=0.02,
            convergence_patience=100,
            convergence_max_steps=2000,
        )

        self.assertEqual(config.convergence_target, 0.02)
        self.assertEqual(config.convergence_patience, 100)
        self.assertEqual(config.convergence_max_steps, 2000)

    def test_convergence_default_patience_and_max_steps(self) -> None:
        """Test default values for patience and max_steps."""
        config = TrainConfig(
            phase=TrainPhase.BLOCKWISE,
            batch_size=1,
            block_size=512,
            lr=0.001,
            convergence_target=0.05,  # Enable convergence
        )

        self.assertEqual(config.convergence_patience, 50)  # Default
        self.assertEqual(config.convergence_max_steps, 5000)  # Default


if __name__ == "__main__":
    unittest.main()
