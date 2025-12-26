"""
Unit tests for the benchmark runner module.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from caramba.benchmark.artifacts import ExperimentMetadata
from caramba.benchmark.runner import BenchmarkRunner, QuickBenchmark
from caramba.config.benchmark import (
    BenchmarkSpec,
    BenchmarkSuite,
    BenchmarkType,
    MemoryBenchmarkConfig,
    PerplexityBenchmarkConfig,
)


class DummyLMModel(nn.Module):
    """Simple language model for testing."""

    def __init__(self, vocab_size: int = 32000, d_model: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.linear(h)
        return self.head(h)


class TestBenchmarkRunner(unittest.TestCase):
    """Tests for BenchmarkRunner."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "artifacts"

        # Create test dataset
        self.dataset_path = Path(self.temp_dir) / "test_data.npy"
        data = np.random.randint(0, 1000, size=10000, dtype=np.uint16)
        np.save(self.dataset_path, data)

        self.metadata = ExperimentMetadata(
            name="test",
            timestamp="2024-12-26T12:00:00",
            manifest_path="/test/manifest.yml",
            teacher_checkpoint="test_ckpt",
            student_config="DBA",
            device="cpu",
        )

        self.teacher = DummyLMModel()
        self.teacher.eval()
        self.student = DummyLMModel()
        self.student.eval()

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_returns_paths(self) -> None:
        """Runner returns dict of artifact paths."""
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="memory",
                    config=MemoryBenchmarkConfig(
                        sequence_lengths=[32],
                        batch_sizes=[1],
                        quantization_modes=["fp16"],
                    ),
                    models=["teacher", "student"],
                    repeats=1,
                ),
            ],
            output_dir=str(self.output_dir),
            formats=["json"],
        )

        runner = BenchmarkRunner(suite, self.device, self.metadata)
        paths = runner.run(self.teacher, self.student)

        self.assertIsInstance(paths, dict)
        self.assertGreater(len(paths), 0)

    def test_output_dir_created(self) -> None:
        """Output directory is created."""
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="memory",
                    config=MemoryBenchmarkConfig(
                        sequence_lengths=[32],
                        batch_sizes=[1],
                        quantization_modes=["fp16"],
                    ),
                    models=["teacher"],
                    repeats=1,
                ),
            ],
            output_dir=str(self.output_dir),
            formats=["json"],
        )

        runner = BenchmarkRunner(suite, self.device, self.metadata)
        runner.run(self.teacher, self.student)

        self.assertTrue(self.output_dir.exists())

    def test_runs_perplexity_benchmark(self) -> None:
        """Runner executes perplexity benchmark."""
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="perplexity",
                    config=PerplexityBenchmarkConfig(
                        dataset=str(self.dataset_path),
                        block_size=32,
                        batch_size=1,
                        num_batches=2,
                    ),
                    models=["teacher", "student"],
                    repeats=1,
                ),
            ],
            output_dir=str(self.output_dir),
            formats=["csv"],
        )

        runner = BenchmarkRunner(suite, self.device, self.metadata)
        paths = runner.run(self.teacher, self.student)

        self.assertIn("perplexity.csv", paths)
        self.assertTrue(paths["perplexity.csv"].exists())

    def test_runs_memory_benchmark(self) -> None:
        """Runner executes memory benchmark."""
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="memory",
                    config=MemoryBenchmarkConfig(
                        sequence_lengths=[32],
                        batch_sizes=[1],
                        quantization_modes=["fp16"],
                    ),
                    models=["teacher", "student"],
                    repeats=1,
                ),
            ],
            output_dir=str(self.output_dir),
            formats=["csv"],
        )

        runner = BenchmarkRunner(suite, self.device, self.metadata)
        paths = runner.run(self.teacher, self.student)

        self.assertIn("memory.csv", paths)
        self.assertTrue(paths["memory.csv"].exists())

    def test_runs_only_specified_models(self) -> None:
        """Runner only benchmarks specified models."""
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="perplexity",
                    config=PerplexityBenchmarkConfig(
                        dataset=str(self.dataset_path),
                        block_size=32,
                        batch_size=1,
                        num_batches=2,
                    ),
                    models=["teacher"],  # Only teacher
                    repeats=1,
                ),
            ],
            output_dir=str(self.output_dir),
            formats=["csv"],
        )

        runner = BenchmarkRunner(suite, self.device, self.metadata)
        paths = runner.run(self.teacher, self.student)

        # Check CSV only has teacher
        csv_content = paths["perplexity.csv"].read_text()
        self.assertIn("teacher", csv_content)
        self.assertNotIn("student", csv_content)

    def test_multiple_benchmarks(self) -> None:
        """Runner handles multiple benchmarks in sequence."""
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="perplexity",
                    config=PerplexityBenchmarkConfig(
                        dataset=str(self.dataset_path),
                        block_size=32,
                        batch_size=1,
                        num_batches=2,
                    ),
                    models=["teacher", "student"],
                    repeats=1,
                ),
                BenchmarkSpec(
                    id="memory",
                    config=MemoryBenchmarkConfig(
                        sequence_lengths=[32],
                        batch_sizes=[1],
                        quantization_modes=["fp16"],
                    ),
                    models=["teacher", "student"],
                    repeats=1,
                ),
            ],
            output_dir=str(self.output_dir),
            formats=["csv", "json"],
        )

        runner = BenchmarkRunner(suite, self.device, self.metadata)
        paths = runner.run(self.teacher, self.student)

        self.assertIn("perplexity.csv", paths)
        self.assertIn("memory.csv", paths)
        self.assertIn("report.json", paths)


class TestQuickBenchmark(unittest.TestCase):
    """Tests for QuickBenchmark."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.temp_dir = tempfile.mkdtemp()

        # Create test dataset
        self.dataset_path = Path(self.temp_dir) / "test_data.npy"
        data = np.random.randint(0, 1000, size=10000, dtype=np.uint16)
        np.save(self.dataset_path, data)

        self.model = DummyLMModel()
        self.model.eval()

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_returns_metrics(self) -> None:
        """Quick benchmark returns metrics dict."""
        benchmark = QuickBenchmark(self.device)
        metrics = benchmark.run(self.model, str(self.dataset_path), num_batches=2)

        self.assertIsInstance(metrics, dict)
        self.assertIn("perplexity", metrics)
        self.assertIn("loss", metrics)
        self.assertIn("kv_bytes_per_token", metrics)

    def test_perplexity_is_positive(self) -> None:
        """Quick benchmark returns positive perplexity."""
        benchmark = QuickBenchmark(self.device)
        metrics = benchmark.run(self.model, str(self.dataset_path), num_batches=2)

        self.assertGreater(metrics["perplexity"], 0)

    def test_loss_is_positive(self) -> None:
        """Quick benchmark returns positive loss."""
        benchmark = QuickBenchmark(self.device)
        metrics = benchmark.run(self.model, str(self.dataset_path), num_batches=2)

        self.assertGreater(metrics["loss"], 0)


if __name__ == "__main__":
    unittest.main()
