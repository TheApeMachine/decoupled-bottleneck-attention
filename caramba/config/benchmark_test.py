"""
Unit tests for the benchmark config module.
"""
from __future__ import annotations

import unittest

from caramba.config.benchmark import (
    AccuracyBenchmarkConfig,
    BenchmarkSpec,
    BenchmarkSuite,
    BenchmarkType,
    GenerationBenchmarkConfig,
    LatencyBenchmarkConfig,
    MemoryBenchmarkConfig,
    PerplexityBenchmarkConfig,
)


class TestBenchmarkType(unittest.TestCase):
    """Tests for BenchmarkType enum."""

    def test_all_types_exist(self) -> None:
        """All benchmark types are defined."""
        self.assertEqual(BenchmarkType.PERPLEXITY.value, "perplexity")
        self.assertEqual(BenchmarkType.LATENCY.value, "latency")
        self.assertEqual(BenchmarkType.MEMORY.value, "memory")
        self.assertEqual(BenchmarkType.ACCURACY.value, "accuracy")
        self.assertEqual(BenchmarkType.GENERATION.value, "generation")


class TestPerplexityBenchmarkConfig(unittest.TestCase):
    """Tests for PerplexityBenchmarkConfig."""

    def test_minimal_config(self) -> None:
        """Minimal config with required fields."""
        cfg = PerplexityBenchmarkConfig(dataset="data.npy")
        self.assertEqual(cfg.dataset, "data.npy")
        self.assertEqual(cfg.type, BenchmarkType.PERPLEXITY)

    def test_defaults(self) -> None:
        """Default values are applied."""
        cfg = PerplexityBenchmarkConfig(dataset="data.npy")
        self.assertEqual(cfg.block_size, 2048)
        self.assertEqual(cfg.batch_size, 1)
        self.assertIsNone(cfg.num_batches)
        self.assertIsNone(cfg.stride)

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        cfg = PerplexityBenchmarkConfig(
            dataset="data.npy",
            block_size=1024,
            batch_size=4,
            num_batches=100,
            stride=512,
        )
        self.assertEqual(cfg.block_size, 1024)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.num_batches, 100)
        self.assertEqual(cfg.stride, 512)


class TestLatencyBenchmarkConfig(unittest.TestCase):
    """Tests for LatencyBenchmarkConfig."""

    def test_minimal_config(self) -> None:
        """Minimal config with defaults."""
        cfg = LatencyBenchmarkConfig()
        self.assertEqual(cfg.type, BenchmarkType.LATENCY)

    def test_defaults(self) -> None:
        """Default values are applied."""
        cfg = LatencyBenchmarkConfig()
        self.assertEqual(cfg.prompt_lengths, [128, 512, 1024, 2048])
        self.assertEqual(cfg.generation_lengths, [128, 256, 512])
        self.assertEqual(cfg.batch_sizes, [1, 4, 8])
        self.assertEqual(cfg.warmup_runs, 3)
        self.assertEqual(cfg.timed_runs, 10)
        self.assertTrue(cfg.use_cache)

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        cfg = LatencyBenchmarkConfig(
            prompt_lengths=[64, 128],
            generation_lengths=[32],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=5,
            use_cache=False,
        )
        self.assertEqual(cfg.prompt_lengths, [64, 128])
        self.assertEqual(cfg.generation_lengths, [32])
        self.assertEqual(cfg.batch_sizes, [1])
        self.assertEqual(cfg.warmup_runs, 1)
        self.assertEqual(cfg.timed_runs, 5)
        self.assertFalse(cfg.use_cache)


class TestMemoryBenchmarkConfig(unittest.TestCase):
    """Tests for MemoryBenchmarkConfig."""

    def test_minimal_config(self) -> None:
        """Minimal config with defaults."""
        cfg = MemoryBenchmarkConfig()
        self.assertEqual(cfg.type, BenchmarkType.MEMORY)

    def test_defaults(self) -> None:
        """Default values are applied."""
        cfg = MemoryBenchmarkConfig()
        self.assertEqual(cfg.sequence_lengths, [512, 1024, 2048, 4096])
        self.assertEqual(cfg.batch_sizes, [1, 4, 8])
        self.assertTrue(cfg.measure_peak)
        self.assertTrue(cfg.measure_kvcache)
        self.assertEqual(cfg.quantization_modes, ["fp16", "q8", "q4"])

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        cfg = MemoryBenchmarkConfig(
            sequence_lengths=[256, 512],
            batch_sizes=[1],
            measure_peak=False,
            measure_kvcache=True,
            quantization_modes=["fp16"],
        )
        self.assertEqual(cfg.sequence_lengths, [256, 512])
        self.assertEqual(cfg.batch_sizes, [1])
        self.assertFalse(cfg.measure_peak)
        self.assertTrue(cfg.measure_kvcache)
        self.assertEqual(cfg.quantization_modes, ["fp16"])


class TestAccuracyBenchmarkConfig(unittest.TestCase):
    """Tests for AccuracyBenchmarkConfig."""

    def test_minimal_config(self) -> None:
        """Minimal config with required fields."""
        cfg = AccuracyBenchmarkConfig(tasks=["hellaswag"])
        self.assertEqual(cfg.type, BenchmarkType.ACCURACY)
        self.assertEqual(cfg.tasks, ["hellaswag"])

    def test_defaults(self) -> None:
        """Default values are applied."""
        cfg = AccuracyBenchmarkConfig(tasks=["hellaswag"])
        self.assertEqual(cfg.num_fewshot, 0)
        self.assertIsNone(cfg.limit)

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        cfg = AccuracyBenchmarkConfig(
            tasks=["hellaswag", "winogrande"],
            num_fewshot=5,
            limit=100,
        )
        self.assertEqual(cfg.tasks, ["hellaswag", "winogrande"])
        self.assertEqual(cfg.num_fewshot, 5)
        self.assertEqual(cfg.limit, 100)


class TestGenerationBenchmarkConfig(unittest.TestCase):
    """Tests for GenerationBenchmarkConfig."""

    def test_minimal_config(self) -> None:
        """Minimal config with required fields."""
        cfg = GenerationBenchmarkConfig(prompts_file="prompts.yml")
        self.assertEqual(cfg.type, BenchmarkType.GENERATION)
        self.assertEqual(cfg.prompts_file, "prompts.yml")

    def test_defaults(self) -> None:
        """Default values are applied."""
        cfg = GenerationBenchmarkConfig(prompts_file="prompts.yml")
        self.assertEqual(cfg.max_new_tokens, 256)
        self.assertAlmostEqual(cfg.temperature, 1.0)
        self.assertAlmostEqual(cfg.top_p, 1.0)
        self.assertAlmostEqual(cfg.repetition_penalty, 1.0)


class TestBenchmarkSpec(unittest.TestCase):
    """Tests for BenchmarkSpec."""

    def test_minimal_spec(self) -> None:
        """Minimal spec with required fields."""
        spec = BenchmarkSpec(
            id="test",
            config=PerplexityBenchmarkConfig(dataset="data.npy"),
        )
        self.assertEqual(spec.id, "test")
        self.assertEqual(spec.models, ["teacher", "student"])
        self.assertEqual(spec.repeats, 1)

    def test_custom_models(self) -> None:
        """Custom models list."""
        spec = BenchmarkSpec(
            id="test",
            config=PerplexityBenchmarkConfig(dataset="data.npy"),
            models=["student"],
        )
        self.assertEqual(spec.models, ["student"])

    def test_custom_repeats(self) -> None:
        """Custom repeats value."""
        spec = BenchmarkSpec(
            id="test",
            config=PerplexityBenchmarkConfig(dataset="data.npy"),
            repeats=3,
        )
        self.assertEqual(spec.repeats, 3)


class TestBenchmarkSuite(unittest.TestCase):
    """Tests for BenchmarkSuite."""

    def test_minimal_suite(self) -> None:
        """Minimal suite with required fields."""
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="test",
                    config=PerplexityBenchmarkConfig(dataset="data.npy"),
                ),
            ],
        )
        self.assertEqual(len(suite.benchmarks), 1)

    def test_defaults(self) -> None:
        """Default values are applied."""
        suite = BenchmarkSuite(benchmarks=[])
        self.assertEqual(suite.output_dir, "artifacts")
        self.assertEqual(suite.formats, ["csv", "json", "png"])
        self.assertEqual(suite.comparison_baseline, "teacher")

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        suite = BenchmarkSuite(
            benchmarks=[],
            output_dir="custom_output",
            formats=["json", "latex"],
            comparison_baseline="student",
        )
        self.assertEqual(suite.output_dir, "custom_output")
        self.assertEqual(suite.formats, ["json", "latex"])
        self.assertEqual(suite.comparison_baseline, "student")


if __name__ == "__main__":
    unittest.main()
