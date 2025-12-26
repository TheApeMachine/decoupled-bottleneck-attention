"""
Unit tests for the latency benchmark module.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn

from caramba.benchmark.latency import (
    LatencyBenchmark,
    LatencyMeasurement,
    LatencyResult,
)
from caramba.config.benchmark import LatencyBenchmarkConfig


class DummyModel(nn.Module):
    """Simple model for latency testing.

    Accepts the `ctx` kwarg for compatibility with Generator-based
    cached latency benchmarks, but ignores it (no actual KV-cache).
    """

    def __init__(self, vocab_size: int = 32000, d_model: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
        # ctx is accepted for Generator compatibility but ignored
        # (this dummy model has no actual KV-cache)
        h = self.embed(x)
        h = self.linear(h)
        return self.head(h)


class TestLatencyMeasurement(unittest.TestCase):
    """Tests for LatencyMeasurement dataclass."""

    def test_measurement_fields(self) -> None:
        """Measurement stores all required fields."""
        m = LatencyMeasurement(
            prompt_len=512,
            gen_len=128,
            batch_size=4,
            prefill_time_ms=10.5,
            decode_time_ms=50.0,
            total_time_ms=60.5,
            tokens_per_second=1000.0,
            time_to_first_token_ms=10.5,
        )
        self.assertEqual(m.prompt_len, 512)
        self.assertEqual(m.gen_len, 128)
        self.assertEqual(m.batch_size, 4)
        self.assertAlmostEqual(m.prefill_time_ms, 10.5)
        self.assertAlmostEqual(m.decode_time_ms, 50.0)
        self.assertAlmostEqual(m.total_time_ms, 60.5)
        self.assertAlmostEqual(m.tokens_per_second, 1000.0)
        self.assertAlmostEqual(m.time_to_first_token_ms, 10.5)


class TestLatencyResult(unittest.TestCase):
    """Tests for LatencyResult dataclass."""

    def test_avg_tokens_per_second_empty(self) -> None:
        """Average is 0 for empty measurements."""
        result = LatencyResult(model_name="test")
        self.assertAlmostEqual(result.avg_tokens_per_second, 0.0)

    def test_avg_tokens_per_second_single(self) -> None:
        """Average for single measurement."""
        result = LatencyResult(
            model_name="test",
            measurements=[
                LatencyMeasurement(
                    prompt_len=128, gen_len=64, batch_size=1,
                    prefill_time_ms=5.0, decode_time_ms=20.0,
                    total_time_ms=25.0, tokens_per_second=100.0,
                    time_to_first_token_ms=5.0,
                )
            ],
        )
        self.assertAlmostEqual(result.avg_tokens_per_second, 100.0)

    def test_avg_tokens_per_second_multiple(self) -> None:
        """Average of multiple measurements."""
        result = LatencyResult(
            model_name="test",
            measurements=[
                LatencyMeasurement(
                    prompt_len=128, gen_len=64, batch_size=1,
                    prefill_time_ms=5.0, decode_time_ms=20.0,
                    total_time_ms=25.0, tokens_per_second=100.0,
                    time_to_first_token_ms=5.0,
                ),
                LatencyMeasurement(
                    prompt_len=256, gen_len=64, batch_size=1,
                    prefill_time_ms=10.0, decode_time_ms=20.0,
                    total_time_ms=30.0, tokens_per_second=200.0,
                    time_to_first_token_ms=10.0,
                ),
            ],
        )
        self.assertAlmostEqual(result.avg_tokens_per_second, 150.0)

    def test_avg_time_to_first_token(self) -> None:
        """Average time to first token."""
        result = LatencyResult(
            model_name="test",
            measurements=[
                LatencyMeasurement(
                    prompt_len=128, gen_len=64, batch_size=1,
                    prefill_time_ms=5.0, decode_time_ms=20.0,
                    total_time_ms=25.0, tokens_per_second=100.0,
                    time_to_first_token_ms=5.0,
                ),
                LatencyMeasurement(
                    prompt_len=256, gen_len=64, batch_size=1,
                    prefill_time_ms=15.0, decode_time_ms=20.0,
                    total_time_ms=35.0, tokens_per_second=100.0,
                    time_to_first_token_ms=15.0,
                ),
            ],
        )
        self.assertAlmostEqual(result.avg_time_to_first_token_ms, 10.0)


class TestLatencyBenchmark(unittest.TestCase):
    """Tests for LatencyBenchmark."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_run_returns_result(self) -> None:
        """Benchmark run returns a LatencyResult."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=1,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertIsInstance(result, LatencyResult)
        self.assertEqual(result.model_name, "test_model")

    def test_measurements_count(self) -> None:
        """Number of measurements matches config combinations."""
        prompt_lens = [16, 32]
        gen_lens = [4, 8]
        batch_sizes = [1]

        config = LatencyBenchmarkConfig(
            prompt_lengths=prompt_lens,
            generation_lengths=gen_lens,
            batch_sizes=batch_sizes,
            warmup_runs=1,
            timed_runs=1,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        expected_count = len(prompt_lens) * len(gen_lens) * len(batch_sizes)
        self.assertEqual(len(result.measurements), expected_count)

    def test_times_are_positive(self) -> None:
        """All timing measurements are positive."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=2,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        for m in result.measurements:
            self.assertGreater(m.prefill_time_ms, 0)
            self.assertGreater(m.decode_time_ms, 0)
            self.assertGreater(m.total_time_ms, 0)
            self.assertGreater(m.tokens_per_second, 0)

    def test_total_equals_prefill_plus_decode(self) -> None:
        """Total time equals prefill + decode."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=2,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        for m in result.measurements:
            expected_total = m.prefill_time_ms + m.decode_time_ms
            self.assertAlmostEqual(m.total_time_ms, expected_total, places=2)

    def test_measurement_config_matches(self) -> None:
        """Measurement stores correct config values."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[32],
            generation_lengths=[8],
            batch_sizes=[2],
            warmup_runs=1,
            timed_runs=1,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertEqual(len(result.measurements), 1)
        m = result.measurements[0]
        self.assertEqual(m.prompt_len, 32)
        self.assertEqual(m.gen_len, 8)
        self.assertEqual(m.batch_size, 2)


class TestLatencyBenchmarkWithCache(unittest.TestCase):
    """Tests for LatencyBenchmark with KV-cache enabled."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_cached_run_returns_result(self) -> None:
        """Cached benchmark run returns a LatencyResult."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=1,
            use_cache=True,  # Enable KV-cache mode
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertIsInstance(result, LatencyResult)
        self.assertEqual(result.model_name, "test_model")

    def test_cached_measurement_has_use_cache_true(self) -> None:
        """Cached measurements have use_cache=True flag."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=1,
            use_cache=True,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        for m in result.measurements:
            self.assertTrue(m.use_cache)

    def test_uncached_measurement_has_use_cache_false(self) -> None:
        """Uncached measurements have use_cache=False flag."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=1,
            use_cache=False,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        for m in result.measurements:
            self.assertFalse(m.use_cache)

    def test_cached_times_are_positive(self) -> None:
        """All cached timing measurements are positive."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=2,
            use_cache=True,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        for m in result.measurements:
            self.assertGreater(m.prefill_time_ms, 0)
            self.assertGreater(m.decode_time_ms, 0)
            self.assertGreater(m.total_time_ms, 0)
            self.assertGreater(m.tokens_per_second, 0)
            self.assertGreater(m.time_to_first_token_ms, 0)

    def test_cached_ttft_includes_first_decode(self) -> None:
        """Cached TTFT is at least prefill time (includes first decode step)."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=2,
            use_cache=True,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        for m in result.measurements:
            # TTFT should be >= prefill (it includes first decode step)
            self.assertGreaterEqual(m.time_to_first_token_ms, m.prefill_time_ms)


class TestLatencyMeasurementUseCache(unittest.TestCase):
    """Tests for use_cache field in LatencyMeasurement."""

    def test_measurement_use_cache_default(self) -> None:
        """use_cache defaults to False."""
        m = LatencyMeasurement(
            prompt_len=128,
            gen_len=64,
            batch_size=1,
            prefill_time_ms=10.0,
            decode_time_ms=50.0,
            total_time_ms=60.0,
            tokens_per_second=1000.0,
            time_to_first_token_ms=10.0,
        )
        self.assertFalse(m.use_cache)

    def test_measurement_use_cache_explicit_true(self) -> None:
        """use_cache can be set to True."""
        m = LatencyMeasurement(
            prompt_len=128,
            gen_len=64,
            batch_size=1,
            prefill_time_ms=10.0,
            decode_time_ms=50.0,
            total_time_ms=60.0,
            tokens_per_second=1000.0,
            time_to_first_token_ms=10.0,
            use_cache=True,
        )
        self.assertTrue(m.use_cache)


class TestTTFTSemanticConsistency(unittest.TestCase):
    """Tests for TTFT semantic consistency between cached and uncached modes.

    TTFT (Time To First Token) should have consistent semantics across modes:
    - Both cached and uncached modes report TTFT as prefill + first decode step
    - This ensures TTFT is comparable across different benchmark configurations
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_uncached_ttft_includes_first_decode(self) -> None:
        """Uncached TTFT includes first decode step (consistent with cached mode)."""
        config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=2,
            use_cache=False,
        )
        model = DummyModel()
        model.eval()

        benchmark = LatencyBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        for m in result.measurements:
            # TTFT should be > prefill (it includes first decode step)
            self.assertGreater(m.time_to_first_token_ms, m.prefill_time_ms)

    def test_ttft_semantics_match_across_modes(self) -> None:
        """Both cached and uncached TTFT have the same semantic definition.

        Both modes should report TTFT as prefill + first decode, meaning
        TTFT > prefill_time for both modes.
        """
        model = DummyModel()
        model.eval()

        # Run uncached benchmark
        uncached_config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=2,
            use_cache=False,
        )
        uncached_benchmark = LatencyBenchmark(uncached_config, self.device)
        uncached_result = uncached_benchmark.run(model, "test_model_uncached")

        # Run cached benchmark
        cached_config = LatencyBenchmarkConfig(
            prompt_lengths=[16],
            generation_lengths=[4],
            batch_sizes=[1],
            warmup_runs=1,
            timed_runs=2,
            use_cache=True,
        )
        cached_benchmark = LatencyBenchmark(cached_config, self.device)
        cached_result = cached_benchmark.run(model, "test_model_cached")

        # Both should have TTFT > prefill (includes first decode step)
        for m in uncached_result.measurements:
            self.assertGreater(
                m.time_to_first_token_ms, m.prefill_time_ms,
                "Uncached TTFT should include first decode step"
            )

        for m in cached_result.measurements:
            self.assertGreaterEqual(
                m.time_to_first_token_ms, m.prefill_time_ms,
                "Cached TTFT should include first decode step"
            )


if __name__ == "__main__":
    unittest.main()
