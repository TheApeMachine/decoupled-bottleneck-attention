"""
Unit tests for the memory benchmark module.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn

from caramba.benchmark.memory import (
    KVCacheAnalysis,
    MemoryBenchmark,
    MemoryMeasurement,
    MemoryResult,
)
from caramba.config.benchmark import MemoryBenchmarkConfig
from caramba.config.layer import AttentionLayerConfig, AttentionMode, LayerType
from caramba.layer.attention import AttentionLayer


class DummyModel(nn.Module):
    """Simple model without attention for basic testing."""

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


class AttentionModel(nn.Module):
    """Model with attention layers for KV-cache analysis."""

    def __init__(self, n_layers: int = 2, d_model: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(32000, d_model)
        self.layers = nn.ModuleList([
            AttentionLayer(AttentionLayerConfig(
                type=LayerType.ATTENTION,
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
            ))
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 32000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            out, _ = layer(h)
            h = out
        return self.head(h)


class DBAModel(nn.Module):
    """Model with DBA attention for decoupled cache analysis."""

    def __init__(
        self,
        n_layers: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        sem_dim: int = 16,
        geo_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(32000, d_model)
        self.layers = nn.ModuleList([
            AttentionLayer(AttentionLayerConfig(
                type=LayerType.ATTENTION,
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                mode=AttentionMode.DECOUPLED,
                sem_dim=sem_dim,
                geo_dim=geo_dim,
            ))
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 32000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            out, _ = layer(h)
            h = out
        return self.head(h)


class TestMemoryMeasurement(unittest.TestCase):
    """Tests for MemoryMeasurement dataclass."""

    def test_measurement_fields(self) -> None:
        """Measurement stores all required fields."""
        m = MemoryMeasurement(
            seq_len=512,
            batch_size=4,
            peak_memory_mb=1024.0,
            kvcache_memory_mb=256.0,
            model_memory_mb=512.0,
            quantization="fp16",
        )
        self.assertEqual(m.seq_len, 512)
        self.assertEqual(m.batch_size, 4)
        self.assertAlmostEqual(m.peak_memory_mb, 1024.0)
        self.assertAlmostEqual(m.kvcache_memory_mb, 256.0)
        self.assertAlmostEqual(m.model_memory_mb, 512.0)
        self.assertEqual(m.quantization, "fp16")


class TestKVCacheAnalysis(unittest.TestCase):
    """Tests for KVCacheAnalysis dataclass."""

    def test_standard_analysis(self) -> None:
        """Analysis for standard attention."""
        analysis = KVCacheAnalysis(
            model_name="test",
            n_layers=16,
            n_kv_heads=8,
            head_dim=64,
            attention_mode="standard",
            bytes_per_token_fp16=2048.0,
            bytes_per_token_q8=1024.0,
            bytes_per_token_q4=512.0,
        )
        self.assertEqual(analysis.n_layers, 16)
        self.assertEqual(analysis.n_kv_heads, 8)
        self.assertEqual(analysis.head_dim, 64)
        self.assertEqual(analysis.attention_mode, "standard")
        self.assertIsNone(analysis.sem_dim)
        self.assertIsNone(analysis.geo_dim)

    def test_dba_analysis(self) -> None:
        """Analysis for DBA attention includes sem/geo dims."""
        analysis = KVCacheAnalysis(
            model_name="test",
            n_layers=16,
            n_kv_heads=8,
            head_dim=64,
            attention_mode="decoupled",
            bytes_per_token_fp16=2048.0,
            bytes_per_token_q8=1024.0,
            bytes_per_token_q4=512.0,
            sem_dim=128,
            geo_dim=256,
            bytes_per_token_dba_fp16=384.0,
            bytes_per_token_dba_q8=192.0,
            bytes_per_token_dba_q4=120.0,
        )
        self.assertEqual(analysis.sem_dim, 128)
        self.assertEqual(analysis.geo_dim, 256)
        self.assertIsNotNone(analysis.bytes_per_token_dba_fp16)
        dba_bytes = analysis.bytes_per_token_dba_fp16
        self.assertIsNotNone(dba_bytes)
        # Type narrowing: after assertIsNotNone, dba_bytes is known to be float
        self.assertAlmostEqual(dba_bytes, 384.0)  # type: ignore[arg-type]
        # Verify q8 and q4 estimates
        self.assertIsNotNone(analysis.bytes_per_token_dba_q8)
        self.assertIsNotNone(analysis.bytes_per_token_dba_q4)


class TestMemoryResult(unittest.TestCase):
    """Tests for MemoryResult dataclass."""

    def test_peak_memory_empty(self) -> None:
        """Peak memory is 0 for empty measurements."""
        result = MemoryResult(model_name="test")
        self.assertAlmostEqual(result.peak_memory_mb, 0.0)

    def test_peak_memory_single(self) -> None:
        """Peak memory for single measurement."""
        result = MemoryResult(
            model_name="test",
            measurements=[
                MemoryMeasurement(
                    seq_len=512,
                    batch_size=1,
                    peak_memory_mb=100.0,
                    kvcache_memory_mb=50.0,
                    model_memory_mb=40.0,
                    quantization="fp16",
                )
            ],
        )
        self.assertAlmostEqual(result.peak_memory_mb, 100.0)

    def test_peak_memory_multiple(self) -> None:
        """Peak memory is max of all measurements."""
        result = MemoryResult(
            model_name="test",
            measurements=[
                MemoryMeasurement(
                    seq_len=512, batch_size=1,
                    peak_memory_mb=100.0, kvcache_memory_mb=50.0,
                    model_memory_mb=40.0, quantization="fp16",
                ),
                MemoryMeasurement(
                    seq_len=1024, batch_size=1,
                    peak_memory_mb=200.0, kvcache_memory_mb=100.0,
                    model_memory_mb=40.0, quantization="fp16",
                ),
                MemoryMeasurement(
                    seq_len=2048, batch_size=1,
                    peak_memory_mb=150.0, kvcache_memory_mb=75.0,
                    model_memory_mb=40.0, quantization="fp16",
                ),
            ],
        )
        self.assertAlmostEqual(result.peak_memory_mb, 200.0)


class TestMemoryBenchmark(unittest.TestCase):
    """Tests for MemoryBenchmark."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_run_returns_result(self) -> None:
        """Benchmark run returns a MemoryResult."""
        config = MemoryBenchmarkConfig(
            sequence_lengths=[32],
            batch_sizes=[1],
            quantization_modes=["fp16"],
        )
        model = DummyModel()
        model.eval()

        benchmark = MemoryBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertIsInstance(result, MemoryResult)
        self.assertEqual(result.model_name, "test_model")

    def test_measurements_count(self) -> None:
        """Number of measurements matches config combinations."""
        seq_lens = [32, 64]
        batch_sizes = [1, 2]
        quants = ["fp16", "q8"]

        config = MemoryBenchmarkConfig(
            sequence_lengths=seq_lens,
            batch_sizes=batch_sizes,
            quantization_modes=quants,
        )
        model = DummyModel()
        model.eval()

        benchmark = MemoryBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        expected_count = len(seq_lens) * len(batch_sizes) * len(quants)
        self.assertEqual(len(result.measurements), expected_count)

    def test_kvcache_analysis_standard(self) -> None:
        """KV-cache analysis for standard attention model."""
        config = MemoryBenchmarkConfig(
            sequence_lengths=[32],
            batch_sizes=[1],
            quantization_modes=["fp16"],
        )
        model = AttentionModel(n_layers=2, d_model=64, n_heads=4)
        model.eval()

        benchmark = MemoryBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertIsNotNone(result.kvcache_analysis)
        analysis = result.kvcache_analysis
        assert analysis is not None
        self.assertEqual(analysis.n_layers, 2)
        self.assertEqual(analysis.n_kv_heads, 4)
        self.assertEqual(analysis.attention_mode, "standard")

    def test_kvcache_analysis_dba(self) -> None:
        """KV-cache analysis for DBA attention model."""
        config = MemoryBenchmarkConfig(
            sequence_lengths=[32],
            batch_sizes=[1],
            quantization_modes=["fp16"],
        )
        model = DBAModel(n_layers=2, d_model=64, n_heads=4, sem_dim=16, geo_dim=32)
        model.eval()

        benchmark = MemoryBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        self.assertIsNotNone(result.kvcache_analysis)
        analysis = result.kvcache_analysis
        assert analysis is not None
        self.assertEqual(analysis.n_layers, 2)
        self.assertEqual(analysis.attention_mode, "decoupled")
        self.assertEqual(analysis.sem_dim, 16)
        self.assertEqual(analysis.geo_dim, 32)
        self.assertIsNotNone(analysis.bytes_per_token_dba_fp16)
        # Verify all quantized DBA estimates are present
        self.assertIsNotNone(analysis.bytes_per_token_dba_q8)
        self.assertIsNotNone(analysis.bytes_per_token_dba_q4)

    def test_kvcache_dba_quantized_estimates(self) -> None:
        """DBA KV-cache estimates maintain correct ratios across quantization levels.

        For DBA, the KV-cache stores sem_dim + geo_dim + v_dim elements per token per layer.
        The byte estimates should scale according to:
          - fp16: 2 bytes/element
          - q8: 1 byte/element
          - q4: 0.625 bytes/element
        """
        config = MemoryBenchmarkConfig(
            sequence_lengths=[32],
            batch_sizes=[1],
            quantization_modes=["fp16", "q8", "q4"],
        )
        # Create DBA model with known dimensions
        model = DBAModel(n_layers=2, d_model=64, n_heads=4, sem_dim=16, geo_dim=32)
        model.eval()

        benchmark = MemoryBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        analysis = result.kvcache_analysis
        assert analysis is not None
        assert analysis.bytes_per_token_dba_fp16 is not None
        assert analysis.bytes_per_token_dba_q8 is not None
        assert analysis.bytes_per_token_dba_q4 is not None

        # Verify ratios are correct
        fp16_to_q8_ratio = analysis.bytes_per_token_dba_fp16 / analysis.bytes_per_token_dba_q8
        self.assertAlmostEqual(fp16_to_q8_ratio, 2.0, places=2)

        fp16_to_q4_ratio = analysis.bytes_per_token_dba_fp16 / analysis.bytes_per_token_dba_q4
        # fp16 (2.0) / q4 (0.625) = 3.2
        self.assertAlmostEqual(fp16_to_q4_ratio, 3.2, places=2)

    def test_kvcache_estimation_scales_with_seq_len(self) -> None:
        """KV-cache memory estimation scales linearly with sequence length."""
        config = MemoryBenchmarkConfig(
            sequence_lengths=[32, 64],
            batch_sizes=[1],
            quantization_modes=["fp16"],
        )
        model = AttentionModel(n_layers=2, d_model=64, n_heads=4)
        model.eval()

        benchmark = MemoryBenchmark(config, self.device)
        result = benchmark.run(model, "test_model")

        # Find measurements for each seq_len
        m32 = next(m for m in result.measurements if m.seq_len == 32)
        m64 = next(m for m in result.measurements if m.seq_len == 64)

        # KV-cache should scale linearly
        ratio = m64.kvcache_memory_mb / m32.kvcache_memory_mb
        self.assertAlmostEqual(ratio, 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
