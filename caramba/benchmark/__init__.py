"""
benchmark provides benchmarking utilities for model comparison.

This module implements:
- Perplexity measurement
- Latency/throughput measurement
- Memory profiling (including KV-cache)
- Result aggregation and artifact generation
"""
from __future__ import annotations

from caramba.benchmark.runner import BenchmarkRunner
from caramba.benchmark.perplexity import PerplexityBenchmark
from caramba.benchmark.latency import LatencyBenchmark
from caramba.benchmark.memory import MemoryBenchmark
from caramba.benchmark.artifacts import ArtifactGenerator

__all__ = [
    "BenchmarkRunner",
    "PerplexityBenchmark",
    "LatencyBenchmark",
    "MemoryBenchmark",
    "ArtifactGenerator",
]
