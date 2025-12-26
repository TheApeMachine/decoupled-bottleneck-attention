"""Benchmark configuration for measuring model quality and performance.

Benchmarks run after training to measure what matters:
- Perplexity: language modeling quality
- Latency: tokens per second
- Memory: KV-cache and peak usage
- Accuracy: downstream task performance
- Generation: text quality assessment
"""
from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field

from caramba.config import PositiveFloat, PositiveInt, Probability


class BenchmarkType(str, enum.Enum):
    """Types of benchmarks available."""

    PERPLEXITY = "perplexity"
    LATENCY = "latency"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    GENERATION = "generation"


class PerplexityBenchmarkConfig(BaseModel):
    """Measure language modeling perplexity on a dataset.

    Lower perplexity = better language modeling. This is the core
    metric for comparing model quality after upcycling.
    """

    type: Literal[BenchmarkType.PERPLEXITY] = BenchmarkType.PERPLEXITY
    dataset: str
    block_size: PositiveInt = 2048
    batch_size: PositiveInt = 1
    num_batches: PositiveInt | None = None
    stride: PositiveInt | None = None


class LatencyBenchmarkConfig(BaseModel):
    """Measure generation speed (tokens per second).

    Tests different prompt lengths, generation lengths, and batch sizes
    to characterize throughput across usage patterns.
    """

    type: Literal[BenchmarkType.LATENCY] = BenchmarkType.LATENCY
    prompt_lengths: list[PositiveInt] = Field(
        default_factory=lambda: [128, 512, 1024, 2048]
    )
    generation_lengths: list[PositiveInt] = Field(
        default_factory=lambda: [128, 256, 512]
    )
    batch_sizes: list[PositiveInt] = Field(default_factory=lambda: [1, 4, 8])
    warmup_runs: PositiveInt = 3
    timed_runs: PositiveInt = 10
    use_cache: bool = True


class MemoryBenchmarkConfig(BaseModel):
    """Measure memory usage (KV-cache and peak).

    For DBA upcycling, we expect significant KV-cache reduction due to
    the compressed attention dimensions.
    """

    type: Literal[BenchmarkType.MEMORY] = BenchmarkType.MEMORY
    sequence_lengths: list[PositiveInt] = Field(
        default_factory=lambda: [512, 1024, 2048, 4096]
    )
    batch_sizes: list[PositiveInt] = Field(default_factory=lambda: [1, 4, 8])
    measure_peak: bool = True
    measure_kvcache: bool = True
    quantization_modes: list[str] = Field(
        default_factory=lambda: ["fp16", "q8", "q4"]
    )


class AccuracyBenchmarkConfig(BaseModel):
    """Measure accuracy on downstream tasks.

    Uses standard evaluation benchmarks like HellaSwag, WinoGrande, etc.
    to assess whether model capabilities are preserved after upcycling.
    """

    type: Literal[BenchmarkType.ACCURACY] = BenchmarkType.ACCURACY
    tasks: list[str]
    num_fewshot: PositiveInt = 0
    limit: PositiveInt | None = None


class GenerationBenchmarkConfig(BaseModel):
    """Assess text generation quality.

    Runs generation on curated prompts to qualitatively evaluate
    the model's output quality.
    """

    type: Literal[BenchmarkType.GENERATION] = BenchmarkType.GENERATION
    prompts_file: str
    max_new_tokens: PositiveInt = 256
    temperature: PositiveFloat = 1.0
    top_p: Probability = 1.0
    repetition_penalty: PositiveFloat = 1.0


# Union of all benchmark config types
BenchmarkConfig = (
    PerplexityBenchmarkConfig
    | LatencyBenchmarkConfig
    | MemoryBenchmarkConfig
    | AccuracyBenchmarkConfig
    | GenerationBenchmarkConfig
)


class BenchmarkSpec(BaseModel):
    """Specification for running a benchmark.

    Wraps a benchmark config with metadata like which models to test
    and how many times to repeat for statistical confidence.
    """

    id: str
    config: BenchmarkConfig = Field(discriminator="type")
    models: list[str] = Field(
        default_factory=lambda: ["teacher", "student"],
        description="Which models to benchmark",
    )
    repeats: PositiveInt = 1


class BenchmarkSuite(BaseModel):
    """Complete benchmark suite for an experiment.

    Collects multiple benchmarks and configures output formats.
    """

    benchmarks: list[BenchmarkSpec]
    output_dir: str = "artifacts"
    formats: list[str] = Field(
        default_factory=lambda: ["csv", "json", "png"],
        description="Output formats: csv, json, png, latex",
    )
    comparison_baseline: str | None = "teacher"
