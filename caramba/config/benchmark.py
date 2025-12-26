"""
benchmark provides config models for benchmarking pipelines.

Benchmarks measure:
- Perplexity (language modeling quality)
- Latency (tokens/second)
- Memory (KV-cache, peak memory)
- Accuracy (task-specific metrics)
"""
from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field

from caramba.config import PositiveFloat, PositiveInt, Probability


class BenchmarkType(str, enum.Enum):
    """Type of benchmark to run."""

    PERPLEXITY = "perplexity"
    LATENCY = "latency"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    GENERATION = "generation"


class PerplexityBenchmarkConfig(BaseModel):
    """Perplexity benchmark configuration."""

    type: Literal[BenchmarkType.PERPLEXITY] = BenchmarkType.PERPLEXITY
    dataset: str  # path to .npy or HF dataset
    block_size: PositiveInt = 2048
    batch_size: PositiveInt = 1
    num_batches: PositiveInt | None = None  # None = all
    stride: PositiveInt | None = None  # for sliding window perplexity


class LatencyBenchmarkConfig(BaseModel):
    """Latency/throughput benchmark configuration."""

    type: Literal[BenchmarkType.LATENCY] = BenchmarkType.LATENCY
    prompt_lengths: list[PositiveInt] = Field(default_factory=lambda: [128, 512, 1024, 2048])
    generation_lengths: list[PositiveInt] = Field(default_factory=lambda: [128, 256, 512])
    batch_sizes: list[PositiveInt] = Field(default_factory=lambda: [1, 4, 8])
    warmup_runs: PositiveInt = 3
    timed_runs: PositiveInt = 10
    use_cache: bool = True


class MemoryBenchmarkConfig(BaseModel):
    """Memory usage benchmark configuration."""

    type: Literal[BenchmarkType.MEMORY] = BenchmarkType.MEMORY
    sequence_lengths: list[PositiveInt] = Field(default_factory=lambda: [512, 1024, 2048, 4096])
    batch_sizes: list[PositiveInt] = Field(default_factory=lambda: [1, 4, 8])
    measure_peak: bool = True
    measure_kvcache: bool = True
    quantization_modes: list[str] = Field(default_factory=lambda: ["fp16", "q8", "q4"])


class AccuracyBenchmarkConfig(BaseModel):
    """Task accuracy benchmark configuration."""

    type: Literal[BenchmarkType.ACCURACY] = BenchmarkType.ACCURACY
    tasks: list[str]  # e.g., ["hellaswag", "winogrande", "arc_easy"]
    num_fewshot: PositiveInt = 0
    limit: PositiveInt | None = None  # limit samples per task


class GenerationBenchmarkConfig(BaseModel):
    """Text generation quality benchmark."""

    type: Literal[BenchmarkType.GENERATION] = BenchmarkType.GENERATION
    prompts_file: str  # path to prompts YAML
    max_new_tokens: PositiveInt = 256
    temperature: PositiveFloat = 1.0
    top_p: Probability = 1.0
    repetition_penalty: PositiveFloat = 1.0


BenchmarkConfig = (
    PerplexityBenchmarkConfig
    | LatencyBenchmarkConfig
    | MemoryBenchmarkConfig
    | AccuracyBenchmarkConfig
    | GenerationBenchmarkConfig
)


class BenchmarkSpec(BaseModel):
    """Specification for a single benchmark run."""

    id: str
    config: BenchmarkConfig = Field(discriminator="type")
    models: list[str] = Field(
        default_factory=lambda: ["teacher", "student"],
        description="Which models to benchmark (teacher, student, or both)",
    )
    repeats: PositiveInt = 1


class BenchmarkSuite(BaseModel):
    """Complete benchmark suite configuration."""

    benchmarks: list[BenchmarkSpec]
    output_dir: str = "artifacts"
    formats: list[str] = Field(
        default_factory=lambda: ["csv", "json", "png"],
        description="Output formats: csv, json, png (charts), latex",
    )
    comparison_baseline: str | None = "teacher"
