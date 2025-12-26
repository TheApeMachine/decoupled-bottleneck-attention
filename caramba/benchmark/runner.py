"""
runner provides the unified benchmark runner.

This orchestrates all benchmarking and artifact generation.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from caramba.benchmark.artifacts import ArtifactGenerator, ExperimentMetadata
from caramba.benchmark.latency import LatencyBenchmark, LatencyResult
from caramba.benchmark.memory import MemoryBenchmark, MemoryResult
from caramba.benchmark.perplexity import PerplexityBenchmark, PerplexityResult
from caramba.config.benchmark import (
    BenchmarkSuite,
    BenchmarkType,
    LatencyBenchmarkConfig,
    MemoryBenchmarkConfig,
    PerplexityBenchmarkConfig,
)


class BenchmarkRunner:
    """
    Unified benchmark runner that executes all configured benchmarks
    and generates paper-ready artifacts.
    """

    def __init__(
        self,
        suite: BenchmarkSuite,
        device: torch.device,
        metadata: ExperimentMetadata,
    ) -> None:
        self.suite = suite
        self.device = device
        self.metadata = metadata
        self.output_dir = Path(suite.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        teacher: nn.Module,
        student: nn.Module,
    ) -> dict[str, Path]:
        """
        Run all benchmarks and generate artifacts.

        Returns a dict of artifact paths.
        """
        print(f"benchmark: running {len(self.suite.benchmarks)} benchmarks")

        # Collect results
        teacher_perplexity: PerplexityResult | None = None
        student_perplexity: PerplexityResult | None = None
        teacher_latency: LatencyResult | None = None
        student_latency: LatencyResult | None = None
        teacher_memory: MemoryResult | None = None
        student_memory: MemoryResult | None = None

        for spec in self.suite.benchmarks:
            print(f"benchmark: {spec.id} ({spec.config.type})")

            for _ in range(spec.repeats):
                match spec.config.type:
                    case BenchmarkType.PERPLEXITY:
                        assert isinstance(spec.config, PerplexityBenchmarkConfig)
                        benchmark = PerplexityBenchmark(spec.config, self.device)

                        if "teacher" in spec.models:
                            result = benchmark.run(teacher, "teacher")
                            if teacher_perplexity is None or result.perplexity < teacher_perplexity.perplexity:
                                teacher_perplexity = result
                            print(f"  teacher: ppl={result.perplexity:.2f}")

                        if "student" in spec.models:
                            result = benchmark.run(student, "student")
                            if student_perplexity is None or result.perplexity < student_perplexity.perplexity:
                                student_perplexity = result
                            print(f"  student: ppl={result.perplexity:.2f}")

                    case BenchmarkType.LATENCY:
                        assert isinstance(spec.config, LatencyBenchmarkConfig)
                        benchmark = LatencyBenchmark(spec.config, self.device)

                        if "teacher" in spec.models:
                            result = benchmark.run(teacher, "teacher")
                            if teacher_latency is None:
                                teacher_latency = result
                            print(f"  teacher: {result.avg_tokens_per_second:.1f} tok/s")

                        if "student" in spec.models:
                            result = benchmark.run(student, "student")
                            if student_latency is None:
                                student_latency = result
                            print(f"  student: {result.avg_tokens_per_second:.1f} tok/s")

                    case BenchmarkType.MEMORY:
                        assert isinstance(spec.config, MemoryBenchmarkConfig)
                        benchmark = MemoryBenchmark(spec.config, self.device)

                        if "teacher" in spec.models:
                            result = benchmark.run(teacher, "teacher")
                            if teacher_memory is None:
                                teacher_memory = result
                            if result.kvcache_analysis:
                                print(f"  teacher: {result.kvcache_analysis.bytes_per_token_fp16:.0f} bytes/tok")

                        if "student" in spec.models:
                            result = benchmark.run(student, "student")
                            if student_memory is None:
                                student_memory = result
                            if result.kvcache_analysis:
                                kv_bytes = result.kvcache_analysis.bytes_per_token_dba_fp16 or result.kvcache_analysis.bytes_per_token_fp16
                                print(f"  student: {kv_bytes:.0f} bytes/tok")

                    case _:
                        print(f"  skipping unsupported benchmark type: {spec.config.type}")

        # Generate artifacts
        print("benchmark: generating artifacts")
        generator = ArtifactGenerator(self.output_dir)
        paths = generator.generate_all(
            metadata=self.metadata,
            teacher_perplexity=teacher_perplexity,
            student_perplexity=student_perplexity,
            teacher_latency=teacher_latency,
            student_latency=student_latency,
            teacher_memory=teacher_memory,
            student_memory=student_memory,
            formats=self.suite.formats,
        )

        print(f"benchmark: generated {len(paths)} artifacts in {self.output_dir}")
        for name, path in paths.items():
            print(f"  - {name}: {path}")

        return paths


class QuickBenchmark:
    """
    Quick benchmark for sanity checking during development.

    Runs a minimal set of benchmarks with reduced iterations.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        model: nn.Module,
        dataset_path: str,
        num_batches: int = 10,
    ) -> dict[str, float]:
        """Run quick benchmark and return metrics dict."""
        model.eval()

        # Quick perplexity
        ppl_config = PerplexityBenchmarkConfig(
            dataset=dataset_path,
            block_size=512,
            batch_size=1,
            num_batches=num_batches,
        )
        ppl_benchmark = PerplexityBenchmark(ppl_config, self.device)
        ppl_result = ppl_benchmark.run(model, "model")

        # Quick memory analysis
        mem_config = MemoryBenchmarkConfig(
            sequence_lengths=[512],
            batch_sizes=[1],
            quantization_modes=["fp16"],
        )
        mem_benchmark = MemoryBenchmark(mem_config, self.device)
        mem_result = mem_benchmark.run(model, "model")

        kv_bytes = 0.0
        if mem_result.kvcache_analysis:
            if mem_result.kvcache_analysis.bytes_per_token_dba_fp16:
                kv_bytes = mem_result.kvcache_analysis.bytes_per_token_dba_fp16
            else:
                kv_bytes = mem_result.kvcache_analysis.bytes_per_token_fp16

        return {
            "perplexity": ppl_result.perplexity,
            "loss": ppl_result.loss,
            "kv_bytes_per_token": kv_bytes,
        }
