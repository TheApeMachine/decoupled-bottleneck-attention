"""Artifact generation for paper-ready outputs.

After benchmarking, we need to present results. This module generates:
- CSV files: Raw data for further analysis
- JSON reports: Structured summary with metadata
- PNG charts: Visual comparisons
- LaTeX tables: Ready for paper inclusion
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from caramba.benchmark.latency import LatencyResult
from caramba.benchmark.memory import MemoryResult
from caramba.benchmark.perplexity import PerplexityResult


@dataclass
class ExperimentMetadata:
    """Metadata describing the experiment."""

    name: str
    timestamp: str
    manifest_path: str
    teacher_checkpoint: str
    student_config: str
    device: str
    notes: str = ""


@dataclass
class ComparisonSummary:
    """Summary comparing teacher and student model performance."""

    teacher_perplexity: float
    student_perplexity: float
    perplexity_ratio: float

    teacher_tokens_per_sec: float
    student_tokens_per_sec: float
    speedup: float

    teacher_kvcache_bytes_per_token: float
    student_kvcache_bytes_per_token: float
    memory_reduction: float

    @property
    def teacher_kvcache_mb_per_token(self) -> float:
        """Teacher KV-cache size in MB per token (for display)."""
        return self.teacher_kvcache_bytes_per_token / (1024 * 1024)

    @property
    def student_kvcache_mb_per_token(self) -> float:
        """Student KV-cache size in MB per token (for display)."""
        return self.student_kvcache_bytes_per_token / (1024 * 1024)


class ArtifactGenerator:
    """Generates paper-ready artifacts from benchmark results.

    Supports multiple output formats: CSV for data analysis, JSON for
    programmatic access, PNG for figures, and LaTeX for direct paper
    inclusion.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Set up the generator with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        *,
        metadata: ExperimentMetadata,
        teacher_perplexity: PerplexityResult | None = None,
        student_perplexity: PerplexityResult | None = None,
        teacher_latency: LatencyResult | None = None,
        student_latency: LatencyResult | None = None,
        teacher_memory: MemoryResult | None = None,
        student_memory: MemoryResult | None = None,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Generate all artifacts and return a dict of paths."""
        formats = formats or ["csv", "json", "png", "latex"]
        generated: dict[str, Path] = {}

        summary = self._compute_summary(
            teacher_perplexity=teacher_perplexity,
            student_perplexity=student_perplexity,
            teacher_latency=teacher_latency,
            student_latency=student_latency,
            teacher_memory=teacher_memory,
            student_memory=student_memory,
        )

        if "json" in formats:
            path = self._write_json_report(metadata, summary)
            generated["report.json"] = path

        if "csv" in formats:
            paths = self._write_csv_files(
                teacher_perplexity=teacher_perplexity,
                student_perplexity=student_perplexity,
                teacher_latency=teacher_latency,
                student_latency=student_latency,
                teacher_memory=teacher_memory,
                student_memory=student_memory,
            )
            generated.update(paths)

        if "png" in formats:
            paths = self._generate_charts(
                summary=summary,
                teacher_latency=teacher_latency,
                student_latency=student_latency,
                teacher_memory=teacher_memory,
                student_memory=student_memory,
            )
            generated.update(paths)

        if "latex" in formats:
            path = self._write_latex_tables(metadata, summary)
            generated["tables.tex"] = path

        return generated

    def _compute_summary(
        self,
        teacher_perplexity: PerplexityResult | None,
        student_perplexity: PerplexityResult | None,
        teacher_latency: LatencyResult | None,
        student_latency: LatencyResult | None,
        teacher_memory: MemoryResult | None,
        student_memory: MemoryResult | None,
    ) -> ComparisonSummary:
        """Compute comparison summary from individual results."""
        t_ppl = teacher_perplexity.perplexity if teacher_perplexity else 0.0
        s_ppl = student_perplexity.perplexity if student_perplexity else 0.0

        t_tps = teacher_latency.avg_tokens_per_second if teacher_latency else 0.0
        s_tps = student_latency.avg_tokens_per_second if student_latency else 0.0

        t_mem = (
            teacher_memory.kvcache_analysis.bytes_per_token_fp16
            if teacher_memory and teacher_memory.kvcache_analysis
            else 0.0
        )
        s_mem = (
            student_memory.kvcache_analysis.bytes_per_token_dba_fp16
            if student_memory
            and student_memory.kvcache_analysis
            and student_memory.kvcache_analysis.bytes_per_token_dba_fp16
            else student_memory.kvcache_analysis.bytes_per_token_fp16
            if student_memory and student_memory.kvcache_analysis
            else 0.0
        )

        return ComparisonSummary(
            teacher_perplexity=t_ppl,
            student_perplexity=s_ppl,
            perplexity_ratio=s_ppl / t_ppl if t_ppl > 0 else 0.0,
            teacher_tokens_per_sec=t_tps,
            student_tokens_per_sec=s_tps,
            speedup=s_tps / t_tps if t_tps > 0 else 0.0,
            teacher_kvcache_bytes_per_token=t_mem if t_mem else 0.0,
            student_kvcache_bytes_per_token=s_mem if s_mem else 0.0,
            memory_reduction=t_mem / s_mem if s_mem > 0 else 0.0,
        )

    def _write_json_report(
        self,
        metadata: ExperimentMetadata,
        summary: ComparisonSummary,
    ) -> Path:
        """Write JSON summary report with metadata and comparison."""
        path = self.output_dir / "report.json"

        report = {
            "metadata": asdict(metadata),
            "summary": asdict(summary),
            "generated_at": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        return path

    def _write_csv_files(
        self,
        teacher_perplexity: PerplexityResult | None,
        student_perplexity: PerplexityResult | None,
        teacher_latency: LatencyResult | None,
        student_latency: LatencyResult | None,
        teacher_memory: MemoryResult | None,
        student_memory: MemoryResult | None,
    ) -> dict[str, Path]:
        """Write CSV files with raw benchmark data."""
        paths: dict[str, Path] = {}

        # Perplexity CSV
        if teacher_perplexity or student_perplexity:
            path = self.output_dir / "perplexity.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "perplexity", "loss", "num_tokens"])
                if teacher_perplexity:
                    writer.writerow(
                        [
                            teacher_perplexity.model_name,
                            teacher_perplexity.perplexity,
                            teacher_perplexity.loss,
                            teacher_perplexity.num_tokens,
                        ]
                    )
                if student_perplexity:
                    writer.writerow(
                        [
                            student_perplexity.model_name,
                            student_perplexity.perplexity,
                            student_perplexity.loss,
                            student_perplexity.num_tokens,
                        ]
                    )
            paths["perplexity.csv"] = path

        # Latency CSV
        if teacher_latency or student_latency:
            path = self.output_dir / "latency.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model",
                        "prompt_len",
                        "gen_len",
                        "batch_size",
                        "prefill_ms",
                        "decode_ms",
                        "total_ms",
                        "tokens_per_sec",
                        "ttft_ms",
                        "use_cache",
                    ]
                )
                for result in [teacher_latency, student_latency]:
                    if result:
                        for m in result.measurements:
                            writer.writerow(
                                [
                                    result.model_name,
                                    m.prompt_len,
                                    m.gen_len,
                                    m.batch_size,
                                    f"{m.prefill_time_ms:.2f}",
                                    f"{m.decode_time_ms:.2f}",
                                    f"{m.total_time_ms:.2f}",
                                    f"{m.tokens_per_second:.2f}",
                                    f"{m.time_to_first_token_ms:.2f}",
                                    getattr(m, "use_cache", False),
                                ]
                            )
            paths["latency.csv"] = path

        # Memory CSV
        if teacher_memory or student_memory:
            path = self.output_dir / "memory.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model",
                        "seq_len",
                        "batch_size",
                        "quantization",
                        "peak_mb",
                        "kvcache_mb",
                        "model_mb",
                    ]
                )
                for result in [teacher_memory, student_memory]:
                    if result:
                        for m in result.measurements:
                            writer.writerow(
                                [
                                    result.model_name,
                                    m.seq_len,
                                    m.batch_size,
                                    m.quantization,
                                    f"{m.peak_memory_mb:.2f}",
                                    f"{m.kvcache_memory_mb:.2f}",
                                    f"{m.model_memory_mb:.2f}",
                                ]
                            )
            paths["memory.csv"] = path

        return paths

    def _generate_charts(
        self,
        summary: ComparisonSummary,
        teacher_latency: LatencyResult | None,
        student_latency: LatencyResult | None,
        teacher_memory: MemoryResult | None,
        student_memory: MemoryResult | None,
    ) -> dict[str, Path]:
        """Generate PNG charts for visual comparison."""
        paths: dict[str, Path] = {}

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return paths

        # Summary bar chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        models = ["Teacher", "Student (DBA)"]
        values = [summary.teacher_perplexity, summary.student_perplexity]
        colors = ["#3498db", "#e74c3c"]
        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel("Perplexity ↓")
        ax.set_title("Language Modeling Quality")
        ax.bar_label(bars, fmt="%.2f")

        ax = axes[1]
        values = [summary.teacher_tokens_per_sec, summary.student_tokens_per_sec]
        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel("Tokens/Second ↑")
        ax.set_title(f"Throughput ({summary.speedup:.2f}x speedup)")
        ax.bar_label(bars, fmt="%.0f")

        ax = axes[2]
        values = [
            summary.teacher_kvcache_mb_per_token,
            summary.student_kvcache_mb_per_token,
        ]
        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel("KV-Cache (MB/token) ↓")
        ax.set_title(f"Memory ({summary.memory_reduction:.1f}x reduction)")
        ax.bar_label(bars, fmt="%.6f")

        plt.tight_layout()
        path = self.output_dir / "summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths["summary.png"] = path

        # Latency vs context length chart
        if teacher_latency and student_latency:
            fig, ax = plt.subplots(figsize=(10, 6))

            all_measurements = (
                teacher_latency.measurements + student_latency.measurements
            )
            if all_measurements:
                batch_sizes = sorted(set(m.batch_size for m in all_measurements))
                gen_lens = sorted(set(m.gen_len for m in all_measurements))

                ref_batch = batch_sizes[0]
                ref_gen_len = gen_lens[len(gen_lens) // 2]

                t_data: dict[int, float] = {}
                s_data: dict[int, float] = {}

                for m in teacher_latency.measurements:
                    if m.batch_size == ref_batch and m.gen_len == ref_gen_len:
                        t_data[m.prompt_len] = m.tokens_per_second

                for m in student_latency.measurements:
                    if m.batch_size == ref_batch and m.gen_len == ref_gen_len:
                        s_data[m.prompt_len] = m.tokens_per_second

                if t_data and s_data:
                    x = sorted(t_data.keys())
                    t_y = [t_data.get(k, 0) for k in x]
                    s_y = [s_data.get(k, 0) for k in x]

                    ax.plot(
                        x, t_y, "o-", label="Teacher", color="#3498db", linewidth=2
                    )
                    ax.plot(
                        x,
                        s_y,
                        "s-",
                        label="Student (DBA)",
                        color="#e74c3c",
                        linewidth=2,
                    )
                    ax.set_xlabel("Prompt Length (tokens)")
                    ax.set_ylabel("Tokens/Second")
                    ax.set_title(
                        f"Throughput vs Context Length "
                        f"(gen_len={ref_gen_len}, batch={ref_batch})"
                    )
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    path = self.output_dir / "latency_vs_context.png"
                    plt.savefig(path, dpi=150, bbox_inches="tight")
                    paths["latency_vs_context.png"] = path

            plt.close()

        # Memory scaling chart
        if teacher_memory and student_memory:
            t_analysis = teacher_memory.kvcache_analysis
            s_analysis = student_memory.kvcache_analysis

            if t_analysis and s_analysis:
                fig, ax = plt.subplots(figsize=(10, 6))

                seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
                t_mem = [
                    t_analysis.bytes_per_token_fp16 * s / (1024 * 1024)
                    for s in seq_lens
                ]

                if s_analysis.bytes_per_token_dba_fp16:
                    s_mem = [
                        s_analysis.bytes_per_token_dba_fp16 * s / (1024 * 1024)
                        for s in seq_lens
                    ]
                else:
                    s_mem = [
                        s_analysis.bytes_per_token_fp16 * s / (1024 * 1024)
                        for s in seq_lens
                    ]

                ax.plot(
                    seq_lens, t_mem, "o-", label="Teacher", color="#3498db", linewidth=2
                )
                ax.plot(
                    seq_lens,
                    s_mem,
                    "s-",
                    label="Student (DBA)",
                    color="#e74c3c",
                    linewidth=2,
                )
                ax.set_xlabel("Sequence Length (tokens)")
                ax.set_ylabel("KV-Cache Memory (MB)")
                ax.set_title("Memory Scaling with Context Length")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale("log", base=2)

                path = self.output_dir / "memory_scaling.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                paths["memory_scaling.png"] = path

                plt.close()

        return paths

    def _write_latex_tables(
        self,
        metadata: ExperimentMetadata,
        summary: ComparisonSummary,
    ) -> Path:
        """Write LaTeX tables for direct paper inclusion."""
        path = self.output_dir / "tables.tex"

        latex = f"""% Auto-generated by Caramba on {datetime.now().isoformat()}
% Experiment: {metadata.name}

\\begin{{table}}[h]
\\centering
\\caption{{DBA Upcycle Results: {metadata.name}}}
\\label{{tab:dba-results}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Teacher}} & \\textbf{{Student (DBA)}} & \\textbf{{Change}} \\\\
\\midrule
Perplexity $\\downarrow$ & {summary.teacher_perplexity:.2f} & {summary.student_perplexity:.2f} & {summary.perplexity_ratio:.2f}$\\times$ \\\\
Throughput (tok/s) $\\uparrow$ & {summary.teacher_tokens_per_sec:.0f} & {summary.student_tokens_per_sec:.0f} & {summary.speedup:.2f}$\\times$ \\\\
KV-Cache (bytes/tok) $\\downarrow$ & {summary.teacher_kvcache_bytes_per_token:.0f} & {summary.student_kvcache_bytes_per_token:.0f} & {summary.memory_reduction:.1f}$\\times$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% Configuration
% Teacher: {metadata.teacher_checkpoint}
% Student: {metadata.student_config}
% Device: {metadata.device}
"""

        with open(path, "w") as f:
            f.write(latex)

        return path
