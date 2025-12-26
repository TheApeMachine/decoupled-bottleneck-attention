"""Experiment orchestration.

The ExperimentRunner coordinates all phases of an experiment:
1. Parse and validate the manifest
2. Run training (upcycling with distillation)
3. Run benchmarks comparing teacher and student
4. Generate artifacts for analysis and publication
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from caramba.benchmark.artifacts import ExperimentMetadata
from caramba.benchmark.runner import BenchmarkRunner
from caramba.config.benchmark import BenchmarkSuite
from caramba.config.group import Group
from caramba.config.manifest import Manifest
from caramba.config.train import TrainConfig
from caramba.console import logger
from caramba.trainer.upcycle import Upcycle


class ExperimentRunner:
    """Unified experiment runner for the complete pipeline.

    Takes a manifest and runs all configured groups through upcycling,
    benchmarking, and artifact generation.

    Usage:
        manifest = Manifest.from_path("llama32_1b_dba.yml")
        runner = ExperimentRunner(manifest)
        artifacts = runner.run()  # Returns paths to generated artifacts
    """

    def __init__(self, manifest: Manifest) -> None:
        """Initialize with a validated manifest."""
        self.manifest = manifest
        self.teacher: nn.Module | None = None
        self.student: nn.Module | None = None

    def run(self, group_name: str | None = None) -> dict[str, Path]:
        """Run the complete experiment pipeline.

        Args:
            group_name: Optional group name to run. If None, runs the first group.

        Returns:
            Dict mapping artifact names to their file paths.
        """
        group = self._find_group(group_name)

        logger.header("Experiment", group.name)
        if group.description:
            logger.info(group.description)
        logger.key_value(
            {
                "Runs": len(group.runs),
                "Benchmarks": len(group.benchmarks) if group.benchmarks else 0,
                "Data": group.data,
            }
        )

        # Get train config from first run
        train_config = self._get_train_config(group)

        # Run upcycle training
        upcycle = Upcycle(self.manifest, group, train_config)

        for i, run in enumerate(group.runs):
            phase_name = run.train.phase.value if run.train else "unknown"
            logger.step(i + 1, len(group.runs), f"Run '{run.id}' ({phase_name})")
            upcycle.run(run)

        # Store references to trained models
        self.teacher = upcycle.teacher
        self.student = upcycle.student

        # Run benchmarks if configured
        artifacts: dict[str, Path] = {}
        if group.benchmarks:
            artifacts = self._run_benchmarks(group, upcycle)

        logger.success(f"Experiment complete • {len(artifacts)} artifacts generated")
        return artifacts

    def _find_group(self, group_name: str | None) -> Group:
        """Find group by name or return first group."""
        if not self.manifest.groups:
            raise ValueError("Manifest has no groups defined")

        if group_name is None:
            return self.manifest.groups[0]

        for group in self.manifest.groups:
            if group.name == group_name:
                return group

        raise ValueError(f"Group '{group_name}' not found in manifest")

    def _get_train_config(self, group: Group) -> TrainConfig:
        """Get train config from the first run with training."""
        for run in group.runs:
            if run.train:
                return run.train
        raise ValueError(f"Group '{group.name}' has no runs with train config")

    def _run_benchmarks(
        self,
        group: Group,
        upcycle: Upcycle,
    ) -> dict[str, Path]:
        """Run benchmarks and generate artifacts."""
        if not group.benchmarks:
            return {}

        # Get output formats from manifest, with default fallback
        default_formats = ["csv", "json", "png", "latex"]
        output_formats = getattr(self.manifest, "output_formats", None)
        if (
            output_formats is None
            or not isinstance(output_formats, list)
            or not output_formats
            or not all(isinstance(f, str) for f in output_formats)
        ):
            if output_formats is not None:
                logger.warning(
                    "Invalid or empty output_formats in manifest, using defaults"
                )
            output_formats = list(default_formats)

        # Build benchmark suite
        suite = BenchmarkSuite(
            benchmarks=group.benchmarks,
            output_dir=f"artifacts/{self.manifest.name or 'experiment'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            formats=output_formats,
        )

        # Build metadata
        train_config = self._get_train_config(group)
        model = getattr(self.manifest, "model", None)
        topology = getattr(model, "topology", None) if model is not None else None
        topology_type = (
            str(getattr(topology, "type", "")) if topology is not None else ""
        )

        metadata = ExperimentMetadata(
            name=self.manifest.name or "experiment",
            timestamp=datetime.now().isoformat(),
            manifest_path=str(self.manifest.name) if self.manifest.name else "",
            teacher_checkpoint=train_config.teacher_ckpt or "",
            student_config=topology_type,
            device=train_config.device,
            notes=self.manifest.notes,
        )

        # Run benchmarks
        runner = BenchmarkRunner(suite, upcycle.device, metadata)
        return runner.run(upcycle.teacher, upcycle.student)


def run_experiment(
    manifest_path: str | Path, group: str | None = None
) -> dict[str, Path]:
    """Convenience function to run an experiment from a manifest path.

    Loads, compiles, and runs the manifest in one call.

    Args:
        manifest_path: Path to manifest YAML/JSON file.
        group: Optional group name to run.

    Returns:
        Dict of generated artifact paths.
    """
    from caramba.compiler import Compiler

    path = Path(manifest_path)
    manifest = Manifest.from_path(path)

    # Lower and validate
    compiler = Compiler()
    manifest = compiler.lowerer.lower_manifest(manifest)
    compiler.validator.validate_manifest(manifest)

    # Run
    runner = ExperimentRunner(manifest)
    return runner.run(group)
