"""
runner provides the unified experiment orchestration.
"""
from __future__ import annotations

import logging
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
from caramba.model import Model
from caramba.trainer.upcycle import Upcycle

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Unified experiment runner that orchestrates the complete pipeline.

    Usage:
        manifest = Manifest.from_path("llama32_1b_dba.yml")
        runner = ExperimentRunner(manifest)
        runner.run()  # Runs everything: upcycle + benchmarks + artifacts
    """

    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        self.teacher: nn.Module | None = None
        self.student: nn.Module | None = None

    def run(self, group_name: str | None = None) -> dict[str, Path]:
        """
        Run the complete experiment pipeline.

        Args:
            group_name: Optional group name to run. If None, runs the first group.

        Returns:
            Dict of generated artifact paths.
        """
        # Find the group to run
        group = self._find_group(group_name)
        logger.info("experiment: running group '%s'", group.name)
        logger.info("  %s", group.description)

        # Get train config from first run
        train_config = self._get_train_config(group)

        # Run upcycle training
        upcycle = Upcycle(self.manifest, group, train_config)
        for run in group.runs:
            logger.info(
                "experiment: run '%s' (%s)",
                run.id,
                run.train.phase if run.train else "no train",
            )
            upcycle.run(run)

        # Store references to trained models
        self.teacher = upcycle.teacher
        self.student = upcycle.student

        # Run benchmarks if configured
        artifacts: dict[str, Path] = {}
        if group.benchmarks:
            logger.info("experiment: running %d benchmarks", len(group.benchmarks))
            artifacts = self._run_benchmarks(group, upcycle)

        logger.info("experiment: complete, generated %d artifacts", len(artifacts))
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
        if output_formats is None:
            output_formats = default_formats
        elif not isinstance(output_formats, list) or not all(isinstance(f, str) for f in output_formats):
            logger.warning("Invalid output_formats in manifest, using defaults")
            output_formats = default_formats

        # Build benchmark suite
        suite = BenchmarkSuite(
            benchmarks=group.benchmarks,
            output_dir=f"artifacts/{self.manifest.name or 'experiment'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            formats=output_formats,
        )

        # Build metadata - safely retrieve topology type
        train_config = self._get_train_config(group)
        model = getattr(self.manifest, "model", None)
        topology = getattr(model, "topology", None) if model is not None else None
        topology_type = str(getattr(topology, "type", "")) if topology is not None else ""

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


def run_experiment(manifest_path: str | Path, group: str | None = None) -> dict[str, Path]:
    """
    Convenience function to run an experiment from a manifest path.

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
