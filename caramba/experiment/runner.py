"""
runner provides the unified experiment orchestration.
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
from caramba.model import Model
from caramba.trainer.upcycle import Upcycle


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
        print(f"experiment: running group '{group.name}'")
        print(f"  {group.description}")

        # Get train config from first run
        train_config = self._get_train_config(group)

        # Run upcycle training
        upcycle = Upcycle(self.manifest, group, train_config)
        for run in group.runs:
            print(f"experiment: run '{run.id}' ({run.train.phase if run.train else 'no train'})")
            upcycle.run(run)

        # Store references to trained models
        self.teacher = upcycle.teacher
        self.student = upcycle.student

        # Run benchmarks if configured
        artifacts: dict[str, Path] = {}
        if group.benchmarks:
            print(f"experiment: running {len(group.benchmarks)} benchmarks")
            artifacts = self._run_benchmarks(group, upcycle)

        print(f"experiment: complete, generated {len(artifacts)} artifacts")
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

        # Build benchmark suite
        suite = BenchmarkSuite(
            benchmarks=group.benchmarks,
            output_dir=f"artifacts/{self.manifest.name or 'experiment'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            formats=["csv", "json", "png", "latex"],
        )

        # Build metadata
        train_config = self._get_train_config(group)
        metadata = ExperimentMetadata(
            name=self.manifest.name or "experiment",
            timestamp=datetime.now().isoformat(),
            manifest_path=str(self.manifest.name) if self.manifest.name else "",
            teacher_checkpoint=train_config.teacher_ckpt or "",
            student_config=str(self.manifest.model.topology.type if hasattr(self.manifest.model, 'topology') else ""),
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
