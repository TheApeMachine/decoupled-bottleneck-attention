"""
experiment provides the unified experiment runner.

This module orchestrates the complete pipeline:
1. Model loading/creation
2. Upcycle (surgery + distillation)
3. Benchmarking (teacher vs student)
4. Artifact generation (CSV, charts, LaTeX)
"""
from __future__ import annotations

from caramba.experiment.runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
