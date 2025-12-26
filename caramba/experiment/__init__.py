"""Unified experiment runner.

Orchestrates the complete experiment pipeline:
1. Model loading/creation from manifest
2. Upcycle (attention surgery + blockwise distillation)
3. Benchmarking (teacher vs student comparison)
4. Artifact generation (CSV, charts, LaTeX tables)

This is the main entry point for running experiments end-to-end.
"""
from __future__ import annotations

from caramba.experiment.runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
