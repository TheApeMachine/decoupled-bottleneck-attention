"""Compiler: lowering passes that turn configs into canonical forms.

Before training, user-facing configs may contain shortcuts (like `repeat: 16`
to define 16 identical layers). The compiler expands these into explicit forms
that the model builder can directly consume.

Pipeline stages:
1. Lower: Expand repeats, apply defaults, normalize structure
2. Validate: Check shape invariants, attention constraints, IO compatibility
3. Plan (optional): Generate human-readable execution plans for debugging
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from caramba.compiler.lower import Lowerer
from caramba.compiler.plan import Planner
from caramba.compiler.validate import Validator

if TYPE_CHECKING:
    from caramba.config.manifest import Manifest

__all__ = ["Compiler", "Lowerer", "Validator", "Planner"]


class Compiler:
    """Runs the full compilation pipeline on manifests.

    Converts user-friendly configs into validated, canonical forms ready
    for model construction.
    """

    lowerer: Lowerer
    validator: Validator
    planner: Planner

    def __init__(self) -> None:
        """Initialize compiler with default passes."""
        self.lowerer = Lowerer()
        self.validator = Validator()
        self.planner = Planner()

    def compile(self, manifest: "Manifest") -> "Manifest":
        """Run the full pipeline: lower → validate → return.

        Args:
            manifest: The raw manifest to compile.

        Returns:
            The lowered and validated manifest.

        Raises:
            ValueError: If validation fails.
        """
        lowered = self.lowerer.lower_manifest(manifest)
        self.validator.validate_manifest(lowered)
        return lowered
