"""
compiler provides lowering passes that turn configs into canonical forms.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from caramba.compiler.lower import Lowerer
from caramba.compiler.validate import Validator
from caramba.compiler.plan import Planner

if TYPE_CHECKING:
    from caramba.config.manifest import Manifest

__all__ = ["Compiler", "Lowerer", "Validator", "Planner"]


class Compiler:
    """
    Compiler provides a pipeline for compiling manifests.

    The compile pipeline consists of three stages:
    1. Lower: Expand repeats and transform configs to canonical form
    2. Validate: Check cross-layer shape invariants and constraints
    3. Plan: Generate the execution plan

    Usage:
        compiler = Compiler()
        manifest = compiler.compile(Manifest.from_path("path/to/manifest.yml"))
        # Or use components directly:
        # manifest = compiler.lowerer.lower_manifest(raw_manifest)
        # compiler.validator.validate_manifest(manifest)
        # plan = compiler.planner.format(manifest)
    """

    lowerer: Lowerer
    validator: Validator
    planner: Planner

    def __init__(self) -> None:
        self.lowerer = Lowerer()
        self.validator = Validator()
        self.planner = Planner()

    def compile(self, manifest: "Manifest") -> "Manifest":
        """
        Run the full compile pipeline: lower -> validate -> return.

        Args:
            manifest: The raw manifest to compile.

        Returns:
            The lowered and validated manifest.

        Raises:
            ValueError: If validation fails.
        """
        from caramba.config.manifest import Manifest

        lowered = self.lowerer.lower_manifest(manifest)
        self.validator.validate_manifest(lowered)
        return lowered