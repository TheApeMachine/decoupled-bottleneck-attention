"""
compiler provides lowering passes that turn configs into canonical forms.
"""
from __future__ import annotations

from caramba.compiler.lower import Lowerer
from caramba.compiler.validate import Validator
from caramba.compiler.plan import Planner

__all__ = ["Compiler", "Lowerer", "Validator", "Planner"]


class Compiler:
    """Compiler provides a pipeline for compiling manifests."""

    lowerer: Lowerer
    validator: Validator
    planner: Planner

    def __init__(self) -> None:
        self.lowerer = Lowerer()
        self.validator = Validator()
        self.planner = Planner()