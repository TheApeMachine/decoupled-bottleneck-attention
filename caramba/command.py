"""
command provides typed CLI command payloads.
"""
from __future__ import annotations

from dataclasses import dataclass

from caramba.config.manifest import Manifest


@dataclass(frozen=True, slots=True)
class RunCommand:
    """
    RunCommand represents a request to run the system from a manifest.
    """

    manifest: Manifest


@dataclass(frozen=True, slots=True)
class CompileCommand:
    """
    CompileCommand represents a request to compile a manifest without building.
    """

    manifest: Manifest
    print_plan: bool


Command = RunCommand | CompileCommand

