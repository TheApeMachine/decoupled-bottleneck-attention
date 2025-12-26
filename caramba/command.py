"""Typed CLI command payloads.

Each command type represents a distinct user intent. The CLI parses arguments
into these typed objects, which are then dispatched to the appropriate handler.
"""
from __future__ import annotations

from dataclasses import dataclass

from caramba.config.manifest import Manifest


@dataclass(frozen=True, slots=True)
class RunCommand:
    """Request to run the system from a manifest (legacy mode)."""

    manifest: Manifest


@dataclass(frozen=True, slots=True)
class CompileCommand:
    """Request to compile a manifest without building or running.

    Useful for validating configs before expensive training runs.
    """

    manifest: Manifest
    print_plan: bool


Command = RunCommand | CompileCommand
