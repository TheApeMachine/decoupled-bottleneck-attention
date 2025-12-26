"""Experiment groups: organizing runs and benchmarks.

A group bundles related training runs and benchmarks together, typically
for comparing different configurations on the same data. Groups share
a data path and can have multiple runs with different settings.
"""
from __future__ import annotations

from pydantic import BaseModel

from caramba.config.benchmark import BenchmarkSpec
from caramba.config.run import Run


class Group(BaseModel):
    """A collection of related training runs and benchmarks.

    Groups provide organization for experiments—all runs in a group
    share the same data and can be compared against each other.
    """

    name: str
    description: str
    data: str
    runs: list[Run]
    benchmarks: list[BenchmarkSpec] | None = None
