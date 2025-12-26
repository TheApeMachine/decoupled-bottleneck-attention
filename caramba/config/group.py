"""
group provides the group module for the training loop.
"""
from __future__ import annotations

from pydantic import BaseModel
from caramba.config.run import Run
from caramba.config.benchmark import BenchmarkSpec


class Group(BaseModel):
    """
    Group provides the group module for the training loop.
    """
    name: str
    description: str
    data: str
    runs: list[Run]
    benchmarks: list[BenchmarkSpec] | None = None