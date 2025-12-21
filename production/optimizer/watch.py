"""Internal watcher record for the dependency graph optimizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class OptimizerLike(Protocol):
    """Minimal Optimizer interface needed by watch callbacks."""

    def has(self, key: str) -> bool: ...
    def get(self, key: str) -> object: ...
    def maybe(self, key: str, default: object = ...) -> object: ...
    def set(self, key: str, value: object) -> None: ...

@dataclass
class Watch:
    """A deferred callback that fires once all dependencies are present."""

    deps: tuple[str, ...]
    fn: Callable[[OptimizerLike], None]
    name: str
    fired: bool = False

