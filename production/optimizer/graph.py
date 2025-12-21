"""Dependency-driven optimizer graph used by `apply_dynamic_config`."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from production.optimizer.watch import OptimizerLike, Watch


class Optimizer:
    """Dependency-driven resolver for intent → concrete runtime config."""

    def __init__(self) -> None:
        self._values: dict[str, object] = {}
        self._watches: list[Watch] = []
        self._pumping: bool = False

    def has(self, key: str) -> bool:
        return str(key) in self._values

    def get(self, key: str) -> object:
        return self._values[str(key)]

    def maybe(self, key: str, default: object = None) -> object:
        return self._values.get(str(key), default)

    def set(self, key: str, value: object) -> None:
        self._values[str(key)] = value
        self._pump()

    def when_ready(
        self,
        deps: Sequence[str],
        fn: Callable[[OptimizerLike], None],
        *,
        name: str | None = None,
    ) -> None:
        watch_name = name or getattr(fn, "__name__", "watch")
        self._watches.append(Watch(tuple(str(d) for d in deps), fn, watch_name))
        self._pump()

    def _pump(self, *, max_iters: int = 512) -> None:
        if self._pumping:
            return
        self._pumping = True
        try:
            for _ in range(int(max_iters)):
                progressed = False
                for w in self._watches:
                    if w.fired:
                        continue
                    if all(d in self._values for d in w.deps):
                        w.fired = True
                        w.fn(self)
                        progressed = True
                if not progressed:
                    return
            raise RuntimeError("Optimizer graph did not converge (possible dependency cycle).")
        finally:
            self._pumping = False

    def apply_to_args(self, args: argparse.Namespace) -> None:
        for k, v in self._values.items():
            try:
                setattr(args, k, v)
            except (TypeError, AttributeError):
                pass

