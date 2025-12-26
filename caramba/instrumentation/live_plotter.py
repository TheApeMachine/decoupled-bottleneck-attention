"""Optional live plotting during training.

Live plotting is useful during development to quickly see if training is
converging. This module is intentionally optional:
- If matplotlib isn't installed, it disables itself.
- Any plotting errors disable plotting rather than breaking training.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any


def _try_import_pyplot() -> Any | None:
    try:
        if importlib.util.find_spec("matplotlib") is None:
            return None
        return importlib.import_module("matplotlib.pyplot")
    except (ImportError, ModuleNotFoundError):
        return None


@dataclass
class LivePlotter:
    """Best-effort matplotlib live plotter for scalar time series."""

    enabled: bool = False
    title: str = "caramba training"
    plot_every: int = 10

    _plt: Any | None = field(init=False, default=None)
    _fig: Any | None = field(init=False, default=None)
    _ax: Any | None = field(init=False, default=None)
    _series: dict[str, list[float]] = field(init=False, default_factory=dict)
    _steps: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        plt = _try_import_pyplot()
        if plt is None:
            self.enabled = False
            return
        try:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.set_title(str(self.title))
            self._plt = plt
            self._fig = fig
            self._ax = ax
        except Exception:
            self.enabled = False
            self._plt = None
            self._fig = None
            self._ax = None

    def update(self, *, step: int, scalars: dict[str, float]) -> None:
        """Update plot with a set of scalar values at a given step."""

        if (
            not self.enabled
            or self._plt is None
            or self._ax is None
            or self._fig is None
        ):
            return

        try:
            ax = self._ax
            fig = self._fig
            step_i = int(step)
            self._steps.append(step_i)
            for k, v in scalars.items():
                self._series.setdefault(str(k), []).append(float(v))

            pe = int(self.plot_every)
            if pe > 1 and (step_i % pe) != 0:
                return

            ax.clear()
            ax.set_title(str(self.title))
            for name, ys in self._series.items():
                # Invariant: every update must append to all series and _steps together.
                # If series lengths diverge, align by taking the last len(ys) steps.
                if len(ys) > len(self._steps):
                    # Should not happen; truncate series to match steps.
                    ys = ys[-len(self._steps):]
                xs = self._steps[-len(ys):]
                ax.plot(xs, ys, label=name)
            try:
                ax.legend()
            except Exception:
                pass
            ax.relim()
            ax.autoscale_view()
            canvas = getattr(fig, "canvas", None)
            if canvas is not None:
                draw = getattr(canvas, "draw", None)
                flush = getattr(canvas, "flush_events", None)
                if callable(draw):
                    draw()
                if callable(flush):
                    flush()
        except Exception:
            # Disable on any plotting failure.
            self.enabled = False

    def close(self) -> None:
        if not self.enabled or self._plt is None:
            return
        try:
            self._plt.ioff()
        except Exception:
            pass
        try:
            if self._fig is not None:
                self._plt.close(self._fig)
        except Exception:
            pass
        self._plt = None
        self._fig = None
        self._ax = None

