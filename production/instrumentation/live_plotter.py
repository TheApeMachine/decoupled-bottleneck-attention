from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from production.instrumentation.utils import coerce_float, coerce_int, coerce_str


class _Line(Protocol):
    def set_data(self, x: Sequence[float | int], y: Sequence[float | int]) -> None: ...


class _Axes(Protocol):
    def plot(
        self, x: Sequence[float | int], y: Sequence[float | int], *, label: str
    ) -> list[_Line]: ...

    def legend(self) -> None: ...

    def set_title(self, title: str) -> None: ...

    def relim(self) -> None: ...

    def autoscale_view(self) -> None: ...


class _Canvas(Protocol):
    def draw(self) -> None: ...

    def flush_events(self) -> None: ...


class _Figure(Protocol):
    canvas: _Canvas


@runtime_checkable
class _Pyplot(Protocol):
    def ion(self) -> None: ...

    def ioff(self) -> None: ...

    def close(self, fig: _Figure) -> None: ...

    def subplots(
        self, nrows: int, ncols: int, *, figsize: tuple[float, float]
    ) -> tuple[_Figure, Sequence[_Axes]]: ...


class LivePlotter:
    """
    Dev-only matplotlib plots. Safe to keep disabled by default.
    """

    enabled: bool
    plt: _Pyplot | None
    fig: _Figure | None
    ax: list[_Axes]
    steps: list[int]
    train_loss: list[float]
    val_loss: list[float]
    l1: _Line | None
    l2: _Line | None

    def __init__(self):
        self.enabled = False
        self.plt = None
        self.fig = None
        self.ax = []
        self.steps = []
        self.train_loss = []
        self.val_loss = []
        self.l1 = None
        self.l2 = None
        try:
            import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

            if not isinstance(plt, _Pyplot):
                raise RuntimeError("matplotlib.pyplot does not match expected API")
            self.plt = plt
            self.plt.ion()

            fig, ax_seq = self.plt.subplots(1, 2, figsize=(10.0, 4.0))
            self.fig = fig
            self.ax = list(ax_seq)

            l1s = self.ax[0].plot([], [], label="train")
            l2s = self.ax[0].plot([], [], label="val")
            self.l1 = l1s[0] if l1s else None
            self.l2 = l2s[0] if l2s else None
            self.ax[0].legend()
            self.ax[0].set_title("Loss")
            self.enabled = True
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            print(f"[warn] Live plot disabled: {e}")
            self.enabled = False

    def maybe_update(self, event: dict[str, object]) -> None:
        if not self.enabled:
            return
        if self.fig is None or self.plt is None or self.l1 is None:
            return
        try:
            et = coerce_str(event.get("type", ""), default="")
            step = coerce_int(event.get("step", 0))
            if et == "train" and "loss" in event:
                loss = coerce_float(event["loss"])
                if loss is not None:
                    self.steps.append(step)
                    self.train_loss.append(loss)
            if et == "eval" and "val_loss" in event:
                vloss = coerce_float(event["val_loss"])
                if vloss is not None:
                    self.val_loss.append(vloss)

            self.l1.set_data(self.steps, self.train_loss)
            if self.val_loss and self.l2 is not None:
                xs = (
                    self.steps[-len(self.val_loss) :]
                    if len(self.val_loss) <= len(self.steps)
                    else list(range(len(self.val_loss)))
                )
                self.l2.set_data(xs, self.val_loss)
            self.ax[0].relim()
            self.ax[0].autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass

    def close(self) -> None:
        if not self.enabled:
            return
        if self.plt is None or self.fig is None:
            return
        try:
            self.plt.ioff()
            self.plt.close(self.fig)
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
