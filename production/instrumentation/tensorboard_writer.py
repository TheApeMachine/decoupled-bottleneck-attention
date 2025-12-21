from __future__ import annotations

import os
from typing import Protocol, cast

from production.instrumentation.utils import coerce_float, coerce_int, coerce_str


class _TBWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class TensorBoardWriter:
    """Optional scalar logging via TensorBoard (requires `tensorboard` package)."""

    writer: _TBWriter | None

    def __init__(self, out_dir: str):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except ImportError as e:
            print(f"[warn] TensorBoard not available: {e}. Disable with --tb=0 or install tensorboard.")
            self.writer = None
            return

        try:
            self.writer = cast(_TBWriter, SummaryWriter(log_dir=os.path.join(str(out_dir), "tb")))
        except (OSError, RuntimeError, ValueError) as e:
            print(f"[warn] TensorBoard init failed: {e}. Disable with --tb=0 or install tensorboard.")
            self.writer = None

    def maybe_log(self, event: dict[str, object]) -> None:
        if self.writer is None:
            return
        try:
            step = coerce_int(event.get("step", 0))
            etype = coerce_str(event.get("type", ""), default="")
            if etype == "train":
                if "loss" in event:
                    v = coerce_float(event["loss"])
                    if v is not None:
                        self.writer.add_scalar("loss/train", v, step)
                if "ppl" in event:
                    v = coerce_float(event["ppl"])
                    if v is not None:
                        self.writer.add_scalar("ppl/train", v, step)
                if "tok_s" in event:
                    v = coerce_float(event["tok_s"])
                    if v is not None:
                        self.writer.add_scalar("perf/tok_s", v, step)
            if etype == "eval":
                if "train_loss" in event:
                    v = coerce_float(event["train_loss"])
                    if v is not None:
                        self.writer.add_scalar("loss/train_eval", v, step)
                if "val_loss" in event:
                    v = coerce_float(event["val_loss"])
                    if v is not None:
                        self.writer.add_scalar("loss/val", v, step)
            if etype == "analysis":
                for k, v in event.items():
                    if k in ("type", "step", "wall_time"):
                        continue
                    fv = coerce_float(v)
                    if fv is not None:
                        self.writer.add_scalar(f"analysis/{k}", fv, step)
        except (OSError, RuntimeError, ValueError, TypeError):
            pass

    def close(self) -> None:
        if self.writer is None:
            return
        try:
            self.writer.flush()
            self.writer.close()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
