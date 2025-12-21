from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Protocol, cast, runtime_checkable

from production.instrumentation.utils import (
    auto_wandb_mode,
    coerce_float,
    coerce_int,
    coerce_str,
    safe_vars,
)


class _WandbRun(Protocol):
    """
    Protocol for the W&B run.
    """
    url: str | None

    def log(self, data: dict[str, float], *, step: int | None = None) -> None: ...

    def finish(self) -> None: ...


@runtime_checkable
class _WandbModule(Protocol):
    """
    Protocol for the W&B module.
    """
    def init(
        self,
        *,
        project: str,
        entity: str | None,
        name: str | None,
        group: str | None,
        tags: list[str] | None,
        mode: str,
        config: dict[str, object],
        **kwargs: object,
    ) -> _WandbRun: ...


class WandBWriter:
    """
    Optional scalar logging via Weights & Biases (requires `wandb` package).
    """
    run: _WandbRun | None

    def __init__(
        self,
        out_dir: str,
        *,
        project: str,
        entity: str | None,
        name: str | None,
        group: str | None,
        tags: list[str] | None,
        mode: str,
        cfg: object,
        args: argparse.Namespace,
    ):
        """
        Initialize the W&B writer.
        """
        self.run = None
        try:
            import wandb  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            print(f"[warn] W&B not available; continuing without it: {e}", file=sys.stderr)
            self.run = None
            return

        mode = str(mode or "auto").lower()
        if mode not in ("auto", "disabled", "online", "offline"):
            mode = "auto"
        if mode in ("auto", "disabled"):
            mode = auto_wandb_mode()

        if name is None:
            name = os.path.basename(str(out_dir).rstrip("/")) or None

        cfg_dict: dict[str, object] = {"args": safe_vars(args)}
        if is_dataclass(cfg) and not isinstance(cfg, type):
            cfg_dict["config"] = cast(dict[str, object], asdict(cfg))
        else:
            raw = getattr(cfg, "__dict__", {})
            cfg_dict["config"] = cast(dict[str, object], raw) if isinstance(raw, dict) else {}

        try:
            if not isinstance(wandb, _WandbModule):
                raise RuntimeError("wandb module does not match expected API")
            wandb_mod = wandb
            self.run = wandb_mod.init(
                project=str(project),
                entity=(str(entity) if entity else None),
                name=(str(name) if name else None),
                group=(str(group) if group else None),
                tags=tags,
                dir=str(out_dir),
                mode=mode,
                config=cfg_dict,
            )
        except (OSError, RuntimeError, ValueError) as e:
            print(f"[warn] W&B init failed; continuing without it: {e}", file=sys.stderr)
            self.run = None
            return

        try:
            if self.run.url:
                print(f"[wandb] run: {self.run.url}", file=sys.stderr)
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass

    def maybe_log(self, event: dict[str, object]) -> None:
        """
        Log an event to W&B.
        """
        if self.run is None:
            return
        try:
            step = coerce_int(event.get("step", 0))
            etype = coerce_str(event.get("type", ""), default="")
            payload: dict[str, float] = {}

            if etype == "train":
                for k in (
                    "loss",
                    "ppl",
                    "lr",
                    "tok_s",
                    "seq_len",
                    "gbs",
                    "ms_step",
                    "ms_fwd",
                    "ms_bwd",
                    "ms_opt",
                ):
                    if k in event:
                        fv = coerce_float(event[k])
                        if fv is not None:
                            payload[k] = fv
                for k in (
                    "mps_mem_alloc_bytes",
                    "mps_mem_driver_bytes",
                    "cuda_mem_alloc_bytes",
                    "cuda_mem_reserved_bytes",
                ):
                    if k in event:
                        fv = coerce_float(event[k])
                        if fv is not None:
                            payload[k] = fv

            elif etype == "eval":
                for k in ("train_loss", "val_loss"):
                    if k in event:
                        fv = coerce_float(event[k])
                        if fv is not None:
                            payload[k] = fv

            elif etype == "analysis":
                for k, v in event.items():
                    if k in ("type", "step", "wall_time"):
                        continue
                    fv = coerce_float(v)
                    if fv is not None:
                        payload[f"analysis/{k}"] = fv

            if payload:
                self.run.log(payload, step=step)
        except (OSError, RuntimeError, ValueError, TypeError):
            pass

    def close(self) -> None:
        """
        Finish the W&B run.
        """
        if self.run is None:
            return
        try:
            self.run.finish()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
