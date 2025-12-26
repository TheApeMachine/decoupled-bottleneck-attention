"""Weights & Biases logging integration.

W&B is optional. This module provides a best-effort wrapper that:
- Initializes a run (when `wandb` is installed and enabled)
- Logs scalar metrics with a consistent key namespace
- Never crashes training if W&B is unavailable or misconfigured
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from caramba.instrumentation.utils import coerce_jsonable


@runtime_checkable
class _WandbRun(Protocol):
    url: str | None

    def log(self, data: dict[str, float], *, step: int | None = None) -> None: ...

    def finish(self) -> None: ...


@runtime_checkable
class _WandbModule(Protocol):
    def init(
        self,
        *,
        project: str,
        entity: str | None,
        name: str | None,
        group: str | None,
        mode: str,
        dir: str | None = None,
        config: dict[str, object] | None = None,
        **kwargs: object,
    ) -> _WandbRun: ...


def _try_import_wandb() -> _WandbModule | None:
    try:
        if importlib.util.find_spec("wandb") is None:
            return None
        mod = importlib.import_module("wandb")
    except (ImportError, ModuleNotFoundError):
        return None
    return mod if isinstance(mod, _WandbModule) else None


def _auto_wandb_mode() -> str:
    """Choose a sensible W&B mode based on environment.

    This only applies when the manifest explicitly sets wandb_mode: auto.
    """

    # If a user has a key, default to online; otherwise offline avoids prompts/network.
    if str(os.environ.get("WANDB_API_KEY", "")).strip():
        return "online"
    # Common CI env var patterns: default to offline.
    if str(os.environ.get("CI", "")).strip():
        return "offline"
    return "offline"


def _serialize_cfg(obj: object) -> dict[str, object]:
    """Best-effort config serialization for W&B init config."""

    if obj is None:
        return {}

    # Pydantic v2 BaseModel has model_dump.
    try:
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            out = dump()
            if isinstance(out, dict):
                return coerce_jsonable(out)
    except Exception:
        pass

    # Dataclasses.
    try:
        import dataclasses

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return coerce_jsonable(dataclasses.asdict(obj))
    except Exception:
        pass

    # Plain dict.
    if isinstance(obj, dict):
        return coerce_jsonable({str(k): v for k, v in obj.items()})

    # Fallback to __dict__.
    try:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            return coerce_jsonable({str(k): v for k, v in d.items()})
    except Exception:
        pass

    return {"repr": repr(obj)}


@dataclass
class WandBWriter:
    """Best-effort W&B writer for scalar metrics."""

    out_dir: Path
    enabled: bool = True
    project: str = ""
    entity: str | None = None
    mode: str = "online"
    run_name: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    config: object | None = None
    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.run: _WandbRun | None = None
        self._consecutive_failures: int = 0

        if not self.enabled:
            return

        wandb = _try_import_wandb()
        if wandb is None:
            self.enabled = False
            return

        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we can't create output dir, still allow wandb (dir is optional).
            pass

        try:
            name = self.run_name
            if name is None:
                name = self.out_dir.name or None
            mode = str(self.mode or "online").strip().lower()
            if mode in ("disabled", "off", "false", "0", "none"):
                self.enabled = False
                self.run = None
                return
            if mode == "auto":
                mode = _auto_wandb_mode()
            cfg = _serialize_cfg(self.config)
            self.run = wandb.init(
                project=str(self.project or ""),
                entity=(str(self.entity) if self.entity else None),
                name=(str(name) if name else None),
                group=(str(self.group) if self.group else None),
                mode=str(mode),
                dir=str(self.out_dir),
                config=cfg,
                tags=self.tags,
            )
        except Exception:
            self.enabled = False
            self.run = None
            return

        # Log run URL for convenience (best-effort).
        try:
            if self.run is not None and getattr(self.run, "url", None):
                # Avoid importing rich logger here; stderr is fine for a one-liner.
                print(f"[wandb] run: {self.run.url}", file=sys.stderr)
        except Exception:
            pass

    def log_scalars(self, *, prefix: str, step: int, scalars: dict[str, float]) -> None:
        """Log scalar metrics to W&B with a prefix namespace."""

        if not self.enabled or self.run is None:
            return
        try:
            payload = {f"{prefix}/{k}": float(v) for k, v in scalars.items()}
            self.run.log(payload, step=int(step))
            # Reset failure counter on success.
            self._consecutive_failures = 0
        except Exception as e:
            self._consecutive_failures += 1
            # Log the failure details (best-effort, to stderr).
            try:
                print(f"[wandb] log failure ({self._consecutive_failures}/{self.max_consecutive_failures}): {e}", file=sys.stderr)
            except Exception:
                pass
            # Only disable after exceeding the threshold.
            if self._consecutive_failures >= self.max_consecutive_failures:
                self.enabled = False

    def close(self) -> None:
        """Finish the W&B run."""

        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass
        self.run = None

