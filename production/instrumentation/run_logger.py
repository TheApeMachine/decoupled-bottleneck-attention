from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import torch

from production.instrumentation.analysis import generate_analysis_png
from production.instrumentation.live_plotter import LivePlotter
from production.instrumentation.tensorboard_writer import TensorBoardWriter
from production.instrumentation.utils import device_summary, env_info, now_iso, safe_vars
from production.instrumentation.wandb_writer import WandBWriter

class _TextWriter(Protocol):
    def write(self, s: str, /) -> int: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class _H5AttrManager(Protocol):
    def __setitem__(self, key: str, value: object) -> None: ...


class _H5Group(Protocol):
    attrs: _H5AttrManager

    def create_dataset(
        self,
        name: str,
        *,
        data: object,
        compression: str | None = None,
        compression_opts: int | None = None,
    ) -> object: ...


class _H5File(Protocol):
    def create_group(self, name: str) -> _H5Group: ...

    def require_group(self, name: str) -> _H5Group: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class _H5pyModule(Protocol):
    def File(self, filename: str, mode: str) -> _H5File: ...


_h5py_mod: _H5pyModule | None
try:
    import h5py as _h5py  # pyright: ignore[reportMissingImports]
except ImportError:
    _h5py_mod = None
else:
    if isinstance(_h5py, _H5pyModule):
        _h5py_mod = _h5py
    else:
        _h5py_mod = None


class RunLogger:
    """JSONL logger + optional HDF5, live plot, tensorboard.

    This is the modularized counterpart to v30's deep instrumentation. It keeps the same public
    surface (log(), h5_write_step(), finalize(), close()) so the runner can remain compatible.
    """

    out_dir: str
    instrument: str
    cfg: object
    args: argparse.Namespace
    device: torch.device
    start_time: float

    train_jsonl_path: str
    summary_path: str
    h5_path: str
    png_path: str

    _jsonl_f: _TextWriter | None
    _h5: _H5File | None
    _live: LivePlotter | None
    _tb: TensorBoardWriter | None
    _wandb: WandBWriter | None

    def __init__(
        self,
        out_dir: str,
        *,
        instrument: str,
        cfg: object,
        args: argparse.Namespace,
        device: torch.device,
        live_plot: bool = False,
        tb: bool = False,
        wandb: bool = False,
    ):
        self.out_dir = str(out_dir)
        self.instrument = str(instrument)
        self.cfg = cfg
        self.args = args
        self.device = device
        self.start_time = time.time()

        os.makedirs(self.out_dir, exist_ok=True)

        self.train_jsonl_path = os.path.join(self.out_dir, "train.jsonl")
        self.summary_path = os.path.join(self.out_dir, "summary.md")
        self.h5_path = os.path.join(self.out_dir, "analysis.h5")
        self.png_path = os.path.join(self.out_dir, "analysis.png")

        self._jsonl_f = None
        self._h5 = None
        self._live = None
        self._tb = None
        self._wandb = None

        if self.instrument != "off":
            resume_mode = bool(getattr(args, "resume", False) or getattr(args, "resume_path", None))
            mode = "a" if (resume_mode and os.path.exists(self.train_jsonl_path)) else "w"
            self._jsonl_f = open(self.train_jsonl_path, mode, encoding="utf-8")
            config_payload: dict[str, object]
            if is_dataclass(cfg) and not isinstance(cfg, type):
                config_payload = cast(dict[str, object], asdict(cfg))
            else:
                raw_cfg = getattr(cfg, "__dict__", {})
                config_payload = cast(dict[str, object], raw_cfg) if isinstance(raw_cfg, dict) else {}
            self.log(
                {
                    "type": ("resume_meta" if resume_mode else "meta"),
                    "step": 0,
                    "env": env_info(device),
                    "argv": list(sys.argv),
                    "args": safe_vars(args),
                    "config": config_payload,
                }
            )

        if self.instrument == "full" and _h5py_mod is not None:
            try:
                self._h5 = _h5py_mod.File(self.h5_path, "w")
                meta = self._h5.create_group("meta")
                meta.attrs["created"] = now_iso()
                meta.attrs["argv"] = " ".join(sys.argv)
            except (OSError, RuntimeError, ValueError, TypeError) as e:
                print(f"[warn] Could not open HDF5 at {self.h5_path}: {e}")
                self._h5 = None

        if live_plot:
            self._live = LivePlotter()
        if tb:
            self._tb = TensorBoardWriter(self.out_dir)
        if wandb:
            tags: list[str] | None = None
            args_dict = safe_vars(args)
            raw_tags = args_dict.get("wandb_tags")
            if raw_tags:
                tags = [t.strip() for t in str(raw_tags).split(",") if t.strip()]

            entity_obj = args_dict.get("wandb_entity")
            name_obj = args_dict.get("wandb_name")
            group_obj = args_dict.get("wandb_group")
            project_obj = args_dict.get("wandb_project")
            mode_obj = args_dict.get("wandb_mode")
            self._wandb = WandBWriter(
                self.out_dir,
                project=str(project_obj or "experiments"),
                entity=(str(entity_obj) if entity_obj is not None else None),
                name=(str(name_obj) if name_obj is not None else None),
                group=(str(group_obj) if group_obj is not None else None),
                tags=tags,
                mode=str(mode_obj or "disabled"),
                cfg=cfg,
                args=args,
            )

        self._write_summary(initial_only=True)

    def close(self) -> None:
        try:
            if self._h5 is not None:
                self._h5.flush()
                self._h5.close()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        try:
            if self._jsonl_f is not None:
                self._jsonl_f.flush()
                self._jsonl_f.close()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        try:
            if self._tb is not None:
                self._tb.close()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        try:
            if self._wandb is not None:
                self._wandb.close()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        try:
            if self._live is not None:
                self._live.close()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass

    def log(self, event: dict[str, object]) -> None:
        if self._jsonl_f is None and self._tb is None and self._wandb is None and self._live is None:
            return
        event = dict(event)
        if "wall_time" not in event:
            event["wall_time"] = now_iso()
        if self._jsonl_f is not None:
            _ = self._jsonl_f.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
            self._jsonl_f.flush()
        if self._tb is not None:
            self._tb.maybe_log(event)
        if self._wandb is not None:
            self._wandb.maybe_log(event)
        if self._live is not None:
            self._live.maybe_update(event)

    def h5_write_step(
        self,
        step: int,
        *,
        group: str,
        tensors: dict[str, torch.Tensor | None],
        attrs: dict[str, object] | None = None,
    ) -> None:
        if self._h5 is None:
            return
        try:
            g = self._h5.require_group(f"{group}/step_{int(step)}")
            if attrs:
                for k, v in attrs.items():
                    try:
                        g.attrs[k] = v
                    except (OSError, RuntimeError, ValueError, TypeError):
                        g.attrs[k] = str(v)
            for name, t in tensors.items():
                if t is None:
                    continue
                arr = t.detach().to("cpu")
                if arr.dtype in (torch.bfloat16, torch.float16):
                    arr = arr.to(torch.float16)
                else:
                    arr = arr.to(torch.float32)
                _ = g.create_dataset(name, data=arr.numpy(), compression="gzip", compression_opts=4)
            self._h5.flush()
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            print(f"[warn] HDF5 write failed at step {step}: {e}")

    def finalize(self, *, best_val: float, last_step: int) -> None:
        try:
            if self.instrument != "off":
                generate_analysis_png(self.train_jsonl_path, self.png_path)
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            print(f"[warn] analysis.png generation failed: {e}")
        self._write_summary(initial_only=False, best_val=best_val, last_step=last_step)

    def _write_summary(
        self,
        *,
        initial_only: bool,
        best_val: float | None = None,
        last_step: int | None = None,
    ) -> None:
        try:
            lines: list[str] = []
            lines.append("# Run Summary")
            lines.append("")
            lines.append(f"- Created: `{now_iso()}`")
            lines.append(f"- Out dir: `{self.out_dir}`")
            lines.append(f"- Device: `{device_summary(self.device)}`")
            lines.append(f"- Command: `{(' '.join(sys.argv))}`")
            lines.append("")
            lines.append("## Model Config")
            lines.append("")
            cfg_payload: dict[str, object]
            if is_dataclass(self.cfg) and not isinstance(self.cfg, type):
                cfg_payload = cast(dict[str, object], asdict(self.cfg))
            else:
                raw_cfg = getattr(self.cfg, "__dict__", {})
                cfg_payload = cast(dict[str, object], raw_cfg) if isinstance(raw_cfg, dict) else {}
            cfg_json = json.dumps(cfg_payload, indent=2, sort_keys=True, default=str)
            lines.append("```json")
            lines.append(cfg_json)
            lines.append("```")
            lines.append("")
            if not initial_only and best_val is not None and last_step is not None:
                ppl = math.exp(best_val) if best_val < 20 else float("inf")
                lines.append("## Results")
                lines.append("")
                lines.append(f"- Last step: `{last_step}`")
                lines.append(f"- Best val loss: `{best_val:.6f}` (ppl `{ppl:.2f}`)")
                lines.append("")
            _ = Path(self.summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            print(f"[warn] Failed to write summary.md: {e}")
