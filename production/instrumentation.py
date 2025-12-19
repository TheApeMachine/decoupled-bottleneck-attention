from __future__ import annotations

import argparse
import datetime
import math
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    import h5py as _h5py  # type: ignore
except Exception:
    _h5py = None  # type: ignore


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def _device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = "cuda"
        return f"cuda:{device.index or 0} ({name})"
    if device.type == "mps":
        return "mps"
    return str(device)


def _env_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "unknown"),
        "device": _device_summary(device),
    }
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


class TensorBoardWriter:
    """Optional scalar logging via TensorBoard (requires `tensorboard` package)."""

    def __init__(self, out_dir: str):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self.writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        except Exception as e:
            print(f"[warn] TensorBoard not available: {e}. Disable with --tb=0 or install tensorboard.")
            self.writer = None

    def maybe_log(self, event: Dict[str, Any]) -> None:
        if self.writer is None:
            return
        try:
            step = int(event.get("step", 0))
            etype = str(event.get("type", ""))
            if etype == "train":
                if "loss" in event:
                    self.writer.add_scalar("loss/train", float(event["loss"]), step)
                if "ppl" in event:
                    self.writer.add_scalar("ppl/train", float(event["ppl"]), step)
                if "tok_s" in event:
                    self.writer.add_scalar("perf/tok_s", float(event["tok_s"]), step)
            if etype == "eval":
                if "train_loss" in event:
                    self.writer.add_scalar("loss/train_eval", float(event["train_loss"]), step)
                if "val_loss" in event:
                    self.writer.add_scalar("loss/val", float(event["val_loss"]), step)
            if etype == "analysis":
                for k, v in event.items():
                    if k in ("type", "step", "wall_time"):
                        continue
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"analysis/{k}", float(v), step)
        except Exception:
            pass

    def close(self) -> None:
        if self.writer is None:
            return
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass


class WandBWriter:
    """Optional scalar logging via Weights & Biases (requires `wandb` package)."""

    def __init__(self, out_dir: str, *, project: str, entity: Optional[str], name: Optional[str], group: Optional[str], tags: Optional[List[str]], mode: str, cfg: Any, args: argparse.Namespace):
        self.run = None
        try:
            import wandb  # type: ignore

            # Make `--wandb` behave like `--tb`: if you enable it, it actually logs.
            # The CLI defaults wandb-mode to "disabled" to avoid surprises, but when --wandb is set
            # we treat "disabled" as "online" (unless the user explicitly asked for offline).
            mode = str(mode or "disabled").lower()
            if mode not in ("disabled", "online", "offline"):
                mode = "disabled"
            if mode == "disabled":
                mode = "online"

            if name is None:
                try:
                    name = os.path.basename(str(out_dir).rstrip("/"))
                except Exception:
                    name = None

            # Config: merge CLI args + model cfg (wandb will flatten/pretty-print).
            cfg_dict: Dict[str, Any] = {}
            try:
                cfg_dict["args"] = vars(args)
            except Exception:
                pass
            try:
                cfg_dict["config"] = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else getattr(cfg, "__dict__", {})
            except Exception:
                pass

            self.run = wandb.init(
                project=str(project),
                entity=(str(entity) if entity else None),
                name=(str(name) if name else None),
                group=(str(group) if group else None),
                tags=tags,
                dir=str(out_dir),
                mode=mode,
                config=cfg_dict,
            )
            # One-time breadcrumb so it's obvious where the run went.
            try:
                url = getattr(self.run, "url", None)
                if url:
                    print(f"[wandb] run: {url}", file=sys.stderr)
            except Exception:
                pass
        except Exception as e:
            # If the user asked for W&B, fail loudly when the package is missing; otherwise we'd
            # silently log only locally and it looks like "nothing happens".
            if isinstance(e, ModuleNotFoundError) and "wandb" in str(e):
                raise ImportError(
                    "W&B logging requested (--wandb) but the `wandb` package is not installed. "
                    "Install it with `pip install wandb` (or `pip install -r requirements.txt`)."
                ) from e
            print(f"[warn] W&B not available: {e}. Disable with --wandb=0 or install wandb.", file=sys.stderr)
            self.run = None

    def maybe_log(self, event: Dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            step = int(event.get("step", 0))
            etype = str(event.get("type", ""))
            payload: Dict[str, Any] = {}

            if etype == "train":
                for k in ("loss", "ppl", "lr", "tok_s", "seq_len", "gbs", "ms_step", "ms_fwd", "ms_bwd", "ms_opt"):
                    if k in event:
                        payload[k] = float(event[k]) if isinstance(event[k], (int, float)) else event[k]
                # device mem (if present)
                for k in ("mps_mem_alloc_bytes", "mps_mem_driver_bytes", "cuda_mem_alloc_bytes", "cuda_mem_reserved_bytes"):
                    if k in event:
                        payload[k] = float(event[k])

            elif etype == "eval":
                for k in ("train_loss", "val_loss"):
                    if k in event:
                        payload[k] = float(event[k])

            elif etype == "analysis":
                for k, v in event.items():
                    if k in ("type", "step", "wall_time"):
                        continue
                    if isinstance(v, (int, float)):
                        payload[f"analysis/{k}"] = float(v)

            if payload:
                # wandb uses `step=` as the x-axis control for charts
                self.run.log(payload, step=step)
        except Exception:
            pass

    def close(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass


class LivePlotter:
    """Dev-only matplotlib plots. Safe to keep disabled by default."""

    def __init__(self):
        self.enabled = False
        try:
            import matplotlib.pyplot as plt  # type: ignore

            self.plt = plt
            self.plt.ion()
            self.fig, self.ax = self.plt.subplots(1, 2, figsize=(10, 4))
            self.steps: List[int] = []
            self.train_loss: List[float] = []
            self.val_loss: List[float] = []
            (self.l1,) = self.ax[0].plot([], [], label="train")
            (self.l2,) = self.ax[0].plot([], [], label="val")
            self.ax[0].legend()
            self.ax[0].set_title("Loss")
            self.enabled = True
        except Exception as e:
            print(f"[warn] Live plot disabled: {e}")
            self.enabled = False

    def maybe_update(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            et = str(event.get("type", ""))
            step = int(event.get("step", 0))
            if et == "train" and "loss" in event:
                self.steps.append(step)
                self.train_loss.append(float(event["loss"]))
            if et == "eval" and "val_loss" in event:
                self.val_loss.append(float(event["val_loss"]))

            self.l1.set_data(self.steps, self.train_loss)
            if self.val_loss:
                xs = self.steps[-len(self.val_loss) :] if len(self.val_loss) <= len(self.steps) else list(range(len(self.val_loss)))
                self.l2.set_data(xs, self.val_loss)
            self.ax[0].relim()
            self.ax[0].autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self.plt.ioff()
            self.plt.close(self.fig)
        except Exception:
            pass


def generate_analysis_png(_train_jsonl_path: str, _out_png: str) -> None:
    # Placeholder: v30 has a rich analysis figure generator.
    # Keeping the hook so CLI parity holds; safe no-op if matplotlib isn't installed.
    return


class RunLogger:
    """JSONL logger + optional HDF5, live plot, tensorboard.

    This is the modularized counterpart to v30's deep instrumentation. It keeps the same public surface
    (log(), h5_write_step(), finalize(), close()) so the runner can remain compatible.
    """

    def __init__(
        self,
        out_dir: str,
        *,
        instrument: str,
        cfg: Any,
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
            self._jsonl_f = open(self.train_jsonl_path, "w", encoding="utf-8")
            self.log({"type": "meta", "step": 0, "env": _env_info(device), "argv": sys.argv, "args": vars(args), "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else getattr(cfg, "__dict__", {})})

        if self.instrument == "full" and _h5py is not None:
            try:
                self._h5 = _h5py.File(self.h5_path, "w")
                meta = self._h5.create_group("meta")
                meta.attrs["created"] = _now_iso()
                meta.attrs["argv"] = " ".join(sys.argv)
            except Exception as e:
                print(f"[warn] Could not open HDF5 at {self.h5_path}: {e}")
                self._h5 = None

        if live_plot:
            self._live = LivePlotter()
        if tb:
            self._tb = TensorBoardWriter(self.out_dir)
        if wandb:
            # Best-effort wandb init: defaults are safe and require only --wandb.
            tags = None
            try:
                raw_tags = getattr(args, "wandb_tags", None)
                if raw_tags:
                    tags = [t.strip() for t in str(raw_tags).split(",") if t.strip()]
            except Exception:
                tags = None
            self._wandb = WandBWriter(
                self.out_dir,
                project=str(getattr(args, "wandb_project", "experiments")),
                entity=getattr(args, "wandb_entity", None),
                name=getattr(args, "wandb_name", None),
                group=getattr(args, "wandb_group", None),
                tags=tags,
                mode=str(getattr(args, "wandb_mode", "disabled")),
                cfg=cfg,
                args=args,
            )

        self._write_summary(initial_only=True)

    def close(self) -> None:
        try:
            if self._h5 is not None:
                self._h5.flush()
                self._h5.close()
        except Exception:
            pass
        try:
            if self._jsonl_f is not None:
                self._jsonl_f.flush()
                self._jsonl_f.close()
        except Exception:
            pass
        try:
            if self._tb is not None:
                self._tb.close()
        except Exception:
            pass
        try:
            if self._wandb is not None:
                self._wandb.close()
        except Exception:
            pass
        try:
            if self._live is not None:
                self._live.close()
        except Exception:
            pass

    def log(self, event: Dict[str, Any]) -> None:
        if self._jsonl_f is None and self._tb is None and self._wandb is None and self._live is None:
            return
        event = dict(event)
        event.setdefault("wall_time", _now_iso())
        if self._jsonl_f is not None:
            self._jsonl_f.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
            self._jsonl_f.flush()
        if self._tb is not None:
            self._tb.maybe_log(event)
        if self._wandb is not None:
            self._wandb.maybe_log(event)
        if self._live is not None:
            self._live.maybe_update(event)

    def h5_write_step(self, step: int, *, group: str, tensors: Dict[str, torch.Tensor], attrs: Optional[Dict[str, Any]] = None) -> None:
        if self._h5 is None:
            return
        try:
            g = self._h5.require_group(f"{group}/step_{int(step)}")
            if attrs:
                for k, v in attrs.items():
                    try:
                        g.attrs[k] = v
                    except Exception:
                        g.attrs[k] = str(v)
            for name, t in tensors.items():
                if t is None:
                    continue
                arr = t.detach().to("cpu")
                if arr.dtype in (torch.bfloat16, torch.float16):
                    arr = arr.to(torch.float16)
                else:
                    arr = arr.to(torch.float32)
                g.create_dataset(name, data=arr.numpy(), compression="gzip", compression_opts=4)
            self._h5.flush()
        except Exception as e:
            print(f"[warn] HDF5 write failed at step {step}: {e}")

    def finalize(self, *, best_val: float, last_step: int) -> None:
        try:
            if self.instrument != "off":
                generate_analysis_png(self.train_jsonl_path, self.png_path)
        except Exception as e:
            print(f"[warn] analysis.png generation failed: {e}")
        self._write_summary(initial_only=False, best_val=best_val, last_step=last_step)

    def _write_summary(self, *, initial_only: bool, best_val: Optional[float] = None, last_step: Optional[int] = None) -> None:
        try:
            lines: List[str] = []
            lines.append("# Run Summary")
            lines.append("")
            lines.append(f"- Created: `{_now_iso()}`")
            lines.append(f"- Out dir: `{self.out_dir}`")
            lines.append(f"- Device: `{_device_summary(self.device)}`")
            lines.append(f"- Command: `{(' '.join(sys.argv))}`")
            lines.append("")
            lines.append("## Model Config")
            lines.append("")
            try:
                cfg_json = json.dumps(asdict(self.cfg), indent=2, sort_keys=True)
            except Exception:
                cfg_json = json.dumps(getattr(self.cfg, "__dict__", {}), indent=2, sort_keys=True, default=str)
            lines.append("```json")
            lines.append(cfg_json)
            lines.append("```")
            lines.append("")
            if not initial_only and best_val is not None and last_step is not None:
                ppl = math.exp(best_val) if best_val < 20 else float("inf")  # type: ignore[name-defined]
                lines.append("## Results")
                lines.append("")
                lines.append(f"- Last step: `{last_step}`")
                lines.append(f"- Best val loss: `{best_val:.6f}` (ppl `{ppl:.2f}`)")
                lines.append("")
            Path(self.summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[warn] Failed to write summary.md: {e}")


