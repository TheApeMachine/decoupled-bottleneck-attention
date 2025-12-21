from __future__ import annotations

import datetime
import os
import sys
from typing import cast

import torch


def _can_connect(host: str, port: int, *, timeout_s: float = 0.5) -> bool:
    """Best-effort TCP connectivity probe (no DNS caching assumptions)."""
    try:
        import socket

        with socket.create_connection((str(host), int(port)), timeout=float(timeout_s)):
            return True
    except OSError:
        return False


def auto_wandb_mode() -> str:
    """Pick a safe W&B mode based on environment and connectivity."""
    env_mode = os.environ.get("WANDB_MODE", None)
    if env_mode:
        m = str(env_mode).strip().lower()
        if m in ("online", "offline", "disabled"):
            return m
    return "online" if _can_connect("api.wandb.ai", 443, timeout_s=0.5) else "offline"


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        # Keep this robust even if CUDA isn't initialized.
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(device)
            except (AssertionError, RuntimeError, ValueError):
                name = "cuda"
            return f"cuda:{device.index or 0} ({name})"
        return "cuda"
    if device.type == "mps":
        return "mps"
    return str(device)


def env_info(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "unknown"),
        "device": device_summary(device),
    }
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except (AssertionError, RuntimeError, ValueError):
        pass
    return info


def coerce_int(value: object, *, default: int = 0) -> int:
    """Best-effort conversion of JSON-y values to int."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def coerce_float(value: object) -> float | None:
    """Best-effort conversion of JSON-y values to float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def coerce_str(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def safe_vars(ns: object) -> dict[str, object]:
    """Like vars(), but typed and safe for non-Namespace inputs."""
    try:
        return cast(dict[str, object], vars(ns))
    except TypeError:
        return {}
