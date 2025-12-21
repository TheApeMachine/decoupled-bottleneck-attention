"""Device capability probing for self-optimization."""

from __future__ import annotations

import torch


def supports_dtype(dev: torch.device, dt: torch.dtype) -> bool:
    """Best-effort dtype probe (must never crash runtime)."""
    try:
        x = torch.ones(8, device=dev, dtype=dt)
        y = (x * 1.0001).sum()
        _ = float(y.detach().to("cpu").item())
        return True
    except (RuntimeError, ValueError, TypeError):
        return False


def choose_param_dtype(device: torch.device) -> torch.dtype:
    """Choose a conservative parameter dtype for this device."""
    if device.type == "cuda":
        if supports_dtype(device, torch.bfloat16):
            return torch.bfloat16
        if supports_dtype(device, torch.float16):
            return torch.float16
        return torch.float32
    if device.type == "mps":
        return torch.float32
    return torch.float32


def choose_amp(device: torch.device) -> tuple[bool, torch.dtype]:
    """Choose autocast enablement and autocast dtype."""
    if device.type not in ("cuda", "mps"):
        return False, torch.bfloat16

    # Prefer bf16 when supported, else fp16.
    if supports_dtype(device, torch.bfloat16):
        dt = torch.bfloat16
    elif supports_dtype(device, torch.float16):
        dt = torch.float16
    else:
        return False, torch.bfloat16
    return True, dt


