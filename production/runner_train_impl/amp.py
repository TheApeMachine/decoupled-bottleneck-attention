"""AMP/GradScaler helpers for training.

Why this exists:
- PyTorch AMP components are optional depending on device/build.
- Centralizing the import and typing avoids sprinkling `type: ignore` across the codebase.
"""

from __future__ import annotations

import contextlib
from contextlib import nullcontext
from typing import Protocol, cast

import torch


class GradScalerLike(Protocol):
    """Minimal GradScaler interface used by the training loop."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor: ...

    def step(self, optimizer: torch.optim.Optimizer) -> None: ...

    def update(self) -> None: ...


class _AmpGradScalerCtor(Protocol):
    def __call__(self, device_type: str) -> object: ...


class _CudaGradScalerCtor(Protocol):
    def __call__(self) -> object: ...


def _get(obj: object, name: str, default: object = None) -> object:
    """Why: prevent `Any` from leaking out of getattr in strict type checking."""
    return cast(object, getattr(obj, name, default))


def make_grad_scaler(*, enabled: bool) -> GradScalerLike | None:
    """Why: return a typed scaler when available, otherwise None."""
    if not enabled:
        return None
    # Prefer dynamic getattr to avoid tight coupling to specific torch versions/stubs.
    try:
        amp_mod = _get(torch, "amp", None)
        if amp_mod is not None:
            gs_obj = _get(amp_mod, "GradScaler", None)
            if gs_obj is not None:
                gs = cast(_AmpGradScalerCtor, gs_obj)
                return cast(GradScalerLike, gs("cuda"))
    except (AttributeError, TypeError, RuntimeError):
        pass

    try:
        cuda_amp = _get(torch.cuda, "amp", None)
        if cuda_amp is None:
            return None
        gs2_obj = _get(cuda_amp, "GradScaler", None)
        if gs2_obj is None:
            return None
        gs2 = cast(_CudaGradScalerCtor, gs2_obj)
        return cast(GradScalerLike, gs2())
    except (AttributeError, TypeError, RuntimeError):
        return None


@contextlib.contextmanager
def autocast_ctx(device: torch.device, *, enabled: bool, dtype: torch.dtype):
    """Why: keep the training loop readable by hiding device-specific autocast branches."""
    if not enabled:
        with nullcontext():
            yield
        return
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=dtype):
            yield
        return
    if device.type == "mps":
        with torch.autocast("mps", dtype=dtype):
            yield
        return
    with torch.autocast("cpu", dtype=torch.bfloat16):
        yield


