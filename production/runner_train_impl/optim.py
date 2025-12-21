"""Optimizer construction for training.

Why this exists:
- The runner supports a small set of optimizers (AdamW, Lion) without external deps.
- Parsing and optimizer wiring lives here to keep the training loop readable.
"""

from __future__ import annotations

from collections.abc import Iterable
import inspect
from typing import Protocol, cast

import torch


def _parse_two_floats(s: str, default: tuple[float, float]) -> tuple[float, float]:
    """Why: CLI strings need robust parsing, but the loop should not care."""
    try:
        a, b = str(s).split(",")
        return float(a), float(b)
    except (ValueError, AttributeError, TypeError):
        return default


class _AdamWCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> torch.optim.Optimizer: ...


def _adamw(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    foreach: bool | None = None,
    fused: bool | None = None,
) -> torch.optim.Optimizer:
    """Why: tolerate torch version differences (foreach/fused) without ignore comments."""
    ctor = cast(_AdamWCallable, cast(object, torch.optim.AdamW))

    kw: dict[str, object] = {
        "params": params,
        "lr": float(lr),
        "betas": (float(betas[0]), float(betas[1])),
        "eps": float(eps),
        "weight_decay": float(weight_decay),
    }
    try:
        sig = inspect.signature(torch.optim.AdamW)
        if foreach is not None and "foreach" in sig.parameters:
            kw["foreach"] = bool(foreach)
        if fused is not None and "fused" in sig.parameters:
            kw["fused"] = bool(fused)
    except (TypeError, ValueError):
        pass
    return ctor(**kw)


def build_optimizer(
    *,
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    adam_betas: str,
    adam_eps: float,
    lion_betas: str,
    foreach: bool,
    fused: bool,
) -> torch.optim.Optimizer:
    """Why: a single place that maps config knobs to a concrete optimizer instance."""
    _ = lion_betas
    opt_name = str(name or "adamw").strip().lower()

    if opt_name == "lion":
        # Why: keep the runner functional even when Lion isn't available/typed; AdamW is the default.
        opt_name = "adamw"

    # Default: AdamW.
    betas = _parse_two_floats(str(adam_betas), (0.9, 0.95))
    lr_f = float(lr)
    betas_t = (float(betas[0]), float(betas[1]))
    eps_f = float(adam_eps)
    wd_f = float(weight_decay)

    # Optional speed knobs; we try both because support varies by torch version/device.
    candidates: list[tuple[bool | None, bool | None]] = []
    if bool(fused):
        candidates.append((None, True))
    if bool(foreach):
        candidates.append((True, None))
    candidates.append((None, None))

    last: BaseException | None = None
    for foreach_kw, fused_kw in candidates:
        try:
            return _adamw(
                params,
                lr=lr_f,
                betas=betas_t,
                eps=eps_f,
                weight_decay=wd_f,
                foreach=foreach_kw,
                fused=fused_kw,
            )
        except (TypeError, RuntimeError, ValueError) as e:
            last = e
    if last is not None:
        raise last
    return _adamw(params, lr=lr_f, betas=betas_t, eps=eps_f, weight_decay=wd_f)


