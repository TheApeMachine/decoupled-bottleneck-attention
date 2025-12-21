"""torch.compile tuning for training."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from production.selfopt_cache import get_cache_entry, set_cache_entry
from production.selfopt_utils import device_sig, hash_cfg, restore_rng, snapshot_rng

from production.optimizer.tuner.train.bench import bench_train_tok_s
from production.optimizer.tuner.train.types import TrainCompilePlan


def tune_torch_compile(
    *,
    cache_path: str | None,
    device: torch.device,
    cfg: object,
    model: torch.nn.Module,
    get_batch: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    mode: str = "reduce-overhead",
    warmup: int = 1,
    iters: int = 2,
    hysteresis: float = 0.03,
    verbose: bool = False,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, TrainCompilePlan]:
    """Decide whether to use torch.compile for training, caching per (device, cfg, shapes)."""
    enabled_default = bool(device.type == "cuda" and hasattr(torch, "compile"))
    mode = str(mode or "default")
    warmup = int(max(0, warmup))
    iters = int(max(1, iters))
    hysteresis = float(max(0.0, hysteresis))

    key = (
        f"{device_sig(device)}|train_compile|cfg={hash_cfg(cfg)}|"
        f"bs={int(batch_size)}|ga={int(grad_accum)}|seq={int(seq_len)}|"
        f"amp={int(bool(amp_enabled))}|ampdt={str(amp_dtype)}|mode={mode}"
    )

    if cache_path:
        cached = get_cache_entry(cache_path, section="train_compile", key=key)
        if isinstance(cached, dict) and "enabled" in cached:
            try:
                want = bool(cached.get("enabled", False))
                plan = TrainCompilePlan(
                    enabled=want, mode=mode, warmup=warmup, iters=iters, hysteresis=hysteresis
                )
                if want and enabled_default:
                    try:
                        return torch.compile(model, mode=mode), plan
                    except Exception:
                        return model, TrainCompilePlan(
                            enabled=False,
                            mode=mode,
                            warmup=warmup,
                            iters=iters,
                            hysteresis=hysteresis,
                        )
                return model, plan
            except Exception:
                pass

    if not enabled_default:
        return model, TrainCompilePlan(
            enabled=False, mode=mode, warmup=warmup, iters=iters, hysteresis=hysteresis
        )

    snap = snapshot_rng(device)
    try:
        tok_s_base = bench_train_tok_s(
            device=device,
            model=model,
            get_batch=get_batch,
            batch_size=batch_size,
            grad_accum=grad_accum,
            seq_len=seq_len,
            warmup=warmup,
            iters=iters,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        if verbose:
            print(f"[selfopt][train] compile probe baseline tok/s={tok_s_base:.0f}")

        try:
            compiled = torch.compile(model, mode=mode)
        except Exception as e:
            if verbose:
                print(f"[selfopt][train] torch.compile failed; continuing without it: {e}")
            plan = TrainCompilePlan(
                enabled=False, mode=mode, warmup=warmup, iters=iters, hysteresis=hysteresis
            )
            if cache_path:
                try:
                    set_cache_entry(
                        str(cache_path),
                        section="train_compile",
                        key=key,
                        value={"enabled": False, "ts": float(time.time())},
                    )
                except Exception:
                    pass
            return model, plan

        tok_s_comp = bench_train_tok_s(
            device=device,
            model=compiled,
            get_batch=get_batch,
            batch_size=batch_size,
            grad_accum=grad_accum,
            seq_len=seq_len,
            warmup=warmup,
            iters=iters,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        if verbose:
            print(f"[selfopt][train] compile probe compiled tok/s={tok_s_comp:.0f}")

        want = bool(tok_s_comp >= tok_s_base * (1.0 + float(hysteresis)))
        plan = TrainCompilePlan(
            enabled=want, mode=mode, warmup=warmup, iters=iters, hysteresis=hysteresis
        )
        if cache_path:
            try:
                set_cache_entry(
                    str(cache_path),
                    section="train_compile",
                    key=key,
                    value={
                        "enabled": bool(want),
                        "tok_s_base": float(tok_s_base),
                        "tok_s_comp": float(tok_s_comp),
                        "ts": float(time.time()),
                    },
                )
            except Exception:
                pass
        return (compiled if want else model), plan
    finally:
        restore_rng(device, snap)


