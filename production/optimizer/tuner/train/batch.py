"""Batch/grad-accum tuning for training."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from production.memory_utils import device_synchronize, empty_device_cache, get_device_mem_stats
from production.selfopt_cache import get_cache_entry, set_cache_entry
from production.selfopt_utils import device_sig, hash_cfg, is_oom_error, restore_rng, snapshot_rng

from production.optimizer.tuner.train.types import TrainBatchPlan


def _default_micro_batches(target_gbs: int) -> list[int]:
    # Prefer larger microbatches first (less grad-accum overhead).
    out: list[int] = []
    g = int(max(1, target_gbs))
    b = g
    while b >= 1:
        out.append(int(b))
        if b == 1:
            break
        b = max(1, b // 2)
    # De-dup while preserving order.
    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def tune_batch_by_seq(
    *,
    cache_path: str | None,
    device: torch.device,
    cfg: object,
    model: torch.nn.Module,
    get_batch: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    seq_lens: list[int],
    target_gbs: int,
    warmup: int = 1,
    iters: int = 2,
    verbose: bool = False,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> TrainBatchPlan:
    """Tune (batch_size, grad_accum) per seq_len, with caching and RNG preservation."""
    seq_lens = sorted({int(s) for s in seq_lens if int(s) > 0})
    if not seq_lens:
        raise ValueError("seq_lens is empty")

    target_gbs = int(target_gbs)
    auto = bool(target_gbs <= 0)
    warmup = int(max(0, warmup))
    iters = int(max(1, iters))

    key = (
        f"{device_sig(device)}|train|cfg={hash_cfg(cfg)}|"
        f"{'auto' if auto else f'gbs={target_gbs}'}|amp={int(bool(amp_enabled))}|ampdt={str(amp_dtype)}"
    )

    if cache_path:
        cached = get_cache_entry(cache_path, section="train_plans", key=key)
        if isinstance(cached, dict) and "by_seq" in cached:
            try:
                by_seq_raw = cached.get("by_seq", {})
                by_seq: dict[int, tuple[int, int]] = {}
                if isinstance(by_seq_raw, dict):
                    for k, v in by_seq_raw.items():
                        if not isinstance(v, (list, tuple)) or len(v) != 2:
                            continue
                        by_seq[int(k)] = (int(v[0]), int(v[1]))
                if by_seq:
                    return TrainBatchPlan(by_seq=by_seq, target_gbs=target_gbs, warmup=warmup, iters=iters)
            except Exception:
                pass

    snap = snapshot_rng(device)
    try:
        model_was_training = bool(model.training)
        model.train()

        bs_list = _default_micro_batches(target_gbs) if not auto else []
        by_seq: dict[int, tuple[int, int]] = {}

        def _auto_max_micro_batch_for(dev: torch.device) -> int:
            # Auto probing is primarily useful on CUDA where OOM boundaries are sharp.
            # On CPU/MPS, probing large micro-batches can become extremely slow; keep bounded.
            if dev.type == "cpu":
                return 256
            if dev.type == "mps":
                return 128
            return 4096

        max_auto_bs = int(_auto_max_micro_batch_for(device))

        for s in seq_lens:
            best_tok_s = -1.0
            best_pair: tuple[int, int] | None = None
            best_peak = 0.0

            if verbose:
                print(
                    f"[selfopt][train] tuning seq_len={s} "
                    + ("(auto gbs)" if auto else f"target_gbs={target_gbs}")
                )

            # Auto mode: probe best micro-batch at ga=1 by doubling until OOM.
            if auto:
                candidates: list[tuple[int, int]] = []
                bs_try = 1
                while bs_try <= max_auto_bs:
                    candidates.append((int(bs_try), 1))
                    bs_try *= 2
                # Add a few intermediate points for smoother peak detection.
                mids: list[tuple[int, int]] = []
                for (b, _ga) in candidates:
                    if b >= 4:
                        mids.append((int(b // 2 + b // 4), 1))
                candidates.extend(mids)
                # de-dup
                seen: set[tuple[int, int]] = set()
                uniq: list[tuple[int, int]] = []
                for b, g in candidates:
                    if (b, g) not in seen:
                        uniq.append((b, g))
                        seen.add((b, g))

                for bs_try, ga_try in uniq:
                    # MPSGraph logits-size guard: B*T*V must not exceed INT_MAX.
                    if device.type == "mps":
                        try:
                            max_elems = 2_147_483_647
                            v = int(getattr(cfg, "vocab_size", 0) or 0)
                            if v > 0 and int(bs_try) * int(s) * int(v) > max_elems:
                                continue
                        except Exception:
                            pass

                    ok = True
                    tok = 0
                    peak = 0.0
                    try:
                        empty_device_cache(device)
                        device_synchronize(device)
                        if amp_enabled:
                            if device.type == "cpu":
                                cast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
                            else:
                                cast_ctx = torch.autocast(device.type, dtype=amp_dtype)
                        else:
                            cast_ctx = nullcontext()

                        # Warmup
                        for _ in range(warmup):
                            model.zero_grad(set_to_none=True)
                            xb, yb = get_batch(int(bs_try), int(s))
                            with cast_ctx:
                                logits, _ = model(xb)
                                loss = F.cross_entropy(
                                    logits.reshape(-1, logits.size(-1)), yb.reshape(-1)
                                )
                            loss.backward()
                        device_synchronize(device)

                        # Timed iters
                        t0 = time.perf_counter()
                        for _ in range(iters):
                            model.zero_grad(set_to_none=True)
                            xb, yb = get_batch(int(bs_try), int(s))
                            with cast_ctx:
                                logits, _ = model(xb)
                                loss = F.cross_entropy(
                                    logits.reshape(-1, logits.size(-1)), yb.reshape(-1)
                                )
                            loss.backward()
                            tok += int(xb.numel())
                            m = get_device_mem_stats(device)
                            peak = max(
                                peak,
                                float(m.get("mps_mem_driver_bytes", 0.0) or 0.0),
                                float(m.get("cuda_mem_reserved_bytes", 0.0) or 0.0),
                            )
                        device_synchronize(device)
                        dt = time.perf_counter() - t0
                        tok_s = float(tok / max(dt, 1e-9))
                    except Exception as e:
                        ok = False
                        tok_s = 0.0
                        if not is_oom_error(e) and verbose:
                            print(f"[selfopt][train] auto bs={bs_try} failed: {e}")
                    finally:
                        try:
                            model.zero_grad(set_to_none=True)
                        except Exception:
                            pass

                    if not ok:
                        continue
                    if tok_s > best_tok_s:
                        best_tok_s = float(tok_s)
                        best_pair = (int(bs_try), int(ga_try))
                        best_peak = float(peak)

            # Targeted mode: match requested global batch size with grad-accum.
            for bs_try in bs_list:
                # MPSGraph logits-size guard: B*T*V must not exceed INT_MAX.
                if device.type == "mps":
                    try:
                        max_elems = 2_147_483_647
                        v = int(getattr(cfg, "vocab_size", 0) or 0)
                        if v > 0 and int(bs_try) * int(s) * int(v) > max_elems:
                            continue
                    except Exception:
                        pass

                ga_try = int((max(1, target_gbs) + int(bs_try) - 1) // int(bs_try))
                ga_try = max(1, ga_try)
                gbs_eff = int(bs_try) * int(ga_try)
                if gbs_eff < max(1, target_gbs // 4):
                    continue

                ok = True
                tok = 0
                peak = 0.0
                try:
                    empty_device_cache(device)
                    device_synchronize(device)
                    if amp_enabled:
                        if device.type == "cpu":
                            cast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
                        else:
                            cast_ctx = torch.autocast(device.type, dtype=amp_dtype)
                    else:
                        cast_ctx = nullcontext()

                    # Warmup
                    for _ in range(warmup):
                        model.zero_grad(set_to_none=True)
                        for _m in range(ga_try):
                            xb, yb = get_batch(int(bs_try), int(s))
                            with cast_ctx:
                                logits, _ = model(xb)
                                loss = F.cross_entropy(
                                    logits.reshape(-1, logits.size(-1)), yb.reshape(-1)
                                )
                            (loss / ga_try).backward()
                    device_synchronize(device)

                    # Timed iters
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        model.zero_grad(set_to_none=True)
                        for _m in range(ga_try):
                            xb, yb = get_batch(int(bs_try), int(s))
                            with cast_ctx:
                                logits, _ = model(xb)
                                loss = F.cross_entropy(
                                    logits.reshape(-1, logits.size(-1)), yb.reshape(-1)
                                )
                            (loss / ga_try).backward()
                            tok += int(xb.numel())
                        m = get_device_mem_stats(device)
                        peak = max(
                            peak,
                            float(m.get("mps_mem_driver_bytes", 0.0) or 0.0),
                            float(m.get("cuda_mem_reserved_bytes", 0.0) or 0.0),
                        )
                    device_synchronize(device)
                    dt = time.perf_counter() - t0
                    tok_s = float(tok / max(dt, 1e-9))
                except Exception as e:
                    ok = False
                    tok_s = 0.0
                    if not is_oom_error(e) and verbose:
                        print(f"[selfopt][train] bs={bs_try} ga={ga_try} failed: {e}")
                finally:
                    try:
                        model.zero_grad(set_to_none=True)
                    except Exception:
                        pass

                if not ok:
                    continue
                if tok_s > best_tok_s:
                    best_tok_s = float(tok_s)
                    best_pair = (int(bs_try), int(ga_try))
                    best_peak = float(peak)

            if best_pair is None:
                best_pair = (1, 1 if auto else max(1, target_gbs))

            by_seq[int(s)] = best_pair
            if verbose:
                bs_b, ga_b = best_pair
                peak_gb = best_peak / (1024.0**3) if best_peak > 0 else 0.0
                print(
                    f"[selfopt][train] best@{s}: bs={bs_b} ga={ga_b} "
                    f"tok/s={best_tok_s:.0f} peak={peak_gb:.2f}GB"
                )

        plan = TrainBatchPlan(by_seq=by_seq, target_gbs=(0 if auto else target_gbs), warmup=warmup, iters=iters)
        if cache_path:
            try:
                payload = {
                    "by_seq": {str(k): [int(v[0]), int(v[1])] for k, v in by_seq.items()},
                    "target_gbs": int(target_gbs),
                    "warmup": int(warmup),
                    "iters": int(iters),
                    "ts": float(time.time()),
                }
                set_cache_entry(str(cache_path), section="train_plans", key=key, value=payload)
            except Exception:
                pass

        if not model_was_training:
            model.eval()
        return plan
    finally:
        restore_rng(device, snap)


