#!/usr/bin/env python3
"""
run_1b_launch.py

Minimal launcher to kick off a ~1B parameter run on a single GPU (CUDA) with sane defaults.
This avoids copy/pasting a 40-flag command line.

It invokes:
  v29_transformer_decoupled_bottleneck_instrumented.py

Defaults are chosen for "it should start" on typical rented GPUs:
  - bf16 params + bf16 autocast
  - gradient checkpointing ON
  - micro-batch 1 + grad accumulation
  - sequence-length curriculum up to 2048
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from typing import List, Tuple, Optional


V29_SCRIPT = "v29_transformer_decoupled_bottleneck_instrumented.py"


def _q(x: str) -> str:
    return shlex.quote(str(x))


def _fmt(argv: List[str]) -> str:
    return " ".join(_q(a) for a in argv)


def _resolve_out_dir(run_root: str, tag: str, variant: str, seed: int, if_exists: str) -> str:
    base = os.path.join(str(run_root), f"{tag}_{variant}_seed{int(seed)}")
    if not os.path.exists(base):
        return base
    if if_exists == "skip":
        return base
    if if_exists == "error":
        raise SystemExit(f"Refusing to overwrite existing out dir: {base} (use --if-exists suffix/skip)")
    k = 2
    while os.path.exists(f"{base}_v{k}"):
        k += 1
    return f"{base}_v{k}"


def _parse_seq_schedule(spec: str) -> List[Tuple[int, int]]:
    """
    "512@0,1024@2000,2048@12000" -> [(0,512),(2000,1024),(12000,2048)] sorted.
    Mirrors v29's behavior (uses step_idx = step-1).
    """
    spec = str(spec).strip()
    if not spec:
        return []
    out: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part or "@" not in part:
            continue
        a, b = part.split("@", 1)
        try:
            seq = int(a)
            st = int(b)
        except Exception:
            continue
        out.append((st, seq))
    out.sort(key=lambda x: x[0])
    return out


def _seq_len_for_step(step_idx: int, *, default_seq_len: int, schedule: List[Tuple[int, int]]) -> int:
    if not schedule:
        return int(default_seq_len)
    cur = int(default_seq_len)
    for st, seq in schedule:
        if step_idx >= int(st):
            cur = int(seq)
    return int(cur)


def _parse_token_count(s: str) -> int:
    """
    Parse token-count strings like '500M', '10B', '2000000000'.
    Mirrors prepare_fineweb.py behavior.
    """
    s = str(s).strip().upper()
    if not s:
        raise ValueError("empty token count")
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


def _estimate_total_tokens(
    *,
    steps: int,
    global_batch: int,
    default_seq_len: int,
    schedule: List[Tuple[int, int]],
) -> int:
    """
    Estimate tokens processed over `steps` optimizer steps:
      tokens = sum_{step_idx=0..steps-1} (global_batch * seq_len(step_idx))
    """
    steps = int(max(0, steps))
    if steps == 0:
        return 0
    if not schedule:
        return int(steps * global_batch * default_seq_len)

    # Build piecewise-constant segments.
    breaks: List[int] = [0]
    for st, _seq in schedule:
        st = int(st)
        if 0 < st < steps:
            breaks.append(st)
    breaks = sorted(set(breaks))

    total = 0
    for i, start in enumerate(breaks):
        end = breaks[i + 1] if (i + 1) < len(breaks) else steps
        seq = _seq_len_for_step(start, default_seq_len=default_seq_len, schedule=schedule)
        total += int((end - start) * global_batch * seq)
    return int(total)


def _steps_for_target_tokens(
    *,
    target_tokens: int,
    global_batch: int,
    default_seq_len: int,
    schedule: List[Tuple[int, int]],
    lo: int = 1,
    hi: int = 5_000_000,
) -> int:
    """
    Find smallest steps such that planned_tokens(steps) >= target_tokens.
    """
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")

    lo = int(max(1, lo))
    hi = int(max(lo, hi))

    # Expand hi if needed (rare, but safe).
    while _estimate_total_tokens(steps=hi, global_batch=global_batch, default_seq_len=default_seq_len, schedule=schedule) < target_tokens:
        hi *= 2
        if hi > 50_000_000:
            raise RuntimeError("target token budget too large; refusing to search >50M steps")

    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        got = _estimate_total_tokens(steps=mid, global_batch=global_batch, default_seq_len=default_seq_len, schedule=schedule)
        if got >= target_tokens:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="python3.12")
    ap.add_argument("--device", type=str, default="cuda", help="cuda recommended; can be cpu/mps for smoke tests")
    ap.add_argument("--data", type=str, default="fineweb_1b.npy")
    ap.add_argument("--data-format", type=str, default="npy", choices=["auto", "text", "npy", "bin", "pt"])
    ap.add_argument("--vocab-size", type=int, default=50257)

    ap.add_argument("--run-root", type=str, default="runs")
    ap.add_argument("--tag", type=str, default="oneb")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--if-exists", type=str, default="suffix", choices=["suffix", "skip", "error"])

    ap.add_argument("--variant", type=str, default="decoupled", choices=["baseline", "gqa_kv2", "bottleneck", "decoupled"])

    # 1B-ish model (≈0.98–1.05B depending on vocab/emb details)
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--d-ff", type=int, default=6144)
    ap.add_argument("--block", type=int, default=2048)

    # Training knobs
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument(
        "--target-tokens",
        type=str,
        default=None,
        help="Optional token budget like '10B' or '500M'. If set, overrides --steps by solving for steps.",
    )
    ap.add_argument("--optimizer", type=str, default="lion", choices=["lion", "adamw"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1, help="Micro-batch size")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--train-seq-len", type=int, default=1024)
    ap.add_argument("--seq-schedule", type=str, default="512@0,1024@2000,2048@12000")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0, help="Set >0 if you want periodic step*.pt files")

    # Memory/speed defaults
    ap.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false", help="Disable grad checkpointing")
    ap.add_argument("--param-dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="Disable autocast (not recommended for 1B on 1 GPU)")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--opt-fused", dest="opt_fused", action="store_true", default=True, help="CUDA-only; ignored on other devices")
    ap.add_argument("--no-opt-fused", dest="opt_fused", action="store_false", help="Disable fused optimizer (if it causes issues)")

    # Noise defaults
    ap.add_argument("--instrument", type=str, default="basic", choices=["off", "basic", "medium", "full"])
    ap.add_argument("--analysis-every", type=int, default=0)
    ap.add_argument("--live", type=str, default="rich", choices=["auto", "off", "basic", "rich"])

    ap.add_argument("--run", action="store_true", help="Execute (default prints command only).")
    args = ap.parse_args()

    # Derived dims for bottleneck/decoupled that mirror the repo convention:
    # sem = d_model/16, geo = d_model/8, attn = sem+geo
    sem_dim = int(args.d_model // 16)
    geo_dim = int(args.d_model // 8)
    attn_dim = int(sem_dim + geo_dim)

    variant_args: Tuple[str, ...]
    if args.variant == "baseline":
        variant_args = ("--attn-mode", "standard")
    elif args.variant == "gqa_kv2":
        variant_args = ("--attn-mode", "gqa", "--kv-head", "2", "--attn-dim", str(int(args.d_model)))
    elif args.variant == "bottleneck":
        variant_args = ("--attn-mode", "bottleneck", "--attn-dim", str(attn_dim), "--null-attn")
    else:
        # decoupled
        variant_args = (
            "--attn-mode",
            "decoupled",
            "--attn-dim",
            str(attn_dim),
            "--sem-dim",
            str(sem_dim),
            "--geo-dim",
            str(geo_dim),
            "--tie-qk",
            "--null-attn",
        )

    out_dir = _resolve_out_dir(args.run_root, args.tag, args.variant, args.seed, args.if_exists)

    global_batch = int(args.batch_size) * int(args.grad_accum)
    sched = _parse_seq_schedule(args.seq_schedule) if args.seq_schedule else []
    target_tokens: Optional[int] = None
    if args.target_tokens:
        target_tokens = _parse_token_count(str(args.target_tokens))
        args.steps = _steps_for_target_tokens(
            target_tokens=target_tokens,
            global_batch=int(global_batch),
            default_seq_len=int(args.train_seq_len),
            schedule=sched,
        )
    planned_tokens = _estimate_total_tokens(
        steps=int(args.steps),
        global_batch=int(global_batch),
        default_seq_len=int(args.train_seq_len),
        schedule=sched,
    )
    planned_tokens_b = planned_tokens / 1e9
    extra = ""
    if target_tokens is not None:
        extra = f" (target={target_tokens/1e9:.3f}B)"
    print(f"[plan] global_batch={global_batch} | steps={int(args.steps)} | planned_tokens≈{planned_tokens_b:.3f}B{extra}")
    if sched:
        print(f"[plan] seq_schedule={args.seq_schedule}")

    argv: List[str] = [
        str(args.python),
        V29_SCRIPT,
        "--mode",
        "train",
        "--device",
        str(args.device),
        "--data",
        str(args.data),
        "--data-format",
        str(args.data_format),
        "--vocab-size",
        str(int(args.vocab_size)),
        "--out-dir",
        out_dir,
        "--seed",
        str(int(args.seed)),
        "--steps",
        str(int(args.steps)),
        "--d-model",
        str(int(args.d_model)),
        "--layers",
        str(int(args.layers)),
        "--n-head",
        str(int(args.n_head)),
        "--d-ff",
        str(int(args.d_ff)),
        "--block",
        str(int(args.block)),
        "--embed-dim",
        str(int(args.d_model)),
        "--optimizer",
        str(args.optimizer),
        "--lr",
        str(float(args.lr)),
        "--batch-size",
        str(int(args.batch_size)),
        "--grad-accum",
        str(int(args.grad_accum)),
        "--train-seq-len",
        str(int(args.train_seq_len)),
        "--seq-schedule",
        str(args.seq_schedule),
        "--eval-every",
        str(int(args.eval_every)),
        "--eval-iters",
        str(int(args.eval_iters)),
        "--log-every",
        str(int(args.log_every)),
        "--save-every",
        str(int(args.save_every)),
        "--instrument",
        str(args.instrument),
        "--analysis-every",
        str(int(args.analysis_every)),
        "--live",
        str(args.live),
        "--param-dtype",
        str(args.param_dtype),
    ] + list(variant_args)

    if args.grad_checkpoint:
        argv.append("--grad-checkpoint")
    if args.amp:
        argv += ["--amp", "--amp-dtype", str(args.amp_dtype)]
    if args.opt_fused:
        argv.append("--opt-fused")

    print(_fmt(argv))
    if args.run:
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()


