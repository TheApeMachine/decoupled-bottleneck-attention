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
import sys
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
    original = str(s).strip()
    if not original:
        raise ValueError("empty token count")

    # Reject negative values explicitly (avoid silently accepting -1K, etc.).
    if original.startswith("-"):
        raise ValueError(f"negative token count: {original!r}")

    s_norm = original.upper()
    if s_norm.endswith(("B", "M", "K")):
        suffix = s_norm[-1]
        num_str = s_norm[:-1]
        mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
        try:
            tokens_f = float(num_str) * mult
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid token count: {original!r}") from e
        if tokens_f < 0:
            raise ValueError(f"negative token count: {original!r}")
        return int(tokens_f)

    try:
        tokens_i = int(s_norm)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid token count: {original!r}") from e
    if tokens_i < 0:
        raise ValueError(f"negative token count: {original!r}")
    return int(tokens_i)


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
    ap.add_argument("--python", type=str, default="python")
    ap.add_argument("--device", type=str, default="cuda", help="cuda recommended; can be cpu/mps for smoke tests")
    ap.add_argument("--data", type=str, default="fineweb_1b.npy")
    ap.add_argument("--data-format", type=str, default="npy", choices=["auto", "text", "npy", "bin", "pt"])
    ap.add_argument("--vocab-size", type=int, default=50257)

    ap.add_argument("--run-root", type=str, default="runs")
    ap.add_argument("--tag", type=str, default="oneb")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--if-exists", type=str, default="suffix", choices=["suffix", "skip", "error"])

    ap.add_argument("--variant", type=str, default="decoupled", choices=["baseline", "gqa", "gqa_kv2", "bottleneck", "decoupled"])
    ap.add_argument("--kv-head", type=int, default=None, help="For --variant gqa: number of KV heads (must divide --n-head). Default: n_head/4.")

    # Optional stabilizers / inductive biases (make opt-in; can be workload-dependent)
    ap.add_argument("--null-attn", dest="null_attn", action="store_true", default=False, help="Enable learnable null token (attend-nowhere).")
    ap.add_argument("--tie-qk", dest="tie_qk", action="store_true", default=False, help="Tie Q/K projections where supported (not supported for gqa unless kv_head==n_head).")

    # 1B-ish model (TinyLlama-ish defaults)
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=22)
    ap.add_argument("--n-head", type=int, default=32)
    ap.add_argument("--d-ff", type=int, default=5632)
    ap.add_argument("--block", type=int, default=2048)

    # Optional explicit interaction dims (otherwise derived from d_model):
    # - default: sem=d_model/16, geo=d_model/8, attn=sem+geo (≈5.33× interaction compression)
    ap.add_argument("--sem-dim", type=int, default=None, help="Decoupled semantic Q/K dim (total across heads). Default: d_model/16.")
    ap.add_argument("--geo-dim", type=int, default=None, help="Decoupled geometric Q/K dim (total across heads). Default: d_model/8.")
    ap.add_argument("--attn-dim", type=int, default=None, help="Value dim (total across heads). Default: sem_dim + geo_dim.")

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
    ap.add_argument("--seq-schedule", type=str, default="1024@0,2048@10000")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0, help="Set >0 if you want periodic step*.pt files")
    ap.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to initialize/resume from.")
    ap.add_argument("--resume", action="store_true", help="Resume training from --ckpt into the existing out dir (no suffix).")
    ap.add_argument("--save-optim", dest="save_optim", action="store_true", default=True,
                    help="Include optimizer/scaler state in periodic/last checkpoints for resume (default: on).")
    ap.add_argument("--no-save-optim", dest="save_optim", action="store_false", help="Disable saving optimizer state in checkpoints.")

    # Memory/speed defaults
    ap.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false", help="Disable grad checkpointing")
    ap.add_argument("--param-dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="Disable autocast (not recommended for 1B on 1 GPU)")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--opt-fused", dest="opt_fused", action="store_true", default=True, help="CUDA-only; ignored on other devices")
    ap.add_argument("--no-opt-fused", dest="opt_fused", action="store_false", help="Disable fused optimizer (if it causes issues)")
    ap.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile in v29 (experimental).")
    ap.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])

    # Noise defaults
    ap.add_argument("--instrument", type=str, default="basic", choices=["off", "basic", "medium", "full"])
    ap.add_argument("--analysis-every", type=int, default=0)
    # Default to no TUI for max training throughput; enable explicitly with --live basic/rich.
    ap.add_argument("--live", type=str, default="off", choices=["auto", "off", "basic", "rich"])

    # Optional Weights & Biases logging
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (requires `wandb` package + `wandb login`).")
    ap.add_argument("--wandb-project", type=str, default="experiments", help="W&B project name.")
    ap.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team (optional).")
    ap.add_argument("--wandb-name", type=str, default=None, help="W&B run name (optional). Defaults to out-dir basename.")
    ap.add_argument("--wandb-group", type=str, default=None, help="W&B group name (optional).")
    ap.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags (optional).")
    ap.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "online", "offline"],
                    help="W&B mode. Default disabled unless --wandb is set; 'offline' writes locally for later sync.")
    ap.add_argument("--wandb-id", type=str, default=None, help="Optional W&B run id (for resume).")

    ap.add_argument("--run", action="store_true", help="Execute (default prints command only).")
    args = ap.parse_args()
    steps_was_explicit = any(a == "--steps" or a.startswith("--steps=") for a in sys.argv[1:])
    original_steps = int(args.steps)

    # Derived / overridden dims
    sem_dim = int(args.sem_dim) if args.sem_dim is not None else int(args.d_model // 16)
    geo_dim = int(args.geo_dim) if args.geo_dim is not None else int(args.d_model // 8)
    attn_dim = int(args.attn_dim) if args.attn_dim is not None else int(sem_dim + geo_dim)

    # Basic divisibility sanity (matches v29 implementation constraints)
    H = int(args.n_head)
    if int(args.d_model) % H != 0:
        raise SystemExit(f"--d-model must be divisible by --n-head (got d_model={args.d_model}, n_head={H})")
    if sem_dim % H != 0:
        raise SystemExit(f"--sem-dim must be divisible by --n-head (got sem_dim={sem_dim}, n_head={H})")
    if geo_dim % H != 0:
        raise SystemExit(f"--geo-dim must be divisible by --n-head (got geo_dim={geo_dim}, n_head={H})")
    if attn_dim % H != 0:
        raise SystemExit(f"--attn-dim must be divisible by --n-head (got attn_dim={attn_dim}, n_head={H})")
    geo_hd = geo_dim // H
    if geo_hd % 2 != 0:
        raise SystemExit(f"RoPE requires even geo head dim; pick geo_dim divisible by 2*n_head (geo_dim={geo_dim}, n_head={H})")

    variant_args: Tuple[str, ...]
    if args.variant == "baseline":
        variant_args = ("--attn-mode", "standard")
    elif args.variant in {"gqa", "gqa_kv2"}:
        kv_head = 2 if args.variant == "gqa_kv2" else (int(args.kv_head) if args.kv_head is not None else max(1, H // 4))
        if H % int(kv_head) != 0:
            raise SystemExit(f"GQA requires n_head % kv_head == 0 (got n_head={H}, kv_head={kv_head})")
        # Note: tie_qk is not supported for gqa unless kv_head == n_head.
        if bool(args.tie_qk):
            raise SystemExit("--tie-qk is not supported for GQA unless kv_head == n_head (use --variant baseline).")
        variant_args = ("--attn-mode", "gqa", "--kv-head", str(int(kv_head)), "--attn-dim", str(int(args.d_model)))
    elif args.variant == "bottleneck":
        variant_args = ("--attn-mode", "bottleneck", "--attn-dim", str(int(attn_dim)))
    else:
        # decoupled
        variant_args = (
            "--attn-mode",
            "decoupled",
            "--attn-dim",
            str(int(attn_dim)),
            "--sem-dim",
            str(int(sem_dim)),
            "--geo-dim",
            str(int(geo_dim)),
        )

    # Append optional stabilizers where supported
    extra_flags: List[str] = []
    if bool(args.null_attn):
        extra_flags.append("--null-attn")
    if bool(args.tie_qk):
        # safe: baseline/bottleneck/decoupled support tie-qk; gqa_kv2 blocked above
        extra_flags.append("--tie-qk")

    # Output dir: for resume we must reuse the base dir (so logs/checkpoints continue).
    base_out_dir = os.path.join(str(args.run_root), f"{args.tag}_{args.variant}_seed{int(args.seed)}")
    if bool(args.resume):
        if not args.ckpt:
            raise SystemExit("--resume requires --ckpt")
        if not os.path.exists(base_out_dir):
            raise SystemExit(f"--resume expected existing out dir: {base_out_dir}")
        out_dir = base_out_dir
    else:
        out_dir = _resolve_out_dir(args.run_root, args.tag, args.variant, args.seed, args.if_exists)

    global_batch = int(args.batch_size) * int(args.grad_accum)
    sched = _parse_seq_schedule(args.seq_schedule) if args.seq_schedule else []

    # Context length sanity: you only *actually* train long-context if block_size supports it.
    # Avoid the easy footgun where seq-schedule asks for 4096/8192 but --block remains 2048.
    max_sched_seq = 0
    try:
        max_sched_seq = max((int(seq) for (_st, seq) in sched), default=0)
    except Exception:
        max_sched_seq = 0
    max_requested_seq = max(int(args.train_seq_len), int(max_sched_seq or 0))
    if max_requested_seq > int(args.block):
        raise SystemExit(
            f"Context schedule exceeds --block. Requested max seq_len={max_requested_seq} "
            f"(train_seq_len={args.train_seq_len}, schedule_max={max_sched_seq or 0}) "
            f"but block={int(args.block)}. Increase --block or reduce --seq-schedule."
        )

    target_tokens: Optional[int] = None
    if args.target_tokens:
        target_tokens = _parse_token_count(str(args.target_tokens))
        args.steps = _steps_for_target_tokens(
            target_tokens=target_tokens,
            global_batch=int(global_batch),
            default_seq_len=int(args.train_seq_len),
            schedule=sched,
        )
        if steps_was_explicit:
            print(
                f"Note: --target-tokens provided; overriding --steps (original {original_steps}) with computed {int(args.steps)} steps."
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
        "--ckpt",
        str(args.ckpt) if args.ckpt else "",
        "--instrument",
        str(args.instrument),
        "--analysis-every",
        str(int(args.analysis_every)),
        "--live",
        str(args.live),
        "--param-dtype",
        str(args.param_dtype),
    ] + list(variant_args) + extra_flags

    # Remove the placeholder --ckpt if not provided (avoid passing empty string)
    if not args.ckpt:
        # argv contains ["--ckpt", ""] as inserted above; remove both
        i = argv.index("--ckpt")
        argv.pop(i)  # flag
        argv.pop(i)  # value
    elif bool(args.resume):
        argv.append("--resume")
    if bool(args.save_optim):
        argv.append("--save-optim")
    if bool(args.compile):
        argv += ["--compile", "--compile-mode", str(args.compile_mode)]

    if args.grad_checkpoint:
        argv.append("--grad-checkpoint")
    if args.amp:
        argv += ["--amp", "--amp-dtype", str(args.amp_dtype)]
    if args.opt_fused:
        argv.append("--opt-fused")

    if bool(args.wandb):
        argv.append("--wandb")
        argv += ["--wandb-project", str(args.wandb_project)]
        if args.wandb_entity:
            argv += ["--wandb-entity", str(args.wandb_entity)]
        if args.wandb_name:
            argv += ["--wandb-name", str(args.wandb_name)]
        if args.wandb_group:
            argv += ["--wandb-group", str(args.wandb_group)]
        if args.wandb_tags:
            argv += ["--wandb-tags", str(args.wandb_tags)]
        # Always forward mode so offline/online behavior is explicit in the printed command.
        argv += ["--wandb-mode", str(args.wandb_mode)]
        if args.wandb_id:
            argv += ["--wandb-id", str(args.wandb_id)]

    print(_fmt(argv))
    if args.run:
        subprocess.run(argv, check=True)
    else:
        print("\n(dry-run) Add --run to execute, e.g. `python run_1b_launch.py ... --run`.")


if __name__ == "__main__":
    main()


