#!/usr/bin/env python3
"""
run_long_context_tests_v29.py

Convenience wrapper to run BOTH:
  - test_rope_extrapolation_v29.py
  - test_needle_haystack_v29.py

…for one or more v29 checkpoints, and write results into assets/ with stable names.

This script is meant to be called from the master paper runner.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple


def _parse_kv(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise ValueError(f"Expected label=path, got: {s}")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k or not v:
        raise ValueError(f"Bad label=path: {s}")
    return k, v


def _fmt(argv: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="python3.12")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--tag", type=str, default="m4max", help="Prefix for assets outputs.")
    ap.add_argument("--ckpt", action="append", required=True, help="label=path (repeatable)")

    ap.add_argument("--data-npy", type=str, default="fineweb_100m.npy")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--len", type=int, default=2_000_000)

    ap.add_argument("--rope-contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--rope-batch-size", type=int, default=1)
    ap.add_argument("--rope-num-batches", type=int, default=50)

    ap.add_argument("--needle-contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--needle-depths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--needle-trials", type=int, default=20)

    ap.add_argument("--run", action="store_true", help="Execute (default prints commands).")
    args = ap.parse_args()

    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.ckpt:
        label, ckpt_path = _parse_kv(spec)

        rope_out = str(assets_dir / f"{args.tag}_{label}_rope_extrapolation.png")
        needle_out = str(assets_dir / f"{args.tag}_{label}_needle_haystack.png")

        rope_cmd = [
            str(args.python),
            "test_rope_extrapolation_v29.py",
            "--ckpt",
            ckpt_path,
            "--data-npy",
            str(args.data_npy),
            "--offset",
            str(int(args.offset)),
            "--len",
            str(int(args.len)),
            "--contexts",
            *[str(int(x)) for x in args.rope_contexts],
            "--batch-size",
            str(int(args.rope_batch_size)),
            "--num-batches",
            str(int(args.rope_num_batches)),
            "--out",
            rope_out,
        ]
        if args.device:
            rope_cmd += ["--device", str(args.device)]

        needle_cmd = [
            str(args.python),
            "test_needle_haystack_v29.py",
            "--ckpt",
            ckpt_path,
            "--context-lengths",
            *[str(int(x)) for x in args.needle_contexts],
            "--depths",
            *[str(float(x)) for x in args.needle_depths],
            "--trials",
            str(int(args.needle_trials)),
            "--out",
            needle_out,
        ]
        if args.device:
            needle_cmd += ["--device", str(args.device)]

        print("\n" + "-" * 80)
        print(f"[{label}] RoPE extrapolation -> {rope_out}")
        print(_fmt(rope_cmd))
        if args.run:
            subprocess.run(rope_cmd, check=True)

        print("\n" + "-" * 80)
        print(f"[{label}] Needle-haystack -> {needle_out}")
        print(_fmt(needle_cmd))
        if args.run:
            subprocess.run(needle_cmd, check=True)


if __name__ == "__main__":
    main()


