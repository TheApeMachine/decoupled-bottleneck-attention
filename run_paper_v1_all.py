#!/usr/bin/env python3
"""
run_paper_v1_all.py

"One button" runner for the experiments/analyses discussed for a strong arXiv v1:
  - Add an additional seed for the main v29 suite (baseline / gqa_kv2 / bottleneck_144 / decoupled_48_96)
  - Run a longer-horizon comparison (baseline vs decoupled)
  - Run a small baseline LR sweep (baseline only) to address "weak baseline" critiques
  - Run decoupled ablations: null token, tie-QK, and no-RoPE
  - Regenerate paper plots + suite summaries
  - Generate the rank-evidence figure (spectrum + entropy effective-rank)
  - Run long-context sanity checks (RoPE extrapolation + passkey needle probe)

This script prints commands by default. Add --run to actually execute them sequentially.
It does not assume any specific environment beyond being run from the repo root.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional


RUNS_DIR = "runs"


def _q(x: str) -> str:
    return shlex.quote(str(x))


def _fmt(argv: List[str]) -> str:
    return " ".join(_q(a) for a in argv)


def _tag_lr(lr: str) -> str:
    # safe-ish tag suffix for file names
    return str(lr).replace(".", "p").replace("-", "m").replace("+", "")


def _pick_latest_run_dir(runs_dir: Path, tag: str, variant: str, seed: int) -> Optional[Path]:
    prefix = f"{tag}_{variant}_seed{seed}"
    cands: List[Path] = []
    if not runs_dir.exists():
        return None
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == prefix or p.name.startswith(prefix + "_v"):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _run(argv: List[str], *, cwd: Path, do_run: bool) -> None:
    print("\n" + "-" * 90)
    print(_fmt(argv))
    if do_run:
        subprocess.run(argv, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="python3.12")
    ap.add_argument("--device", type=str, required=True, help="cpu|mps|cuda")
    ap.add_argument("--data-npy", type=str, default="fineweb_100m.npy")
    ap.add_argument("--runs-dir", type=str, default=RUNS_DIR)
    ap.add_argument("--assets-dir", type=str, default="assets")

    # Main v29 suite (additional seed)
    ap.add_argument("--main-tag", type=str, default="m4max")
    ap.add_argument("--add-seed", type=int, default=1339, help="Extra seed to run for the main suite")

    # Longer run (baseline vs decoupled)
    ap.add_argument("--long-tag", type=str, default="m4max_long12k")
    ap.add_argument("--long-seed", type=int, default=1337)
    ap.add_argument("--long-steps", type=int, default=12000)

    # Baseline LR sweep (address baseline fairness)
    ap.add_argument("--lr-sweep-seed", type=int, default=1337)
    ap.add_argument("--lr-sweep-lrs", type=str, default="2e-4,4e-4")

    # Ablations (decoupled only)
    ap.add_argument("--abl-seed", type=int, default=1337)
    ap.add_argument("--abl-steps", type=int, default=6000)

    # Rank evidence (uses main suite ckpts)
    ap.add_argument("--rank-seed", type=int, default=1337)
    ap.add_argument("--rank-seq-len", type=int, default=1024)
    ap.add_argument("--rank-offset", type=int, default=0)

    # Long-context probes (uses main suite ckpts)
    ap.add_argument("--probe-seed", type=int, default=1337)

    ap.add_argument("--run", action="store_true", help="Actually execute commands.")
    args = ap.parse_args()

    repo = Path(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = repo / str(args.runs_dir)

    # 1) Additional seed for main suite (all variants)
    _run(
        [
            str(args.python),
            "run_v29_suite.py",
            "--device",
            str(args.device),
            "--data",
            str(args.data_npy),
            "--tag",
            str(args.main_tag),
            "--seeds",
            str(int(args.add_seed)),
            "--if-exists",
            "suffix",
            "--run",
        ],
        cwd=repo,
        do_run=args.run,
    )

    # 2) Regenerate suite summary + plots (auto-detect seeds)
    _run(
        [
            str(args.python),
            "summarize_v29_suite.py",
            "--tag",
            str(args.main_tag),
            "--assets",
            str(args.assets_dir),
            "--runs",
            str(args.runs_dir),
            "--write",
            "--plot",
        ],
        cwd=repo,
        do_run=args.run,
    )

    # 3) Longer training horizon: baseline vs decoupled
    _run(
        [
            str(args.python),
            "run_v29_suite.py",
            "--device",
            str(args.device),
            "--data",
            str(args.data_npy),
            "--tag",
            str(args.long_tag),
            "--seeds",
            str(int(args.long_seed)),
            "--only",
            "baseline,decoupled_48_96",
            "--steps",
            str(int(args.long_steps)),
            "--if-exists",
            "suffix",
            "--run",
        ],
        cwd=repo,
        do_run=args.run,
    )

    # 4) Baseline LR sweep (baseline only, short horizon)
    lrs = [x.strip() for x in str(args.lr_sweep_lrs).split(",") if x.strip()]
    for lr in lrs:
        tag = f"{args.main_tag}_baseline_lr{_tag_lr(lr)}"
        _run(
            [
                str(args.python),
                "run_v29_suite.py",
                "--device",
                str(args.device),
                "--data",
                str(args.data_npy),
                "--tag",
                str(tag),
                "--seeds",
                str(int(args.lr_sweep_seed)),
                "--only",
                "baseline",
                "--lr",
                str(lr),
                "--if-exists",
                "suffix",
                "--run",
            ],
            cwd=repo,
            do_run=args.run,
        )

    # 5) Decoupled ablations (single seed)
    ablations = [
        ("dec_null", "null"),
        ("dec_no_null", "no_null"),
        ("dec_tieqk", "tie_qk"),
        ("dec_no_rope", "no_rope"),
    ]
    for tag_suffix, abl in ablations:
        tag = f"{args.main_tag}_{tag_suffix}"
        _run(
            [
                str(args.python),
                "run_v29_suite.py",
                "--device",
                str(args.device),
                "--data",
                str(args.data_npy),
                "--tag",
                str(tag),
                "--seeds",
                str(int(args.abl_seed)),
                "--only",
                "decoupled_48_96",
                "--steps",
                str(int(args.abl_steps)),
                "--ablations",
                str(abl),
                "--if-exists",
                "suffix",
                "--run",
            ],
            cwd=repo,
            do_run=args.run,
        )

    # Resolve ckpts for rank evidence + probes from the main suite (latest runs)
    baseline_dir = _pick_latest_run_dir(runs_dir, str(args.main_tag), "baseline", int(args.rank_seed))
    dec_dir = _pick_latest_run_dir(runs_dir, str(args.main_tag), "decoupled_48_96", int(args.rank_seed))
    if baseline_dir is None or dec_dir is None:
        print("\n" + "-" * 90)
        print("[warn] Could not find main suite run dirs for rank/probe steps; skipping those unless you run them manually.")
        return

    baseline_ckpt = baseline_dir / "best.pt"
    dec_ckpt = dec_dir / "best.pt"

    # 6) Rank evidence figure (paper appendix expects assets/m4max_rank_evidence.png)
    _run(
        [
            str(args.python),
            "plot_rank_evidence.py",
            "--ckpt",
            f"baseline={str(baseline_ckpt)}",
            "--ckpt",
            f"decoupled={str(dec_ckpt)}",
            "--data-npy",
            str(args.data_npy),
            "--offset",
            str(int(args.rank_offset)),
            "--seq-len",
            str(int(args.rank_seq_len)),
            "--device",
            str(args.device),
            "--out",
            str(Path(args.assets_dir) / f"{args.main_tag}_rank_evidence.png"),
        ],
        cwd=repo,
        do_run=args.run,
    )

    # 7) Long-context probes (RoPE extrapolation + needle)
    probe_cmd = [
        str(args.python),
        "run_long_context_tests_v29.py",
        "--python",
        str(args.python),
        "--device",
        str(args.device),
        "--tag",
        str(args.main_tag),
        "--ckpt",
        f"baseline={str(baseline_ckpt)}",
        "--ckpt",
        f"decoupled={str(dec_ckpt)}",
        "--data-npy",
        str(args.data_npy),
        "--offset",
        "0",
        "--len",
        "2000000",
    ]
    if args.run:
        probe_cmd.append("--run")
    _run(probe_cmd, cwd=repo, do_run=args.run)

    print("\n" + "=" * 90)
    print("Done. Next:")
    print(f"- Compile paper.tex (appendix will now include seed1338 plots, rank evidence, and long-context probes).")
    print(f"- If you ran the extra seed, re-run summarize_v29_suite.py and update Table 2 numbers accordingly.")


if __name__ == "__main__":
    main()


