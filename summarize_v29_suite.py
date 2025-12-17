#!/usr/bin/env python3
"""
summarize_v29_suite.py

Summarize a completed v29 suite (baseline / gqa_kv2 / bottleneck_144 / decoupled_48_96)
across one or more seeds and optionally generate paper plots per seed without overwriting.

This script is intended to be run locally (matplotlib is only needed if you also run
generate_run_visualizations.py; this script itself does not import matplotlib).
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RUNS_DIR_DEFAULT = "runs"
ASSETS_DIR_DEFAULT = "assets"
PLOTTER = "generate_run_visualizations.py"


VARIANTS: Tuple[str, ...] = (
    "baseline",
    "gqa_kv2",
    "bottleneck_144",
    "decoupled_48_96",
)


@dataclass(frozen=True)
class RunMetrics:
    seed: int
    variant: str
    run_dir: str
    best_val_loss: float
    best_ppl: float
    kv_128k: Optional[str]  # keep as string to preserve units/format


_RE_BEST = re.compile(r"^-\s*Best val loss:\s*`([0-9.]+)`\s*\(ppl\s*`([0-9.]+)`\)\s*$")
_RE_KV_128K = re.compile(r"^-\s*This run policy @ 128k:\s*`(.+)`\s*$")


def _find_run_dirs(runs_dir: Path, tag: str, variant: str, seed: int) -> List[Path]:
    # Allow suffixes like _v2/_v3, etc.
    prefix = f"{tag}_{variant}_seed{seed}"
    out: List[Path] = []
    if not runs_dir.exists():
        return out
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == prefix or p.name.startswith(prefix + "_v"):
            out.append(p)
    # newest first
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return out


def _parse_summary(summary_path: Path) -> Tuple[float, float, Optional[str]]:
    best_loss: Optional[float] = None
    best_ppl: Optional[float] = None
    kv_128k: Optional[str] = None

    for line in summary_path.read_text().splitlines():
        m = _RE_BEST.match(line.strip())
        if m:
            best_loss = float(m.group(1))
            best_ppl = float(m.group(2))
            continue
        m = _RE_KV_128K.match(line.strip())
        if m:
            kv_128k = m.group(1).strip()

    if best_loss is None or best_ppl is None:
        raise ValueError(f"Could not parse best val loss / ppl from {summary_path}")
    return best_loss, best_ppl, kv_128k


def _discover_seeds(runs_dir: Path, tag: str) -> List[int]:
    seeds: set[int] = set()
    pat = re.compile(rf"^{re.escape(tag)}_({'|'.join(map(re.escape, VARIANTS))})_seed(\d+)(?:_v\d+)?$")
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        seeds.add(int(m.group(2)))
    return sorted(seeds)


def _fmt_mean_std(xs: List[float], digits: int = 3) -> str:
    if not xs:
        return "---"
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f}"
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs)
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run_plotter(tag: str, out_dir: str, baseline_dir: str, gqa_dir: str, bottleneck_dir: str, decoupled_dir: str) -> None:
    argv = [
        "python3.12",
        PLOTTER,
        "--tag",
        tag,
        "--run",
        f"baseline={baseline_dir}",
        "--run",
        f"bottleneck={bottleneck_dir}",
        "--run",
        f"gqa={gqa_dir}",
        "--run",
        f"decoupled={decoupled_dir}",
        "--baseline-label",
        "baseline",
        "--out",
        out_dir,
    ]
    subprocess.run(argv, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default=RUNS_DIR_DEFAULT)
    ap.add_argument("--assets", type=str, default=ASSETS_DIR_DEFAULT)
    ap.add_argument("--tag", type=str, default="m4max", help="Prefix used by the runner (e.g. m4max).")
    ap.add_argument("--seeds", type=str, default="auto", help="'auto' or comma-separated seeds, e.g. 1337,1338")
    ap.add_argument("--write", action="store_true", help="Write summary files into assets/.")
    ap.add_argument("--plot", action="store_true", help="Call generate_run_visualizations.py per seed.")
    ap.add_argument("--plot-seed-tag-prefix", type=str, default=None,
                    help="Override per-seed plot tag prefix. Default: <tag>_seed")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    assets_dir = Path(args.assets)

    if str(args.seeds).strip() == "auto":
        seeds = _discover_seeds(runs_dir, str(args.tag))
    else:
        seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    if not seeds:
        raise SystemExit("No seeds found. If --seeds auto, ensure runs/<tag>_<variant>_seed* exists.")

    metrics: List[RunMetrics] = []
    missing: List[str] = []
    for seed in seeds:
        for variant in VARIANTS:
            candidates = _find_run_dirs(runs_dir, str(args.tag), variant, seed)
            if not candidates:
                missing.append(f"{args.tag}_{variant}_seed{seed}")
                continue
            run_dir = candidates[0]
            summary = run_dir / "summary.md"
            if not summary.exists():
                missing.append(str(summary))
                continue
            best_loss, best_ppl, kv_128k = _parse_summary(summary)
            metrics.append(
                RunMetrics(
                    seed=seed,
                    variant=variant,
                    run_dir=str(run_dir),
                    best_val_loss=best_loss,
                    best_ppl=best_ppl,
                    kv_128k=kv_128k,
                )
            )

    if missing:
        print("[warn] missing:")
        for m in missing:
            print(" -", m)

    by_variant: Dict[str, List[RunMetrics]] = {v: [] for v in VARIANTS}
    for m in metrics:
        by_variant[m.variant].append(m)

    # Markdown summary
    lines: List[str] = []
    lines.append("# v29 Suite Summary")
    lines.append("")
    lines.append(f"- tag: `{args.tag}`")
    lines.append(f"- seeds: `{','.join(map(str, seeds))}`")
    lines.append("")
    lines.append("## Best val loss / ppl (per seed)")
    lines.append("")
    lines.append("| variant | seed | best_val_loss | best_ppl | kv@128k | run_dir |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for v in VARIANTS:
        for m in sorted(by_variant[v], key=lambda x: x.seed):
            lines.append(f"| {m.variant} | {m.seed} | {m.best_val_loss:.6f} | {m.best_ppl:.2f} | {m.kv_128k or '---'} | `{m.run_dir}` |")
    lines.append("")
    lines.append("## Aggregate across seeds (mean ± std)")
    lines.append("")
    lines.append("| variant | best_val_loss | best_ppl |")
    lines.append("|---|---:|---:|")
    for v in VARIANTS:
        xs_loss = [m.best_val_loss for m in by_variant[v]]
        xs_ppl = [m.best_ppl for m in by_variant[v]]
        lines.append(f"| {v} | {_fmt_mean_std(xs_loss, digits=3)} | {_fmt_mean_std(xs_ppl, digits=1)} |")
    lines.append("")

    md = "\n".join(lines) + "\n"
    print(md)

    # LaTeX snippet (rows only; you can paste into a tabular)
    # Note: kv@128k kept out of LaTeX row because units vary; use your existing memory table/figure.
    tex_lines: List[str] = []
    tex_lines.append("% Auto-generated by summarize_v29_suite.py")
    tex_lines.append("% Columns: Model & Params & Val Loss & Val PPL \\\\")
    tex_lines.append(r"Standard Baseline & 139.8M & " + _fmt_mean_std([m.best_val_loss for m in by_variant["baseline"]], 3) + " & " + _fmt_mean_std([m.best_ppl for m in by_variant["baseline"]], 1) + r" \\")
    tex_lines.append(r"GQA (kv=2) & 128.0M & " + _fmt_mean_std([m.best_val_loss for m in by_variant["gqa_kv2"]], 3) + " & " + _fmt_mean_std([m.best_ppl for m in by_variant["gqa_kv2"]], 1) + r" \\")
    tex_lines.append(r"Bottleneck (rank 144) & 116.8M & " + _fmt_mean_std([m.best_val_loss for m in by_variant["bottleneck_144"]], 3) + " & " + _fmt_mean_std([m.best_ppl for m in by_variant["bottleneck_144"]], 1) + r" \\")
    tex_lines.append(r"Decoupled 48/96 & 116.8M & " + _fmt_mean_std([m.best_val_loss for m in by_variant["decoupled_48_96"]], 3) + " & " + _fmt_mean_std([m.best_ppl for m in by_variant["decoupled_48_96"]], 1) + r" \\")
    tex = "\n".join(tex_lines) + "\n"

    if args.write:
        _write_text(assets_dir / f"{args.tag}_suite_summary.md", md)
        _write_text(assets_dir / f"{args.tag}_suite_rows.tex", tex)
        print(f"[write] {assets_dir / f'{args.tag}_suite_summary.md'}")
        print(f"[write] {assets_dir / f'{args.tag}_suite_rows.tex'}")

    if args.plot:
        prefix = args.plot_seed_tag_prefix or f"{args.tag}_seed"
        for seed in seeds:
            # Require all variants for plotting
            dirs: Dict[str, str] = {}
            for v in VARIANTS:
                c = _find_run_dirs(runs_dir, str(args.tag), v, seed)
                if not c:
                    dirs = {}
                    break
                dirs[v] = str(c[0])
            if not dirs:
                print(f"[plot] skip seed={seed} (missing variant dirs)")
                continue
            plot_tag = f"{prefix}{seed}"
            print(f"[plot] seed={seed} tag={plot_tag}")
            _run_plotter(
                tag=plot_tag,
                out_dir=str(assets_dir),
                baseline_dir=dirs["baseline"],
                gqa_dir=dirs["gqa_kv2"],
                bottleneck_dir=dirs["bottleneck_144"],
                decoupled_dir=dirs["decoupled_48_96"],
            )


if __name__ == "__main__":
    main()


