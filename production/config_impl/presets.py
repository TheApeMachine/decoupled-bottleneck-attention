"""Experiment presets for training/sampling harnesses (legacy).

Why this exists:
- Paper harnesses need consistent configurations across runs.
- We keep presets separate so the rest of the config system can focus on intent
  and derived values.
"""

from __future__ import annotations

import argparse
import math
import sys


# Bottleneck/decoupled KV reduction target (d_model / attn_dim).
_BOTTLENECK_RATIO: float = 5.3333333333


def _round_up(x: float, multiple: int) -> int:
    multiple = int(max(1, multiple))
    return int(math.ceil(float(x) / float(multiple)) * multiple)


def _lcm(a: int, b: int) -> int:
    a = int(abs(a))
    b = int(abs(b))
    if a == 0 or b == 0:
        return int(max(a, b))
    return int(a // math.gcd(a, b) * b)


def _round_nearest(x: float, multiple: int) -> int:
    multiple = int(max(1, multiple))
    return int(round(float(x) / float(multiple)) * multiple)


EXP_PRESETS: dict[str, dict[str, object]] = {
    "paper_baseline": dict(attn_mode="standard"),
    "paper_bottleneck": dict(attn_mode="bottleneck", null_attn=True),
    # Decoupled flagship path: keep null_attn off by default to avoid extra branching and to preserve
    # fused/streaming decode fast paths. See `production/ablate_null_attn.py` for an explicit ablation.
    "paper_decoupled": dict(attn_mode="decoupled", tie_qk=True, null_attn=False, rope=True),
    "paper_gqa": dict(attn_mode="gqa"),
    # Training-oriented preset: expresses intent only; runtime performance is auto-tuned.
    "train_decoupled_fast": dict(attn_mode="decoupled", tie_qk=True, rope=True, null_attn=False),
}


def _argv_has_flag(flag: str) -> bool:
    """Why: distinguish defaults from explicit user overrides (argparse can't tell)."""
    if flag in sys.argv:
        return True
    prefix = str(flag) + "="
    return any(str(a).startswith(prefix) for a in sys.argv)


def apply_exp_preset(args: argparse.Namespace) -> None:
    """Why: keep experiment configurations consistent across runs/harnesses."""
    if not getattr(args, "exp", None):
        return
    exp = str(args.exp)
    if exp not in EXP_PRESETS and exp != "paper_all":
        raise ValueError(f"Unknown experiment preset: {exp}")

    # For paper_all, we don't set mode here; the runner loops over EXP_PRESETS.
    if exp == "paper_all":
        return

    preset = EXP_PRESETS[exp]

    # attn_mode
    if not _argv_has_flag("--attn-mode") and "attn_mode" in preset:
        args.attn_mode = preset["attn_mode"]

    # Experiment-specific dims (derived from d_model; no size tables).
    d_model = int(getattr(args, "d_model", 0) or 0)
    if d_model <= 0:
        return
    mode = str(getattr(args, "attn_mode", "") or "")

    if mode in ("standard", "gqa"):
        if int(getattr(args, "attn_dim", 0) or 0) <= 0:
            args.attn_dim = int(d_model)

    if mode == "bottleneck":
        if int(getattr(args, "attn_dim", 0) or 0) <= 0:
            args.attn_dim = int(max(32, _round_up(float(d_model) / float(_BOTTLENECK_RATIO), 32)))

    if mode == "decoupled":
        nh = int(getattr(args, "n_head", 0) or 0)
        rope_enabled = not bool(getattr(args, "no_rope", False))

        # Why: decoupled dims must be head-divisible; RoPE prefers even geo_head_dim.
        sem_geo_multiple = int(2 * nh) if (rope_enabled and nh > 0) else int(max(1, nh))
        attn_multiple = _lcm(32, int(max(1, sem_geo_multiple))) if nh > 0 else 32

        if int(getattr(args, "attn_dim", 0) or 0) <= 0:
            args.attn_dim = int(
                max(attn_multiple, _round_up(float(d_model) / float(_BOTTLENECK_RATIO), attn_multiple))
            )

        attn_dim = int(args.attn_dim)

        if int(getattr(args, "sem_dim", 0) or 0) <= 0 or int(getattr(args, "geo_dim", 0) or 0) <= 0:
            if nh <= 0:
                sem = int(max(32, _round_up(float(attn_dim) / 3.0, 32)))
                sem = min(sem, attn_dim - 32)
                geo = int(attn_dim - sem)
            else:
                sem = int(_round_nearest(float(attn_dim) / 3.0, sem_geo_multiple))
                sem = int(max(sem_geo_multiple, min(sem, attn_dim - sem_geo_multiple)))
                geo = int(attn_dim - sem)
            args.sem_dim = int(sem)
            args.geo_dim = int(geo)

    if mode == "gqa":
        if getattr(args, "kv_head", None) is None:
            nh = int(getattr(args, "n_head", 0) or 0)
            if nh > 0:
                cand = max(1, nh // 4)
                while cand > 1 and (nh % cand) != 0:
                    cand -= 1
                args.kv_head = int(max(1, cand))

    # Bool toggles (only set if user didn't explicitly toggle)
    if "null_attn" in preset:
        if (not _argv_has_flag("--null-attn")) and (not _argv_has_flag("--no-null-attn")):
            args.null_attn = bool(preset["null_attn"])
    if "tie_qk" in preset:
        if (not _argv_has_flag("--tie-qk")) and (not _argv_has_flag("--no-tie-qk")):
            args.tie_qk = bool(preset["tie_qk"])
    if "rope" in preset:
        if (not _argv_has_flag("--no-rope")) and (not _argv_has_flag("--rope")):
            if preset["rope"]:
                args.no_rope = False
            else:
                args.no_rope = True


