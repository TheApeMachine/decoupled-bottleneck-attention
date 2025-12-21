"""Output directory naming helpers.

Why this exists:
- Some harnesses rely on a predictable `runs/{exp}_{tag}` layout.
- Keeping this logic centralized avoids ad-hoc path building in runners.
"""

from __future__ import annotations

import argparse
import os


def default_out_dir(args: argparse.Namespace) -> str | None:
    """Why: provide a stable default when the user doesn't pass --out-dir."""
    if getattr(args, "out_dir", None):
        return str(args.out_dir)
    exp = getattr(args, "exp", None)
    run_root = getattr(args, "run_root", "runs")
    tag = getattr(args, "run_tag", None)
    if not exp or exp == "paper_all":
        return None
    name = str(exp).replace("paper_", "")
    if tag:
        name = f"{name}_{tag}"
    return os.path.join(str(run_root), name)


