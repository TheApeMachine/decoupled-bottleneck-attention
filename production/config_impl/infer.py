"""
Inference utilities for intent-first workflows.

The public CLI deliberately avoids exposing many architecture knobs.
We infer missing context (like depth or dataset scale) from filenames/paths so
runs remain reproducible without requiring extra flags.
"""

from __future__ import annotations

import os
import re


def infer_layers_from_out_dir(out_dir: str) -> int | None:
    """
    infer_layers_from_out_dir allows encoding depth in run directories
    without exposing `--layers` everywhere.
    """
    try:
        base = os.path.basename(str(out_dir).rstrip("/")).lower()
    except (TypeError, OSError):
        return None
    m = re.search(r"(?:^|[_\\-])(?:l|layers)(\d+)(?:$|[_\\-])", base)
    if not m:
        return None
    try:
        n = int(m.group(1))
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None


def infer_dataset_tokens_from_path(path: str) -> int | None:
    """
    infer_dataset_tokens_from_path derives dataset scale cheaply without scanning files.

    Supported examples:
    - `fineweb_20b.npy`   -> 20_000_000_000
    - `fineweb_100m.npy`  -> 100_000_000 tokens
    - `fineweb_1b.npy`    -> 1_000_000_000 tokens
    """
    try:
        base = os.path.basename(str(path)).lower()
        if m := re.search(r"(\d+)([bm])", base):
            return int(m.group(1)) * (1_000_000 if m.group(2) == "m" else 1_000_000_000)

        if os.path.exists(meta_path := f"{path}.meta"):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line and (kv := line.split(":", 1))[0].strip().lower() == "tokens":
                        n = int(kv[1].strip().replace("_", ""))
                        return n if n > 0 else None
    except (ValueError, TypeError, OSError):
        pass
    return None
