"""Token-ID spec loader for calibration / verification."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def load_token_ids_spec(spec: str) -> list[int]:
    """Load token IDs from either:
    - a path to a file containing whitespace-separated ints
    - a path to a .npy file (np.load)
    - an inline whitespace-separated string of ints
    """
    s = str(spec)
    p = Path(s)
    if os.path.exists(s):
        if p.suffix == ".npy":
            arr = np.load(str(p), mmap_mode="r")
            arr = np.asarray(arr).reshape(-1)
            if arr.dtype != np.int64:
                arr = arr.astype(np.int64, copy=False)
            return [int(x) for x in arr.tolist()]
        raw = p.read_text(encoding="utf-8", errors="ignore")
        return [int(t) for t in raw.strip().split() if t.strip()]
    return [int(t) for t in s.strip().split() if t.strip()]


