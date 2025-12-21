"""Count parsing/formatting utilities for intent flags."""

from __future__ import annotations

import math
import re


class CountCodec:
    """Parse/format human-friendly counts like `100m`, `1b`, `1.5b`, `2e9`."""

    _COUNT_RE: re.Pattern[str] = re.compile(
        r"^\s*(\d+(?:\.\d+)?)([kmbt])?\s*$", flags=re.IGNORECASE
    )

    @staticmethod
    def parse(spec: object | None) -> int | None:
        if spec is None:
            return None
        if isinstance(spec, int):
            return int(spec) if int(spec) > 0 else None
        s = str(spec).strip().lower().replace("_", "")
        if not s:
            return None
        try:
            v = float(s)
            if math.isfinite(v) and v > 0:
                return int(v)
        except (ValueError, OverflowError):
            pass
        m = CountCodec._COUNT_RE.match(s)
        if not m:
            return None
        num = float(m.group(1))
        suf = (m.group(2) or "").lower()
        mult_map = {
            "": 1,
            "k": 1_000,
            "m": 1_000_000,
            "b": 1_000_000_000,
            "t": 1_000_000_000_000,
        }
        mult = mult_map.get(suf, 1)
        v = int(num * mult)
        return v if v > 0 else None

    @staticmethod
    def format(n: int) -> str:
        n = int(n)
        if n % 1_000_000_000_000 == 0:
            return f"{n // 1_000_000_000_000}t"
        if n % 1_000_000_000 == 0:
            return f"{n // 1_000_000_000}b"
        if n % 1_000_000 == 0:
            return f"{n // 1_000_000}m"
        if n % 1_000 == 0:
            return f"{n // 1_000}k"
        return str(n)
