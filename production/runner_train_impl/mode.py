"""Attention-mode parsing for training runner.

Why this exists:
- CLI/config flows historically used string values like "standard"/"decoupled".
- The model config uses a `Mode` enum; we translate here to keep the rest of the
  runner free of stringly-typed branching.
"""

from __future__ import annotations

from production.model.config import Mode


def mode_from_str(s: str | None) -> Mode:
    """Why: normalize legacy string modes into the canonical `Mode` enum."""
    v = str(s or "").strip().lower()
    match v:
        case "standard" | "baseline" | "base":
            return Mode.BASELINE
        case "gqa":
            return Mode.GQA
        case "bottleneck":
            return Mode.BOTTLENECK
        case "decoupled":
            return Mode.DECOUPLED
        case _:
            # Default keeps legacy behavior (bottleneck path is the conservative choice).
            return Mode.BOTTLENECK


