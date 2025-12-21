"""Small helpers for bucketing runtime tuning by prefix length."""

from __future__ import annotations

__all__ = ["pow2_bucket"]

def pow2_bucket(n: int) -> int:
    """Bucket `n` to the next power-of-two (used for prefix-length caching)."""
    if n <= 0:
        return 0
    return 1 << (int(n - 1).bit_length())


# Back-compat alias (avoid importing underscore names across modules).
_pow2_bucket = pow2_bucket


