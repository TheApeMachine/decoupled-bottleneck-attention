"""
Triton availability for fused decoupled attention kernels.

Fused decode kernels are optional; most environments (CPU/MPS) won't have Triton.
We keep type-checking happy without requiring Triton to be installed.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

__all__ = ["TRITON_AVAILABLE", "triton_decoupled_q4q8q4_available"]


# At type-check time we force this off so Triton-only code can live behind runtime guards.
try:
    _triton_spec = importlib.util.find_spec("triton") is not None
    _triton_lang_spec = importlib.util.find_spec("triton.language") is not None
except (ImportError, ValueError, AttributeError):
    _triton_spec = False
    _triton_lang_spec = False

TRITON_AVAILABLE: bool = False if TYPE_CHECKING else bool(_triton_spec and _triton_lang_spec)


def triton_decoupled_q4q8q4_available() -> bool:
    """Check if fused decoupled q4/q8/q4 decode kernels can be used."""
    return bool(TRITON_AVAILABLE)
