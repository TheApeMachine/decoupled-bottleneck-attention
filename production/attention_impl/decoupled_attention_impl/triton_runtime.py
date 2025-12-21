"""
Triton availability for decoupled attention.

Fused decode kernels are optional; most environments (CPU/MPS) won't have Triton.
We must keep type-checking happy without requiring Triton to be installed.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

__all__ = ["TRITON_AVAILABLE", "triton_decoupled_q4q8q4_available"]


def _spec_exists(name: str) -> bool:
    """Why: detect optional dependencies without importing them."""
    try:
        return importlib.util.find_spec(str(name)) is not None
    except (ImportError, ValueError, AttributeError):
        return False


# At type-check time we force this off so Triton-only code can live behind runtime guards.
TRITON_AVAILABLE: bool = False if TYPE_CHECKING else bool(_spec_exists("triton") and _spec_exists("triton.language"))


def triton_decoupled_q4q8q4_available() -> bool:
    """Why: central predicate for whether fused decoupled q4/q8/q4 decode kernels can be used."""
    return bool(TRITON_AVAILABLE)


