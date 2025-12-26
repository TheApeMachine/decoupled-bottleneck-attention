"""Triton availability detection.

Fused attention kernels require Triton, which only works on CUDA. This module
provides safe runtime detection so code can import and type-check without
Triton installed, using fallback implementations on CPU/MPS.
"""
from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

__all__ = ["TRITON_AVAILABLE", "triton_decoupled_q4q8q4_available"]


# At type-check time, force this off so Triton-only code stays behind guards
try:
    _triton_spec = importlib.util.find_spec("triton") is not None
    _triton_lang_spec = importlib.util.find_spec("triton.language") is not None
except (ImportError, ValueError, AttributeError):
    _triton_spec = False
    _triton_lang_spec = False

TRITON_AVAILABLE: bool = (
    False if TYPE_CHECKING else bool(_triton_spec and _triton_lang_spec)
)


def triton_decoupled_q4q8q4_available() -> bool:
    """Check if fused decoupled q4/q8/q4 decode kernels can be used."""
    return bool(TRITON_AVAILABLE)
