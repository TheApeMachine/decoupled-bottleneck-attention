"""Optional Triton availability checks."""

from __future__ import annotations

__all__ = ["TRITON_AVAILABLE", "_triton_decoupled_q4q8q4_available"]


try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TRITON_AVAILABLE = False


def triton_decoupled_q4q8q4_available() -> bool:
    """Whether the fused decoupled q4/q8/q4 decode kernels are available."""
    return bool(TRITON_AVAILABLE)


# Back-compat alias (avoid importing underscore names across modules).
_triton_decoupled_q4q8q4_available = triton_decoupled_q4q8q4_available


