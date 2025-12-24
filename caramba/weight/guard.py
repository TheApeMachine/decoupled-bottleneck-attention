"""
guard provides small validation helpers for weight containers.
"""
from __future__ import annotations


def require_int(name: str, value: object, *, ge: int | None = None) -> int:
    """
    require_int validates that value is an int, optionally bounded below.
    """
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value)!r}")
    if ge is not None and value < ge:
        raise ValueError(f"{name} must be >= {ge}, got {value}")
    return value


def require_bool(name: str, value: object) -> bool:
    """
    require_bool validates that value is a bool.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value)!r}")
    return value


def require_float(name: str, value: object) -> float:
    """
    require_float validates that value is a float or int.
    """
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a float, got {type(value)!r}")
    return float(value)

