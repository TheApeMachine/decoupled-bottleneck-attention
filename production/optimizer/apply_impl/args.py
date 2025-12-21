"""Argument helpers for dynamic config.

Why this exists:
- `argparse.Namespace` is effectively untyped/dynamic, and our codebase bans `Any`.
- Centralizing typed getters prevents `Any` from leaking into the optimizer graph.
"""

from __future__ import annotations

from typing import cast


def arg(obj: object, name: str, default: object = None) -> object:
    """Typed getattr helper (prevents Any from leaking out of dynamic objects)."""
    return cast(object, getattr(obj, name, default))


