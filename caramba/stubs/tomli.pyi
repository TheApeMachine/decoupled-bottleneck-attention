"""
tomli provides TOML parsing for Python < 3.11.
"""

from __future__ import annotations

from typing import Any, BinaryIO


def load(fp: BinaryIO, /) -> dict[str, Any]: ...


def loads(s: str | bytes, /) -> dict[str, Any]: ...


