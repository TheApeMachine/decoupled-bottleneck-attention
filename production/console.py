"""
Console helpers.

We use `rich` for nicer console UX when available, but always keep a safe fallback
that works in minimal environments (pipes, CI logs, missing deps).
"""

from __future__ import annotations

import os
import sys
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol, runtime_checkable, cast


@runtime_checkable
class ConsoleLike(Protocol):
    """Minimal surface we rely on for console output."""

    def print(self, *objects: object, **kwargs: object) -> None: ...

    def rule(self, title: str | None = None) -> None: ...

    def status(self, status: str, *, spinner: str | None = None) -> AbstractContextManager[object]: ...


class PlainConsole:
    """Fallback console when `rich` is unavailable."""

    def print(self, *objects: object, **kwargs: object) -> None:
        sep = cast(str, kwargs.get("sep", " "))
        end = cast(str, kwargs.get("end", "\n"))
        flush = bool(kwargs.get("flush", False))
        file_obj = kwargs.get("file", sys.stdout)
        try:
            file = cast(object, file_obj)
            # Best-effort compatibility with rich-style kwargs; ignore style/highlight.
            _ = kwargs.get("style", None)
            _ = kwargs.get("highlight", None)
            print(*(str(o) for o in objects), sep=sep, end=end, file=file, flush=flush)
        except (OSError, ValueError, TypeError):
            try:
                print(*(str(o) for o in objects), flush=True)
            except Exception:
                pass

    def rule(self, title: str | None = None) -> None:
        msg = f"--- {title} ---" if title else "---"
        self.print(msg)

    def status(self, _status: str, *, spinner: str | None = None) -> AbstractContextManager[object]:
        _ = spinner
        return nullcontext()


def rich_enabled() -> bool:
    """Global toggle (for users/CI): set NO_RICH=1 to disable rich output."""
    v = str(os.environ.get("NO_RICH", "")).strip().lower()
    return v not in ("1", "true", "yes", "on")


def rich_live_enabled() -> bool:
    """Only use live renderers (progress/status) when stdout is a TTY."""
    try:
        return bool(rich_enabled() and sys.stdout.isatty())
    except Exception:
        return False


_CONSOLE: ConsoleLike | None = None


def get_console() -> ConsoleLike:
    """Return a cached console instance (rich when available, otherwise plain)."""
    global _CONSOLE
    if _CONSOLE is not None:
        return _CONSOLE

    if not rich_enabled():
        _CONSOLE = PlainConsole()
        return _CONSOLE

    try:
        from rich.console import Console  # pyright: ignore[reportMissingImports]
    except ImportError:
        _CONSOLE = PlainConsole()
        return _CONSOLE

    try:
        # Let rich decide terminal behavior; it degrades gracefully in pipes.
        c = Console()
        _CONSOLE = cast(ConsoleLike, c)
        return _CONSOLE
    except (OSError, ValueError, TypeError):
        _CONSOLE = PlainConsole()
        return _CONSOLE


