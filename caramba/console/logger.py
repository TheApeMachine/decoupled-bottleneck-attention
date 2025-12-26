"""logger is a custom logger based on rich."""
from __future__ import annotations
from typing import Any, Generator

from rich.console import Console
from rich.progress import track


class Logger:
    """
    Logger wraps rich Console and provides a unified interface for logging.
    """
    def __init__(self) -> None:
        self.console = Console()

    def log(self, message: str) -> None:
        """
        log is a generic log message.
        """
        self.console.print(message)

    def info(self, message: str) -> None:
        """
        info is a generic info message.
        """
        self.console.print(f"[bold green]{message}[/bold green]")

    def warning(self, message: str) -> None:
        """
        warning is a generic warning message.
        """
        self.console.print(f"[bold yellow]{message}[/bold yellow]")

    def error(self, message: str) -> None:
        """
        error is a generic error message.
        """
        self.console.print(f"[bold red]{message}[/bold red]")

    def inspect(self, obj: object, **kwargs: Any) -> None:
        """
        inspect produces a rich report on an object using Rich's inspect.
        """
        self.console.print(obj, **kwargs)

    def progress(self, total: int, description: str) -> Generator[int, None, None]:
        """
        progress tracks a progress bar.
        """
        yield from track(range(total), description=description, console=self.console)