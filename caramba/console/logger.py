"""logger is a custom logger based on rich."""
from __future__ import annotations
from typing import Generator

from rich.console import Console
from rich import inspect
from rich.progress import track


class Logger:
    """
    Logger wraps rich Console and provides a unified interface for logging.
    """
    def __init__(self):
        self.console = Console()

    def log(self, message: str):
        """
        log is a generic log message.
        """
        self.console.print(message)

    def info(self, message: str):
        """
        info is a generic info message.
        """
        self.console.print(f"[bold green]{message}[/bold green]")

    def warning(self, message: str):
        """
        warning is a generic warning message.
        """
        self.console.print(f"[bold yellow]{message}[/bold yellow]")

    def error(self, message: str):
        """
        error is a generic error message.
        """
        self.console.print(f"[bold red]{message}[/bold red]")


    def inspect(self, obj: object):
        """
        inspect produces a rich report on an object.
        """
        inspect(obj)

    def progress(self, total: int, description: str) -> Generator[int, None, None]:
        """
        progress tracks a progress bar.
        """
        yield from track(range(total), description=description, console=self.console)