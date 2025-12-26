"""Rich-based logger with caramba theming.

Training runs produce lots of output. This logger makes it readable with:
- Semantic colors (cyan=info, green=success, amber=warning, red=error)
- Structured output (tables, panels, key-value pairs)
- Progress bars and spinners
- Training-specific helpers for consistent metrics display
"""
from __future__ import annotations

from typing import Any, Generator

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    track,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


# Caramba color theme - distinctive and cohesive
CARAMBA_THEME = Theme(
    {
        "info": "bold #7dcfff",  # Soft cyan - informational
        "success": "bold #9ece6a",  # Muted green - success
        "warning": "bold #e0af68",  # Warm amber - warnings
        "error": "bold #f7768e",  # Soft coral red - errors
        "highlight": "bold #bb9af7",  # Lavender purple - emphasis
        "muted": "dim #565f89",  # Slate gray - secondary info
        "metric": "#7aa2f7",  # Sky blue - metrics/numbers
        "path": "italic #73daca",  # Teal - file paths
        "step": "#ff9e64",  # Orange - step/progress indicators
    }
)


class Logger:
    """Unified logging interface with rich console output.

    Wraps Rich Console to provide semantic log levels, structured data
    display, and progress tracking—all with consistent theming.
    """

    def __init__(self) -> None:
        """Initialize with the caramba theme."""
        self.console = Console(theme=CARAMBA_THEME)

    # ─────────────────────────────────────────────────────────────────────
    # Basic Logging
    # ─────────────────────────────────────────────────────────────────────

    def log(self, message: str) -> None:
        """Log a generic message."""
        self.console.print(message)

    def info(self, message: str) -> None:
        """Log an informational message (cyan ℹ)."""
        self.console.print(f"[info]ℹ[/info] {message}")

    def success(self, message: str) -> None:
        """Log a success message (green ✓)."""
        self.console.print(f"[success]✓[/success] {message}")

    def warning(self, message: str) -> None:
        """Log a warning message (amber ⚠)."""
        self.console.print(f"[warning]⚠[/warning] {message}")

    def error(self, message: str) -> None:
        """Log an error message (red ✗)."""
        self.console.print(f"[error]✗[/error] {message}")

    # ─────────────────────────────────────────────────────────────────────
    # Structured Output
    # ─────────────────────────────────────────────────────────────────────

    def header(self, title: str, subtitle: str | None = None) -> None:
        """Print a prominent section header.

        Use this to mark major phases like "Blockwise Training" or
        "Benchmark Results".
        """
        header_text = Text()
        header_text.append("━" * 3 + " ", style="muted")
        header_text.append(title, style="highlight")
        if subtitle:
            header_text.append(f" • {subtitle}", style="muted")
        header_text.append(" " + "━" * 40, style="muted")
        self.console.print()
        self.console.print(header_text)
        self.console.print()

    def subheader(self, text: str) -> None:
        """Print a subtle subheader for subsections."""
        self.console.print(f"[muted]──[/muted] [highlight]{text}[/highlight]")

    def panel(
        self, content: str, title: str | None = None, style: str = "muted"
    ) -> None:
        """Display content in a bordered panel."""
        self.console.print(Panel(content, title=title, border_style=style))

    def table(
        self,
        title: str | None = None,
        columns: list[str] | None = None,
        rows: list[list[str]] | None = None,
    ) -> Table:
        """Create and optionally populate a styled table.

        If columns and rows are provided, prints immediately. Otherwise
        returns the Table for manual population.
        """
        table = Table(
            title=title,
            title_style="highlight",
            header_style="info",
            border_style="muted",
            row_styles=["", "dim"],
        )

        if columns and rows:
            for col in columns:
                table.add_column(col)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)

        return table

    def key_value(self, data: dict[str, Any], title: str | None = None) -> None:
        """Display key-value pairs in a clean format."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="muted")
        table.add_column("Value", style="metric")

        for key, value in data.items():
            table.add_row(f"{key}:", str(value))

        if title:
            self.subheader(title)
        self.console.print(table)

    def metric(self, name: str, value: float | int | str, unit: str = "") -> None:
        """Display a single metric with formatting.

        Floats are formatted to 4 decimal places.
        """
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        self.console.print(
            f"  [muted]{name}:[/muted] [metric]{formatted}[/metric]{unit}"
        )

    def step(self, current: int, total: int | None = None, message: str = "") -> None:
        """Display a step indicator for multi-phase operations."""
        if total:
            prefix = f"[step][{current}/{total}][/step]"
        else:
            prefix = f"[step][{current}][/step]"
        self.console.print(f"{prefix} {message}")

    def path(self, filepath: str, label: str = "") -> None:
        """Display a file path with optional label."""
        if label:
            self.console.print(f"  [muted]{label}:[/muted] [path]{filepath}[/path]")
        else:
            self.console.print(f"  [path]{filepath}[/path]")

    # ─────────────────────────────────────────────────────────────────────
    # Inspection
    # ─────────────────────────────────────────────────────────────────────

    def inspect(self, obj: object, **kwargs: Any) -> None:
        """Print an object with rich formatting.

        Delegates to console.print() which handles dicts, lists, etc.
        """
        self.console.print(obj, **kwargs)

    # ─────────────────────────────────────────────────────────────────────
    # Progress Tracking
    # ─────────────────────────────────────────────────────────────────────

    def progress(self, total: int, description: str) -> Generator[int, None, None]:
        """Track iteration progress with a styled progress bar."""
        yield from track(
            range(total),
            description=f"[info]{description}[/info]",
            console=self.console,
        )

    def spinner(self, description: str = "Processing...") -> Progress:
        """Create a spinner for indeterminate progress.

        Usage:
            with logger.spinner("Loading model...") as progress:
                task = progress.add_task("", total=None)
                # ... do work ...
        """
        return Progress(
            SpinnerColumn(style="info"),
            TextColumn("[info]{task.description}[/info]"),
            console=self.console,
            transient=True,
        )

    def progress_bar(self) -> Progress:
        """Create a rich progress bar for fine-grained control.

        Usage:
            with logger.progress_bar() as progress:
                task = progress.add_task("Training...", total=1000)
                for step in range(1000):
                    progress.update(task, advance=1)
        """
        return Progress(
            SpinnerColumn(style="info"),
            TextColumn("[info]{task.description}[/info]"),
            BarColumn(bar_width=40, style="muted", complete_style="success"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Training-Specific Helpers
    # ─────────────────────────────────────────────────────────────────────

    def training_step(
        self,
        phase: str,
        step: int,
        loss: float,
        extras: dict[str, float] | None = None,
    ) -> None:
        """Log a training step with consistent formatting.

        Args:
            phase: Training phase (e.g., "blockwise", "global")
            step: Current step number
            loss: Loss value
            extras: Additional metrics like {"ce": 0.5, "diff": 0.1}
        """
        parts = [
            f"[step]{phase}[/step] step=[metric]{step}[/metric] "
            f"loss=[metric]{loss:.6f}[/metric]"
        ]
        if extras:
            extra_str = " ".join(
                f"{k}=[metric]{v:.4f}[/metric]" for k, v in extras.items()
            )
            parts.append(f"({extra_str})")
        self.console.print(" ".join(parts))

    def benchmark_result(
        self,
        name: str,
        model: str,
        value: float,
        unit: str = "",
    ) -> None:
        """Log a benchmark result with consistent formatting."""
        self.console.print(
            f"  [muted]{model}:[/muted] [metric]{value:.2f}[/metric]{unit}"
        )

    def artifacts_summary(self, artifacts: dict[str, Any]) -> None:
        """Display a summary of generated artifacts."""
        self.console.print()
        self.success(f"Generated {len(artifacts)} artifacts:")
        for name, path in artifacts.items():
            self.console.print(f"    [muted]•[/muted] {name}: [path]{path}[/path]")


# ─────────────────────────────────────────────────────────────────────────────
# Module-Level Singleton
# ─────────────────────────────────────────────────────────────────────────────

_logger: Logger | None = None


def get_logger() -> Logger:
    """Get or create the singleton Logger instance.

    Using a singleton ensures consistent theming and avoids creating
    multiple Console instances.
    """
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger
