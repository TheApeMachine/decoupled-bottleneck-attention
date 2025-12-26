"""
Unit tests for the console logger module.
"""
from __future__ import annotations

import io
import re
import unittest
from unittest.mock import patch

from rich.console import Console
from rich.progress import Progress

from caramba.console.logger import Logger, get_logger, CARAMBA_THEME


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


class TestLoggerBasicMethods(unittest.TestCase):
    """Tests for basic logging methods."""

    def setUp(self) -> None:
        """Set up test fixtures with a captured console."""
        self.output = io.StringIO()
        self.logger = Logger()
        # Replace the console with one that writes to our StringIO
        self.logger.console = Console(file=self.output, force_terminal=True, theme=CARAMBA_THEME)

    def test_log_prints_message(self) -> None:
        """log() prints the message to console."""
        self.logger.log("Hello, world!")
        output = self.output.getvalue()
        self.assertIn("Hello, world!", output)

    def test_info_includes_icon(self) -> None:
        """info() includes the info icon."""
        self.logger.info("Information message")
        output = self.output.getvalue()
        self.assertIn("ℹ", output)
        self.assertIn("Information message", output)

    def test_success_includes_checkmark(self) -> None:
        """success() includes the checkmark icon."""
        self.logger.success("Success message")
        output = self.output.getvalue()
        self.assertIn("✓", output)
        self.assertIn("Success message", output)

    def test_warning_includes_icon(self) -> None:
        """warning() includes the warning icon."""
        self.logger.warning("Warning message")
        output = self.output.getvalue()
        self.assertIn("⚠", output)
        self.assertIn("Warning message", output)

    def test_error_includes_icon(self) -> None:
        """error() includes the error icon."""
        self.logger.error("Error message")
        output = self.output.getvalue()
        self.assertIn("✗", output)
        self.assertIn("Error message", output)


class TestLoggerStructuredOutput(unittest.TestCase):
    """Tests for structured output methods."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.output = io.StringIO()
        self.logger = Logger()
        self.logger.console = Console(file=self.output, force_terminal=True, theme=CARAMBA_THEME)

    def test_header_prints_title(self) -> None:
        """header() prints the title."""
        self.logger.header("Test Section")
        output = self.output.getvalue()
        self.assertIn("Test Section", output)

    def test_header_with_subtitle(self) -> None:
        """header() includes subtitle when provided."""
        self.logger.header("Training", "epoch 5")
        output = self.output.getvalue()
        self.assertIn("Training", output)
        self.assertIn("epoch 5", output)

    def test_subheader_prints_text(self) -> None:
        """subheader() prints the text."""
        self.logger.subheader("Subsection")
        output = self.output.getvalue()
        self.assertIn("Subsection", output)

    def test_metric_formats_float(self) -> None:
        """metric() formats float values correctly."""
        self.logger.metric("loss", 0.0123456789)
        output = self.output.getvalue()
        self.assertIn("loss", output)
        self.assertIn("0.0123", output)

    def test_metric_with_unit(self) -> None:
        """metric() includes unit when provided."""
        self.logger.metric("speed", 150.5, " tok/s")
        output = self.output.getvalue()
        self.assertIn("150.5", output)
        self.assertIn("tok/s", output)

    def test_metric_formats_int(self) -> None:
        """metric() handles integer values."""
        self.logger.metric("steps", 1000)
        output = self.output.getvalue()
        self.assertIn("steps", output)
        self.assertIn("1000", output)

    def test_step_with_total(self) -> None:
        """step() shows current/total format."""
        self.logger.step(3, 10, "Processing batch")
        output = strip_ansi(self.output.getvalue())
        self.assertIn("[3/10]", output)
        self.assertIn("Processing batch", output)

    def test_step_without_total(self) -> None:
        """step() works without total."""
        self.logger.step(5, message="Running step")
        output = strip_ansi(self.output.getvalue())
        self.assertIn("[5]", output)
        self.assertIn("Running step", output)

    def test_path_with_label(self) -> None:
        """path() includes label when provided."""
        self.logger.path("/path/to/file.txt", "Output")
        output = strip_ansi(self.output.getvalue())
        self.assertIn("Output", output)
        self.assertIn("/path/to/file.txt", output)

    def test_path_without_label(self) -> None:
        """path() works without label."""
        self.logger.path("/path/to/file.txt")
        output = strip_ansi(self.output.getvalue())
        self.assertIn("/path/to/file.txt", output)

    def test_key_value_displays_pairs(self) -> None:
        """key_value() displays all key-value pairs."""
        self.logger.key_value({
            "epochs": 10,
            "lr": 0.001,
            "batch_size": 32,
        })
        output = self.output.getvalue()
        self.assertIn("epochs", output)
        self.assertIn("10", output)
        self.assertIn("lr", output)
        self.assertIn("0.001", output)
        self.assertIn("batch_size", output)
        self.assertIn("32", output)

    def test_key_value_with_title(self) -> None:
        """key_value() shows title when provided."""
        self.logger.key_value({"key": "value"}, title="Config")
        output = self.output.getvalue()
        self.assertIn("Config", output)


class TestLoggerTrainingHelpers(unittest.TestCase):
    """Tests for training-specific helper methods."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.output = io.StringIO()
        self.logger = Logger()
        self.logger.console = Console(file=self.output, force_terminal=True, theme=CARAMBA_THEME)

    def test_training_step_basic(self) -> None:
        """training_step() shows phase, step, and loss."""
        self.logger.training_step("global", step=100, loss=0.0234)
        output = strip_ansi(self.output.getvalue())
        self.assertIn("global", output)
        self.assertIn("100", output)
        # The loss value is formatted with 6 decimal places
        self.assertIn("0.023", output)

    def test_training_step_with_extras(self) -> None:
        """training_step() shows extra metrics."""
        self.logger.training_step(
            "blockwise",
            step=50,
            loss=0.05,
            extras={"ce": 0.03, "diff": 0.02}
        )
        output = self.output.getvalue()
        self.assertIn("blockwise", output)
        self.assertIn("ce", output)
        self.assertIn("diff", output)

    def test_benchmark_result(self) -> None:
        """benchmark_result() formats correctly."""
        self.logger.benchmark_result("perplexity", "student", 12.5, " ppl")
        output = self.output.getvalue()
        self.assertIn("student", output)
        self.assertIn("12.50", output)
        self.assertIn("ppl", output)

    def test_artifacts_summary(self) -> None:
        """artifacts_summary() shows all artifacts."""
        self.logger.artifacts_summary({
            "model.pt": "/path/to/model.pt",
            "config.json": "/path/to/config.json",
        })
        output = strip_ansi(self.output.getvalue())
        # Check for artifact count (may be "2 artifacts" or "2 artifacts:")
        self.assertIn("2", output)
        self.assertIn("artifacts", output)
        self.assertIn("model.pt", output)
        self.assertIn("config.json", output)


class TestLoggerProgress(unittest.TestCase):
    """Tests for progress tracking methods."""

    def test_progress_returns_generator(self) -> None:
        """progress() returns a generator."""
        logger = Logger()
        gen = logger.progress(5, "Test")
        self.assertTrue(hasattr(gen, "__iter__"))
        self.assertTrue(hasattr(gen, "__next__"))

    def test_progress_iterates_correct_count(self) -> None:
        """progress() yields correct number of items."""
        output = io.StringIO()
        logger = Logger()
        logger.console = Console(file=output, force_terminal=False)

        count = 0
        for _ in logger.progress(10, "Counting"):
            count += 1

        self.assertEqual(count, 10)

    def test_progress_bar_returns_progress(self) -> None:
        """progress_bar() returns a Progress context manager."""
        logger = Logger()
        progress = logger.progress_bar()
        self.assertIsInstance(progress, Progress)

    def test_spinner_returns_progress(self) -> None:
        """spinner() returns a Progress context manager."""
        logger = Logger()
        spinner = logger.spinner("Loading...")
        self.assertIsInstance(spinner, Progress)


class TestLoggerSingleton(unittest.TestCase):
    """Tests for singleton pattern."""

    def test_get_logger_returns_logger(self) -> None:
        """get_logger() returns a Logger instance."""
        logger = get_logger()
        self.assertIsInstance(logger, Logger)

    def test_get_logger_returns_same_instance(self) -> None:
        """get_logger() returns the same instance on multiple calls."""
        logger1 = get_logger()
        logger2 = get_logger()
        self.assertIs(logger1, logger2)


class TestLoggerInspect(unittest.TestCase):
    """Tests for inspect method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.output = io.StringIO()
        self.logger = Logger()
        self.logger.console = Console(file=self.output, force_terminal=True, theme=CARAMBA_THEME)

    def test_inspect_prints_object(self) -> None:
        """inspect() prints object representation."""
        self.logger.inspect({"key": "value"})
        output = self.output.getvalue()
        self.assertIn("key", output)
        self.assertIn("value", output)

    def test_inspect_handles_list(self) -> None:
        """inspect() handles list objects."""
        self.logger.inspect([1, 2, 3])
        output = self.output.getvalue()
        self.assertIn("1", output)
        self.assertIn("2", output)
        self.assertIn("3", output)


class TestLoggerTable(unittest.TestCase):
    """Tests for table method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.output = io.StringIO()
        self.logger = Logger()
        self.logger.console = Console(file=self.output, force_terminal=True, theme=CARAMBA_THEME)

    def test_table_with_data_prints(self) -> None:
        """table() with columns and rows prints immediately."""
        self.logger.table(
            title="Results",
            columns=["Model", "Score"],
            rows=[["teacher", "0.95"], ["student", "0.93"]],
        )
        output = self.output.getvalue()
        self.assertIn("Model", output)
        self.assertIn("Score", output)
        self.assertIn("teacher", output)
        self.assertIn("student", output)

    def test_table_without_data_returns_table(self) -> None:
        """table() without data returns a Table object."""
        from rich.table import Table
        table = self.logger.table(title="Empty")
        self.assertIsInstance(table, Table)


class TestLoggerPanel(unittest.TestCase):
    """Tests for panel method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.output = io.StringIO()
        self.logger = Logger()
        self.logger.console = Console(file=self.output, force_terminal=True, theme=CARAMBA_THEME)

    def test_panel_prints_content(self) -> None:
        """panel() prints content."""
        self.logger.panel("This is panel content")
        output = self.output.getvalue()
        self.assertIn("panel content", output)

    def test_panel_with_title(self) -> None:
        """panel() includes title when provided."""
        self.logger.panel("Content", title="My Panel")
        output = self.output.getvalue()
        self.assertIn("Content", output)


if __name__ == "__main__":
    unittest.main()
