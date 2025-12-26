"""Console-script entrypoint for the caramba package.

This allows running caramba as `python -m caramba` or via the console script.
All commands are routed through the unified CLI.
"""
from __future__ import annotations

import sys

from caramba.cli import main as cli_main


def main(argv: list[str] | None = None) -> None:
    """Entrypoint for the `caramba` console script.

    Routes everything through the unified CLI to ensure consistent behavior
    for all command types (compile, run, experiment).
    """
    exit_code = cli_main(argv)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
