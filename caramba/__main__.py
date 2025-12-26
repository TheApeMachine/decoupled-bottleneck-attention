"""
__main__ provides the console-script entrypoint for the caramba package.

All commands are routed through the unified CLI in caramba.cli.main().
"""
from __future__ import annotations

import sys

from caramba.cli import main as cli_main


def main(argv: list[str] | None = None) -> None:
    """
    main is the entrypoint for the `caramba` console script.

    Routes everything through the unified CLI to ensure consistent behavior
    for all command types (compile, run, legacy).
    """
    exit_code = cli_main(argv)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
