#!/usr/bin/env python3
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    # Thin, stable entrypoint. All implementation lives under production/.
    from production.cli import parse_args, run

    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


