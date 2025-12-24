"""
__main__ provides the console-script entrypoint for the caramba package.
"""
from __future__ import annotations

import sys
import traceback

from caramba.cli import CLI
from caramba.command import CompileCommand, RunCommand
from caramba.compiler.plan import format_manifest_plan
from caramba.config.mode import Mode
from caramba.trainer import Trainer


def main(argv: list[str] | None = None) -> None:
    """
    main is the entrypoint for the `caramba` console script.
    """
    try:
        command = CLI().parse_command(argv)

        match command:
            case CompileCommand() as c:
                if c.print_plan:
                    print(format_manifest_plan(c.manifest))
                return
            case RunCommand() as c:
                intent = c.manifest
            case _:
                raise ValueError(f"Invalid command payload: {type(command)!r}")

        for group in intent.groups:
            for run in group.runs:
                match run.mode:
                    case Mode.TRAIN:
                        Trainer(
                            manifest=intent,
                        ).run()
                    case _:
                        raise ValueError(f"Invalid mode: {run.mode}")
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
        if code == 0:
            raise
        sys.exit(code)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        print(f"details: {e!r}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print("runtime error while running caramba.", file=sys.stderr)
        print(f"details: {e!r}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

