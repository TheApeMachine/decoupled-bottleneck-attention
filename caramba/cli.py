"""Command-line interface for the caramba system.

Designed as a minimal, intent-first CLI that's easy to use and understand.
Advanced optimization knobs are intentionally hidden—one of the platform's
core principles is self-optimization, auto-tuning, and auto-fitting.

Commands:
- compile: Parse → lower → validate a manifest without building
- run: Full experiment pipeline (upcycle + benchmarks + artifacts)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from caramba.command import Command, CompileCommand, RunCommand
from caramba.compiler import Compiler
from caramba.config.manifest import Manifest
from caramba.config.mode import Mode
from caramba.console import logger


@dataclass(frozen=True, slots=True)
class ExperimentCommand:
    """Command to run a full experiment with benchmarks and artifact generation."""

    manifest: Manifest
    group: str | None


class _Args(argparse.Namespace):
    """Typed namespace for CLI arguments."""

    command: str | None = None
    compile_manifest: Path | None = None
    print_plan: bool = False

    # Experiment command args
    experiment_manifest: Path | None = None
    group: str | None = None

    entity: str | None = None
    project: str | None = None
    manifest: Path = Path("caramba/manifest.json")
    mode: str = "train"
    exp: str | None = None
    data: str | None = None
    ckpt: str | None = None
    resume: str | None = None
    seed: int = 1337


class CLI(argparse.ArgumentParser):
    """Minimal, intent-first command-line interface.

    Provides subcommands for compiling and running experiments, plus
    legacy arguments for backward compatibility.
    """

    def __init__(self) -> None:
        """Set up CLI with subcommands and legacy arguments."""
        super().__init__(
            prog="caramba",
            description="Caramba - A research platform for efficient AI.",
            epilog="For more information, see https://github.com/theapemachine/caramba",
        )

        _ = self.add_argument(
            "--version",
            action="version",
            version="%(prog)s 0.1.0",
            help="Show the version and exit.",
        )

        subparsers = self.add_subparsers(
            dest="command",
            parser_class=argparse.ArgumentParser,
        )

        # Compile command
        compile_parser = subparsers.add_parser(
            "compile",
            help="Compile a manifest (parse → lower → validate), without building.",
        )
        _ = compile_parser.add_argument(
            "compile_manifest",
            type=Path,
            metavar="manifest",
            help="Manifest path (.json, .yml, or .yaml).",
        )
        _ = compile_parser.add_argument(
            "--print-plan",
            action="store_true",
            default=False,
            dest="print_plan",
            help="Print the lowered graph/plan.",
        )

        # Run command (full experiment pipeline)
        run_parser = subparsers.add_parser(
            "run",
            help="Run a full experiment: upcycle + benchmarks + artifact generation.",
        )
        _ = run_parser.add_argument(
            "experiment_manifest",
            type=Path,
            metavar="manifest",
            help="Manifest path (.json, .yml, or .yaml).",
        )
        _ = run_parser.add_argument(
            "--group",
            type=str,
            default=None,
            help="Group name to run. If not specified, runs the first group.",
        )

        # Legacy arguments for backward compatibility
        _ = self.add_argument(
            "--entity",
            type=str,
            default=None,
            help="The entity is a label for the institution or user running the experiments.",
        )
        _ = self.add_argument(
            "--project",
            type=str,
            default=None,
            help="The project is a label for the project the experiments belong to.",
        )
        _ = self.add_argument(
            "--manifest",
            type=Path,
            default=Path("caramba/manifest.json"),
            help=" ".join(
                [
                    "Use the given manifest to run the system.",
                    "This also enables the 'expert' configuration mode.",
                    "If not provided, the default manifest will be used.",
                    "The default manifest is located at caramba/manifest.json.",
                ]
            ),
        )
        _ = self.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "sample", "chat"],
            help=" ".join(
                [
                    "Run mode. 'train' for training,",
                    "'sample' for sampling from a checkpoint,",
                    "'chat' for chatting with the model.",
                ]
            ),
        )
        _ = self.add_argument(
            "--exp",
            type=str,
            default=None,
            choices=[
                "baseline",
                "gqa",
                "bottleneck",
                "decoupled",
            ],
            help=" ".join(
                [
                    "Experiment preset (only used in 'train' mode).",
                    "Available presets:",
                    "- baseline: Standard attention",
                    "- gqa: GQA attention",
                    "- bottleneck: Bottleneck attention",
                    "- decoupled: Decoupled Bottleneck attention",
                ]
            ),
        )
        _ = self.add_argument(
            "--data",
            type=str,
            default=None,
            help=" ".join(
                [
                    "Dataset path (only used in 'train' mode).",
                    "If not provided, the dataset will be downloaded automatically.",
                ]
            ),
        )
        _ = self.add_argument(
            "--ckpt",
            type=str,
            default=None,
            help=" ".join(
                [
                    "Checkpoint path (only used in 'sample' and 'chat' modes).",
                    "If not provided, the latest checkpoint will be used.",
                ]
            ),
        )
        _ = self.add_argument(
            "--resume",
            type=str,
            default=None,
            help=" ".join(
                [
                    "Resume training from an explicit checkpoint path (only used in 'train' mode).",
                    "If not provided, the training will start from scratch.",
                ]
            ),
        )
        _ = self.add_argument(
            "--seed",
            type=int,
            default=1337,
            help=" ".join(
                [
                    "Random seed for reproducibility.",
                    "If not provided, the seed will be set to 1337.",
                ]
            ),
        )

    def _get_mode(self, mode: str) -> Mode:
        """Convert mode string to Mode enum."""
        match mode:
            case "train":
                return Mode.TRAIN
            case "sample":
                return Mode.SAMPLE
            case "chat":
                return Mode.CHAT
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    def _load_and_validate_manifest(self, manifest_path: Path) -> Manifest:
        """Load a manifest from path, lower it, and validate it."""
        compiler = Compiler()
        manifest = compiler.lowerer.lower_manifest(Manifest.from_path(manifest_path))
        compiler.validator.validate_manifest(manifest)
        return manifest

    def parse_command(self, argv: list[str] | None = None) -> Command | ExperimentCommand:
        """Parse CLI arguments into a typed command payload."""
        args = self.parse_args(argv, namespace=_Args())

        match args.command:
            case "compile":
                if args.compile_manifest is None:
                    raise ValueError("compile requires a manifest path.")
                manifest = self._load_and_validate_manifest(args.compile_manifest)
                return CompileCommand(
                    manifest=manifest,
                    print_plan=bool(args.print_plan),
                )
            case "run":
                if args.experiment_manifest is None:
                    raise ValueError("run requires a manifest path.")
                manifest = self._load_and_validate_manifest(args.experiment_manifest)
                return ExperimentCommand(
                    manifest=manifest,
                    group=args.group,
                )
            case None:
                manifest = self._parse_run_manifest(args)
                Compiler().validator.validate_manifest(manifest)
                return RunCommand(manifest=manifest)
            case _:
                raise ValueError(f"Invalid command: {args.command}")

    def _parse_run_manifest(self, args: _Args) -> Manifest:
        """Build or load a manifest for the legacy run command."""
        if args.manifest.exists():
            return Compiler().lowerer.lower_manifest(Manifest.from_path(args.manifest))

        raise ValueError(
            f"Manifest file not found: {args.manifest}. Provide --manifest PATH to "
            "an existing manifest file."
        )

    def parse(self, argv: list[str] | None = None) -> Manifest:
        """Parse CLI arguments and return a Manifest object.

        If --manifest is provided, ignores all other arguments and loads
        the manifest from the file.
        """
        command = self.parse_command(argv)
        match command:
            case RunCommand() as c:
                return c.manifest
            case ExperimentCommand() as c:
                return c.manifest
            case CompileCommand():
                raise ValueError(
                    "compile is not supported via CLI.parse(); use `caramba compile` "
                    "via the console script entrypoint."
                )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Returns exit code (0 for success, non-zero for failure).
    """
    cli = CLI()

    try:
        command = cli.parse_command(argv)

        match command:
            case CompileCommand() as cmd:
                if cmd.print_plan:
                    from caramba.compiler.plan import Planner

                    logger.log(Planner().format(cmd.manifest))
                logger.success("Manifest compiled successfully")
                return 0

            case ExperimentCommand() as cmd:
                from caramba.experiment import ExperimentRunner

                runner = ExperimentRunner(cmd.manifest)
                artifacts = runner.run(cmd.group)
                logger.success("Experiment complete!")
                logger.artifacts_summary(artifacts)
                return 0

            case RunCommand() as cmd:
                # Legacy run behavior
                from caramba.experiment import ExperimentRunner

                runner = ExperimentRunner(cmd.manifest)
                runner.run()
                return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
