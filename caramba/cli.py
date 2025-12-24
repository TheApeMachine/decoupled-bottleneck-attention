"""
cli provides the command line interface for the caramba system.

It is designed to be a minimal, intent-first CLI that is easy to use and understand.
Advanced optimization knobs are intentionally hidden, given one of the core principles
of the platform is self-optimization, auto-tuning, and auto-fitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from caramba.command import Command, CompileCommand, RunCommand
from caramba.config.mode import Mode
from caramba.config.manifest import Manifest
from caramba.compiler import lower_manifest, validate_manifest


class _Args(argparse.Namespace):
    command: str | None = None
    compile_manifest: Path | None = None
    print_plan: bool = False

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
    """
    CLI is a minimal, intent-first command line interface.
    """
    def __init__(self) -> None:
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

        _ = self.add_argument(
            "--entity",
            type=str,
            default=None,
            help="The entity is a label for the institution or user that is running the experiments.",
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
            help=" ".join([
                "Use the given manifest to run the system.",
                "This also enables the 'expert' configuration mode.",
                "If not provided, the default manifest will be used.",
                "The default manifest is located at caramba/manifest.json.",
            ]),
        )
        _ = self.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "sample", "chat"],
            help=" ".join([
                "Run mode. 'train' for training,",
                "'sample' for sampling from a checkpoint,",
                "'chat' for chatting with the model.",
            ]),
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
            help=" ".join([
                "Experiment preset (only used in 'train' mode).",
                "Available presets:",
                "- baseline: Standard attention",
                "- gqa: GQA attention",
                "- bottleneck: Bottleneck attention",
                "- decoupled: Decoupled Bottleneck attention",
            ]),
        )
        _ = self.add_argument(
            "--data",
            type=str,
            default=None,
            help=" ".join([
                "Dataset path (only used in 'train' mode).",
                "If not provided, the dataset will be downloaded automatically.",
            ])
        )
        _ = self.add_argument(
            "--ckpt",
            type=str,
            default=None,
            help=" ".join([
                "Checkpoint path (only used in 'sample' and 'chat' modes).",
                "If not provided, the latest checkpoint will be used.",
            ]),
        )
        _ = self.add_argument(
            "--resume",
            type=str,
            default=None,
            help=" ".join([
                "Resume training from an explicit checkpoint path (only used in 'train' mode).",
                "If not provided, the training will start from scratch.",
            ]),
        )
        _ = self.add_argument(
            "--seed",
            type=int,
            default=1337,
            help=" ".join([
                "Random seed for reproducibility.",
                "If not provided, the seed will be set to 1337.",
            ]),
        )

    def _get_mode(self, mode: str) -> Mode:
        """
        get the mode from the string.
        """
        match mode:
            case "train":
                return Mode.TRAIN
            case "sample":
                return Mode.SAMPLE
            case "chat":
                return Mode.CHAT
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    def parse_command(self, argv: list[str] | None = None) -> Command:
        """
        parse_command parses the CLI arguments into a typed command payload.
        """
        args = self.parse_args(argv, namespace=_Args())

        match args.command:
            case "compile":
                if args.compile_manifest is None:
                    raise ValueError("compile requires a manifest path.")
                manifest = lower_manifest(Manifest.from_path(args.compile_manifest))
                validate_manifest(manifest)
                return CompileCommand(
                    manifest=manifest,
                    print_plan=bool(args.print_plan),
                )
            case None:
                manifest = self._parse_run_manifest(args)
                validate_manifest(manifest)
                return RunCommand(manifest=manifest)
            case _:
                raise ValueError(f"Invalid command: {args.command}")

    def _parse_run_manifest(self, args: _Args) -> Manifest:
        """
        _parse_run_manifest builds or loads a manifest for the run command.
        """
        if args.manifest.exists():
            return lower_manifest(Manifest.from_path(args.manifest))

        raise ValueError(
            f"Manifest file not found: {args.manifest}. Provide --manifest PATH to "
            "an existing manifest file."
        )

    def parse(self, argv: list[str] | None = None) -> Manifest:
        """
        parse the CLI arguments and return a Manifest object.

        if --manifest is provided, it will ignore all other arguments,
        and serialize the manifest.json file onto the Manifest object.
        otherwise, it will use the arguments to construct a Manifest object.
        """
        command = self.parse_command(argv)
        match command:
            case RunCommand() as c:
                return c.manifest
            case CompileCommand():
                raise ValueError(
                    "compile is not supported via CLI.parse(); use `caramba compile` "
                    "via the console script entrypoint."
                )