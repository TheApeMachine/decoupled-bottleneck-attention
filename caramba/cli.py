"""
cli provides the command line interface for the caramba system.

It is designed to be a minimal, intent-first CLI that is easy to use and understand.
Advanced optimization knobs are intentionally hidden, given one of the core principles
of the platform is self-optimization, auto-tuning, and auto-fitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from caramba.config.mode import Mode
from caramba.config.model import ModelConfig, ModelType
from caramba.config.topology import TopologyConfig, TopologyType
from caramba.config.defaults import Defaults
from caramba.config.group import Group
from caramba.config.manifest import Manifest
from caramba.config.run import Run


class _Args(argparse.Namespace):
    entity: str = ""
    project: str = ""
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
        _ = self.add_argument(
            "--entity",
            type=str,
            default=None,
            help="The entity is a label for the institution or user that is running the experiments.",
            required=True,
        )
        _ = self.add_argument(
            "--project",
            type=str,
            default=None,
            help="The project is a label for the project the experiments belong to.",
            required=True,
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

    def parse(self, argv: list[str] | None = None) -> Manifest:
        """
        parse the CLI arguments and return a Manifest object.

        if --manifest is provided, it will ignore all other arguments,
        and serialize the manifest.json file onto the Manifest object.
        otherwise, it will use the arguments to construct a Manifest object.
        """
        args = self.parse_args(argv, namespace=_Args())

        if args.manifest.exists():
            return Manifest.from_path(args.manifest)

        return Manifest(
            version=1,
            name="",
            notes="",
            model=ModelConfig(
                type=ModelType.TRANSFORMER,
                topology=TopologyConfig(
                    type=TopologyType.STACKED,
                    layers=[],
                ),
            ),
            defaults=Defaults(
                wandb_entity=args.entity,
                wandb_project=args.project,
            ),
            groups=[
                Group(
                    name="default",
                    description="Default group",
                    data="",
                    runs=[
                        Run(
                            id="default",
                            mode=self._get_mode(args.mode),
                            exp="default",
                            seed=args.seed,
                            steps=1000,
                            expected={
                                "attn_mode": "standard",
                            },
                        ),
                    ],
                )
            ],
        )