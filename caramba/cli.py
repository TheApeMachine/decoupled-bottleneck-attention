"""Command-line interface for the caramba system.

Designed as a minimal, intent-first CLI that's easy to use and understand.
Advanced optimization knobs are intentionally hidden—one of the platform's
core principles is self-optimization, auto-tuning, and auto-fitting.

Commands:
- compile: Parse → lower → validate a manifest without building
- run: Full experiment pipeline (upcycle + benchmarks + artifacts)
- paper: AI-assisted paper drafting from experiments
- review: AI-assisted paper review with experiment proposals
- research: Autonomous research loop (write → review → experiment → repeat)
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


@dataclass(frozen=True, slots=True)
class PaperCommand:
    """Command to draft/update a paper from a manifest."""

    manifest: Manifest
    manifest_path: Path
    output_dir: Path | None
    update: bool


@dataclass(frozen=True, slots=True)
class ReviewCommand:
    """Command to review a paper and propose improvements."""

    manifest: Manifest
    manifest_path: Path
    paper_dir: Path
    strictness: str


@dataclass(frozen=True, slots=True)
class ResearchCommand:
    """Command to run the autonomous research loop."""

    manifest: Manifest
    manifest_path: Path
    output_dir: Path | None
    max_iterations: int


class _Args(argparse.Namespace):
    """Typed namespace for CLI arguments."""

    command: str | None = None
    compile_manifest: Path | None = None
    print_plan: bool = False

    # Experiment command args
    experiment_manifest: Path | None = None
    group: str | None = None

    # Paper command args
    paper_manifest: Path | None = None
    paper_output_dir: Path | None = None
    paper_update: bool = False

    # Review command args
    review_manifest: Path | None = None
    review_paper_dir: Path | None = None
    review_strictness: str = "conference"

    # Research loop command args
    research_manifest: Path | None = None
    research_output_dir: Path | None = None
    research_max_iterations: int = 5

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

        # Paper command (AI-assisted paper drafting)
        paper_parser = subparsers.add_parser(
            "paper",
            help="Draft or update a paper using AI from experiment manifest.",
        )
        _ = paper_parser.add_argument(
            "paper_manifest",
            type=Path,
            metavar="manifest",
            help="Manifest path (.json, .yml, or .yaml) with paper configuration.",
        )
        _ = paper_parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            dest="paper_output_dir",
            help="Output directory for paper files. Defaults to artifacts/{name}/paper/",
        )
        _ = paper_parser.add_argument(
            "--update",
            action="store_true",
            default=False,
            dest="paper_update",
            help="Update existing paper with new experiment results.",
        )

        # Review command (AI-assisted paper review)
        review_parser = subparsers.add_parser(
            "review",
            help="Review a paper using AI and propose improvements/experiments.",
        )
        _ = review_parser.add_argument(
            "review_manifest",
            type=Path,
            metavar="manifest",
            help="Manifest path (.json, .yml, or .yaml).",
        )
        _ = review_parser.add_argument(
            "--paper-dir",
            type=Path,
            default=None,
            dest="review_paper_dir",
            help="Directory containing paper.tex. Defaults to artifacts/{name}/paper/",
        )
        _ = review_parser.add_argument(
            "--strictness",
            type=str,
            default="conference",
            dest="review_strictness",
            choices=["workshop", "conference", "journal", "top_venue"],
            help="Review strictness level.",
        )

        # Research loop command (autonomous research)
        research_parser = subparsers.add_parser(
            "research",
            help="Run autonomous research loop: write → review → experiment → repeat.",
        )
        _ = research_parser.add_argument(
            "research_manifest",
            type=Path,
            metavar="manifest",
            help="Manifest path (.json, .yml, or .yaml) with paper configuration.",
        )
        _ = research_parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            dest="research_output_dir",
            help="Output directory for research artifacts.",
        )
        _ = research_parser.add_argument(
            "--max-iterations",
            type=int,
            default=5,
            dest="research_max_iterations",
            help="Maximum research loop iterations (default: 5).",
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

    def parse_command(self, argv: list[str] | None = None) -> Command | ExperimentCommand | PaperCommand | ReviewCommand | ResearchCommand:
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
            case "paper":
                if args.paper_manifest is None:
                    raise ValueError("paper requires a manifest path.")
                manifest = self._load_and_validate_manifest(args.paper_manifest)
                if manifest.paper is None:
                    raise ValueError(
                        "Manifest does not have a 'paper' section. "
                        "Add paper configuration to enable AI-assisted drafting."
                    )
                return PaperCommand(
                    manifest=manifest,
                    manifest_path=args.paper_manifest,
                    output_dir=args.paper_output_dir,
                    update=args.paper_update,
                )
            case "review":
                if args.review_manifest is None:
                    raise ValueError("review requires a manifest path.")
                manifest = self._load_and_validate_manifest(args.review_manifest)

                # Determine paper directory
                if args.review_paper_dir:
                    paper_dir = args.review_paper_dir
                else:
                    paper_dir = Path("artifacts") / (manifest.name or "experiment") / "paper"

                return ReviewCommand(
                    manifest=manifest,
                    manifest_path=args.review_manifest,
                    paper_dir=paper_dir,
                    strictness=args.review_strictness,
                )
            case "research":
                if args.research_manifest is None:
                    raise ValueError("research requires a manifest path.")
                manifest = self._load_and_validate_manifest(args.research_manifest)
                if manifest.paper is None:
                    raise ValueError(
                        "Manifest does not have a 'paper' section. "
                        "Add paper configuration to enable the research loop."
                    )
                return ResearchCommand(
                    manifest=manifest,
                    manifest_path=args.research_manifest,
                    output_dir=args.research_output_dir,
                    max_iterations=args.research_max_iterations,
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
            case PaperCommand() as c:
                return c.manifest
            case ReviewCommand() as c:
                return c.manifest
            case ResearchCommand() as c:
                return c.manifest
            case CompileCommand():
                raise ValueError(
                    "compile is not supported via CLI.parse(); use `caramba compile` "
                    "via the console script entrypoint."
                )
            case _:
                raise ValueError(f"Unsupported command for parse(): {command}")


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

            case PaperCommand() as cmd:
                from caramba.paper import PaperDrafter

                paper_config = cmd.manifest.paper
                if paper_config is None:
                    logger.error("No paper configuration found in manifest")
                    return 1

                logger.header("Paper Drafter", paper_config.title)

                # Determine output directory
                if cmd.output_dir:
                    output_dir = cmd.output_dir
                else:
                    output_dir = Path("artifacts") / (cmd.manifest.name or "experiment") / paper_config.output_dir

                drafter = PaperDrafter(paper_config, output_dir)
                paper_path = drafter.draft_sync(
                    manifest=cmd.manifest,
                    manifest_path=cmd.manifest_path,
                )

                logger.success(f"Paper drafted: {paper_path}")
                logger.key_value({
                    "Output": str(paper_path),
                    "References": str(paper_path.parent / "references.bib"),
                })
                return 0

            case ReviewCommand() as cmd:
                from caramba.paper import PaperReviewer, ReviewConfig

                # Build review config
                review_config = cmd.manifest.review or ReviewConfig()
                review_config = ReviewConfig(
                    **{**review_config.model_dump(), "strictness": cmd.strictness}
                )

                logger.header("Paper Reviewer", f"Reviewing paper in {cmd.paper_dir}")

                reviewer = PaperReviewer(review_config, cmd.paper_dir)
                result = reviewer.review_sync(
                    manifest=cmd.manifest,
                    manifest_path=cmd.manifest_path,
                )

                logger.success("Review complete!")
                logger.key_value({
                    "Score": f"{result.overall_score:.1f}/10",
                    "Recommendation": result.recommendation.value,
                    "Strengths": len(result.strengths),
                    "Weaknesses": len(result.weaknesses),
                    "Proposed experiments": len(result.proposed_experiments),
                })

                if result.proposed_experiments:
                    logger.info("Proposed experiments:")
                    for exp in result.proposed_experiments:
                        logger.info(f"  - {exp.name}: {exp.rationale[:60]}...")

                return 0

            case ResearchCommand() as cmd:
                from caramba.paper import ResearchLoop, ResearchLoopConfig

                paper_config = cmd.manifest.paper
                if paper_config is None:
                    logger.error("No paper configuration found in manifest")
                    return 1

                review_config = cmd.manifest.review

                logger.header("Research Loop", f"🔄 {paper_config.title}")

                loop_config = ResearchLoopConfig(max_iterations=cmd.max_iterations)

                # Determine output directory
                if cmd.output_dir:
                    output_dir = cmd.output_dir
                else:
                    output_dir = Path("artifacts") / (cmd.manifest.name or "experiment") / paper_config.output_dir

                loop = ResearchLoop(
                    paper_config=paper_config,
                    review_config=review_config,
                    loop_config=loop_config,
                    output_dir=output_dir,
                )

                result = loop.run_sync(
                    manifest=cmd.manifest,
                    manifest_path=cmd.manifest_path,
                )

                if result.success:
                    logger.success(f"🎉 Research loop completed successfully!")
                else:
                    logger.warning(f"Research loop ended: {result.message}")

                logger.key_value({
                    "Iterations": len(result.iterations),
                    "Final score": f"{result.final_score:.1f}/10" if result.final_score else "N/A",
                    "Experiments run": result.total_experiments_run,
                    "Output": str(result.paper_path) if result.paper_path else "N/A",
                })

                return 0 if result.success else 1

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
