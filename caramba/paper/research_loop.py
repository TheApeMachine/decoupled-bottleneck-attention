"""Autonomous research loop: write → review → experiment → repeat.

The ResearchLoop coordinates the complete autonomous research cycle:
1. Write/update paper draft
2. Review paper and identify gaps
3. If style fixes needed: update paper
4. If new experiments needed: generate manifests and run them
5. Update paper with new results
6. Repeat until approved or max iterations

This is the heart of the autonomous research system.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from caramba.console import logger

if TYPE_CHECKING:
    from caramba.config.manifest import Manifest
    from caramba.config.paper import PaperConfig
    from caramba.paper.review import ReviewConfig, ReviewResult


class LoopAction(str, Enum):
    """Actions taken during the research loop."""

    WRITE_DRAFT = "write_draft"
    UPDATE_DRAFT = "update_draft"
    REVIEW = "review"
    STYLE_FIX = "style_fix"
    RUN_EXPERIMENT = "run_experiment"
    APPROVE = "approve"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"


@dataclass
class LoopIteration:
    """Record of a single iteration in the research loop."""

    iteration: int
    action: LoopAction
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)
    review_score: float | None = None
    experiments_proposed: int = 0
    experiments_run: int = 0


@dataclass
class ResearchLoopResult:
    """Final result of the research loop."""

    success: bool
    final_action: LoopAction
    iterations: list[LoopIteration]
    paper_path: Path | None
    final_score: float | None
    total_experiments_run: int
    message: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_action": self.final_action.value,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "action": it.action.value,
                    "timestamp": it.timestamp,
                    "review_score": it.review_score,
                    "experiments_proposed": it.experiments_proposed,
                    "experiments_run": it.experiments_run,
                    "details": it.details,
                }
                for it in self.iterations
            ],
            "paper_path": str(self.paper_path) if self.paper_path else None,
            "final_score": self.final_score,
            "total_experiments_run": self.total_experiments_run,
            "message": self.message,
        }


@dataclass
class ResearchLoopConfig:
    """Configuration for the research loop."""

    # Iteration limits
    max_iterations: int = 5
    max_experiments_per_iteration: int = 2
    max_total_experiments: int = 5

    # Approval thresholds
    min_score_to_approve: float = 7.5
    auto_approve_score: float = 9.0

    # Behavior
    auto_run_experiments: bool = True
    require_human_approval_for_experiments: bool = False
    save_all_versions: bool = True

    # Timeouts (seconds)
    experiment_timeout: int = 3600  # 1 hour per experiment
    review_timeout: int = 300  # 5 minutes for review


class ResearchLoop:
    """Autonomous research loop coordinator.

    Orchestrates the complete cycle of writing, reviewing, experimenting,
    and iterating until the paper is approved or limits are reached.

    Usage:
        loop = ResearchLoop(paper_config, review_config)
        result = await loop.run(manifest)
    """

    def __init__(
        self,
        paper_config: "PaperConfig",
        review_config: "ReviewConfig | None" = None,
        loop_config: ResearchLoopConfig | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        """Initialize the research loop.

        Args:
            paper_config: Configuration for paper drafting.
            review_config: Configuration for paper review.
            loop_config: Configuration for the loop itself.
            output_dir: Base output directory.
        """
        from caramba.paper.review import ReviewConfig

        self.paper_config = paper_config
        self.review_config = review_config or ReviewConfig()
        self.loop_config = loop_config or ResearchLoopConfig()

        if output_dir is None:
            output_dir = Path("artifacts") / paper_config.output_dir
        self.output_dir = Path(output_dir)

        self.iterations: list[LoopIteration] = []
        self.total_experiments_run = 0

    async def run(
        self,
        manifest: "Manifest | None" = None,
        manifest_path: Path | str | None = None,
        experiment_results: dict[str, Any] | None = None,
        artifacts: dict[str, Path] | None = None,
    ) -> ResearchLoopResult:
        """Run the complete research loop.

        Args:
            manifest: The base experiment manifest.
            manifest_path: Path to the manifest file.
            experiment_results: Initial experiment results.
            artifacts: Initial artifacts.

        Returns:
            ResearchLoopResult with the outcome of the loop.
        """
        logger.header("Research Loop", f"🔄 Starting autonomous research cycle")
        logger.key_value({
            "Paper": self.paper_config.title,
            "Max iterations": self.loop_config.max_iterations,
            "Min score to approve": self.loop_config.min_score_to_approve,
        })

        self.output_dir.mkdir(parents=True, exist_ok=True)

        current_results = experiment_results or {}
        current_artifacts = artifacts or {}
        current_manifest_path = Path(manifest_path) if manifest_path else None

        paper_path: Path | None = None
        final_score: float | None = None

        for iteration in range(1, self.loop_config.max_iterations + 1):
            logger.header("Iteration", f"{iteration}/{self.loop_config.max_iterations}")

            try:
                # Step 1: Write or update paper
                paper_path = await self._write_or_update_paper(
                    iteration=iteration,
                    manifest=manifest,
                    manifest_path=current_manifest_path,
                    experiment_results=current_results,
                    artifacts=current_artifacts,
                )

                # Step 2: Review paper
                review = await self._review_paper(
                    iteration=iteration,
                    paper_path=paper_path,
                    manifest=manifest,
                    manifest_path=current_manifest_path,
                    experiment_results=current_results,
                )

                final_score = review.overall_score

                # Step 3: Check if approved
                if review.overall_score >= self.loop_config.auto_approve_score:
                    logger.success(f"🎉 Paper auto-approved with score {review.overall_score:.1f}!")
                    self._record_iteration(
                        iteration, LoopAction.APPROVE, review_score=review.overall_score
                    )
                    return self._build_result(
                        success=True,
                        final_action=LoopAction.APPROVE,
                        paper_path=paper_path,
                        final_score=final_score,
                        message=f"Paper approved with score {review.overall_score:.1f}/10",
                    )

                if (
                    review.overall_score >= self.loop_config.min_score_to_approve
                    and review.style_fixes_only
                ):
                    logger.success(f"✓ Paper approved with minor fixes (score {review.overall_score:.1f})")
                    # Do one more style fix iteration then approve
                    await self._apply_style_fixes(iteration, review, paper_path)
                    self._record_iteration(
                        iteration, LoopAction.APPROVE, review_score=review.overall_score
                    )
                    return self._build_result(
                        success=True,
                        final_action=LoopAction.APPROVE,
                        paper_path=paper_path,
                        final_score=final_score,
                        message=f"Paper approved after style fixes (score {review.overall_score:.1f}/10)",
                    )

                # Step 4: Handle needed changes
                if review.needs_new_experiments:
                    if self.total_experiments_run >= self.loop_config.max_total_experiments:
                        logger.warning(
                            f"Max experiments reached ({self.loop_config.max_total_experiments}). "
                            "Stopping loop."
                        )
                        break

                    # Run proposed experiments
                    new_results, new_artifacts = await self._run_proposed_experiments(
                        iteration=iteration,
                        review=review,
                        base_manifest=manifest,
                    )

                    # Merge results
                    current_results = {**current_results, **new_results}
                    current_artifacts = {**current_artifacts, **new_artifacts}

                    self._record_iteration(
                        iteration,
                        LoopAction.RUN_EXPERIMENT,
                        review_score=review.overall_score,
                        experiments_proposed=len(review.proposed_experiments),
                        experiments_run=len(new_results),
                    )
                else:
                    # Just style fixes needed
                    await self._apply_style_fixes(iteration, review, paper_path)
                    self._record_iteration(
                        iteration,
                        LoopAction.STYLE_FIX,
                        review_score=review.overall_score,
                    )

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                self._record_iteration(
                    iteration, LoopAction.ERROR, details={"error": str(e)}
                )
                return self._build_result(
                    success=False,
                    final_action=LoopAction.ERROR,
                    paper_path=paper_path,
                    final_score=final_score,
                    message=f"Loop failed at iteration {iteration}: {e}",
                )

        # Max iterations reached
        logger.warning(f"Max iterations ({self.loop_config.max_iterations}) reached")
        return self._build_result(
            success=False,
            final_action=LoopAction.MAX_ITERATIONS,
            paper_path=paper_path,
            final_score=final_score,
            message=f"Max iterations reached. Final score: {final_score:.1f}/10" if final_score else "Max iterations reached",
        )

    def run_sync(
        self,
        manifest: "Manifest | None" = None,
        manifest_path: Path | str | None = None,
        experiment_results: dict[str, Any] | None = None,
        artifacts: dict[str, Path] | None = None,
    ) -> ResearchLoopResult:
        """Synchronous wrapper for run()."""
        return asyncio.run(
            self.run(
                manifest=manifest,
                manifest_path=manifest_path,
                experiment_results=experiment_results,
                artifacts=artifacts,
            )
        )

    async def _write_or_update_paper(
        self,
        iteration: int,
        manifest: "Manifest | None",
        manifest_path: Path | None,
        experiment_results: dict[str, Any],
        artifacts: dict[str, Path],
    ) -> Path:
        """Write initial draft or update existing paper."""
        from caramba.paper.drafter import PaperDrafter

        paper_path = self.output_dir / "paper.tex"
        action = LoopAction.UPDATE_DRAFT if paper_path.exists() else LoopAction.WRITE_DRAFT

        logger.step(1, 4, f"{'Updating' if paper_path.exists() else 'Writing'} paper draft")

        drafter = PaperDrafter(self.paper_config, self.output_dir)
        result_path = await drafter.draft(
            manifest=manifest,
            manifest_path=manifest_path,
            experiment_results=experiment_results,
            artifacts=artifacts,
        )

        # Save version if configured
        if self.loop_config.save_all_versions:
            version_dir = self.output_dir / "versions"
            version_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(result_path, version_dir / f"paper_iter{iteration}_{timestamp}.tex")

        self._record_iteration(iteration, action)
        return result_path

    async def _review_paper(
        self,
        iteration: int,
        paper_path: Path,
        manifest: "Manifest | None",
        manifest_path: Path | None,
        experiment_results: dict[str, Any],
    ) -> "ReviewResult":
        """Review the current paper draft."""
        from caramba.paper.reviewer import PaperReviewer

        logger.step(2, 4, "Reviewing paper")

        reviewer = PaperReviewer(self.review_config, self.output_dir)
        review = await reviewer.review(
            manifest=manifest,
            manifest_path=manifest_path,
            experiment_results=experiment_results,
        )

        # Log review summary
        logger.key_value({
            "Score": f"{review.overall_score:.1f}/10",
            "Recommendation": review.recommendation.value,
            "Strengths": len(review.strengths),
            "Weaknesses": len(review.weaknesses),
            "Proposed experiments": len(review.proposed_experiments),
        })

        # Save review
        review_path = self.output_dir / f"review_iter{iteration}.json"
        review_path.write_text(
            json.dumps({
                "score": review.overall_score,
                "recommendation": review.recommendation.value,
                "summary": review.summary,
                "strengths": review.strengths,
                "weaknesses": review.weaknesses,
                "proposed_experiments": [
                    {"name": e.name, "rationale": e.rationale}
                    for e in review.proposed_experiments
                ],
            }, indent=2),
            encoding="utf-8",
        )

        return review

    async def _run_proposed_experiments(
        self,
        iteration: int,
        review: "ReviewResult",
        base_manifest: "Manifest | None",
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """Run experiments proposed by the reviewer."""
        logger.step(3, 4, f"Running {len(review.proposed_experiments)} proposed experiments")

        all_results: dict[str, Any] = {}
        all_artifacts: dict[str, Path] = {}

        # Limit experiments per iteration
        experiments_to_run = review.proposed_experiments[
            : self.loop_config.max_experiments_per_iteration
        ]

        for i, experiment in enumerate(experiments_to_run):
            if self.total_experiments_run >= self.loop_config.max_total_experiments:
                logger.warning("Max total experiments reached, skipping remaining")
                break

            logger.info(f"Running experiment {i+1}/{len(experiments_to_run)}: {experiment.name}")

            # Check if manifest was generated
            manifest_path = self.output_dir / f"proposed_{experiment.name.lower().replace(' ', '_')}.yml"

            if manifest_path.exists():
                try:
                    results, artifacts = await self._run_single_experiment(
                        manifest_path, experiment.name
                    )
                    all_results[experiment.name] = results
                    all_artifacts.update(artifacts)
                    self.total_experiments_run += 1
                except Exception as e:
                    logger.error(f"Experiment '{experiment.name}' failed: {e}")
            else:
                logger.warning(f"Manifest not found for '{experiment.name}', skipping")

        return all_results, all_artifacts

    async def _run_single_experiment(
        self,
        manifest_path: Path,
        experiment_name: str,
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """Run a single experiment from its manifest."""
        # Import here to avoid circular imports
        from caramba.experiment.runner import run_experiment

        logger.info(f"Executing experiment from {manifest_path}")

        # This is synchronous but we wrap it
        # In a real implementation, you might want to use subprocess or async execution
        try:
            artifacts = run_experiment(manifest_path)

            # Build results summary
            results = {
                "experiment_name": experiment_name,
                "manifest_path": str(manifest_path),
                "artifacts": {name: str(path) for name, path in artifacts.items()},
                "completed": True,
            }

            # Try to load report.json for metrics
            for name, path in artifacts.items():
                if name == "report.json" and path.exists():
                    with open(path) as f:
                        report = json.load(f)
                        results["metrics"] = report.get("summary", {})

            return results, artifacts

        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            return {"error": str(e), "completed": False}, {}

    async def _apply_style_fixes(
        self,
        iteration: int,
        review: "ReviewResult",
        paper_path: Path,
    ) -> None:
        """Apply style fixes based on review feedback."""
        logger.step(4, 4, "Applying style fixes")

        # The next iteration's paper update will incorporate the review feedback
        # We could also have a dedicated style-fix agent here
        logger.info(f"Style fixes will be applied in next paper update")
        logger.info(f"Weaknesses to address: {', '.join(review.weaknesses[:3])}")

    def _record_iteration(
        self,
        iteration: int,
        action: LoopAction,
        review_score: float | None = None,
        experiments_proposed: int = 0,
        experiments_run: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an iteration for the history."""
        self.iterations.append(
            LoopIteration(
                iteration=iteration,
                action=action,
                timestamp=datetime.now().isoformat(),
                details=details or {},
                review_score=review_score,
                experiments_proposed=experiments_proposed,
                experiments_run=experiments_run,
            )
        )

    def _build_result(
        self,
        success: bool,
        final_action: LoopAction,
        paper_path: Path | None,
        final_score: float | None,
        message: str,
    ) -> ResearchLoopResult:
        """Build the final result object."""
        result = ResearchLoopResult(
            success=success,
            final_action=final_action,
            iterations=self.iterations,
            paper_path=paper_path,
            final_score=final_score,
            total_experiments_run=self.total_experiments_run,
            message=message,
        )

        # Save result to file
        result_path = self.output_dir / "research_loop_result.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

        return result


# ============================================================================
# Convenience Functions
# ============================================================================


async def run_research_loop(
    paper_config: "PaperConfig",
    manifest: "Manifest | None" = None,
    manifest_path: Path | str | None = None,
    review_config: "ReviewConfig | None" = None,
    loop_config: ResearchLoopConfig | None = None,
    output_dir: Path | str | None = None,
    experiment_results: dict[str, Any] | None = None,
    artifacts: dict[str, Path] | None = None,
) -> ResearchLoopResult:
    """Run the complete research loop.

    Args:
        paper_config: Configuration for paper writing.
        manifest: Base experiment manifest.
        manifest_path: Path to manifest file.
        review_config: Configuration for review.
        loop_config: Configuration for the loop.
        output_dir: Output directory.
        experiment_results: Initial experiment results.
        artifacts: Initial artifacts.

    Returns:
        ResearchLoopResult with the outcome.
    """
    loop = ResearchLoop(
        paper_config=paper_config,
        review_config=review_config,
        loop_config=loop_config,
        output_dir=output_dir,
    )

    return await loop.run(
        manifest=manifest,
        manifest_path=manifest_path,
        experiment_results=experiment_results,
        artifacts=artifacts,
    )


def run_research_loop_sync(
    paper_config: "PaperConfig",
    manifest: "Manifest | None" = None,
    manifest_path: Path | str | None = None,
    review_config: "ReviewConfig | None" = None,
    loop_config: ResearchLoopConfig | None = None,
    output_dir: Path | str | None = None,
    experiment_results: dict[str, Any] | None = None,
    artifacts: dict[str, Path] | None = None,
) -> ResearchLoopResult:
    """Synchronous version of run_research_loop()."""
    return asyncio.run(
        run_research_loop(
            paper_config=paper_config,
            manifest=manifest,
            manifest_path=manifest_path,
            review_config=review_config,
            loop_config=loop_config,
            output_dir=output_dir,
            experiment_results=experiment_results,
            artifacts=artifacts,
        )
    )
