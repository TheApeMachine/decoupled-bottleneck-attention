"""Paper reviewer orchestration using OpenAI Agent SDK.

The PaperReviewer coordinates an AI agent to review academic papers,
identify weaknesses, and propose improvements including new experiments.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents import Agent, Runner
from agents.tool import Tool

from caramba.console import logger
from caramba.paper.review import ReviewConfig, ReviewResult, ReviewActionType
from caramba.paper.reviewer_tools import REVIEWER_TOOLS
from caramba.paper.tools import PaperState, set_state

if TYPE_CHECKING:
    from caramba.config.manifest import Manifest


# ============================================================================
# System Instructions for Different Personas
# ============================================================================

REVIEWER_PERSONAS = {
    "senior_researcher": """You are a senior ML researcher with 15+ years of experience reviewing papers for top venues (NeurIPS, ICML, ICLR). You are thorough, fair, and constructive. You focus on:
- Novelty and significance of contributions
- Soundness of methodology
- Completeness of experiments
- Clarity of presentation
- Proper contextualization in related work

You provide actionable feedback and, when needed, specific suggestions for additional experiments.""",

    "methodology_expert": """You are a methodology expert who specializes in experimental design and statistical rigor. You focus on:
- Experimental design and controls
- Statistical significance and proper evaluation
- Reproducibility concerns
- Ablation studies and analysis
- Potential confounds and biases

You are particularly attentive to claims that lack proper experimental support.""",

    "practitioner": """You are an ML practitioner who evaluates papers for practical relevance and reproducibility. You focus on:
- Practical applicability of the work
- Computational requirements and efficiency claims
- Ease of reproduction
- Real-world deployment considerations
- Missing baselines that practitioners would expect

You emphasize experiments that would help practitioners decide whether to adopt this work.""",
}

BASE_INSTRUCTIONS = """## Your Role

You are reviewing an academic paper and providing constructive feedback. Your goal is to help strengthen the paper by identifying:
1. Weaknesses in argumentation or presentation
2. Gaps in experimental evidence
3. Missing related work or context
4. Opportunities for additional experiments

## Review Process

1. First, analyze the paper structure using `analyze_paper_structure()`
2. Read key sections using `read_paper_section()`
3. Check experimental claims using `check_experimental_claims()`
4. Check citation coverage for key topics using `check_citation_coverage()`
5. Review available experiment results using `get_experiment_results_summary()`

## When Proposing Experiments

If you identify gaps that require new experiments:
1. Use `propose_experiment()` to formally propose each experiment
2. Explain clearly why existing results don't address the gap
3. Use `generate_experiment_manifest()` to create runnable configs

## Submitting Your Review

After your analysis, use `submit_review()` to provide:
- Overall score (0-10)
- Recommendation (approve/style_fix/new_experiment/major_revision)
- Summary of your review
- List of strengths and weaknesses

## Review Standards

{strictness_guidelines}

## Important Guidelines

- Be constructive, not just critical
- Prioritize issues by importance
- For each weakness, suggest how to address it
- Only propose experiments that are truly necessary
- Consider the paper's scope and contribution claims
"""

STRICTNESS_GUIDELINES = {
    "workshop": """Workshop papers should present interesting ideas even if not fully developed.
- Novel ideas are more important than complete experiments
- Accept if the core contribution is clear and promising
- Score 6+ is acceptable for publication""",

    "conference": """Conference papers should have solid contributions with adequate evaluation.
- Expect thorough experiments comparing to relevant baselines
- Ablation studies are appreciated but not always required
- Score 7+ is the typical acceptance threshold""",

    "journal": """Journal papers should be comprehensive with extensive evaluation.
- Expect thorough experiments, ablations, and analysis
- Multiple datasets/settings should be evaluated
- Score 8+ is expected for acceptance""",

    "top_venue": """Top venue papers must be exceptional in novelty and rigor.
- Novel, significant contribution is mandatory
- Comprehensive experiments with strong baselines
- Thorough ablation and analysis required
- Score 8.5+ for acceptance, most papers are rejected""",
}


# ============================================================================
# PaperReviewer Class
# ============================================================================


class PaperReviewer:
    """Orchestrates AI-assisted paper review.

    Uses the OpenAI Agent SDK to review papers, identify issues,
    and propose experiments to strengthen the work.

    Usage:
        config = ReviewConfig(strictness="conference")
        reviewer = PaperReviewer(config)
        result = await reviewer.review(paper_dir)
    """

    def __init__(
        self,
        config: ReviewConfig,
        output_dir: Path | str | None = None,
    ) -> None:
        """Initialize the paper reviewer.

        Args:
            config: Review configuration.
            output_dir: Directory containing paper.tex and outputs.
        """
        self.config = config

        if output_dir is None:
            output_dir = Path("artifacts/paper")
        self.output_dir = Path(output_dir)

        # Build instructions
        persona = REVIEWER_PERSONAS.get(
            config.reviewer_persona, REVIEWER_PERSONAS["senior_researcher"]
        )
        strictness = STRICTNESS_GUIDELINES.get(
            config.strictness, STRICTNESS_GUIDELINES["conference"]
        )

        instructions = persona + "\n\n" + BASE_INSTRUCTIONS.format(
            strictness_guidelines=strictness
        )

        if config.custom_instructions:
            instructions += f"\n\n## Additional Instructions\n\n{config.custom_instructions}"

        # Create the agent
        tools: list[Tool] = list(REVIEWER_TOOLS)  # type: ignore[assignment]
        self.agent = Agent(
            name="Paper Reviewer",
            instructions=instructions,
            tools=tools,
            model=config.model,
        )

    async def review(
        self,
        paper_dir: Path | str | None = None,
        manifest: "Manifest | None" = None,
        manifest_path: Path | str | None = None,
        experiment_results: dict[str, Any] | None = None,
    ) -> ReviewResult:
        """Review a paper and return structured feedback.

        Args:
            paper_dir: Directory containing paper.tex.
            manifest: The experiment manifest (optional).
            manifest_path: Path to the manifest file.
            experiment_results: Dict of experiment results/metrics.

        Returns:
            ReviewResult with comments and proposed experiments.
        """
        if paper_dir:
            self.output_dir = Path(paper_dir)

        logger.header("Paper Reviewer", f"Reviewing paper in {self.output_dir}")

        # Check that paper exists
        tex_path = self.output_dir / "paper.tex"
        if not tex_path.exists():
            raise FileNotFoundError(f"No paper.tex found in {self.output_dir}")

        # Set up state for tools (reuse from drafter tools)
        from caramba.config.paper import PaperConfig

        # Create a minimal paper config for state
        paper_config = PaperConfig(title="Review Target")

        state = PaperState(
            output_dir=self.output_dir,
            paper_config=paper_config,
            manifest_path=Path(manifest_path) if manifest_path else None,
            experiment_results=experiment_results,
        )
        set_state(state)

        # Build the review task
        task = self._build_review_prompt()

        # Run the agent
        logger.info(f"Running review agent with {self.config.model}...")

        try:
            result = await Runner.run(self.agent, input=task)
            logger.success("Review complete!")

        except Exception as e:
            logger.error(f"Review failed: {e}")
            raise

        # Parse the review result from state
        review_data = getattr(state, "_review_result", None)

        if review_data:
            return self._parse_review_result(review_data)
        else:
            # Try to load from file
            review_path = self.output_dir / "review.json"
            if review_path.exists():
                review_data = json.loads(review_path.read_text())
                return self._parse_review_result(review_data)

        # Fallback: create result from agent output
        return ReviewResult(
            overall_score=5.0,
            recommendation=ReviewActionType.MAJOR_REVISION,
            summary=result.final_output or "Review completed but structured output not captured.",
            comments=[],
            strengths=[],
            weaknesses=[],
            proposed_experiments=[],
        )

    def review_sync(
        self,
        paper_dir: Path | str | None = None,
        manifest: "Manifest | None" = None,
        manifest_path: Path | str | None = None,
        experiment_results: dict[str, Any] | None = None,
    ) -> ReviewResult:
        """Synchronous wrapper for review()."""
        return asyncio.run(
            self.review(
                paper_dir=paper_dir,
                manifest=manifest,
                manifest_path=manifest_path,
                experiment_results=experiment_results,
            )
        )

    def _build_review_prompt(self) -> str:
        """Build the review task prompt."""
        focus_areas = []
        if self.config.check_methodology:
            focus_areas.append("methodology and approach")
        if self.config.check_experiments:
            focus_areas.append("experimental design and coverage")
        if self.config.check_results:
            focus_areas.append("results and their interpretation")
        if self.config.check_writing:
            focus_areas.append("writing quality and clarity")
        if self.config.check_citations:
            focus_areas.append("related work and citations")

        prompt = f"""Please review the paper located at {self.output_dir}/paper.tex

## Review Focus Areas
{chr(10).join(f"- {area}" for area in focus_areas)}

## Review Standards
- Strictness level: {self.config.strictness}
- Minimum score to approve: {self.config.min_score_to_approve}

## Instructions

1. Start by analyzing the paper structure
2. Read and evaluate each major section
3. Check that experimental claims are supported
4. Verify citation coverage for key topics
5. Identify any gaps that need addressing

If you find issues that require new experiments:
- Propose up to {self.config.max_proposed_experiments} experiments
- Generate manifests for proposed experiments
- Prioritize experiments by impact

Finally, submit your complete review with:
- Overall score (0-10)
- Clear recommendation
- Strengths and weaknesses
- Summary of findings

Begin your review now."""

        return prompt

    def _parse_review_result(self, data: dict) -> ReviewResult:
        """Parse review data into ReviewResult model."""
        from caramba.paper.review import ProposedExperiment

        # Map recommendation string to enum
        rec_map = {
            "approve": ReviewActionType.APPROVE,
            "style_fix": ReviewActionType.STYLE_FIX,
            "clarification": ReviewActionType.CLARIFICATION,
            "new_experiment": ReviewActionType.NEW_EXPERIMENT,
            "major_revision": ReviewActionType.MAJOR_REVISION,
        }

        recommendation = rec_map.get(
            data.get("recommendation", "").lower(),
            ReviewActionType.MAJOR_REVISION,
        )

        # Parse proposed experiments
        experiments = []
        for exp in data.get("proposed_experiments", []):
            experiments.append(
                ProposedExperiment(
                    name=exp.get("name", "unnamed"),
                    rationale=exp.get("rationale", ""),
                    hypothesis=exp.get("hypothesis", ""),
                    group_name=exp.get("name", "experiment").lower().replace(" ", "_"),
                    group_description=exp.get("hypothesis", ""),
                    variables={v: "vary" for v in exp.get("key_variables", [])},
                    benchmarks=exp.get("benchmarks", ["perplexity"]),
                    priority=exp.get("priority", 2),
                )
            )

        return ReviewResult(
            overall_score=float(data.get("overall_score", 5.0)),
            recommendation=recommendation,
            summary=data.get("summary", ""),
            comments=[],  # Could parse if we add structured comments
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            proposed_experiments=experiments,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


async def review_paper(
    paper_dir: Path | str,
    config: ReviewConfig | None = None,
    manifest: "Manifest | None" = None,
    manifest_path: Path | str | None = None,
    experiment_results: dict[str, Any] | None = None,
) -> ReviewResult:
    """Convenience function to review a paper.

    Args:
        paper_dir: Directory containing paper.tex.
        config: Review configuration (uses defaults if None).
        manifest: Experiment manifest (optional).
        manifest_path: Path to manifest file.
        experiment_results: Experiment results dict.

    Returns:
        ReviewResult with feedback and proposed experiments.
    """
    if config is None:
        config = ReviewConfig()

    reviewer = PaperReviewer(config, output_dir=paper_dir)
    return await reviewer.review(
        manifest=manifest,
        manifest_path=manifest_path,
        experiment_results=experiment_results,
    )


def review_paper_sync(
    paper_dir: Path | str,
    config: ReviewConfig | None = None,
    manifest: "Manifest | None" = None,
    manifest_path: Path | str | None = None,
    experiment_results: dict[str, Any] | None = None,
) -> ReviewResult:
    """Synchronous version of review_paper()."""
    return asyncio.run(
        review_paper(
            paper_dir=paper_dir,
            config=config,
            manifest=manifest,
            manifest_path=manifest_path,
            experiment_results=experiment_results,
        )
    )
