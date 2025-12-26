"""Paper drafting orchestration using OpenAI Agent SDK.

The PaperDrafter coordinates an AI agent to write and update academic papers
based on experiment manifests and results. It handles:

- Creating new papers from scratch
- Updating existing drafts with new results
- Managing citations and references
- Incorporating generated figures and tables
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents import Agent, Runner
from agents.tool import Tool

from caramba.config.paper import PaperConfig, PaperSection
from caramba.console import logger
from caramba.paper.tools import ALL_TOOLS, PaperState, set_state

if TYPE_CHECKING:
    from caramba.config.manifest import Manifest


# ============================================================================
# System Instructions
# ============================================================================

SYSTEM_INSTRUCTIONS = """You are an expert academic paper writer specializing in machine learning and AI research. You write clear, precise, and well-structured scientific papers.

## Your Capabilities

You have tools to:
1. Read and write LaTeX files (paper.tex, references.bib)
2. Search academic databases (arXiv, Semantic Scholar) for citations
3. Access experiment manifests and results
4. Include figures and artifacts from experiments

## Paper Writing Guidelines

### Style
- Use precise, technical language appropriate for ML/AI venues
- Write in third person ("We propose..." or "This paper presents...")
- Be concise but thorough
- Use active voice where appropriate
- Avoid hyperbole and unsupported claims

### Structure
- Abstract: ~150-250 words summarizing problem, approach, and key results
- Introduction: Motivate the problem, state contributions, outline paper
- Related Work: Position work relative to prior art, cite relevant papers
- Methodology: Clearly explain the approach with formal notation where helpful
- Experiments: Describe setup, datasets, baselines, metrics
- Results: Present findings with supporting figures/tables
- Discussion: Analyze results, discuss limitations, future work
- Conclusion: Summarize contributions and impact

### Citations
- Always cite relevant prior work
- Use \\cite{key} for inline citations
- Search for citations before writing sections
- Prefer recent, high-quality papers from top venues

### Figures
- Include all relevant generated figures from experiments
- Write informative captions
- Reference figures in the text (e.g., "As shown in Figure~\\ref{fig:results}")

### Mathematics
- Use proper LaTeX math notation
- Define notation before using it
- Number important equations

## Workflow

1. First, check if a paper already exists (read_tex_file)
2. If not, get the template and customize it (get_paper_template)
3. Read the experiment manifest and results
4. Search for relevant citations for the topic
5. Write each section with appropriate content
6. Include figures from experiment artifacts
7. Ensure proper citations are added

When updating an existing paper:
1. Read the current paper
2. Get new experiment results
3. Update relevant sections (use update_section)
4. Add any new citations needed
5. Include new figures

Always ensure the paper compiles correctly with proper LaTeX syntax."""


# ============================================================================
# PaperDrafter Class
# ============================================================================


class PaperDrafter:
    """Orchestrates AI-assisted paper drafting from experiments.

    Uses the OpenAI Agent SDK to coordinate paper writing, handling both
    new paper creation and updates to existing drafts.

    Usage:
        config = PaperConfig(title="My Paper", ...)
        drafter = PaperDrafter(config)
        result = await drafter.draft(manifest, artifacts)
    """

    def __init__(
        self,
        config: PaperConfig,
        output_dir: Path | str | None = None,
    ) -> None:
        """Initialize the paper drafter.

        Args:
            config: Paper configuration from manifest.
            output_dir: Base output directory for paper files.
                       Defaults to artifacts/{paper_dir}/
        """
        self.config = config

        # Set up output directory
        if output_dir is None:
            output_dir = Path("artifacts") / config.output_dir
        self.output_dir = Path(output_dir)

        # Build custom instructions
        instructions = SYSTEM_INSTRUCTIONS
        if config.custom_instructions:
            instructions += f"\n\n## Additional Instructions\n\n{config.custom_instructions}"

        # Create the agent
        # Cast tools to the expected type for the Agent
        tools: list[Tool] = list(ALL_TOOLS)  # type: ignore[assignment]
        self.agent = Agent(
            name="Paper Drafter",
            instructions=instructions,
            tools=tools,
            model=config.model,
        )

    async def draft(
        self,
        manifest: "Manifest | None" = None,
        manifest_path: Path | str | None = None,
        experiment_results: dict[str, Any] | None = None,
        artifacts: dict[str, Path] | None = None,
    ) -> Path:
        """Draft or update a paper based on experiment data.

        Args:
            manifest: The experiment manifest (optional).
            manifest_path: Path to the manifest file.
            experiment_results: Dict of experiment results/metrics.
            artifacts: Dict mapping artifact names to paths.

        Returns:
            Path to the generated paper.tex file.
        """
        logger.header("Paper Drafter", self.config.title)

        # Set up state for tools
        state = PaperState(
            output_dir=self.output_dir,
            paper_config=self.config,
            manifest_path=Path(manifest_path) if manifest_path else None,
            experiment_results=experiment_results,
            artifacts=artifacts,
        )
        set_state(state)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine the task based on whether a paper exists
        if state.tex_path.exists():
            task = self._build_update_prompt(manifest, experiment_results, artifacts)
            logger.info("Updating existing paper draft...")
        else:
            task = self._build_create_prompt(manifest, experiment_results, artifacts)
            logger.info("Creating new paper draft...")

        # Run the agent
        logger.info(f"Running paper drafting agent with {self.config.model}...")

        try:
            result = await Runner.run(self.agent, input=task)
            logger.success("Paper drafting complete!")

            # Log the result summary
            if result.final_output:
                # Truncate long outputs for logging
                summary = result.final_output[:500]
                if len(result.final_output) > 500:
                    summary += "..."
                logger.info(f"Agent output: {summary}")

        except Exception as e:
            logger.error(f"Paper drafting failed: {e}")
            raise

        # Return the path to the paper
        return state.tex_path

    def draft_sync(
        self,
        manifest: "Manifest | None" = None,
        manifest_path: Path | str | None = None,
        experiment_results: dict[str, Any] | None = None,
        artifacts: dict[str, Path] | None = None,
    ) -> Path:
        """Synchronous wrapper for draft().

        Convenient for calling from synchronous code.
        """
        return asyncio.run(
            self.draft(
                manifest=manifest,
                manifest_path=manifest_path,
                experiment_results=experiment_results,
                artifacts=artifacts,
            )
        )

    def _build_create_prompt(
        self,
        manifest: "Manifest | None",
        experiment_results: dict[str, Any] | None,
        artifacts: dict[str, Path] | None,
    ) -> str:
        """Build the prompt for creating a new paper."""
        prompt = f"""Please create a new academic paper with the following specifications:

## Paper Details
- Title: {self.config.title}
- Authors: {', '.join(self.config.authors) if self.config.authors else 'To be specified'}
- Type: {self.config.paper_type.value}

## Required Sections
{self._format_sections()}

## Instructions

1. First, get the paper template using get_paper_template()
2. Read the experiment manifest using get_experiment_manifest()
3. Get the experiment results using get_experiment_results()
4. Search for relevant citations for the research topic
5. Write each section with substantive content based on the experiment
6. Include available figures using include_figure()
7. Add all necessary citations to references.bib
8. Write the complete paper to paper.tex

## Keywords for Citation Search
{', '.join(self.config.keywords) if self.config.keywords else 'machine learning, attention mechanism, transformer'}

Please create a complete, well-structured paper draft."""

        return prompt

    def _build_update_prompt(
        self,
        manifest: "Manifest | None",
        experiment_results: dict[str, Any] | None,
        artifacts: dict[str, Path] | None,
    ) -> str:
        """Build the prompt for updating an existing paper."""
        prompt = f"""Please update the existing paper draft with new experiment results.

## Paper Details
- Title: {self.config.title}

## Instructions

1. Read the current paper using read_tex_file()
2. Get the latest experiment results using get_experiment_results()
3. List available artifacts using list_artifacts()
4. Update the relevant sections (especially Results and Experiments) with new data
5. Add any new figures that should be included
6. Search for and add any additional citations if needed
7. Ensure the paper reflects all current experiment findings

Focus on integrating new results while maintaining the existing structure and content.
Create a new version of the paper with the updates."""

        return prompt

    def _format_sections(self) -> str:
        """Format the configured sections for the prompt."""
        section_names = {
            PaperSection.ABSTRACT: "Abstract",
            PaperSection.INTRODUCTION: "Introduction",
            PaperSection.RELATED_WORK: "Related Work",
            PaperSection.METHODOLOGY: "Methodology",
            PaperSection.EXPERIMENTS: "Experiments",
            PaperSection.RESULTS: "Results",
            PaperSection.DISCUSSION: "Discussion",
            PaperSection.CONCLUSION: "Conclusion",
            PaperSection.APPENDIX: "Appendix",
        }
        return "\n".join(
            f"- {section_names.get(s, s.value)}" for s in self.config.sections
        )


# ============================================================================
# Convenience Functions
# ============================================================================


async def draft_paper(
    config: PaperConfig,
    manifest: "Manifest | None" = None,
    manifest_path: Path | str | None = None,
    experiment_results: dict[str, Any] | None = None,
    artifacts: dict[str, Path] | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """Convenience function to draft a paper.

    Creates a PaperDrafter and runs it with the provided data.

    Args:
        config: Paper configuration.
        manifest: Experiment manifest (optional).
        manifest_path: Path to manifest file.
        experiment_results: Experiment results dict.
        artifacts: Generated artifacts dict.
        output_dir: Output directory for paper files.

    Returns:
        Path to the generated paper.tex file.
    """
    drafter = PaperDrafter(config, output_dir)
    return await drafter.draft(
        manifest=manifest,
        manifest_path=manifest_path,
        experiment_results=experiment_results,
        artifacts=artifacts,
    )


def draft_paper_sync(
    config: PaperConfig,
    manifest: "Manifest | None" = None,
    manifest_path: Path | str | None = None,
    experiment_results: dict[str, Any] | None = None,
    artifacts: dict[str, Path] | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """Synchronous version of draft_paper()."""
    return asyncio.run(
        draft_paper(
            config=config,
            manifest=manifest,
            manifest_path=manifest_path,
            experiment_results=experiment_results,
            artifacts=artifacts,
            output_dir=output_dir,
        )
    )
