"""Paper configuration for AI-assisted paper drafting.

Defines configuration for automatic paper/technical report generation
using AI agents. The paper drafter can create new papers from experiments,
update existing drafts with new results, and search for citations.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PaperType(str, Enum):
    """Type of paper/document to generate."""

    PAPER = "paper"
    TECHNICAL_REPORT = "technical_report"
    BLOG_POST = "blog_post"
    ARXIV = "arxiv"


class PaperSection(str, Enum):
    """Standard paper sections."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"


class CitationConfig(BaseModel):
    """Configuration for citation search and management."""

    enabled: bool = True
    max_citations: int = Field(default=20, ge=1, le=100)
    sources: list[str] = Field(
        default_factory=lambda: ["arxiv", "semantic_scholar", "google_scholar"]
    )
    prefer_recent: bool = True
    recent_years: int = Field(default=5, ge=1, le=20)


class PaperConfig(BaseModel):
    """Configuration for AI-assisted paper drafting.

    This is added to the manifest to enable paper generation from experiments.
    The agent will create a new paper if none exists, or update an existing
    draft with new experiment results.
    """

    enabled: bool = True
    title: str = Field(description="Title of the paper")
    authors: list[str] = Field(default_factory=list)
    paper_type: PaperType = PaperType.PAPER
    output_dir: str = Field(
        default="paper",
        description="Subdirectory within artifacts for paper outputs",
    )

    # Content configuration
    abstract_max_words: int = Field(default=250, ge=50, le=500)
    sections: list[PaperSection] = Field(
        default_factory=lambda: [
            PaperSection.ABSTRACT,
            PaperSection.INTRODUCTION,
            PaperSection.RELATED_WORK,
            PaperSection.METHODOLOGY,
            PaperSection.EXPERIMENTS,
            PaperSection.RESULTS,
            PaperSection.DISCUSSION,
            PaperSection.CONCLUSION,
        ]
    )

    # Citation configuration
    citations: CitationConfig = Field(default_factory=CitationConfig)

    # Model configuration
    model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for drafting",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Versioning
    auto_version: bool = Field(
        default=True,
        description="Create numbered versions instead of overwriting",
    )
    max_versions: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of versions to keep",
    )

    # Keywords for citation search
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to guide citation search",
    )

    # Custom instructions for the agent
    custom_instructions: str | None = Field(
        default=None,
        description="Additional instructions for the paper writing agent",
    )
