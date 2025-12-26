"""Paper review configuration and result models.

Defines the review process configuration and the structured output
from the reviewer agent, including actions to take.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReviewActionType(str, Enum):
    """Types of actions the reviewer can recommend."""

    APPROVE = "approve"  # Paper is ready, no changes needed
    STYLE_FIX = "style_fix"  # Stylistic changes within existing content
    CLARIFICATION = "clarification"  # Needs clarification, can be resolved with existing data
    NEW_EXPERIMENT = "new_experiment"  # Requires new experiments to address
    MAJOR_REVISION = "major_revision"  # Significant restructuring needed


class ReviewSeverity(str, Enum):
    """Severity levels for review comments."""

    CRITICAL = "critical"  # Must be addressed before publication
    MAJOR = "major"  # Should be addressed, significant impact
    MINOR = "minor"  # Nice to have, small impact
    SUGGESTION = "suggestion"  # Optional improvement


class ReviewComment(BaseModel):
    """A single review comment on the paper."""

    section: str = Field(description="Section this comment applies to")
    severity: ReviewSeverity = Field(description="Severity of the issue")
    action_type: ReviewActionType = Field(description="Type of action needed")
    comment: str = Field(description="The review comment/critique")
    suggestion: str = Field(description="Suggested fix or improvement")

    # For new experiments
    experiment_rationale: str | None = Field(
        default=None,
        description="Why this experiment is needed (if action_type is NEW_EXPERIMENT)",
    )


class ProposedExperiment(BaseModel):
    """A proposed new experiment to address reviewer feedback."""

    name: str = Field(description="Short name for the experiment")
    rationale: str = Field(description="Why this experiment is needed")
    hypothesis: str = Field(description="What we expect to learn/show")

    # Manifest components
    group_name: str = Field(description="Name for the experiment group")
    group_description: str = Field(description="Description of what the group tests")

    # Key parameters to vary/test
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to test/vary in the experiment",
    )

    # Benchmark types needed
    benchmarks: list[str] = Field(
        default_factory=lambda: ["perplexity"],
        description="Types of benchmarks to run (perplexity, latency, memory)",
    )

    # Priority
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Priority 1-5 (1=highest)",
    )


class ReviewResult(BaseModel):
    """Complete review result from the reviewer agent."""

    # Overall assessment
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall paper quality score (0-10)",
    )
    recommendation: ReviewActionType = Field(
        description="Overall recommendation for the paper",
    )
    summary: str = Field(description="Executive summary of the review")

    # Detailed comments
    comments: list[ReviewComment] = Field(
        default_factory=list,
        description="Individual review comments",
    )

    # Strengths and weaknesses
    strengths: list[str] = Field(
        default_factory=list,
        description="Key strengths of the paper",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Key weaknesses to address",
    )

    # Proposed experiments (if any)
    proposed_experiments: list[ProposedExperiment] = Field(
        default_factory=list,
        description="New experiments proposed to strengthen the paper",
    )

    # Quick stats
    @property
    def needs_new_experiments(self) -> bool:
        """Check if any comments require new experiments."""
        return any(
            c.action_type == ReviewActionType.NEW_EXPERIMENT for c in self.comments
        ) or len(self.proposed_experiments) > 0

    @property
    def style_fixes_only(self) -> bool:
        """Check if only style fixes are needed."""
        return all(
            c.action_type in (ReviewActionType.STYLE_FIX, ReviewActionType.CLARIFICATION)
            for c in self.comments
        ) and len(self.proposed_experiments) == 0

    @property
    def critical_issues(self) -> list[ReviewComment]:
        """Get all critical issues."""
        return [c for c in self.comments if c.severity == ReviewSeverity.CRITICAL]


class ReviewConfig(BaseModel):
    """Configuration for the paper review process."""

    enabled: bool = True

    # Review focus areas
    check_methodology: bool = True
    check_experiments: bool = True
    check_results: bool = True
    check_writing: bool = True
    check_citations: bool = True

    # Strictness
    strictness: str = Field(
        default="conference",
        description="Review strictness: 'workshop', 'conference', 'journal', 'top_venue'",
    )

    # Thresholds
    min_score_to_approve: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Minimum score to approve without changes",
    )

    # Model settings
    model: str = Field(
        default="gpt-5.2",
        description="OpenAI model to use for reviewing",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)  # Lower for more consistent reviews

    # Experiment generation
    auto_generate_experiments: bool = Field(
        default=True,
        description="Automatically generate experiment manifests for proposed experiments",
    )
    max_proposed_experiments: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of experiments to propose",
    )

    # Review persona
    reviewer_persona: str = Field(
        default="senior_researcher",
        description="Reviewer persona: 'senior_researcher', 'methodology_expert', 'practitioner'",
    )

    # Custom review instructions
    custom_instructions: str | None = Field(
        default=None,
        description="Additional instructions for the reviewer",
    )
