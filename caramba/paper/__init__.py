"""AI-assisted paper drafting and review for experiments.

This module provides automated paper generation and review from experiment
results using OpenAI's Agent SDK. It can:

- Create new LaTeX papers from experiment manifests and results
- Update existing drafts with new experiment data
- Search and cite relevant literature
- Generate figures and tables from artifacts
- Review papers and identify weaknesses
- Propose and generate new experiments to strengthen papers
- Run autonomous research loops (write → review → experiment → repeat)
"""
from caramba.paper.drafter import PaperDrafter
from caramba.paper.reviewer import PaperReviewer
from caramba.paper.research_loop import ResearchLoop, ResearchLoopConfig
from caramba.paper.review import ReviewConfig, ReviewResult

__all__ = [
    "PaperDrafter",
    "PaperReviewer",
    "ResearchLoop",
    "ResearchLoopConfig",
    "ReviewConfig",
    "ReviewResult",
]
