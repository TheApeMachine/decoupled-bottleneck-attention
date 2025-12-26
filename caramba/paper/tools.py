"""Tools for the paper drafting agent.

Provides function tools that the AI agent can use to:
- Read and write LaTeX files
- Manage BibTeX citations
- Search academic databases
- Include figures and assets
- Read experiment results
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, Field

from agents import function_tool

if TYPE_CHECKING:
    from caramba.config.paper import PaperConfig


# ============================================================================
# Data Models
# ============================================================================


class Citation(BaseModel):
    """A citation entry for the bibliography."""

    key: str = Field(description="BibTeX citation key")
    title: str = Field(description="Paper title")
    authors: list[str] = Field(description="List of author names")
    year: int = Field(description="Publication year")
    venue: str = Field(description="Journal or conference name")
    doi: str | None = Field(default=None, description="DOI if available")
    url: str | None = Field(default=None, description="URL to the paper")
    abstract: str | None = Field(default=None, description="Paper abstract")


class SearchResult(BaseModel):
    """Result from an academic search."""

    citations: list[Citation] = Field(default_factory=list)
    query: str = Field(description="The search query used")
    source: str = Field(description="The source database")
    total_results: int = Field(default=0, description="Total results found")


class SectionContent(BaseModel):
    """Content for a paper section."""

    name: str = Field(description="Section name (e.g., 'introduction')")
    content: str = Field(description="LaTeX content for the section")
    subsections: list["SectionContent"] = Field(default_factory=list)


class ExperimentSummary(BaseModel):
    """Summary of experiment results for the paper."""

    name: str = Field(description="Experiment name")
    description: str = Field(description="Brief description")
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    config_summary: str = Field(description="Summary of configuration")


# ============================================================================
# Paper State (shared across tools)
# ============================================================================


@dataclass
class PaperState:
    """Shared state for paper drafting tools."""

    output_dir: Path
    paper_config: "PaperConfig"
    manifest_path: Path | None = None
    experiment_results: dict | None = None
    artifacts: dict[str, Path] | None = None

    @property
    def tex_path(self) -> Path:
        """Path to the main .tex file."""
        return self.output_dir / "paper.tex"

    @property
    def bib_path(self) -> Path:
        """Path to the references.bib file."""
        return self.output_dir / "references.bib"

    @property
    def figures_dir(self) -> Path:
        """Path to the figures directory."""
        return self.output_dir / "figures"


# Global state - will be set by the drafter before running agent
_state: PaperState | None = None


def set_state(state: PaperState) -> None:
    """Set the global paper state for tools to use."""
    global _state
    _state = state


def get_state() -> PaperState:
    """Get the global paper state."""
    if _state is None:
        raise RuntimeError("Paper state not initialized. Call set_state first.")
    return _state


# ============================================================================
# File Operations Tools
# ============================================================================


@function_tool
def read_tex_file() -> str:
    """Read the current paper.tex file contents.

    Returns the full LaTeX content of the paper, or a message if the file
    doesn't exist yet.
    """
    state = get_state()
    if state.tex_path.exists():
        return state.tex_path.read_text(encoding="utf-8")
    return "FILE_NOT_FOUND: No paper.tex exists yet. Use write_tex_file to create one."


@function_tool
def write_tex_file(
    content: Annotated[str, "The complete LaTeX content for the paper"],
) -> str:
    """Write the complete paper.tex file.

    This overwrites the entire file. Use update_section for incremental changes.
    """
    state = get_state()
    state.output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally create a versioned backup
    if state.paper_config.auto_version and state.tex_path.exists():
        _create_backup(state.tex_path, state.paper_config.max_versions)

    state.tex_path.write_text(content, encoding="utf-8")
    return f"Successfully wrote paper.tex ({len(content)} characters)"


@function_tool
def update_section(
    section_name: Annotated[str, "Name of the section to update (e.g., 'introduction')"],
    content: Annotated[str, "New LaTeX content for this section"],
) -> str:
    """Update a specific section in the paper.

    Finds the section by name and replaces its content between
    \\section{Name} and the next \\section or \\end{document}.
    """
    state = get_state()

    if not state.tex_path.exists():
        return "ERROR: No paper.tex exists. Use write_tex_file first to create the document."

    tex_content = state.tex_path.read_text(encoding="utf-8")

    # Find and replace the section
    # Pattern matches \section{Name} through the next \section or \end{document}
    pattern = rf"(\\section\{{{re.escape(section_name)}\}})(.*?)(\\section|\Z|\\end\{{document\}})"
    match = re.search(pattern, tex_content, re.DOTALL | re.IGNORECASE)

    if match:
        # Replace the section content
        new_tex = (
            tex_content[: match.start()]
            + f"\\section{{{section_name}}}\n{content}\n\n"
            + match.group(3)
            + tex_content[match.end() :]
        )
        state.tex_path.write_text(new_tex, encoding="utf-8")
        return f"Successfully updated section '{section_name}'"

    return f"Section '{section_name}' not found. Available sections: {_find_sections(tex_content)}"


@function_tool
def read_bib_file() -> str:
    """Read the current references.bib file contents."""
    state = get_state()
    if state.bib_path.exists():
        return state.bib_path.read_text(encoding="utf-8")
    return "FILE_NOT_FOUND: No references.bib exists yet. Use add_citation to create entries."


@function_tool
def add_citation(
    key: Annotated[str, "Unique BibTeX key for this citation"],
    entry_type: Annotated[str, "BibTeX entry type (article, inproceedings, misc, etc.)"],
    title: Annotated[str, "Paper title"],
    authors: Annotated[str, "Authors in 'Last, First and Last, First' format"],
    year: Annotated[int, "Publication year"],
    venue: Annotated[str, "Journal or conference name"],
    doi: Annotated[str, "DOI if available, or empty string"] = "",
    url: Annotated[str, "URL to the paper, or empty string"] = "",
    volume: Annotated[str, "Volume number, or empty string"] = "",
    pages: Annotated[str, "Page numbers, or empty string"] = "",
) -> str:
    """Add a citation to the references.bib file.

    Creates a properly formatted BibTeX entry and appends it to the file.
    """
    state = get_state()
    state.output_dir.mkdir(parents=True, exist_ok=True)

    # Build the BibTeX entry
    entry_lines = [f"@{entry_type}{{{key},"]
    entry_lines.append(f"  title = {{{title}}},")
    entry_lines.append(f"  author = {{{authors}}},")
    entry_lines.append(f"  year = {{{year}}},")

    if entry_type == "article":
        entry_lines.append(f"  journal = {{{venue}}},")
    else:
        entry_lines.append(f"  booktitle = {{{venue}}},")

    if doi:
        entry_lines.append(f"  doi = {{{doi}}},")
    if url:
        entry_lines.append(f"  url = {{{url}}},")
    if volume:
        entry_lines.append(f"  volume = {{{volume}}},")
    if pages:
        entry_lines.append(f"  pages = {{{pages}}},")

    entry_lines.append("}")
    entry = "\n".join(entry_lines)

    # Check if citation already exists
    existing = ""
    if state.bib_path.exists():
        existing = state.bib_path.read_text(encoding="utf-8")
        if f"@{entry_type}{{{key}," in existing or f"@{entry_type.upper()}{{{key}," in existing:
            return f"Citation '{key}' already exists in references.bib"

    # Append the new entry
    with open(state.bib_path, "a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write("\n" + entry + "\n")

    return f"Successfully added citation '{key}'"


# ============================================================================
# Search Tools
# ============================================================================


@function_tool
def search_arxiv(
    query: Annotated[str, "Search query for arXiv"],
    max_results: Annotated[int, "Maximum number of results to return"] = 10,
) -> SearchResult:
    """Search arXiv for relevant papers.

    Returns a list of citations that can be added to the bibliography.
    """
    try:
        # Build the arXiv API query
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(max_results, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        # Make the request
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode("utf-8")

        # Parse the Atom feed (simple regex parsing to avoid dependencies)
        citations = []
        entries = re.findall(r"<entry>(.*?)</entry>", data, re.DOTALL)

        for entry in entries:
            title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            authors_matches = re.findall(r"<name>(.*?)</name>", entry)
            published_match = re.search(r"<published>(\d{4})", entry)
            summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            id_match = re.search(r"<id>(.*?)</id>", entry)

            if title_match and authors_matches and published_match:
                title = title_match.group(1).strip().replace("\n", " ")
                year = int(published_match.group(1))
                arxiv_id = id_match.group(1).split("/")[-1] if id_match else ""
                key = f"{authors_matches[0].split()[-1].lower()}{year}_{arxiv_id.replace('.', '_')}"

                citations.append(
                    Citation(
                        key=key[:30],  # Limit key length
                        title=title,
                        authors=authors_matches[:5],  # Limit authors
                        year=year,
                        venue="arXiv preprint",
                        url=id_match.group(1) if id_match else None,
                        abstract=summary_match.group(1).strip()[:500] if summary_match else None,
                    )
                )

        return SearchResult(
            citations=citations,
            query=query,
            source="arxiv",
            total_results=len(citations),
        )

    except Exception as e:
        return SearchResult(
            citations=[],
            query=query,
            source="arxiv",
            total_results=0,
        )


@function_tool
def search_semantic_scholar(
    query: Annotated[str, "Search query for Semantic Scholar"],
    max_results: Annotated[int, "Maximum number of results to return"] = 10,
) -> SearchResult:
    """Search Semantic Scholar for relevant papers.

    Returns a list of citations that can be added to the bibliography.
    """
    try:
        # Build the Semantic Scholar API query
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(max_results, 50),
            "fields": "title,authors,year,venue,externalIds,abstract",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        # Make the request
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Caramba/1.0")

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        citations = []
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            if not authors or not paper.get("title"):
                continue

            year = paper.get("year", 2024)
            first_author_last = authors[0].split()[-1].lower() if authors else "unknown"
            key = f"{first_author_last}{year}_{len(citations)}"

            external_ids = paper.get("externalIds", {})
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")

            citations.append(
                Citation(
                    key=key[:30],
                    title=paper.get("title", ""),
                    authors=authors[:5],
                    year=year,
                    venue=paper.get("venue", "Unknown venue") or "Unknown venue",
                    doi=doi,
                    url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                    abstract=paper.get("abstract", "")[:500] if paper.get("abstract") else None,
                )
            )

        return SearchResult(
            citations=citations,
            query=query,
            source="semantic_scholar",
            total_results=data.get("total", len(citations)),
        )

    except Exception as e:
        return SearchResult(
            citations=[],
            query=query,
            source="semantic_scholar",
            total_results=0,
        )


# ============================================================================
# Experiment Data Tools
# ============================================================================


@function_tool
def get_experiment_manifest() -> str:
    """Get the experiment manifest YAML/JSON content.

    Returns the raw manifest file content that defines the experiment.
    """
    state = get_state()
    if state.manifest_path and state.manifest_path.exists():
        return state.manifest_path.read_text(encoding="utf-8")
    return "No manifest file available."


@function_tool
def get_experiment_results() -> str:
    """Get the experiment results as JSON.

    Returns benchmark results, metrics, and other experiment outputs.
    """
    state = get_state()
    if state.experiment_results:
        return json.dumps(state.experiment_results, indent=2, default=str)
    return "No experiment results available yet."


@function_tool
def list_artifacts() -> list[str]:
    """List all available artifact files from the experiment.

    Returns paths to generated figures, tables, and data files that
    can be included in the paper.
    """
    state = get_state()
    if state.artifacts:
        return [str(p) for p in state.artifacts.values()]

    # Also check the artifacts directory
    artifacts_list = []
    if state.output_dir.parent.exists():
        for ext in ["*.png", "*.pdf", "*.csv", "*.json", "*.tex"]:
            artifacts_list.extend(str(p) for p in state.output_dir.parent.glob(ext))
    return artifacts_list


@function_tool
def include_figure(
    artifact_path: Annotated[str, "Path to the artifact file (e.g., 'summary.png')"],
    caption: Annotated[str, "Figure caption"],
    label: Annotated[str, "LaTeX label for referencing (e.g., 'fig:summary')"],
    width: Annotated[str, "Figure width (e.g., '0.8\\textwidth')"] = "0.8\\textwidth",
) -> str:
    """Generate LaTeX code to include a figure in the paper.

    Copies the figure to the paper's figures directory and returns
    the LaTeX includegraphics command.
    """
    import shutil

    state = get_state()
    state.figures_dir.mkdir(parents=True, exist_ok=True)

    # Find the artifact
    source = Path(artifact_path)
    if not source.is_absolute():
        # Try to find it in common locations
        candidates = [
            Path(artifact_path),
            state.output_dir.parent / artifact_path,
            Path("artifacts") / artifact_path,
        ]
        if state.artifacts:
            for name, path in state.artifacts.items():
                if name == artifact_path or str(path).endswith(artifact_path):
                    source = path
                    break
        for candidate in candidates:
            if candidate.exists():
                source = candidate
                break

    if not source.exists():
        return f"ERROR: Artifact '{artifact_path}' not found."

    # Copy to figures directory
    dest = state.figures_dir / source.name
    shutil.copy(source, dest)

    # Generate LaTeX
    relative_path = f"figures/{source.name}"
    latex = f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width={width}]{{{relative_path}}}
    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{figure}}"""

    return latex


# ============================================================================
# Template Tools
# ============================================================================


@function_tool
def get_paper_template() -> str:
    """Get a LaTeX paper template to start from.

    Returns a complete LaTeX document template with standard sections.
    """
    state = get_state()
    config = state.paper_config

    authors_str = " \\and ".join(config.authors) if config.authors else "Author Name"

    template = f"""\\documentclass[11pt]{{article}}

% ============================================================================
% Packages
% ============================================================================
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{cleveref}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}
\\usepackage{{xcolor}}
\\usepackage{{listings}}
\\usepackage[margin=1in]{{geometry}}

% ============================================================================
% Document Info
% ============================================================================
\\title{{{config.title}}}
\\author{{{authors_str}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

% ============================================================================
% Abstract
% ============================================================================
\\begin{{abstract}}
[Abstract goes here - maximum {config.abstract_max_words} words]
\\end{{abstract}}

% ============================================================================
% Introduction
% ============================================================================
\\section{{Introduction}}
\\label{{sec:introduction}}

[Introduction goes here]

% ============================================================================
% Related Work
% ============================================================================
\\section{{Related Work}}
\\label{{sec:related_work}}

[Related work goes here]

% ============================================================================
% Methodology
% ============================================================================
\\section{{Methodology}}
\\label{{sec:methodology}}

[Methodology goes here]

% ============================================================================
% Experiments
% ============================================================================
\\section{{Experiments}}
\\label{{sec:experiments}}

[Experiments description goes here]

% ============================================================================
% Results
% ============================================================================
\\section{{Results}}
\\label{{sec:results}}

[Results go here]

% ============================================================================
% Discussion
% ============================================================================
\\section{{Discussion}}
\\label{{sec:discussion}}

[Discussion goes here]

% ============================================================================
% Conclusion
% ============================================================================
\\section{{Conclusion}}
\\label{{sec:conclusion}}

[Conclusion goes here]

% ============================================================================
% References
% ============================================================================
\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""
    return template


# ============================================================================
# Utilities
# ============================================================================


def _find_sections(tex_content: str) -> list[str]:
    """Find all section names in a LaTeX document."""
    matches = re.findall(r"\\section\{([^}]+)\}", tex_content)
    return matches


def _create_backup(file_path: Path, max_versions: int) -> None:
    """Create a versioned backup of a file."""
    backup_dir = file_path.parent / "versions"
    backup_dir.mkdir(exist_ok=True)

    # Find next version number
    existing = list(backup_dir.glob(f"{file_path.stem}_v*.tex"))
    version = len(existing) + 1

    if version > max_versions:
        # Remove oldest version
        oldest = min(existing, key=lambda p: p.stat().st_mtime)
        oldest.unlink()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{file_path.stem}_v{version:03d}_{timestamp}.tex"
    backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")


# ============================================================================
# Tool Collection
# ============================================================================

ALL_TOOLS = [
    read_tex_file,
    write_tex_file,
    update_section,
    read_bib_file,
    add_citation,
    search_arxiv,
    search_semantic_scholar,
    get_experiment_manifest,
    get_experiment_results,
    list_artifacts,
    include_figure,
    get_paper_template,
]
