"""Tools for the paper reviewer agent.

Provides function tools that the AI reviewer can use to:
- Analyze paper sections
- Check experimental coverage
- Propose new experiments
- Generate experiment manifests
- Search for missing citations
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, Field

from agents import function_tool

if TYPE_CHECKING:
    from caramba.paper.review import ReviewConfig

from caramba.paper.tools import get_state


# ============================================================================
# Review Analysis Tools
# ============================================================================


@function_tool
def analyze_paper_structure() -> str:
    """Analyze the structure of the current paper.

    Returns a summary of sections, their lengths, and basic statistics
    to help identify structural issues.
    """
    state = get_state()

    if not state.tex_path.exists():
        return "ERROR: No paper.tex found to analyze."

    content = state.tex_path.read_text(encoding="utf-8")

    # Find sections
    sections = re.findall(r"\\section\{([^}]+)\}", content)
    subsections = re.findall(r"\\subsection\{([^}]+)\}", content)

    # Count figures, tables, equations
    figures = len(re.findall(r"\\begin\{figure\}", content))
    tables = len(re.findall(r"\\begin\{table\}", content))
    equations = len(re.findall(r"\\begin\{equation\}", content))
    equations += len(re.findall(r"\\\[", content))  # Display math

    # Count citations
    citations = len(re.findall(r"\\cite\{[^}]+\}", content))

    # Word count (rough)
    text_only = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", content)
    text_only = re.sub(r"\\[a-zA-Z]+", "", text_only)
    text_only = re.sub(r"[{}\\%$]", "", text_only)
    words = len(text_only.split())

    # Section lengths
    section_info = []
    for i, section in enumerate(sections):
        pattern = rf"\\section\{{{re.escape(section)}\}}(.*?)(\\section|\\end\{{document\}})"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            section_text = match.group(1)
            section_words = len(section_text.split())
            section_info.append(f"  - {section}: ~{section_words} words")

    return f"""Paper Structure Analysis:
=========================

Sections ({len(sections)}):
{chr(10).join(section_info)}

Subsections: {len(subsections)}
Figures: {figures}
Tables: {tables}
Equations: {equations}
Citations: {citations}
Approximate word count: {words}

Section list: {', '.join(sections)}
"""


@function_tool
def check_experimental_claims() -> str:
    """Check for experimental claims that may need supporting evidence.

    Scans the paper for claims that should be backed by experiments
    and checks if corresponding results are present.
    """
    state = get_state()

    if not state.tex_path.exists():
        return "ERROR: No paper.tex found."

    content = state.tex_path.read_text(encoding="utf-8")

    # Patterns that suggest claims needing evidence
    claim_patterns = [
        (r"we show that", "empirical claim"),
        (r"we demonstrate", "demonstration claim"),
        (r"outperforms", "performance claim"),
        (r"achieves .{0,20}(accuracy|perplexity|speedup|reduction)", "metric claim"),
        (r"significant(ly)? (better|faster|more efficient)", "comparative claim"),
        (r"state-of-the-art", "SOTA claim"),
        (r"\d+(\.\d+)?[x×] (faster|speedup|reduction|improvement)", "quantitative claim"),
        (r"ablation", "ablation study reference"),
    ]

    findings = []
    for pattern, claim_type in claim_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].replace("\n", " ").strip()
            findings.append(f"  [{claim_type}] ...{context}...")

    # Check for results section content
    results_match = re.search(
        r"\\section\{Results\}(.*?)(\\section|\\end\{document\})",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    has_results = bool(results_match and len(results_match.group(1).strip()) > 100)

    # Check for figure references
    figure_refs = re.findall(r"(Figure|Fig\.)~?\\ref\{([^}]+)\}", content)
    table_refs = re.findall(r"Table~?\\ref\{([^}]+)\}", content)

    return f"""Experimental Claims Analysis:
==============================

Claims found ({len(findings)}):
{chr(10).join(findings[:15])}
{"... and more" if len(findings) > 15 else ""}

Results section present: {"Yes" if has_results else "NO - MISSING!"}
Figure references: {len(figure_refs)}
Table references: {len(table_refs)}

Recommendation: Ensure each claim has corresponding evidence in Results section.
"""


@function_tool
def check_citation_coverage(
    topic_keywords: Annotated[str, "Comma-separated keywords to check citation coverage for"],
) -> str:
    """Check if the paper adequately cites relevant work for given topics.

    Analyzes the bibliography and paper content to identify potential
    citation gaps.
    """
    state = get_state()
    keywords = [k.strip().lower() for k in topic_keywords.split(",")]

    # Read current citations
    citations = {}
    if state.bib_path.exists():
        bib_content = state.bib_path.read_text(encoding="utf-8")
        # Parse BibTeX entries
        entries = re.findall(r"@\w+\{([^,]+),([^@]*)", bib_content, re.DOTALL)
        for key, entry_content in entries:
            title_match = re.search(r"title\s*=\s*\{([^}]+)\}", entry_content, re.IGNORECASE)
            if title_match:
                citations[key] = title_match.group(1).lower()

    # Check keyword coverage
    coverage = {}
    for keyword in keywords:
        matching_citations = [
            key for key, title in citations.items() if keyword in title
        ]
        coverage[keyword] = matching_citations

    # Read paper to see where citations appear
    tex_content = ""
    if state.tex_path.exists():
        tex_content = state.tex_path.read_text(encoding="utf-8")

    related_work_match = re.search(
        r"\\section\{Related Work\}(.*?)(\\section|\\end\{document\})",
        tex_content,
        re.DOTALL | re.IGNORECASE,
    )
    related_work_citations = 0
    if related_work_match:
        related_work_citations = len(
            re.findall(r"\\cite\{", related_work_match.group(1))
        )

    report = [f"Citation Coverage Analysis for: {', '.join(keywords)}", "=" * 50, ""]

    for keyword, cites in coverage.items():
        if cites:
            report.append(f"✓ '{keyword}': {len(cites)} citations - {', '.join(cites[:3])}")
        else:
            report.append(f"✗ '{keyword}': NO CITATIONS FOUND - consider adding!")

    report.append("")
    report.append(f"Total citations in bibliography: {len(citations)}")
    report.append(f"Citations in Related Work section: {related_work_citations}")

    if related_work_citations < 10:
        report.append("\n⚠️ Warning: Related Work section may need more citations.")

    return "\n".join(report)


@function_tool
def read_paper_section(
    section_name: Annotated[str, "Name of the section to read (e.g., 'Introduction')"],
) -> str:
    """Read a specific section of the paper for detailed review.

    Returns the full content of the specified section.
    """
    state = get_state()

    if not state.tex_path.exists():
        return "ERROR: No paper.tex found."

    content = state.tex_path.read_text(encoding="utf-8")

    # Find the section
    pattern = rf"\\section\{{{re.escape(section_name)}\}}(.*?)(\\section|\\end\{{document\}})"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        section_content = match.group(1).strip()
        return f"=== {section_name} ===\n\n{section_content}"

    # Try partial match
    sections = re.findall(r"\\section\{([^}]+)\}", content)
    matching = [s for s in sections if section_name.lower() in s.lower()]

    if matching:
        return f"Section '{section_name}' not found exactly. Did you mean: {', '.join(matching)}?"

    return f"Section '{section_name}' not found. Available sections: {', '.join(sections)}"


@function_tool
def get_experiment_results_summary() -> str:
    """Get a summary of available experiment results.

    Returns key metrics and findings from the experiments to help
    assess if claims are adequately supported.
    """
    state = get_state()

    if not state.experiment_results:
        return "No experiment results available."

    results = state.experiment_results

    summary = ["Experiment Results Summary", "=" * 30, ""]

    # Basic info
    if "experiment_name" in results:
        summary.append(f"Experiment: {results['experiment_name']}")
    if "group_name" in results:
        summary.append(f"Group: {results['group_name']}")

    # Benchmark summary
    if "benchmark_summary" in results:
        summary.append("\nBenchmark Results:")
        for key, value in results["benchmark_summary"].items():
            if isinstance(value, float):
                summary.append(f"  {key}: {value:.4f}")
            else:
                summary.append(f"  {key}: {value}")

    # Runs
    if "runs" in results:
        summary.append(f"\nRuns completed: {len(results['runs'])}")
        for run in results["runs"][:5]:
            summary.append(f"  - {run.get('id', 'unknown')}: {run.get('steps', '?')} steps")

    # Artifacts
    if "artifacts" in results:
        summary.append(f"\nArtifacts generated: {len(results['artifacts'])}")
        for name in list(results["artifacts"].keys())[:10]:
            summary.append(f"  - {name}")

    return "\n".join(summary)


# ============================================================================
# Experiment Proposal Tools
# ============================================================================


@function_tool
def propose_experiment(
    name: Annotated[str, "Short name for the experiment"],
    rationale: Annotated[str, "Why this experiment is needed"],
    hypothesis: Annotated[str, "What we expect to show/learn"],
    experiment_type: Annotated[str, "Type: 'ablation', 'comparison', 'scaling', 'robustness'"],
    key_variables: Annotated[str, "Comma-separated list of variables to test"],
    benchmarks: Annotated[str, "Comma-separated benchmarks: 'perplexity', 'latency', 'memory'"],
    priority: Annotated[int, "Priority 1-5 (1=highest)"] = 2,
) -> str:
    """Propose a new experiment to address a gap in the paper.

    Records the experiment proposal for later manifest generation.
    """
    # Store in state for later retrieval
    state = get_state()

    # Initialize proposals list if needed
    if not hasattr(state, "_proposed_experiments"):
        state._proposed_experiments = []  # type: ignore[attr-defined]

    proposal = {
        "name": name,
        "rationale": rationale,
        "hypothesis": hypothesis,
        "experiment_type": experiment_type,
        "key_variables": [v.strip() for v in key_variables.split(",")],
        "benchmarks": [b.strip() for b in benchmarks.split(",")],
        "priority": priority,
    }

    state._proposed_experiments.append(proposal)  # type: ignore[attr-defined]

    return f"""Experiment Proposed: {name}
--------------------------------
Type: {experiment_type}
Priority: {priority}/5
Variables: {key_variables}
Benchmarks: {benchmarks}

Rationale: {rationale}

Hypothesis: {hypothesis}

This experiment will be added to the review output for manifest generation.
"""


@function_tool
def generate_experiment_manifest(
    experiment_name: Annotated[str, "Name of the proposed experiment"],
    base_on_existing: Annotated[bool, "Whether to base on existing manifest"] = True,
) -> str:
    """Generate a YAML manifest for a proposed experiment.

    Creates a runnable experiment manifest based on the proposal.
    """
    state = get_state()

    # Find the proposal
    proposals = getattr(state, "_proposed_experiments", [])
    proposal = None
    for p in proposals:
        if p["name"] == experiment_name:
            proposal = p
            break

    if not proposal:
        return f"ERROR: No proposal found with name '{experiment_name}'"

    # Get base manifest if available
    base_config = {}
    if base_on_existing and state.manifest_path and state.manifest_path.exists():
        # Read existing manifest for reference
        manifest_content = state.manifest_path.read_text(encoding="utf-8")
        # Extract key values we might want to reuse
        base_config["reference"] = "Based on existing manifest"

    # Generate the manifest
    exp_type = proposal["experiment_type"]
    variables = proposal["key_variables"]
    benchmarks = proposal["benchmarks"]

    # Build benchmark configs
    benchmark_configs = []
    for bench in benchmarks:
        if bench == "perplexity":
            benchmark_configs.append("""      - id: perplexity
        config:
          type: perplexity
          dataset: "fineweb_100m.npy"
          block_size: 2048
          batch_size: 1
          num_batches: 100
        models: ["teacher", "student"]
        repeats: 1""")
        elif bench == "latency":
            benchmark_configs.append("""      - id: latency
        config:
          type: latency
          prompt_lengths: [128, 512, 1024, 2048]
          generation_lengths: [64, 128]
          batch_sizes: [1]
          warmup_runs: 3
          timed_runs: 10
        models: ["teacher", "student"]
        repeats: 1""")
        elif bench == "memory":
            benchmark_configs.append("""      - id: memory
        config:
          type: memory
          sequence_lengths: [512, 1024, 2048, 4096]
          batch_sizes: [1]
          measure_peak: true
          measure_kvcache: true
        models: ["teacher", "student"]
        repeats: 1""")

    # Build variable experiments based on type
    runs_config = ""
    if exp_type == "ablation":
        runs_config = f"""    # Ablation study: testing impact of {', '.join(variables)}
    runs:
      - id: baseline
        mode: train
        exp: {experiment_name}_baseline
        seed: 42
        steps: 1000
        train:
          phase: global
          batch_size: 1
          block_size: 2048
          lr: 0.00005
          device: mps
          dtype: float32"""
    elif exp_type == "scaling":
        runs_config = f"""    # Scaling study: varying {', '.join(variables)}
    runs:
      - id: scale_small
        mode: train
        exp: {experiment_name}_small
        seed: 42
        steps: 500
        train:
          phase: global
          batch_size: 1
          block_size: 1024
          lr: 0.0001
          device: mps
          dtype: float32
      - id: scale_medium
        mode: train
        exp: {experiment_name}_medium
        seed: 42
        steps: 500
        train:
          phase: global
          batch_size: 1
          block_size: 2048
          lr: 0.00005
          device: mps
          dtype: float32"""
    else:
        runs_config = f"""    runs:
      - id: experiment
        mode: train
        exp: {experiment_name}
        seed: 42
        steps: 1000
        train:
          phase: global
          batch_size: 1
          block_size: 2048
          lr: 0.00005
          device: mps
          dtype: float32"""

    manifest = f"""# ============================================================================
# Generated Experiment: {experiment_name}
# ============================================================================
# Rationale: {proposal['rationale']}
# Hypothesis: {proposal['hypothesis']}
# Type: {exp_type}
# Variables: {', '.join(variables)}
# ============================================================================

version: 1
name: {experiment_name.lower().replace(' ', '_')}
notes: "{proposal['rationale']}"

# Add to existing experiment or run standalone
groups:
  - name: {experiment_name.lower().replace(' ', '_')}
    description: "{proposal['hypothesis']}"
    data: "fineweb_100m.npy"

{runs_config}

    benchmarks:
{chr(10).join(benchmark_configs)}
"""

    # Save to output directory
    output_path = state.output_dir / f"proposed_{experiment_name.lower().replace(' ', '_')}.yml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest, encoding="utf-8")

    return f"""Generated manifest saved to: {output_path}

{manifest[:1500]}{"..." if len(manifest) > 1500 else ""}
"""


@function_tool
def submit_review(
    overall_score: Annotated[float, "Overall paper score 0-10"],
    recommendation: Annotated[str, "Recommendation: 'approve', 'style_fix', 'new_experiment', 'major_revision'"],
    summary: Annotated[str, "Executive summary of the review"],
    strengths: Annotated[str, "Comma-separated list of paper strengths"],
    weaknesses: Annotated[str, "Comma-separated list of paper weaknesses"],
) -> str:
    """Submit the final review with overall assessment.

    This should be called after all analysis and proposals are complete.
    """
    state = get_state()

    # Get proposed experiments
    proposals = getattr(state, "_proposed_experiments", [])

    review = {
        "overall_score": overall_score,
        "recommendation": recommendation,
        "summary": summary,
        "strengths": [s.strip() for s in strengths.split(",")],
        "weaknesses": [w.strip() for w in weaknesses.split(",")],
        "proposed_experiments": proposals,
        "needs_new_experiments": len(proposals) > 0,
    }

    # Save review to file
    review_path = state.output_dir / "review.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")

    # Store in state for retrieval
    state._review_result = review  # type: ignore[attr-defined]

    return f"""
╔══════════════════════════════════════════════════════════════════╗
║                      REVIEW SUBMITTED                            ║
╠══════════════════════════════════════════════════════════════════╣
║ Score: {overall_score:.1f}/10
║ Recommendation: {recommendation.upper()}
╠══════════════════════════════════════════════════════════════════╣
║ Summary:
║ {summary[:60]}...
╠══════════════════════════════════════════════════════════════════╣
║ Strengths: {len(review['strengths'])}
║ Weaknesses: {len(review['weaknesses'])}
║ Proposed Experiments: {len(proposals)}
╚══════════════════════════════════════════════════════════════════╝

Review saved to: {review_path}
"""


# ============================================================================
# Tool Collection
# ============================================================================

REVIEWER_TOOLS = [
    analyze_paper_structure,
    check_experimental_claims,
    check_citation_coverage,
    read_paper_section,
    get_experiment_results_summary,
    propose_experiment,
    generate_experiment_manifest,
    submit_review,
]
