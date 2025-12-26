"""Tests for the paper drafting module."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caramba.config.paper import (
    CitationConfig,
    PaperConfig,
    PaperSection,
    PaperType,
)
from caramba.paper.drafter import PaperDrafter
from caramba.paper.tools import (
    Citation,
    PaperState,
    SearchResult,
    _create_backup,
    _find_sections,
    get_state,
    set_state,
)


class TestPaperConfig:
    """Test PaperConfig validation and defaults."""

    def test_default_config(self) -> None:
        """Test default paper configuration."""
        config = PaperConfig(title="Test Paper")

        assert config.enabled is True
        assert config.title == "Test Paper"
        assert config.paper_type == PaperType.PAPER
        assert config.model == "gpt-4o"
        assert len(config.sections) == 8

    def test_custom_config(self) -> None:
        """Test custom paper configuration."""
        config = PaperConfig(
            title="Custom Paper",
            authors=["Author 1", "Author 2"],
            paper_type=PaperType.ARXIV,
            sections=[PaperSection.ABSTRACT, PaperSection.CONCLUSION],
            model="gpt-4",
        )

        assert config.title == "Custom Paper"
        assert len(config.authors) == 2
        assert config.paper_type == PaperType.ARXIV
        assert len(config.sections) == 2

    def test_citation_config_defaults(self) -> None:
        """Test citation configuration defaults."""
        config = CitationConfig()

        assert config.enabled is True
        assert config.max_citations == 20
        assert "arxiv" in config.sources


class TestPaperTools:
    """Test paper drafting tools."""

    def test_find_sections(self) -> None:
        """Test finding sections in LaTeX content."""
        tex = r"""
\section{Introduction}
Some text.
\section{Related Work}
More text.
\section{Methodology}
Even more text.
"""
        sections = _find_sections(tex)
        assert sections == ["Introduction", "Related Work", "Methodology"]

    def test_create_backup(self) -> None:
        """Test backup creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tex_file = tmp_path / "paper.tex"
            tex_file.write_text("Test content")

            _create_backup(tex_file, max_versions=5)

            backup_dir = tmp_path / "versions"
            assert backup_dir.exists()
            backups = list(backup_dir.glob("*.tex"))
            assert len(backups) == 1

    def test_paper_state(self) -> None:
        """Test PaperState properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PaperConfig(title="Test")
            state = PaperState(
                output_dir=Path(tmpdir),
                paper_config=config,
            )

            assert state.tex_path == Path(tmpdir) / "paper.tex"
            assert state.bib_path == Path(tmpdir) / "references.bib"
            assert state.figures_dir == Path(tmpdir) / "figures"

    def test_set_and_get_state(self) -> None:
        """Test global state management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PaperConfig(title="Test")
            state = PaperState(
                output_dir=Path(tmpdir),
                paper_config=config,
            )

            set_state(state)
            retrieved = get_state()

            assert retrieved is state


class TestCitation:
    """Test Citation model."""

    def test_citation_creation(self) -> None:
        """Test creating a citation."""
        citation = Citation(
            key="smith2024_test",
            title="A Test Paper",
            authors=["John Smith", "Jane Doe"],
            year=2024,
            venue="NeurIPS",
            doi="10.1234/test",
        )

        assert citation.key == "smith2024_test"
        assert len(citation.authors) == 2
        assert citation.venue == "NeurIPS"


class TestSearchResult:
    """Test SearchResult model."""

    def test_empty_search_result(self) -> None:
        """Test empty search result."""
        result = SearchResult(
            citations=[],
            query="test",
            source="arxiv",
            total_results=0,
        )

        assert len(result.citations) == 0
        assert result.source == "arxiv"

    def test_search_result_with_citations(self) -> None:
        """Test search result with citations."""
        citations = [
            Citation(
                key="test1",
                title="Paper 1",
                authors=["Author 1"],
                year=2024,
                venue="ICML",
            ),
            Citation(
                key="test2",
                title="Paper 2",
                authors=["Author 2"],
                year=2023,
                venue="NeurIPS",
            ),
        ]

        result = SearchResult(
            citations=citations,
            query="machine learning",
            source="semantic_scholar",
            total_results=100,
        )

        assert len(result.citations) == 2
        assert result.total_results == 100


class TestPaperDrafter:
    """Test PaperDrafter class."""

    def test_drafter_initialization(self) -> None:
        """Test drafter initialization."""
        config = PaperConfig(
            title="Test Paper",
            authors=["Test Author"],
        )

        drafter = PaperDrafter(config)

        assert drafter.config == config
        assert drafter.agent.name == "Paper Drafter"

    def test_drafter_custom_output_dir(self) -> None:
        """Test drafter with custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PaperConfig(title="Test")
            drafter = PaperDrafter(config, output_dir=tmpdir)

            assert drafter.output_dir == Path(tmpdir)

    def test_format_sections(self) -> None:
        """Test section formatting for prompts."""
        config = PaperConfig(
            title="Test",
            sections=[
                PaperSection.ABSTRACT,
                PaperSection.INTRODUCTION,
                PaperSection.CONCLUSION,
            ],
        )
        drafter = PaperDrafter(config)

        formatted = drafter._format_sections()

        assert "Abstract" in formatted
        assert "Introduction" in formatted
        assert "Conclusion" in formatted

    def test_build_create_prompt(self) -> None:
        """Test creating the initial paper prompt."""
        config = PaperConfig(
            title="My Paper Title",
            authors=["John Doe"],
            keywords=["AI", "ML"],
        )
        drafter = PaperDrafter(config)

        prompt = drafter._build_create_prompt(None, None, None)

        assert "My Paper Title" in prompt
        assert "John Doe" in prompt
        assert "AI" in prompt

    def test_build_update_prompt(self) -> None:
        """Test creating the update paper prompt."""
        config = PaperConfig(title="Existing Paper")
        drafter = PaperDrafter(config)

        prompt = drafter._build_update_prompt(None, None, None)

        assert "update" in prompt.lower()
        assert "Existing Paper" in prompt
