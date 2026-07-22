"""
Unit tests for UI modules: ResponseFormatter and ServiceSelector.
No live services required.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# ResponseFormatter — citations
# ---------------------------------------------------------------------------

class TestFormatCitations:
    @pytest.fixture
    def formatter(self):
        from src.ui.response_formatter import ResponseFormatter
        return ResponseFormatter()

    def test_empty_docs_returns_empty(self, formatter):
        assert formatter.format_citations([]) == ""

    def test_single_citation(self, formatter):
        docs = [{"author": "Doe", "title": "RF Study", "date": "2025-01-15"}]
        result = formatter.format_citations(docs)
        assert "[1]" in result
        assert "Doe" in result
        assert "RF Study" in result

    def test_multiple_citations_numbered(self, formatter):
        docs = [
            {"author": "Alice", "title": "Doc A"},
            {"author": "Bob", "title": "Doc B"},
            {"author": "Charlie", "title": "Doc C"},
        ]
        result = formatter.format_citations(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_missing_author_shows_unknown(self, formatter):
        docs = [{"title": "No Author Doc"}]
        result = formatter.format_citations(docs)
        assert "Unknown" in result

    def test_missing_title_shows_untitled(self, formatter):
        docs = [{"author": "Alice"}]
        result = formatter.format_citations(docs)
        assert "Untitled" in result

    def test_alternate_key_names(self, formatter):
        docs = [{"Author": "Smith", "Title": "Big Study", "Date": "2025-06"}]
        result = formatter.format_citations(docs)
        assert "Smith" in result
        assert "Big Study" in result

    def test_source_as_title_fallback(self, formatter):
        docs = [{"source": "logbook_entry_123"}]
        result = formatter.format_citations(docs)
        assert "logbook_entry_123" in result


# ---------------------------------------------------------------------------
# ResponseFormatter — answer with citations
# ---------------------------------------------------------------------------

class TestFormatAnswerWithCitations:
    @pytest.fixture
    def formatter(self):
        from src.ui.response_formatter import ResponseFormatter
        return ResponseFormatter()

    def test_no_docs_returns_answer_only(self, formatter):
        result = formatter.format_answer_with_citations("The answer is 42.", [])
        assert result == "The answer is 42."

    def test_with_docs_appends_references(self, formatter):
        docs = [{"author": "Alice", "title": "Study"}]
        result = formatter.format_answer_with_citations("Answer text.", docs)
        assert result.startswith("Answer text.")
        assert "**References:**" in result
        assert "Alice" in result


# ---------------------------------------------------------------------------
# ResponseFormatter — Mermaid KG subgraph
# ---------------------------------------------------------------------------

class TestFormatMermaidKG:
    @pytest.fixture
    def formatter(self):
        from src.ui.response_formatter import ResponseFormatter
        return ResponseFormatter()

    def test_empty_graph(self, formatter):
        result = formatter.format_mermaid_kg_subgraph([], [])
        assert "```mermaid" in result
        assert "graph TD" in result
        assert result.endswith("```")

    def test_dict_nodes(self, formatter):
        nodes = [
            {"id": "person_1", "label": "Doe"},
            {"id": "system_1", "name": "SR RF"},
        ]
        result = formatter.format_mermaid_kg_subgraph(nodes, [])
        assert 'person_1["Doe"]' in result
        assert 'system_1["SR RF"]' in result

    def test_string_nodes(self, formatter):
        nodes = ["Beam Loss", "Sector 35"]
        result = formatter.format_mermaid_kg_subgraph(nodes, [])
        assert 'Beam_Loss["Beam Loss"]' in result

    def test_dict_edges(self, formatter):
        nodes = [{"id": "A"}, {"id": "B"}]
        edges = [{"source": "A", "target": "B", "relation": "CAUSED_BY"}]
        result = formatter.format_mermaid_kg_subgraph(nodes, edges)
        assert 'A -->|"CAUSED_BY"| B' in result

    def test_tuple_edges(self, formatter):
        nodes = [{"id": "A"}, {"id": "B"}]
        edges = [("A", "B", "WORKS_ON")]
        result = formatter.format_mermaid_kg_subgraph(nodes, edges)
        assert 'A -->|"WORKS_ON"| B' in result

    def test_tuple_edge_without_label(self, formatter):
        edges = [("A", "B")]
        result = formatter.format_mermaid_kg_subgraph([], edges)
        assert 'A -->|""| B' in result


# ---------------------------------------------------------------------------
# ResponseFormatter — confidence
# ---------------------------------------------------------------------------

class TestFormatConfidence:
    @pytest.fixture
    def formatter(self):
        from src.ui.response_formatter import ResponseFormatter
        return ResponseFormatter()

    def test_high_confidence(self, formatter):
        result = formatter.format_confidence(0.9)
        assert "High" in result
        assert "90%" in result

    def test_medium_confidence(self, formatter):
        result = formatter.format_confidence(0.65)
        assert "Medium" in result

    def test_low_confidence(self, formatter):
        result = formatter.format_confidence(0.45)
        assert "Low" in result

    def test_very_low_confidence(self, formatter):
        result = formatter.format_confidence(0.1)
        assert "Very Low" in result

    def test_zero_confidence(self, formatter):
        result = formatter.format_confidence(0.0)
        assert "Very Low" in result
        assert "0%" in result

    def test_exactly_at_threshold(self, formatter):
        result = formatter.format_confidence(0.8)
        assert "High" in result

    def test_just_below_threshold(self, formatter):
        result = formatter.format_confidence(0.79)
        assert "Medium" in result


# ---------------------------------------------------------------------------
# ServiceSelector
# ---------------------------------------------------------------------------

class TestServiceSelector:
    @pytest.fixture
    def selector(self):
        from src.ui.service_selector import ServiceSelector
        return ServiceSelector()

    def test_unknown_mode_falls_back_to_rag(self, selector):
        mock_settings = MagicMock()
        mock_components = {}
        with patch("src.ui.service_selector.ServiceSelector._create_service") as mock_create:
            mock_service = MagicMock()
            mock_create.return_value = mock_service
            result = selector.get_service("nonexistent_mode", mock_settings, mock_components)
            mock_create.assert_called_with("rag", mock_settings, mock_components)

    def test_caches_service_instance(self, selector):
        mock_settings = MagicMock()
        mock_components = {}
        with patch("src.ui.service_selector.ServiceSelector._create_service") as mock_create:
            mock_service = MagicMock()
            mock_create.return_value = mock_service
            result1 = selector.get_service("rag", mock_settings, mock_components)
            result2 = selector.get_service("rag", mock_settings, mock_components)
            # Should only create once
            assert mock_create.call_count == 1
            assert result1 is result2

    def test_different_modes_get_separate_instances(self, selector):
        mock_settings = MagicMock()
        mock_components = {}
        with patch("src.ui.service_selector.ServiceSelector._create_service") as mock_create:
            mock_create.side_effect = [MagicMock(), MagicMock()]
            r1 = selector.get_service("rag", mock_settings, mock_components)
            r2 = selector.get_service("daily_summary", mock_settings, mock_components)
            assert r1 is not r2
            assert mock_create.call_count == 2

    def test_all_supported_modes(self, selector):
        from src.ui.service_selector import _SUPPORTED_MODES
        expected = {"rag", "agentic_rag", "daily_summary", "troubleshooting", "studies_copilot"}
        assert set(_SUPPORTED_MODES) == expected
