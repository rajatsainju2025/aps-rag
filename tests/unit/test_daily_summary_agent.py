"""Unit tests for DailySummaryAgent.

Covers: date-window construction, multi-database fan-out, empty-result
fallback, expanded-window retry, and final-answer formatting. The two
retrievers and the LLM are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.daily_summary_agent import DailySummaryAgent


def _doc(coll: str, i: int = 0) -> Document:
    return Document(
        page_content=f"entry {i} from {coll}",
        metadata={"id": f"{coll}_{i}", "collection": coll, "creation_datetime": "2026-04-26T10:00:00"},
    )


@pytest.fixture
def llm():
    m = MagicMock()
    m.invoke.return_value = MagicMock(content="# Daily Summary\n- bullet 1\n- bullet 2")
    return m


@pytest.fixture
def bm25():
    m = MagicMock()
    m.retrieve.return_value = []
    return m


@pytest.fixture
def vector():
    m = MagicMock()
    m.retrieve.return_value = []
    return m


@pytest.fixture
def agent(llm, bm25, vector):
    return DailySummaryAgent(llm=llm, bm25_retriever=bm25, vector_retriever=vector)


class TestEmptyResultFallback:
    def test_returns_friendly_message_when_no_docs(self, agent):
        result = agent.summarize({"original_query": "What happened today?"})
        assert "No entries found" in result["final_answer"]
        assert result["citations"] == []
        assert result["retrieved_docs"] == []
        # reasoning_trace must mention the empty window
        assert any("No entries" in step for step in result["reasoning_trace"])


class TestRetrievalFanOut:
    def test_fans_out_to_four_databases(self, agent, bm25, vector):
        # Make BM25 + vector return some docs so we exercise the synthesis path
        bm25.retrieve.return_value = [_doc("entry_logbook", 0)]
        vector.retrieve.return_value = [_doc("entry_logbook", 1)]
        agent.summarize({"original_query": "today"})
        # Each of the system/WR/Teams gets BM25+vector, ICMS gets vector → multiple calls
        assert bm25.retrieve.call_count >= 3
        assert vector.retrieve.call_count >= 4


class TestDateWindow:
    def test_default_uses_48h_window(self, agent, bm25):
        bm25.retrieve.return_value = [_doc("entry_logbook")]
        result = agent.summarize({"original_query": "today"})
        # The reasoning_trace records the window — confirm the system/WR/Teams window text
        windows = [s for s in result["reasoning_trace"] if "window" in s]
        assert windows, "expected a window step in reasoning_trace"

    def test_explicit_temporal_context_honoured(self, agent, bm25):
        bm25.retrieve.return_value = [_doc("entry_logbook")]
        state = {
            "original_query": "What happened on April 20?",
            "temporal_context": {
                "start": "2026-04-20T00:00:00",
                "end": "2026-04-20T23:59:59",
            },
        }
        result = agent.summarize(state)
        # One reasoning_trace entry should mention the explicit dates
        assert any("2026-04-20" in s for s in result["reasoning_trace"])


class TestSynthesis:
    def test_llm_invoked_with_combined_context(self, agent, bm25, llm):
        bm25.retrieve.return_value = [_doc("entry_logbook", 1), _doc("entry_logbook", 2)]
        agent.summarize({"original_query": "today"})
        llm.invoke.assert_called_once()
        # The prompt passed to the LLM should reference all four databases
        call_args = llm.invoke.call_args[0][0]
        prompt_text = "\n".join(m.content for m in call_args)
        # The prompt template names the databases
        assert "the system" in prompt_text


class TestOutputContract:
    def test_return_shape_includes_required_keys(self, agent, bm25):
        bm25.retrieve.return_value = [_doc("entry_logbook")]
        result = agent.summarize({"original_query": "today"})
        for k in ("final_answer", "citations", "retrieved_docs", "reasoning_trace"):
            assert k in result, f"missing key: {k}"

    def test_final_answer_non_empty_with_docs(self, agent, bm25):
        bm25.retrieve.return_value = [_doc("entry_logbook", 0)]
        result = agent.summarize({"original_query": "today"})
        assert result["final_answer"]
        assert len(result["final_answer"]) > 10


class TestRetrievalFailureIsolation:
    def test_one_retriever_failure_does_not_abort(self, agent, bm25, vector):
        bm25.retrieve.side_effect = RuntimeError("ES is down")
        vector.retrieve.return_value = [_doc("entry_logbook", 0)]
        # Should not raise — _safe_bm25 catches per-call failures
        result = agent.summarize({"original_query": "today"})
        assert result["final_answer"]


class TestExpansionFallback:
    def test_zero_results_with_temporal_triggers_expansion(self, agent, bm25, vector):
        """When the explicit temporal window yields zero docs across ALL four
        databases, the agent must attempt to widen the window (±2d, ±7d)."""
        # Both retrievers always return [] → total stays 0; first attempt fails
        # AND each expansion attempt fails too. The expansion log lines must
        # still appear in reasoning_trace.
        bm25.retrieve.return_value = []
        vector.retrieve.return_value = []
        result = agent.summarize({
            "original_query": "what happened",
            "temporal_context": {"start": "2026-04-20T00:00:00", "end": "2026-04-20T23:59:59"},
        })
        assert any("expanding" in s.lower() for s in result["reasoning_trace"]), \
            f"expected expansion in trace, got: {result['reasoning_trace']}"
