"""
Unit tests for the agents module.
Covers: AgentState construction, CorrectiveAgent decision logic
        (confidence threshold, re-plan trigger, max-rounds guard),
        and _parse_json / _format_context helpers.
No live LLM, Neo4j, or Qdrant required.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_required_keys_present(self):
        from src.agents.state import AgentState
        keys = AgentState.__annotations__
        required = {
            "messages", "session_id", "service_mode",
            "original_query", "enhanced_query", "query_intent",
            "selected_databases", "temporal_context",
            "plan", "reasoning_trace",
            "retrieved_docs", "graph_context", "web_results", "retrieval_sources",
            "draft_answer", "final_answer", "citations",
            "confidence_score", "needs_correction", "correction_round",
            "tool_calls", "tool_results",
            "memory_context",
            "next_node", "error", "iteration_count",
        }
        missing = required - set(keys)
        assert not missing, f"AgentState missing keys: {missing}"

    def test_service_mode_annotation_is_str(self):
        from src.agents.state import AgentState
        assert AgentState.__annotations__["service_mode"] is str

    def test_correction_round_annotation_is_int(self):
        from src.agents.state import AgentState
        assert AgentState.__annotations__["correction_round"] is int

    def test_confidence_score_annotation_is_float(self):
        from src.agents.state import AgentState
        assert AgentState.__annotations__["confidence_score"] is float


# ---------------------------------------------------------------------------
# CorrectiveAgent — helpers
# ---------------------------------------------------------------------------

class TestCorrectiveAgentHelpers:
    @pytest.fixture
    def agent(self):
        from src.agents.corrective_agent import CorrectiveAgent
        mock_llm = MagicMock()
        return CorrectiveAgent(llm=mock_llm, quality_threshold=0.6, max_rounds=3)

    def test_parse_json_clean(self, agent):
        raw = '{"overall_score": 0.8, "relevance_fraction": 0.9}'
        result = agent._parse_json(raw)
        assert result["overall_score"] == 0.8
        assert result["relevance_fraction"] == 0.9

    def test_parse_json_strips_markdown_fences(self, agent):
        raw = "```json\n{\"overall_score\": 0.75}\n```"
        result = agent._parse_json(raw)
        assert result["overall_score"] == 0.75

    def test_parse_json_handles_extra_text(self, agent):
        raw = 'Sure, here is the JSON: {"overall_score": 0.5} end'
        result = agent._parse_json(raw)
        assert result["overall_score"] == 0.5

    def test_parse_json_returns_empty_dict_on_failure(self, agent):
        result = agent._parse_json("not json at all")
        assert result == {}

    def test_format_context_truncates_long_docs(self, agent):
        from langchain_core.documents import Document
        # format_context_for_llm caps each doc at 1500 chars and total at 12k.
        doc = Document(page_content="x" * 5000)
        context = agent._format_context([doc])
        # Allow for the [Source N (...)] header overhead (~80 chars)
        assert len(context) < 1700

    def test_format_context_no_docs(self, agent):
        result = agent._format_context([])
        assert result == "No context"

    def test_format_context_numbers_docs(self, agent):
        from langchain_core.documents import Document
        # Aligned with reasoning_agent: format_context_for_llm uses
        # "[Source N (...)]" headers (not "[N]") so the corrective evaluator
        # sees the same numbering the synthesis LLM cited against.
        docs = [Document(page_content=f"Doc {i}") for i in range(3)]
        result = agent._format_context(docs)
        assert "Source 1" in result
        assert "Source 2" in result
        assert "Source 3" in result


# ---------------------------------------------------------------------------
# CorrectiveAgent — evaluate() decision logic
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> Dict[str, Any]:
    base = {
        "original_query": "What caused the beam loss?",
        "enhanced_query": "What caused the beam loss in sector 35?",
        "draft_answer": "The beam loss was caused by a kicker fault.",
        "retrieved_docs": [],
        "correction_round": 0,
        "confidence_score": 0.0,
        "needs_correction": False,
    }
    base.update(overrides)
    return base


def _mock_llm_response(overall_score, relevance_fraction, refined_query=None):
    payload = {
        "relevance_fraction": relevance_fraction,
        "factual_grounding": 0.8,
        "completeness": 0.7,
        "overall_score": overall_score,
        "issues": [],
        "refined_query": refined_query,
    }
    mock = MagicMock()
    mock.content = json.dumps(payload)
    return mock


class TestCorrectiveAgentEvaluate:
    @pytest.fixture
    def agent(self):
        from src.agents.corrective_agent import CorrectiveAgent
        return CorrectiveAgent(
            llm=MagicMock(),
            quality_threshold=0.6,
            irrelevant_threshold=0.5,
            max_rounds=3,
        )

    def test_high_confidence_accepts_draft(self, agent):
        agent.llm.invoke.return_value = _mock_llm_response(
            overall_score=0.85, relevance_fraction=0.9
        )
        result = agent.evaluate(_make_state())
        assert result["needs_correction"] is False
        assert result["final_answer"] == "The beam loss was caused by a kicker fault."
        assert result["confidence_score"] == pytest.approx(0.85)

    def test_low_confidence_triggers_correction(self, agent):
        agent.llm.invoke.return_value = _mock_llm_response(
            overall_score=0.4, relevance_fraction=0.9, refined_query="refined query"
        )
        result = agent.evaluate(_make_state())
        assert result["needs_correction"] is True
        assert result["correction_round"] == 1
        assert result["enhanced_query"] == "refined query"

    def test_high_irrelevance_triggers_correction(self, agent):
        # relevance_fraction=0.3 → 70% irrelevant (> 50% threshold)
        agent.llm.invoke.return_value = _mock_llm_response(
            overall_score=0.75, relevance_fraction=0.3, refined_query="better query"
        )
        result = agent.evaluate(_make_state())
        assert result["needs_correction"] is True

    def test_exactly_at_threshold_does_not_trigger(self, agent):
        # overall_score=0.6 == quality_threshold → should NOT trigger correction
        agent.llm.invoke.return_value = _mock_llm_response(
            overall_score=0.6, relevance_fraction=0.6
        )
        result = agent.evaluate(_make_state())
        assert result["needs_correction"] is False

    def test_max_rounds_forces_accept(self, agent):
        # correction_round already at max — must accept regardless of score
        state = _make_state(correction_round=3, draft_answer="best we have")
        result = agent.evaluate(state)
        assert result["needs_correction"] is False
        assert result["final_answer"] == "best we have"
        # Phase 0 fail-closed: returns max(0.3, state.confidence_score). State
        # default confidence_score is 0.0, so result floor is 0.3.
        assert result["confidence_score"] == 0.3
        assert result["quality_status"] == "max_rounds_exhausted"
        # LLM should NOT be called
        agent.llm.invoke.assert_not_called()

    def test_correction_round_incremented(self, agent):
        agent.llm.invoke.return_value = _mock_llm_response(
            overall_score=0.2, relevance_fraction=0.2, refined_query="try again"
        )
        state = _make_state(correction_round=1)
        result = agent.evaluate(state)
        assert result["correction_round"] == 2

    def test_llm_failure_forces_correction_when_rounds_remain(self, agent):
        # Phase 0 fail-closed: evaluator error triggers another correction
        # round (rather than silently accepting an unverified draft) as long
        # as max_rounds hasn't been exhausted.
        agent.llm.invoke.side_effect = RuntimeError("LLM timeout")
        result = agent.evaluate(_make_state(correction_round=0))
        assert result["needs_correction"] is True
        assert result["confidence_score"] == 0.0
        assert result["quality_status"] == "evaluator_error"
        assert result["correction_round"] == 1

    def test_llm_failure_at_last_round_accepts_unverified(self, agent):
        # Phase 0 fail-closed last-resort: if we're already at max_rounds-1,
        # we have no more rounds to try, so accept but flag as unverified.
        agent.llm.invoke.side_effect = RuntimeError("LLM timeout")
        result = agent.evaluate(_make_state(correction_round=2))  # max=3
        assert result["needs_correction"] is False
        assert result["quality_status"] == "unverified"

    def test_no_refined_query_accepts_even_if_low_score(self, agent):
        # Low score but LLM gave no refined_query → accept draft
        agent.llm.invoke.return_value = _mock_llm_response(
            overall_score=0.3, relevance_fraction=0.9, refined_query=None
        )
        result = agent.evaluate(_make_state())
        assert result["needs_correction"] is False
        assert "final_answer" in result

    def test_kg_validation_skipped_without_neo4j(self, agent):
        agent.neo4j = None
        agent.llm.invoke.return_value = _mock_llm_response(0.9, 0.9)
        # Should complete without error even though _validate_kg_entities would need neo4j
        result = agent.evaluate(_make_state())
        assert "confidence_score" in result
