"""Extended unit tests for CorrectiveAgent.

Existing `test_agents.py` covers `_parse_json` and a couple of evaluate paths.
This file adds:
  - KG entity validation (with mocked Neo4j)
  - Hallucination cross-check via the LLM evaluation prompt
  - Irrelevant-doc threshold trigger
  - Quality-threshold trigger
  - Max-rounds boundary (force-accept with degraded status)
  - Evaluator failure / fail-closed behavior
  - _escalate_strategy mapping
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.corrective_agent import CorrectiveAgent


def _docs(n: int = 3) -> list[Document]:
    return [
        Document(page_content=f"text {i}", metadata={"id": f"doc_{i}", "collection": "entry_logbook"})
        for i in range(n)
    ]


def _state(**overrides):
    base = {
        "original_query": "What happened?",
        "enhanced_query": "What happened in the run?",
        "draft_answer": "Beam current dropped at 03:00 [source_1].",
        "retrieved_docs": _docs(3),
        "memory_context": {},
        "correction_round": 0,
        "retrieval_strategy": "HYBRID_3WAY",
        "extracted_entities": [],
        "reasoning_trace": [],
    }
    base.update(overrides)
    return base


def _llm_returning(payload: dict):
    """Build a mock LLM whose `.invoke()` returns a message with JSON content."""
    m = MagicMock()
    m.invoke.return_value = MagicMock(content=json.dumps(payload))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Threshold triggers
# ─────────────────────────────────────────────────────────────────────────────
class TestQualityThresholdTrigger:
    def test_low_score_triggers_correction(self):
        llm = _llm_returning({
            "overall_score": 0.3, "relevance_fraction": 0.9,
            "issues": ["unsupported claim"], "refined_query": "What time did beam drop?",
        })
        agent = CorrectiveAgent(llm=llm, quality_threshold=0.5, irrelevant_threshold=0.5, max_rounds=3)
        result = agent.evaluate(_state())
        assert result["needs_correction"] is True
        assert result["correction_round"] == 1
        assert result["enhanced_query"] == "What time did beam drop?"
        assert result["quality_status"] == "needs_correction"

    def test_high_score_passes(self):
        llm = _llm_returning({"overall_score": 0.85, "relevance_fraction": 0.9, "issues": []})
        agent = CorrectiveAgent(llm=llm, quality_threshold=0.4, max_rounds=3)
        result = agent.evaluate(_state())
        assert result["needs_correction"] is False
        assert result["quality_status"] == "verified"
        assert result["final_answer"]


class TestRelevanceThresholdTrigger:
    def test_high_irrelevant_fraction_triggers_correction(self):
        # relevance=0.4 → irrelevant_fraction=0.6 > threshold (0.5)
        llm = _llm_returning({
            "overall_score": 0.7, "relevance_fraction": 0.4,
            "issues": ["docs off-topic"], "refined_query": "different angle",
        })
        agent = CorrectiveAgent(llm=llm, quality_threshold=0.4, irrelevant_threshold=0.5, max_rounds=3)
        result = agent.evaluate(_state())
        assert result["needs_correction"] is True

    def test_low_irrelevant_fraction_passes(self):
        # relevance=0.9 → irrelevant_fraction=0.1 < threshold
        llm = _llm_returning({"overall_score": 0.6, "relevance_fraction": 0.9, "issues": []})
        agent = CorrectiveAgent(llm=llm, quality_threshold=0.4, irrelevant_threshold=0.5, max_rounds=3)
        result = agent.evaluate(_state())
        assert result["needs_correction"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Max-rounds boundary
# ─────────────────────────────────────────────────────────────────────────────
class TestMaxRoundsBoundary:
    def test_max_rounds_force_accepts_with_degraded_status(self):
        llm = _llm_returning({"overall_score": 0.05, "relevance_fraction": 0.0, "issues": []})
        agent = CorrectiveAgent(llm=llm, quality_threshold=0.4, max_rounds=3)
        result = agent.evaluate(_state(correction_round=3, draft_answer="weak draft"))
        assert result["quality_status"] == "max_rounds_exhausted"
        assert result["final_answer"] == "weak draft"
        assert result["needs_correction"] is False
        # LLM should NOT have been invoked — early exit
        llm.invoke.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator failure (fail-closed)
# ─────────────────────────────────────────────────────────────────────────────
class TestEvaluatorFailure:
    def test_llm_exception_below_max_triggers_retry(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM timeout")
        agent = CorrectiveAgent(llm=llm, quality_threshold=0.4, max_rounds=3)
        result = agent.evaluate(_state(correction_round=0))
        assert result["needs_correction"] is True
        assert result["quality_status"] == "evaluator_error"
        assert result["confidence_score"] == 0.0

    def test_llm_exception_at_last_round_marks_unverified(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM timeout")
        agent = CorrectiveAgent(llm=llm, quality_threshold=0.4, max_rounds=3)
        # correction_round=2 means next would be 3 == max_rounds → no more retries
        result = agent.evaluate(_state(correction_round=2))
        assert result["needs_correction"] is False
        assert result["quality_status"] == "unverified"
        assert result["final_answer"]


# ─────────────────────────────────────────────────────────────────────────────
# KG entity validation
# ─────────────────────────────────────────────────────────────────────────────
class TestKGValidation:
    def test_three_unverified_entities_force_correction(self):
        llm = _llm_returning({
            "overall_score": 0.9, "relevance_fraction": 1.0,
            "issues": [], "refined_query": "follow-up",
        })
        neo4j = MagicMock()
        neo4j.fulltext_search.return_value = []  # nothing exists in KG
        agent = CorrectiveAgent(llm=llm, neo4j_client=neo4j, quality_threshold=0.4, max_rounds=3)
        # Pass 3 named entities that the KG doesn't recognize
        state = _state(extracted_entities=["Phantom1", "Ghost2", "Spectre3"])
        result = agent.evaluate(state)
        # Even though score is high, 3+ unverified KG entities forces correction
        assert result["needs_correction"] is True
        # KG penalty should also reduce overall score: 0.9 - (3 * 0.05) = 0.75
        assert result["confidence_score"] == pytest.approx(0.75, abs=0.01)

    def test_kg_validation_skipped_when_no_neo4j(self):
        llm = _llm_returning({"overall_score": 0.9, "relevance_fraction": 1.0, "issues": []})
        agent = CorrectiveAgent(llm=llm, neo4j_client=None, quality_threshold=0.4, max_rounds=3)
        result = agent.evaluate(_state(extracted_entities=["Anything"]))
        # No KG = no penalty, no issues; high score passes
        assert result["needs_correction"] is False
        assert result["confidence_score"] == 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Strategy escalation
# ─────────────────────────────────────────────────────────────────────────────
class TestStrategyEscalation:
    @pytest.fixture
    def agent(self):
        return CorrectiveAgent(llm=MagicMock(), max_rounds=3)

    def test_round_0_keeps_strategy(self, agent):
        assert agent._escalate_strategy("HYBRID_3WAY", 0) == "HYBRID_3WAY"
        assert agent._escalate_strategy("VECTOR_ONLY", 0) == "VECTOR_ONLY"

    def test_round_0_escalates_kg_only_immediately(self, agent):
        # KG traversal is deterministic on the same query, so re-retrieving
        # at round 0 with KG_ONLY pulls the identical doc set — a no-op
        # round that doubles latency without changing the answer
        # (observed on AR-04 in the 50-q eval). Escalate immediately.
        assert agent._escalate_strategy("KG_ONLY", 0) == "HYBRID_3WAY"

    def test_round_1_swaps_to_complementary(self, agent):
        assert agent._escalate_strategy("HYBRID_3WAY", 1) == "VECTOR_ONLY"
        assert agent._escalate_strategy("VECTOR_ONLY", 1) == "HYBRID_3WAY"
        # KG_ONLY → HYBRID_3WAY (avoiding empty KG result trap)
        assert agent._escalate_strategy("KG_ONLY", 1) == "HYBRID_3WAY"

    def test_final_round_uses_max_coverage(self, agent):
        assert agent._escalate_strategy("VECTOR_ONLY", 2) == "HYBRID_3WAY"
        assert agent._escalate_strategy("KG_ONLY", 2) == "HYBRID_3WAY"
