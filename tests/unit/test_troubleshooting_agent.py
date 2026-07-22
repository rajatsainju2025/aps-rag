"""Unit tests for TroubleshootingAgent.diagnose."""
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

from src.agents.troubleshooting_agent import TroubleshootingAgent


def _doc(coll: str, i: int = 0) -> Document:
    return Document(page_content=f"text {i}", metadata={"id": f"{coll}_{i}", "collection": coll})


@pytest.fixture
def llm():
    """Default LLM mock — returns a runbook-shaped response."""
    m = MagicMock()
    # _extract_symptoms uses a JSON shape; subsequent runbook generation uses prose
    responses = iter([
        # symptom extraction
        MagicMock(content=json.dumps({
            "symptoms": ["beam loss"],
            "systems": ["SR"],
            "components": ["BPM"],
        })),
        # runbook
        MagicMock(content="## Runbook\n1. Check BPM\n2. Verify SR alignment"),
    ])
    m.invoke.side_effect = lambda *args, **kwargs: next(responses)
    return m


@pytest.fixture
def neo4j():
    n = MagicMock()
    n.fulltext_search.return_value = []
    n.run_cypher.return_value = []
    return n


@pytest.fixture
def vector():
    m = MagicMock()
    m.retrieve.return_value = []
    return m


@pytest.fixture
def bm25():
    m = MagicMock()
    m.retrieve.return_value = []
    return m


@pytest.fixture
def agent(llm, neo4j, vector, bm25):
    return TroubleshootingAgent(
        llm=llm,
        neo4j_client=neo4j,
        vector_retriever=vector,
        bm25_retriever=bm25,
    )


class TestDiagnoseHappyPath:
    def test_returns_required_keys(self, agent):
        result = agent.diagnose({"original_query": "SR beam loss after kicker fire"})
        for key in ("final_answer", "retrieved_docs", "graph_context"):
            assert key in result, f"missing key: {key}"
        assert result["final_answer"]


class TestSymptomExtraction:
    def test_symptom_extraction_invokes_llm(self, agent, llm):
        agent.diagnose({"original_query": "high vacuum pressure on sector 12"})
        # First LLM call is symptom extraction (returns JSON)
        assert llm.invoke.call_count >= 1


class TestKGQueriesRespectClient:
    def test_kg_disabled_when_client_missing(self, llm, vector, bm25):
        agent_no_kg = TroubleshootingAgent(
            llm=llm, neo4j_client=None, vector_retriever=vector, bm25_retriever=bm25,
        )
        result = agent_no_kg.diagnose({"original_query": "issue X"})
        # graph_context should still exist (empty list) but shouldn't raise
        assert "graph_context" in result
        assert result["graph_context"] == []

    def test_kg_failures_dont_abort(self, llm, neo4j, vector, bm25):
        """A Neo4j outage must not crash the runbook flow."""
        neo4j.fulltext_search.side_effect = RuntimeError("Neo4j down")
        neo4j.run_cypher.side_effect = RuntimeError("Neo4j down")
        agent = TroubleshootingAgent(
            llm=llm, neo4j_client=neo4j, vector_retriever=vector, bm25_retriever=bm25,
        )
        result = agent.diagnose({"original_query": "issue X"})
        assert result["final_answer"]


class TestRetrievalFanOut:
    def test_calls_both_retrievers(self, agent, vector, bm25):
        agent.diagnose({"original_query": "kicker timing drift"})
        # Each of the system/ICMS/WR fans out to both vector and bm25
        assert vector.retrieve.call_count >= 1
        assert bm25.retrieve.call_count >= 1


class TestRunbookGeneration:
    def test_runbook_text_propagates_to_final_answer(self, agent):
        result = agent.diagnose({"original_query": "issue"})
        # Our LLM mock returns the runbook on the second invoke
        assert "Runbook" in result["final_answer"]

    def test_llm_failure_surfaces_error_text(self, vector, bm25, neo4j):
        broken = MagicMock()
        broken.invoke.side_effect = [
            MagicMock(content='{"symptoms":[],"systems":[],"components":[]}'),
            RuntimeError("LLM timeout"),
        ]
        agent = TroubleshootingAgent(
            llm=broken, neo4j_client=neo4j, vector_retriever=vector, bm25_retriever=bm25,
        )
        result = agent.diagnose({"original_query": "x"})
        assert "Error" in result["final_answer"]


class TestEnhancedQueryPreference:
    def test_uses_enhanced_query_when_present(self, agent, llm):
        agent.diagnose({
            "original_query": "vague",
            "enhanced_query": "specific kicker mistiming question",
        })
        # The first prompt to the LLM is symptom extraction; it includes the query.
        first_call_messages = llm.invoke.call_args_list[0][0][0]
        text = "\n".join(getattr(m, "content", "") for m in first_call_messages)
        assert "specific kicker" in text
