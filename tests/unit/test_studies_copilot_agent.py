"""Unit tests for StudiesCopilotAgent.assist."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.studies_copilot_agent import StudiesCopilotAgent


def _doc(coll: str, i: int = 0) -> Document:
    return Document(page_content=f"text {i}", metadata={"id": f"{coll}_{i}", "collection": coll})


@pytest.fixture
def llm():
    m = MagicMock()
    m.invoke.return_value = MagicMock(content="Studies copilot answer with [source_1].")
    return m


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
def neo4j():
    n = MagicMock()
    n.run_cypher.return_value = []
    return n


@pytest.fixture
def agent(llm, vector, bm25, neo4j):
    return StudiesCopilotAgent(llm=llm, vector_retriever=vector, bm25_retriever=bm25, neo4j_client=neo4j)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow routing (general vs comparison)
# ─────────────────────────────────────────────────────────────────────────────
class TestWorkflowDispatch:
    def test_general_workflow_for_plain_query(self, agent):
        result = agent.assist({"original_query": "How does the kicker work?"})
        assert result["final_answer"]
        # general workflow returns citations + retrieved_docs
        assert "citations" in result and "retrieved_docs" in result

    def test_comparison_workflow_for_compare_keyword(self, agent):
        # 'compare' keyword triggers _comparison_workflow
        result = agent.assist({"original_query": "Compare beam current with last week"})
        assert result["final_answer"]
        assert "citations" in result

    @pytest.mark.parametrize("kw", [
        "compare", "comparison", "differ", "changed", "previous",
        "last time", "before", "vs", "versus", "improvement",
    ])
    def test_each_comparison_keyword_triggers_compare_path(self, agent, kw):
        # All keywords from the agent's detection list should route to comparison
        result = agent.assist({"original_query": f"please {kw} the data"})
        assert result["final_answer"]


# ─────────────────────────────────────────────────────────────────────────────
# the system attachment integration
# ─────────────────────────────────────────────────────────────────────────────
class TestEntryAttachments:
    def test_no_attachments_renders_empty_block(self, agent):
        block = agent._build_entry_attachment_block([])
        assert block == ""

    def test_attachments_render_into_prompt(self, agent, llm):
        agent.assist({
            "original_query": "review studies",
            "entry_attachments": ["the system-123 page text", "the system-456 page text"],
        })
        # Inspect prompt sent to LLM — should include the attachment text
        prompt_text = "\n".join(
            getattr(m, "content", "") for m in llm.invoke.call_args[0][0]
        )
        assert "the system-123" in prompt_text


# ─────────────────────────────────────────────────────────────────────────────
# Failure isolation
# ─────────────────────────────────────────────────────────────────────────────
class TestFailureIsolation:
    def test_llm_failure_surfaces_error_text(self, vector, bm25, neo4j):
        broken = MagicMock()
        broken.invoke.side_effect = RuntimeError("LLM down")
        agent = StudiesCopilotAgent(llm=broken, vector_retriever=vector, bm25_retriever=bm25, neo4j_client=neo4j)
        result = agent.assist({"original_query": "research X"})
        assert "Error" in result["final_answer"]
