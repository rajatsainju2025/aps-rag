"""Unit tests for RetrievalAgent — multi-strategy fan-out, fusion, rerank."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.retrieval_agent import RetrievalAgent


def _doc(id_: str, content: str = "txt", coll: str = "entry_logbook") -> Document:
    return Document(page_content=content, metadata={"id": id_, "collection": coll})


@pytest.fixture
def vector():
    m = MagicMock()
    m.retrieve.return_value = [_doc("v1"), _doc("v2")]
    return m


@pytest.fixture
def bm25():
    m = MagicMock()
    m.retrieve.return_value = [_doc("b1")]
    return m


@pytest.fixture
def graph():
    m = MagicMock()
    m.retrieve.return_value = [_doc("g1")]
    m.retrieve_entity_centric.return_value = [_doc("g1")]
    return m


@pytest.fixture
def fusion():
    m = MagicMock()
    # Echo input docs as the "fused" output for assertion clarity
    m.fuse.side_effect = lambda v, b, g, **kw: list(v) + list(b) + list(g)
    return m


@pytest.fixture
def reranker():
    m = MagicMock()
    m.rerank.side_effect = lambda q, docs, **kw: docs
    return m


@pytest.fixture
def enhancer():
    m = MagicMock()
    m.enhance.return_value = {
        "enhanced_query": "enhanced beam current question",
        "temporal_context": None,
        "query_variants": ["enhanced beam current question"],
    }
    return m


@pytest.fixture
def agent(vector, bm25, graph, fusion, reranker, enhancer):
    return RetrievalAgent(
        vector_retriever=vector,
        bm25_retriever=bm25,
        graph_retriever=graph,
        fusion=fusion,
        reranker=reranker,
        query_enhancer=enhancer,
    )


def _state(**overrides):
    base = {
        "original_query": "what is the beam current?",
        "enhanced_query": "what is the beam current?",
        "query_intent": "factual",
        "selected_databases": ["entry_logbook"],
        "retrieval_strategy": "HYBRID_3WAY",
        "extracted_entities": [],
        "query_variants": ["what is the beam current?"],
        "reasoning_trace": [],
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Strategy: HYBRID_3WAY — all three retrievers fire
# ─────────────────────────────────────────────────────────────────────────────
class TestHybrid3Way:
    def test_calls_vector_bm25_and_graph(self, agent, vector, bm25, graph):
        agent.retrieve(_state())
        vector.retrieve.assert_called()
        bm25.retrieve.assert_called()
        graph.retrieve.assert_called()

    def test_returns_required_keys(self, agent):
        out = agent.retrieve(_state())
        for key in ("retrieved_docs", "graph_context", "retrieval_sources",
                    "temporal_context", "query_variants", "reasoning_trace"):
            assert key in out

    def test_sources_listed_when_each_returns_results(self, agent):
        out = agent.retrieve(_state())
        assert "vector" in out["retrieval_sources"]
        assert "bm25" in out["retrieval_sources"]
        assert "graph" in out["retrieval_sources"]

    def test_fusion_invoked(self, agent, fusion):
        agent.retrieve(_state())
        fusion.fuse.assert_called_once()

    def test_reranker_invoked(self, agent, reranker):
        agent.retrieve(_state())
        reranker.rerank.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Strategy: VECTOR_ONLY — only vector fires
# ─────────────────────────────────────────────────────────────────────────────
class TestVectorOnly:
    def test_only_vector_called(self, agent, vector, bm25, graph):
        agent.retrieve(_state(retrieval_strategy="VECTOR_ONLY"))
        vector.retrieve.assert_called()
        bm25.retrieve.assert_not_called()
        graph.retrieve.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Strategy: KG_ONLY — only graph fires
# ─────────────────────────────────────────────────────────────────────────────
class TestKGOnly:
    def test_only_graph_called(self, agent, vector, bm25, graph):
        agent.retrieve(_state(retrieval_strategy="KG_ONLY", extracted_entities=["BPM"]))
        graph.retrieve_entity_centric.assert_called()
        vector.retrieve.assert_not_called()
        bm25.retrieve.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Failure isolation
# ─────────────────────────────────────────────────────────────────────────────
class TestFailureIsolation:
    def test_vector_failure_does_not_abort(self, agent, vector):
        vector.retrieve.side_effect = RuntimeError("Qdrant down")
        out = agent.retrieve(_state())
        # Should still return a dict with retrieved_docs (from BM25 + graph)
        assert "retrieved_docs" in out

    def test_graph_failure_does_not_abort(self, agent, graph):
        graph.retrieve.side_effect = RuntimeError("Neo4j down")
        out = agent.retrieve(_state())
        # Should still return; sources won't include graph
        assert "graph" not in out["retrieval_sources"]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query variants
# ─────────────────────────────────────────────────────────────────────────────
class TestMultiQueryVariants:
    def test_each_variant_triggers_retrieval(self, agent, vector):
        agent.retrieve(_state(query_variants=["v1", "v2", "v3"]))
        # Vector retrieve called once per variant in HYBRID_3WAY
        assert vector.retrieve.call_count == 3
