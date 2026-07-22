"""Unit tests for NaiveRAGService — single-pass hybrid RAG."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.services.naive_rag_service import NaiveRAGService


def _doc(coll: str = "entry_logbook", i: int = 0) -> Document:
    return Document(
        page_content=f"text {i}",
        metadata={
            "id": f"{coll}_{i}",
            "collection": coll,
            "creation_datetime": "2026-04-25T10:00:00",
        },
    )


def _settings():
    agent = SimpleNamespace(reranker_top_k=10, top_k=10)
    return SimpleNamespace(agent=agent)


@pytest.fixture
def llm():
    m = MagicMock()
    m.invoke.return_value = MagicMock(content="The beam current is 100 mA [source_1].")
    return m


@pytest.fixture
def vector():
    m = MagicMock()
    m.retrieve.return_value = [_doc("entry_logbook", 0)]
    return m


@pytest.fixture
def bm25():
    m = MagicMock()
    m.retrieve.return_value = [_doc("entry_logbook", 1)]
    return m


@pytest.fixture
def fusion():
    m = MagicMock()
    m.fuse.side_effect = lambda v, b, g, **kw: list(v) + list(b)
    return m


@pytest.fixture
def reranker():
    m = MagicMock()
    m.rerank.side_effect = lambda q, docs: docs
    return m


@pytest.fixture
def enhancer():
    m = MagicMock()
    m.enhance.return_value = {
        "enhanced_query": "enhanced",
        "temporal_context": None,
        "query_variants": ["enhanced"],
        "synonym_expansion": {"total_expansions": 0},
    }
    return m


@pytest.fixture
def collection_router():
    m = MagicMock()
    m.route.return_value = ["entry_logbook"]
    return m


@pytest.fixture
def components(llm, vector, bm25, fusion, reranker, enhancer):
    return {
        "synthesis_llm": llm,
        "vector_retriever": vector,
        "bm25_retriever": bm25,
        "fusion": fusion,
        "reranker": reranker,
        "query_enhancer": enhancer,
        "fast_llm": MagicMock(),
    }


@pytest.fixture
def service(components, collection_router):
    s = NaiveRAGService(_settings(), components)
    s._collection_router = collection_router  # inject the mocked router
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Single-pass synthesis
# ─────────────────────────────────────────────────────────────────────────────
class TestSinglePassSynthesis:
    def test_returns_required_keys(self, service):
        out = service.query("what is the beam current?", session_id="s1")
        for k in ("answer", "citations", "openwebui_citations", "sources",
                  "service_mode", "reasoning_trace", "latency_ms"):
            assert k in out

    def test_invokes_llm_once(self, service, llm):
        service.query("query", session_id="s1")
        assert llm.invoke.call_count == 1

    def test_service_mode_is_naive_rag(self, service):
        out = service.query("query", session_id="s1")
        assert out["service_mode"] == "naive_rag"

    def test_runs_retrieval(self, service, vector, bm25):
        service.query("query", session_id="s1")
        vector.retrieve.assert_called()
        bm25.retrieve.assert_called()

    def test_runs_fusion_and_rerank(self, service, fusion, reranker):
        service.query("query", session_id="s1")
        fusion.fuse.assert_called_once()
        reranker.rerank.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Empty results path
# ─────────────────────────────────────────────────────────────────────────────
class TestEmptyResults:
    def test_no_docs_returns_friendly_message(self, service, vector, bm25, llm):
        vector.retrieve.return_value = []
        bm25.retrieve.return_value = []
        out = service.query("obscure query", session_id="s1")
        assert "No relevant documents" in out["answer"]
        # LLM should NOT be called when there are zero docs
        llm.invoke.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Query enhancement integration
# ─────────────────────────────────────────────────────────────────────────────
class TestQueryEnhancement:
    def test_enhanced_query_used_for_retrieval(self, service, enhancer, vector):
        enhancer.enhance.return_value = {
            "enhanced_query": "ENHANCED beam current",
            "temporal_context": None,
            "query_variants": ["ENHANCED beam current"],
            "synonym_expansion": {"total_expansions": 0},
        }
        service.query("vague q", session_id="s1")
        # The variant fed to vector retriever should be the enhanced text
        first_call_q = vector.retrieve.call_args_list[0][0][0]
        assert "ENHANCED" in first_call_q


# ─────────────────────────────────────────────────────────────────────────────
# Multi-collection routing
# ─────────────────────────────────────────────────────────────────────────────
class TestCollectionRouting:
    def test_explicit_selection_overrides_router(self, service, collection_router):
        service.query("q", session_id="s1", selected_databases=["icms_collection"])
        # Explicit selection bypasses LLM-based collection router
        collection_router.route.assert_not_called()

    def test_all_databases_invokes_router(self, service, collection_router):
        service.query("q", session_id="s1", selected_databases=["all_databases"])
        collection_router.route.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning trace
# ─────────────────────────────────────────────────────────────────────────────
class TestReasoningTrace:
    def test_trace_records_collection_count(self, service):
        out = service.query("q", session_id="s1", selected_databases=["entry_logbook"])
        first = out["reasoning_trace"][0]
        assert "collection" in first.lower()

    def test_trace_records_doc_counts(self, service):
        out = service.query("q", session_id="s1", selected_databases=["entry_logbook"])
        # Some step mentions "vector" or "BM25" with counts
        joined = " ".join(out["reasoning_trace"])
        assert "vector" in joined.lower() or "bm25" in joined.lower()
