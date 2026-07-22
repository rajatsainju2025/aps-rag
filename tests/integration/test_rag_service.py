"""
Integration tests for RAGService.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_mock_components():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Test answer from RAG")

    mock_vector = MagicMock()
    mock_vector.retrieve.return_value = []

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = []

    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = []

    mock_fusion = MagicMock()
    mock_fusion.fuse.return_value = []

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = []

    mock_enhancer = MagicMock()
    mock_enhancer.enhance.return_value = {"enhanced_query": "test query", "temporal_context": None}

    return {
        "fast_llm": mock_llm,
        "synthesis_llm": mock_llm,
        "vector_retriever": mock_vector,
        "bm25_retriever": mock_bm25,
        "graph_retriever": mock_graph,
        "fusion": mock_fusion,
        "reranker": mock_reranker,
        "query_enhancer": mock_enhancer,
        "neo4j_client": None,
        "memory_manager": None,
        "tools": {},
    }


def _make_mock_settings():
    settings = MagicMock()
    settings.agent.quality_threshold = 0.6
    settings.agent.max_correction_rounds = 3
    settings.agent.top_k = 10
    settings.agent.reranker_top_k = 5
    return settings


@pytest.fixture
def rag_service():
    from src.services.rag_service import RAGService
    settings = _make_mock_settings()
    components = _make_mock_components()

    with patch("src.agents.orchestrator.create_orchestrator") as mock_orch_factory:
        mock_orch = MagicMock()
        mock_orch.invoke.return_value = {
            "final_answer": "Test answer",
            "citations": [{"title": "Doc1", "author": "Author1"}],
            "retrieved_docs": [],
            "correction_round": 0,
            "reasoning_trace": [],
        }
        mock_orch_factory.return_value = mock_orch
        service = RAGService(settings=settings, components=components)
        service._orchestrator = mock_orch
        yield service


def test_rag_service_query(rag_service):
    result = rag_service.query("What is the beam current?", session_id="test-session")
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert result["service_mode"] == "rag"
    assert "latency_ms" in result
    assert "citations" in result


def test_health_check(rag_service):
    result = rag_service.health_check()
    assert result["status"] == "ok"
    assert result["service_mode"] == "rag"
