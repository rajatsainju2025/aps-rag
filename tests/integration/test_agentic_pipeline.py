"""
Integration tests for the agentic RAG pipeline.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_mock_components():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Agentic answer")

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    mock_enhancer = MagicMock()
    mock_enhancer.enhance.return_value = {"enhanced_query": "test", "temporal_context": None}

    return {
        "fast_llm": mock_llm,
        "synthesis_llm": mock_llm,
        "vector_retriever": mock_retriever,
        "bm25_retriever": mock_retriever,
        "graph_retriever": mock_retriever,
        "fusion": MagicMock(fuse=MagicMock(return_value=[])),
        "reranker": MagicMock(rerank=MagicMock(return_value=[])),
        "query_enhancer": mock_enhancer,
        "neo4j_client": None,
        "memory_manager": None,
        "tools": {},
    }


def _make_mock_settings():
    s = MagicMock()
    s.agent.quality_threshold = 0.6
    s.agent.max_correction_rounds = 3
    s.agent.top_k = 10
    s.agent.reranker_top_k = 5
    return s


_MOCK_ORCH_RESULT = {
    "final_answer": "Agentic answer with reasoning",
    "citations": [],
    "retrieved_docs": [],
    "correction_round": 1,
    "reasoning_trace": ["Step 1: retrieved docs", "Step 2: synthesized"],
    "graph_context": [],
    "tool_results": [],
}


def _make_orch_service(service_class):
    settings = _make_mock_settings()
    components = _make_mock_components()
    mock_orch = MagicMock()
    mock_orch.invoke.return_value = _MOCK_ORCH_RESULT

    with patch("src.agents.orchestrator.create_orchestrator", return_value=mock_orch):
        svc = service_class(settings=settings, components=components)
        svc._orchestrator = mock_orch
        return svc


def test_agentic_rag_query():
    from src.services.agentic_rag_service import AgenticRAGService
    service = _make_orch_service(AgenticRAGService)
    result = service.query("Explain the corrector magnet fault", session_id="test")

    assert "answer" in result
    assert result["service_mode"] == "agentic_rag"
    assert "reasoning_trace" in result
    assert "correction_rounds" in result
    assert isinstance(result["correction_rounds"], int)
    assert "latency_ms" in result


def test_service_modes():
    from src.ui.service_selector import ServiceSelector

    settings = _make_mock_settings()
    components = _make_mock_components()
    mock_orch = MagicMock()
    mock_orch.invoke.return_value = _MOCK_ORCH_RESULT

    selector = ServiceSelector()

    modes = ["rag", "agentic_rag", "daily_summary", "troubleshooting", "studies_copilot"]
    with patch("src.agents.orchestrator.create_orchestrator", return_value=mock_orch):
        for mode in modes:
            service = selector.get_service(mode, settings, components)
            assert service.get_service_mode() == mode, f"Mode mismatch for {mode}"
            # Check caching
            service2 = selector.get_service(mode, settings, components)
            assert service is service2, f"Service not cached for {mode}"
