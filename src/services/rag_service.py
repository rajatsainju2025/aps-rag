"""
Standard enhanced RAG service (no corrective loop).
"""
import logging
import time
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from src.agents.event_emitter import current_event_emitter
from src.services.base_service import BaseService
from src.utils.citation_formatter import format_answer_with_citations

logger = logging.getLogger(__name__)


class RAGService(BaseService):
    """Standard RAG — retrieve + synthesize, single pass.

    Also serves as the canonical implementation for ``service_mode="naive_rag"``
    (see ServiceSelector) — both modes share the orchestrator's
    ``naive_rag_node`` (retrieve + synthesize, no corrective loop, with the
    tool-execution loop). The state still carries ``service_mode="rag"`` so the
    orchestrator's ``route_to_service`` doesn't auto-upgrade the request to
    agentic_rag; only the public-facing label varies via ``mode_label``.
    """

    def __init__(
        self,
        settings: Any,
        components: Dict[str, Any],
        mode_label: str = "rag",
    ) -> None:
        self.settings = settings
        self.components = components
        self._orchestrator = None
        self._mode_label = mode_label

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from src.agents.orchestrator import create_orchestrator
            self._orchestrator = create_orchestrator(self.settings, self.components)
        return self._orchestrator

    def get_service_mode(self) -> str:
        return self._mode_label

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start = time.time()
        orchestrator = self._get_orchestrator()

        state = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "enhanced_query": query,
            "session_id": session_id or "default",
            "service_mode": "rag",
            "selected_databases": kwargs.get("selected_databases", ["all_databases"]),
            "query_intent": "",
            "plan": "",
            "reasoning_trace": [],
            "retrieved_docs": [],
            "graph_context": [],
            "web_results": [],
            "retrieval_sources": [],
            "draft_answer": "",
            "final_answer": "",
            "citations": [],
            "confidence_score": 0.0,
            "needs_correction": False,
            "correction_round": 0,
            "tool_calls": [],
            "tool_results": [],
            "memory_context": {},
            "next_node": "",
            "error": None,
            "iteration_count": 0,
            "temporal_context": None,
            "retrieval_strategy_override": kwargs.get("retrieval_strategy"),
        }

        emitter = kwargs.get("event_emitter")
        token = current_event_emitter.set(emitter) if emitter is not None else None
        try:
            result = orchestrator.invoke(state)
        except Exception as exc:
            logger.error("RAG query failed: %s", exc)
            result = {"final_answer": f"Error: {exc}", "citations": [], "retrieved_docs": []}
        finally:
            if token is not None:
                current_event_emitter.reset(token)

        latency_ms = (time.time() - start) * 1000

        # Apply inline citation markers + formatted source footer
        raw_answer = result.get("final_answer", "")
        retrieved_docs = result.get("retrieved_docs", [])
        owui_citations = []
        if retrieved_docs:
            formatted_answer, citations, owui_citations = format_answer_with_citations(
                raw_answer, retrieved_docs
            )
        else:
            formatted_answer = raw_answer
            citations = result.get("citations", [])

        return {
            "answer": formatted_answer,
            "citations": citations,
            "openwebui_citations": owui_citations,
            "sources": [d.metadata if hasattr(d, "metadata") else d for d in retrieved_docs],
            "service_mode": self._mode_label,
            "latency_ms": round(latency_ms, 2),
            # Surface the orchestrator's reasoning_trace so observers (the
            # counting verifier's tool-detection check, OpenWebUI's <think>
            # rendering) can see which tools fired. Mirrors AgenticRAGService.
            "reasoning_trace": result.get("reasoning_trace") or [],
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "service_mode": self._mode_label}
