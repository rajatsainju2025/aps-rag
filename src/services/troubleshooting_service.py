"""
Troubleshooting service — fault chain analysis.
"""
import logging
import time
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from src.services.base_service import BaseService
from src.utils.citation_formatter import format_answer_with_citations

logger = logging.getLogger(__name__)


class TroubleshootingService(BaseService):
    """Diagnoses the facility system faults and provides remediation procedures."""

    service_mode = "troubleshooting"

    def __init__(self, settings: Any, components: Dict[str, Any]) -> None:
        self.settings = settings
        self.components = components
        self._orchestrator = None

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from src.agents.orchestrator import create_orchestrator
            self._orchestrator = create_orchestrator(self.settings, self.components)
        return self._orchestrator

    def get_service_mode(self) -> str:
        return self.service_mode

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
            "service_mode": "troubleshooting",
            "selected_databases": kwargs.get("selected_databases", ["all_databases"]),
            "query_intent": "troubleshooting",
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
        }

        try:
            result = orchestrator.invoke(state)
        except Exception as exc:
            logger.error("Troubleshooting query failed: %s", exc)
            result = {"final_answer": f"Error: {exc}", "citations": [], "retrieved_docs": []}

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
            "fault_chain": result.get("graph_context", []),
            "procedure": result.get("draft_answer", ""),
            "citations": citations,
            "openwebui_citations": owui_citations,
            "sources": [d.metadata if hasattr(d, "metadata") else d for d in retrieved_docs],
            "service_mode": "troubleshooting",
            "reasoning_trace": result.get("reasoning_trace", []),
            "latency_ms": round(latency_ms, 2),
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "service_mode": "troubleshooting"}
