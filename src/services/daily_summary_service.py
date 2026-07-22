"""
Daily Summary service.
"""
import logging
import time
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from src.services.base_service import BaseService
from src.utils.citation_formatter import format_daily_summary_citations

logger = logging.getLogger(__name__)


class DailySummaryService(BaseService):
    """Generates daily operational summaries from the facility data."""

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
        return "daily_summary"

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start = time.time()
        orchestrator = self._get_orchestrator()
        date_str: Optional[str] = kwargs.get("date")

        effective_query = f"{query} date:{date_str}" if date_str else query

        state = {
            "messages": [HumanMessage(content=effective_query)],
            "original_query": effective_query,
            "enhanced_query": effective_query,
            "session_id": session_id or "default",
            "service_mode": "daily_summary",
            "selected_databases": kwargs.get("selected_databases", ["all_databases"]),
            "query_intent": "temporal",
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
            "temporal_context": {"date": date_str} if date_str else None,
        }

        try:
            result = orchestrator.invoke(state)
        except Exception as exc:
            logger.error("DailySummary query failed: %s", exc)
            result = {"final_answer": f"Error: {exc}", "citations": [], "retrieved_docs": []}

        latency_ms = (time.time() - start) * 1000
        retrieved_docs = result.get("retrieved_docs", [])

        # Apply inline citation markers → clickable links + source footer
        raw_answer = result.get("final_answer", "")
        if retrieved_docs:
            formatted_answer, citations, owui_citations = format_daily_summary_citations(
                raw_answer, retrieved_docs
            )
        else:
            formatted_answer = raw_answer
            citations = result.get("citations", [])
            owui_citations = []

        return {
            "answer": formatted_answer,
            "summary": formatted_answer,
            "citations": citations,
            "openwebui_citations": owui_citations,
            "sources": [d.metadata if hasattr(d, "metadata") else d for d in retrieved_docs],
            "date": date_str,
            "service_mode": "daily_summary",
            "reasoning_trace": result.get("reasoning_trace", []),
            "latency_ms": round(latency_ms, 2),
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "service_mode": "daily_summary"}
