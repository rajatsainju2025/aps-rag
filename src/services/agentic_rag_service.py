"""
Agentic RAG service — full ReAct + corrective loop.
"""
import logging
import re
import time
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from src.agents.event_emitter import current_event_emitter
from src.services.base_service import BaseService
from src.utils.citation_formatter import format_answer_with_citations

logger = logging.getLogger(__name__)

# Resume signal: short numeric reply ("1", "2.", "clarify 1", "option 2")
# arriving in a session that has a pending clarification.
_CHOICE_RE = re.compile(r"^\s*(?:clarify\s+|option\s+)?([1-9])\s*\.?\s*$", re.IGNORECASE)


def _render_clarification_markdown(pending: Dict[str, Any]) -> str:
    """Render a pending clarification as numbered markdown for OpenWebUI."""
    lines = [f"**{pending.get('question', 'Please clarify:')}**", ""]
    for opt in pending.get("options", []):
        lines.append(f"{opt['id']}. **{opt['label']}** — {opt['interpretation']}")
    lines.append("")
    lines.append("_Reply with the number of your choice (e.g. `1`)._")
    return "\n".join(lines)


class AgenticRAGService(BaseService):
    """Full agentic RAG with ReAct reasoning and corrective loop."""

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
        return "agentic_rag"

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start = time.time()
        orchestrator = self._get_orchestrator()
        sid = session_id or "default"

        # ── Resume path ──────────────────────────────────────────────────
        # If the prior turn surfaced a clarification AND this turn looks like
        # a numeric choice ("1", "clarify 2"), inject the chosen interpretation
        # into the query and clear the pending state. Anything else clears the
        # pending state and proceeds as a fresh query.
        memory = self.components.get("memory_manager")
        clarification_choice: Optional[str] = None
        pending = memory.get_session_state(sid, "clarification_pending") if memory else None
        if pending:
            m = _CHOICE_RE.match(query or "")
            if m:
                choice_id = m.group(1)
                chosen = next(
                    (o for o in pending.get("options", []) if str(o.get("id")) == choice_id),
                    None,
                )
                if chosen:
                    clarification_choice = choice_id
                    original = pending.get("original_query", query)
                    query = f"{original} (clarified: {chosen['interpretation']})"
            # Clear pending whether or not we matched — stale state shouldn't linger.
            memory.set_session_state(sid, "clarification_pending", None)

        state = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "enhanced_query": query,
            "session_id": sid,
            "service_mode": "agentic_rag",
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
            "query_variants": [query],
            "clarification_pending": None,
            "clarification_choice": clarification_choice,
            "retrieval_strategy_override": kwargs.get("retrieval_strategy"),
        }

        emitter = kwargs.get("event_emitter")
        token = current_event_emitter.set(emitter) if emitter is not None else None
        try:
            result = orchestrator.invoke(state, {"recursion_limit": 50})
        except Exception as exc:
            logger.error("AgenticRAG query failed: %s", exc, exc_info=True)
            result = {
                "final_answer": f"Error: {exc}",
                "citations": [],
                "retrieved_docs": [],
                "reasoning_trace": [],
                "correction_round": 0,
            }
        finally:
            if token is not None:
                current_event_emitter.reset(token)

        latency_ms = (time.time() - start) * 1000

        # ── Clarification surfacing ─────────────────────────────────────
        # If the gate emitted options, persist them for the next turn and
        # render the question as the visible answer. No retrieval, no
        # citations — the user picks before we plan.
        pending_out = result.get("clarification_pending")
        if pending_out:
            persisted = dict(pending_out)
            persisted["original_query"] = state["original_query"]
            if memory:
                memory.set_session_state(sid, "clarification_pending", persisted)
            return {
                "answer": _render_clarification_markdown(persisted),
                "citations": [],
                "openwebui_citations": [],
                "sources": [],
                "service_mode": "agentic_rag",
                "reasoning_trace": [],
                "correction_rounds": 0,
                "latency_ms": round(latency_ms, 2),
                "clarification": persisted,
            }

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
            "service_mode": "agentic_rag",
            "reasoning_trace": result.get("reasoning_trace", []),
            "correction_rounds": result.get("correction_round", 0),
            "latency_ms": round(latency_ms, 2),
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "service_mode": "agentic_rag"}
