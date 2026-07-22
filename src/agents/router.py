"""
Query router — intent classification + collection routing using gemini-1.5-flash.
Determines retrieval strategy and target collections for each query.
"""
import logging
from typing import Dict, Any, List

from src.utils.json_parser import parse_json_with_repair

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import get_prompt
from .state import AgentState
from src.vectorstore.collection_registry import COLLECTION_OPTIONS

logger = logging.getLogger(__name__)

_VALID_SERVICE_MODES = {"naive_rag", "agentic_rag", "daily_summary", "troubleshooting", "studies_copilot"}

# Intent → service_mode mapping (used when LLM doesn't return explicit service_mode)
_INTENT_TO_SERVICE: Dict[str, str] = {
    "factual": "naive_rag",
    "general": "naive_rag",
    "temporal": "agentic_rag",
    "troubleshooting": "troubleshooting",
    "comparative": "agentic_rag",
    "multi_hop": "agentic_rag",
    "code_generation": "agentic_rag",
    "mathematical": "agentic_rag",
    "counting": "agentic_rag",
    "pv_query": "agentic_rag",
    "daily_summary": "daily_summary",
    "studies_copilot": "studies_copilot",
}


def _infer_service_mode(intent: str) -> str:
    """Fallback: infer service_mode from intent type."""
    return _INTENT_TO_SERVICE.get(intent, "naive_rag")


class QueryRouter:
    """
    LLM-based query router using gemini-1.5-flash for low-latency classification.
    """

    def __init__(self, llm):
        self.llm = llm

    def route(self, state: AgentState) -> Dict[str, Any]:
        """
        Classify query intent and determine routing.
        Updates state fields: query_intent, selected_databases, retrieval_strategy.
        """
        query = state.get("enhanced_query") or state.get("original_query", "")

        messages = [
            SystemMessage(content=get_prompt("ROUTER_PROMPT")),
            HumanMessage(content=f"Classify this query:\n\n{query}"),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content
            routing = self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Router LLM failed: {e} — using defaults")
            routing = {
                "intent": "general",
                "collections": ["all_databases"],
                "retrieval_strategy": "HYBRID_3WAY",
                "service_mode": "naive_rag",
                "entities": [],
                "reasoning": "fallback routing",
            }

        # Validate collections
        collections = routing.get("collections", ["all_databases"])
        valid_collections = [c for c in collections if c in COLLECTION_OPTIONS]
        if not valid_collections:
            valid_collections = ["all_databases"]

        # Determine service_mode — prefer explicit LLM routing, fallback on intent
        service_mode = routing.get("service_mode", "")
        if service_mode not in _VALID_SERVICE_MODES:
            service_mode = _infer_service_mode(routing.get("intent", "general"))

        # Log routing decision
        logger.info(
            f"Router: intent={routing.get('intent')} | "
            f"service_mode={service_mode} | "
            f"strategy={routing.get('retrieval_strategy')} | "
            f"collections={valid_collections}"
        )

        # Update reasoning trace for UI thinking display
        reasoning_trace = state.get("reasoning_trace", [])
        intent = routing.get("intent", "general")
        strategy = routing.get("retrieval_strategy", "HYBRID_3WAY")
        reasoning_trace.append(
            f"Routing: intent={intent}, strategy={strategy}, "
            f"collections={valid_collections}"
        )

        return {
            "query_intent": intent,
            "selected_databases": valid_collections,
            "retrieval_strategy": strategy,
            "service_mode": service_mode,
            "extracted_entities": routing.get("entities", []),
            "reasoning_trace": reasoning_trace,
        }

    def _parse_json(self, raw: str) -> Dict:
        """Extract JSON from LLM response with repair fallback."""
        result = parse_json_with_repair(raw, llm=self.llm, context="router")
        return result if isinstance(result, dict) else {}
