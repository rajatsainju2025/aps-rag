"""ClarificationAgent — gates counting queries for scope ambiguity.

Single LLM call (gpt-4o-mini critique LLM, same as AnswerAuditor). Given the user's
raw query, decides whether the query is genuinely ambiguous in a way the
system would have to GUESS to answer (e.g. "is part of" — participation vs
mention?). When ambiguous, returns a structured ``{question, options}`` dict
the orchestrator surfaces to the user before retrieval/planning. Otherwise
returns ``None`` and execution proceeds normally.

Fail-open by design: any malformed LLM output, missing fields, or exception
returns ``None`` so a broken gate never blocks normal query flow.

Pattern: pre-flight clarification gate (LangGraph human-in-the-loop /
DSPy Suggest). Cost: at most one cheap critique-LLM call per counting query;
zero cost on resume turns or non-counting queries.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import get_prompt
from src.utils.json_parser import parse_json_with_repair

logger = logging.getLogger(__name__)


class ClarificationAgent:
    """Pre-flight scope-ambiguity detector for counting queries."""

    def __init__(self, llm: Any):
        self.llm = llm

    def maybe_clarify(self, query: str) -> Optional[Dict[str, Any]]:
        """Return ``{"question", "options"}`` if ambiguous, else ``None``.

        ``options`` is a list of ``{"id", "label", "interpretation"}`` dicts.
        At least 2 options are required for a real choice — fewer means
        there is only one defensible interpretation.
        """
        try:
            prompt = get_prompt("CLARIFICATION_PROMPT").format(query=query)
            response = self.llm.invoke([
                SystemMessage(content="You are a query disambiguation gate. Respond with JSON only."),
                HumanMessage(content=prompt),
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            result = parse_json_with_repair(raw, llm=self.llm, context="clarification")
        except Exception as e:
            logger.warning("Clarification gate failed (fail-open): %s", e)
            return None

        if not isinstance(result, dict) or not result.get("ambiguous"):
            return None
        opts = result.get("options")
        question = result.get("question")
        if not isinstance(opts, list) or len(opts) < 2 or not isinstance(question, str):
            return None
        # Validate option shape — drop anything missing required keys
        clean_opts = [
            {"id": str(o["id"]), "label": str(o["label"]), "interpretation": str(o["interpretation"])}
            for o in opts
            if isinstance(o, dict) and all(k in o for k in ("id", "label", "interpretation"))
        ]
        if len(clean_opts) < 2:
            return None
        return {"question": question, "options": clean_opts}
