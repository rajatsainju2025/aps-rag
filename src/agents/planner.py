"""
Planner node — multi-step plan generation using gpt-4o.
Creates a structured JSON research plan given the query and routing decision.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import get_prompt
from src.utils.json_parser import JSON_REPAIR_PROMPT, strip_cot_blocks
from .state import AgentState

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Generates multi-step retrieval and reasoning plans."""

    def __init__(self, llm):
        self.llm = llm

    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Generate a research plan for the current query."""
        logger.debug(
            "Planner using LLM: %s (model=%s)",
            type(self.llm).__name__,
            getattr(self.llm, "model_name", getattr(self.llm, "model", "unknown")),
        )
        query = state.get("enhanced_query") or state.get("original_query", "")
        intent = state.get("query_intent", "general")
        collections = state.get("selected_databases", [])
        strategy = state.get("retrieval_strategy", "HYBRID_3WAY")
        correction_round = state.get("correction_round", 0)
        entities = state.get("extracted_entities", [])
        prev_issues = state.get("correction_issues", [])

        correction_note = ""
        if correction_round > 0:
            correction_guidance = {
                1: "Broaden query terms, add synonyms, expand date range, try alternate collection.",
                2: "Switch to fundamentally different strategy (e.g., graph→vector or vector→graph).",
            }.get(correction_round, "Decompose into sub-questions, search all relevant collections.")
            issues_str = "; ".join(prev_issues[:3]) if prev_issues else "low relevance or completeness"
            correction_note = (
                f"\n\n[CORRECTION ROUND {correction_round}]\n"
                f"Previous attempt issues: {issues_str}\n"
                f"Correction guidance: {correction_guidance}"
            )

        entities_str = f"\nExtracted entities: {entities}" if entities else ""

        context = (
            f"Query: {query}\n"
            f"Intent: {intent}\n"
            f"Collections: {collections}\n"
            f"Retrieval strategy: {strategy}"
            + entities_str
            + correction_note
        )

        messages = [
            SystemMessage(content=get_prompt("PLANNER_PROMPT")),
            HumanMessage(content=context),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_plan = response.content.strip()
        except Exception as e:
            logger.warning(f"Planner failed: {e}")
            raw_plan = f"1. Search {collections} for information about: {query}\n2. Synthesize answer from results"

        # Parse structured JSON plan from LLM output
        plan_steps = self._parse_plan(raw_plan)

        # Counting / aggregation queries: force a data_analysis step at the
        # head of the plan if the LLM didn't include one. The reasoning_agent
        # also injects a hard directive at synthesis time, but pre-seeding the
        # plan ensures the orchestrator can route this to deterministic
        # pandas execution rather than relying on the LLM to remember.
        if state.get("counting_intent") and not any(
            isinstance(s, dict) and s.get("tool") == "data_analysis"
            for s in plan_steps
        ):
            ds = self._infer_data_source(collections, query)
            if ds:
                plan_steps.insert(0, {
                    "step": 0,
                    "tool": "data_analysis",
                    "query": query,
                    "collections": collections,
                    "data_source": ds,
                    "rationale": (
                        "Counting/aggregation query — use data_analysis tool for "
                        "deterministic count instead of LLM estimation."
                    ),
                })
                logger.info("Planner: prepended data_analysis step (data_source=%s)", ds)

        logger.debug(f"Plan generated: {len(plan_steps)} steps")

        # Update reasoning trace for UI thinking display
        reasoning_trace = state.get("reasoning_trace", [])
        n_steps = len(plan_steps)
        step_summary = ", ".join(
            f"{s.get('tool', '?')}({(s.get('query') or '')[:30]})"
            for s in plan_steps[:4]
            if isinstance(s, dict)  # guard against malformed string items
        )
        reasoning_trace.append(
            f"Planning: {n_steps}-step plan for intent={intent} → {step_summary}"
        )

        return {
            "plan": raw_plan,
            "plan_steps": plan_steps,
            "reasoning_trace": reasoning_trace,
        }

    @staticmethod
    def _infer_data_source(
        collections: List[str], query: str = ""
    ) -> Optional[str]:
        """Map a list of selected collections (and the query text) to a
        data_analysis data_source.

        Tries the explicit collections first. If none match (e.g. OpenWebUI
        passes `["all_databases"]`), falls back to keyword inspection of the
        query — looking for "work request", "logbook"/"entry"/"study", or
        "teams". Returns None only when neither signal points anywhere.
        """
        for c in collections or []:
            cl = str(c).lower()
            if "work_request" in cl:
                return "work_requests"
            if "entry" in cl or "logbook" in cl:
                return "entry_logbook"
            if "teams" in cl:
                return "teams_messages"
        # Fallback: inspect the query text for the OpenWebUI ["all_databases"] case
        ql = (query or "").lower()
        if any(k in ql for k in ("work request", "wrq", "work-request", "wr ")):
            return "work_requests"
        if any(k in ql for k in ("logbook", "entry", "study", "beam study", "studies")):
            return "entry_logbook"
        if "teams" in ql or "chat message" in ql:
            return "teams_messages"
        return None

    def _parse_plan(self, raw: str) -> List[Dict[str, Any]]:
        """Extract structured plan steps from LLM JSON output.

        Strategy:
          1. Strip CoT blocks (<scratchpad>, <thinking>, …)
          2. Extract the outermost JSON array via balanced-bracket scan
             (string-aware so braces inside strings don't confuse depth)
          3. On parse failure, ask the LLM to repair the JSON once
          4. On total failure, return a single-step synthesize fallback
        """
        cleaned = strip_cot_blocks(raw)

        parsed = self._extract_array(cleaned)
        if parsed is None:
            parsed = self._llm_repair(raw)

        if parsed is None:
            logger.warning("Could not parse plan JSON — using fallback single-step plan")
            return [{"step": 1, "tool": "synthesize", "query": None, "collections": [], "rationale": "Fallback"}]

        return self._normalize_steps(parsed)

    @staticmethod
    def _extract_array(text: str) -> Optional[List[Any]]:
        """Find the outermost JSON array in `text` and parse it.

        Returns the parsed list, or None if no valid array is found.
        Walks the string character-by-character with depth + quote tracking
        so brackets inside JSON strings don't perturb the depth count.
        """
        text = re.sub(r"```\w*\s*", "", text)
        if text.endswith("```"):
            text = text[:-3]
        start = text.find("[")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:i + 1])
                        return result if isinstance(result, list) else None
                    except json.JSONDecodeError:
                        return None
        return None

    def _llm_repair(self, raw: str) -> Optional[List[Any]]:
        """Ask the planner LLM to fix malformed JSON once."""
        try:
            response = self.llm.invoke([
                HumanMessage(content=JSON_REPAIR_PROMPT.format(broken_json=raw[:3000]))
            ])
            repaired = response.content if hasattr(response, "content") else str(response)
            return self._extract_array(strip_cot_blocks(repaired))
        except Exception as e:
            logger.warning(f"Planner JSON repair pass failed: {e}")
            return None

    @staticmethod
    def _normalize_steps(steps: list) -> List[Dict[str, Any]]:
        """Ensure every item in the parsed plan is a dict.

        The LLM sometimes returns a JSON array of bare strings
        (e.g. ["search for X", "synthesize"]) instead of objects.
        Convert those to proper step dicts so downstream code that
        calls s.get('tool') won't crash.
        """
        normalized = []
        for i, s in enumerate(steps, 1):
            if isinstance(s, dict):
                normalized.append(s)
            elif isinstance(s, str):
                normalized.append({
                    "step": i,
                    "tool": "text_search",
                    "query": s[:200],
                    "collections": [],
                    "rationale": f"Coerced from string: {s[:80]}",
                })
            else:
                logger.warning(f"Skipping non-dict/non-str plan item: {type(s)}")
        return normalized or [{"step": 1, "tool": "synthesize", "query": None, "collections": [], "rationale": "Fallback after normalization"}]
