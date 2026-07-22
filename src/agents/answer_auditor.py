"""AnswerAuditor — evidence-based critique of `data_analysis` count results.

Fires only when the executed pandas code emits a JSON dict with
``{count, matched_sample, near_miss_sample}`` (the entity-count contract from
the data_analysis tool description). For all other tool outputs, returns
``{"retry": False}`` unconditionally — pure aggregations, errors, plain
numbers all pass through with zero LLM cost.

Reuses the fast critique LLM (gpt-4o-mini). One LLM call per audited tool result;
caller is responsible for capping retries (current limit: 1 per query, gated
by ``state["audit_rounds"]`` in reasoning_agent).

Pattern: actor + critic with verbal feedback (Reflexion / DSPy Suggest /
langgraph-reflection). See plan SOTA validation table.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import get_prompt
from src.utils.json_parser import parse_json_with_repair

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("count", "matched_sample", "near_miss_sample")


class AnswerAuditor:
    """Reviews a `data_analysis` count result against its near-miss sample.

    The auditor is a no-op for any tool output that doesn't follow the
    entity-count dict contract — including pure aggregations, plain numbers,
    error envelopes, and free-form prints. This keeps cost zero on the
    common case.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    def audit(self, query: str, code: str, tool_output: Any) -> Dict[str, Any]:
        """Return ``{"retry": bool, "guidance": str | None}``.

        Fail-open: any exception or malformed LLM output → ``{"retry": False}``
        so a broken auditor never blocks the synthesis path.
        """
        payload = self._extract_evidence_payload(tool_output)
        if payload is None:
            return {"retry": False}
        try:
            prompt = get_prompt("AUDITOR_PROMPT").format(
                query=query,
                code=(code or "")[:2000],
                count=payload["count"],
                matched=json.dumps(payload["matched_sample"], indent=2, default=str)[:1500],
                near_miss=json.dumps(payload["near_miss_sample"], indent=2, default=str)[:1500],
            )
            response = self.llm.invoke([
                SystemMessage(content="You are a data analysis auditor. Respond with JSON only."),
                HumanMessage(content=prompt),
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            result = parse_json_with_repair(raw, llm=self.llm, context="auditor")
            if not isinstance(result, dict) or "retry" not in result:
                logger.warning("Auditor returned non-dict or missing 'retry' — no-op")
                return {"retry": False}
            return result
        except Exception as e:
            logger.warning("Auditor failed (fail-open): %s", e)
            return {"retry": False}

    @staticmethod
    def _extract_evidence_payload(tool_output: Any) -> Optional[Dict[str, Any]]:
        """Find a JSON dict with the entity-count keys in the tool stdout.

        The data_analysis tool wraps subprocess output as
        ``{"stdout": "...", "stderr": "...", "returncode": int, ...}``.
        Look for the LLM-emitted dict in the last printed line of stdout.
        """
        # Unwrap the standard {stdout: ...} envelope; fall back to raw string.
        stdout = ""
        try:
            outer = tool_output if isinstance(tool_output, dict) else json.loads(str(tool_output))
            if isinstance(outer, dict):
                stdout = outer.get("stdout", "") or ""
            else:
                stdout = str(tool_output)
        except Exception:
            stdout = str(tool_output)

        # Scan from the last line backwards — the contract is "print the dict
        # at the end". Tolerate trailing whitespace and intermediate prints.
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict) and all(k in payload for k in _REQUIRED_KEYS):
                return payload
        return None
