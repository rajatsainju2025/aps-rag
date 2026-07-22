"""Make Argo (which has no native tool-use API) look tools-capable to LangGraph.

`langgraph.prebuilt.create_react_agent` requires a chat model whose response
can carry `AIMessage.tool_calls`. Argo only returns plain text. This adapter
wraps `ArgoLLMClientAG`:

  1. `bind_tools(tools)` stores the tools and returns a new instance.
  2. The next `_generate()` injects a strict JSON-mode system prompt that
     describes the bound tools and asks Argo for an envelope of the form:

        {"reply": "...", "tool_calls": [{"name": "...", "args": {...}}, ...]}

  3. The response is parsed; on success an `AIMessage` is returned with
     `tool_calls` populated. On parse failure the raw text becomes the
     reply (no tool calls), and a `argo_structured_output_parse_failures`
     counter is incremented for observability.

Design notes:
  - `supports_tool_calls = True` flag on this class so callers can detect
    capability without `isinstance` checks.
  - We don't try to validate `args` against each tool's Pydantic schema here.
    `ToolNode` does validation on its side and surfaces errors back to the
    LLM via `ToolMessage(status="error")`.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from src.agents._tool_schema_render import render_tool_schemas
from src.integrations.argo.llm_client import ArgoLLMClientAG

logger = logging.getLogger(__name__)


# Module-level counter — cheap observability without a metrics dep.
# Reset by tests; production code can read it for monitoring.
PARSE_FAILURE_COUNT: int = 0


_JSON_MODE_INSTRUCTION = """\
You are operating in TOOL-USE mode. Below are the tools you may call.

{tool_schemas}

You MUST respond with a single JSON object — no surrounding prose, no code \
fences, no commentary. The JSON object must match this shape exactly:

{{"reply": "<your natural-language answer to the user>", "tool_calls": [\
{{"name": "<tool_name>", "args": {{"<arg>": <value>, ...}}}}, ...]}}

Rules:
  - When you can answer directly without tools, set `tool_calls` to `[]` and \
put the full answer in `reply`.
  - When you need a tool, populate `tool_calls` with one or more entries. \
You may call multiple tools in parallel by listing them all in `tool_calls`. \
`reply` may be a short status message ("Looking up …") or "" — it will not \
be shown to the user until tools have run.
  - `args` MUST be a JSON object whose keys match the tool's `properties`. \
Use real values, never the placeholder names from the schema.
  - Output JSON ONLY. No prefix, no suffix, no markdown.
"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first valid JSON object out of arbitrary model output.

    LLMs frequently wrap JSON in code fences or prefix it with chatter
    despite explicit instructions. We try strict parse first, then
    fall back to a greedy regex match for the outermost {...}.
    """
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(text)
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class ToolCapableArgoLLM(ArgoLLMClientAG):
    """Argo wrapper that emits AIMessage.tool_calls via JSON-mode prompting."""

    # ClassVar so Pydantic doesn't treat this as an instance field — it's a
    # capability flag callers (orchestrator) read without instantiating.
    supports_tool_calls: ClassVar[bool] = True

    # Bound tools, set by bind_tools(). Stored as a list of BaseTool; not
    # serialized in repr to keep logs readable.
    bound_tools: List[Any] = []

    model_config = {"arbitrary_types_allowed": True}

    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[BaseTool],
        **kwargs: Any,
    ) -> "ToolCapableArgoLLM":
        """Return a new instance with the given tools bound.

        LangGraph's `create_react_agent` calls this once at graph build
        time, then drives the resulting model through repeated `.invoke()`
        calls — so the bound tools persist across the loop.
        """
        copy = self.model_copy(deep=False)
        copy.bound_tools = list(tools)
        return copy

    def _build_system_prompt(self) -> str:
        tools_dict = {t.name: t for t in self.bound_tools}
        return _JSON_MODE_INSTRUCTION.format(
            tool_schemas=render_tool_schemas(tools_dict),
        )

    def _generate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # If no tools are bound we behave exactly like the parent — useful
        # for nodes that share the LLM instance but don't need tool-use.
        if not self.bound_tools:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        # Prepend our JSON-mode system prompt. We do this on a per-call copy
        # of `messages` so we don't mutate the caller's list.
        prefixed = [SystemMessage(content=self._build_system_prompt()), *messages]
        result = super()._generate(prefixed, stop=stop, run_manager=run_manager, **kwargs)

        # `result.generations[0].message` is the AIMessage from the parent.
        original_text = result.generations[0].message.content if result.generations else ""
        parsed = _extract_first_json_object(original_text or "")

        if parsed is None or not isinstance(parsed, dict) or "reply" not in parsed:
            global PARSE_FAILURE_COUNT
            PARSE_FAILURE_COUNT += 1
            logger.warning(
                "ToolCapableArgoLLM: failed to parse JSON envelope (count=%d). "
                "Falling back to raw text reply.",
                PARSE_FAILURE_COUNT,
            )
            return result

        reply: str = str(parsed.get("reply", "") or "")
        raw_tool_calls = parsed.get("tool_calls", []) or []
        tool_calls: List[Dict[str, Any]] = []
        for tc in raw_tool_calls:
            if not isinstance(tc, dict) or "name" not in tc:
                continue
            args = tc.get("args", {})
            if not isinstance(args, dict):
                args = {"input": args}
            tool_calls.append({
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": tc["name"],
                "args": args,
                "type": "tool_call",
            })

        message = AIMessage(content=reply, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])
