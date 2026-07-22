"""Thin wrapper around `langgraph.prebuilt.create_react_agent`.

Used by the orchestrator's agentic_rag path when
`settings.agent.use_create_react_agent = True` AND the configured LLM
declares `supports_tool_calls = True` (e.g. `ToolCapableArgoLLM`).

The wrapper is intentionally minimal: it builds the prebuilt agent once at
construction, and `.run()` invokes it with the conversation messages.
Tool execution happens inside LangGraph's `ToolNode`; events are emitted
through the same contextvar-bound emitter that `tool_executor.execute`
uses, so R5 streaming continues to work for this path.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

# LangGraph 1.0 deprecated `langgraph.prebuilt.create_react_agent` in favour
# of `langchain.agents.create_agent`. Prefer the new symbol when available.
try:
    from langchain.agents import create_agent as _create_react_agent  # type: ignore
except ImportError:  # pragma: no cover — older langchain
    from langgraph.prebuilt import create_react_agent as _create_react_agent

from .event_emitter import get_emitter

logger = logging.getLogger(__name__)


class ReactExecutor:
    """Wraps `create_react_agent` and translates its output to the orchestrator's
    state shape.

    The prebuilt agent loops `LLM → tool calls → tool results → LLM` until
    the model emits an AIMessage with no tool_calls. Termination is
    LLM-driven (matches R4); the recursion limit is the safety net.
    """

    def __init__(
        self,
        llm: Any,
        tools: Sequence[BaseTool],
        system_prompt: Optional[str] = None,
        recursion_limit: int = 25,
    ):
        self._llm = llm
        self._tools = list(tools)
        self._recursion_limit = recursion_limit
        # `create_react_agent` calls `.bind_tools(tools)` on the model
        # internally — but our adapter expects `bind_tools` already called
        # (it copies tools onto a new instance). Either path works here.
        # `langchain.agents.create_agent` (1.0+) uses `system_prompt`;
        # `langgraph.prebuilt.create_react_agent` uses `prompt`.
        try:
            self._graph = _create_react_agent(
                model=llm,
                tools=self._tools,
                system_prompt=system_prompt,
            )
        except TypeError:
            self._graph = _create_react_agent(
                model=llm,
                tools=self._tools,
                prompt=system_prompt,
            )

    def run(
        self,
        question: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> Dict[str, Any]:
        """Drive the ReAct loop to completion.

        Returns:
            {
              "final_answer": str,        # last AIMessage content
              "tool_results": list[dict], # parallel to legacy ToolExecutor output
              "messages": list[BaseMessage],
            }
        """
        messages: List[BaseMessage] = list(history or [])
        messages.append(HumanMessage(content=question))

        emitter = get_emitter()
        emitter.emit("synthesis_start", {"executor": "react_executor"})

        result = self._graph.invoke(
            {"messages": messages},
            {"recursion_limit": self._recursion_limit},
        )

        out_messages: List[BaseMessage] = result.get("messages", [])

        # Last AIMessage with no tool_calls is the final answer.
        final_answer = ""
        for msg in reversed(out_messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Translate ToolMessages into the legacy {tool, result, error} shape so
        # the rest of the orchestrator (corrective_node, citation formatter)
        # can consume them unchanged.
        tool_results: List[Dict[str, Any]] = []
        for msg in out_messages:
            if isinstance(msg, ToolMessage):
                tool_results.append({
                    "tool": msg.name or "unknown",
                    "result": msg.content if isinstance(msg.content, str) else str(msg.content),
                    "error": getattr(msg, "status", "success") == "error",
                })

        emitter.emit("synthesis_end", {"executor": "react_executor", "n_tool_results": len(tool_results)})

        return {
            "final_answer": final_answer,
            "tool_results": tool_results,
            "messages": out_messages,
        }
