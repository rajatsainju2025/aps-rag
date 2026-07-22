"""End-to-end test: ReactExecutor + ToolCapableArgoLLM drive a full ReAct
loop using a mocked Argo response.

This is the integration glue between R2 (structured-output adapter) and
LangGraph's `create_react_agent`. We verify:
  - The model emits a tool_call → ToolNode runs the tool → result is fed
    back → model emits a final answer with no tool_call → loop terminates.
  - The translated tool_results have the legacy `{tool, result, error}` shape.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.react_executor import ReactExecutor
from src.integrations.argo.structured_tools_adapter import ToolCapableArgoLLM


class _Args(BaseModel):
    expression: str = Field(..., description="Math expression to evaluate")


class _MockCalculator(BaseTool):
    name: str = "calculator"
    description: str = "Evaluate a math expression and return the result."
    args_schema: type = _Args

    def _run(self, expression: str) -> str:  # pragma: no cover — trivial
        return f"<result of {expression}>"


def _llm() -> ToolCapableArgoLLM:
    return ToolCapableArgoLLM(
        model="gpt-4o",
        username="test",
        base_url="https://example.invalid",
        max_tokens=2048,
        timeout=10.0,
    )


def test_react_executor_runs_one_tool_then_emits_final_answer():
    """Two-turn loop:
        Turn 1 → AIMessage(tool_calls=[calculator(2+2)])
        Turn 2 → AIMessage(content="Answer is 4.", tool_calls=[])
    """
    llm = _llm()

    # Sequence of fake parent _generate responses (the adapter parses each
    # one's content as a JSON envelope, then synthesizes AIMessage).
    responses = iter([
        ChatResult(generations=[ChatGeneration(message=AIMessage(
            content='{"reply": "Computing.", '
                    '"tool_calls": [{"name": "calculator", "args": {"expression": "2+2"}}]}'
        ))]),
        ChatResult(generations=[ChatGeneration(message=AIMessage(
            content='{"reply": "The answer is 4.", "tool_calls": []}'
        ))]),
    ])

    def _fake_super_generate(messages, stop=None, run_manager=None, **kw):
        return next(responses)

    with patch.object(
        ToolCapableArgoLLM.__bases__[0], "_generate", autospec=False, side_effect=_fake_super_generate
    ):
        executor = ReactExecutor(llm=llm, tools=[_MockCalculator()])
        result = executor.run(question="What is two plus two?")

    assert "answer is 4" in result["final_answer"].lower()
    assert len(result["tool_results"]) == 1
    tr = result["tool_results"][0]
    assert tr["tool"] == "calculator"
    assert "<result of 2+2>" == tr["result"]
    assert tr["error"] is False


def test_react_executor_no_tool_call_returns_immediately():
    """Single-turn: LLM answers without calling any tool."""
    llm = _llm()

    one_response = ChatResult(generations=[ChatGeneration(message=AIMessage(
        content='{"reply": "Pi is approximately 3.14159.", "tool_calls": []}'
    ))])

    def _fake_super_generate(messages, stop=None, run_manager=None, **kw):
        return one_response

    with patch.object(
        ToolCapableArgoLLM.__bases__[0], "_generate", autospec=False, side_effect=_fake_super_generate
    ):
        executor = ReactExecutor(llm=llm, tools=[_MockCalculator()])
        result = executor.run(question="What is pi?")

    assert "3.14159" in result["final_answer"]
    assert result["tool_results"] == []
