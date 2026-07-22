"""End-to-end: NativeToolsArgoLLM (Anthropic shape) drives a full ReAct loop
through `create_react_agent`, with every Argo `/chat/` round-trip mocked.

Asserts:
  - The first /chat/ call returns a tool_use → ToolNode runs the tool.
  - The second /chat/ call sees the ToolMessage and returns a final answer
    with no tool_calls → the loop terminates.
  - `ReactExecutor.run()` translates the ToolMessage into our legacy
    `{tool, result, error}` shape so corrective_node / formatter consume it
    unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.react_executor import ReactExecutor
from src.integrations.argo.native_tools_client import NativeToolsArgoLLM


class _CalcArgs(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class _CalcTool(BaseTool):
    name: str = "add"
    description: str = "Add two numbers together."
    args_schema: type = _CalcArgs

    def _run(self, a: float, b: float) -> str:
        return f"{a + b}"


def _mock_resp(envelope: dict, status: int = 200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = envelope
    m.text = json.dumps(envelope)
    return m


def _llm() -> NativeToolsArgoLLM:
    return NativeToolsArgoLLM(
        model="claude-3-5-sonnet",
        username="test",
        base_url="https://example.invalid/argoapi/api/v1/resource",
        max_tokens=2048,
        timeout=10.0,
    )


def test_react_loop_runs_one_anthropic_tool_then_emits_final_answer():
    # Two /chat/ envelopes: turn 1 → tool_use, turn 2 → final answer with empty tool_calls.
    envelopes = iter([
        {"response": {
            "content": "Computing 2 + 2 for you.",
            "tool_calls": [{
                "id": "toolu_01",
                "input": {"a": 2, "b": 2},
                "name": "add",
                "type": "tool_use",
            }],
        }},
        {"response": {
            "content": "The answer is 4.",
            "tool_calls": [],
        }},
    ])

    with patch("src.integrations.argo.native_tools_client.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = lambda *a, **kw: _mock_resp(next(envelopes))
        mock_client_cls.return_value = mock_client

        executor = ReactExecutor(llm=_llm(), tools=[_CalcTool()])
        result = executor.run(question="What is 2 + 2?")

    # Final answer surfaced
    assert "answer is 4" in result["final_answer"].lower()

    # Tool result translated to legacy shape
    assert len(result["tool_results"]) == 1
    tr = result["tool_results"][0]
    assert tr["tool"] == "add"
    assert tr["result"] == "4.0"
    assert tr["error"] is False

    # Both /chat/ calls hit the right endpoint with bound tools attached
    assert mock_client.post.call_count == 2
    for call in mock_client.post.call_args_list:
        url = call.args[0]
        body = call.kwargs["json"]
        assert url.endswith("/chat/")
        assert "tools" in body
        assert body["tools"][0]["name"] == "add"
        assert body["tool_choice"] == {"type": "auto"}


def test_react_loop_terminates_immediately_when_no_tool_call():
    envelope = {"response": {
        "content": "Pi is roughly 3.14159.",
        "tool_calls": [],
    }}

    with patch("src.integrations.argo.native_tools_client.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = _mock_resp(envelope)
        mock_client_cls.return_value = mock_client

        executor = ReactExecutor(llm=_llm(), tools=[_CalcTool()])
        result = executor.run(question="What is pi?")

    assert "3.14159" in result["final_answer"]
    assert result["tool_results"] == []
    # Just one /chat/ round-trip — no tool, no second turn
    assert mock_client.post.call_count == 1
