"""Vendor-flow expansion for `NativeToolsArgoLLM` + `create_react_agent`.

The existing `test_react_with_native_tools.py` covers the Anthropic shape.
This file adds OpenAI (`gpt-4o`) and Gemini (`gemini-1.5-flash`) full-loop
parity tests, plus mixed scenarios.
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
    a: float = Field(..., description="First")
    b: float = Field(..., description="Second")


class _CalcTool(BaseTool):
    name: str = "add"
    description: str = "Add two numbers."
    args_schema: type = _CalcArgs

    def _run(self, a: float, b: float) -> str:
        return f"{a + b}"


def _llm(model: str) -> NativeToolsArgoLLM:
    return NativeToolsArgoLLM(
        model=model,
        username="test",
        base_url="https://example.invalid/argoapi/api/v1/resource",
        max_tokens=2048,
        timeout=10.0,
    )


def _mock_resp(envelope: dict, status: int = 200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = envelope
    m.text = json.dumps(envelope)
    return m


def _mock_httpx(envelopes: list[dict]):
    """Patch httpx.Client to return a sequence of envelopes."""
    iterator = iter(envelopes)

    def _post(*a, **kw):
        return _mock_resp(next(iterator))

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.side_effect = _post
    return mock_client


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI vendor flow
# ─────────────────────────────────────────────────────────────────────────────
class TestOpenAIFlow:
    def test_one_tool_call_then_final_answer(self):
        envelopes = [
            {"response": {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"arguments": '{"a": 5, "b": 3}', "name": "add"},
                    "type": "function",
                }],
            }},
            {"response": {
                "content": "Result: 8.",
                "tool_calls": [],
            }},
        ]
        mock_client = _mock_httpx(envelopes)
        with patch("src.integrations.argo.native_tools_client.httpx.Client",
                   return_value=mock_client):
            executor = ReactExecutor(llm=_llm("gpt-4o"), tools=[_CalcTool()])
            result = executor.run(question="What is 5 + 3?")

        assert "Result: 8" in result["final_answer"]
        assert result["tool_results"][0]["tool"] == "add"
        assert result["tool_results"][0]["result"] == "8.0"

        # Both calls must hit /chat/ with OpenAI-shaped tools and string tool_choice
        for call in mock_client.post.call_args_list:
            url = call.args[0]
            body = call.kwargs["json"]
            assert url.endswith("/chat/")
            assert body["tools"][0]["type"] == "function"
            assert body["tool_choice"] == "auto"

    def test_no_tool_call_returns_immediately(self):
        envelopes = [{"response": {"content": "7 GeV.", "tool_calls": []}}]
        mock_client = _mock_httpx(envelopes)
        with patch("src.integrations.argo.native_tools_client.httpx.Client",
                   return_value=mock_client):
            executor = ReactExecutor(llm=_llm("gpt-4o"), tools=[_CalcTool()])
            result = executor.run(question="What is the facility energy?")
        assert "7 GeV" in result["final_answer"]
        assert result["tool_results"] == []
        assert mock_client.post.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Gemini vendor flow
# ─────────────────────────────────────────────────────────────────────────────
class TestGeminiFlow:
    def test_one_tool_call_then_final_answer(self):
        envelopes = [
            {"response": {
                "content": None,
                # Gemini quirk: tool_calls is sometimes a single object not a list
                "tool_calls": {"id": None, "args": {"a": 8, "b": 5}, "name": "add"},
            }},
            {"response": {
                "content": "Result: 13.",
                "tool_calls": [],
            }},
        ]
        mock_client = _mock_httpx(envelopes)
        with patch("src.integrations.argo.native_tools_client.httpx.Client",
                   return_value=mock_client):
            executor = ReactExecutor(llm=_llm("gemini-1.5-flash"), tools=[_CalcTool()])
            result = executor.run(question="What is 8 + 5?")

        assert "Result: 13" in result["final_answer"]
        assert result["tool_results"][0]["result"] == "13.0"

        # Wire format check: Gemini omits tool_choice and uses uppercase types
        first_body = mock_client.post.call_args_list[0].kwargs["json"]
        assert "tool_choice" not in first_body
        assert first_body["tools"][0]["parameters"]["type"] == "OBJECT"
        # Top-level required (Gemini quirk)
        assert "required" in first_body["tools"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-vendor parity
# ─────────────────────────────────────────────────────────────────────────────
class TestVendorParity:
    @pytest.mark.parametrize("model,call_envelope", [
        ("gpt-4o", {
            "id": "call_1",
            "function": {"arguments": '{"a": 1, "b": 2}', "name": "add"},
            "type": "function",
        }),
        ("claude-3-5-sonnet", {
            "id": "tool_1",
            "input": {"a": 1, "b": 2},
            "name": "add",
            "type": "tool_use",
        }),
        ("gemini-1.5-flash", {
            "id": None,
            "args": {"a": 1, "b": 2},
            "name": "add",
        }),
    ])
    def test_each_vendor_produces_same_react_outcome(self, model, call_envelope):
        envelopes = [
            {"response": {"content": None, "tool_calls": [call_envelope]}},
            {"response": {"content": "Final: 3.", "tool_calls": []}},
        ]
        mock_client = _mock_httpx(envelopes)
        with patch("src.integrations.argo.native_tools_client.httpx.Client",
                   return_value=mock_client):
            executor = ReactExecutor(llm=_llm(model), tools=[_CalcTool()])
            result = executor.run(question="1 + 2?")

        assert "Final: 3" in result["final_answer"]
        assert result["tool_results"][0]["tool"] == "add"
        assert result["tool_results"][0]["result"] == "3.0"


# ─────────────────────────────────────────────────────────────────────────────
# Empty tool_calls envelope across vendors
# ─────────────────────────────────────────────────────────────────────────────
class TestEmptyToolCalls:
    @pytest.mark.parametrize("model", ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-flash"])
    def test_no_tool_calls_means_no_extra_round(self, model):
        envelopes = [{"response": {"content": "Direct answer.", "tool_calls": []}}]
        mock_client = _mock_httpx(envelopes)
        with patch("src.integrations.argo.native_tools_client.httpx.Client",
                   return_value=mock_client):
            executor = ReactExecutor(llm=_llm(model), tools=[_CalcTool()])
            result = executor.run(question="hi")
        assert "Direct answer" in result["final_answer"]
        assert mock_client.post.call_count == 1
