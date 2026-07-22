"""Tests for `NativeToolsArgoLLM` — Argo `/chat/` native tool calling."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.integrations.argo.native_tools_client import (
    NativeToolsArgoLLM,
    _from_anthropic_response,
    _from_gemini_response,
    _from_openai_response,
    _to_anthropic_tools,
    _to_gemini_tools,
    _to_openai_tools,
    _vendor_for_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
class _CalcArgs(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class _CalcTool(BaseTool):
    name: str = "add"
    description: str = "Add two numbers together."
    args_schema: type = _CalcArgs

    def _run(self, a: float, b: float) -> str:
        return str(a + b)


class _StockArgs(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol like AAPL")


class _StockTool(BaseTool):
    name: str = "get_stock_price"
    description: str = "Get the current stock price for a ticker."
    args_schema: type = _StockArgs

    def _run(self, ticker: str) -> str:
        return f"<price for {ticker}>"


def _make_llm(model: str = "claude-3-5-sonnet") -> NativeToolsArgoLLM:
    return NativeToolsArgoLLM(
        model=model,
        username="test-user",
        base_url="https://example.invalid/argoapi/api/v1/resource",
        max_tokens=2048,
        timeout=10.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vendor detection
# ─────────────────────────────────────────────────────────────────────────────
class TestVendorDetection:
    @pytest.mark.parametrize("model,vendor", [
        ("gpt-4o", "openai"),
        ("gpt-4o", "openai"),
        ("o1-preview", "openai"),
        ("o3", "openai"),
        ("claude-3-5-sonnet", "anthropic"),
        ("claude-3-opus", "anthropic"),
        ("claude-3-5-haiku", "anthropic"),
        ("gemini-1.5-flash", "gemini"),
        ("gemini-1.5-pro", "gemini"),
        ("mystery_model", "unknown"),
    ])
    def test_vendor_detection(self, model: str, vendor: str):
        assert _vendor_for_model(model) == vendor


# ─────────────────────────────────────────────────────────────────────────────
# Request shape adapters (BaseTool list → vendor `tools` field)
# ─────────────────────────────────────────────────────────────────────────────
class TestOpenAIRequestShape:
    def test_wraps_in_function_envelope(self):
        out = _to_openai_tools([_CalcTool()])
        assert len(out) == 1
        assert out[0]["type"] == "function"
        fn = out[0]["function"]
        assert fn["name"] == "add"
        assert fn["description"] == "Add two numbers together."
        assert fn["parameters"]["type"] == "object"
        assert set(fn["parameters"]["required"]) == {"a", "b"}
        assert "a" in fn["parameters"]["properties"]
        # Pydantic 'title' field stripped
        assert "title" not in fn["parameters"]["properties"]["a"]


class TestAnthropicRequestShape:
    def test_uses_input_schema_field(self):
        out = _to_anthropic_tools([_StockTool()])
        assert len(out) == 1
        assert out[0]["name"] == "get_stock_price"
        # Anthropic uses input_schema (not parameters)
        assert "input_schema" in out[0]
        assert "parameters" not in out[0]
        assert out[0]["input_schema"]["required"] == ["ticker"]


class TestGeminiRequestShape:
    def test_required_at_top_level_not_in_parameters(self):
        out = _to_gemini_tools([_CalcTool()])
        assert len(out) == 1
        # Gemini quirk: `required` lives at the top level, NOT inside `parameters`.
        assert "required" in out[0]
        assert set(out[0]["required"]) == {"a", "b"}
        assert "required" not in out[0]["parameters"]

    def test_uppercase_type_names(self):
        out = _to_gemini_tools([_CalcTool()])
        assert out[0]["parameters"]["type"] == "OBJECT"
        # Numeric properties should also be uppercased (NUMBER, not number)
        for prop in out[0]["parameters"]["properties"].values():
            assert prop["type"] == "NUMBER"


# ─────────────────────────────────────────────────────────────────────────────
# Response shape adapters (vendor response → unified ToolCall list)
# ─────────────────────────────────────────────────────────────────────────────
class TestOpenAIResponseShape:
    def test_parses_arguments_json_string(self):
        raw = [{
            "id": "call_abc123",
            "function": {"arguments": '{"a": 8, "b": 5}', "name": "add"},
            "type": "function",
        }]
        out = _from_openai_response(raw)
        assert len(out) == 1
        assert out[0]["id"] == "call_abc123"
        assert out[0]["name"] == "add"
        assert out[0]["args"] == {"a": 8, "b": 5}
        assert out[0]["type"] == "tool_call"

    def test_handles_invalid_json_arguments(self):
        raw = [{
            "id": "call_x",
            "function": {"arguments": "not-json{", "name": "add"},
            "type": "function",
        }]
        out = _from_openai_response(raw)
        assert out[0]["args"]["_raw"] == "not-json{"

    def test_empty_response(self):
        assert _from_openai_response(None) == []
        assert _from_openai_response([]) == []


class TestAnthropicResponseShape:
    def test_parses_input_dict(self):
        raw = [{
            "id": "toolu_vrtx_01X",
            "input": {"ticker": "MSFT"},
            "name": "get_stock_price",
            "type": "tool_use",
        }]
        out = _from_anthropic_response(raw)
        assert len(out) == 1
        assert out[0]["id"] == "toolu_vrtx_01X"
        assert out[0]["name"] == "get_stock_price"
        assert out[0]["args"] == {"ticker": "MSFT"}

    def test_synthesizes_id_when_missing(self):
        raw = [{"input": {"ticker": "MSFT"}, "name": "get_stock_price"}]
        out = _from_anthropic_response(raw)
        assert out[0]["id"].startswith("call_")


class TestGeminiResponseShape:
    def test_parses_args_dict(self):
        raw = [{"id": None, "args": {"a": 8, "b": 5}, "name": "add"}]
        out = _from_gemini_response(raw)
        assert len(out) == 1
        assert out[0]["name"] == "add"
        assert out[0]["args"] == {"a": 8, "b": 5}
        # ID synthesized because Gemini doesn't return one
        assert out[0]["id"].startswith("call_")

    def test_handles_single_object_response(self):
        """Gemini quirk: notes show response as single object, not list."""
        raw = {"id": None, "args": {"a": 1, "b": 2}, "name": "add"}
        out = _from_gemini_response(raw)
        assert len(out) == 1
        assert out[0]["name"] == "add"


# ─────────────────────────────────────────────────────────────────────────────
# bind_tools contract
# ─────────────────────────────────────────────────────────────────────────────
class TestBindTools:
    def test_returns_new_instance_with_tools(self):
        llm = _make_llm()
        bound = llm.bind_tools([_CalcTool()])
        assert bound is not llm
        assert len(bound.bound_tools) == 1
        assert llm.bound_tools == []

    def test_supports_tool_calls_classvar_is_true(self):
        assert NativeToolsArgoLLM.supports_tool_calls is True


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint URL
# ─────────────────────────────────────────────────────────────────────────────
class TestEndpointURL:
    def test_uses_chat_endpoint_not_streamchat(self):
        llm = _make_llm()
        url = llm._get_chat_url()
        assert url.endswith("/chat/")
        assert "streamchat" not in url


# ─────────────────────────────────────────────────────────────────────────────
# Payload assembly
# ─────────────────────────────────────────────────────────────────────────────
class TestPayloadAssembly:
    def test_no_tools_means_no_tools_field(self):
        llm = _make_llm()
        payload = llm._attach_tools({"messages": []})
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_anthropic_payload_uses_dict_tool_choice(self):
        llm = _make_llm("claude-3-5-sonnet").bind_tools([_CalcTool()])
        payload = llm._attach_tools({"messages": []})
        assert payload["tool_choice"] == {"type": "auto"}
        assert payload["tools"][0]["name"] == "add"
        assert "input_schema" in payload["tools"][0]

    def test_openai_payload_uses_string_tool_choice(self):
        llm = _make_llm("gpt-4o").bind_tools([_CalcTool()])
        payload = llm._attach_tools({"messages": []})
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["type"] == "function"

    def test_gemini_payload_omits_tool_choice(self):
        llm = _make_llm("gemini-1.5-flash").bind_tools([_CalcTool()])
        payload = llm._attach_tools({"messages": []})
        assert "tool_choice" not in payload
        assert payload["tools"][0]["parameters"]["type"] == "OBJECT"


# ─────────────────────────────────────────────────────────────────────────────
# _generate end-to-end with mocked httpx
# ─────────────────────────────────────────────────────────────────────────────
class TestGenerateRoundTrip:
    """Patch httpx.Client at the module level so the real `_generate` runs."""

    def _mock_response(self, envelope: dict, status: int = 200):
        m = MagicMock()
        m.status_code = status
        m.json.return_value = envelope
        m.text = json.dumps(envelope)
        return m

    def test_anthropic_round_trip_with_tool_call(self):
        llm = _make_llm("claude-3-5-sonnet").bind_tools([_StockTool()])
        envelope = {"response": {
            "content": "I'll get the stock price.",
            "tool_calls": [{
                "id": "toolu_01",
                "input": {"ticker": "MSFT"},
                "name": "get_stock_price",
                "type": "tool_use",
            }],
        }}

        with patch("src.integrations.argo.native_tools_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = self._mock_response(envelope)
            mock_client_cls.return_value = mock_client

            result = llm._generate([HumanMessage(content="What's MSFT price?")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert "stock price" in msg.content
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_stock_price"
        assert msg.tool_calls[0]["args"] == {"ticker": "MSFT"}

        # Verify the wire request hit /chat/ with tools attached
        post_args, post_kwargs = mock_client.post.call_args
        assert post_args[0].endswith("/chat/")
        sent = post_kwargs["json"]
        assert "tools" in sent
        assert sent["tool_choice"] == {"type": "auto"}

    def test_openai_round_trip_parses_arguments_string(self):
        llm = _make_llm("gpt-4o").bind_tools([_CalcTool()])
        envelope = {"response": {
            "content": None,
            "tool_calls": [{
                "id": "call_xyz",
                "function": {"arguments": '{"a": 8, "b": 5}', "name": "add"},
                "type": "function",
            }],
        }}

        with patch("src.integrations.argo.native_tools_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = self._mock_response(envelope)
            mock_client_cls.return_value = mock_client

            result = llm._generate([HumanMessage(content="What is 8 plus 5?")])

        msg = result.generations[0].message
        assert msg.content == ""  # null → ""
        assert msg.tool_calls[0]["args"] == {"a": 8, "b": 5}

    def test_no_tools_bound_still_hits_chat_endpoint(self):
        """Per user directive: planner uses /chat/ even without tools."""
        llm = _make_llm("claude-3-5-sonnet")  # no bind_tools
        envelope = {"response": {
            "content": "Plain reply, no tools called.",
            "tool_calls": [],
        }}

        with patch("src.integrations.argo.native_tools_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = self._mock_response(envelope)
            mock_client_cls.return_value = mock_client

            result = llm._generate([HumanMessage(content="hello")])

        assert result.generations[0].message.content == "Plain reply, no tools called."
        post_args, post_kwargs = mock_client.post.call_args
        assert post_args[0].endswith("/chat/")
        assert "tools" not in post_kwargs["json"]

    def test_plain_string_response_envelope(self):
        """Argo /chat/ returns `{"response": "<text>"}` (string, not dict) when
        no tools are involved — the actual planner case. Pre-fix, the code
        only handled the dict shape and silently dropped the content, which
        produced empty plans → "Could not parse plan JSON" → synthesize-only
        fallback for every PV / multi_hop query in the Apr 30 50-q eval."""
        llm = _make_llm("claude-3-5-sonnet")
        envelope = {"response": "Hello there, how are you?"}

        with patch("src.integrations.argo.native_tools_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = self._mock_response(envelope)
            mock_client_cls.return_value = mock_client

            result = llm._generate([HumanMessage(content="say hi")])

        msg = result.generations[0].message
        assert msg.content == "Hello there, how are you?"
        assert msg.tool_calls == []

    def test_top_level_content_field_envelope(self):
        """Defensive: tolerate envelopes that put `content`/`tool_calls` at
        the top level (rare variant some Argo deployments have returned)."""
        llm = _make_llm("claude-3-5-sonnet")
        envelope = {"content": "top-level content", "tool_calls": []}

        with patch("src.integrations.argo.native_tools_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = self._mock_response(envelope)
            mock_client_cls.return_value = mock_client

            result = llm._generate([HumanMessage(content="say hi")])

        assert result.generations[0].message.content == "top-level content"

    def test_http_error_raises(self):
        llm = _make_llm("claude-3-5-sonnet")
        with patch("src.integrations.argo.native_tools_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            err_resp = MagicMock()
            err_resp.status_code = 500
            err_resp.text = "internal error"
            mock_client.post.return_value = err_resp
            mock_client_cls.return_value = mock_client

            with pytest.raises(Exception, match="HTTP 500"):
                llm._generate([HumanMessage(content="hi")])
