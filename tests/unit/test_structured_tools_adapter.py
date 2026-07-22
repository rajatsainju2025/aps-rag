"""Tests for the structured-output Argo adapter (R2)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.integrations.argo import structured_tools_adapter
from src.integrations.argo.structured_tools_adapter import ToolCapableArgoLLM


class _Args(BaseModel):
    query: str = Field(..., description="Search query")


class _FakeTool(BaseTool):
    name: str = "channel_finder"
    description: str = "Find an EPICS channel by description."
    args_schema: type = _Args

    def _run(self, query: str) -> str:
        return f"<channels for '{query}'>"


def _make_llm() -> ToolCapableArgoLLM:
    return ToolCapableArgoLLM(
        model="gpt-4o",
        username="test-user",
        base_url="https://example.invalid",
        max_tokens=2048,
        timeout=10.0,
    )


@pytest.fixture(autouse=True)
def _reset_parse_counter():
    structured_tools_adapter.PARSE_FAILURE_COUNT = 0
    yield
    structured_tools_adapter.PARSE_FAILURE_COUNT = 0


class TestBindToolsContract:
    def test_bind_tools_returns_new_instance_with_tools(self):
        llm = _make_llm()
        bound = llm.bind_tools([_FakeTool()])
        assert bound is not llm
        assert len(bound.bound_tools) == 1
        assert bound.bound_tools[0].name == "channel_finder"
        # Original unchanged
        assert llm.bound_tools == []

    def test_supports_tool_calls_capability_flag_is_true(self):
        assert ToolCapableArgoLLM.supports_tool_calls is True


class TestEnvelopeParsing:
    def test_valid_envelope_with_tool_calls(self):
        llm = _make_llm().bind_tools([_FakeTool()])
        fake_response = ChatResult(generations=[
            ChatGeneration(message=AIMessage(
                content='{"reply": "Looking up the beam current PV.", '
                        '"tool_calls": [{"name": "channel_finder", "args": {"query": "beam current"}}]}'
            ))
        ])
        with patch.object(
            ToolCapableArgoLLM.__bases__[0], "_generate", return_value=fake_response
        ):
            result = llm._generate([HumanMessage(content="What is the beam current?")])

        msg = result.generations[0].message
        assert msg.content == "Looking up the beam current PV."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "channel_finder"
        assert msg.tool_calls[0]["args"] == {"query": "beam current"}
        assert msg.tool_calls[0]["id"].startswith("call_")

    def test_envelope_with_empty_tool_calls(self):
        llm = _make_llm().bind_tools([_FakeTool()])
        fake_response = ChatResult(generations=[
            ChatGeneration(message=AIMessage(
                content='{"reply": "The the facility storage ring runs at 7 GeV.", "tool_calls": []}'
            ))
        ])
        with patch.object(
            ToolCapableArgoLLM.__bases__[0], "_generate", return_value=fake_response
        ):
            result = llm._generate([HumanMessage(content="What energy is the facility ring?")])

        msg = result.generations[0].message
        assert msg.content == "The the facility storage ring runs at 7 GeV."
        assert msg.tool_calls == []

    def test_parallel_tool_calls(self):
        llm = _make_llm().bind_tools([_FakeTool()])
        fake_response = ChatResult(generations=[
            ChatGeneration(message=AIMessage(
                content=(
                    '{"reply": "Querying both PVs.", "tool_calls": ['
                    '{"name": "channel_finder", "args": {"query": "current"}},'
                    '{"name": "channel_finder", "args": {"query": "lifetime"}}'
                    ']}'
                )
            ))
        ])
        with patch.object(
            ToolCapableArgoLLM.__bases__[0], "_generate", return_value=fake_response
        ):
            result = llm._generate([HumanMessage(content="Get current and lifetime.")])

        msg = result.generations[0].message
        assert len(msg.tool_calls) == 2
        # Each gets a unique id for the tool-result correlation
        assert msg.tool_calls[0]["id"] != msg.tool_calls[1]["id"]


class TestFallbackBehaviour:
    def test_malformed_json_increments_counter_and_returns_raw_text(self):
        llm = _make_llm().bind_tools([_FakeTool()])
        fake_response = ChatResult(generations=[
            ChatGeneration(message=AIMessage(
                content="Sorry, I'm not in JSON mode today!"
            ))
        ])
        with patch.object(
            ToolCapableArgoLLM.__bases__[0], "_generate", return_value=fake_response
        ):
            result = llm._generate([HumanMessage(content="hi")])

        # Raw text passes through, no tool_calls
        assert result.generations[0].message.content == "Sorry, I'm not in JSON mode today!"
        assert result.generations[0].message.tool_calls == []
        assert structured_tools_adapter.PARSE_FAILURE_COUNT == 1

    def test_envelope_wrapped_in_code_fences_still_parses(self):
        llm = _make_llm().bind_tools([_FakeTool()])
        fake_response = ChatResult(generations=[
            ChatGeneration(message=AIMessage(
                content='```json\n{"reply": "ok", "tool_calls": []}\n```'
            ))
        ])
        with patch.object(
            ToolCapableArgoLLM.__bases__[0], "_generate", return_value=fake_response
        ):
            result = llm._generate([HumanMessage(content="hi")])

        assert result.generations[0].message.content == "ok"
        assert structured_tools_adapter.PARSE_FAILURE_COUNT == 0

    def test_no_bound_tools_skips_json_mode(self):
        """With no tools bound, the adapter must behave like the plain Argo client."""
        llm = _make_llm()  # no bind_tools call
        plain_response = ChatResult(generations=[
            ChatGeneration(message=AIMessage(content="just a plain reply"))
        ])
        with patch.object(
            ToolCapableArgoLLM.__bases__[0], "_generate", return_value=plain_response
        ) as mocked_super:
            result = llm._generate([HumanMessage(content="hi")])
            # The system prompt should NOT have been injected
            (passed_messages,), _ = mocked_super.call_args
            assert all("TOOL-USE mode" not in (m.content or "") for m in passed_messages)

        assert result.generations[0].message.content == "just a plain reply"
