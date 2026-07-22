"""
Unit tests for multi-round tool execution in ReasoningAgent.
Covers: tool call extraction across rounds, archiver_read parsing.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.reasoning_agent import ReasoningAgent


@pytest.fixture
def agent():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="test response")
    return ReasoningAgent(llm=mock_llm)


class TestToolCallExtraction:
    def test_first_round_extracts_tools(self, agent):
        draft = "Let me look that up.\nTOOL_CALL: channel_finder(beam current)"
        tool_calls = agent._extract_tool_calls(draft)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "channel_finder"
        assert tool_calls[0]["args"] == "beam current"

    def test_archiver_read_pipe_parsing(self, agent):
        draft = "TOOL_CALL: archiver_read(S-DCCT:CurrentM||1h)"
        tool_calls = agent._extract_tool_calls(draft)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "archiver_read"
        assert tool_calls[0]["args"]["pv_names"] == "S-DCCT:CurrentM"
        assert tool_calls[0]["args"]["start"] == "1h"

    def test_archiver_read_multi_pv_pipe(self, agent):
        draft = "TOOL_CALL: archiver_read(PV1, PV2||24h)"
        tool_calls = agent._extract_tool_calls(draft)
        assert len(tool_calls) == 1
        assert tool_calls[0]["args"]["pv_names"] == "PV1, PV2"
        assert tool_calls[0]["args"]["start"] == "24h"

    def test_data_analysis_still_works(self, agent):
        draft = "TOOL_CALL: data_analysis(df.head()||work_requests)"
        tool_calls = agent._extract_tool_calls(draft)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "data_analysis"
        assert tool_calls[0]["args"]["data_source"] == "work_requests"


class TestConfidenceDrivenTermination:
    """After R4, ReasoningAgent.synthesize trusts the LLM's emitted TOOL_CALL
    directives — the orchestrator enforces the safety cap, not the agent."""

    def test_second_round_allowed(self, agent):
        """With 1 prior tool result, a second round of tool calls is allowed."""
        state = {
            "original_query": "What is the beam current?",
            "enhanced_query": "What is the beam current?",
            "retrieved_docs": [],
            "memory_context": {},
            "tool_results": [{"tool": "channel_finder", "result": "...", "error": False}],
            "reasoning_trace": [],
            "correction_round": 0,
        }
        agent.llm.invoke.return_value = MagicMock(
            content="<think>using pv_read</think>\nTOOL_CALL: pv_read(S-DCCT:CurrentM)"
        )
        result = agent.synthesize(state)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "pv_read"

    def test_third_round_still_allowed(self, agent):
        """With 2 prior tool results, a third tool call (e.g. archiver_read)
        is still permitted — the old hard 2-round cap is removed."""
        state = {
            "original_query": "What is the beam current and trend?",
            "enhanced_query": "What is the beam current and trend?",
            "retrieved_docs": [],
            "memory_context": {},
            "tool_results": [
                {"tool": "channel_finder", "result": "...", "error": False},
                {"tool": "pv_read", "result": "...", "error": False},
            ],
            "reasoning_trace": [],
            "correction_round": 0,
        }
        agent.llm.invoke.return_value = MagicMock(
            content="<think>need archiver too</think>\nTOOL_CALL: archiver_read(S-DCCT:CurrentM||1h)"
        )
        result = agent.synthesize(state)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "archiver_read"

    def test_no_more_tool_calls_when_llm_emits_none(self, agent):
        """LLM-driven termination: if the draft has no TOOL_CALL directives,
        synthesize returns an empty tool_calls list regardless of prior count."""
        state = {
            "original_query": "What is the beam current?",
            "enhanced_query": "What is the beam current?",
            "retrieved_docs": [],
            "memory_context": {},
            "tool_results": [
                {"tool": "channel_finder", "result": "...", "error": False},
                {"tool": "pv_read", "result": "100.2 mA", "error": False},
            ],
            "reasoning_trace": [],
            "correction_round": 0,
        }
        agent.llm.invoke.return_value = MagicMock(
            content="The beam current is 100.2 mA (S-DCCT:CurrentM)."
        )
        result = agent.synthesize(state)
        assert result["tool_calls"] == []


class TestPlannedPVToolInstructions:
    def test_pv_search_plan_injects_tool_call(self, agent):
        state = {
            "plan_steps": [
                {"step": 1, "tool": "pv_search", "query": "beam current"},
                {"step": 2, "tool": "synthesize", "query": None},
            ]
        }

        instructions = agent._format_plan_tool_instructions(state, tool_results=[])

        assert "TOOL_CALL: pv_search(beam current)" in instructions
