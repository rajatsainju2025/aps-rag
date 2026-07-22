"""Unit tests for ToolExecutor — dispatch, error handling, and event emission.

ToolExecutor is the heart of agent ↔ tool interaction. These tests pin down:
- Dispatch by name from the registered tools dict
- Missing-tool error path
- String vs dict args (LangChain BaseTool accepts both)
- Per-call exception handling (one failure does not abort the batch)
- Event emission for start / result / error including duration_ms
- Empty tool_calls is a no-op returning `{"tool_results": []}`
- The legacy {tool, result, error} dict shape stays intact (consumed by reasoning_agent)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.event_emitter import current_event_emitter
from src.agents.tool_executor import ToolExecutor


class _Recorder:
    """Captures emitted events for assertion."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


@pytest.fixture
def calc_tool():
    t = MagicMock()
    t.run.return_value = "13"
    return t


@pytest.fixture
def slow_tool():
    t = MagicMock()
    t.run.return_value = "<slow>"
    return t


@pytest.fixture
def broken_tool():
    t = MagicMock()
    t.run.side_effect = RuntimeError("disk full")
    return t


def _state(*calls: dict) -> dict:
    return {"tool_calls": list(calls)}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch and arg shapes
# ─────────────────────────────────────────────────────────────────────────────
class TestDispatch:
    def test_no_pending_calls_returns_empty(self, calc_tool):
        ex = ToolExecutor(tools={"calculator": calc_tool})
        result = ex.execute({"tool_calls": []})
        assert result == {"tool_results": []}

    def test_string_args_passed_through(self, calc_tool):
        ex = ToolExecutor(tools={"calculator": calc_tool})
        result = ex.execute(_state({"tool": "calculator", "args": "8 + 5"}))
        calc_tool.run.assert_called_once_with("8 + 5")
        assert result["tool_results"][0]["result"] == "13"
        assert result["tool_results"][0]["error"] is False

    def test_dict_args_use_tool_input_kw(self, calc_tool):
        ex = ToolExecutor(tools={"calculator": calc_tool})
        ex.execute(_state({"tool": "calculator", "args": {"expression": "8+5"}}))
        # When args is a dict the executor passes tool_input=<dict>
        calc_tool.run.assert_called_once_with(tool_input={"expression": "8+5"})

    def test_clears_tool_calls_after_dispatch(self, calc_tool):
        ex = ToolExecutor(tools={"calculator": calc_tool})
        result = ex.execute(_state({"tool": "calculator", "args": "1+1"}))
        assert result["tool_calls"] == []


class TestMissingTool:
    def test_unregistered_tool_returns_error_record(self):
        ex = ToolExecutor(tools={})
        result = ex.execute(_state({"tool": "ghost", "args": "x"}))
        tr = result["tool_results"][0]
        assert tr["tool"] == "ghost"
        assert tr["error"] is True
        assert "not available" in tr["result"]


class TestPartialFailure:
    def test_one_failure_does_not_abort_others(self, calc_tool, broken_tool):
        ex = ToolExecutor(tools={"calculator": calc_tool, "broken": broken_tool})
        result = ex.execute(_state(
            {"tool": "broken", "args": "x"},
            {"tool": "calculator", "args": "1+1"},
        ))
        assert len(result["tool_results"]) == 2
        assert result["tool_results"][0]["error"] is True
        assert "disk full" in result["tool_results"][0]["result"]
        assert result["tool_results"][1]["error"] is False
        assert result["tool_results"][1]["result"] == "13"


# ─────────────────────────────────────────────────────────────────────────────
# Event emission (ContextVar-bound emitter)
# ─────────────────────────────────────────────────────────────────────────────
class TestEventEmission:
    def test_emits_start_then_result_for_success(self, calc_tool):
        ex = ToolExecutor(tools={"calculator": calc_tool})
        rec = _Recorder()
        token = current_event_emitter.set(rec)
        try:
            ex.execute(_state({"tool": "calculator", "args": "1+1"}))
        finally:
            current_event_emitter.reset(token)
        kinds = [e[0] for e in rec.events]
        assert kinds == ["tool_call_start", "tool_call_result"]
        # duration_ms key present
        assert "duration_ms" in rec.events[1][1]

    def test_emits_error_for_exception(self, broken_tool):
        ex = ToolExecutor(tools={"broken": broken_tool})
        rec = _Recorder()
        token = current_event_emitter.set(rec)
        try:
            ex.execute(_state({"tool": "broken", "args": "x"}))
        finally:
            current_event_emitter.reset(token)
        kinds = [e[0] for e in rec.events]
        assert kinds == ["tool_call_start", "tool_call_error"]
        assert rec.events[1][1]["error"] == "disk full"

    def test_emits_error_for_missing_tool(self):
        ex = ToolExecutor(tools={})
        rec = _Recorder()
        token = current_event_emitter.set(rec)
        try:
            ex.execute(_state({"tool": "ghost", "args": "x"}))
        finally:
            current_event_emitter.reset(token)
        kinds = [e[0] for e in rec.events]
        assert kinds == ["tool_call_start", "tool_call_error"]

    def test_no_emitter_bound_is_no_op(self, calc_tool):
        """When no ContextVar emitter is set, the executor must still work."""
        # Ensure no emitter leaked from a prior test
        try:
            current_event_emitter.set(None)
        except LookupError:
            pass
        ex = ToolExecutor(tools={"calculator": calc_tool})
        result = ex.execute(_state({"tool": "calculator", "args": "1+1"}))
        assert result["tool_results"][0]["result"] == "13"

    def test_each_call_in_batch_gets_paired_events(self, calc_tool, broken_tool, monkeypatch):
        # Strict per-call interleaving (start, result, start, error, ...) is a
        # property of the serial dispatch path. The parallel path emits all
        # `start` events before the pool runs and all `result`/`error` events
        # after it finishes, so this assertion only holds when parallel
        # dispatch is disabled. Parallel-path event semantics are exercised
        # in tests/unit/test_parallel_retrieval.py.
        monkeypatch.setenv("ENTRY_PARALLEL_RETRIEVAL", "0")
        ex = ToolExecutor(tools={"calculator": calc_tool, "broken": broken_tool})
        rec = _Recorder()
        token = current_event_emitter.set(rec)
        try:
            ex.execute(_state(
                {"tool": "calculator", "args": "1+1"},
                {"tool": "broken", "args": "x"},
                {"tool": "calculator", "args": "2+2"},
            ))
        finally:
            current_event_emitter.reset(token)
        kinds = [e[0] for e in rec.events]
        assert kinds == [
            "tool_call_start", "tool_call_result",
            "tool_call_start", "tool_call_error",
            "tool_call_start", "tool_call_result",
        ]
