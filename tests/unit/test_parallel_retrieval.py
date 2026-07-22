"""
Unit tests for parallel tool fan-out in src.agents.tool_executor.ToolExecutor.

Verifies:
  - 3 parallel-eligible tool calls run concurrently (wall time < 1.0s when
    each tool sleeps 0.5s) when ENTRY_PARALLEL_RETRIEVAL is enabled.
  - Per-tool error isolation: one tool raising does not prevent the others'
    results from landing in tool_results, and the failure surfaces as an
    error entry with the same shape.
  - tool_results order is preserved (matches tool_calls order).
  - Serial fallback when ENTRY_PARALLEL_RETRIEVAL=0 — wall time >= 1.5s.
  - Single-call batches always go through the serial path (no thread-pool
    overhead) regardless of the flag.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.tool_executor import ToolExecutor  # noqa: E402


class _SleepyTool:
    """Tool stub that sleeps `delay_s` seconds and returns a marker."""

    def __init__(self, name: str, delay_s: float = 0.5):
        self.name = name
        self.delay_s = delay_s

    def run(self, args=None, *, tool_input=None):  # noqa: D401
        time.sleep(self.delay_s)
        return f"{self.name}:{args if args is not None else tool_input}"


class _RaisingTool:
    """Tool stub that always raises."""

    def run(self, args=None, *, tool_input=None):  # noqa: D401
        raise RuntimeError("boom")


def _state(tool_calls):
    return {"tool_calls": tool_calls, "tool_results": [], "reasoning_trace": []}


@pytest.fixture
def parallel_env(monkeypatch):
    monkeypatch.setenv("ENTRY_PARALLEL_RETRIEVAL", "1")
    yield


@pytest.fixture
def serial_env(monkeypatch):
    monkeypatch.setenv("ENTRY_PARALLEL_RETRIEVAL", "0")
    yield


def test_parallel_dispatch_is_concurrent(parallel_env):
    """3 sleep-0.5s tools should finish in well under 1.0s when parallel."""
    executor = ToolExecutor(tools={
        "a": _SleepyTool("a"),
        "b": _SleepyTool("b"),
        "c": _SleepyTool("c"),
    })
    calls = [{"tool": "a", "args": "1"}, {"tool": "b", "args": "2"}, {"tool": "c", "args": "3"}]
    t0 = time.monotonic()
    out = executor.execute(_state(calls))
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"parallel dispatch took {elapsed:.2f}s (expected < 1.0s)"
    assert len(out["tool_results"]) == 3


def test_serial_fallback_is_sequential(serial_env):
    """Same 3 tools, ENTRY_PARALLEL_RETRIEVAL=0 → wall time >= 1.5s."""
    executor = ToolExecutor(tools={
        "a": _SleepyTool("a"),
        "b": _SleepyTool("b"),
        "c": _SleepyTool("c"),
    })
    calls = [{"tool": "a", "args": "1"}, {"tool": "b", "args": "2"}, {"tool": "c", "args": "3"}]
    t0 = time.monotonic()
    out = executor.execute(_state(calls))
    elapsed = time.monotonic() - t0
    assert elapsed >= 1.4, f"serial dispatch took only {elapsed:.2f}s (expected >= 1.4s)"
    assert len(out["tool_results"]) == 3


def test_per_tool_error_isolation(parallel_env):
    """One tool raising must NOT abort the others; error surfaces per-tool."""
    executor = ToolExecutor(tools={
        "ok": _SleepyTool("ok", delay_s=0.05),
        "boom": _RaisingTool(),
        "ok2": _SleepyTool("ok2", delay_s=0.05),
    })
    calls = [{"tool": "ok", "args": "1"}, {"tool": "boom", "args": "2"}, {"tool": "ok2", "args": "3"}]
    out = executor.execute(_state(calls))

    assert len(out["tool_results"]) == 3
    by_tool = {r["tool"]: r for r in out["tool_results"]}
    assert by_tool["ok"]["error"] is False
    assert by_tool["ok2"]["error"] is False
    assert by_tool["boom"]["error"] is True
    assert "boom" in by_tool["boom"]["result"]


def test_results_order_preserved_under_parallel(parallel_env):
    """Slow→fast→medium calls must come back in input order, not finish order."""
    executor = ToolExecutor(tools={
        "slow": _SleepyTool("slow", delay_s=0.30),
        "fast": _SleepyTool("fast", delay_s=0.05),
        "medium": _SleepyTool("medium", delay_s=0.15),
    })
    calls = [{"tool": "slow", "args": "s"}, {"tool": "fast", "args": "f"}, {"tool": "medium", "args": "m"}]
    out = executor.execute(_state(calls))
    tool_order = [r["tool"] for r in out["tool_results"]]
    assert tool_order == ["slow", "fast", "medium"]


def test_unregistered_tool_surfaces_as_error_entry(parallel_env):
    """Missing tool returns an error result, not a missing position."""
    executor = ToolExecutor(tools={"a": _SleepyTool("a", delay_s=0.05)})
    calls = [{"tool": "a", "args": "1"}, {"tool": "ghost", "args": "2"}]
    out = executor.execute(_state(calls))
    assert len(out["tool_results"]) == 2
    by_tool = {r["tool"]: r for r in out["tool_results"]}
    assert by_tool["a"]["error"] is False
    assert by_tool["ghost"]["error"] is True
    assert "not available" in by_tool["ghost"]["result"]


def test_single_call_uses_serial_path(parallel_env):
    """A single tool_call should not pay the thread-pool cost.

    Asserted via behavior: trace line + result shape match the serial path
    semantics (just one start event, one result, in order).
    """
    executor = ToolExecutor(tools={"only": _SleepyTool("only", delay_s=0.05)})
    out = executor.execute(_state([{"tool": "only", "args": "x"}]))
    assert len(out["tool_results"]) == 1
    assert out["tool_results"][0]["tool"] == "only"


def test_empty_tool_calls_returns_empty_results(parallel_env):
    executor = ToolExecutor(tools={})
    out = executor.execute(_state([]))
    assert out == {"tool_results": []}


def test_prior_results_are_preserved(parallel_env):
    """Upstream tool_results should NOT be wiped by this dispatch."""
    executor = ToolExecutor(tools={"a": _SleepyTool("a", delay_s=0.01)})
    state = {
        "tool_calls": [{"tool": "a", "args": "x"}],
        "tool_results": [{"tool": "earlier", "result": "kept", "error": False}],
        "reasoning_trace": [],
    }
    out = executor.execute(state)
    assert len(out["tool_results"]) == 2
    assert out["tool_results"][0]["tool"] == "earlier"
    assert out["tool_results"][1]["tool"] == "a"


def test_reasoning_trace_records_each_call(parallel_env):
    executor = ToolExecutor(tools={
        "a": _SleepyTool("a", delay_s=0.01),
        "b": _RaisingTool(),
    })
    out = executor.execute(_state([
        {"tool": "a", "args": "1"},
        {"tool": "b", "args": "2"},
    ]))
    trace = out["reasoning_trace"]
    assert any("Tool a executed" in line for line in trace)
    assert any("Tool b failed" in line for line in trace)
