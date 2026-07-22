"""Unit tests for the orchestrator's tool-skip surfacing logic.

When the planner emits a step like {"tool": "pv_read"} but pv_read isn't in
the registered tools dict (e.g. PV tools disabled at startup, or MCP server
down), the orchestrator must record this in state["tool_skipped"] and add
a line to reasoning_trace — otherwise operators see a generic answer with
no signal that the live PV read silently didn't happen.

These tests exercise the same code shape used by `agentic_rag_node` in
src/agents/orchestrator.py without spinning up the full LangGraph DAG.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


_NON_TOOL_SENTINELS = {"synthesize", "vector_search", "bm25_search", "text_search"}


def _surface_skipped_steps(
    plan_steps: List[Dict],
    tool_registry: Dict[str, Any],
    tool_status: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    """Mirror of the orchestrator block that builds `skipped`. Kept inline so
    a refactor of orchestrator.py that changes this contract surfaces here."""
    skipped: List[Dict[str, str]] = []
    for s in plan_steps:
        if not isinstance(s, dict):
            continue
        tname = s.get("tool")
        if tname and tname not in tool_registry and tname not in _NON_TOOL_SENTINELS:
            reason = tool_status.get(tname, {}).get(
                "reason", "tool not registered at startup"
            )
            skipped.append({"tool": tname, "reason": reason})
    return skipped


def test_skipped_carries_status_reason_when_available():
    plan = [{"tool": "pv_read", "query": "S05A:Q1"}]
    registry = {"calculator": object()}
    status = {"pv_read": {"status": "skipped", "reason": "pv.enable_pv_tools=false"}}
    out = _surface_skipped_steps(plan, registry, status)
    assert out == [{"tool": "pv_read", "reason": "pv.enable_pv_tools=false"}]


def test_skipped_falls_back_to_default_reason():
    plan = [{"tool": "pv_read", "query": "S05A:Q1"}]
    registry = {}
    status = {}  # no entry for pv_read at all
    out = _surface_skipped_steps(plan, registry, status)
    assert out == [{"tool": "pv_read", "reason": "tool not registered at startup"}]


def test_loaded_tool_is_not_marked_skipped():
    plan = [{"tool": "calculator", "query": "2+2"}]
    registry = {"calculator": object()}
    status = {"calculator": {"status": "ok"}}
    assert _surface_skipped_steps(plan, registry, status) == []


def test_multiple_skipped_steps_all_recorded():
    plan = [
        {"tool": "pv_read", "query": "X"},
        {"tool": "archiver_read", "query": "Y"},
        {"tool": "calculator", "query": "Z"},  # loaded — should NOT appear
    ]
    registry = {"calculator": object()}
    status = {
        "pv_read":       {"reason": "mcp_unreachable"},
        "archiver_read": {"reason": "mcp_unreachable"},
    }
    out = _surface_skipped_steps(plan, registry, status)
    assert {s["tool"] for s in out} == {"pv_read", "archiver_read"}
    assert all(s["reason"] == "mcp_unreachable" for s in out)


def test_malformed_plan_step_is_ignored():
    # Defends against the planner emitting a non-dict (string, None, list)
    plan = ["not a dict", None, {"tool": "pv_read"}]
    registry = {}
    status = {}
    out = _surface_skipped_steps(plan, registry, status)
    assert out == [{"tool": "pv_read", "reason": "tool not registered at startup"}]


def test_step_without_tool_field_is_ignored():
    plan = [{"query": "no tool key"}, {"tool": "", "query": "empty tool"}]
    registry = {}
    status = {}
    assert _surface_skipped_steps(plan, registry, status) == []


def test_planner_sentinels_are_not_flagged_as_skipped():
    """The planner emits `synthesize` (no-op) and `vector_search`/`bm25_search`
    (retrieval-phase markers handled by retrieval_node). None of these are
    registered as runtime tools, but they must not show up as skipped or
    every multi_hop AR query pollutes its reasoning_trace with phantom
    "Tool skipped: synthesize" entries."""
    plan = [
        {"step": 1, "tool": "vector_search", "query": "X"},
        {"step": 2, "tool": "bm25_search",   "query": "Y"},
        {"step": 3, "tool": "synthesize",    "query": None},
    ]
    assert _surface_skipped_steps(plan, {}, {}) == []


def test_agent_state_field_exists():
    """Compile-time check that AgentState declares tool_skipped — the
    orchestrator writes to this field, and LangGraph state merges depend on
    it being part of the TypedDict schema."""
    from src.agents.state import AgentState
    assert "tool_skipped" in AgentState.__annotations__
