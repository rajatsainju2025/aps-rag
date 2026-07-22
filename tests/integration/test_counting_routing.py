"""Integration tests for counting-question routing.

These tests verify that a counting query like "How many work requests
entered in 2025 for kickers in Linac, PAR, or Booster?" gets routed
through the data_analysis tool path:

  1. is_counting_query() flags it
  2. PlannerAgent.plan() prepends a data_analysis plan step
  3. ReasoningAgent._format_counting_directive() emits the hard directive

We exercise the contract directly (no live LLM) so the test runs offline
and stays fast. End-to-end correctness over a live API is covered by the
counting_verifier harness.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.planner import PlannerAgent  # noqa: E402
from src.agents.reasoning_agent import ReasoningAgent  # noqa: E402
from src.retrieval.query_enhancement import is_counting_query  # noqa: E402


COUNTING_QUERY = (
    "How many work requests entered in year 2025 exist for kickers "
    "in the Linac, PAR, or Booster?"
)
NON_COUNTING_QUERY = "What is a kicker magnet and how does it work?"


# ---------------------------------------------------------------------------
# 1. Detection
# ---------------------------------------------------------------------------
def test_is_counting_query_flags_explicit_count():
    assert is_counting_query(COUNTING_QUERY) is True


def test_is_counting_query_does_not_flag_definition_query():
    assert is_counting_query(NON_COUNTING_QUERY) is False


def test_is_counting_query_handles_empty_query():
    assert is_counting_query("") is False
    assert is_counting_query(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Planner forces data_analysis at head of plan when counting_intent is set
# ---------------------------------------------------------------------------
def _make_mock_llm(response_text: str):
    """Mock LLM that returns a fixed plan JSON regardless of input."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=MagicMock(content=response_text))
    return llm


def test_planner_prepends_data_analysis_step_for_counting_intent():
    # LLM returns a plain retrieval plan with no data_analysis step
    plan_json = (
        '[{"step": 1, "tool": "vector_search", "query": "kickers 2025", '
        '"collections": ["work_requests"], "rationale": "find kicker WRs"}]'
    )
    planner = PlannerAgent(llm=_make_mock_llm(plan_json))
    state = {
        "original_query": COUNTING_QUERY,
        "enhanced_query": COUNTING_QUERY,
        "query_intent": "counting",
        "counting_intent": True,
        "selected_databases": ["work_requests"],
        "retrieval_strategy": "HYBRID_3WAY",
        "correction_round": 0,
        "extracted_entities": [],
        "correction_issues": [],
        "reasoning_trace": [],
    }
    result = planner.plan(state)
    plan_steps = result["plan_steps"]
    assert len(plan_steps) >= 1
    head = plan_steps[0]
    assert head.get("tool") == "data_analysis", (
        f"Expected data_analysis at head of plan, got: {plan_steps}"
    )
    assert head.get("data_source") == "work_requests"


def test_planner_does_not_prepend_when_counting_intent_is_false():
    plan_json = (
        '[{"step": 1, "tool": "vector_search", "query": "kicker", '
        '"collections": ["work_requests"], "rationale": ""}]'
    )
    planner = PlannerAgent(llm=_make_mock_llm(plan_json))
    state = {
        "original_query": NON_COUNTING_QUERY,
        "enhanced_query": NON_COUNTING_QUERY,
        "query_intent": "factual",
        "counting_intent": False,
        "selected_databases": ["work_requests"],
        "retrieval_strategy": "HYBRID_3WAY",
        "correction_round": 0,
        "extracted_entities": [],
        "correction_issues": [],
        "reasoning_trace": [],
    }
    result = planner.plan(state)
    plan_steps = result["plan_steps"]
    assert all(s.get("tool") != "data_analysis" for s in plan_steps), (
        f"Plan should not contain data_analysis when counting_intent is False: {plan_steps}"
    )


def test_planner_does_not_duplicate_data_analysis_when_already_present():
    plan_json = (
        '[{"step": 1, "tool": "data_analysis", "query": "count kickers", '
        '"collections": ["work_requests"], "rationale": "LLM picked tool itself"}]'
    )
    planner = PlannerAgent(llm=_make_mock_llm(plan_json))
    state = {
        "original_query": COUNTING_QUERY,
        "enhanced_query": COUNTING_QUERY,
        "query_intent": "counting",
        "counting_intent": True,
        "selected_databases": ["work_requests"],
        "retrieval_strategy": "HYBRID_3WAY",
        "correction_round": 0,
        "extracted_entities": [],
        "correction_issues": [],
        "reasoning_trace": [],
    }
    result = planner.plan(state)
    n_data_analysis = sum(
        1 for s in result["plan_steps"] if s.get("tool") == "data_analysis"
    )
    assert n_data_analysis == 1, (
        f"Should not duplicate data_analysis step, got {n_data_analysis}"
    )


def test_planner_skips_injection_when_no_recognized_collection():
    """If selected_databases is e.g. ['icms_collection'], data_analysis has
    no matching data_source so we don't inject — the synthesis-time directive
    still nudges the LLM."""
    plan_json = '[{"step": 1, "tool": "vector_search", "query": "X", "collections": ["icms_collection"], "rationale": ""}]'
    planner = PlannerAgent(llm=_make_mock_llm(plan_json))
    state = {
        "original_query": "How many ICMS docs mention safety?",
        "enhanced_query": "How many ICMS docs mention safety?",
        "query_intent": "counting",
        "counting_intent": True,
        "selected_databases": ["icms_collection"],
        "retrieval_strategy": "HYBRID_3WAY",
        "correction_round": 0,
        "extracted_entities": [],
        "correction_issues": [],
        "reasoning_trace": [],
    }
    result = planner.plan(state)
    assert all(s.get("tool") != "data_analysis" for s in result["plan_steps"])


# ---------------------------------------------------------------------------
# 3. ReasoningAgent emits the hard directive when counting_intent is set
# ---------------------------------------------------------------------------
def test_counting_directive_emitted_when_intent_set_and_no_results():
    """Stage B reworded the directive to use native function-calling vocabulary
    (still mandates calling the tool, just drops the legacy `TOOL_CALL:`
    syntax mandate). The test now asserts the persistent contract: counting
    intent triggers a directive that names the data_analysis tool and the
    correct data_source."""
    state = {"counting_intent": True, "selected_databases": ["work_requests"]}
    out = ReasoningAgent._format_counting_directive(state, [])
    assert "COUNTING / AGGREGATION" in out
    # Tool name must appear so the model knows which tool to call.
    assert "data_analysis" in out
    # Data source hint is what makes this query routable.
    assert 'data_source="work_requests"' in out
    # The directive must not encourage the legacy `TOOL_CALL:` text protocol.
    assert "TOOL_CALL:" not in out, (
        "Stage B drops the prompt-based TOOL_CALL: syntax; the directive "
        "should describe tool USE in vendor-neutral terms only."
    )


def test_counting_directive_silent_when_intent_not_set():
    state = {"counting_intent": False, "selected_databases": ["work_requests"]}
    assert ReasoningAgent._format_counting_directive(state, []) == ""


def test_counting_directive_silent_after_data_analysis_runs():
    """Once data_analysis returned a result, don't re-instruct (would cause
    the LLM to re-run the same analysis or get confused)."""
    state = {"counting_intent": True, "selected_databases": ["work_requests"]}
    tool_results = [{"tool": "data_analysis", "result": '{"stdout": "31"}'}]
    assert ReasoningAgent._format_counting_directive(state, tool_results) == ""


def test_counting_directive_data_source_inferred_from_databases():
    cases = [
        (["entry_logbook"], "entry_logbook"),
        (["teams_messages"], "teams_messages"),
        (["work_requests"], "work_requests"),
    ]
    for databases, expected_ds in cases:
        state = {"counting_intent": True, "selected_databases": databases}
        out = ReasoningAgent._format_counting_directive(state, [])
        assert f'data_source="{expected_ds}"' in out, (
            f"Expected {expected_ds!r} hint for {databases!r}, got: {out}"
        )


def test_counting_directive_falls_back_when_database_unknown():
    state = {"counting_intent": True, "selected_databases": ["icms_collection"]}
    out = ReasoningAgent._format_counting_directive(state, [])
    # Falls back to enumerating the available data sources
    assert "Choose `data_source`" in out
    assert "work_requests" in out  # must list available options


# ---------------------------------------------------------------------------
# 4. Query-text fallback (OpenWebUI sends selected_databases=["all_databases"])
# ---------------------------------------------------------------------------
def test_infer_data_source_uses_query_text_when_collection_is_all_databases():
    # OpenWebUI passes ["all_databases"] which doesn't match any specific
    # data source — fall back to keyword inspection of the query.
    assert PlannerAgent._infer_data_source(
        ["all_databases"],
        "How many work requests entered in 2025 for kickers?"
    ) == "work_requests"
    assert PlannerAgent._infer_data_source(
        ["all_databases"],
        "How many logbook entries mention vacuum leaks?"
    ) == "entry_logbook"
    # No keywords -> still None (the directive will then enumerate options)
    assert PlannerAgent._infer_data_source(
        ["all_databases"],
        "How many things are there?"
    ) is None


def test_directive_uses_query_text_to_resolve_data_source():
    state = {
        "counting_intent": True,
        "selected_databases": ["all_databases"],
        "enhanced_query": "How many work requests entered in 2025 for kickers?",
    }
    out = ReasoningAgent._format_counting_directive(state, [])
    assert 'data_source="work_requests"' in out, (
        "OpenWebUI all_databases path should resolve via query text"
    )


def test_directive_forbids_deferral_phrasing():
    """Specific guard against the 'Do you want me to run...' failure mode."""
    state = {"counting_intent": True, "selected_databases": ["work_requests"]}
    out = ReasoningAgent._format_counting_directive(state, [])
    assert "Do you want me to run" in out  # explicitly listed as forbidden
    assert "MANDATORY TOOL USE" in out
