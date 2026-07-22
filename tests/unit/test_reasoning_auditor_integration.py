"""Integration: ReasoningAgent._maybe_audit short-circuits to a retry tool_call.

Exercises the actual gate logic in `ReasoningAgent.synthesize` — including the
auditor short-circuit path — with stub LLMs (no network calls). Asserts:

  - When counting_intent + data_analysis result with entity-count payload
    AND the auditor says retry, synthesis returns a single data_analysis
    tool_call instead of invoking the synthesis LLM.
  - When the auditor approves, normal synthesis proceeds (synthesis LLM is
    called).
  - When counting_intent is False, the auditor never fires.
  - When audit_rounds >= 1, the auditor never re-fires.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.reasoning_agent import ReasoningAgent


def _entity_envelope(count: int = 3) -> str:
    inner = json.dumps({
        "count": count,
        "matched_sample": [{"id": "the system-A"}],
        "near_miss_sample": [{"id": "the system-B", "personnel_involved": ["Doe"]}],
    })
    return json.dumps({"stdout": inner, "stderr": "", "returncode": 0})


class _StubResponse:
    def __init__(self, content: str = "", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _StubLLM:
    """Stub primary LLM — records invocations so tests can assert it was
    (or wasn't) called by the synthesize() path."""
    def __init__(self):
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return _StubResponse(content="A drafted answer.", tool_calls=[])

    def bind_tools(self, tools):
        return self  # Stub claims native binding succeeded


def _build_state(*, counting_intent: bool, tool_results, audit_rounds: int = 0):
    return {
        "original_query": "How many entries is Jane Doe part of?",
        "enhanced_query": "How many entries is Jane Doe part of?",
        "retrieved_docs": [],
        "memory_context": {},
        "tool_results": tool_results,
        "counting_intent": counting_intent,
        "audit_rounds": audit_rounds,
        "reasoning_trace": [],
        "correction_round": 0,
    }


@pytest.fixture
def actor_llm():
    return _StubLLM()


@pytest.fixture
def auditor_llm_retry():
    """Auditor that says retry."""
    llm = MagicMock()
    llm.invoke.return_value = _StubResponse(
        content='{"retry": true, "guidance": "widen filter to also match \'Doe\'"}',
    )
    return llm


@pytest.fixture
def auditor_llm_approve():
    """Auditor that approves the count."""
    llm = MagicMock()
    llm.invoke.return_value = _StubResponse(
        content='{"retry": false, "guidance": null}',
    )
    return llm


def _data_analysis_result(*, args=None) -> dict:
    return {
        "tool": "data_analysis",
        "args": args or {
            "data_source": "entry_logbook",
            "analysis_code": "df[df['personnel_involved'] == 'Jane Doe']",
        },
        "result": _entity_envelope(count=3),
        "error": False,
    }


# ── Auditor fires + short-circuits to retry ─────────────────────────────


def test_auditor_short_circuits_to_retry_tool_call(actor_llm, auditor_llm_retry):
    agent = ReasoningAgent(llm=actor_llm, critique_llm=actor_llm,
                           auditor_llm=auditor_llm_retry, tools={})
    state = _build_state(counting_intent=True, tool_results=[_data_analysis_result()])

    out = agent.synthesize(state)

    # Synthesis LLM must NOT have been called — auditor short-circuited
    assert actor_llm.invocations == [], "actor LLM should not be invoked on auditor retry"
    # Result must contain a single data_analysis retry call with auditor guidance
    assert len(out["tool_calls"]) == 1
    call = out["tool_calls"][0]
    assert call["tool"] == "data_analysis"
    assert call["args"]["data_source"] == "entry_logbook"
    assert "AUDITOR FEEDBACK" in call["args"]["analysis_code"]
    assert "Doe" in call["args"]["analysis_code"].lower()
    assert out["audit_rounds"] == 1
    # Trace recorded the retry
    assert any("Auditor" in t for t in out["reasoning_trace"])


# ── Auditor approves → normal synthesis ─────────────────────────────────


def test_auditor_approves_falls_through_to_synthesis(actor_llm, auditor_llm_approve):
    agent = ReasoningAgent(llm=actor_llm, critique_llm=actor_llm,
                           auditor_llm=auditor_llm_approve, tools={})
    state = _build_state(counting_intent=True, tool_results=[_data_analysis_result()])

    out = agent.synthesize(state)

    # Auditor invoked, then synthesis LLM invoked normally
    auditor_llm_approve.invoke.assert_called_once()
    assert len(actor_llm.invocations) == 1
    assert out["draft_answer"] == "A drafted answer."
    # No new data_analysis retry emitted
    assert out["tool_calls"] == []
    # No audit_rounds bump on approval
    assert "audit_rounds" not in out or out.get("audit_rounds") == 0


# ── Auditor never fires when counting_intent=False ──────────────────────


def test_auditor_skipped_when_not_counting(actor_llm, auditor_llm_retry):
    agent = ReasoningAgent(llm=actor_llm, critique_llm=actor_llm,
                           auditor_llm=auditor_llm_retry, tools={})
    state = _build_state(counting_intent=False, tool_results=[_data_analysis_result()])

    out = agent.synthesize(state)

    auditor_llm_retry.invoke.assert_not_called()
    assert len(actor_llm.invocations) == 1, "actor LLM must run normally"
    assert out["draft_answer"] == "A drafted answer."


# ── Retry budget exhausted ──────────────────────────────────────────────


def test_auditor_skipped_when_audit_rounds_exhausted(actor_llm, auditor_llm_retry):
    agent = ReasoningAgent(llm=actor_llm, critique_llm=actor_llm,
                           auditor_llm=auditor_llm_retry, tools={})
    state = _build_state(
        counting_intent=True,
        tool_results=[_data_analysis_result()],
        audit_rounds=1,  # already retried once
    )

    out = agent.synthesize(state)

    auditor_llm_retry.invoke.assert_not_called()
    assert len(actor_llm.invocations) == 1
    assert out["draft_answer"] == "A drafted answer."
