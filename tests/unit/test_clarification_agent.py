"""Unit tests for ClarificationAgent.

Covers the gate's three exit shapes:
  - Returns dict when LLM emits ambiguous=true with ≥2 valid options
  - Returns None when LLM emits ambiguous=false
  - Returns None (fail-open) on exception, malformed JSON, or invalid options
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.clarification_agent import ClarificationAgent


def _resp(content: str) -> MagicMock:
    return MagicMock(content=content)


# ── Ambiguous → returns dict ────────────────────────────────────────────


def test_returns_dict_when_llm_says_ambiguous():
    llm = MagicMock()
    llm.invoke.return_value = _resp(
        '{"ambiguous": true,'
        ' "question": "What did you mean by \\"is part of\\"?",'
        ' "options": ['
        '  {"id":"1","label":"Participant","interpretation":"in personnel_involved or created_by"},'
        '  {"id":"2","label":"Mentioned","interpretation":"named anywhere in the entry"}'
        ' ]}'
    )
    agent = ClarificationAgent(llm)
    out = agent.maybe_clarify("How many entries is Doe part of?")
    assert out is not None
    assert "is part of" in out["question"]
    assert len(out["options"]) == 2
    assert {o["id"] for o in out["options"]} == {"1", "2"}
    assert out["options"][0]["label"] == "Participant"


# ── Unambiguous → returns None ──────────────────────────────────────────


def test_returns_none_when_llm_says_unambiguous():
    llm = MagicMock()
    llm.invoke.return_value = _resp('{"ambiguous": false}')
    agent = ClarificationAgent(llm)
    assert agent.maybe_clarify("How many WRs for kicker repairs in 6 months?") is None


# ── Fail-open paths ─────────────────────────────────────────────────────


def test_fail_open_on_exception():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("LLM crashed")
    agent = ClarificationAgent(llm)
    assert agent.maybe_clarify("anything") is None


def test_fail_open_on_malformed_json():
    llm = MagicMock()
    llm.invoke.return_value = _resp("not json at all")
    agent = ClarificationAgent(llm)
    assert agent.maybe_clarify("anything") is None


def test_fail_open_on_missing_options_key():
    llm = MagicMock()
    llm.invoke.return_value = _resp('{"ambiguous": true, "question": "?"}')
    agent = ClarificationAgent(llm)
    assert agent.maybe_clarify("anything") is None


def test_fail_open_when_only_one_option():
    """A 'choice' with one option is not a real choice — fail open."""
    llm = MagicMock()
    llm.invoke.return_value = _resp(
        '{"ambiguous": true, "question": "?", "options":'
        ' [{"id":"1","label":"A","interpretation":"x"}]}'
    )
    agent = ClarificationAgent(llm)
    assert agent.maybe_clarify("anything") is None


def test_drops_malformed_options_and_fails_open_below_threshold():
    """Options missing required keys are dropped; if <2 remain, return None."""
    llm = MagicMock()
    llm.invoke.return_value = _resp(
        '{"ambiguous": true, "question": "?", "options": ['
        ' {"id":"1","label":"A","interpretation":"x"},'
        ' {"id":"2","label":"B"},'  # missing interpretation → dropped
        ' "not a dict"'
        ']}'
    )
    agent = ClarificationAgent(llm)
    # Only one valid option survives → not enough to ask
    assert agent.maybe_clarify("anything") is None


def test_normalizes_option_ids_to_strings():
    """LLM may emit numeric ids; coerce to strings for downstream regex match."""
    llm = MagicMock()
    llm.invoke.return_value = _resp(
        '{"ambiguous": true, "question": "?", "options": ['
        ' {"id":1,"label":"A","interpretation":"x"},'
        ' {"id":2,"label":"B","interpretation":"y"}'
        ']}'
    )
    agent = ClarificationAgent(llm)
    out = agent.maybe_clarify("anything")
    assert out is not None
    assert all(isinstance(o["id"], str) for o in out["options"])
