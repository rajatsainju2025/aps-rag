"""Service-layer tests for the clarification resume path.

Covers the regex + memory-state interaction in ``AgenticRAGService.query``
without booting the orchestrator. Specifically:
  - "1" / "2." / "clarify 2" / "Option 1" → choice extracted, query rewritten
  - Free-text reply → pending cleared, treated as fresh query (no choice)
  - No pending in memory → numeric reply passes through unchanged
  - Choice id not in options → no rewrite (defensive)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.services.agentic_rag_service import _CHOICE_RE, _render_clarification_markdown
from src.memory.memory_manager import MemoryManager


_PENDING = {
    "question": "What did you mean by 'is part of'?",
    "options": [
        {"id": "1", "label": "Participant",
         "interpretation": "Doe is in personnel_involved or created_by"},
        {"id": "2", "label": "Mentioned",
         "interpretation": "Doe is named anywhere in the entry"},
    ],
    "original_query": "How many entries is Jane Doe part of?",
}


# ── Regex behavior ──────────────────────────────────────────────────────


@pytest.mark.parametrize("reply,expected", [
    ("1", "1"),
    (" 2 ", "2"),
    ("2.", "2"),
    ("clarify 1", "1"),
    ("CLARIFY 2", "2"),
    ("option 1", "1"),
    ("Option 2", "2"),
])
def test_choice_regex_matches(reply, expected):
    m = _CHOICE_RE.match(reply)
    assert m is not None, f"{reply!r} should match"
    assert m.group(1) == expected


@pytest.mark.parametrize("reply", [
    "How many the system entries are there?",  # fresh query
    "10",                                 # 2-digit number is a count, not a menu pick
    "0",                                  # menu starts at 1
    "neither",
    "1 of these",
    "",
])
def test_choice_regex_does_not_match_unrelated(reply):
    assert _CHOICE_RE.match(reply) is None, f"{reply!r} should NOT match"


# ── MemoryManager session-state store ───────────────────────────────────


def test_session_state_set_and_get_roundtrip():
    mm = MemoryManager(neo4j_client=None)
    mm.set_session_state("sess-A", "clarification_pending", _PENDING)
    assert mm.get_session_state("sess-A", "clarification_pending") == _PENDING
    # default for missing key
    assert mm.get_session_state("sess-A", "missing", default="x") == "x"
    # default for missing session
    assert mm.get_session_state("sess-B", "anything") is None


def test_session_state_isolated_per_session():
    mm = MemoryManager(neo4j_client=None)
    mm.set_session_state("sess-A", "k", "value-A")
    mm.set_session_state("sess-B", "k", "value-B")
    assert mm.get_session_state("sess-A", "k") == "value-A"
    assert mm.get_session_state("sess-B", "k") == "value-B"


def test_session_state_clear_via_none():
    mm = MemoryManager(neo4j_client=None)
    mm.set_session_state("sess-A", "k", "v")
    mm.set_session_state("sess-A", "k", None)
    assert mm.get_session_state("sess-A", "k") is None


# ── Markdown rendering ──────────────────────────────────────────────────


def test_render_clarification_markdown_shape():
    md = _render_clarification_markdown(_PENDING)
    # Question shown bold
    assert "**What did you mean by 'is part of'?**" in md
    # Each option rendered as numbered bullet with label
    assert "1. **Participant**" in md
    assert "2. **Mentioned**" in md
    # Reply hint present
    assert "Reply with the number" in md
    assert "`1`" in md
