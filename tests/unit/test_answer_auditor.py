"""Unit tests for AnswerAuditor.

Covers:
  - _extract_evidence_payload: dict-detection edge cases (no LLM call)
  - audit(): no-op fast paths and the LLM-driven retry path
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

from src.agents.answer_auditor import AnswerAuditor


def _envelope(stdout: str) -> str:
    """Wrap stdout in the standard data_analysis MCP envelope."""
    return json.dumps({"stdout": stdout, "stderr": "", "returncode": 0})


def _entity_dict(count: int = 3, matched=None, near_miss=None) -> dict:
    return {
        "count": count,
        "matched_sample": matched if matched is not None else [{"id": "the system-A"}],
        "near_miss_sample": near_miss if near_miss is not None else [{"id": "the system-B"}],
    }


# ── _extract_evidence_payload ───────────────────────────────────────────


class TestExtractEvidencePayload:
    def test_plain_number_no_op(self):
        assert AnswerAuditor._extract_evidence_payload(_envelope("42\n")) is None

    def test_dict_at_end_of_stdout(self):
        d = _entity_dict()
        out = _envelope(json.dumps(d))
        assert AnswerAuditor._extract_evidence_payload(out) == d

    def test_dict_after_intermediate_prints(self):
        d = _entity_dict(count=17)
        out = _envelope(f"Loading data...\nIntermediate count: 17\n{json.dumps(d)}\n")
        assert AnswerAuditor._extract_evidence_payload(out) == d

    def test_error_envelope_no_op(self):
        out = json.dumps({"error": "timeout", "stdout": "", "returncode": -1})
        assert AnswerAuditor._extract_evidence_payload(out) is None

    def test_raw_string_no_envelope(self):
        # Auditor should tolerate non-JSON tool output
        assert AnswerAuditor._extract_evidence_payload("count is 17, done") is None

    def test_dict_missing_required_key(self):
        out = _envelope(json.dumps({"count": 5, "matched_sample": []}))
        # Missing near_miss_sample → no-op
        assert AnswerAuditor._extract_evidence_payload(out) is None

    def test_dict_passed_directly_not_string(self):
        d = _entity_dict()
        # Caller may pass the parsed envelope dict instead of its JSON string
        outer = {"stdout": json.dumps(d), "stderr": "", "returncode": 0}
        assert AnswerAuditor._extract_evidence_payload(outer) == d


# ── audit() ─────────────────────────────────────────────────────────────


class TestAudit:
    def test_no_op_on_plain_number(self):
        llm = MagicMock()
        auditor = AnswerAuditor(llm)
        verdict = auditor.audit(query="how many?", code="print(len(df))",
                                tool_output=_envelope("42\n"))
        assert verdict == {"retry": False}
        # Auditor must NOT call the LLM when there's no entity-count payload
        llm.invoke.assert_not_called()

    def test_retry_when_llm_says_so(self):
        d = _entity_dict(count=3)
        llm = MagicMock()
        # parse_json_with_repair just returns parsed dict if valid JSON
        llm.invoke.return_value = MagicMock(
            content='{"retry": true, "guidance": "widen filter to also match \'Doe\'"}'
        )
        auditor = AnswerAuditor(llm)
        verdict = auditor.audit(
            query="How many entries is Jane Doe part of?",
            code="df[df['personnel_involved'] == 'Jane Doe']",
            tool_output=_envelope(json.dumps(d)),
        )
        assert verdict["retry"] is True
        assert "Doe" in verdict["guidance"].lower()
        llm.invoke.assert_called_once()

    def test_no_retry_when_llm_approves(self):
        d = _entity_dict(count=175)
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='{"retry": false, "guidance": null}')
        auditor = AnswerAuditor(llm)
        verdict = auditor.audit(query="how many?", code="...",
                                tool_output=_envelope(json.dumps(d)))
        assert verdict["retry"] is False

    def test_fail_open_on_llm_exception(self):
        d = _entity_dict()
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM crashed")
        auditor = AnswerAuditor(llm)
        verdict = auditor.audit(query="how many?", code="...",
                                tool_output=_envelope(json.dumps(d)))
        # Fail-open: never block synthesis on a broken auditor
        assert verdict == {"retry": False}

    def test_fail_open_on_malformed_llm_output(self):
        d = _entity_dict()
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="not json at all")
        auditor = AnswerAuditor(llm)
        verdict = auditor.audit(query="how many?", code="...",
                                tool_output=_envelope(json.dumps(d)))
        assert verdict == {"retry": False}
