"""Unit tests for DataAnalysisTool — schema description, TODAY preamble, sandbox.

These tests cover the contract the LLM and verifier rely on:
  - The tool description includes a live schema cheatsheet (column names,
    value enums, date ranges) so the LLM doesn't hallucinate column names.
  - The subprocess preamble injects TODAY/THIS_YEAR/YESTERDAY computed from
    the server clock — fixing the "this year = 2024" date drift.
  - Sandbox safety still blocks os.system/subprocess/eval/exec patterns.
  - Round-trip: a typical counting analysis returns the right number.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tools.data_analysis import (  # noqa: E402
    DataAnalysisTool,
    WHITELISTED_DATA_FILES,
    _build_schema_summary,
    _build_tool_description,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wr_available() -> bool:
    p = WHITELISTED_DATA_FILES.get("work_requests")
    return bool(p and Path(p).exists())


_skip_no_wr = pytest.mark.skipif(
    not _wr_available(),
    reason="work_requests_processed.json not present on this host",
)


# ---------------------------------------------------------------------------
# Schema-aware description (Phase 1)
# ---------------------------------------------------------------------------
@_skip_no_wr
def test_schema_summary_lists_columns_and_value_enums():
    summary = _build_schema_summary("work_requests")
    assert "WRQ_NO" in summary
    assert "SUBMISSION_DATE" in summary
    assert "MACHINE" in summary
    # Value enums for low-cardinality columns
    assert "Storage Ring" in summary
    assert "Linac" in summary
    assert "COMPLETED" in summary  # STATUS enum
    # Date range for SUBMISSION_DATE (ISO 8601)
    assert "2015-" in summary  # earliest year
    assert "ISO 8601" in summary


@_skip_no_wr
def test_schema_summary_does_not_misflag_non_date_columns():
    """`created_by` (a username) was being mis-labelled as a date column
    because of the substring 'created' in the column name. Detector now
    requires the column's first non-null value to look like ISO YYYY-MM-DD."""
    entry_summary = _build_schema_summary("entry_logbook")
    if not entry_summary:  # entry data not present on this host
        pytest.skip("entry_logbook source not available")
    # Find the "Date columns" section and assert created_by isn't in it.
    if "Date columns" in entry_summary:
        date_section = entry_summary.split("Date columns")[1]
        # Stop at the next ### header if any
        date_section = date_section.split("###")[0]
        assert "created_by" not in date_section, (
            "created_by (a username field) was mis-flagged as a date column"
        )


@_skip_no_wr
def test_tool_description_documents_preamble_constants():
    """The LLM-visible description must list TODAY/THIS_YEAR/YESTERDAY."""
    desc = _build_tool_description()
    assert "TODAY" in desc
    assert "THIS_YEAR" in desc
    assert "YESTERDAY" in desc
    # And must mention the data analysis use case
    assert "counting" in desc.lower()


def test_tool_description_assigned_to_class():
    tool = DataAnalysisTool()
    assert tool.description, "DataAnalysisTool must expose a non-empty description"
    # Reasonable size — small enough to bound LLM context cost. Upper bound
    # bumped from 8000 → 12000 when defensive directive + per-column hints +
    # entity-count template were added (Phases 1-3a of the auditor work).
    assert 800 < len(tool.description) < 12000, (
        f"tool description length {len(tool.description)} outside expected band"
    )


# ---------------------------------------------------------------------------
# TODAY preamble injection (Phase 2)
# ---------------------------------------------------------------------------
@_skip_no_wr
def test_today_preamble_is_servers_current_date():
    tool = DataAnalysisTool()
    out = json.loads(tool._run(
        analysis_code="print(TODAY); print(THIS_YEAR); print(YESTERDAY)",
        data_source="work_requests",
    ))
    assert out["returncode"] == 0, f"stderr={out.get('stderr')!r}"
    lines = out["stdout"].strip().splitlines()
    today = date.today()
    assert lines[0] == today.isoformat()
    assert lines[1] == str(today.year)
    # YESTERDAY is TODAY - 1 day
    from datetime import timedelta
    assert lines[2] == (today - timedelta(days=1)).isoformat()


@_skip_no_wr
def test_today_value_passed_in_at_call_time(monkeypatch):
    """Confirm the date is computed per-call (not frozen at module import).

    Stage C note: ``DataAnalysisTool._run`` now dispatches over MCP to a fresh
    subprocess, where this monkeypatch wouldn't reach. We call the
    ``data_analysis_impl`` module function directly — that's the function the
    MCP server uses, so it's the right contract to test.
    """
    import src.tools.data_analysis as da

    fake = date(2030, 1, 15)

    class _FakeDate(date):
        @classmethod
        def today(cls):
            return fake

    monkeypatch.setattr(da, "date", _FakeDate)
    out = json.loads(da.data_analysis_impl(
        analysis_code="print(TODAY)",
        data_source="work_requests",
    ))
    assert out["returncode"] == 0
    assert out["stdout"].strip() == "2030-01-15"


# ---------------------------------------------------------------------------
# Sandbox safety (regression — already-existing block list)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden_snippet", [
    "import os; os.system('echo pwned')",
    "import subprocess; subprocess.run(['ls'])",
    "exec('print(1)')",
    "eval('1+1')",
    "__import__('os').system('echo pwned')",
])
def test_sandbox_blocks_forbidden_patterns(forbidden_snippet):
    tool = DataAnalysisTool()
    out = json.loads(tool._run(analysis_code=forbidden_snippet, data_source="work_requests"))
    # Either the safety check rejects (returncode 1 + 'Forbidden' error), or
    # the missing data file rejects first (returncode 1 + 'not found').
    # Both are acceptable failures; the must-not-happen case is returncode 0.
    assert out["returncode"] != 0


@_skip_no_wr
def test_sandbox_allows_pandas_string_with_forbidden_substring():
    """Make sure the regex strips comments and string literals so that a
    pandas filter mentioning the literal word 'subprocess' doesn't trip."""
    tool = DataAnalysisTool()
    out = json.loads(tool._run(
        analysis_code='print(df.shape[0])  # using pandas, not subprocess',
        data_source="work_requests",
    ))
    assert out["returncode"] == 0, f"stderr={out.get('stderr')!r}"


# ---------------------------------------------------------------------------
# Round-trip (the original failing query from RS's critique)
# ---------------------------------------------------------------------------
@_skip_no_wr
def test_kicker_count_round_trip():
    """The exact analysis the LLM would run for the user's failing query."""
    tool = DataAnalysisTool()
    code = (
        "mask = ("
        "    df['SUBMISSION_DATE'].str.startswith('2025') &"
        "    df['MACHINE'].isin(['Linac', 'PAR', 'Booster']) &"
        "    df['DESCRIPTION'].str.contains('kicker', case=False, na=False)"
        ")\n"
        "print(mask.sum())\n"
    )
    out = json.loads(tool._run(analysis_code=code, data_source="work_requests"))
    assert out["returncode"] == 0, f"stderr={out.get('stderr')!r}"
    n = int(out["stdout"].strip())
    # Tight assertion: this number is stable for any 2026 run because 2025 is closed.
    assert n == 31, (
        f"Expected 31 (canonical), got {n}. If data was refreshed, update the "
        f"test and counting_qa_groundtruth.yaml in tandem."
    )


@_skip_no_wr
def test_unknown_data_source_returns_error():
    """Stage C: unknown data_source surfaces as a structured MCP error envelope
    so the LLM can read .error/.hint and self-correct on the next ReAct
    iteration. The envelope shape replaces the pre-Stage-C
    ``{"returncode": 1, "error": ...}`` shape."""
    tool = DataAnalysisTool()
    out = json.loads(tool._run(analysis_code="print(1)", data_source="not_a_real_source"))
    assert out["status"] == "error"
    assert "Unknown data_source" in out["error"]
    assert out.get("tool") == "data_analysis"
    # The error response includes a hint pointing the LLM at valid sources
    assert "hint" in out
    assert "work_requests" in out["hint"]
