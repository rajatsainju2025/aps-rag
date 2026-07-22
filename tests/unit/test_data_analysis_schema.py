"""Schema-summary regression tests for data_analysis tool.

The Doe bug came from `personnel_involved` (list-valued) silently
dropping out of the LLM-visible schema because `Series.unique()` raises
TypeError on list cells. These tests pin the new behaviour:
  - List-valued columns appear with the apply+any idiom hint.
  - High-cardinality free-text columns get a substring directive.
  - Low-cardinality enums still list values.
  - The defensive directive + entity-count template are present in the
    tool description.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Skip module if pandas is unavailable (environments without scientific deps)
pd = pytest.importorskip("pandas")

from src.tools.data_analysis import DataAnalysisTool, _build_schema_summary


@pytest.fixture(scope="module")
def description() -> str:
    return DataAnalysisTool().description


@pytest.fixture(scope="module")
def entry_summary() -> str:
    return _build_schema_summary("entry_logbook")


@pytest.fixture(scope="module")
def wr_summary() -> str:
    return _build_schema_summary("work_requests")


# ── Tool description (Phase 1 + 3a) ─────────────────────────────────────


class TestToolDescription:
    def test_defensive_idiom_present(self, description):
        assert "Defensive filter idiom" in description
        assert "NEVER use exact equality" in description
        assert "case=False, na=False" in description

    def test_list_valued_apply_idiom_documented(self, description):
        assert "apply(lambda xs:" in description
        assert "any(" in description

    def test_entity_count_template_present(self, description):
        assert "entity-count queries" in description
        assert "matched_sample" in description
        assert "near_miss_sample" in description
        assert "json.dumps" in description


# ── Schema enrichment (Phase 2) ─────────────────────────────────────────


class TestDatasetSchemaHints:
    """Dataset schema-hint coverage for list-valued personnel fields."""

    def test_personnel_involved_flagged_list_valued(self, entry_summary):
        if not entry_summary:
            pytest.skip("entry_logbook source not available in this env")
        # The previously-missing column must now appear with the correct hint.
        assert "personnel_involved" in entry_summary
        assert "list-valued" in entry_summary
        assert "apply(lambda xs:" in entry_summary

    def test_keywords_flagged_list_valued(self, entry_summary):
        if not entry_summary:
            pytest.skip("entry_logbook source not available in this env")
        assert "`keywords` (list-valued)" in entry_summary

    def test_high_card_columns_get_substring_directive(self, entry_summary):
        if not entry_summary:
            pytest.skip("entry_logbook source not available in this env")
        # Free-text columns should carry the .str.contains hint
        assert "str.contains" in entry_summary
        assert "Do NOT exact-match" in entry_summary

    def test_list_hints_appear_before_high_card_hints(self, entry_summary):
        """Truncation should drop the LEAST critical info (high-card lines)
        first, so list-valued hints must appear earlier in the output."""
        if not entry_summary:
            pytest.skip("entry_logbook source not available in this env")
        idx_list = entry_summary.find("(list-valued)")
        idx_high = entry_summary.find("high-cardinality free text")
        assert idx_list != -1
        assert idx_high != -1
        assert idx_list < idx_high


class TestWorkRequestsSchemaHints:
    """Work requests have low-cardinality enums (STATUS, MACHINE) — those
    must continue to enumerate actual values, not get the high-card hint."""

    def test_low_card_enum_lists_values(self, wr_summary):
        if not wr_summary:
            pytest.skip("work_requests source not available in this env")
        # STATUS has 7 distinct values — should still be enumerated
        assert "`STATUS` values" in wr_summary
        assert "COMPLETED" in wr_summary

    def test_machine_low_card_lists_values(self, wr_summary):
        if not wr_summary:
            pytest.skip("work_requests source not available in this env")
        assert "`MACHINE` values" in wr_summary
        assert "Storage Ring" in wr_summary
