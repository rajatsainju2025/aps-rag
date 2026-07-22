"""
Unit tests for src.utils.temporal_filter — specifically
build_temporal_context_block, which surfaces the temporal-expansion stage
to the LLM synthesis prompt so off-window docs aren't presented as in-range.
"""
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from src.utils.temporal_filter import build_temporal_context_block


class TestBuildTemporalContextBlock:
    def test_empty_when_range_missing(self):
        assert build_temporal_context_block(None, None) == ""
        assert build_temporal_context_block("", "") == ""
        assert build_temporal_context_block(None, "2026-05-25T00:00:00") == ""
        assert build_temporal_context_block("2026-05-25T00:00:00", None) == ""

    def test_strict_default_when_stage_omitted(self):
        block = build_temporal_context_block(
            "2026-05-25T00:00:00", "2026-05-25T23:59:59"
        )
        assert "2026-05-25 to 2026-05-25" in block
        assert "exact match" in block.lower()

    def test_strict_explicit(self):
        block = build_temporal_context_block(
            "2026-05-25T00:00:00", "2026-05-25T23:59:59", stage="strict"
        )
        assert "exact match" in block.lower()
        assert "widened" not in block.lower()
        assert "no documents matched" not in block.lower()

    def test_expanded_warns_off_window(self):
        block = build_temporal_context_block(
            "2026-05-25T00:00:00", "2026-05-25T23:59:59", stage="expanded"
        )
        # The block must announce the widening AND tell the LLM not to
        # present off-window docs as if they fell in-range.
        assert "30 days" in block
        assert "nearest" in block.lower()
        assert "do not present" in block.lower()
        assert "2026-05-25 to 2026-05-25" in block

    def test_unfiltered_states_no_records(self):
        block = build_temporal_context_block(
            "2026-05-25T00:00:00", "2026-05-25T23:59:59", stage="unfiltered"
        )
        assert "no documents matched" in block.lower()
        assert "no entries exist" in block.lower()

    def test_three_stages_produce_distinct_text(self):
        """strict / expanded / unfiltered must all read differently — the
        whole point of the helper is to make the difference visible to the LLM."""
        s = build_temporal_context_block("2026-05-25T00:00:00", "2026-05-25T23:59:59", "strict")
        e = build_temporal_context_block("2026-05-25T00:00:00", "2026-05-25T23:59:59", "expanded")
        u = build_temporal_context_block("2026-05-25T00:00:00", "2026-05-25T23:59:59", "unfiltered")
        assert len({s, e, u}) == 3, "Stages must produce distinct strings"

    def test_unknown_stage_falls_back_gracefully(self):
        """An unexpected stage value must still produce a usable block, not crash."""
        block = build_temporal_context_block(
            "2026-05-25T00:00:00", "2026-05-25T23:59:59", stage="something-new"
        )
        assert "2026-05-25" in block
