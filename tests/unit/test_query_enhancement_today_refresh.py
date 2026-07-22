"""
Regression test for the "stale today" bug in QueryEnhancer.

Background: TemporalDateParser (upstream in entry_rag_app) sets self.today =
datetime.now() once in __init__ and never refreshes. A long-running FastAPI
process would then resolve "yesterday" against the start-of-process date,
not the wall clock. QueryEnhancer.enhance() must rebind today before every
parse_query call so wall-clock drift can't shift the temporal window.
"""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class _StubDateParser:
    """Minimal TemporalDateParser stand-in that records today values per call."""

    def __init__(self):
        # Simulate the upstream class: today is set once at init.
        self.today = datetime(2000, 1, 1)
        self.today_str = "01/01/2000"
        self.seen_today_strs = []

    def parse_query(self, _query):
        # Record what today_str was when parse_query was called.
        self.seen_today_strs.append(self.today_str)
        return None  # No temporal intent — irrelevant to this test.

    def get_date_range_filter(self, _date_range):
        return {"metadata": {}}


class TestTodayRefresh:
    @pytest.fixture
    def enhancer_with_stub(self):
        from src.retrieval.query_enhancement import QueryEnhancer

        enh = QueryEnhancer()
        stub = _StubDateParser()
        enh._date_parser = stub
        return enh, stub

    def test_today_refreshed_per_call(self, enhancer_with_stub):
        """Two calls on different wall-clock dates must see different today_str."""
        enhancer, stub = enhancer_with_stub

        day_one = datetime(2026, 5, 25, 10, 0, 0)
        day_two = datetime(2026, 5, 27, 10, 0, 0)

        with patch(
            "src.retrieval.query_enhancement.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = day_one
            # Preserve other datetime attrs the module might touch.
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            enhancer.enhance("anything", generate_variants=False)

            mock_dt.now.return_value = day_two
            enhancer.enhance("anything else", generate_variants=False)

        assert stub.seen_today_strs == ["05/25/2026", "05/27/2026"], (
            f"Expected today to be refreshed per call, got {stub.seen_today_strs!r}. "
            "The QueryEnhancer is not rebinding self._date_parser.today before "
            "parse_query — see Fix 1a in the plan."
        )

    def test_today_not_frozen_at_init(self, enhancer_with_stub):
        """The init-time today (year 2000) must never leak into parse_query."""
        enhancer, stub = enhancer_with_stub

        with patch(
            "src.retrieval.query_enhancement.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 26, 12, 0, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            enhancer.enhance("yesterday's studies", generate_variants=False)

        assert stub.seen_today_strs == ["05/26/2026"]
        assert "2000" not in stub.seen_today_strs[0]
