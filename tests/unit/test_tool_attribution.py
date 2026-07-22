"""
Unit tests for citation_formatter.replace_tool_attribution_markers.

Covers:
  - same-sentence entry ID → swap [tool: data_analysis] for clickable link
  - no nearby entry ID → render italic badge
  - sentence boundaries isolate the swap (later sentence's tool marker
    doesn't pick up an earlier sentence's entry ID)
  - loose prose forms ("Entry ID 10406", "WRQ 5021") match
  - canonical forms (ENTRY_10406, WRQ_5021) match
  - missing URL → bracketed display_id without link
  - empty/missing citations list → all markers become badges
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.citation_formatter import (  # noqa: E402
    _DATA_ANALYSIS_BADGE,
    replace_tool_attribution_markers,
)


def _entry(num: int, url: str = None) -> dict:
    return {
        "id": str(num),
        "display_id": f"ENTRY_{num}",
        "url": url or f"https://your-host.example.com/entry/views/itemDomainLogbook/view?id={num}",
        "index": 1,
    }


def _wrq(num: int, url: str = None) -> dict:
    return {
        "id": str(num),
        "display_id": f"WRQ_{num}",
        "url": url or f"http://localhost:8003/api/entries/work_requests/{num}.html",
        "index": 2,
    }


def test_no_tool_marker_passthrough():
    text = "Regular answer with [source_1] only."
    assert replace_tool_attribution_markers(text, [_entry(10406)]) == text


def test_loose_prose_id_swaps_to_link():
    text = '2026-03-24 — "[2026/03/24/1] BLS quenches," Entry ID 10406 [tool: data_analysis].'
    out = replace_tool_attribution_markers(text, [_entry(10406)])
    assert "[tool: data_analysis]" not in out
    assert "[ENTRY_10406](https://your-host.example.com/" in out


def test_canonical_id_swaps_to_link():
    text = "Entry ENTRY_10406 confirms the trip [tool: data_analysis]."
    out = replace_tool_attribution_markers(text, [_entry(10406)])
    assert "[ENTRY_10406](https://your-host.example.com/" in out
    assert "[tool: data_analysis]" not in out


def test_wrq_prose_form_matches():
    text = "Work request WRQ 5021 was created [tool: data_analysis]."
    out = replace_tool_attribution_markers(text, [_wrq(5021)])
    assert "[WRQ_5021](http://localhost:8003/api/entries/work_requests/5021.html)" in out


def test_no_nearby_id_renders_badge():
    text = "Three full recent entries are visible in the result [tool: data_analysis]."
    out = replace_tool_attribution_markers(text, [_entry(10406)])
    assert _DATA_ANALYSIS_BADGE in out
    assert "[tool: data_analysis]" not in out


def test_sentence_boundary_isolates_swap():
    text = (
        "Entry the system 10406 is the BLS quench case. "
        "Three more entries are partially truncated [tool: data_analysis]."
    )
    out = replace_tool_attribution_markers(text, [_entry(10406)])
    # The tool marker is in the SECOND sentence, which has no entry ID.
    # It should become the badge, not the ENTRY_10406 link.
    assert _DATA_ANALYSIS_BADGE in out
    assert "[ENTRY_10406]" not in out  # not pulled across sentence boundary


def test_two_sentences_each_get_correct_treatment():
    text = (
        "Entry ID 10406 shows the failure [tool: data_analysis]. "
        "Overall trends are stable [tool: data_analysis]."
    )
    out = replace_tool_attribution_markers(text, [_entry(10406)])
    # First sentence: link.  Second sentence: badge.
    assert "[ENTRY_10406](https://your-host.example.com/" in out
    assert _DATA_ANALYSIS_BADGE in out
    assert "[tool: data_analysis]" not in out


def test_missing_url_falls_back_to_bracketed_id():
    cite = {"id": "10406", "display_id": "ENTRY_10406", "url": "", "index": 1}
    text = "the system 10406 [tool: data_analysis]."
    out = replace_tool_attribution_markers(text, [cite])
    assert "[ENTRY_10406]" in out
    assert "[ENTRY_10406](" not in out  # no URL → no markdown link
    assert "[tool: data_analysis]" not in out


def test_empty_citations_all_become_badges():
    text = "First [tool: data_analysis]. Second the system 10406 [tool: data_analysis]."
    out = replace_tool_attribution_markers(text, [])
    assert out.count(_DATA_ANALYSIS_BADGE) == 2
    assert "[tool: data_analysis]" not in out


def test_unknown_id_in_sentence_falls_through_to_badge():
    text = "the system 99999 [tool: data_analysis]."  # 99999 not in citations
    out = replace_tool_attribution_markers(text, [_entry(10406)])
    assert _DATA_ANALYSIS_BADGE in out
    assert "[ENTRY_10406]" not in out  # don't borrow an unrelated citation


def test_case_insensitive_tool_marker():
    text = "the system 10406 [Tool: Data_Analysis]."
    out = replace_tool_attribution_markers(text, [_entry(10406)])
    assert "[ENTRY_10406](https://your-host.example.com/" in out


def test_marker_with_extra_whitespace():
    text = "the system 10406 [tool:  data_analysis ]."
    out = replace_tool_attribution_markers(text, [_entry(10406)])
    assert "[ENTRY_10406](https://your-host.example.com/" in out
