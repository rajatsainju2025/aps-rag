"""Unit tests for src/agents/pv_intent.py."""
import pytest

from src.agents.pv_intent import classify_pv_intent


class TestRouteA_LiteralPV:
    """Route A — literal EPICS PV addresses → pv_read."""

    @pytest.mark.parametrize("query,expected_pvs", [
        ("what is the current value of S35A:H1:CurrentAO?", ["S35A:H1:CurrentAO"]),
        ("S-DCCT:CurrentM right now", ["S-DCCT:CurrentM"]),
        ("read PV LTS:H1:CurrentAO", ["LTS:H1:CurrentAO"]),
        ("show me BR:H1:SP and SR:H2:RBV", ["BR:H1:SP", "SR:H2:RBV"]),
    ])
    def test_literal_pv_addresses_match(self, query, expected_pvs):
        intent = classify_pv_intent(query)
        assert intent is not None, f"expected match for {query!r}"
        assert intent["route"] == "pv_read"
        assert intent["matched"] == expected_pvs
        assert intent["args"]["pv_names"] == ",".join(expected_pvs)
        assert intent["label"] == "literal PV address"

    def test_literal_pv_wins_over_phrase(self):
        # Both route triggers present; Route A must win (faster, more direct).
        intent = classify_pv_intent("current value of S35A:H1:CurrentAO")
        assert intent["route"] == "pv_read"


class TestRouteA_FalsePositives:
    """Route A must NOT match document namespaces, protocols, or timestamps."""

    @pytest.mark.parametrize("query", [
        "the the system:LOG entry from yesterday",
        "ICMS:DOC-1234 status",
        "HTTP:GET request failed",
        "see https://example.com/path",
        "TODO:CLEANUP in the orchestrator",
        "timestamp 2026-05-26T14:30:00 looks wrong",
        "iso date 2026-05-26 in the log",
        "how do I align the BPMs?",          # no colon at all
        "S-Bend power supply tripped",        # hyphen but no colon
    ])
    def test_excluded_patterns_do_not_match(self, query):
        intent = classify_pv_intent(query)
        # Either no match, or if matched it's via Route B (not Route A spurious)
        assert intent is None or intent["route"] != "pv_read"


class TestRouteB_DescriptionPhrase:
    """Route B — description-based phrases → pv_search."""

    @pytest.mark.parametrize("query", [
        "what is the beam current right now?",
        "current value of beam current",
        "live value of the H1 corrector",
        "real-time reading of the DCCT",
        "what is the storage ring current currently?",
        "read PV for the main bend magnets",
        "EPICS value of the beam position",
    ])
    def test_description_phrases_match(self, query):
        intent = classify_pv_intent(query)
        assert intent is not None, f"expected match for {query!r}"
        assert intent["route"] == "pv_search"
        assert intent["args"]["query"] == query.strip()
        assert intent["label"] == "description phrase"

    @pytest.mark.parametrize("query", [
        "what was the beam current last Thursday",
        "current value of beam current on 2026-05-20",
        "what was the H1 corrector reading yesterday",
        "live value of the DCCT in March 2025",
        "what was happening at the moment we lost beam?",  # "was" → historical
    ])
    def test_historical_queries_suppress_route_b(self, query):
        intent = classify_pv_intent(query)
        assert intent is None, (
            f"expected None for historical query {query!r}, got {intent!r}"
        )


class TestFallThrough:
    """Queries that should NOT trigger any PV intent."""

    @pytest.mark.parametrize("query", [
        "",
        "   ",
        "how do I align the BPMs?",
        "explain the difference between a quadrupole and a sextupole",
        "who is on shift tonight?",
        "summarize the last week of logbook entries",
    ])
    def test_no_match_falls_through(self, query):
        assert classify_pv_intent(query) is None


class TestEdgeCases:
    def test_too_many_pvs_capped(self):
        # 6 PV-shaped tokens; only first 5 should be returned.
        q = "read A:B C:D E:F G:H I:J K:L M:N right now"
        intent = classify_pv_intent(q)
        assert intent is not None
        assert intent["route"] == "pv_read"
        assert len(intent["matched"]) <= 5

    def test_duplicate_pvs_deduplicated(self):
        intent = classify_pv_intent("S35A:H1:CurrentAO and S35A:H1:CurrentAO again")
        assert intent is not None
        assert intent["matched"] == ["S35A:H1:CurrentAO"]
