"""Unit tests for src/services/pv_short_circuit.py."""
import json

import pytest

from src.services.pv_short_circuit import try_pv_short_circuit


class _FakeTool:
    """Minimal stand-in for PVReaderTool / PVSearchTool with controllable output."""

    def __init__(self, payload, raises=None):
        self.payload = payload
        self.raises = raises
        self.last_call = None

    def _run(self, **kwargs):
        self.last_call = kwargs
        if self.raises is not None:
            raise self.raises
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


def _components(pv_read=None, pv_search=None):
    tools = {}
    if pv_read is not None:
        tools["pv_read"] = pv_read
    if pv_search is not None:
        tools["pv_search"] = pv_search
    return {"tools": tools}


class TestNoMatch:
    def test_returns_none_for_non_pv_query(self):
        comps = _components(pv_read=_FakeTool({"readings": []}))
        assert try_pv_short_circuit("how do I align BPMs?", comps, "naive_rag", "sid") is None


class TestRouteA_LiteralPV:
    def test_single_pv_formatted_naturally(self):
        payload = {
            "readings": [{
                "pv": "S35A:H1:CurrentAO",
                "value": 12.345,
                "timestamp": "2026-05-26T14:30:21Z",
                "status": "OK",
                "units": "A",
            }],
            "count": 1,
        }
        tool = _FakeTool(payload)
        comps = _components(pv_read=tool)

        result = try_pv_short_circuit(
            "what is the current value of S35A:H1:CurrentAO?",
            comps, "naive_rag", "sid",
        )

        assert result is not None
        assert result["service_mode"] == "naive_rag"  # label preserved
        assert tool.last_call == {"pv_names": "S35A:H1:CurrentAO"}
        assert "S35A:H1:CurrentAO" in result["answer"]
        assert "12.345" in result["answer"]
        assert "2026-05-26T14:30:21Z" in result["answer"]
        # Reasoning trace surfaces the bypass + tool name
        trace = result["reasoning_trace"]
        assert any("PV intent detected" in t for t in trace)
        assert any("bypassed retrieval" in t for t in trace)
        assert any("pv_read" in t for t in trace)
        assert any("1 reading" in t for t in trace)
        # Empty retrieval fields so the UI doesn't show stale sources
        assert result["citations"] == []
        assert result["openwebui_citations"] == []
        assert result["sources"] == []

    def test_multi_pv_bulleted(self):
        payload = {
            "readings": [
                {"pv": "S35A:H1:CurrentAO", "value": 1.0, "timestamp": "t1"},
                {"pv": "S35A:H2:CurrentAO", "value": 2.0, "timestamp": "t2"},
            ],
            "count": 2,
        }
        comps = _components(pv_read=_FakeTool(payload))
        result = try_pv_short_circuit(
            "S35A:H1:CurrentAO and S35A:H2:CurrentAO",
            comps, "agentic_rag", "sid",
        )
        assert result is not None
        assert "Live values for 2 PV(s)" in result["answer"]
        assert "- " in result["answer"]  # bulleted

    def test_tool_unavailable_clear_error(self):
        # No pv_read tool registered at all.
        result = try_pv_short_circuit(
            "what is the current value of S35A:H1:CurrentAO?",
            {"tools": {}}, "naive_rag", "sid",
        )
        assert result is not None
        assert "unavailable" in result["answer"].lower()
        assert "pv_read" in result["answer"]
        # Reasoning trace must record the bypass + abort, NOT silently fall through
        trace = result["reasoning_trace"]
        assert any("unavailable" in t.lower() for t in trace)
        assert any("aborted" in t.lower() for t in trace)

    def test_tool_returns_error_surfaces_to_user(self):
        comps = _components(pv_read=_FakeTool({"error": "PV not found", "readings": []}))
        result = try_pv_short_circuit(
            "what is the current value of S35A:H1:CurrentAO?",
            comps, "naive_rag", "sid",
        )
        assert result is not None
        assert "Could not read" in result["answer"]
        assert "PV not found" in result["answer"]

    def test_tool_raises_exception(self):
        comps = _components(pv_read=_FakeTool({}, raises=TimeoutError("CA timeout")))
        result = try_pv_short_circuit(
            "S35A:H1:CurrentAO",
            comps, "naive_rag", "sid",
        )
        assert result is not None
        assert "failed" in result["answer"].lower()
        assert "CA timeout" in result["answer"]


class TestRouteB_DescriptionPhrase:
    def test_description_routes_to_pv_search(self):
        payload = {
            "readings": [{"pv": "S-DCCT:CurrentM", "value": 102.3, "timestamp": "t"}],
            "resolved_pvs": [{"pv": "S-DCCT:CurrentM"}],
            "count": 1,
        }
        tool = _FakeTool(payload)
        comps = _components(pv_search=tool)

        result = try_pv_short_circuit(
            "what is the beam current right now?",
            comps, "naive_rag", "sid",
        )
        assert result is not None
        assert tool.last_call == {"query": "what is the beam current right now?"}
        assert "S-DCCT:CurrentM" in result["answer"]
        assert "102.3" in result["answer"]
        # Provenance line from pv_search
        assert "Resolved from query" in result["answer"]

    def test_pv_search_dict_shaped_readings(self):
        # pv_search returns readings as a DICT keyed by PV name (not a list).
        # This is the real shape observed from MCP `controls/pv_search`.
        payload = {
            "resolved_pvs": [
                "LTS:CM1:beamCurrentAO",
                "LTS:H1:CurrentAO",
                "LTS:H1:CurrentAI",
            ],
            "readings": {
                "LTS:CM1:beamCurrentAO": {"value": 0.0},
                "LTS:H1:CurrentAO": {"value": 0.0},
                "LTS:H1:CurrentAI": {"value": 0.0003814755474174092},
            },
            "errors": None,
        }
        comps = _components(pv_search=_FakeTool(payload))
        result = try_pv_short_circuit(
            "what is the beam current right now?",
            comps, "naive_rag", "sid",
        )
        assert result is not None
        # Must not crash, must surface all three PVs
        for pv in payload["resolved_pvs"]:
            assert pv in result["answer"]
        assert "Live values for 3 PV(s)" in result["answer"]
        assert "Resolved from query" in result["answer"]
        # Reasoning trace should reflect successful readings, not error
        assert any("3 reading(s)" in t for t in result["reasoning_trace"])

    def test_pv_search_per_pv_errors(self):
        # pv_search may return per-PV errors in the "errors" dict.
        payload = {
            "resolved_pvs": ["GOOD:PV", "BAD:PV"],
            "readings": {"GOOD:PV": {"value": 12.3}},
            "errors": {"BAD:PV": "timeout"},
        }
        comps = _components(pv_search=_FakeTool(payload))
        result = try_pv_short_circuit(
            "what is the live value of beam current",
            comps, "naive_rag", "sid",
        )
        assert result is not None
        assert "GOOD:PV" in result["answer"]
        assert "12.3" in result["answer"]
        assert "BAD:PV" in result["answer"]  # surfaced in errors footer

    def test_historical_query_falls_through(self):
        # Even though "current value of" is in the query, the historical
        # date suppresses Route B → falls through (returns None).
        comps = _components(pv_search=_FakeTool({"readings": []}))
        result = try_pv_short_circuit(
            "current value of beam current on 2026-05-20",
            comps, "naive_rag", "sid",
        )
        assert result is None
