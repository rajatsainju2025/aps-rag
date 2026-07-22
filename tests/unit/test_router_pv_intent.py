"""
Unit tests for PV query routing.
Covers: pv_query intent mapping and router dispatch.
"""
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.router import _INTENT_TO_SERVICE, _infer_service_mode


class TestPVQueryIntentMapping:
    def test_pv_query_in_intent_map(self):
        assert "pv_query" in _INTENT_TO_SERVICE

    def test_pv_query_routes_to_agentic_rag(self):
        assert _INTENT_TO_SERVICE["pv_query"] == "agentic_rag"

    def test_infer_service_mode_pv_query(self):
        assert _infer_service_mode("pv_query") == "agentic_rag"
