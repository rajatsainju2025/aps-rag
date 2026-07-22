"""Unit tests for the /health/pv endpoint handler.

Bypasses FastAPI by calling the async route function directly with a patched
`api.main.components` module attribute. Verifies:
  - All four PV tools are reported, even if some are missing from tool_status
  - `status` is "ok" only when every PV tool is "ok"
  - Unknown / missing tools surface as {"status": "unknown"}
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _call_pv_health(tool_status: dict) -> dict:
    """Invoke the async pv_health handler with a synthetic components dict."""
    from api.routes.health import pv_health
    fake_components = {"tool_status": tool_status}
    fake_main = type(sys)("api.main")
    fake_main.components = fake_components
    with patch.dict(sys.modules, {"api.main": fake_main}):
        return asyncio.run(pv_health())


def test_all_pv_tools_ok_returns_overall_ok():
    status = {
        "channel_finder": {"status": "ok", "details": {"backend": "local_json"}},
        "pv_read":        {"status": "ok"},
        "archiver_read":  {"status": "ok"},
        "pv_search":      {"status": "ok"},
    }
    out = _call_pv_health(status)
    assert out["status"] == "ok"
    assert set(out["tools"].keys()) == {"channel_finder", "pv_read", "archiver_read", "pv_search"}
    assert out["tools"]["channel_finder"]["status"] == "ok"


def test_one_pv_tool_skipped_returns_degraded():
    status = {
        "channel_finder": {"status": "ok"},
        "pv_read":        {"status": "ok"},
        "archiver_read":  {"status": "skipped", "reason": "mcp_unreachable"},
        "pv_search":      {"status": "ok"},
    }
    out = _call_pv_health(status)
    assert out["status"] == "degraded"
    assert out["tools"]["archiver_read"]["status"] == "skipped"
    assert out["tools"]["archiver_read"]["reason"] == "mcp_unreachable"


def test_missing_pv_tool_surfaces_as_unknown():
    # Only two PV tools present in tool_status; the other two should default
    status = {
        "channel_finder": {"status": "ok"},
        "pv_read":        {"status": "ok"},
    }
    out = _call_pv_health(status)
    assert out["status"] == "degraded"
    assert out["tools"]["archiver_read"] == {"status": "unknown"}
    assert out["tools"]["pv_search"] == {"status": "unknown"}


def test_pv_tools_disabled_returns_skipped_for_all():
    status = {name: {"status": "skipped", "reason": "pv.enable_pv_tools=false"}
              for name in ("channel_finder", "pv_read", "archiver_read", "pv_search")}
    out = _call_pv_health(status)
    assert out["status"] == "degraded"
    for name in ("channel_finder", "pv_read", "archiver_read", "pv_search"):
        assert out["tools"][name]["status"] == "skipped"
        assert out["tools"][name]["reason"] == "pv.enable_pv_tools=false"


def test_empty_tool_status_returns_all_unknown():
    out = _call_pv_health({})
    assert out["status"] == "degraded"
    assert all(v == {"status": "unknown"} for v in out["tools"].values())
