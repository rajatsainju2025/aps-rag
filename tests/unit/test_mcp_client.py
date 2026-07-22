"""Unit tests for MCPClientManager — config loading + sync wrapper.

The actual stdio MCP session is mocked via patch — these tests verify
the client's contract surface, not the upstream MCP library behavior.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tools.mcp_client import MCPClientManager


@pytest.fixture
def mcp_config(tmp_path):
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({
        "mcpServers": {
            "controls": {
                "command": "python",
                "args": ["-m", "mcp_servers.controls"],
                "env": {"EPICS_CA_ADDR_LIST": "host:5064"},
            },
            "archiver": {
                "command": "python",
                "args": ["-m", "mcp_servers.archiver"],
            },
        }
    }))
    return str(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────
class TestConfigLoading:
    def test_loads_valid_config(self, mcp_config):
        mgr = MCPClientManager(mcp_config)
        assert mgr.is_available
        assert mgr._config is not None
        assert "controls" in mgr._config["mcpServers"]
        assert "archiver" in mgr._config["mcpServers"]

    def test_missing_file_marks_unavailable(self, tmp_path):
        mgr = MCPClientManager(str(tmp_path / "does-not-exist.json"))
        assert mgr.is_available is False


# ─────────────────────────────────────────────────────────────────────────────
# Server lookup
# ─────────────────────────────────────────────────────────────────────────────
class TestServerLookup:
    def test_get_server_config_returns_dict(self, mcp_config):
        mgr = MCPClientManager(mcp_config)
        cfg = mgr._get_server_config("controls")
        assert cfg["command"] == "python"
        assert cfg["args"] == ["-m", "mcp_servers.controls"]

    def test_unknown_server_raises_keyerror(self, mcp_config):
        mgr = MCPClientManager(mcp_config)
        with pytest.raises(KeyError):
            mgr._get_server_config("ghost")


# ─────────────────────────────────────────────────────────────────────────────
# call_tool_sync — runs the async path inside a thread
# ─────────────────────────────────────────────────────────────────────────────
class TestCallToolSync:
    def test_dispatches_through_invoke_once(self, mcp_config):
        """Stage D: call_tool_sync routes through ``_invoke_once`` (which goes
        through the persistent session pool in production). Patching
        ``_invoke_once`` is the documented test seam for both the retry layer
        and the session pool."""
        mgr = MCPClientManager(mcp_config)

        def _fake_call(server, tool, args):
            return f"<result for {server}/{tool} with {args}>"

        with patch.object(MCPClientManager, "_invoke_once", side_effect=_fake_call):
            out = mgr.call_tool_sync("controls", "channel_find", {"q": "current"})
        assert "controls/channel_find" in out
        assert "{'q': 'current'}" in out


# ─────────────────────────────────────────────────────────────────────────────
# Env var substitution shape
# ─────────────────────────────────────────────────────────────────────────────
class TestEnvSubstitution:
    def test_static_env_value_passed_through(self, mcp_config):
        """Env vars from the config are forwarded to the subprocess.

        We can't spawn a real subprocess here; we just verify the config
        carries the env mapping unchanged.
        """
        mgr = MCPClientManager(mcp_config)
        cfg = mgr._get_server_config("controls")
        assert cfg["env"]["EPICS_CA_ADDR_LIST"] == "host:5064"
