"""Stage D — persistent MCP session pool + image/math tool shims.

Covers the contract surface of the persistent session pool added in
[src/tools/mcp_client.py](../../src/tools/mcp_client.py):

  - ``_SessionPool`` reuses one ClientSession per server across N calls
    (proves no per-call subprocess respawn — the Stage D win).
  - ``call_tool_sync`` returns a structured error envelope on transport
    failures inside the pool (no raised exceptions cross the boundary).
  - On a transport-level failure, the dead session is torn down so the
    next attempt opens a fresh one (crash recovery).
  - ``shutdown_now`` / atexit integration drains sessions cleanly.

Plus the new BaseTool→MCP dispatch shims for ``image_search`` and
``math_reasoning`` (mirrors the Stage C parameterized shim test).

The session-pool tests use the **compute** MCP server because it has the
lightest startup cost (SymPy import only); they are skipped if FastMCP isn't
installed in the env.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


pytest_plugins: list[str] = []


def _has_fastmcp() -> bool:
    try:
        import mcp.server.fastmcp  # noqa: F401
        import mcp.client.stdio  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


_skip_no_mcp = pytest.mark.skipif(not _has_fastmcp(), reason="mcp library not installed")


# ---------------------------------------------------------------------------
# BaseTool → MCP dispatch shims (Stage D coverage that mirrors Stage C)
# ---------------------------------------------------------------------------
def _patched_client(monkeypatch, response: str = '{"status":"ok","result":"OK","error":null}'):
    mock_client = MagicMock()
    mock_client.call_tool_sync = MagicMock(return_value=response)
    monkeypatch.setattr("src.tools.mcp_client.get_default_client", lambda: mock_client)
    return mock_client


@pytest.mark.parametrize("tool_factory,run_kwargs,expected_server,expected_tool,expected_args", [
    (lambda: __import__("src.tools.image_search", fromlist=["ImageSearchTool"]).ImageSearchTool(),
     {"query": "beam profile plot", "top_k": 3},
     "image", "image_search", {"query": "beam profile plot", "top_k": 3}),
    (lambda: __import__("src.tools.math_reasoning", fromlist=["MathReasoningTool"]).MathReasoningTool(llm=None),
     {"problem": "compute synchrotron tune for V_RF=9 MV"},
     "math", "math_reasoning", {"problem": "compute synchrotron tune for V_RF=9 MV"}),
])
def test_basetool_shim_dispatches_to_correct_mcp_endpoint(
    monkeypatch, tool_factory, run_kwargs, expected_server, expected_tool, expected_args
):
    mock_client = _patched_client(monkeypatch)
    tool = tool_factory()
    tool._run(**run_kwargs)
    mock_client.call_tool_sync.assert_called_once()
    pos = mock_client.call_tool_sync.call_args.args
    assert pos[0] == expected_server
    assert pos[1] == expected_tool
    for k, v in expected_args.items():
        assert pos[2].get(k) == v


def test_math_reasoning_constructor_keeps_legacy_llm_kwarg():
    """``llm`` kwarg is no longer used (Stage D moves the LLM into the math
    MCP server) but is preserved for backward compatibility with the
    api/main.py wiring."""
    from src.tools.math_reasoning import MathReasoningTool
    sentinel = object()
    tool = MathReasoningTool(llm=sentinel)  # must not raise
    assert tool.llm is sentinel


def test_image_search_returns_unwrapped_envelope_on_ok(monkeypatch):
    """unwrap_mcp_envelope re-serializes complex dict results so the LLM sees
    structured JSON, not a Python repr."""
    from src.tools.image_search import ImageSearchTool
    payload = {"query": "x", "results_markdown": "![img](url)", "count": 1}
    mock_client = _patched_client(monkeypatch, response=json.dumps({
        "status": "ok", "result": payload, "error": None,
    }))
    tool = ImageSearchTool()
    out = tool._run(query="x", top_k=1)
    assert json.loads(out) == payload


def test_image_search_passes_through_error_envelope(monkeypatch):
    """On status=error, the full envelope is returned so the LLM can read
    the error message + hint and self-correct on the next ReAct iteration."""
    from src.tools.image_search import ImageSearchTool
    err_env = {
        "status": "error", "result": None,
        "error": "image index is still building — try again shortly",
        "tool": "image_search",
        "hint": "first call may need to wait ~tens of seconds",
    }
    _patched_client(monkeypatch, response=json.dumps(err_env))
    tool = ImageSearchTool()
    out = tool._run(query="x")
    parsed = json.loads(out)
    assert parsed["status"] == "error"
    assert "still building" in parsed["error"]
    assert "hint" in parsed


# ---------------------------------------------------------------------------
# _SessionPool unit tests (do not require FastMCP — uses async stubs)
# ---------------------------------------------------------------------------
class _StubAsyncCtx:
    """Async context manager that returns a fixed value on enter."""
    def __init__(self, value):
        self._value = value
    async def __aenter__(self):
        return self._value
    async def __aexit__(self, exc_type, exc, tb):
        return False


class _StubSession:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        # FastMCP CallToolResult shape: .content is a list of blocks with .text
        block = type("B", (), {"text": json.dumps({"status": "ok", "result": f"{name}({args})", "error": None})})()
        return type("R", (), {"content": [block]})()


def _patch_mcp_stdio(monkeypatch, sessions: list[_StubSession]):
    """Replace stdio_client and ClientSession with stubs that hand out the
    pre-supplied list of sessions in order."""
    import src.tools.mcp_client as mc
    iter_sessions = iter(sessions)

    class _StubStdioClient:
        def __init__(self, params):
            self.params = params
        async def __aenter__(self):
            return (object(), object())
        async def __aexit__(self, *a):
            return False

    class _StubClientSession:
        def __init__(self, read, write):
            self._sess = next(iter_sessions)
        async def __aenter__(self):
            return self._sess
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr("mcp.client.stdio.stdio_client", lambda params: _StubStdioClient(params))
    monkeypatch.setattr("mcp.client.session.ClientSession", _StubClientSession)
    return mc


def _fresh_pool():
    """Build a fresh _SessionPool — never the singleton, so tests are isolated."""
    from src.tools.mcp_client import _SessionPool
    return _SessionPool()


def _make_stdio_params():
    """Build a dummy StdioServerParameters; the stubs ignore the contents."""
    from mcp.client.stdio import StdioServerParameters
    return StdioServerParameters(command="python", args=["-c", "pass"])


@_skip_no_mcp
def test_session_pool_reuses_one_connection_across_many_calls(monkeypatch):
    """The Stage D contract: N calls to the same server → ONE session opened.
    (Pre-Stage-D, every call would respawn the subprocess.)"""
    sess = _StubSession()
    _patch_mcp_stdio(monkeypatch, [sess])
    pool = _fresh_pool()
    try:
        params = _make_stdio_params()
        for _ in range(5):
            out = pool.call("compute", params, "calculator", {"expression": "2+2"}, timeout=10)
            assert "calculator" in out
        assert pool.connect_count("compute") == 1, (
            f"expected 1 connect, got {pool.connect_count('compute')}"
        )
        assert len(sess.calls) == 5
    finally:
        pool.shutdown_now()


@_skip_no_mcp
def test_session_pool_reopens_after_transport_error(monkeypatch):
    """When a session call raises BrokenPipeError, the pool must tear down
    the dead entry. The next call opens a FRESH session — proves crash
    recovery without a separate watchdog thread."""
    sess1 = _StubSession()
    sess2 = _StubSession()

    # Make sess1's call_tool raise BrokenPipeError; sess2's call works normally.
    async def _broken(*a, **kw):
        raise BrokenPipeError("simulated subprocess crash")
    sess1.call_tool = _broken  # type: ignore[assignment]

    _patch_mcp_stdio(monkeypatch, [sess1, sess2])
    pool = _fresh_pool()
    try:
        params = _make_stdio_params()
        # First call dies inside sess1 — the pool surfaces the BrokenPipeError
        with pytest.raises(BrokenPipeError):
            pool.call("compute", params, "calculator", {"expression": "1+1"}, timeout=10)
        # Sess1 should have been torn down; the live session is the new sess2.
        out = pool.call("compute", params, "calculator", {"expression": "1+1"}, timeout=10)
        assert "calculator" in out
        assert sess2.initialized, "sess2 was not initialized — pool didn't reopen"
        assert pool.connect_count("compute") == 1, (
            "after teardown the new task starts at connect_count=1 (its own counter)"
        )
    finally:
        pool.shutdown_now()


@_skip_no_mcp
def test_session_pool_isolates_sessions_per_server(monkeypatch):
    """Two distinct server names → two distinct sessions, both cached."""
    sess_a = _StubSession()
    sess_b = _StubSession()
    _patch_mcp_stdio(monkeypatch, [sess_a, sess_b])
    pool = _fresh_pool()
    try:
        params = _make_stdio_params()
        pool.call("compute", params, "calculator", {"expression": "1+1"}, timeout=10)
        pool.call("graph", params, "graph_search", {"query": "kicker"}, timeout=10)
        assert pool.session_count() == 2
        assert pool.connect_count("compute") == 1
        assert pool.connect_count("graph") == 1
    finally:
        pool.shutdown_now()


@_skip_no_mcp
def test_session_pool_shutdown_drains_sessions(monkeypatch):
    """``shutdown_now`` (and the atexit hook it shares) closes all live
    sessions and stops the background loop."""
    sess = _StubSession()
    _patch_mcp_stdio(monkeypatch, [sess])
    pool = _fresh_pool()
    params = _make_stdio_params()
    pool.call("compute", params, "calculator", {"expression": "1+1"}, timeout=10)
    assert pool.session_count() == 1
    pool.shutdown_now()
    assert pool.session_count() == 0


# ---------------------------------------------------------------------------
# call_tool_sync × persistent pool integration (mocked at _invoke_once)
# ---------------------------------------------------------------------------
def test_call_tool_sync_routes_through_invoke_once(monkeypatch):
    """``call_tool_sync`` must dispatch each attempt through ``_invoke_once``
    so monkey-patches at that seam (tests + production retry layer) keep
    working through the Stage D refactor."""
    from src.tools.mcp_client import MCPClientManager
    mgr = MCPClientManager.__new__(MCPClientManager)
    mgr._config = {"mcpServers": {"compute": {"command": "echo"}}}
    mgr._config_path = "/tmp/fake.json"
    mgr._available = True

    seen = []

    def _record(server, tool, args):
        seen.append((server, tool, args))
        return '{"status":"ok","result":"OK","error":null}'

    monkeypatch.setattr(mgr, "_invoke_once", _record)
    out = mgr.call_tool_sync("compute", "calculator", {"expression": "2+2"})
    assert json.loads(out) == {"status": "ok", "result": "OK", "error": None}
    assert seen == [("compute", "calculator", {"expression": "2+2"})]


def test_call_tool_sync_returns_envelope_on_pool_failure(monkeypatch):
    """If the persistent pool fails irrecoverably, ``call_tool_sync`` MUST
    return a structured error envelope (never a raised exception) so the
    LLM's ReAct loop sees an error message it can read."""
    from src.tools import mcp_client
    from src.tools.mcp_client import MCPClientManager
    mgr = MCPClientManager.__new__(MCPClientManager)
    mgr._config = {"mcpServers": {"compute": {"command": "echo"}}}
    mgr._config_path = "/tmp/fake.json"
    mgr._available = True

    def _always_pipe(*a, **kw):
        raise BrokenPipeError("pool says: subprocess gone")
    monkeypatch.setattr(mgr, "_invoke_once", _always_pipe)
    monkeypatch.setattr(mcp_client, "_MCP_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(mcp_client, "_MCP_RETRY_BASE_DELAY_S", 0.0)

    out = mgr.call_tool_sync("compute", "calculator", {"expression": "2+2"})
    parsed = json.loads(out)
    assert parsed["status"] == "error"
    assert "BrokenPipeError" in parsed["error"]
    assert parsed["server"] == "compute"
    assert parsed["tool"] == "calculator"
    assert parsed["attempts"] == 2


def test_session_pool_module_singleton_is_one_instance():
    """``get_session_pool`` always returns the SAME pool — so retries from
    different threads share session state."""
    from src.tools.mcp_client import get_session_pool
    assert get_session_pool() is get_session_pool()


# ---------------------------------------------------------------------------
# .mcp.json now lists math + image (drives auto-discovery in api/main.py)
# ---------------------------------------------------------------------------
def test_mcp_json_lists_all_six_servers():
    """Stage C+D end-state: 6 MCP servers (compute, controls, archiver,
    graph, math, image)."""
    cfg_path = Path(_ROOT) / "agentic-graph-rag" / ".mcp.json"
    if not cfg_path.exists():
        # When run from inside agentic-graph-rag/, _ROOT points there directly
        cfg_path = Path(_ROOT) / ".mcp.json"
    assert cfg_path.exists(), f".mcp.json not found at {cfg_path}"
    cfg = json.loads(cfg_path.read_text())
    servers = set(cfg["mcpServers"].keys())
    assert servers == {"compute", "controls", "archiver", "graph", "math", "image"}


def test_math_server_env_threads_argo_creds():
    """The math MCP server reads Argo creds from env (the .mcp.json must
    forward them via ${VAR:-default} substitution)."""
    cfg_path = Path(_ROOT) / ".mcp.json"
    cfg = json.loads(cfg_path.read_text())
    math_env = cfg["mcpServers"]["math"]["env"]
    assert "ARGO_USERNAME" in math_env
    assert "ARGO_BASE_URL" in math_env
    # Substitution syntax preserved (not eagerly resolved in JSON)
    assert math_env["ARGO_USERNAME"].startswith("${")
