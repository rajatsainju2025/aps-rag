"""Stage C unit tests — MCP server layer + tool shims.

Covers the new Stage C contracts:

  1. ``mcp_servers/_common.py`` — ``ok``/``err`` response shape, the
     ``mcp_tool_safe`` decorator catches exceptions into structured errors,
     and ``add_project_root_to_path`` wires up imports correctly.
  2. ``src/tools/mcp_client.py`` — env-var expansion (``${VAR:-default}``),
     retry classifier (transient errors retry; tool-level errors don't),
     ``unwrap_mcp_envelope`` legacy-shape preservation, and singleton.
  3. ``src/tools/*.py`` BaseTool shims — each ``_run`` calls
     ``get_default_client().call_tool_sync(<server>, <tool>, args)``.
     Mock the client; assert the call was dispatched correctly.
  4. ``mcp_servers/*.py`` — each server boots and exposes the expected
     tool names via ``list_tools``.

Live MCP-subprocess round-trips (the heavier integration test) are
deliberately scoped out of this file — they're covered by the live
verifier rerun in Stage C-10. These tests run offline in <2s.
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


# ---------------------------------------------------------------------------
# 1. _common: response shape + mcp_tool_safe decorator
# ---------------------------------------------------------------------------
def test_ok_envelope_shape():
    from mcp_servers._common import ok
    env = json.loads(ok(result=42, query="anything"))
    assert env == {"status": "ok", "result": 42, "error": None, "query": "anything"}


def test_err_envelope_shape():
    from mcp_servers._common import err
    env = json.loads(err("bad input", hint="try again"))
    assert env["status"] == "error"
    assert env["error"] == "bad input"
    assert env["result"] is None
    assert env["hint"] == "try again"


def test_ok_handles_complex_result_payload():
    from mcp_servers._common import ok
    env = json.loads(ok(result={"channels": ["S:CurrentM", "S:LifetimeM"], "count": 2}))
    assert env["result"] == {"channels": ["S:CurrentM", "S:LifetimeM"], "count": 2}


def test_mcp_tool_safe_passes_through_normal_return():
    from mcp_servers._common import mcp_tool_safe, ok
    @mcp_tool_safe
    def my_tool(x: int) -> str:
        return ok(result=x * 2)
    env = json.loads(my_tool(7))
    assert env["status"] == "ok" and env["result"] == 14


def test_mcp_tool_safe_catches_exception_returns_structured_error():
    """The decorator's whole point — never let an exception cross stdio."""
    from mcp_servers._common import mcp_tool_safe
    @mcp_tool_safe
    def buggy_tool(x: int) -> str:
        raise ValueError("invalid input")
    env = json.loads(buggy_tool(0))
    assert env["status"] == "error"
    assert "ValueError" in env["error"]
    assert "invalid input" in env["error"]
    assert env["tool"] == "buggy_tool"
    # Traceback excerpt is included so debugging via the LLM trace is possible
    assert "traceback_excerpt" in env


def test_mcp_tool_safe_preserves_function_name():
    """functools.wraps keeps __name__ so the FastMCP introspection is correct."""
    from mcp_servers._common import mcp_tool_safe, ok
    @mcp_tool_safe
    def my_named_tool() -> str:
        return ok()
    assert my_named_tool.__name__ == "my_named_tool"


def test_add_project_root_to_path_idempotent():
    from mcp_servers._common import add_project_root_to_path
    root1 = add_project_root_to_path()
    root2 = add_project_root_to_path()
    assert root1 == root2
    # The contract is "root is reachable on sys.path", not exact-count-of-1 —
    # other test files insert their own (sometimes non-resolved) variant of
    # the same path, so duplicates are an environmental detail outside our control.
    assert str(root1) in sys.path


# ---------------------------------------------------------------------------
# 2. mcp_client: env expansion, retry classifier, envelope unwrap, singleton
# ---------------------------------------------------------------------------
def test_env_var_expansion_with_value(monkeypatch):
    from src.tools.mcp_client import _expand_env_value
    monkeypatch.setenv("MY_TEST_VAR", "hello-world")
    assert _expand_env_value("${MY_TEST_VAR}") == "hello-world"


def test_env_var_expansion_with_default(monkeypatch):
    from src.tools.mcp_client import _expand_env_value
    monkeypatch.delenv("UNSET_TEST_VAR", raising=False)
    assert _expand_env_value("${UNSET_TEST_VAR:-fallback}") == "fallback"


def test_env_var_expansion_passes_through_plain_value():
    from src.tools.mcp_client import _expand_env_value
    assert _expand_env_value("plain-string") == "plain-string"
    assert _expand_env_value("/some/path") == "/some/path"


def test_env_var_expansion_handles_unset_no_default(monkeypatch):
    from src.tools.mcp_client import _expand_env_value
    monkeypatch.delenv("ALSO_UNSET", raising=False)
    assert _expand_env_value("${ALSO_UNSET}") == ""


@pytest.mark.parametrize("exc_cls,is_retriable", [
    (BrokenPipeError, True),
    (ConnectionResetError, True),
    (ConnectionRefusedError, True),
    (TimeoutError, True),
    (json.JSONDecodeError("x", "doc", 0), True),
    (ValueError, False),
    (KeyError, False),
    (RuntimeError, False),
])
def test_retry_classifier_only_retries_transport_errors(exc_cls, is_retriable):
    """Transport failures are worth retrying; tool-level errors are not."""
    from src.tools.mcp_client import _is_retriable_exception
    inst = exc_cls if isinstance(exc_cls, BaseException) else exc_cls("test")
    assert _is_retriable_exception(inst) is is_retriable


def test_unwrap_mcp_envelope_returns_inner_result_for_ok_string():
    """Legacy contract: calculator returned `'4'` not `'{"result":"4"}'`. Stage C
    must preserve this so prompt heuristics keyed off the old shape still work."""
    from src.tools.mcp_client import unwrap_mcp_envelope
    assert unwrap_mcp_envelope('{"status":"ok","result":"4","error":null}') == "4"


def test_unwrap_mcp_envelope_serializes_complex_ok_result():
    from src.tools.mcp_client import unwrap_mcp_envelope
    env = '{"status":"ok","result":{"channels":[1,2,3]},"error":null}'
    assert unwrap_mcp_envelope(env) == '{"channels": [1, 2, 3]}'


def test_unwrap_mcp_envelope_returns_full_envelope_on_error():
    """On error, surface the WHOLE envelope so the LLM can read .error/.hint
    and self-correct on the next ReAct iteration."""
    from src.tools.mcp_client import unwrap_mcp_envelope
    env = '{"status":"error","result":null,"error":"bad data_source","hint":"pick from X"}'
    assert unwrap_mcp_envelope(env) == env


def test_unwrap_mcp_envelope_passthrough_for_malformed_json():
    from src.tools.mcp_client import unwrap_mcp_envelope
    assert unwrap_mcp_envelope("not json") == "not json"
    assert unwrap_mcp_envelope('{"no_status_field": true}') == '{"no_status_field": true}'


def test_unwrap_mcp_envelope_handles_null_result():
    from src.tools.mcp_client import unwrap_mcp_envelope
    assert unwrap_mcp_envelope('{"status":"ok","result":null,"error":null}') == ""


def test_get_default_client_is_singleton():
    """Module-level singleton — every BaseTool shim shares one client instance."""
    from src.tools.mcp_client import get_default_client
    c1 = get_default_client()
    c2 = get_default_client()
    assert c1 is c2


# ---------------------------------------------------------------------------
# 3. BaseTool MCP-shim wiring — each shim calls (server, tool, args)
# ---------------------------------------------------------------------------
def _patched_client(monkeypatch, response: str = '{"status":"ok","result":"OK","error":null}'):
    """Patch get_default_client to return a mock; track call_tool_sync."""
    mock_client = MagicMock()
    mock_client.call_tool_sync = MagicMock(return_value=response)
    monkeypatch.setattr("src.tools.mcp_client.get_default_client", lambda: mock_client)
    return mock_client


@pytest.mark.parametrize("tool_factory,run_kwargs,expected_server,expected_tool,expected_args", [
    # tool_factory: callable → BaseTool instance
    # expected_args: subset of args that MUST be in the dispatched call
    (lambda: __import__("src.tools.calculator", fromlist=["CalculatorTool"]).CalculatorTool(),
     {"expression": "1+1"},
     "compute", "calculator", {"expression": "1+1"}),
    (lambda: __import__("src.tools.code_executor", fromlist=["CodeExecutorTool"]).CodeExecutorTool(),
     {"code": "print(1)"},
     "compute", "code_executor", {"code": "print(1)"}),
    (lambda: __import__("src.tools.data_analysis", fromlist=["DataAnalysisTool"]).DataAnalysisTool(),
     {"analysis_code": "print(len(df))", "data_source": "work_requests"},
     "compute", "data_analysis",
     {"analysis_code": "print(len(df))", "data_source": "work_requests"}),
    (lambda: __import__("src.tools.graph_search", fromlist=["GraphSearchTool"]).GraphSearchTool(),
     {"query": "kicker"},
     "graph", "graph_search", {"query": "kicker"}),
    (lambda: __import__("src.tools.channel_finder", fromlist=["ChannelFinderTool"]).ChannelFinderTool(),
     {"query": "S-DCCT", "max_results": 5},
     "controls", "channel_find", {"query": "S-DCCT", "max_results": 5}),
    (lambda: __import__("src.tools.archiver_reader", fromlist=["ArchiverReaderTool"]).ArchiverReaderTool(),
     {"pv_names": "S:CurrentM", "start": "1h"},
     "archiver", "archiver_read", {"pv_names": ["S:CurrentM"], "start": "1h"}),
    (lambda: __import__("src.tools.pv_search", fromlist=["PVSearchTool"]).PVSearchTool(),
     {"query": "PAR kicker", "max_pvs": 2},
     "controls", "pv_search", {"query": "PAR kicker", "max_pvs": 2}),
])
def test_basetool_shim_dispatches_to_correct_mcp_endpoint(
    monkeypatch, tool_factory, run_kwargs, expected_server, expected_tool, expected_args
):
    """Each LangChain BaseTool's _run must dispatch to the right
    (server, tool, args) on its singleton MCPClientManager."""
    mock_client = _patched_client(monkeypatch)
    tool = tool_factory()
    tool._run(**run_kwargs)
    mock_client.call_tool_sync.assert_called_once()
    args = mock_client.call_tool_sync.call_args
    # call_args is .args (positional) or .kwargs (named)
    pos = args.args
    assert pos[0] == expected_server, f"wrong server: {pos[0]} != {expected_server}"
    assert pos[1] == expected_tool, f"wrong tool: {pos[1]} != {expected_tool}"
    dispatched_args = pos[2]
    for k, v in expected_args.items():
        assert dispatched_args.get(k) == v, (
            f"arg {k} mismatch: dispatched={dispatched_args.get(k)!r} expected={v!r}"
        )


def test_pv_reader_shim_splits_csv_pv_names(monkeypatch):
    """PVReaderTool takes a CSV string and must split it into a list before
    dispatching to controls/channel_read (which expects ``channels: list[str]``)."""
    mock_client = _patched_client(monkeypatch)
    from src.tools.pv_reader import PVReaderTool
    PVReaderTool()._run(pv_names="S:A, S:B, S:C")
    args = mock_client.call_tool_sync.call_args.args
    assert args[0] == "controls"
    assert args[1] == "channel_read"
    assert args[2]["channels"] == ["S:A", "S:B", "S:C"]


def test_basetool_shim_returns_unwrapped_ok_payload(monkeypatch):
    """When MCP returns ok, the shim returns the inner result string —
    preserving pre-Stage-C behavior (calculator returned '4', not the envelope)."""
    _patched_client(monkeypatch, response='{"status":"ok","result":"5","error":null}')
    from src.tools.calculator import CalculatorTool
    assert CalculatorTool()._run(expression="2+3") == "5"


def test_basetool_shim_returns_full_envelope_on_error(monkeypatch):
    """When MCP returns error, the shim returns the FULL envelope so the LLM
    sees the structured error fields — this is what enables ReAct self-correct."""
    err_env = '{"status":"error","result":null,"error":"bad","hint":"fix it"}'
    _patched_client(monkeypatch, response=err_env)
    from src.tools.calculator import CalculatorTool
    out = CalculatorTool()._run(expression="???")
    parsed = json.loads(out)
    assert parsed["status"] == "error"
    assert parsed["error"] == "bad"
    assert parsed["hint"] == "fix it"


# ---------------------------------------------------------------------------
# 4. Each MCP server boots and exposes the expected tool names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("server_module,expected_tools", [
    ("mcp_servers.compute", {"calculator", "code_executor", "data_analysis"}),
    ("mcp_servers.controls", {"channel_read", "channel_find", "pv_search"}),
    ("mcp_servers.archiver", {"archiver_read"}),
    ("mcp_servers.graph", {"graph_search"}),
])
def test_mcp_server_exposes_expected_tools(server_module, expected_tools):
    import asyncio
    import importlib
    mod = importlib.import_module(server_module)
    tools = asyncio.run(mod.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == expected_tools, (
        f"{server_module} tool set mismatch: got={names} expected={expected_tools}"
    )


# ---------------------------------------------------------------------------
# 5. Pure-impl functions still callable directly (used by MCP servers)
# ---------------------------------------------------------------------------
def test_calculator_impl_works_without_mcp():
    """The MCP server calls calculator_impl directly — must not require MCP."""
    from src.tools.calculator import calculator_impl
    assert calculator_impl("2+2") == "4"


def test_code_executor_impl_works_without_mcp():
    from src.tools.code_executor import code_executor_impl
    out = json.loads(code_executor_impl("print('hi')"))
    assert out["returncode"] == 0
    assert "hi" in out["stdout"]


def test_code_executor_impl_blocks_forbidden_pattern():
    from src.tools.code_executor import code_executor_impl
    out = json.loads(code_executor_impl("import os; os.system('echo x')"))
    assert out["returncode"] != 0
    assert "Forbidden" in out["stderr"]


def test_data_analysis_impl_unknown_data_source_returns_error():
    """The error envelope must match the MCP server's error contract — used by
    the LLM to self-correct (pick a valid data_source on retry)."""
    from src.tools.data_analysis import data_analysis_impl
    out = json.loads(data_analysis_impl("print(1)", "bogus_source"))
    assert out["returncode"] == 1
    assert "Unknown data_source" in out["error"]
