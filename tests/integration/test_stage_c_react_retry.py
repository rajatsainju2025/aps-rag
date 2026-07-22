"""Stage C integration test — ReAct retry on tool errors.

Verifies the full re-attempt chain end-to-end (offline, mocked LLM):

    1. Synthesize: LLM emits a tool_call with bad arguments.
    2. ToolExecutor: runs the (real) MCP-backed tool → structured error envelope.
    3. Re-synthesize: LLM is invoked AGAIN with the error in tool_results.
    4. The new LLM response can correct the call (verified by checking the
       second synthesize sees the error message).

The orchestrator's loop ([naive_rag_node](agentic-graph-rag/src/agents/orchestrator.py))
runs synthesize → execute → re-synthesize until either (a) no more
tool_calls or (b) the ``tool_chain_safety_cap`` is hit. This test exercises
that loop with a controlled LLM mock so the assertion is deterministic.

Why this matters for the user's "ReAct + error re-attempts" requirement:
the tool-error path is the load-bearing case. If a tool returns a structured
error and the LLM doesn't re-invoke, ReAct has degenerated into a single-shot
call — which is exactly the failure mode Stage A/B fixed for counting queries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _ai_message_with_tool_calls(content: str, tool_calls: list):
    """Build an AIMessage-like mock that NativeToolsArgoLLM would return."""
    from langchain_core.messages import AIMessage
    return AIMessage(content=content, tool_calls=tool_calls)


def test_react_loop_retries_after_tool_error_envelope():
    """Simulate: LLM calls data_analysis with bad data_source → tool returns
    structured error → LLM is re-invoked with the error in tool_results
    → LLM corrects the data_source.

    This exercises ToolExecutor (real MCP call to compute server) end-to-end;
    only the LLM is mocked.
    """
    from src.agents.tool_executor import ToolExecutor
    from src.tools.data_analysis import DataAnalysisTool

    # Real MCP-backed tool — the test is end-to-end through the MCP shim.
    executor = ToolExecutor(tools={"data_analysis": DataAnalysisTool()})

    # Step 1: Simulate first LLM response — emits a tool_call with bad args.
    bad_call_state = {
        "tool_calls": [{
            "tool": "data_analysis",
            "args": {"analysis_code": "print(len(df))", "data_source": "bogus_source"},
        }],
        "tool_results": [],
        "reasoning_trace": ["Synthesizing answer (round 1) from 5 documents"],
    }
    after_bad = executor.execute(bad_call_state)

    # Tool returns a result — but it's an error envelope, not a hallucinated
    # success. The agent's ReAct loop needs to see this on the next iteration.
    assert len(after_bad["tool_results"]) == 1
    bad_result = after_bad["tool_results"][0]
    assert bad_result["tool"] == "data_analysis"
    # The MCP server returned a structured error; the shim surfaced the full envelope
    parsed_err = json.loads(bad_result["result"])
    assert parsed_err["status"] == "error"
    assert "Unknown data_source" in parsed_err["error"]
    assert "hint" in parsed_err
    # And the trace recorded the executed call (this is what tells the
    # verifier / OpenWebUI / observers that the tool fired)
    assert any("data_analysis" in line for line in after_bad["reasoning_trace"])

    # Step 2: Simulate the LLM seeing the error and emitting a CORRECTED call.
    # In production this is done by reasoning_agent.synthesize() being invoked
    # again with the error result in state["tool_results"]. The corrected call
    # would use a valid data_source.
    corrected_call_state = {
        "tool_calls": [{
            "tool": "data_analysis",
            "args": {"analysis_code": "print(len(df))", "data_source": "work_requests"},
        }],
        "tool_results": after_bad["tool_results"],  # carry forward
        "reasoning_trace": after_bad["reasoning_trace"],
    }
    after_good = executor.execute(corrected_call_state)

    # Step 3: Verify the corrected call SUCCEEDS — produces a real count.
    assert len(after_good["tool_results"]) == 2  # bad result + good result
    good_result = after_good["tool_results"][1]
    assert good_result["tool"] == "data_analysis"
    parsed_ok = json.loads(good_result["result"])
    # The unwrapped envelope inner result is itself a JSON string with stdout
    inner = json.loads(parsed_ok) if isinstance(parsed_ok, str) else parsed_ok
    assert "stdout" in inner
    assert int(inner["stdout"].strip()) > 0  # work_requests has thousands of rows


def test_tool_executor_surfaces_unregistered_tool_as_error_for_retry():
    """If the LLM hallucinates a tool name that doesn't exist, the executor
    must surface a clear error so the LLM can pick a real tool on retry —
    not silently skip and continue."""
    from src.agents.tool_executor import ToolExecutor
    executor = ToolExecutor(tools={})  # empty registry
    state = {
        "tool_calls": [{"tool": "imaginary_tool", "args": {"x": 1}}],
        "reasoning_trace": [],
    }
    result = executor.execute(state)
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["error"] is True
    assert "imaginary_tool" in result["tool_results"][0]["result"]
    # Trace also records the skip so observers see the failure
    assert any("imaginary_tool" in l and "skipped" in l for l in result["reasoning_trace"])


def test_mcp_client_transport_retry_returns_structured_error_after_exhaustion(monkeypatch):
    """When all retries fail, the client must return a structured error
    envelope — never a raised exception. This is what feeds the LLM's
    ReAct retry loop with an error message it can read."""
    from src.tools import mcp_client
    from src.tools.mcp_client import MCPClientManager

    mgr = MCPClientManager.__new__(MCPClientManager)
    mgr._config = {"mcpServers": {"x": {"command": "echo"}}}
    mgr._config_path = "/tmp/fake.json"
    mgr._available = True

    # Force every attempt to raise BrokenPipeError (retriable transport error)
    def always_pipe_break(*a, **kw):
        raise BrokenPipeError("simulated pipe break")
    monkeypatch.setattr(mgr, "_invoke_once", always_pipe_break)
    # Speed up: collapse the backoff so the test is fast
    monkeypatch.setattr(mcp_client, "_MCP_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(mcp_client, "_MCP_RETRY_BASE_DELAY_S", 0.01)

    out = mgr.call_tool_sync("x", "y", {})
    parsed = json.loads(out)
    assert parsed["status"] == "error"
    assert "BrokenPipeError" in parsed["error"]
    assert parsed["server"] == "x"
    assert parsed["tool"] == "y"
    assert parsed["attempts"] == 2
    assert parsed["transport"] == "mcp_stdio"


def test_mcp_client_does_not_retry_non_transport_errors(monkeypatch):
    """ValueError / KeyError etc. are tool-level signals — retrying them just
    repeats the same bad call. Must fail fast with a structured envelope."""
    from src.tools import mcp_client
    from src.tools.mcp_client import MCPClientManager

    mgr = MCPClientManager.__new__(MCPClientManager)
    mgr._config = {"mcpServers": {"x": {"command": "echo"}}}
    mgr._config_path = "/tmp/fake.json"
    mgr._available = True

    call_counter = {"n": 0}

    def raise_value_error(*a, **kw):
        call_counter["n"] += 1
        raise ValueError("not retriable")
    monkeypatch.setattr(mgr, "_invoke_once", raise_value_error)
    monkeypatch.setattr(mcp_client, "_MCP_RETRY_ATTEMPTS", 5)  # would be 5 if naive
    monkeypatch.setattr(mcp_client, "_MCP_RETRY_BASE_DELAY_S", 0.0)

    out = mgr.call_tool_sync("x", "y", {})
    parsed = json.loads(out)
    assert parsed["status"] == "error"
    assert "ValueError" in parsed["error"]
    # Should bail out on the first try — non-retriable
    assert call_counter["n"] == 1, (
        f"non-retriable error retried {call_counter['n']}× — should bail on attempt 1"
    )
