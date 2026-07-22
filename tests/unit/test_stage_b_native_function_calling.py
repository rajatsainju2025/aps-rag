"""Stage B unit tests — native function calling end-to-end.

Stage B replaces the prompt-based ``TOOL_CALL: name(args)`` regex protocol
with native function calling via ``NativeToolsArgoLLM``. The contracts:

  1. ``LLMFactory.get_synthesis_llm`` returns a ``NativeToolsArgoLLM``
     instance (not the legacy text-only ``ArgoLLMClientAG``).
  2. ``ReasoningAgent.__init__`` calls ``llm.bind_tools(...)`` when:
     - ``tools`` is non-empty, AND
     - the LLM class advertises ``supports_tool_calls = True``.
     Otherwise it falls through to the legacy prompt-based path.
  3. ``ReasoningAgent._tool_calls_from_response`` extracts native
     ``response.tool_calls`` and converts ``[{"name", "args": dict}]`` to
     the orchestrator's ``[{"tool", "args": dict}]`` shape.
  4. ``ToolExecutor.execute`` appends a ``Tool X executed (N ms)`` line to
     ``state["reasoning_trace"]`` so the verifier (and OpenWebUI) can detect
     tool usage from the response trace.
  5. The counting verifier's ``extract_answer_count`` skips year-shaped
     candidates so phrasings like *"31 matching 2025 work requests"* extract
     31, not 2025.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# 1. LLMFactory.get_synthesis_llm now returns NativeToolsArgoLLM
# ---------------------------------------------------------------------------
def test_get_synthesis_llm_returns_native_tools_client():
    """The synthesis LLM must support native tool calling so reasoning_agent
    can bind tools natively (instead of prompt-based TOOL_CALL: directives)."""
    from src.integrations.argo.llm_client import LLMFactory
    from src.integrations.argo.native_tools_client import NativeToolsArgoLLM
    llm = LLMFactory.get_synthesis_llm(username="testuser", base_url="http://x")
    assert isinstance(llm, NativeToolsArgoLLM)
    assert llm.supports_tool_calls is True


def test_get_synthesis_llm_uses_chat_endpoint_not_streamchat():
    """NativeToolsArgoLLM must hit /chat/ (not /streamchat/) — that's the
    Argo endpoint that supports tool calling."""
    from src.integrations.argo.llm_client import LLMFactory
    llm = LLMFactory.get_synthesis_llm(username="testuser", base_url="http://x")
    assert llm._get_chat_url().endswith("/chat/"), (
        f"Expected /chat/ endpoint, got {llm._get_chat_url()}"
    )


# ---------------------------------------------------------------------------
# 2. ReasoningAgent native tool binding
# ---------------------------------------------------------------------------
def _fake_native_llm():
    """Build a mock LLM that advertises supports_tool_calls and tracks bind_tools."""
    llm = MagicMock()
    llm.__class__ = type("FakeNativeLLM", (), {"supports_tool_calls": True})
    bound_copy = MagicMock()
    bound_copy.__class__ = llm.__class__
    llm.bind_tools = MagicMock(return_value=bound_copy)
    return llm, bound_copy


def _fake_legacy_llm():
    """Mock LLM without native tool calling support."""
    llm = MagicMock()
    llm.__class__ = type("FakeLegacyLLM", (), {"supports_tool_calls": False})
    return llm


def test_reasoning_agent_binds_tools_natively_when_supported():
    from src.agents.reasoning_agent import ReasoningAgent
    llm, bound = _fake_native_llm()
    tools = {"calculator": MagicMock(name="calc"),
             "data_analysis": MagicMock(name="da")}
    ra = ReasoningAgent(llm=llm, tools=tools)
    # Two bind_tools calls: full toolbox (default self.llm) and gated
    # variant without data_analysis (self._llm_without_data_analysis).
    assert llm.bind_tools.call_count == 2
    full_call_args = llm.bind_tools.call_args_list[0].args[0]
    gated_call_args = llm.bind_tools.call_args_list[1].args[0]
    assert len(full_call_args) == 2
    assert len(gated_call_args) == 1  # data_analysis filtered out
    assert tools["data_analysis"] not in gated_call_args
    assert ra.llm is bound
    assert ra._tools_bound_natively is True
    assert ra._llm_without_data_analysis is bound
    # Tool schemas block stays empty when native binding succeeded
    assert ra._tool_schemas_block == ""


def test_reasoning_agent_gated_llm_skipped_when_no_data_analysis():
    """When data_analysis isn't in the tools dict, no gated variant is built."""
    from src.agents.reasoning_agent import ReasoningAgent
    llm, _bound = _fake_native_llm()
    tools = {"calculator": MagicMock(name="calc")}
    ra = ReasoningAgent(llm=llm, tools=tools)
    llm.bind_tools.assert_called_once()
    assert ra._llm_without_data_analysis is None


def test_synthesize_hides_data_analysis_for_non_counting_query():
    """Behaviour: non-counting query → gated LLM (no data_analysis)."""
    from src.agents.reasoning_agent import ReasoningAgent
    llm = MagicMock()
    llm.__class__ = type("FakeNativeLLM", (), {"supports_tool_calls": True})
    bound_full = MagicMock(name="bound_full")
    bound_gated = MagicMock(name="bound_gated")
    for b in (bound_full, bound_gated):
        b.invoke = MagicMock(return_value=MagicMock(content="answer", tool_calls=[]))
    llm.bind_tools = MagicMock(side_effect=[bound_full, bound_gated])
    tools = {"calculator": MagicMock(name="calc"),
             "data_analysis": MagicMock(name="da")}
    ra = ReasoningAgent(llm=llm, tools=tools, critique_llm=MagicMock())
    state = {
        "original_query": "latest entry by Chenran",
        "enhanced_query": "latest entry by Chenran",
        "counting_intent": False,
        "retrieved_docs": [], "tool_results": [], "memory_context": {},
        "reasoning_trace": [], "correction_round": 0,
    }
    ra.synthesize(state)
    assert bound_gated.invoke.called, "non-counting query must use gated LLM"
    assert not bound_full.invoke.called, "non-counting query must NOT use full LLM"


def test_synthesize_exposes_data_analysis_for_counting_query():
    """Behaviour: counting query → full LLM (data_analysis available)."""
    from src.agents.reasoning_agent import ReasoningAgent
    llm = MagicMock()
    llm.__class__ = type("FakeNativeLLM", (), {"supports_tool_calls": True})
    bound_full = MagicMock(name="bound_full")
    bound_gated = MagicMock(name="bound_gated")
    for b in (bound_full, bound_gated):
        b.invoke = MagicMock(return_value=MagicMock(content="answer", tool_calls=[]))
    llm.bind_tools = MagicMock(side_effect=[bound_full, bound_gated])
    tools = {"calculator": MagicMock(name="calc"),
             "data_analysis": MagicMock(name="da")}
    ra = ReasoningAgent(llm=llm, tools=tools, critique_llm=MagicMock())
    state = {
        "original_query": "how many entries by Chenran",
        "enhanced_query": "how many entries by Chenran",
        "counting_intent": True,
        "retrieved_docs": [], "tool_results": [], "memory_context": {},
        "reasoning_trace": [], "correction_round": 0,
    }
    ra.synthesize(state)
    assert bound_full.invoke.called, "counting query must use full LLM"
    assert not bound_gated.invoke.called, "counting query must NOT use gated LLM"


def test_reasoning_agent_falls_back_to_prompt_path_for_legacy_llm(monkeypatch):
    """Patch render_tool_schemas so we don't need to fabricate real BaseTool
    instances — the contract under test is just 'fallback path renders SOMETHING
    into _tool_schemas_block when bind_tools is unavailable'."""
    import src.agents.reasoning_agent as ra_mod
    monkeypatch.setattr(ra_mod, "render_tool_schemas", lambda tools: "FAKE_SCHEMA_BLOCK")
    llm = _fake_legacy_llm()
    tools = {"calculator": MagicMock(name="calc")}
    ra = ra_mod.ReasoningAgent(llm=llm, tools=tools)
    # Legacy LLMs don't have bind_tools called
    assert not hasattr(llm, "bind_tools") or not llm.bind_tools.called
    assert ra.llm is llm  # not replaced
    assert ra._tools_bound_natively is False
    assert ra._tool_schemas_block == "FAKE_SCHEMA_BLOCK"


def test_reasoning_agent_no_tools_no_binding():
    """With empty tools dict, bind_tools is NOT called even on a native LLM
    (no point binding zero tools)."""
    from src.agents.reasoning_agent import ReasoningAgent
    llm, _ = _fake_native_llm()
    ra = ReasoningAgent(llm=llm, tools={})
    llm.bind_tools.assert_not_called()
    assert ra._tools_bound_natively is False


def test_reasoning_agent_bind_tools_failure_falls_back_gracefully(monkeypatch):
    """If bind_tools raises (vendor mapping missing, network etc.), we must
    fall back to the prompt-based path rather than crashing the agent."""
    import src.agents.reasoning_agent as ra_mod
    monkeypatch.setattr(ra_mod, "render_tool_schemas", lambda tools: "FAKE_SCHEMA_BLOCK")
    llm = MagicMock()
    llm.__class__ = type("BrokenNativeLLM", (), {"supports_tool_calls": True})
    llm.bind_tools = MagicMock(side_effect=RuntimeError("vendor down"))
    ra = ra_mod.ReasoningAgent(llm=llm, tools={"x": MagicMock()})
    assert ra._tools_bound_natively is False
    assert ra._tool_schemas_block == "FAKE_SCHEMA_BLOCK"


# ---------------------------------------------------------------------------
# 3. _tool_calls_from_response — native → legacy shape conversion
# ---------------------------------------------------------------------------
def test_tool_calls_from_response_handles_native_aimessage():
    from src.agents.reasoning_agent import ReasoningAgent
    response = MagicMock()
    response.tool_calls = [
        {"id": "call_1", "name": "data_analysis",
         "args": {"analysis_code": "print(31)", "data_source": "work_requests"},
         "type": "tool_call"},
    ]
    out = ReasoningAgent._tool_calls_from_response(response)
    assert out == [{
        "tool": "data_analysis",
        "args": {"analysis_code": "print(31)", "data_source": "work_requests"},
    }]


def test_tool_calls_from_response_handles_multiple_calls():
    from src.agents.reasoning_agent import ReasoningAgent
    response = MagicMock()
    response.tool_calls = [
        {"id": "1", "name": "calculator", "args": {"expression": "2+2"}},
        {"id": "2", "name": "channel_finder", "args": {"query": "kicker"}},
    ]
    out = ReasoningAgent._tool_calls_from_response(response)
    assert [c["tool"] for c in out] == ["calculator", "channel_finder"]


def test_tool_calls_from_response_returns_empty_for_none_response():
    from src.agents.reasoning_agent import ReasoningAgent
    assert ReasoningAgent._tool_calls_from_response(None) == []


def test_tool_calls_from_response_returns_empty_for_no_tool_calls():
    from src.agents.reasoning_agent import ReasoningAgent
    response = MagicMock()
    response.tool_calls = []
    assert ReasoningAgent._tool_calls_from_response(response) == []


def test_tool_calls_from_response_skips_malformed_entries():
    """Entries without a name or that aren't dicts must be skipped, not crash."""
    from src.agents.reasoning_agent import ReasoningAgent
    response = MagicMock()
    response.tool_calls = [
        {"id": "1", "name": "calculator", "args": {"expression": "1+1"}},
        "not-a-dict",
        {"id": "2"},  # no name
        {"id": "3", "name": "data_analysis", "args": {"analysis_code": "x"}},
    ]
    out = ReasoningAgent._tool_calls_from_response(response)
    assert len(out) == 2
    assert {c["tool"] for c in out} == {"calculator", "data_analysis"}


def test_tool_calls_from_response_defaults_args_to_empty_dict():
    """Some vendors return tool_calls without an args field (e.g. zero-arg
    tools). Must surface as {} not None."""
    from src.agents.reasoning_agent import ReasoningAgent
    response = MagicMock()
    response.tool_calls = [{"id": "1", "name": "calculator"}]  # no args
    out = ReasoningAgent._tool_calls_from_response(response)
    assert out == [{"tool": "calculator", "args": {}}]


# ---------------------------------------------------------------------------
# 4. ToolExecutor surfaces tool usage in reasoning_trace
# ---------------------------------------------------------------------------
def test_tool_executor_appends_executed_line_to_trace():
    from src.agents.tool_executor import ToolExecutor
    fake_tool = MagicMock()
    fake_tool.run = MagicMock(return_value="31")
    executor = ToolExecutor(tools={"data_analysis": fake_tool})
    state = {
        "tool_calls": [{"tool": "data_analysis", "args": {"analysis_code": "x", "data_source": "work_requests"}}],
        "reasoning_trace": ["prior step"],
    }
    out = executor.execute(state)
    trace = out["reasoning_trace"]
    assert any("data_analysis" in line and "executed" in line for line in trace), (
        f"Expected 'Tool data_analysis executed' line in trace, got: {trace}"
    )
    # Prior trace lines must be preserved
    assert "prior step" in trace


def test_tool_executor_appends_failed_line_on_exception():
    from src.agents.tool_executor import ToolExecutor
    fake_tool = MagicMock()
    fake_tool.run = MagicMock(side_effect=ValueError("bad input"))
    executor = ToolExecutor(tools={"calculator": fake_tool})
    state = {"tool_calls": [{"tool": "calculator", "args": "1+"}], "reasoning_trace": []}
    out = executor.execute(state)
    assert any("calculator" in l and "failed" in l for l in out["reasoning_trace"])


def test_tool_executor_appends_skipped_line_when_tool_missing():
    from src.agents.tool_executor import ToolExecutor
    executor = ToolExecutor(tools={})  # registry empty
    state = {"tool_calls": [{"tool": "data_analysis", "args": {}}], "reasoning_trace": []}
    out = executor.execute(state)
    assert any("data_analysis" in l and "skipped" in l for l in out["reasoning_trace"])


def test_tool_executor_no_tool_calls_returns_empty_results():
    from src.agents.tool_executor import ToolExecutor
    executor = ToolExecutor(tools={})
    out = executor.execute({"tool_calls": [], "reasoning_trace": []})
    assert out == {"tool_results": []}


# ---------------------------------------------------------------------------
# 5. counting_verifier extract_answer_count fixes
# ---------------------------------------------------------------------------
def test_extract_answer_count_skips_year_shaped_match():
    """The 'X matching YYYY work requests' bug: the regex used to grab YYYY
    as the count. Now must skip year-shaped candidates and pick the real count."""
    sys.path.insert(0, str(Path(_ROOT) / "evaluation"))
    from counting_verifier import extract_answer_count
    ans = "I found 31 matching 2025 work requests for Linac, PAR, or Booster."
    assert extract_answer_count(ans) == 31


@pytest.mark.parametrize("ans, expected", [
    ("Total: 24", 24),
    ("The count is 35.", 35),
    ("The answer is 121 [source_1].", 121),
    ("In 2025, 580 work requests targeted the Storage Ring.", 580),
    ("There are 31 matching 2025 work requests.", 31),
    ("There were 31 work requests entered in 2025.", 31),
    ("No clear count available.", None),
    ("In 2025, work was done.", None),  # year-only, no count
])
def test_extract_answer_count_known_phrasings(ans, expected):
    sys.path.insert(0, str(Path(_ROOT) / "evaluation"))
    from counting_verifier import extract_answer_count
    assert extract_answer_count(ans) == expected
