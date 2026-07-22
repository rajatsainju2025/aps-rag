"""End-to-end orchestrator tests (mocked agents).

Drives queries through the full LangGraph DAG with each agent mocked at the
class boundary. Goal: catch DAG-wiring regressions without depending on the
behavior of any specific agent. Pattern: patch each agent class so that its
construction returns a MagicMock with the methods the orchestrator calls.

Coverage:
  - Service dispatch routes to each of the 5 service nodes (5 tests)
  - Naive-RAG single-pass with no tools (1 test)
  - Naive-RAG re-synthesizes after a tool call (1 test)
  - Agentic-RAG corrective loop triggers re-retrieve (1 test)
  - Agentic-RAG hits max_correction_rounds and force-accepts (1 test)
  - check_tools routes to tool_node when tool_calls present (1 test)
  - check_tools enforces safety cap when tool_results accumulate (1 test)
  - format_response strips TOOL_CALL artifacts and writes memory (2 tests)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Mock agent factory — every class becomes a MagicMock returning fixed nodes
# ─────────────────────────────────────────────────────────────────────────────
def _doc(i: int) -> Document:
    return Document(page_content=f"text {i}", metadata={"id": f"d{i}", "collection": "entry_logbook"})


def _settings(**overrides):
    """Minimal settings shape for the orchestrator constructor."""
    agent = SimpleNamespace(
        use_create_react_agent=False,
        tool_chain_safety_cap=8,
        max_correction_rounds=3,
        quality_threshold=0.4,
        irrelevant_doc_threshold=0.5,
    )
    for k, v in overrides.items():
        setattr(agent, k, v)
    return SimpleNamespace(agent=agent)


def _components(**overrides):
    """Minimal components dict — orchestrator only reads, doesn't introspect."""
    base = {
        "fast_llm": MagicMock(),
        "synthesis_llm": MagicMock(),
        "critique_llm": MagicMock(),
        "vector_retriever": MagicMock(),
        "bm25_retriever": MagicMock(),
        "graph_retriever": MagicMock(),
        "fusion": MagicMock(),
        "reranker": MagicMock(),
        "query_enhancer": MagicMock(),
        "neo4j_client": None,
        "memory_manager": None,
        "tools": {},
    }
    base.update(overrides)
    return base


def _initial_state(query: str = "What is the beam current?", service_mode: str = "naive_rag"):
    """Minimal AgentState satisfying the typed dict — orchestrator nodes will fill rest."""
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "test-session",
        "service_mode": service_mode,
        "original_query": query,
        "enhanced_query": query,
        "query_intent": "",
        "selected_databases": ["all_databases"],
        "temporal_context": None,
        "query_variants": [query],
        "plan": "",
        "plan_steps": [],
        "reasoning_trace": [],
        "retrieved_docs": [],
        "graph_context": [],
        "web_results": [],
        "retrieval_sources": [],
        "draft_answer": "",
        "final_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "needs_correction": False,
        "correction_round": 0,
        "correction_issues": [],
        "quality_status": "",
        "retrieval_strategy": "HYBRID_3WAY",
        "extracted_entities": [],
        "tool_calls": [],
        "tool_results": [],
        "memory_context": {},
        "pv_addresses": [],
        "pv_readings": [],
        "pv_history": [],
        "entry_attachments": [],
        "next_node": "",
        "error": None,
        "iteration_count": 0,
    }


def _build_with_mocks(
    *,
    routing_mode: str = "naive_rag",
    plan_steps: list | None = None,
    retrieved_docs: list[Document] | None = None,
    synthesis_result: dict | None = None,
    corrective_result: dict | None = None,
    daily_summary_result: dict | None = None,
    troubleshooting_result: dict | None = None,
    studies_result: dict | None = None,
    tool_executor_result: dict | None = None,
    settings_overrides: dict | None = None,
    extra_components: dict | None = None,
):
    """Construct the orchestrator with every internal agent class patched to a mock."""
    from src.agents import orchestrator as orch_mod

    # Default canned returns
    docs = retrieved_docs if retrieved_docs is not None else [_doc(i) for i in range(2)]
    synthesis = synthesis_result if synthesis_result is not None else {
        "draft_answer": "The beam current is 100 mA.",
        "tool_calls": [],
        "tool_results": [],
        "citations": [],
        "reasoning_trace": ["synth done"],
    }
    corrective = corrective_result if corrective_result is not None else {
        "final_answer": synthesis["draft_answer"],
        "confidence_score": 0.85,
        "needs_correction": False,
        "quality_status": "verified",
        "reasoning_trace": ["corrective passed"],
    }

    # Build mock instances
    router_inst = MagicMock()
    router_inst.route.return_value = {
        "service_mode": routing_mode,
        "query_intent": "factual",
        "selected_databases": ["entry_logbook"],
        "extracted_entities": [],
        "retrieval_strategy": "HYBRID_3WAY",
    }

    planner_inst = MagicMock()
    planner_inst.plan.return_value = {
        "plan": "1. Retrieve\n2. Synthesize",
        "plan_steps": plan_steps if plan_steps is not None else [],
    }

    retrieval_inst = MagicMock()
    retrieval_inst.retrieve.return_value = {
        "retrieved_docs": docs,
        "retrieval_sources": ["vector"],
    }

    reasoning_inst = MagicMock()
    reasoning_inst.synthesize.return_value = synthesis

    corrective_inst = MagicMock()
    corrective_inst.evaluate.return_value = corrective

    tool_exec_inst = MagicMock()
    tool_exec_inst.execute.return_value = tool_executor_result if tool_executor_result is not None \
        else {"tool_results": [{"tool": "calc", "result": "13", "error": False}], "tool_calls": []}

    daily_inst = MagicMock()
    daily_inst.summarize.return_value = daily_summary_result if daily_summary_result is not None \
        else {"final_answer": "<daily summary>", "reasoning_trace": []}

    trouble_inst = MagicMock()
    trouble_inst.diagnose.return_value = troubleshooting_result if troubleshooting_result is not None \
        else {"final_answer": "<troubleshooting>", "reasoning_trace": []}

    studies_inst = MagicMock()
    studies_inst.assist.return_value = studies_result if studies_result is not None \
        else {"final_answer": "<studies copilot>", "reasoning_trace": []}

    # Patch the agent classes inside orchestrator module
    patches = [
        patch.object(orch_mod, "QueryRouter", return_value=router_inst),
        patch.object(orch_mod, "PlannerAgent", return_value=planner_inst),
        patch.object(orch_mod, "RetrievalAgent", return_value=retrieval_inst),
        patch.object(orch_mod, "ReasoningAgent", return_value=reasoning_inst),
        patch.object(orch_mod, "CorrectiveAgent", return_value=corrective_inst),
        patch.object(orch_mod, "ToolExecutor", return_value=tool_exec_inst),
        patch.object(orch_mod, "DailySummaryAgent", return_value=daily_inst),
        patch.object(orch_mod, "TroubleshootingAgent", return_value=trouble_inst),
        patch.object(orch_mod, "StudiesCopilotAgent", return_value=studies_inst),
    ]
    for p in patches:
        p.start()

    settings = _settings(**(settings_overrides or {}))
    components = _components(**(extra_components or {}))
    graph = orch_mod.create_orchestrator(settings, components)

    # Return a context that stops the patches when the test is done
    handles = SimpleNamespace(
        graph=graph,
        router=router_inst,
        planner=planner_inst,
        retrieval=retrieval_inst,
        reasoning=reasoning_inst,
        corrective=corrective_inst,
        tool_exec=tool_exec_inst,
        daily=daily_inst,
        trouble=trouble_inst,
        studies=studies_inst,
        _patches=patches,
    )
    return handles


def _stop(handles):
    for p in handles._patches:
        p.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Service dispatch — each of the 5 modes routes to the right node
# ─────────────────────────────────────────────────────────────────────────────
class TestServiceDispatch:
    @pytest.mark.parametrize("mode,expected_node", [
        ("naive_rag", "reasoning"),
        ("agentic_rag", "reasoning"),
        ("daily_summary", "daily"),
        ("troubleshooting", "trouble"),
        ("studies_copilot", "studies"),
    ])
    def test_dispatch_per_mode(self, mode, expected_node):
        h = _build_with_mocks(routing_mode=mode)
        try:
            state = _initial_state(service_mode=mode)
            result = h.graph.invoke(state, {"recursion_limit": 50})
        finally:
            _stop(h)

        # Whichever specialized node owns this mode must have been hit.
        assert result.get("final_answer", "") != ""
        if expected_node == "daily":
            h.daily.summarize.assert_called()
        elif expected_node == "trouble":
            h.trouble.diagnose.assert_called()
        elif expected_node == "studies":
            h.studies.assist.assert_called()
        else:
            h.reasoning.synthesize.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# Naive-RAG path
# ─────────────────────────────────────────────────────────────────────────────
class TestNaiveRAG:
    def test_single_pass_no_tools(self):
        h = _build_with_mocks(routing_mode="naive_rag")
        try:
            result = h.graph.invoke(_initial_state(service_mode="naive_rag"), {"recursion_limit": 50})
        finally:
            _stop(h)
        assert "100 mA" in result["final_answer"]
        h.retrieval.retrieve.assert_called()
        h.reasoning.synthesize.assert_called()
        # No tools emitted → tool_executor never invoked
        h.tool_exec.execute.assert_not_called()

    def test_resynthesizes_after_tool_call(self):
        # First synthesize call returns a tool_call; second returns the final answer.
        synth_returns = iter([
            {"draft_answer": "Looking up.", "tool_calls": [{"tool": "calc", "args": "1+1"}],
             "tool_results": [], "citations": [], "reasoning_trace": []},
            {"draft_answer": "Final answer is 13.", "tool_calls": [],
             "tool_results": [], "citations": [], "reasoning_trace": []},
        ])
        h = _build_with_mocks(routing_mode="naive_rag")
        h.reasoning.synthesize.side_effect = lambda state: next(synth_returns)
        try:
            result = h.graph.invoke(_initial_state(service_mode="naive_rag"), {"recursion_limit": 50})
        finally:
            _stop(h)
        assert "Final answer" in result["final_answer"]
        assert h.reasoning.synthesize.call_count == 2
        h.tool_exec.execute.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Agentic-RAG corrective loop
# ─────────────────────────────────────────────────────────────────────────────
class TestAgenticRAGCorrectiveLoop:
    def test_corrective_triggers_reretrieval(self):
        # First eval needs_correction=True; second eval passes.
        corrective_returns = iter([
            {"confidence_score": 0.2, "needs_correction": True, "correction_round": 1,
             "enhanced_query": "refined", "retrieval_strategy": "VECTOR_ONLY",
             "reasoning_trace": [], "correction_issues": ["low score"], "quality_status": "needs_correction",
             "query_variants": []},
            {"final_answer": "Verified answer.", "confidence_score": 0.9,
             "needs_correction": False, "quality_status": "verified", "reasoning_trace": []},
        ])
        h = _build_with_mocks(routing_mode="agentic_rag")
        h.corrective.evaluate.side_effect = lambda state: next(corrective_returns)
        try:
            result = h.graph.invoke(_initial_state(service_mode="agentic_rag"), {"recursion_limit": 50})
        finally:
            _stop(h)
        assert "Verified answer" in result["final_answer"]
        # planner ran twice (initial + after correction triggered re-plan)
        assert h.planner.plan.call_count == 2
        assert h.corrective.evaluate.call_count == 2

    def test_max_correction_rounds_force_accepts(self):
        """When corrective always returns needs_correction=True, the
        correction_router must exit to format_node once correction_round
        reaches max_correction_rounds — preventing an infinite loop."""
        h = _build_with_mocks(
            routing_mode="agentic_rag",
            settings_overrides={"max_correction_rounds": 1},
        )
        h.corrective.evaluate.return_value = {
            "confidence_score": 0.1,
            "needs_correction": True,
            "correction_round": 1,
            "enhanced_query": "refined",
            "retrieval_strategy": "HYBRID_3WAY",
            "reasoning_trace": [],
            "correction_issues": ["low score"],
            "quality_status": "needs_correction",
            "query_variants": [],
        }
        try:
            result = h.graph.invoke(_initial_state(service_mode="agentic_rag"), {"recursion_limit": 50})
        finally:
            _stop(h)
        # Loop must terminate (no GraphRecursionError) and produce a final_answer
        assert result["final_answer"], "expected a final answer after max-rounds force-exit"
        # corrective evaluated exactly once because router exits when round >= max
        assert h.corrective.evaluate.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tool-chain safety cap (R4)
# ─────────────────────────────────────────────────────────────────────────────
class TestToolChainSafetyCap:
    def test_check_tools_routes_to_tool_node_when_calls_present(self):
        # Reasoning emits one tool call; tool_executor runs; reasoning emits no more → corrective
        synth_returns = iter([
            {"draft_answer": "Need tool.", "tool_calls": [{"tool": "calc", "args": "x"}],
             "tool_results": [], "citations": [], "reasoning_trace": []},
            {"draft_answer": "Done with tool.", "tool_calls": [],
             "tool_results": [], "citations": [], "reasoning_trace": []},
        ])
        h = _build_with_mocks(routing_mode="agentic_rag")
        h.reasoning.synthesize.side_effect = lambda state: next(synth_returns)
        try:
            h.graph.invoke(_initial_state(service_mode="agentic_rag"), {"recursion_limit": 50})
        finally:
            _stop(h)
        h.tool_exec.execute.assert_called_once()

    def test_safety_cap_forces_termination(self):
        """When reasoning keeps emitting tool calls, check_tools must
        terminate the loop once tool_results count >= safety_cap."""
        h = _build_with_mocks(
            routing_mode="agentic_rag",
            settings_overrides={"tool_chain_safety_cap": 2},
        )
        # Reasoning ALWAYS emits a tool call; do NOT include `tool_results`
        # in the return so LangGraph's TypedDict merge preserves whatever
        # tool_executor accumulated (matches real ReasoningAgent.synthesize).
        h.reasoning.synthesize.return_value = {
            "draft_answer": "Calling tool.",
            "tool_calls": [{"tool": "calc", "args": "x"}],
            "citations": [],
            "reasoning_trace": [],
        }
        # tool_executor returns ONE result per call and clears tool_calls.
        # The state's tool_results list grows because LangGraph merges by key.
        call_counter = {"n": 0}
        def _exec(state):
            call_counter["n"] += 1
            prior = state.get("tool_results") or []
            return {
                "tool_results": prior + [{"tool": "calc", "result": f"ok{call_counter['n']}", "error": False}],
                "tool_calls": [],
            }
        h.tool_exec.execute.side_effect = _exec
        try:
            h.graph.invoke(_initial_state(service_mode="agentic_rag"), {"recursion_limit": 50})
        finally:
            _stop(h)
        # Cap is 2 → after 2 successful tool runs, check_tools routes to corrective.
        # Allow tool_exec to be called at most safety_cap times.
        assert call_counter["n"] <= 2, f"tool_executor called {call_counter['n']}x, cap was 2"
        # And the corrective node was eventually reached (loop did terminate)
        h.corrective.evaluate.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# format_response — TOOL_CALL artifact stripping + memory writeback
# ─────────────────────────────────────────────────────────────────────────────
class TestFormatResponse:
    def test_strips_tool_call_lines_from_final_answer(self):
        synth = {
            "draft_answer": "Here's the result.\nTOOL_CALL: calculator(2+2)\nThe answer is 4.",
            "tool_calls": [], "tool_results": [], "citations": [], "reasoning_trace": [],
        }
        h = _build_with_mocks(routing_mode="naive_rag", synthesis_result=synth)
        try:
            result = h.graph.invoke(_initial_state(service_mode="naive_rag"), {"recursion_limit": 50})
        finally:
            _stop(h)
        assert "TOOL_CALL:" not in result["final_answer"]
        assert "answer is 4" in result["final_answer"]

    def test_writes_to_memory_when_manager_present(self):
        memory = MagicMock()
        h = _build_with_mocks(
            routing_mode="naive_rag",
            extra_components={"memory_manager": memory},
        )
        try:
            h.graph.invoke(_initial_state(service_mode="naive_rag"), {"recursion_limit": 50})
        finally:
            _stop(h)
        memory.add_turn.assert_any_call("test-session", "user", "What is the beam current?")
        # add_turn called twice: once for user, once for assistant
        assert memory.add_turn.call_count >= 2
