"""Stage A unit tests — naive_rag redirect to RAGService.

Stage A of the MCP-everywhere plan routes ``service_mode="naive_rag"`` through
``RAGService`` (orchestrator-backed, has the tool-execution loop) instead of
the legacy ``NaiveRAGService`` (no tool calling). The contract:

    1. ServiceSelector("naive_rag") returns a RAGService instance with
       ``mode_label="naive_rag"``.
    2. RAGService.get_service_mode() reflects mode_label, not a hardcoded "rag".
    3. RAGService.query() reports ``service_mode=mode_label`` in the response.
    4. The state dict passed to the orchestrator carries
       ``service_mode="rag"`` (NOT mode_label) so that ``route_to_service``
       does NOT auto-upgrade naive_rag to agentic_rag — the explicit "rag"
       value triggers the user_mode != "naive_rag" branch in
       orchestrator.route_to_service, locking the request to naive_rag_node.
    5. service_mode="rag" still works (backward compat) and now also goes
       through RAGService.
    6. NaiveRAGService class still exists (used by legacy harness/tests) but
       is no longer reachable via ServiceSelector.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.services.rag_service import RAGService  # noqa: E402
from src.ui.service_selector import ServiceSelector  # noqa: E402


# ---------------------------------------------------------------------------
# 1. ServiceSelector routing
# ---------------------------------------------------------------------------
def test_service_selector_routes_naive_rag_to_rag_service():
    """naive_rag mode must instantiate RAGService, not NaiveRAGService."""
    selector = ServiceSelector()
    mock_settings = MagicMock()
    mock_components = {"synthesis_llm": MagicMock(), "tools": {}}
    svc = selector.get_service("naive_rag", mock_settings, mock_components)
    assert isinstance(svc, RAGService), (
        f"naive_rag should route to RAGService, got {type(svc).__name__}"
    )


def test_service_selector_routes_naive_rag_with_correct_mode_label():
    """The RAGService instance must remember it was created for naive_rag."""
    selector = ServiceSelector()
    mock_settings = MagicMock()
    mock_components = {"synthesis_llm": MagicMock(), "tools": {}}
    svc = selector.get_service("naive_rag", mock_settings, mock_components)
    assert svc.get_service_mode() == "naive_rag"


def test_service_selector_rag_mode_still_works():
    """Backward compat: explicit service_mode='rag' still returns RAGService."""
    selector = ServiceSelector()
    mock_settings = MagicMock()
    mock_components = {"synthesis_llm": MagicMock(), "tools": {}}
    svc = selector.get_service("rag", mock_settings, mock_components)
    assert isinstance(svc, RAGService)
    assert svc.get_service_mode() == "rag"


def test_service_selector_naive_rag_and_rag_get_separate_instances():
    """Cache key is the mode string — naive_rag and rag get separate instances
    so their mode_label labels don't bleed into each other's responses."""
    selector = ServiceSelector()
    mock_settings = MagicMock()
    mock_components = {"synthesis_llm": MagicMock(), "tools": {}}
    naive_svc = selector.get_service("naive_rag", mock_settings, mock_components)
    rag_svc = selector.get_service("rag", mock_settings, mock_components)
    assert naive_svc is not rag_svc
    assert naive_svc.get_service_mode() == "naive_rag"
    assert rag_svc.get_service_mode() == "rag"


def test_service_selector_does_not_import_naive_rag_service_for_naive_rag():
    """The legacy NaiveRAGService must not be instantiated for naive_rag mode."""
    selector = ServiceSelector()
    mock_settings = MagicMock()
    mock_components = {"synthesis_llm": MagicMock(), "tools": {}}
    # If NaiveRAGService were imported, this patch path would be hit
    with patch("src.services.naive_rag_service.NaiveRAGService") as legacy:
        selector.get_service("naive_rag", mock_settings, mock_components)
        legacy.assert_not_called()


# ---------------------------------------------------------------------------
# 2. RAGService mode_label semantics
# ---------------------------------------------------------------------------
def test_rag_service_default_mode_label_is_rag():
    """Backward compat: omitted mode_label defaults to 'rag'."""
    svc = RAGService(MagicMock(), {"synthesis_llm": MagicMock()})
    assert svc.get_service_mode() == "rag"


def test_rag_service_mode_label_propagates_to_health_check():
    svc_naive = RAGService(MagicMock(), {"synthesis_llm": MagicMock()},
                           mode_label="naive_rag")
    svc_rag = RAGService(MagicMock(), {"synthesis_llm": MagicMock()},
                        mode_label="rag")
    assert svc_naive.health_check()["service_mode"] == "naive_rag"
    assert svc_rag.health_check()["service_mode"] == "rag"


# ---------------------------------------------------------------------------
# 3. State + response semantics (the router-lock contract)
# ---------------------------------------------------------------------------
def _make_rag_service_with_mocked_orchestrator(mode_label: str):
    """Construct a RAGService whose orchestrator.invoke is mockable."""
    svc = RAGService(MagicMock(), {"synthesis_llm": MagicMock()},
                     mode_label=mode_label)
    mock_orch = MagicMock()
    mock_orch.invoke.return_value = {
        "final_answer": "stub answer",
        "citations": [],
        "retrieved_docs": [],
    }
    svc._orchestrator = mock_orch  # bypass create_orchestrator
    return svc, mock_orch


def test_query_passes_state_service_mode_rag_for_router_lock():
    """The state dict passed to the orchestrator must carry service_mode='rag'
    (NOT mode_label). This is what makes route_to_service treat it as 'user
    explicitly chose a non-default pipeline' and skip auto-upgrade to
    agentic_rag. If we passed mode_label='naive_rag' here, the router would
    be free to upgrade naive queries to agentic.
    """
    svc, mock_orch = _make_rag_service_with_mocked_orchestrator("naive_rag")
    svc.query("how many work requests in 2025?", session_id="t-1")
    state_passed = mock_orch.invoke.call_args.args[0]
    assert state_passed["service_mode"] == "rag", (
        "state.service_mode must be 'rag' (router lock) regardless of mode_label"
    )


def test_query_response_reports_mode_label():
    """The user-visible response.service_mode reflects the caller's mode."""
    svc, _ = _make_rag_service_with_mocked_orchestrator("naive_rag")
    out = svc.query("anything", session_id="t-2")
    assert out["service_mode"] == "naive_rag"


def test_query_response_for_rag_label():
    svc, _ = _make_rag_service_with_mocked_orchestrator("rag")
    out = svc.query("anything", session_id="t-3")
    assert out["service_mode"] == "rag"


def test_query_response_includes_reasoning_trace():
    """Stage B+ contract: RAGService must surface the orchestrator's
    reasoning_trace in the response so observers (counting verifier's
    tool-detection, OpenWebUI's <think> rendering) can see what happened.
    Was previously dropped — only AgenticRAGService surfaced it.
    """
    svc, mock_orch = _make_rag_service_with_mocked_orchestrator("naive_rag")
    mock_orch.invoke.return_value = {
        "final_answer": "stub",
        "citations": [],
        "retrieved_docs": [],
        "reasoning_trace": ["Synthesizing answer from 5 documents",
                            "Tool data_analysis executed (3500 ms)"],
    }
    out = svc.query("how many?", session_id="t-trace")
    assert "reasoning_trace" in out
    assert any("data_analysis" in line for line in out["reasoning_trace"]), (
        "Tool execution lines must reach the response trace"
    )


def test_query_response_reasoning_trace_defaults_to_empty_list():
    """If the orchestrator didn't populate a trace, the response field is
    [] (not None) so downstream consumers can iterate without nil checks."""
    svc, mock_orch = _make_rag_service_with_mocked_orchestrator("naive_rag")
    mock_orch.invoke.return_value = {
        "final_answer": "stub", "citations": [], "retrieved_docs": [],
        # no reasoning_trace key
    }
    out = svc.query("q", session_id="t-trace-empty")
    assert out["reasoning_trace"] == []


def test_query_state_includes_required_orchestrator_fields():
    """Sanity: regression guard for the state dict shape (a previous bug
    saw fields silently dropped here, breaking downstream nodes)."""
    svc, mock_orch = _make_rag_service_with_mocked_orchestrator("naive_rag")
    svc.query("q", session_id="t-4", selected_databases=["work_requests"])
    state = mock_orch.invoke.call_args.args[0]
    required = {
        "messages", "original_query", "enhanced_query", "session_id",
        "service_mode", "selected_databases", "query_intent",
        "reasoning_trace", "retrieved_docs", "draft_answer",
        "final_answer", "citations", "tool_calls", "tool_results",
    }
    missing = required - set(state.keys())
    assert not missing, f"state missing required fields: {missing}"
    # And selected_databases should pass through
    assert state["selected_databases"] == ["work_requests"]


# ---------------------------------------------------------------------------
# 4. NaiveRAGService still exists (legacy harness needs it) but is unreachable
#    via ServiceSelector
# ---------------------------------------------------------------------------
def test_naive_rag_service_class_still_importable_for_legacy_users():
    """evaluation/naive_latency_harness.py + tests/unit/test_naive_rag_service.py
    still import NaiveRAGService directly. Don't break those just yet —
    Stage E will delete the class after a stability period."""
    from src.services.naive_rag_service import NaiveRAGService  # noqa: F401


def test_service_selector_module_does_not_import_or_instantiate_naive_rag_service():
    """Structural guard: scan service_selector.py for any IMPORT or
    INSTANTIATION of NaiveRAGService — neither should appear after Stage A.
    A docstring/comment mention is fine (we may reference the legacy class
    in a 'why this exists' comment); we're guarding against dead code paths,
    not text mentions.
    """
    import inspect
    import re
    import src.ui.service_selector as ss_module
    src_text = inspect.getsource(ss_module)
    # Strip comments and docstrings before searching, so a comment mention
    # like '# legacy NaiveRAGService' doesn't trigger a false positive.
    code_only = re.sub(r"#[^\n]*", "", src_text)
    code_only = re.sub(r'"""[\s\S]*?"""', "", code_only)
    code_only = re.sub(r"'''[\s\S]*?'''", "", code_only)
    assert "NaiveRAGService" not in code_only, (
        "service_selector.py should no longer import or instantiate "
        "NaiveRAGService after Stage A — naive_rag now goes through RAGService."
    )
