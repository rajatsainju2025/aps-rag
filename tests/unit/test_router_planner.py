"""
Unit tests for QueryRouter and PlannerAgent.
Uses mocked LLMs — no live API calls.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm_json(payload: dict) -> MagicMock:
    mock = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps(payload)
    mock.invoke.return_value = resp
    return mock


def _mock_llm_text(text: str) -> MagicMock:
    mock = MagicMock()
    resp = MagicMock()
    resp.content = text
    mock.invoke.return_value = resp
    return mock


def _base_state(**overrides):
    state = {
        "original_query": "What caused the beam loss in sector 35?",
        "enhanced_query": "What caused the beam loss in sector 35?",
        "query_intent": "",
        "selected_databases": [],
        "service_mode": "rag",
        "correction_round": 0,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# QueryRouter._parse_json
# ---------------------------------------------------------------------------

class TestRouterParseJson:
    @pytest.fixture
    def router(self):
        from src.agents.router import QueryRouter
        return QueryRouter(llm=MagicMock())

    def test_clean_json(self, router):
        raw = '{"intent": "factual", "collections": ["entry_logbook"]}'
        result = router._parse_json(raw)
        assert result["intent"] == "factual"

    def test_markdown_fenced_json(self, router):
        raw = '```json\n{"intent": "temporal"}\n```'
        result = router._parse_json(raw)
        assert result["intent"] == "temporal"

    def test_json_with_preamble(self, router):
        raw = 'Here is my response:\n{"intent": "general"}\nDone.'
        result = router._parse_json(raw)
        assert result["intent"] == "general"

    def test_no_json_returns_empty(self, router):
        assert router._parse_json("not json at all") == {}


# ---------------------------------------------------------------------------
# QueryRouter.route
# ---------------------------------------------------------------------------

class TestRouterRoute:
    def test_successful_routing(self):
        from src.agents.router import QueryRouter
        llm = _mock_llm_json({
            "intent": "troubleshooting",
            "collections": ["entry_logbook"],
            "retrieval_strategy": "HYBRID_3WAY",
            "entities": ["sector 35"],
            "reasoning": "Fault diagnosis query",
        })
        router = QueryRouter(llm=llm)
        result = router.route(_base_state())
        assert result["query_intent"] == "troubleshooting"
        assert result["selected_databases"] == ["entry_logbook"]
        assert result["retrieval_strategy"] == "HYBRID_3WAY"

    def test_invalid_collection_filtered(self):
        from src.agents.router import QueryRouter
        llm = _mock_llm_json({
            "intent": "factual",
            "collections": ["entry_logbook", "fake_collection"],
            "retrieval_strategy": "VECTOR_ONLY",
            "entities": [],
        })
        router = QueryRouter(llm=llm)
        result = router.route(_base_state())
        assert "fake_collection" not in result["selected_databases"]
        assert "entry_logbook" in result["selected_databases"]

    def test_all_invalid_collections_fallback_to_all_databases(self):
        from src.agents.router import QueryRouter
        llm = _mock_llm_json({
            "intent": "general",
            "collections": ["fake_1", "fake_2"],
            "retrieval_strategy": "HYBRID_3WAY",
            "entities": [],
        })
        router = QueryRouter(llm=llm)
        result = router.route(_base_state())
        assert result["selected_databases"] == ["all_databases"]

    def test_llm_failure_returns_defaults(self):
        from src.agents.router import QueryRouter
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("API timeout")
        router = QueryRouter(llm=llm)
        result = router.route(_base_state())
        assert result["query_intent"] == "general"
        assert result["selected_databases"] == ["all_databases"]
        assert result["retrieval_strategy"] == "HYBRID_3WAY"

    def test_uses_enhanced_query_over_original(self):
        from src.agents.router import QueryRouter
        llm = _mock_llm_json({
            "intent": "temporal",
            "collections": ["entry_logbook"],
            "retrieval_strategy": "HYBRID_3WAY",
            "entities": [],
        })
        router = QueryRouter(llm=llm)
        state = _base_state(enhanced_query="Enhanced: beam loss sector 35 last week")
        router.route(state)
        # Verify the LLM received the enhanced query
        call_args = llm.invoke.call_args[0][0]
        assert "Enhanced: beam loss" in call_args[1].content

    def test_missing_enhanced_query_uses_original(self):
        from src.agents.router import QueryRouter
        llm = _mock_llm_json({
            "intent": "general",
            "collections": ["all_databases"],
            "retrieval_strategy": "HYBRID_3WAY",
            "entities": [],
        })
        router = QueryRouter(llm=llm)
        state = _base_state(enhanced_query="")
        router.route(state)
        call_args = llm.invoke.call_args[0][0]
        assert "beam loss" in call_args[1].content

    def test_all_databases_is_valid_collection(self):
        from src.agents.router import QueryRouter
        llm = _mock_llm_json({
            "intent": "general",
            "collections": ["all_databases"],
            "retrieval_strategy": "HYBRID_3WAY",
            "entities": [],
        })
        router = QueryRouter(llm=llm)
        result = router.route(_base_state())
        assert "all_databases" in result["selected_databases"]


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class TestPlannerAgent:
    def test_successful_plan(self):
        from src.agents.planner import PlannerAgent
        llm = _mock_llm_text(
            "1. Search entry_logbook for beam loss events\n"
            "2. Use [graph_search] to find related faults\n"
            "3. Synthesize answer"
        )
        planner = PlannerAgent(llm=llm)
        result = planner.plan(_base_state())
        assert "plan" in result
        assert "beam loss" in result["plan"]
        assert "graph_search" in result["plan"]

    def test_llm_failure_returns_fallback_plan(self):
        from src.agents.planner import PlannerAgent
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        planner = PlannerAgent(llm=llm)
        state = _base_state(selected_databases=["entry_logbook"])
        result = planner.plan(state)
        assert "plan" in result
        assert "Search" in result["plan"]
        assert "Synthesize" in result["plan"]

    def test_correction_round_adds_note(self):
        from src.agents.planner import PlannerAgent
        llm = _mock_llm_text("1. Broaden search\n2. Synthesize")
        planner = PlannerAgent(llm=llm)
        state = _base_state(correction_round=2)
        planner.plan(state)
        # First invocation is the plan generation; later calls may be JSON-repair attempts.
        call_args = llm.invoke.call_args_list[0][0][0]
        prompt_text = call_args[1].content
        assert "CORRECTION ROUND 2" in prompt_text.upper()

    def test_no_correction_note_on_round_zero(self):
        from src.agents.planner import PlannerAgent
        llm = _mock_llm_text("1. Search\n2. Synthesize")
        planner = PlannerAgent(llm=llm)
        state = _base_state(correction_round=0)
        planner.plan(state)
        call_args = llm.invoke.call_args_list[0][0][0]
        prompt_text = call_args[1].content
        assert "correction round" not in prompt_text

    def test_plan_includes_query_and_intent(self):
        from src.agents.planner import PlannerAgent
        llm = _mock_llm_text("1. Search\n2. Synthesize")
        planner = PlannerAgent(llm=llm)
        state = _base_state(query_intent="troubleshooting")
        planner.plan(state)
        call_args = llm.invoke.call_args_list[0][0][0]
        prompt_text = call_args[1].content
        assert "troubleshooting" in prompt_text
        assert "beam loss" in prompt_text
