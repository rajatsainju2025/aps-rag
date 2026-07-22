"""Unit tests for SemanticRouter — schema-aware intent classification."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.semantic_router import SemanticRouter


@pytest.fixture
def llm():
    return MagicMock()


@pytest.fixture
def router(llm):
    return SemanticRouter(llm=llm)


def _wrap(payload: dict) -> MagicMock:
    return MagicMock(content=json.dumps(payload))


# ─────────────────────────────────────────────────────────────────────────────
# Routing decision parsing
# ─────────────────────────────────────────────────────────────────────────────
class TestRoutingTypes:
    def test_single_database_routing(self, router, llm):
        llm.invoke.return_value = _wrap({
            "routing_type": "single",
            "target_database": "entry_logbook",
            "confidence_score": 0.9,
            "reasoning": "Query matches the logbook",
            "extracted_entities": ["beam current"],
            "suggested_retrieval_strategy": "VECTOR_ONLY",
        })
        out = router.route("what is the beam current trend?")
        assert out["routing_type"] == "single"
        assert out["target_database"] == "entry_logbook"
        assert out["confidence_score"] == 0.9

    def test_multi_database_routing(self, router, llm):
        llm.invoke.return_value = _wrap({
            "routing_type": "multi",
            "targets": [
                {"target_database": "entry_logbook", "weight": 0.7},
                {"target_database": "icms_collection", "weight": 0.3},
            ],
            "overall_confidence": 0.8,
            "reasoning": "Spans logbook + tech docs",
            "extracted_entities": [],
            "suggested_retrieval_strategy": "HYBRID_3WAY",
        })
        out = router.route("what is documented about the kicker?")
        assert out["routing_type"] == "multi"
        assert len(out["targets"]) == 2

    def test_fallback_routing(self, router, llm):
        """If LLM raises, the router must return a safe fallback."""
        llm.invoke.side_effect = RuntimeError("LLM down")
        out = router.route("vague query")
        assert out["routing_type"] == "fallback"
        assert out["target_database"] == "all_databases"
        assert out["suggested_retrieval_strategy"] == "HYBRID_3WAY"


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing robustness
# ─────────────────────────────────────────────────────────────────────────────
class TestJSONParsing:
    def test_strips_routing_analysis_block(self, router, llm):
        raw = (
            "<routing_analysis>\n"
            "I think this is about beam dynamics, so the system is best.\n"
            "</routing_analysis>\n"
            '{"routing_type": "single", "target_database": "entry_logbook", '
            '"confidence_score": 0.9, "extracted_entities": []}'
        )
        llm.invoke.return_value = MagicMock(content=raw)
        out = router.route("question")
        assert out["target_database"] == "entry_logbook"

    def test_strips_markdown_code_fences(self, router, llm):
        raw = '```json\n{"routing_type": "single", "target_database": "icms_collection"}\n```'
        llm.invoke.return_value = MagicMock(content=raw)
        out = router.route("question")
        assert out["target_database"] == "icms_collection"

    def test_invalid_json_returns_fallback_dict(self, router, llm):
        llm.invoke.return_value = MagicMock(content="not even close to json")
        out = router.route("question")
        # Parser returns a dict with whatever defaults — must not raise
        assert isinstance(out, dict)


# ─────────────────────────────────────────────────────────────────────────────
# route_to_state translates to AgentState shape
# ─────────────────────────────────────────────────────────────────────────────
class TestRouteToState:
    def test_single_target_collapses_to_list(self, router, llm):
        llm.invoke.return_value = _wrap({
            "routing_type": "single",
            "target_database": "entry_logbook",
            "confidence_score": 0.85,
            "extracted_entities": ["beam"],
            "suggested_retrieval_strategy": "VECTOR_ONLY",
        })
        state = router.route_to_state("query")
        assert state["selected_databases"] == ["entry_logbook"]
        assert state["retrieval_strategy"] == "VECTOR_ONLY"
        assert state["extracted_entities"] == ["beam"]

    def test_multi_targets_become_list(self, router, llm):
        llm.invoke.return_value = _wrap({
            "routing_type": "multi",
            "targets": [
                {"target_database": "entry_logbook"},
                {"target_database": "icms_collection"},
            ],
            "overall_confidence": 0.7,
            "extracted_entities": [],
        })
        state = router.route_to_state("query")
        assert set(state["selected_databases"]) == {"entry_logbook", "icms_collection"}

    def test_fallback_yields_all_databases(self, router, llm):
        llm.invoke.side_effect = RuntimeError("LLM crashed")
        state = router.route_to_state("query")
        assert state["selected_databases"] == ["all_databases"]
        assert state["retrieval_strategy"] == "HYBRID_3WAY"


# ─────────────────────────────────────────────────────────────────────────────
# Schema injection
# ─────────────────────────────────────────────────────────────────────────────
class TestSchemaInjection:
    def test_default_schemas_used_when_not_provided(self, llm):
        from src.agents.semantic_router import DEFAULT_DATABASE_SCHEMAS
        r = SemanticRouter(llm=llm)
        assert r.schemas == DEFAULT_DATABASE_SCHEMAS

    def test_custom_schemas_override_default(self, llm):
        custom = "entry_logbook: my custom description"
        r = SemanticRouter(llm=llm, database_schemas=custom)
        assert r.schemas == custom

    def test_schemas_passed_to_llm_prompt(self, llm):
        custom = "entry_logbook: SUPER CUSTOM DESCRIPTION ABCDEF"
        r = SemanticRouter(llm=llm, database_schemas=custom)
        llm.invoke.return_value = _wrap({
            "routing_type": "single", "target_database": "entry_logbook",
            "extracted_entities": [],
        })
        r.route("query")
        # The system message should contain our custom schema string
        sent = llm.invoke.call_args[0][0]
        text = "\n".join(getattr(m, "content", "") for m in sent)
        assert "SUPER CUSTOM DESCRIPTION" in text
