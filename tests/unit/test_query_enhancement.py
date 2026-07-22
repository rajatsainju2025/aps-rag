"""
Unit tests for QueryEnhancer (fallback pipeline).
Tests the fallback classes since entry_rag_app imports may not be available.
No live LLM or external services required.
"""
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestQueryEnhancer:
    @pytest.fixture
    def enhancer(self):
        from src.retrieval.query_enhancement import QueryEnhancer
        return QueryEnhancer()

    def test_enhance_returns_required_keys(self, enhancer):
        result = enhancer.enhance("What caused the beam loss?")
        required = {"original_query", "enhanced_query", "temporal_context", "has_temporal"}
        assert required.issubset(set(result.keys()))

    def test_original_query_preserved(self, enhancer):
        result = enhancer.enhance("Test query here")
        assert result["original_query"] == "Test query here"

    def test_enhanced_query_not_empty(self, enhancer):
        result = enhancer.enhance("What happened yesterday?")
        assert len(result["enhanced_query"]) > 0

    def test_empty_query(self, enhancer):
        result = enhancer.enhance("")
        assert result["original_query"] == ""
        assert result["enhanced_query"] == ""

    def test_has_temporal_is_bool(self, enhancer):
        result = enhancer.enhance("What happened last week?")
        assert isinstance(result["has_temporal"], bool)


class TestFallbackClasses:
    """Test the fallback classes that are used when entry_rag_app imports fail."""

    def test_fallback_temporal_processor(self):
        from src.retrieval.query_enhancement import TemporalProcessor
        tp = TemporalProcessor()
        # Fallback returns None
        result = tp.parse_temporal_context("What happened yesterday?")
        # Either returns a dict (real import) or None (fallback)
        assert result is None or isinstance(result, dict)

    def test_fallback_acronym_resolver(self):
        from src.retrieval.query_enhancement import AcronymResolver
        ar = AcronymResolver()
        result = ar.resolve("SR RF cavity fault")
        # Fallback returns the same text; real import may expand acronyms
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_query_intent_constants(self):
        from src.retrieval.query_enhancement import QueryIntent
        assert QueryIntent.FACTUAL == "factual"
        assert QueryIntent.TEMPORAL == "temporal"
        assert QueryIntent.TROUBLESHOOTING == "troubleshooting"
        assert QueryIntent.GENERAL == "general"


class TestEnhancerExceptionHandling:
    def test_acronym_resolver_exception_falls_back(self):
        from src.retrieval.query_enhancement import QueryEnhancer
        enhancer = QueryEnhancer()
        # Monkey-patch to force exception
        original_resolve = enhancer.acronym_resolver.resolve
        enhancer.acronym_resolver.resolve = lambda q: (_ for _ in ()).throw(ValueError("bad"))
        result = enhancer.enhance("test query")
        # Should fall back to original query
        assert result["enhanced_query"] == "test query"

    def test_temporal_processor_exception_falls_back(self):
        from src.retrieval.query_enhancement import QueryEnhancer
        enhancer = QueryEnhancer()
        enhancer.temporal_processor.parse_temporal_context = lambda q: (_ for _ in ()).throw(ValueError("bad"))
        result = enhancer.enhance("What happened last week?")
        assert result["temporal_context"] is None
        assert result["has_temporal"] is False
