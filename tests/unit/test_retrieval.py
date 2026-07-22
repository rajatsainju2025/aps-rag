"""
Unit tests for the retrieval module.
Covers: AdaptiveHybridFusion (weights, RRF scores, dedup, cross-collection
        normalization) and GraphProximityReranker (score ordering, graph-
        proximity bonus, no-model passthrough).
No live Qdrant / Elasticsearch / Neo4j required.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from langchain_core.documents import Document


def _doc(content: str, doc_id: str = "", collection: str = "entry_logbook",
         score: float = 1.0) -> Document:
    meta = {"collection": collection, "score": score}
    if doc_id:
        meta["id"] = doc_id
    return Document(page_content=content, metadata=meta)


# ---------------------------------------------------------------------------
# AdaptiveHybridFusion
# ---------------------------------------------------------------------------

class TestAdaptiveWeights:
    def test_default_weights_for_unknown_intent(self):
        from src.retrieval.hybrid_fusion import ADAPTIVE_WEIGHTS, DEFAULT_WEIGHTS
        assert "factual" in ADAPTIVE_WEIGHTS
        assert "temporal" in ADAPTIVE_WEIGHTS
        assert DEFAULT_WEIGHTS == (0.4, 0.3, 0.3)

    def test_factual_intent_boosts_graph(self):
        from src.retrieval.hybrid_fusion import ADAPTIVE_WEIGHTS
        v, b, g = ADAPTIVE_WEIGHTS["factual"]
        assert g > v, "Graph weight should dominate for factual queries"
        assert g > b

    def test_temporal_intent_boosts_bm25(self):
        from src.retrieval.hybrid_fusion import ADAPTIVE_WEIGHTS
        v, b, g = ADAPTIVE_WEIGHTS["temporal"]
        assert b > v, "BM25 weight should dominate for temporal queries"
        assert b > g

    def test_mathematical_intent_boosts_vector(self):
        from src.retrieval.hybrid_fusion import ADAPTIVE_WEIGHTS
        v, b, g = ADAPTIVE_WEIGHTS["mathematical"]
        assert v > b
        assert v > g

    def test_all_weight_triples_sum_to_one(self):
        from src.retrieval.hybrid_fusion import ADAPTIVE_WEIGHTS, DEFAULT_WEIGHTS
        for intent, (v, b, g) in ADAPTIVE_WEIGHTS.items():
            total = v + b + g
            assert abs(total - 1.0) < 1e-9, f"Weights for '{intent}' sum to {total}"
        assert abs(sum(DEFAULT_WEIGHTS) - 1.0) < 1e-9


class TestHybridFusionRRF:
    @pytest.fixture
    def fusion(self):
        from src.retrieval.hybrid_fusion import AdaptiveHybridFusion
        return AdaptiveHybridFusion(rrf_k=60)

    def test_empty_inputs_return_empty(self, fusion):
        result = fusion.fuse([], [], [], query_intent="general")
        assert result == []

    def test_single_list_returns_all_docs(self, fusion):
        docs = [_doc("alpha", "a"), _doc("beta", "b")]
        result = fusion.fuse(docs, [], [], query_intent="general")
        assert len(result) == 2

    def test_dedup_same_doc_id(self, fusion):
        doc_v = _doc("shared content", "shared_id")
        doc_b = _doc("shared content", "shared_id")
        result = fusion.fuse([doc_v], [doc_b], [], query_intent="general")
        assert len(result) == 1, "Duplicate doc IDs should be deduplicated"

    def test_rrf_score_is_weighted(self, fusion):
        # With factual weights (graph=0.5), graph rank-1 doc should outscore vector rank-1
        v_doc = _doc("vector result", "v1")
        g_doc = _doc("graph result", "g1")
        result = fusion.fuse([v_doc], [], [g_doc], query_intent="factual")
        scores = {d.metadata["id"]: d.metadata["rrf_score"] for d in result}
        # Both at rank 1, so score = weight * (1/61); graph should be higher
        assert scores["g1"] > scores["v1"]

    def test_higher_rank_docs_score_higher(self, fusion):
        docs = [_doc(f"doc {i}", f"id_{i}") for i in range(5)]
        result = fusion.fuse(docs, [], [], query_intent="general")
        scores = [d.metadata["rrf_score"] for d in result]
        # First in input list (rank 1) should get highest score
        assert scores == sorted(scores, reverse=True)

    def test_final_rank_metadata_set(self, fusion):
        docs = [_doc("alpha", "a"), _doc("beta", "b")]
        result = fusion.fuse(docs, [], [], query_intent="general")
        for i, doc in enumerate(result):
            assert doc.metadata["final_rank"] == i + 1

    def test_fusion_method_tag_in_metadata(self, fusion):
        docs = [_doc("content", "id1")]
        result = fusion.fuse(docs, [], [], query_intent="temporal")
        assert result[0].metadata["fusion_method"] == "adaptive_3way_rrf"
        assert result[0].metadata["query_intent"] == "temporal"

    def test_three_lists_all_included(self, fusion):
        v = [_doc("vector", "v1")]
        b = [_doc("bm25", "b1")]
        g = [_doc("graph", "g1")]
        result = fusion.fuse(v, b, g, query_intent="general")
        ids = {d.metadata["id"] for d in result}
        assert ids == {"v1", "b1", "g1"}


class TestCrossCollectionNormalization:
    @pytest.fixture
    def fusion(self):
        from src.retrieval.hybrid_fusion import AdaptiveHybridFusion
        return AdaptiveHybridFusion(rrf_k=60)

    def test_normalize_per_collection_bounds(self, fusion):
        docs = [
            _doc("low", "l", collection="entry_logbook", score=0.2),
            _doc("high", "h", collection="entry_logbook", score=0.8),
        ]
        normalized = fusion._normalize_per_collection(docs)
        scores = [d.metadata["normalized_score"] for d in normalized]
        assert max(scores) == pytest.approx(1.0)
        assert min(scores) == pytest.approx(0.0)

    def test_normalize_single_doc_per_collection_gets_one(self, fusion):
        docs = [_doc("only", "o", collection="entry_logbook", score=0.5)]
        normalized = fusion._normalize_per_collection(docs)
        assert normalized[0].metadata["normalized_score"] == pytest.approx(1.0)

    def test_normalize_separates_collections(self, fusion):
        docs = [
            _doc("a1", "a1", collection="entry_logbook", score=0.1),
            _doc("a2", "a2", collection="entry_logbook", score=0.9),
            _doc("b1", "b1", collection="icms_collection", score=0.5),
            _doc("b2", "b2", collection="icms_collection", score=0.5),
        ]
        normalized = fusion._normalize_per_collection(docs)
        norm_by_id = {d.metadata["id"]: d.metadata["normalized_score"] for d in normalized}
        # entry_logbook: low=0.0, high=1.0
        assert norm_by_id["a1"] == pytest.approx(0.0)
        assert norm_by_id["a2"] == pytest.approx(1.0)
        # icms: both same score → both 1.0
        assert norm_by_id["b1"] == pytest.approx(1.0)
        assert norm_by_id["b2"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GraphProximityReranker
# ---------------------------------------------------------------------------

class TestGraphProximityReranker:
    @pytest.fixture
    def reranker(self):
        from src.retrieval.reranker import GraphProximityReranker
        r = GraphProximityReranker(top_k=5, graph_proximity_bonus=0.1)
        # Force passthrough mode: _load_model returns None → uses rrf_score from metadata
        r._load_model = lambda: None
        return r

    def test_empty_documents_return_empty(self, reranker):
        result = reranker.rerank("query", [])
        assert result == []

    def test_returns_top_k_only(self, reranker):
        docs = [_doc(f"doc {i}", f"id_{i}") for i in range(10)]
        for doc in docs:
            doc.metadata["rrf_score"] = 0.5
        result = reranker.rerank("query", docs)
        assert len(result) <= 5

    def test_higher_rrf_score_ranked_first(self, reranker):
        low = _doc("low score", "low")
        low.metadata["rrf_score"] = 0.1
        high = _doc("high score", "high")
        high.metadata["rrf_score"] = 0.9
        result = reranker.rerank("query", [low, high])
        assert result[0].metadata["id"] == "high"

    def test_rerank_score_and_rank_set(self, reranker):
        docs = [_doc("content", "id1")]
        docs[0].metadata["rrf_score"] = 0.7
        result = reranker.rerank("query", docs)
        assert "rerank_score" in result[0].metadata
        assert result[0].metadata["rerank_rank"] == 1

    def test_graph_proximity_bonus_applied(self, reranker):
        doc = _doc("relevant doc", "doc_relevant")
        doc.metadata["rrf_score"] = 0.5

        mock_neo4j = MagicMock()
        mock_neo4j.fulltext_search.return_value = [{"id": "entity_1"}]
        mock_neo4j.get_documents_for_entity.return_value = [{"doc_id": "doc_relevant"}]

        result = reranker.rerank(
            "query",
            [doc],
            query_entities=["Doe"],
            neo4j_client=mock_neo4j,
        )
        # Bonus should be applied: 0.5 + 0.1 = 0.6
        assert result[0].metadata["rerank_score"] == pytest.approx(0.6)
        assert result[0].metadata.get("graph_proximity_bonus") == pytest.approx(0.1)

    def test_graph_proximity_bonus_not_applied_without_neo4j(self, reranker):
        doc = _doc("doc", "id1")
        doc.metadata["rrf_score"] = 0.5
        result = reranker.rerank("query", [doc], query_entities=["Doe"], neo4j_client=None)
        assert "graph_proximity_bonus" not in result[0].metadata

    def test_graph_proximity_bonus_not_applied_without_entities(self, reranker):
        doc = _doc("doc", "id1")
        doc.metadata["rrf_score"] = 0.5
        mock_neo4j = MagicMock()
        result = reranker.rerank("query", [doc], query_entities=None, neo4j_client=mock_neo4j)
        assert "graph_proximity_bonus" not in result[0].metadata
        mock_neo4j.fulltext_search.assert_not_called()

    def test_graph_proximity_error_handled_gracefully(self, reranker):
        doc = _doc("doc", "id1")
        doc.metadata["rrf_score"] = 0.5

        mock_neo4j = MagicMock()
        mock_neo4j.fulltext_search.side_effect = RuntimeError("Neo4j down")

        # Should not raise, just skip the bonus
        result = reranker.rerank("query", [doc], query_entities=["Entity"], neo4j_client=mock_neo4j)
        assert len(result) == 1
        assert result[0].metadata.get("graph_proximity_bonus", 0.0) == 0.0

    def test_doc_id_falls_back_to_hash(self, reranker):
        # Document with no 'id' or 'file_id' in metadata
        doc = Document(page_content="content without id", metadata={"rrf_score": 0.5})
        result = reranker.rerank("query", [doc])
        assert len(result) == 1
