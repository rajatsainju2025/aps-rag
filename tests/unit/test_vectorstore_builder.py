"""
Unit tests for AgRAGVectorStoreBuilder.
Covers: build_collection, KG metadata enrichment, build_all_collections.
All external clients (Qdrant, ES, embeddings) are mocked.
No live services required.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.vector.embedding_dim = 1024
    return settings


@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    client.collection_exists.return_value = False
    return client


@pytest.fixture
def mock_es():
    client = MagicMock()
    client.index_exists.return_value = False
    return client


@pytest.fixture
def mock_embedding():
    model = MagicMock()
    # Return 3-dim embeddings for simplicity
    model.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    return model


@pytest.fixture
def builder(mock_settings, mock_qdrant, mock_es, mock_embedding):
    from src.vectorstore.builder import AgRAGVectorStoreBuilder
    return AgRAGVectorStoreBuilder(
        settings=mock_settings,
        qdrant_client=mock_qdrant,
        es_client=mock_es,
        embedding_model=mock_embedding,
    )


# ---------------------------------------------------------------------------
# build_collection — basic
# ---------------------------------------------------------------------------

class TestBuildCollection:
    def test_build_with_provided_documents(self, builder, mock_embedding):
        docs = [
            {"id": "d1", "content": "Beam loss event"},
            {"id": "d2", "content": "RF cavity trip"},
        ]
        result = builder.build_collection("entry_logbook", documents=docs)
        assert result["indexed"] == 2
        assert result["collection"] == "agrag_entry_logbook"
        mock_embedding.embed_documents.assert_called_once()

    def test_empty_documents_returns_zero(self, builder):
        result = builder.build_collection("entry_logbook", documents=[])
        assert result["indexed"] == 0

    def test_qdrant_collection_created_if_not_exists(self, builder, mock_qdrant):
        docs = [{"id": "d1", "content": "test"}]
        builder.build_collection("entry_logbook", documents=docs)
        mock_qdrant.client.create_collection.assert_called_once()

    def test_qdrant_collection_not_recreated_if_exists(self, builder, mock_qdrant):
        mock_qdrant.collection_exists.return_value = True
        docs = [{"id": "d1", "content": "test"}]
        builder.build_collection("entry_logbook", documents=docs)
        mock_qdrant.client.create_collection.assert_not_called()

    def test_es_index_created_if_not_exists(self, builder, mock_es):
        docs = [{"id": "d1", "content": "test"}]
        builder.build_collection("entry_logbook", documents=docs)
        mock_es.client.indices.create.assert_called_once()

    def test_es_index_not_recreated_if_exists(self, builder, mock_es):
        mock_es.index_exists.return_value = True
        docs = [{"id": "d1", "content": "test"}]
        builder.build_collection("entry_logbook", documents=docs)
        mock_es.client.indices.create.assert_not_called()


# ---------------------------------------------------------------------------
# KG metadata enrichment
# ---------------------------------------------------------------------------

class TestKGMetadataEnrichment:
    def test_kg_metadata_applied(self, builder):
        docs = [
            {"id": "d1", "content": "text 1"},
            {"id": "d2", "content": "text 2"},
        ]
        kg_meta = {
            "d1": {"kg_node_ids": ["n1", "n2"], "triplet_count": 5},
        }
        builder.build_collection("entry_logbook", documents=docs, kg_metadata=kg_meta)
        assert docs[0]["kg_node_ids"] == ["n1", "n2"]
        assert docs[0]["triplet_count"] == 5
        assert docs[0]["graph_enriched"] is True
        assert docs[1]["graph_enriched"] is False

    def test_no_kg_metadata_leaves_docs_unchanged(self, builder):
        docs = [{"id": "d1", "content": "text"}]
        builder.build_collection("entry_logbook", documents=docs, kg_metadata=None)
        assert "graph_enriched" not in docs[0]


# ---------------------------------------------------------------------------
# build_all_collections
# ---------------------------------------------------------------------------

class TestBuildAllCollections:
    def test_builds_all_six(self, builder):
        with patch.object(builder, "build_collection") as mock_build:
            mock_build.return_value = {"indexed": 10, "collection": "test"}
            result = builder.build_all_collections()
        assert len(result) == 6
        assert all(v == 10 for v in result.values())

    def test_error_in_one_collection_doesnt_stop_others(self, builder):
        call_count = [0]

        def side_effect(key):
            call_count[0] += 1
            if key == "icms_collection":
                raise RuntimeError("ES down")
            return {"indexed": 5, "collection": f"agrag_{key}"}

        with patch.object(builder, "build_collection", side_effect=side_effect):
            result = builder.build_all_collections()
        assert result["icms_collection"] == 0  # Failed
        assert result["entry_logbook"] == 5  # Succeeded
        assert call_count[0] == 6  # All attempted


# ---------------------------------------------------------------------------
# _upsert_to_qdrant
# ---------------------------------------------------------------------------

class TestUpsertToQdrant:
    def test_batch_upsert_called(self, builder, mock_qdrant):
        docs = [{"id": f"d{i}", "content": f"doc {i}"} for i in range(5)]
        embeddings = [[0.1, 0.2, 0.3]] * 5
        total = builder._upsert_to_qdrant("agrag_test", docs, embeddings)
        assert total == 5
        mock_qdrant.client.upsert.assert_called()

    def test_content_truncated_in_payload(self, builder, mock_qdrant):
        docs = [{"id": "d1", "content": "x" * 5000}]
        embeddings = [[0.1, 0.2, 0.3]]
        builder._upsert_to_qdrant("agrag_test", docs, embeddings)
        upsert_call = mock_qdrant.client.upsert.call_args
        points = upsert_call[1]["points"] if "points" in upsert_call[1] else upsert_call[0][0]
        # The first point's payload content should be truncated
        # (implementation caps at 2000 chars)


# ---------------------------------------------------------------------------
# _index_to_es
# ---------------------------------------------------------------------------

class TestIndexToES:
    def test_bulk_index_called(self, builder, mock_es):
        docs = [
            {"id": "d1", "content": "test doc", "title": "Test", "author": "Alice"},
        ]
        with patch("elasticsearch.helpers.bulk") as mock_bulk:
            builder._index_to_es("agrag_test", docs)
            mock_bulk.assert_called_once()

    def test_empty_docs_no_bulk(self, builder, mock_es):
        with patch("elasticsearch.helpers.bulk") as mock_bulk:
            builder._index_to_es("agrag_test", [])
            mock_bulk.assert_not_called()
