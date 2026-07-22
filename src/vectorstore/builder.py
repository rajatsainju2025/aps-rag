"""
Vectorstore builder.

Loads normalized documents, chunks them, embeds each chunk, and writes vectors
to Qdrant + a keyword index to Elasticsearch. Optionally enriches documents
with knowledge-graph metadata (kg_node_ids, triplet_count, graph_enriched).

Supports two chunking strategies (selectable at build time):
  - "whole-doc"          : 1 vector per normalized record (fast, coarse-grained)
  - "granular-chunking"  : 3-stage recursive splitting (default; fine-grained,
                           ~3-15x more vectors). See src/vectorstore/granular_chunking.py
"""
import sys
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from .collection_registry import COLLECTIONS, get_collection_names

logger = logging.getLogger(__name__)

# Valid chunking strategy values
VALID_STRATEGIES = ("whole-doc", "granular-chunking")


class AgRAGVectorStoreBuilder:
    """
    Builds _ag_rag Qdrant collections and ES indices.

    Key difference from entry_rag_app builder:
    - Adds kg_node_ids, triplet_count, graph_enriched=True to all documents
    - Creates new collections with _ag_rag suffix
    - Never modifies existing entry_rag_app collections
    - Supports selectable chunking_strategy: "whole-doc" or "granular-chunking"
    """

    def __init__(
        self,
        settings,
        qdrant_client,
        es_client,
        embedding_model,
        chunking_strategy: str = "whole-doc",
    ):
        self.settings = settings
        self.qdrant = qdrant_client
        self.es = es_client
        self.embedding_model = embedding_model

        if chunking_strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown chunking_strategy '{chunking_strategy}'. "
                f"Valid options: {VALID_STRATEGIES}"
            )
        self.chunking_strategy = chunking_strategy

        # Lazy-init the GranularChunker only when needed
        self._granular_chunker = None

    def _get_granular_chunker(self):
        """Lazy-load GranularChunker to avoid import overhead for whole-doc builds."""
        if self._granular_chunker is None:
            from .granular_chunking import GranularChunker

            chunking_cfg = getattr(self.settings, "chunking", None)
            kwargs = {}
            if chunking_cfg:
                kwargs = {
                    "pre_chunk_size": getattr(chunking_cfg, "pre_chunk_size", 1000),
                    "pre_chunk_overlap": getattr(chunking_cfg, "pre_chunk_overlap", 200),
                    "max_chunk_size": getattr(chunking_cfg, "max_chunk_size", 4000),
                    "chunk_overlap": getattr(chunking_cfg, "chunk_overlap", 200),
                }
            self._granular_chunker = GranularChunker(**kwargs)
        return self._granular_chunker

    def build_collection(
        self,
        collection_key: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        kg_metadata: Optional[Dict[str, Any]] = None,
        qdrant_name_override: Optional[str] = None,
        es_index_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a single collection using the configured chunking strategy.

        Args:
            collection_key: e.g., 'entry_logbook'
            documents: Pre-loaded normalized documents (if None, loads from data_path)
            kg_metadata: Dict mapping doc_id → {kg_node_ids, triplet_count}
            qdrant_name_override: If set, build into this Qdrant collection name
                                  instead of the default (used for staging builds).
            es_index_override: If set, build into this ES index name.

        Returns:
            {'indexed': N, 'collection': name, 'strategy': str}
        """
        names = get_collection_names(collection_key, self.chunking_strategy)
        qdrant_name = qdrant_name_override or names["qdrant_collection"]
        es_index = es_index_override or names["es_index"]

        if documents is None:
            documents = self._load_documents(collection_key)

        if not documents:
            logger.warning(f"No documents for collection {collection_key}")
            return {"indexed": 0, "collection": qdrant_name, "strategy": self.chunking_strategy}

        # Enrich with KG metadata if available
        if kg_metadata:
            for doc in documents:
                doc_id = doc.get("id", "")
                if doc_id in kg_metadata:
                    doc["kg_node_ids"] = kg_metadata[doc_id].get("kg_node_ids", [])
                    doc["triplet_count"] = kg_metadata[doc_id].get("triplet_count", 0)
                    doc["graph_enriched"] = True
                else:
                    doc["graph_enriched"] = False

        # ---- Strategy dispatch ----
        if self.chunking_strategy == "granular-chunking":
            indexed = self._build_granular(collection_key, qdrant_name, es_index, documents)
        else:
            indexed = self._build_whole_doc(collection_key, qdrant_name, es_index, documents)

        logger.info(
            f"Built {collection_key} [{self.chunking_strategy}]: "
            f"{indexed} vectors in {qdrant_name}"
        )
        return {"indexed": indexed, "collection": qdrant_name, "strategy": self.chunking_strategy}

    # ------------------------------------------------------------------
    # Strategy: whole-doc (original behaviour — 1 vector per record)
    # ------------------------------------------------------------------

    def _build_whole_doc(
        self,
        collection_key: str,
        qdrant_name: str,
        es_index: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Original whole-doc pipeline: embed raw content, 1 vector per record."""
        texts = [doc.get("content", "") or doc.get("page_content", "") for doc in documents]

        logger.info(f"[whole-doc] Embedding {len(texts)} documents for {collection_key}...")
        embeddings = self.embedding_model.embed_documents(texts)

        indexed = self._upsert_to_qdrant(qdrant_name, documents, embeddings)
        self._index_to_es(es_index, documents)
        return indexed

    # ------------------------------------------------------------------
    # Strategy: granular-chunking (entry_rag_app-style recursive splitting)
    # ------------------------------------------------------------------

    def _build_granular(
        self,
        collection_key: str,
        qdrant_name: str,
        es_index: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Granular-chunking pipeline:
          1. GranularChunker splits normalized dicts → List[Document]
          2. Embed each chunk's page_content
          3. Upsert to Qdrant + ES with chunk metadata
        """
        chunker = self._get_granular_chunker()
        chunk_docs: List[Document] = chunker.chunk_documents(documents, collection_key)

        if not chunk_docs:
            logger.warning(f"[granular-chunking] No chunks produced for {collection_key}")
            return 0

        texts = [d.page_content for d in chunk_docs]

        logger.info(
            f"[granular-chunking] {collection_key}: {len(documents)} docs → "
            f"{len(chunk_docs)} chunks. Embedding..."
        )
        embeddings = self.embedding_model.embed_documents(texts)

        # Convert Documents to dicts for the Qdrant/ES helpers
        doc_dicts = []
        for d in chunk_docs:
            entry = {**d.metadata, "content": d.page_content}
            doc_dicts.append(entry)

        indexed = self._upsert_to_qdrant(qdrant_name, doc_dicts, embeddings)
        self._index_to_es(es_index, doc_dicts)
        return indexed

    def build_all_collections(self) -> Dict[str, Any]:
        """Build all 6 collections using the configured chunking strategy."""
        results = {}
        for key in COLLECTIONS:
            try:
                result = self.build_collection(key)
                results[key] = result
            except Exception as e:
                logger.error(f"Failed to build collection {key}: {e}")
                results[key] = {"indexed": 0, "error": str(e)}
        return results

    def _load_documents(self, collection_key: str) -> List[Dict[str, Any]]:
        """Load and normalize documents using DocumentLoader."""
        from src.utils.document_loader import DocumentLoader

        loader = DocumentLoader()
        return loader.load_collection(collection_key)

    def _upsert_to_qdrant(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """Upsert documents and embeddings to Qdrant."""
        from qdrant_client.http import models

        vector_size = len(embeddings[0]) if embeddings else self.settings.vector.embedding_dim

        # Create collection if not exists, or recreate if dimension mismatch
        if self.qdrant.collection_exists(collection_name):
            info = self.qdrant.client.get_collection(collection_name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != vector_size:
                logger.warning(
                    f"Dimension mismatch for {collection_name}: "
                    f"existing={existing_dim}, required={vector_size}. "
                    f"Deleting and recreating."
                )
                self.qdrant.client.delete_collection(collection_name)
                self.qdrant.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Recreated Qdrant collection: {collection_name} (dim={vector_size})")
        else:
            self.qdrant.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {collection_name} (dim={vector_size})")

        # Batch upsert
        batch_size = 100
        total = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i: i + batch_size]
            batch_embeddings = embeddings[i: i + batch_size]

            points = []
            for j, (doc, emb) in enumerate(zip(batch_docs, batch_embeddings)):
                # Deterministic UUID from chunk ID — idempotent upserts,
                # no collision between incremental batches
                chunk_id = doc.get("id", f"doc_{i}_{j}")
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))
                payload = {k: v for k, v in doc.items() if k != "content"}
                payload["content"] = doc.get("content", "")[:2000]  # Truncate for storage
                points.append(
                    models.PointStruct(id=point_id, vector=emb, payload=payload)
                )

            self.qdrant.client.upsert(
                collection_name=collection_name, points=points
            )
            total += len(points)

        return total

    def _index_to_es(self, index_name: str, documents: List[Dict[str, Any]]) -> None:
        """Index documents to Elasticsearch."""
        # Create index if not exists
        if not self.es.index_exists(index_name):
            self.es.client.indices.create(
                index=index_name,
                body={
                    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "title": {"type": "text"},
                            "author": {"type": "keyword"},
                            "system": {"type": "keyword"},
                            "collection": {"type": "keyword"},
                            "creation_datetime": {"type": "date", "ignore_malformed": True},
                        }
                    },
                },
            )
            logger.info(f"Created ES index: {index_name}")

        # Bulk index
        from elasticsearch.helpers import bulk

        actions = []
        for doc in documents:
            actions.append({
                "_index": index_name,
                "_id": doc.get("id", ""),
                "_source": {
                    "content": doc.get("content", "")[:5000],
                    "title": doc.get("title", doc.get("entry_filename", "")),
                    "author": doc.get("author", ""),
                    "system": doc.get("system", ""),
                    "collection": doc.get("collection", ""),
                    "creation_datetime": doc.get("creation_datetime", ""),
                    "file_id": doc.get("file_id", doc.get("id", "")),
                },
            })

        if actions:
            try:
                bulk(self.es.client, actions, raise_on_error=False)
                logger.info(f"ES indexed {len(actions)} docs to {index_name}")
            except Exception as e:
                logger.error(f"ES bulk index failed for {index_name}: {e}")
