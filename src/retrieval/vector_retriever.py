"""
Vector retriever — semantic search against Qdrant _ag_rag collections.
Uses Argo embeddings API (v3large) for query encoding.

Supports both chunking strategies:
  - "whole-doc"          → searches agrag_* collections
  - "granular-chunking"  → searches agrag_gc_* collections (default)
"""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Qdrant-backed semantic retriever for _ag_rag collections.
    """

    def __init__(
        self,
        qdrant_client,
        embedding_model,
        top_k: int = 100,
        chunking_strategy: str = "granular-chunking",
    ):
        self.qdrant = qdrant_client
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.chunking_strategy = chunking_strategy

    def retrieve(
        self,
        query: str,
        collections: List[str],
        top_k: Optional[int] = None,
        date_filter: Optional[Dict[str, str]] = None,
        chunking_strategy: Optional[str] = None,
    ) -> List[Document]:
        """
        Semantic search across one or more _ag_rag collections.

        Args:
            query: User query string
            collections: List of collection keys (e.g., ['entry_logbook', 'icms_collection'])
                         Use ['all_databases'] for federated search.
            top_k: Number of results per collection
            date_filter: Optional temporal filter dict
            chunking_strategy: Override instance-level strategy for this query

        Returns:
            List of LangChain Document objects with metadata
        """
        from src.vectorstore.collection_registry import (
            get_collection_names,
            STRATEGY_PREFIXES,
            DEFAULT_STRATEGY,
        )

        k = top_k or self.top_k
        strategy = chunking_strategy or self.chunking_strategy
        query_vector = self.embedding_model.embed_query(query)

        all_docs: List[Document] = []

        # Resolve target (collection_key, qdrant_collection) pairs.
        prefix = STRATEGY_PREFIXES.get(strategy, STRATEGY_PREFIXES[DEFAULT_STRATEGY])
        if "all_databases" in collections:
            # Federated: enumerate every ingested collection with this strategy's
            # prefix, so databases added under data/raw/ are queryable with no
            # code change once ingested.
            targets = [(name[len(prefix):], name) for name in self.qdrant.list_collections(prefix)]
        else:
            targets = [(k_, get_collection_names(k_, strategy)["qdrant_collection"]) for k_ in collections]

        for coll_key, qdrant_name in targets:
            if not self.qdrant.collection_exists(qdrant_name):
                logger.warning(f"Collection {qdrant_name} does not exist in Qdrant")
                continue

            try:
                results = self.qdrant.search(
                    collection_name=qdrant_name,
                    query_vector=query_vector,
                    top_k=k,
                    date_range=date_filter,
                )
                for r in results:
                    payload = r["payload"]
                    doc = Document(
                        page_content=payload.get("content") or "",
                        metadata={
                            "score": r["score"],
                            "retriever_type": "vector",
                            "collection": coll_key,
                            "chunking_strategy": strategy,
                            **payload,
                        },
                    )
                    all_docs.append(doc)
                logger.debug(f"Vector search {qdrant_name}: {len(results)} results")
            except Exception as e:
                logger.error(f"Vector search failed for {qdrant_name}: {e}")

        return all_docs
