"""
BM25 retriever — Elasticsearch full-text search against _ag_rag indices.
"""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Elasticsearch BM25 retriever for _ag_rag indices.
    """

    def __init__(self, es_client, top_k: int = 100, chunking_strategy: str = "granular-chunking"):
        self.es = es_client
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
        BM25 search across one or more _ag_rag ES indices.

        Args:
            query: User query string (expanded/enhanced preferred)
            collections: Collection keys; use ['all_databases'] for federated
            top_k: Results per index
            date_filter: Optional date range filter
            chunking_strategy: Override instance-level strategy for this query
        """
        from src.vectorstore.collection_registry import (
            get_collection_names,
            STRATEGY_PREFIXES,
            DEFAULT_STRATEGY,
        )

        k = top_k or self.top_k
        strategy = chunking_strategy or self.chunking_strategy
        all_docs: List[Document] = []

        # Resolve target (collection_key, es_index) pairs.
        prefix = STRATEGY_PREFIXES.get(strategy, STRATEGY_PREFIXES[DEFAULT_STRATEGY])
        if "all_databases" in collections:
            # Federated: enumerate every ingested index with this strategy's
            # prefix, so databases added under data/raw/ are queryable with no
            # code change once ingested.
            targets = [(name[len(prefix):], name) for name in self.es.list_indices(prefix)]
        else:
            targets = [(k_, get_collection_names(k_, strategy)["es_index"]) for k_ in collections]

        for coll_key, index_name in targets:
            if not self.es.index_exists(index_name):
                logger.warning(f"ES index {index_name} does not exist")
                continue

            try:
                results = self.es.bm25_search(
                    index_name=index_name,
                    query=query,
                    top_k=k,
                    date_filter=date_filter,
                )
                for r in results:
                    payload = r["payload"]
                    doc = Document(
                        page_content=payload.get("content") or "",
                        metadata={
                            "score": r["score"],
                            "retriever_type": "bm25",
                            "collection": coll_key,
                            **payload,
                        },
                    )
                    all_docs.append(doc)
                logger.debug(f"BM25 search {index_name}: {len(results)} results")
            except Exception as e:
                logger.error(f"BM25 search failed for {index_name}: {e}")

        return all_docs
