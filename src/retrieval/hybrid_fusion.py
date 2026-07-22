"""
3-way hybrid retrieval fusion with adaptive weighting.

Fuses dense (vector), sparse (BM25), and graph retrieval via Reciprocal Rank
Fusion (RRF), with query-type-adaptive weight tuning.

Novel contribution: adaptive weights based on query-intent classification.
"""
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Adaptive weight profiles per query intent
# (vector_weight, bm25_weight, graph_weight)
ADAPTIVE_WEIGHTS = {
    "factual":        (0.3, 0.2, 0.5),   # Entity/fact queries → graph↑
    "temporal":       (0.3, 0.5, 0.2),   # Date queries → BM25↑
    "troubleshooting":(0.3, 0.2, 0.5),   # Fault queries → graph↑ (fault chains)
    "comparative":    (0.4, 0.3, 0.3),   # Balanced for comparison
    "multi_hop":      (0.3, 0.2, 0.5),   # Multi-hop → graph↑
    "general":        (0.4, 0.3, 0.3),   # Default balanced
    "mathematical":   (0.5, 0.3, 0.2),   # Math → semantic vector↑
    "code_generation":(0.5, 0.3, 0.2),   # Code → semantic vector↑
}
DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)


class AdaptiveHybridFusion:
    """
    3-way RRF fusion with adaptive weights.

    Fuses results from:
    1. VectorRetriever (Qdrant semantic)
    2. BM25Retriever (Elasticsearch keyword)
    3. GraphRetriever (Neo4j KG + GDS)

    For 'all_databases': applies cross-collection score normalization before fusion.
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        graph_docs: List[Document],
        query_intent: str = "general",
        cross_collection: bool = False,
    ) -> List[Document]:
        """
        Fuse 3 ranked lists using weighted RRF with adaptive weights.

        Args:
            vector_docs: Semantic search results
            bm25_docs: BM25 keyword results
            graph_docs: Graph traversal results
            query_intent: Determines adaptive weight profile
            cross_collection: If True, normalize scores per collection first

        Returns:
            Fused and ranked document list
        """
        weights = ADAPTIVE_WEIGHTS.get(query_intent, DEFAULT_WEIGHTS)
        w_vector, w_bm25, w_graph = weights

        logger.debug(
            f"Fusion weights for intent '{query_intent}': "
            f"vector={w_vector}, bm25={w_bm25}, graph={w_graph}"
        )

        if cross_collection:
            vector_docs = self._normalize_per_collection(vector_docs)
            bm25_docs = self._normalize_per_collection(bm25_docs)

        doc_scores: Dict[str, float] = defaultdict(float)
        doc_objects: Dict[str, Document] = {}

        def add_results(docs: List[Document], weight: float, retriever_type: str):
            for rank, doc in enumerate(docs, start=1):
                doc_id = self._get_doc_id(doc)
                doc_objects[doc_id] = doc
                rrf_score = weight * (1.0 / (self.rrf_k + rank))
                doc_scores[doc_id] += rrf_score
                # Track retriever contributions in metadata
                if doc_id in doc_objects:
                    doc_objects[doc_id].metadata[f"{retriever_type}_rank"] = rank
                    doc_objects[doc_id].metadata[f"{retriever_type}_rrf"] = rrf_score

        add_results(vector_docs, w_vector, "vector")
        add_results(bm25_docs, w_bm25, "bm25")
        add_results(graph_docs, w_graph, "graph")

        sorted_ids = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)

        fused: List[Document] = []
        for i, doc_id in enumerate(sorted_ids):
            doc = doc_objects[doc_id]
            doc.metadata["rrf_score"] = doc_scores[doc_id]
            doc.metadata["final_rank"] = i + 1
            doc.metadata["fusion_method"] = "adaptive_3way_rrf"
            doc.metadata["query_intent"] = query_intent
            fused.append(doc)

        logger.info(
            f"3-way fusion: {len(vector_docs)} vector + {len(bm25_docs)} BM25 + "
            f"{len(graph_docs)} graph → {len(fused)} fused docs"
        )
        return fused

    def _get_doc_id(self, doc: Document) -> str:
        """Stable document ID for deduplication."""
        content = doc.page_content or ""
        doc_id = doc.metadata.get("id") or doc.metadata.get("file_id") or ""
        return doc_id if doc_id else str(hash(content))

    def _normalize_per_collection(self, docs: List[Document]) -> List[Document]:
        """
        Cross-collection score normalization.
        Groups by collection, min-max normalizes scores within each, then merges.
        """
        collection_groups: Dict[str, List[Document]] = defaultdict(list)
        for doc in docs:
            coll = doc.metadata.get("collection", "unknown")
            collection_groups[coll].append(doc)

        normalized: List[Document] = []
        for coll, coll_docs in collection_groups.items():
            scores = [d.metadata.get("score", 0.0) for d in coll_docs]
            min_s, max_s = min(scores), max(scores)
            for doc in coll_docs:
                if max_s > min_s:
                    norm = (doc.metadata.get("score", 0.0) - min_s) / (max_s - min_s)
                else:
                    norm = 1.0
                doc.metadata["normalized_score"] = norm
                normalized.append(doc)

        normalized.sort(key=lambda d: d.metadata.get("normalized_score", 0.0), reverse=True)
        return normalized


# Alias for backwards-compatible imports
HybridFusion = AdaptiveHybridFusion
