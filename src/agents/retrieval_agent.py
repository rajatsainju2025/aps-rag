"""Retrieval agent node — executes 3-way hybrid retrieval based on router decision.
Dispatches to vector, BM25, and graph retrievers in parallel.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from langchain_core.documents import Document

from .event_emitter import get_emitter
from .state import AgentState
from src.utils.temporal_filter import run_with_temporal_expansion

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Executes multi-source retrieval based on routing strategy.
    Runs vector + BM25 + graph retrieval in parallel using ThreadPoolExecutor.
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        graph_retriever,
        fusion,
        reranker,
        query_enhancer,
    ):
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.graph = graph_retriever
        self.fusion = fusion
        self.reranker = reranker
        self.enhancer = query_enhancer

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Execute multi-query retrieval and return fused, reranked documents."""
        query = state.get("enhanced_query") or state.get("original_query", "")
        intent = state.get("query_intent", "general")
        collections = state.get("selected_databases", ["all_databases"])
        # Eval override pins the strategy regardless of router/naive defaults.
        strategy = state.get("retrieval_strategy_override") or state.get(
            "retrieval_strategy", "HYBRID_3WAY"
        )

        # ── 1. Enhance query (synonym + temporal + variants) ───────────────
        enhancement = self.enhancer.enhance(query)
        enhanced_q = enhancement["enhanced_query"]
        temporal = enhancement.get("temporal_context")
        # Use variants from state if the orchestrator already ran enhancement,
        # otherwise use the freshly generated variants.
        query_variants: List[str] = (
            state.get("query_variants")
            or enhancement.get("query_variants")
            or [enhanced_q]
        )
        if len(query_variants) > 1:
            logger.info(
                "RetrievalAgent multi-query: %d variants", len(query_variants)
            )

        cross_collection = "all_databases" in collections

        # ── 2. Retrieve for every query variant ───────────────────────────
        all_vector_raw: List[Document] = []
        all_bm25_raw: List[Document] = []
        graph_docs: List[Document] = []

        # BM25 pre-filter (ES can apply date filter at query time)
        date_filter_es = (
            {"gte": temporal["start"], "lte": temporal["end"]}
            if temporal and temporal.get("start") and temporal.get("end")
            else None
        )

        em = get_emitter()

        if strategy == "VECTOR_ONLY":
            em.emit("thinking_start", {"status": "Vector search..."})
            for variant in query_variants:
                all_vector_raw.extend(
                    self._safe_retrieve(self.vector.retrieve, variant, collections)
                )

        elif strategy == "BM25_ONLY":
            em.emit("thinking_start", {"status": "Lexical (BM25) search..."})
            for variant in query_variants:
                all_bm25_raw.extend(
                    self._safe_retrieve(
                        self.bm25.retrieve, variant, collections, None, date_filter_es
                    )
                )

        elif strategy == "VECTOR_BM25":
            # Hybrid vector + BM25 (no graph). Without this branch the strategy
            # fell through to the vector-only default fallback below.
            em.emit("thinking_start", {"status": "Vector search..."})
            em.emit("thinking_start", {"status": "Lexical (BM25) search..."})
            for variant in query_variants:
                all_vector_raw.extend(
                    self._safe_retrieve(self.vector.retrieve, variant, collections)
                )
                all_bm25_raw.extend(
                    self._safe_retrieve(
                        self.bm25.retrieve, variant, collections, None, date_filter_es
                    )
                )

        elif strategy == "KG_ONLY":
            em.emit("thinking_start", {"status": "Graph search..."})
            entities = state.get("extracted_entities", [enhanced_q])
            graph_docs = self._safe_retrieve(
                self.graph.retrieve_entity_centric, entities
            )

        elif strategy in ("HYBRID_3WAY", "WEB_SEARCH"):
            em.emit("thinking_start", {"status": "Vector search..."})
            em.emit("thinking_start", {"status": "Lexical (BM25) search..."})
            em.emit("thinking_start", {"status": "Graph search..."})
            entities = state.get("extracted_entities", [])
            # Determine adaptive graph subtypes:
            # - entity_centric: always (baseline KG retrieval)
            # - path_based: if 2+ entities detected (find paths between them)
            # - community_based: if intent is troubleshooting (fault community docs)
            graph_subtypes = ["entity_centric"]
            if len(entities) >= 2:
                graph_subtypes.append("path_based")
                logger.info("Adaptive graph: adding path_based (entities=%d)", len(entities))
            if intent == "troubleshooting":
                graph_subtypes.append("community_based")
                logger.info("Adaptive graph: adding community_based (troubleshooting intent)")

            # Fan-out: one set of futures per variant for vector + BM25
            n_graph_futures = len(graph_subtypes)
            max_workers = min(12, len(query_variants) * 2 + n_graph_futures)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map: dict = {}
                for variant in query_variants:
                    future_map[executor.submit(
                        self.vector.retrieve, variant, collections
                    )] = ("vector", variant)
                    future_map[executor.submit(
                        self.bm25.retrieve, variant, collections,
                        None, date_filter_es,
                    )] = ("bm25", variant)
                # Graph retrieval: one future per subtype
                graph_futures = {}
                for subtype in graph_subtypes:
                    f = executor.submit(
                        self.graph.retrieve, enhanced_q, entities, subtype
                    )
                    graph_futures[f] = subtype

                all_futures = list(future_map.keys()) + list(graph_futures.keys())
                for future in as_completed(all_futures):
                    if future in graph_futures:
                        subtype = graph_futures[future]
                        try:
                            result = future.result(timeout=30) or []
                            graph_docs.extend(result)
                            logger.debug("Graph (%s): %d docs", subtype, len(result))
                        except Exception as exc:
                            logger.warning("Graph retrieval (%s) failed: %s", subtype, exc)
                        continue
                    src, _ = future_map[future]
                    try:
                        result = future.result(timeout=30) or []
                        if src == "vector":
                            all_vector_raw.extend(result)
                        else:
                            all_bm25_raw.extend(result)
                    except Exception as exc:
                        logger.warning("Retrieval future (%s) failed: %s", src, exc)
        else:
            # Default fallback
            all_vector_raw = self._safe_retrieve(self.vector.retrieve, enhanced_q, collections)

        # Deduplicate within each source
        vector_docs = self._dedup_docs(all_vector_raw)
        bm25_docs = self._dedup_docs(all_bm25_raw)
        graph_docs = self._dedup_docs(graph_docs)
        logger.info(
            "RetrievalAgent raw: vector=%d bm25=%d graph=%d  (from %d variants)",
            len(vector_docs), len(bm25_docs), len(graph_docs), len(query_variants),
        )

        # ── 3. Post-retrieval temporal filtering (±30-day expansion fallback) ──
        tc_start = temporal.get("start") if temporal else None
        tc_end = temporal.get("end") if temporal else None

        if tc_start and tc_end and (vector_docs or bm25_docs):
            combined_raw = self._dedup_docs(vector_docs + bm25_docs)

            def _re_retrieve_no_filter() -> List[Document]:
                rv: List[Document] = []
                rb: List[Document] = []
                for variant in query_variants:
                    rv.extend(self._safe_retrieve(self.vector.retrieve, variant, collections))
                    rb.extend(self._safe_retrieve(self.bm25.retrieve, variant, collections))
                return self._dedup_docs(rv + rb)

            filtered_docs, stage = run_with_temporal_expansion(
                retrieve_fn=_re_retrieve_no_filter,
                initial_docs=combined_raw,
                start_iso=tc_start,
                end_iso=tc_end,
            )
            if stage != "strict":
                logger.info("RetrievalAgent temporal expansion stage: %s", stage)
            vector_docs = filtered_docs
            bm25_docs = []

        # ── 4. 3-way fusion + rerank ───────────────────────────────────────
        fused = self.fusion.fuse(
            vector_docs, bm25_docs, graph_docs,
            query_intent=intent,
            cross_collection=cross_collection,
        )

        entities = state.get("extracted_entities", [])
        em.emit("thinking_start", {"status": "Reranking..."})
        reranked = self.reranker.rerank(
            query, fused, query_entities=entities, query_intent=intent,
            neo4j_client=getattr(self.graph, "neo4j", None),
        )

        sources = []
        if vector_docs:
            sources.append("vector")
        if bm25_docs:
            sources.append("bm25")
        if graph_docs:
            sources.append("graph")

        logger.info(
            "RetrievalAgent complete: %d reranked docs  (sources: %s)",
            len(reranked), ", ".join(sources) or "none",
        )

        # Update reasoning trace for UI thinking display
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"Retrieval: {len(vector_docs)} vector + {len(bm25_docs)} BM25 + "
            f"{len(graph_docs)} graph docs from {len(query_variants)} query variant(s)"
        )
        reasoning_trace.append(f"After fusion + rerank: {len(reranked)} documents")

        return {
            "retrieved_docs": reranked,
            "graph_context": [
                {"id": d.metadata.get("id", ""), "content": d.page_content,
                 "collection": d.metadata.get("collection", "")}
                for d in graph_docs
            ],
            "retrieval_sources": sources,
            "temporal_context": temporal,
            "query_variants": query_variants,
            "reasoning_trace": reasoning_trace,
        }

    @staticmethod
    def _dedup_docs(docs: List[Document]) -> List[Document]:
        """Deduplicate documents by ID or content prefix, keeping highest-score copy."""
        seen: Dict[Any, Document] = {}
        for doc in docs:
            key = (
                doc.metadata.get("id")
                or doc.metadata.get("doc_id")
                or hash((doc.page_content or '')[:100])
            )
            if key not in seen:
                seen[key] = doc
            else:
                existing_score = seen[key].metadata.get("score", 0.0)
                new_score = doc.metadata.get("score", 0.0)
                if new_score > existing_score:
                    seen[key] = doc
        return list(seen.values())

    def _safe_retrieve(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs) or []
        except Exception as e:
            logger.warning(f"Retriever {fn.__name__} failed: {e}")
            return []
