"""
LLM-based reranker + graph-proximity bonus — powered by Argo API.

Replaces the local cross-encoder approach with an Argo LLM call that scores
query-document relevance on a 0-10 scale, keeping the graph-proximity bonus
as an additive boost.
"""
import logging
import re
from typing import List, Optional

from langchain_core.documents import Document

from config.prompts import get_prompt

logger = logging.getLogger(__name__)



class GraphProximityReranker:
    """
    Argo-LLM reranker with graph-proximity bonus.

    Sends a batch of (query, doc-snippet) pairs to the Argo LLM and asks for
    relevance scores 0-10.  Falls back to existing RRF scores when the LLM
    is unavailable.

    Graph-proximity bonus: +0.1 per query entity a doc is connected to in the KG.
    This encourages documents that are central to the query entity neighborhood.
    """

    def __init__(
        self,
        llm=None,
        top_k: int = 50,
        graph_proximity_bonus: float = 0.1,
        max_doc_chars: int = 800,
        # Legacy kwargs accepted but ignored so callers don't break
        model_name: str | None = None,
    ):
        self.llm = llm
        self.top_k = top_k
        self.graph_proximity_bonus = graph_proximity_bonus
        self.max_doc_chars = max_doc_chars

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: List[Document],
        query_entities: Optional[List[str]] = None,
        neo4j_client=None,
        query_intent: str = "general",
    ) -> List[Document]:
        """
        Rerank documents using Argo LLM relevance scoring + graph-proximity bonus.

        Args:
            query: Original user query
            documents: Fused retrieval results (post-RRF)
            query_entities: Named entities in query (for proximity bonus)
            neo4j_client: If provided, computes graph-proximity bonus
            query_intent: Intent type from router (e.g., temporal, troubleshooting)
        """
        if not documents:
            return []

        scores = self._llm_score(query, documents, query_intent=query_intent)

        # Apply graph-proximity bonus
        if query_entities and neo4j_client and self.graph_proximity_bonus > 0:
            scores = self._apply_graph_proximity(
                documents, scores, query_entities, neo4j_client
            )

        # Sort by final score and return top_k
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        result: List[Document] = []
        for rank, (score, doc) in enumerate(scored[: self.top_k], start=1):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_rank"] = rank
            result.append(doc)

        logger.info(f"Reranking: {len(documents)} → {len(result)} documents")
        return result

    # ------------------------------------------------------------------
    # LLM-based scoring via Argo
    # ------------------------------------------------------------------
    def _llm_score(self, query: str, documents: List[Document],
                   query_intent: str = "general") -> List[float]:
        """Score documents via the Argo LLM; fall back to RRF scores on error."""
        if self.llm is None:
            logger.warning("No LLM provided for reranking — using RRF scores")
            return [doc.metadata.get("rrf_score", 0.0) for doc in documents]

        # Build numbered document list with collection metadata hint
        doc_lines = []
        for i, doc in enumerate(documents):
            collection = doc.metadata.get("collection", "")
            date = str(doc.metadata.get("creation_datetime", ""))[:10]
            meta = f" [{collection}, {date}]" if collection or date else ""
            doc_lines.append(f"[{i+1}]{meta} {(doc.page_content or '')[:self.max_doc_chars]}")
        doc_block = "\n\n".join(doc_lines)

        # Map intent to human-readable query type hint
        _intent_labels = {
            "temporal": "temporal (date-range based — boost docs from specified time window)",
            "troubleshooting": "troubleshooting (fault diagnosis — boost docs with fault+resolution info)",
            "counting": "counting/statistical (boost docs with structured data or numerical summaries)",
            "code_generation": "code/API (boost docs with code examples or API specs)",
            "factual": "factual lookup (boost docs with direct, specific answers)",
            "multi_hop": "multi-hop reasoning (boost docs covering multiple related aspects)",
        }
        query_type = _intent_labels.get(query_intent, f"{query_intent} (general semantic relevance)")

        prompt = get_prompt("RERANK_PROMPT").format(query=query, documents=doc_block, query_type=query_type)

        try:
            response = self.llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            scores = self._parse_scores(raw, len(documents))
            return scores
        except Exception as e:
            logger.warning(f"LLM reranking failed: {e} — using RRF scores")
            return [doc.metadata.get("rrf_score", 0.0) for doc in documents]

    @staticmethod
    def _parse_scores(text: str, expected: int) -> List[float]:
        """Extract comma-separated integers from LLM output, stripping CoT blocks."""
        # Strip <thinking> block if present
        import re as _re
        text = _re.sub(r"<thinking>.*?</thinking>", "", text, flags=_re.DOTALL).strip()
        numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
        if len(numbers) >= expected:
            return numbers[:expected]
        # Pad with 0 if LLM returned fewer
        return numbers + [0.0] * (expected - len(numbers))

    # ------------------------------------------------------------------
    # Graph-proximity bonus (unchanged from original)
    # ------------------------------------------------------------------
    def _apply_graph_proximity(
        self,
        documents: List[Document],
        scores: List[float],
        query_entities: List[str],
        neo4j_client,
    ) -> List[float]:
        """Add graph-proximity bonus to reranker scores."""
        return _apply_graph_proximity_bonus(
            documents, scores, query_entities, neo4j_client,
            bonus_per_entity=self.graph_proximity_bonus,
        )


# =============================================================================
# Shared graph-proximity bonus (free function — used by both reranker classes)
# =============================================================================
def _minmax(xs: List[float]) -> List[float]:
    """Min-max normalise to [0,1]; all-equal (incl. empty-range) → all 0.0."""
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def _apply_graph_proximity_bonus(
    documents: List[Document],
    scores: List[float],
    query_entities: List[str],
    neo4j_client,
    bonus_per_entity: float = 0.1,
) -> List[float]:
    """Add an additive bonus per query-entity that a document is connected to
    in the Neo4j knowledge graph. Caps at 1 hit per entity per doc.

    The KG lookups depend only on the query entities, not on the document, so we
    resolve each entity's connected doc-id set ONCE per query (≈O(entities) Neo4j
    calls) and then membership-test each doc in memory — previously this ran the
    same fulltext_search for every (doc, entity) pair (O(docs×entities) calls),
    which made reranking a 588-doc list prohibitively slow.
    """
    # entity → set of doc_ids connected to it in the KG (computed once)
    entity_doc_sets: List[set] = []
    for entity in query_entities:
        connected_ids: set = set()
        try:
            for er in neo4j_client.fulltext_search(entity, limit=3):
                for c in neo4j_client.get_documents_for_entity(er["id"]):
                    did = c.get("doc_id")
                    if did:
                        connected_ids.add(did)
        except Exception:
            pass
        entity_doc_sets.append(connected_ids)

    adjusted = list(scores)
    for i, doc in enumerate(documents):
        doc_id = doc.metadata.get("id") or doc.metadata.get("file_id") or ""
        if not doc_id:
            continue
        bonus = sum(bonus_per_entity for s in entity_doc_sets if doc_id in s)
        if bonus > 0:
            adjusted[i] += bonus
            doc.metadata["graph_proximity_bonus"] = bonus
    return adjusted


# =============================================================================
# Jina Reranker v3 cross-encoder + graph-proximity bonus  (new default)
# =============================================================================
class JinaCrossEncoderReranker:
    """
    Cross-encoder reranker using Jina Reranker v3 (jinaai/jina-reranker-v3).

    Replaces the per-query LLM call with deterministic local GPU inference:
      - 0.6B-parameter model based on Qwen3-0.6B
      - Last-but-not-late interaction architecture (state-of-the-art on BEIR)
      - 131K context, processes up to 64 documents in one forward pass
      - License: CC BY-NC 4.0  (non-commercial; the facility national-lab use approved)

    Keeps the graph-proximity bonus identical to GraphProximityReranker so the
    KG signal that distinguishes this RAG system is preserved.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3",
        top_k: int = 50,
        graph_proximity_bonus: float = 0.1,
        max_doc_chars: int = 2000,
        device: Optional[str] = None,
        batch_size: int = 32,
        rrf_blend: float = 0.5,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.graph_proximity_bonus = graph_proximity_bonus
        self.max_doc_chars = max_doc_chars
        self.batch_size = batch_size
        # Weight on cross-encoder relevance vs. fusion RRF in the final ranking.
        # rrf_blend=1.0 → pure cross-encoder (old behaviour, discards the adaptive
        # 3-way weights); <1.0 lets the fusion's graph-favouring weighting survive.
        self.rrf_blend = rrf_blend
        self._device = device  # None → let CrossEncoder pick (cuda if available)
        self._model = None     # lazy load — first rerank() call materialises it

    def _ensure_model(self):
        if self._model is not None:
            return
        # Jina v3 uses a custom architecture (last-but-not-late interaction with
        # a Qwen3 backbone) — sentence-transformers' generic CrossEncoder loader
        # silently mis-loads the score head and produces 0-valued scores, so we
        # use AutoModel + trust_remote_code as the model card requires.
        from transformers import AutoModel
        logger.info(
            "Loading Jina reranker %s (device=%s)...",
            self.model_name, self._device or "auto",
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, dtype="auto", trust_remote_code=True,
        )
        if self._device:
            self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("Jina reranker %s loaded", self.model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        query_entities: Optional[List[str]] = None,
        neo4j_client=None,
        query_intent: str = "general",
    ) -> List[Document]:
        if not documents:
            return []

        self._ensure_model()

        # Jina v3's rerank() takes raw strings and returns ordered dicts with
        # {index, relevance_score, document}. We need to map results back to
        # the original Document objects, so we feed truncated text and use the
        # returned `index` field to reattach metadata.
        doc_strings = [
            (doc.page_content or "")[: self.max_doc_chars] for doc in documents
        ]
        try:
            results = self._model.rerank(query, doc_strings)
            # results: list[dict] sorted by relevance — extract scores keyed by
            # original index so we can apply graph-bonus before final sort.
            scores: List[float] = [0.0] * len(documents)
            for r in results:
                idx = r.get("index")
                if idx is not None and 0 <= idx < len(documents):
                    scores[idx] = float(r.get("relevance_score", 0.0))
        except Exception as exc:
            logger.warning(
                "Jina rerank failed (%s) — falling back to RRF order", exc
            )
            scores = [doc.metadata.get("rrf_score", 0.0) for doc in documents]

        if query_entities and neo4j_client and self.graph_proximity_bonus > 0:
            scores = _apply_graph_proximity_bonus(
                documents, scores, query_entities, neo4j_client,
                bonus_per_entity=self.graph_proximity_bonus,
            )

        # Blend cross-encoder relevance with the fusion RRF score so the adaptive
        # 3-way weighting (esp. the graph channel) survives reranking instead of
        # being discarded. Both signals are min-max normalised to [0,1] first
        # because rrf_score (~1e-3) and cross-encoder scores (~0-1) differ in
        # magnitude. rrf_blend=1.0 reproduces the old pure-cross-encoder order.
        a = min(1.0, max(0.0, self.rrf_blend))   # clamp; a=1.0 → pure cross-encoder
        if a < 1.0:
            rrf = [d.metadata.get("rrf_score", 0.0) for d in documents]
            scores = [a * s + (1.0 - a) * r
                      for s, r in zip(_minmax(scores), _minmax(rrf))]

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        result: List[Document] = []
        for rank, (score, doc) in enumerate(scored[: self.top_k], start=1):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_rank"] = rank
            result.append(doc)

        logger.info(
            "JinaCrossEncoderReranker: %d → %d documents", len(documents), len(result)
        )
        return result
