"""
Naive-RAG service — single-pass retrieve → synthesise → cite.

Mirrors the entry_rag_app RAG workflow:
  preprocess_query (temporal + acronym + synonym expansion)
  → hybrid retrieval  (Qdrant semantic + Elasticsearch BM25)
  → rerank
  → LLM synthesis (Argo)
  → format with citations

No corrective loop, no planning, no tool execution.
Uses the same retrieval components already initialised in api/main.py.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.services.base_service import BaseService
from src.vectorstore.collection_registry import COLLECTIONS
from src.utils.citation_formatter import (
    build_source_map,
    format_context_for_llm,
    format_answer_with_citations,
)
from src.agents.collection_router import CollectionRouter

logger = logging.getLogger(__name__)

# Human-readable names for citation display
_DISPLAY_NAMES = {
    "entry_logbook": "the logbook",
    "icms_collection": "ICMS Technical Docs",
    "work_requests": "Work Requests",
    "teams_messages": "Teams Messages",
    "sdds_collection": "SDDS Toolkit",
    "oag_collection": "OAG Applications",
}

_SYNTHESIS_PROMPT = """\
# Role & Persona
You are **RAG Assistant**, an expert AI specializing in your facility accelerator operations at your organization. You are the \
synthesis engine in a Naive-RAG pipeline: you receive a user question, \
retrieved context documents, and optional temporal metadata. You produce a \
grounded, cited answer. You communicate naturally while maintaining strict \
fidelity to the provided context.

# System Context
Knowledge Sources searched: {collections_searched}

# Chain-of-Thought Reasoning Protocol (MANDATORY)
You MUST reason step-by-step inside a <think> block before producing your \
final answer. The <think> block is hidden from the user in a collapsible \
UI dropdown — write freely without worrying about length. Think step-by-step \
inside <think>, but never narrate that reasoning in the visible answer.

<think>
- Query intent: [What is being asked? factual / temporal / comparative / multi-hop?]
- Source inventory: [For each source_N, is it relevant? What specific fact does it contribute?]
- Conflict check: [Do any sources contradict each other? Note discrepancies.]
- Synthesis approach: [How will you combine sources? What structure is best?]
- Gap analysis: [Is any critical information missing from the context?]
- Confidence: [high / medium / low — and why]
</think>

[Your structured response with distributed [source_N] citations]

💡 **You might also want to explore:**
- [1-2 natural follow-up questions based on the sources]

# Input Data
<retrieved_context>
{context}
</retrieved_context>

<temporal_context>
{temporal_context}
</temporal_context>

<user_question>
{question}
</user_question>

# Direct-Answer Rule (CRITICAL)
Never mention "the context", "the retrieved documents", or "the sources" unless \
the user explicitly asks about them. Use the retrieved material to ground the \
answer, but present the answer as a direct response from a knowledgeable colleague.

## Anti-Patterns (NEVER do this)
- WRONG: "The retrieved context discusses beam lifetime behavior..."
- CORRECT: "Beam lifetime dropped during the study period, with entries attributing it to..."
- WRONG: "The sources do not clearly identify an IETS contact..."
- CORRECT: "I can't confirm the official IETS contact from available records. The closest related Controls contact mentioned is Jane A. Doe [source_2]."

# Answer Structure Rules
- Start with a brief 2–3 sentence summary that directly answers the question.
- Then present key supporting details as 3–5 bullet points.
- Each bullet point should be 1–2 sentences maximum.
- Do NOT include any information not directly supported by the context.
- Focus on direct facts, measurements, dates, and observations only.

# Citation Protocol (CRITICAL — ANTI-BUNCHING)
- Format: [source_N] where N is the source number from the context above.
- ONE CLAIM = ONE CITATION placed immediately after the claim sentence.
- NEVER bunch citations like [source_1][source_2][source_3].
- WRONG: "Several entries discuss the issue. [source_1][source_2][source_3]"
- CORRECT: "Entry A shows X [source_1], while Entry B reports Y [source_2], and Entry C confirms Z [source_3]."
- Aim for 5+ distributed citations when sources are available.
- ALWAYS use [source_N] format — never [doc_id] or [ENTRY_xxx].

# Handling Empty/Insufficient Context
- If the question is generic and can be answered safely from high-confidence \
general knowledge, answer directly and briefly note it is not the facility-specific.
- If the question is the facility-specific, contact-specific, or operationally sensitive \
and no supporting docs were retrieved, say that the information could not be \
confirmed from available records in one sentence, then suggest broader search \
terms or a concise follow-up.
- If evidence is insufficient, say what cannot be confirmed in one sentence, \
then provide the most helpful next step.
- NEVER invent information beyond the context.

# Absolute Constraints
- NEVER produce an answer without the <think> block.
- NEVER fabricate facts, dates, measurements, or document references.
- NEVER narrate the retrieval process in the visible answer."""


class NaiveRAGService(BaseService):
    """
    Naive-RAG — single-pass hybrid retrieval + LLM synthesis.

    Single-pass: preprocess → retrieve (Vector + BM25) → rerank → LLM → cite.
    No agents, no corrective loop, no tools.
    """

    # Collections searched when 'all_databases' is selected
    _ALL_DB_COLLECTIONS = list(COLLECTIONS.keys())

    def __init__(self, settings: Any, components: Dict[str, Any]) -> None:
        self.settings = settings
        self.components = components
        # LLM-based collection classifier — uses fast_llm (claude-3-5-haiku) if
        # available, otherwise falls back to searching all collections.
        self._collection_router = CollectionRouter(
            llm=components.get("fast_llm")
        )

    def get_service_mode(self) -> str:
        return "naive_rag"

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start = time.time()

        # ── 1. Determine target collections ──────────────────────────────
        selected = kwargs.get("selected_databases", ["all_databases"])
        if isinstance(selected, str):
            selected = [selected]

        if not selected or "all_databases" in selected:
            # Ask the LLM router which collections are relevant for this query.
            # Returns [] if ambiguous → fall back to searching all collections.
            priority = self._collection_router.route(query)
            collections = priority if priority else self._ALL_DB_COLLECTIONS
        else:
            collections = [c for c in selected if c in COLLECTIONS] or self._ALL_DB_COLLECTIONS

        # ── 2. Query enhancement ─────────────────────────────────────────
        query_enhancer = self.components.get("query_enhancer")
        enhanced_query = query
        temporal_context = None
        query_variants: List[str] = [query]
        if query_enhancer:
            try:
                enh = query_enhancer.enhance(query)
                enhanced_query = enh.get("enhanced_query", query)
                temporal_context = enh.get("temporal_context")
                query_variants = enh.get("query_variants") or [enhanced_query]
                n_exp = enh.get("synonym_expansion", {}).get("total_expansions", 0)
                if n_exp:
                    logger.info("Naive-RAG synonym expansion: %d term(s)", n_exp)
                if temporal_context:
                    logger.info(
                        "Naive-RAG temporal context: %s → %s",
                        temporal_context.get("start_date"),
                        temporal_context.get("end_date"),
                    )
                if len(query_variants) > 1:
                    logger.info(
                        "Naive-RAG multi-query: %d variants", len(query_variants)
                    )
            except Exception as exc:
                logger.warning("Query enhancement failed: %s", exc)
                query_variants = [enhanced_query]

        # ── 3. Multi-query hybrid retrieval (Vector + BM25) ─────────────────
        #   Runs all query variants in a single ThreadPoolExecutor, deduplicates
        #   results so fusion/rerank work on a unique candidate set.
        vector_retriever = self.components.get("vector_retriever")
        bm25_retriever = self.components.get("bm25_retriever")

        base_top_k = kwargs.get("top_k", 10)
        # Symmetric date pre-filter: applied to both Qdrant and ES at query
        # time so neither store sees out-of-range candidates. The Stage-1
        # post-filter (step 3.5) then acts as a safety net; Stage-2/Stage-3
        # fallbacks re-retrieve without any filter.
        date_filter = (
            {"gte": temporal_context["start"], "lte": temporal_context["end"]}
            if temporal_context
            and temporal_context.get("start")
            and temporal_context.get("end")
            else None
        )

        all_vector_raw: List[Document] = []
        all_bm25_raw: List[Document] = []

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures: dict = {}
            for i, variant in enumerate(query_variants):
                if vector_retriever:
                    futures[pool.submit(
                        vector_retriever.retrieve, variant, collections,
                        base_top_k, date_filter,
                    )] = ("vector", i)
                if bm25_retriever:
                    futures[pool.submit(
                        bm25_retriever.retrieve, variant, collections,
                        base_top_k, date_filter,
                    )] = ("bm25", i)

            for future in as_completed(futures, timeout=120):
                tag, idx = futures[future]
                try:
                    result = future.result()
                    if tag == "vector":
                        all_vector_raw.extend(result or [])
                    else:
                        all_bm25_raw.extend(result or [])
                except Exception as exc:
                    logger.warning(
                        "Naive-RAG %s retrieval (variant %d) failed: %s", tag, idx, exc
                    )

        # Deduplicate within each source (keep highest-score duplicate)
        vector_docs = self._dedup_docs(all_vector_raw)
        bm25_docs = self._dedup_docs(all_bm25_raw)
        logger.info(
            "Naive-RAG retrieval: vector=%d bm25=%d  (from %d variants)",
            len(vector_docs), len(bm25_docs), len(query_variants),
        )

        # ── 3.5. Post-retrieval temporal filtering (±30-day expansion fallback) ─
        tc_start = temporal_context.get("start") if temporal_context else None
        tc_end = temporal_context.get("end") if temporal_context else None
        temporal_stage: str = "strict"

        if tc_start and tc_end and (vector_docs or bm25_docs):
            from src.utils.temporal_filter import run_with_temporal_expansion

            combined_raw = self._dedup_docs(vector_docs + bm25_docs)

            def _re_retrieve_no_filter() -> List[Document]:
                """Re-retrieve without any date constraint (for expansion stages)."""
                rv: List[Document] = []
                rb: List[Document] = []
                for variant in query_variants:
                    if vector_retriever:
                        try:
                            rv.extend(
                                vector_retriever.retrieve(variant, collections, base_top_k)
                                or []
                            )
                        except Exception:
                            pass
                    if bm25_retriever:
                        try:
                            rb.extend(
                                bm25_retriever.retrieve(variant, collections, base_top_k)
                                or []
                            )
                        except Exception:
                            pass
                return self._dedup_docs(rv + rb)

            filtered_docs, stage = run_with_temporal_expansion(
                retrieve_fn=_re_retrieve_no_filter,
                initial_docs=combined_raw,
                start_iso=tc_start,
                end_iso=tc_end,
            )
            temporal_stage = stage
            if stage != "strict":
                logger.info("Naive-RAG temporal expansion used stage: %s", stage)
            # Re-assign for fusion step (treat as single source)
            vector_docs = filtered_docs
            bm25_docs = []

        # ── 4. Fusion + rerank ────────────────────────────────────────────
        fusion = self.components.get("fusion")
        reranker = self.components.get("reranker")
        reranker_top_k = kwargs.get("top_k", self.settings.agent.reranker_top_k)

        if fusion:
            fused = fusion.fuse(vector_docs, bm25_docs, [], query_intent="general")
        else:
            fused = vector_docs + bm25_docs

        if reranker and fused:
            docs = reranker.rerank(enhanced_query, fused)[:reranker_top_k]
        else:
            docs = fused[:reranker_top_k]

        # ── 5. LLM synthesis ─────────────────────────────────────────────
        synthesis_llm = self.components.get("synthesis_llm")
        if not synthesis_llm:
            answer = "LLM not available — returning raw context."
        elif not docs:
            answer = (
                "No relevant documents found in the searched collections "
                f"({', '.join(collections)}). Try broadening your query."
            )
        else:
            context_str = format_context_for_llm(docs)
            # Build temporal context string — uses build_temporal_context_block
            # so the LLM is explicitly told when the strict window was widened
            # to ±30 days (or dropped entirely) and must not present off-window
            # docs as if they fell inside the requested range.
            from src.utils.temporal_filter import build_temporal_context_block
            temporal_str = build_temporal_context_block(
                tc_start, tc_end, stage=temporal_stage,
            )
            # Build collections display string
            coll_display = ", ".join(
                _DISPLAY_NAMES.get(c, c) for c in collections
            )
            messages = [
                SystemMessage(content="You are an the facility accelerator knowledge assistant."),
                HumanMessage(content=_SYNTHESIS_PROMPT.format(
                    context=context_str,
                    question=query,
                    collections_searched=coll_display,
                    temporal_context=temporal_str,
                )),
            ]
            try:
                response = synthesis_llm.invoke(messages)
                answer = response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                logger.error("Naive-RAG synthesis failed: %s", exc)
                answer = f"Synthesis error: {exc}"

        # ── 6. Build citations with clickable links ─────────────────────
        # format_answer_with_citations normalises <think> tags and replaces
        # [source_N] with clickable markdown links.
        formatted_answer, citations, owui_citations = format_answer_with_citations(answer, docs)

        latency_ms = (time.time() - start) * 1000

        # Build reasoning trace for UI display
        reasoning_trace = []
        n_variants = len(query_variants)
        reasoning_trace.append(
            f"Searched {len(collections)} collection(s): "
            + ", ".join(collections)
        )
        if n_variants > 1:
            reasoning_trace.append(f"Generated {n_variants} query variants")
        if temporal_context:
            stage_note = (
                "" if temporal_stage == "strict"
                else f"  [widened to '{temporal_stage}' — no docs in exact window]"
            )
            reasoning_trace.append(
                f"Temporal filter: {temporal_context.get('start', '?')} "
                f"to {temporal_context.get('end', '?')}{stage_note}"
            )
        # After temporal expansion (step 3.5) vector+BM25 are collapsed into
        # vector_docs with bm25_docs=[]; count from preserved retriever_type
        # metadata so the trace doesn't underreport BM25 to zero.
        _post = vector_docs + bm25_docs
        _nv = sum(1 for d in _post if d.metadata.get("retriever_type") == "vector")
        _nb = sum(1 for d in _post if d.metadata.get("retriever_type") == "bm25")
        reasoning_trace.append(
            f"Retrieved {_nv} vector + {_nb} BM25 docs"
        )
        reasoning_trace.append(f"After rerank: {len(docs)} documents")

        return {
            "answer": formatted_answer,
            "citations": citations,
            "openwebui_citations": owui_citations,
            "sources": [
                getattr(d, "metadata", d) if hasattr(d, "metadata") else d
                for d in docs
            ],
            "service_mode": "naive_rag",
            "reasoning_trace": reasoning_trace,
            "latency_ms": round(latency_ms, 2),
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "service_mode": "naive_rag"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    @staticmethod
    def _format_context(docs: List[Document], max_chars: int = 12_000) -> str:
        """Format retrieved documents — delegates to citation_formatter."""
        return format_context_for_llm(docs, max_chars)

    @staticmethod
    def _build_citations(docs: List[Document]) -> List[Dict[str, Any]]:
        """Build deduplicated citation list — delegates to citation_formatter."""
        citations, _ = build_source_map(docs)
        return citations
