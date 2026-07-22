"""
Studies Copilot Agent — accelerator physics research assistant.

Databases accessed: the system (logbook), ICMS (technical docs)

Supports:
1. Study comparison: compare current experiment notes with past experiments
2. Parameter analysis: extract and compare key parameters across studies
3. Expert routing: find relevant domain experts via KG PageRank (Neo4j GDS)
4. Improvement suggestions: reason through differences and suggest improvements
5. Live study assistance: answer questions during machine studies time
6. the system URL reading: fetch and analyse linked the logbook entries inline
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import get_prompt
from .state import AgentState

logger = logging.getLogger(__name__)

# ─── Prompts ───────────────────────────────────────────────────────────────────

# Injected into prompts when the user provides the system URL(s) in their query.
_ENTRY_ATTACHMENT_SECTION = """
<linked_entry_entries>
{entry_attachments}
</linked_entry_entries>
"""




class StudiesCopilotAgent:
    """
    Research assistant for accelerator physics studies.
    Combines KG expert routing + multi-database retrieval + comparison workflows.
    """

    def __init__(self, llm, vector_retriever, bm25_retriever,
                 neo4j_client=None):
        self.llm = llm
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.neo4j = neo4j_client

    def assist(self, state: AgentState) -> Dict[str, Any]:
        """Route to appropriate workflow based on query type."""
        query = state.get("enhanced_query") or state.get("original_query", "")

        # Detect comparison intent
        comparison_keywords = [
            "compare", "comparison", "differ", "changed", "previous",
            "last time", "before", "vs", "versus", "improvement",
        ]
        is_comparison = any(kw in query.lower() for kw in comparison_keywords)

        if is_comparison:
            return self._comparison_workflow(state)
        else:
            return self._general_assist_workflow(state)

    def _build_entry_attachment_block(self, entry_attachments: List[str]) -> str:
        """Format fetched the system pages into a prompt attachment block.

        Returns an empty string if no attachments, so the placeholder
        disappears cleanly from the prompt.
        """
        if not entry_attachments:
            return ""
        combined = "\n\n".join(entry_attachments)
        return _ENTRY_ATTACHMENT_SECTION.format(entry_attachments=combined)

    def _general_assist_workflow(self, state: AgentState) -> Dict[str, Any]:
        """Standard research assistance: retrieve studies + tech docs + experts."""
        query = state.get("enhanced_query") or state.get("original_query", "")
        entities = state.get("extracted_entities", [])
        entry_attachments: List[str] = state.get("entry_attachments", [])

        # Parallel retrieval from the system + ICMS
        study_docs, tech_docs = self._retrieve_parallel(query)

        # Get expert rankings via Neo4j GDS PageRank
        experts = self._get_experts(entities)

        messages = [
            SystemMessage(content="You are an the facility accelerator physics research copilot."),
            HumanMessage(
                content=get_prompt("STUDIES_PROMPT").format(
                    query=query,
                    entry_attachment_block=self._build_entry_attachment_block(entry_attachments),
                    docs=self._format_docs(study_docs[:10]),
                    experts=self._format_experts(experts),
                    tech_docs=self._format_docs(tech_docs[:8]),
                )
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"Studies copilot generation failed: {e}")
            answer = f"Error generating response: {e}"

        all_docs = study_docs + tech_docs
        citations = self._build_citations(all_docs)
        return {"final_answer": answer, "citations": citations, "retrieved_docs": all_docs}

    def _comparison_workflow(self, state: AgentState) -> Dict[str, Any]:
        """Compare current study with past experiments.

        Strategy:
        1. Retrieve RECENT logbook entries (last 48h) as "current study"
           (or use linked the system entries if provided)
        2. Extract topic/system keywords from current entries
        3. Retrieve HISTORICAL entries for the same topic/system
        4. Retrieve ICMS technical docs for reference
        5. LLM compares parameters and suggests improvements
        """
        query = state.get("enhanced_query") or state.get("original_query", "")
        entry_attachments: List[str] = state.get("entry_attachments", [])

        # Step 1: Get recent entries (current study)
        current_entries = self._retrieve_recent_entries(query, hours=48)
        logger.info(f"Comparison: found {len(current_entries)} recent entries")

        # Step 2: Extract key topic from current entries for historical search
        topic_query = self._extract_study_topic(query, current_entries)
        logger.info(f"Comparison: searching historical entries for topic: {topic_query[:100]}")

        # Step 3-4: Parallel retrieval of historical + ICMS
        past_entries, tech_docs = self._retrieve_historical(topic_query)
        logger.info(
            f"Comparison: found {len(past_entries)} historical entries, "
            f"{len(tech_docs)} ICMS docs"
        )

        messages = [
            SystemMessage(
                content="You are an the facility accelerator physics research copilot specializing in study comparison."
            ),
            HumanMessage(
                content=get_prompt("COMPARISON_PROMPT").format(
                    query=query,
                    entry_attachment_block=self._build_entry_attachment_block(entry_attachments),
                    current_entries=self._format_docs(current_entries[:15]),
                    past_entries=self._format_docs(past_entries[:15]),
                    tech_docs=self._format_docs(tech_docs[:8]),
                )
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"Comparison generation failed: {e}")
            answer = f"Error generating comparison: {e}"

        all_docs = current_entries + past_entries + tech_docs
        citations = self._build_citations(all_docs)
        return {"final_answer": answer, "citations": citations, "retrieved_docs": all_docs}

    # ─── Retrieval helpers ─────────────────────────────────────────────────────

    def _retrieve_parallel(self, query: str) -> tuple:
        """Retrieve from the system + ICMS in parallel."""
        study_docs: List[Document] = []
        tech_docs: List[Document] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            f_study = executor.submit(
                self._safe_vector, query, ["entry_logbook", "icms_collection"], 20
            )
            f_tech = executor.submit(
                self._safe_bm25, query, ["icms_collection", "sdds_collection", "oag_collection"], 10
            )
            for future in as_completed([f_study, f_tech]):
                try:
                    result = future.result(timeout=30)
                    if future == f_study:
                        study_docs = result
                    else:
                        tech_docs = result
                except Exception as e:
                    logger.warning(f"Studies retrieval failed: {e}")

        return study_docs, tech_docs

    def _retrieve_recent_entries(self, query: str, hours: int = 48) -> List[Document]:
        """Retrieve recent logbook entries (last N hours) as the 'current study'."""
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        start = (now - timedelta(hours=hours)).isoformat()
        end = now.isoformat()
        date_filter = {"gte": start, "lte": end}

        try:
            return self.bm25.retrieve(
                query=query,
                collections=["entry_logbook"],
                top_k=30,
                date_filter=date_filter,
            )
        except Exception as e:
            logger.warning(f"Recent entries retrieval failed: {e}")
            # Fallback: vector search without date filter
            return self._safe_vector(query, ["entry_logbook"], 20)

    def _retrieve_historical(self, topic_query: str) -> tuple:
        """Retrieve historical (older) entries and ICMS docs for comparison."""
        past: List[Document] = []
        tech: List[Document] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            f_past = executor.submit(
                self._safe_vector, topic_query, ["entry_logbook"], 30
            )
            f_tech = executor.submit(
                self._safe_bm25, topic_query, ["icms_collection"], 15
            )
            for future in as_completed([f_past, f_tech]):
                try:
                    result = future.result(timeout=30)
                    if future == f_past:
                        past = result
                    else:
                        tech = result
                except Exception as e:
                    logger.warning(f"Historical retrieval failed: {e}")

        return past, tech

    def _extract_study_topic(self, query: str, current_docs: List[Document]) -> str:
        """Extract the core topic/system from current study for historical search."""
        # Combine query with top keywords from current entries
        if not current_docs:
            return query

        # Take content snippets from first few entries to form a topic query
        snippets = []
        for doc in current_docs[:5]:
            snippets.append((doc.page_content or '')[:200])
        combined = " ".join(snippets)

        # Use LLM to extract the core study topic
        try:
            messages = [
                SystemMessage(
                    content="Extract the core topic/system being studied in 10 words or less. "
                            "Return ONLY the topic phrase, nothing else."
                ),
                HumanMessage(content=f"Query: {query}\n\nStudy notes:\n{combined[:1000]}"),
            ]
            response = self.llm.invoke(messages)
            topic = response.content.strip()
            return f"{topic} {query}"  # Combine for broader historical search
        except Exception:
            return query

    # ─── Safe retrieval wrappers ───────────────────────────────────────────────

    def _safe_vector(self, query, collections, top_k) -> List[Document]:
        try:
            return self.vector.retrieve(query, collections, top_k=top_k)
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
            return []

    def _safe_bm25(self, query, collections, top_k) -> List[Document]:
        try:
            return self.bm25.retrieve(query, collections, top_k=top_k)
        except Exception as e:
            logger.warning(f"BM25 retrieval failed: {e}")
            return []

    # ─── Experts ───────────────────────────────────────────────────────────────

    def _get_experts(self, entities: List[str]) -> List[Dict]:
        """Get top experts using Neo4j GDS PageRank and expertise routing."""
        experts = []

        # Neo4j GDS PageRank for most central Person nodes
        if self.neo4j:
            try:
                ranked = self.neo4j.pagerank()
                experts = [r for r in ranked if r.get("type") == "Person"][:5]
            except Exception:
                pass

        # Fall back to Neo4j expert query if no PageRank results
        if not experts and self.neo4j and entities:
            for entity in entities[:2]:
                try:
                    results = self.neo4j.expert_query(entity, top_n=3)
                    experts.extend(results)
                except Exception:
                    pass

        return experts

    # ─── Formatting ────────────────────────────────────────────────────────────

    def _format_docs(self, docs: List, max_docs: int = 10, start_idx: int = 1) -> str:
        """Format docs with [Source N] numbering for citation_formatter compatibility."""
        lines = []
        for i, doc in enumerate(docs[:max_docs], start=start_idx):
            doc_id = doc.metadata.get("file_id", "")
            date = str(doc.metadata.get("creation_datetime", ""))[:10]
            author = doc.metadata.get("author", "")
            collection = doc.metadata.get("collection", "")
            content = (doc.page_content or '')[:400]
            lines.append(
                f"[Source {i} ({collection}: {doc_id}, {date}, {author})]\n{content}"
            )
        return "\n\n".join(lines) if lines else "No relevant entries found."

    def _format_experts(self, experts: List[Dict]) -> str:
        if not experts:
            return "Expert information not available."
        lines = []
        for e in experts:
            name = e.get("name") or e.get("person", "unknown")
            score = e.get("rank") or e.get("doc_count", 0)
            lines.append(f"- {name} (relevance: {score})")
        return "\n".join(lines)

    def _build_citations(self, docs: List[Document]) -> List[Dict]:
        citations = []
        seen = set()
        for doc in docs[:20]:
            doc_id = doc.metadata.get("file_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                citations.append({
                    "id": doc_id,
                    "collection": doc.metadata.get("collection", ""),
                    "date": doc.metadata.get("creation_datetime", ""),
                    "author": doc.metadata.get("author", ""),
                })
        return citations
