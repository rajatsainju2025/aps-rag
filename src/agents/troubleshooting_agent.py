"""
Troubleshooting Agent — fault diagnosis using KG fault chains + multi-database search.

Databases accessed: the system (logbook), ICMS (technical docs), Work Requests

Pipeline:
  1. Symptom extraction (LLM)
  2. KG fault chain traversal (Neo4j)
  3. Parallel retrieval from the system + ICMS + Work Requests
  4. Runbook synthesis (LLM)
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

from src.utils.json_parser import parse_json_with_repair

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import get_prompt
from .state import AgentState

logger = logging.getLogger(__name__)




class TroubleshootingAgent:
    """
    RCA specialist using KG fault chain traversal + entity-centric graph
    retrieval + multi-database search.

    Searches the logbook, ICMS technical docs, Work Requests, and the
    Knowledge Graph (Neo4j entity-centric + fulltext + fault chains).
    """

    def __init__(self, llm, neo4j_client, vector_retriever, bm25_retriever,
                 graph_retriever=None):
        self.llm = llm
        self.neo4j = neo4j_client
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.graph_retriever = graph_retriever

    def diagnose(self, state: AgentState) -> Dict[str, Any]:
        """Full troubleshooting pipeline with deep KG integration."""
        query = state.get("enhanced_query") or state.get("original_query", "")

        # Step 1: Extract symptoms
        symptoms, systems, components = self._extract_symptoms(query)
        logger.info(f"Troubleshooting: symptoms={symptoms}, systems={systems}, components={components}")

        # Step 2a: KG fault chain traversal (Fault→RESOLVED_BY→Procedure)
        kg_faults = self._traverse_fault_chains(symptoms + systems + components)

        # Step 2b: KG entity-centric retrieval (fulltext → neighbours → docs)
        kg_entity_docs = self._retrieve_kg_entities(symptoms + systems + components)

        # Step 2c: KG fulltext search for broader context
        kg_fulltext_results = self._fulltext_search_kg(symptoms + systems + components)

        # Step 2d: Expert lookup for affected systems
        kg_experts = self._find_experts(systems)

        # Step 3: Parallel multi-database retrieval
        search_query = " ".join(symptoms + systems + components)
        entry_docs, icms_docs, wr_docs = self._retrieve_all_databases(search_query)

        logger.info(
            f"Troubleshooting retrieval: the system={len(entry_docs)}, "
            f"ICMS={len(icms_docs)}, WR={len(wr_docs)}, "
            f"KG faults={len(kg_faults)}, KG entities={len(kg_entity_docs)}, "
            f"KG fulltext={len(kg_fulltext_results)}, KG experts={len(kg_experts)}"
        )

        # Step 4: Generate runbook
        # Bound every variable-length section to keep total synthesis input
        # under Argo's effective token ceiling. kg_faults is the biggest
        # driver — complex queries (lifetime, BLM, timing) return many chains.
        messages = [
            SystemMessage(content=(
                "You are the senior the facility accelerator on-call diagnostician. "
                "Operators execute your output during active faults. "
                "Be terse, cite every claim, never fabricate. Safety first."
            )),
            HumanMessage(
                content=get_prompt("RUNBOOK_PROMPT").format(
                    symptoms=", ".join(symptoms),
                    systems=", ".join(systems),
                    kg_faults=self._format_kg_faults(kg_faults[:5])[:4000],
                    kg_entities=self._format_kg_entities(kg_fulltext_results[:10]),
                    kg_experts=self._format_experts(kg_experts),
                    entry_docs=self._format_docs(entry_docs[:6], "the system"),
                    icms_docs=self._format_docs(icms_docs[:5], "ICMS"),
                    wr_docs=self._format_wr_docs(wr_docs[:5]),
                )
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            runbook = response.content.strip()
        except Exception as e:
            logger.error(f"Runbook generation failed: {e}")
            runbook = f"Error generating runbook: {e}"

        # Combine all retrieved docs for citation formatting
        all_docs = entry_docs + icms_docs + wr_docs + kg_entity_docs

        return {
            "final_answer": runbook,
            "retrieved_docs": all_docs[:20],
            "graph_context": kg_faults,
        }

    def _extract_symptoms(self, query: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract symptoms, systems, components from query using fast LLM."""
        messages = [
            SystemMessage(content=(
                "You are an the facility accelerator fault diagnosis expert. Extract structured "
                "symptom metadata from fault reports. Respond with a <think> "
                "analysis block then JSON only."
            )),
            HumanMessage(content=get_prompt("SYMPTOM_EXTRACTION_PROMPT").format(query=query)),
        ]
        try:
            response = self.llm.invoke(messages)
            raw = response.content.strip()
            data = parse_json_with_repair(raw, llm=self.llm, context="symptom_extraction")
            if not isinstance(data, dict):
                data = {}
            severity = data.get("severity", "MEDIUM")
            logger.info(f"Symptom extraction: severity={severity}, "
                        f"symptoms={data.get('symptoms', [])}")
            return (
                data.get("symptoms", [query]),
                data.get("systems", []),
                data.get("components", []),
            )
        except Exception:
            return [query], [], []

    def _traverse_fault_chains(self, keywords: List[str]) -> List[Dict]:
        """Traverse KG fault chains for matching symptoms."""
        if not self.neo4j:
            return []
        try:
            return self.neo4j.fault_chain_query(keywords[:5])
        except Exception as e:
            logger.warning(f"KG fault traversal failed: {e}")
            return []

    def _retrieve_kg_entities(self, keywords: List[str]) -> List[Document]:
        """Entity-centric KG retrieval via graph_retriever.

        Finds entity nodes matching keywords, then retrieves connected documents.
        """
        if not self.graph_retriever:
            return []
        try:
            return self.graph_retriever.retrieve_entity_centric(keywords[:8], top_k=10)
        except Exception as e:
            logger.warning(f"KG entity retrieval failed: {e}")
            return []

    def _fulltext_search_kg(self, keywords: List[str]) -> List[Dict]:
        """Fulltext search across KG node names and descriptions.

        Returns related entities (systems, components, faults, procedures)
        even when no Fault→RESOLVED_BY→Procedure path exists.
        """
        if not self.neo4j:
            return []
        results = []
        seen_ids = set()
        for kw in keywords[:5]:
            try:
                hits = self.neo4j.fulltext_search(kw, limit=5)
                for h in hits:
                    eid = h.get("id", "")
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        results.append(h)
            except Exception as e:
                logger.warning(f"KG fulltext search failed for '{kw}': {e}")
        return results[:15]

    def _find_experts(self, systems: List[str]) -> List[Dict]:
        """Find domain experts for the affected systems via KG."""
        if not self.neo4j or not systems:
            return []
        experts = []
        for system in systems[:3]:
            try:
                hits = self.neo4j.expert_query(system, top_n=3)
                experts.extend(hits)
            except Exception as e:
                logger.warning(f"KG expert query failed for '{system}': {e}")
        return experts

    def _retrieve_all_databases(self, query: str) -> Tuple[List[Document], List[Document], List[Document]]:
        """Parallel retrieval from the system, ICMS, and Work Requests."""
        entry_docs: List[Document] = []
        icms_docs: List[Document] = []
        wr_docs: List[Document] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            f_entry = executor.submit(
                self._safe_retrieve, self.vector.retrieve,
                query, ["entry_logbook"], 15,
            )
            f_icms = executor.submit(
                self._safe_retrieve, self.bm25.retrieve,
                query, ["icms_collection"], 10,
            )
            f_wr = executor.submit(
                self._safe_retrieve, self.bm25.retrieve,
                query, ["work_requests"], 10,
            )

            for future in as_completed([f_entry, f_icms, f_wr]):
                try:
                    result = future.result(timeout=30)
                    if future == f_entry:
                        entry_docs = result
                    elif future == f_icms:
                        icms_docs = result
                    else:
                        wr_docs = result
                except Exception as e:
                    logger.warning(f"Troubleshooting retrieval failed: {e}")

        return entry_docs, icms_docs, wr_docs

    def _safe_retrieve(self, retriever_fn, query, collections, top_k):
        try:
            return retriever_fn(query, collections, top_k=top_k) or []
        except Exception as e:
            logger.warning(f"Retriever failed for {collections}: {e}")
            return []

    def _format_kg_faults(self, faults: List[Dict]) -> str:
        if not faults:
            return "No matching fault patterns found in knowledge graph."
        lines = []
        for f in faults:
            lines.append(
                f"Fault: {f.get('fault')}\n"
                f"Description: {f.get('fault_desc', 'N/A')}\n"
                f"Procedure: {f.get('procedure')}\n"
                f"Steps: {f.get('steps', 'N/A')}\n"
                f"Frequency: {f.get('frequency', 'unknown')}"
            )
        return "\n\n".join(lines)

    def _format_kg_entities(self, entities: List[Dict]) -> str:
        """Format fulltext KG search results."""
        if not entities:
            return "No related entities found in knowledge graph."
        lines = []
        for e in entities:
            lines.append(
                f"- {e.get('type', 'Entity')}: {e.get('name', 'unknown')} "
                f"(relevance: {e.get('score', 0):.2f})"
            )
        return "\n".join(lines)

    def _format_experts(self, experts: List[Dict]) -> str:
        """Format expert query results."""
        if not experts:
            return "No domain experts identified for these systems."
        lines = []
        for ex in experts:
            lines.append(
                f"- {ex.get('person', 'unknown')} "
                f"({ex.get('doc_count', 0)} related log entries)"
            )
        return "\n".join(lines)

    def _format_docs(self, docs: List, label: str, max_docs: int = 8) -> str:
        lines = []
        for doc in docs[:max_docs]:
            doc_id = doc.metadata.get("file_id", "")
            date = str(doc.metadata.get("creation_datetime", ""))[:10]
            author = doc.metadata.get("author", "")
            content = (doc.page_content or '')[:400]
            lines.append(f"[{doc_id}] ({date}, {author})\n{content}")
        return "\n\n".join(lines) if lines else f"No {label} documents found."

    def _format_wr_docs(self, docs: List, max_docs: int = 6) -> str:
        lines = []
        for doc in docs[:max_docs]:
            wrq = doc.metadata.get("WRQ_NO", doc.metadata.get("file_id", ""))
            status = doc.metadata.get("STATUS", "")
            date = str(doc.metadata.get("SUBMISSION_DATE", doc.metadata.get("creation_datetime", "")))[:10]
            content = (doc.page_content or '')[:300]
            lines.append(f"[WR-{wrq}] ({date}, status: {status})\n{content}")
        return "\n\n".join(lines) if lines else "No related work requests found."
