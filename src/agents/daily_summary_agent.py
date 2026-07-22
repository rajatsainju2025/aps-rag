"""
Daily Summary Agent — multi-database digest for the last 24–48 hours.

Queries each database individually with **hard date filters**:
  - the logbook:   last 24–48 hours
  - Work Requests:  last 24–48 hours
  - Teams Messages: last 24–48 hours
  - ICMS:           last 7 days  (ICMS updates are infrequent)

Then synthesises an executive-level structured summary.
"""
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import get_prompt
from .state import AgentState

logger = logging.getLogger(__name__)


class DailySummaryAgent:
    """Generates daily operations summaries from the system, ICMS, WR, and Teams."""

    def __init__(self, llm, bm25_retriever, vector_retriever):
        self.llm = llm
        self.bm25 = bm25_retriever
        self.vector = vector_retriever

    def summarize(self, state: AgentState) -> Dict[str, Any]:
        """Generate daily summary across all four databases."""
        query = state.get("original_query", "")
        temporal = state.get("temporal_context")

        # ── Resolve hard date windows ──────────────────────────────────
        now = datetime.now(timezone.utc)

        # Try to honour an explicit temporal context from the query enhancer,
        # but ALWAYS guarantee a concrete date range (never None).
        if temporal and temporal.get("start") and temporal.get("end"):
            recent_start = temporal["start"]
            recent_end = temporal["end"]
        else:
            # Default: last 48 hours
            recent_start = (now - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
            recent_end = now.strftime("%Y-%m-%dT%H:%M:%S")

        # ICMS window: always 7 days (ICMS updates are infrequent)
        icms_start = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        icms_end = now.strftime("%Y-%m-%dT%H:%M:%S")

        date_range_display = (
            f"{recent_start[:10]} to {recent_end[:10]}"
        )

        # ── Reasoning trace ────────────────────────────────────────────
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"Daily Summary: hybrid retrieval (BM25 + Qdrant) across 4 databases"
        )
        reasoning_trace.append(
            f"the system/WR/Teams window: {recent_start[:10]}–{recent_end[:10]}, "
            f"ICMS window: {icms_start[:10]}–{icms_end[:10]} (7-day lookback)"
        )

        # ── Parallel retrieval with hard date filters ──────────────────
        recent_filter = {"gte": recent_start, "lte": recent_end}
        icms_filter = {"gte": icms_start, "lte": icms_end}

        entry_docs, wr_docs, icms_docs, teams_docs = self._retrieve_all_databases(
            recent_filter, icms_filter
        )
        total = len(entry_docs) + len(icms_docs) + len(wr_docs) + len(teams_docs)

        # ── Auto-expand if the explicit temporal window yields nothing ─
        # Try progressively wider windows: +2 days → +7 days
        if total == 0 and temporal and temporal.get("start"):
            for expand_days, label in [(2, "±2 days"), (7, "±7 days")]:
                expanded_start = (
                    datetime.fromisoformat(recent_start.replace("Z", "+00:00"))
                    - timedelta(days=expand_days)
                ).strftime("%Y-%m-%dT%H:%M:%S")
                expanded_end = (
                    datetime.fromisoformat(recent_end.replace("Z", "+00:00"))
                    + timedelta(days=expand_days)
                ).strftime("%Y-%m-%dT%H:%M:%S")
                expanded_filter = {"gte": expanded_start, "lte": expanded_end}
                logger.info(
                    f"Daily summary: 0 results in {date_range_display} — "
                    f"expanding to {label}: {expanded_start[:10]}–{expanded_end[:10]}"
                )
                reasoning_trace.append(
                    f"⚠️ No results in {date_range_display} — expanding {label}"
                )
                entry_docs, wr_docs, icms_docs, teams_docs = self._retrieve_all_databases(
                    expanded_filter, icms_filter
                )
                total = len(entry_docs) + len(icms_docs) + len(wr_docs) + len(teams_docs)
                if total > 0:
                    date_range_display = f"{expanded_start[:10]} to {expanded_end[:10]} (expanded {label})"
                    break

        reasoning_trace.append(
            f"Retrieved: the system={len(entry_docs)}, WR={len(wr_docs)}, "
            f"ICMS={len(icms_docs)}, Teams={len(teams_docs)}"
        )

        if total == 0:
            reasoning_trace.append("No entries found in any database for the target window")
            return {
                "final_answer": (
                    f"No entries found across the system, Work Requests, ICMS, or Teams "
                    f"for {date_range_display}.\n\n"
                    "This could mean a quiet period or that the data pipelines "
                    "haven't synced yet today."
                ),
                "citations": [],
                "retrieved_docs": [],
                "reasoning_trace": reasoning_trace,
            }

        # ── Format entries per database ────────────────────────────────
        # Capped to keep synthesis input under Argo's effective token ceiling.
        # Retrieval top_k is unchanged; full doc set is still in retrieved_docs
        # for citations and the UI sidebar.
        entry_str = self._format_entries(entry_docs, max_entries=10)
        wr_str = self._format_wr_entries(wr_docs, max_entries=10)
        icms_str = self._format_entries(icms_docs, max_entries=5)
        teams_str = self._format_entries(teams_docs, max_entries=5)

        messages = [
            SystemMessage(
                content=(
                    "You are the Executive Operations Agent for the facility. "
                    "Synthesize ONLY the pre-retrieved entries below into a "
                    "structured operational brief. Never fabricate data. "
                    "Skip any entry outside the stated date window."
                )
            ),
            HumanMessage(
                content=get_prompt("DAILY_SUMMARY_PROMPT").format(
                    date_range=date_range_display,
                    n_entries=total,
                    n_entry=len(entry_docs),
                    n_wr=len(wr_docs),
                    n_icms=len(icms_docs),
                    n_teams=len(teams_docs),
                    entry_entries=entry_str or "No the logbook entries found for this period.",
                    wr_entries=wr_str or "No work requests found for this period.",
                    icms_entries=icms_str or "No ICMS documents found for the past week.",
                    teams_entries=teams_str or "No Teams messages found for this period.",
                )
            ),
        ]

        reasoning_trace.append("Generating executive summary via LLM")

        try:
            response = self.llm.invoke(messages)
            summary = response.content.strip()
        except Exception as e:
            logger.error(f"Daily summary generation failed: {e}")
            summary = f"Error generating summary: {e}"

        # ── Build citations (with content snippets for sidebar preview) ─
        citations = []
        for doc in (entry_docs + wr_docs + icms_docs + teams_docs)[:40]:
            citations.append({
                "id": doc.metadata.get("file_id", doc.metadata.get("WRQ_NO", "")),
                "collection": doc.metadata.get("collection", "unknown"),
                "date": doc.metadata.get(
                    "creation_datetime",
                    doc.metadata.get("SUBMISSION_DATE", ""),
                ),
                "author": doc.metadata.get(
                    "author", doc.metadata.get("SUBMITTED_BY", "")
                ),
                "content_snippet": (doc.page_content or "")[:500],
            })

        return {
            "final_answer": summary,
            "citations": citations,
            "retrieved_docs": entry_docs + wr_docs + icms_docs + teams_docs,
            "reasoning_trace": reasoning_trace,
        }

    # ------------------------------------------------------------------
    # Retrieval  (hybrid: BM25 + Qdrant vector per database)
    # ------------------------------------------------------------------
    def _retrieve_all_databases(
        self,
        recent_filter: Dict[str, str],
        icms_filter: Dict[str, str],
    ) -> tuple:
        """Retrieve from all 4 databases using BM25 + vector search in parallel.

        Runs 8 parallel retrieval calls (BM25 + vector for each of 4 databases),
        then deduplicates per database so the LLM sees a single merged set.
        """
        entry_bm25: List[Document] = []
        entry_vec: List[Document] = []
        wr_bm25: List[Document] = []
        wr_vec: List[Document] = []
        icms_bm25: List[Document] = []
        icms_vec: List[Document] = []
        teams_bm25: List[Document] = []
        teams_vec: List[Document] = []

        # Broad query for vector search — captures semantic neighbours
        summary_query = "recent operations beam faults maintenance work"

        with ThreadPoolExecutor(max_workers=8) as executor:
            # ── BM25 (Elasticsearch) — wildcard query, rely on date filter ──
            fb_entry = executor.submit(
                self._safe_bm25, "*", ["entry_logbook"], 200, recent_filter,
            )
            fb_wr = executor.submit(
                self._safe_bm25, "*", ["work_requests"], 100, recent_filter,
            )
            fb_icms = executor.submit(
                self._safe_bm25, "*", ["icms_collection"], 50, icms_filter,
            )
            fb_teams = executor.submit(
                self._safe_bm25, "*", ["teams_messages"], 50, recent_filter,
            )

            # ── Vector (Qdrant) — semantic search with date filter ──────────
            fv_entry = executor.submit(
                self._safe_vector, summary_query, ["entry_logbook"], 100, recent_filter,
            )
            fv_wr = executor.submit(
                self._safe_vector, summary_query, ["work_requests"], 50, recent_filter,
            )
            fv_icms = executor.submit(
                self._safe_vector, summary_query, ["icms_collection"], 30, icms_filter,
            )
            fv_teams = executor.submit(
                self._safe_vector, summary_query, ["teams_messages"], 30, recent_filter,
            )

            future_map = {
                fb_entry: ("bm25", "entry"), fb_wr: ("bm25", "wr"),
                fb_icms: ("bm25", "icms"), fb_teams: ("bm25", "teams"),
                fv_entry: ("vec", "entry"), fv_wr: ("vec", "wr"),
                fv_icms: ("vec", "icms"), fv_teams: ("vec", "teams"),
            }
            results: Dict[str, List[Document]] = {
                "bm25_entry": [], "bm25_wr": [], "bm25_icms": [], "bm25_teams": [],
                "vec_entry": [], "vec_wr": [], "vec_icms": [], "vec_teams": [],
            }

            for future in as_completed(future_map, timeout=90):
                src, db = future_map[future]
                key = f"{src}_{db}"
                try:
                    results[key] = future.result(timeout=60)
                except Exception as e:
                    logger.warning("DailySummary %s/%s retrieval failed: %s", src, db, e)

        # ── Merge + deduplicate per database ───────────────────────────
        entry_docs = self._dedup_docs(results["bm25_entry"] + results["vec_entry"])
        wr_docs = self._dedup_docs(results["bm25_wr"] + results["vec_wr"])
        icms_docs = self._dedup_docs(results["bm25_icms"] + results["vec_icms"])
        teams_docs = self._dedup_docs(results["bm25_teams"] + results["vec_teams"])

        logger.info(
            "DailySummary hybrid retrieval: the system=%d (bm25=%d+vec=%d), "
            "WR=%d (bm25=%d+vec=%d), ICMS=%d (bm25=%d+vec=%d), "
            "Teams=%d (bm25=%d+vec=%d)",
            len(entry_docs), len(results["bm25_entry"]), len(results["vec_entry"]),
            len(wr_docs), len(results["bm25_wr"]), len(results["vec_wr"]),
            len(icms_docs), len(results["bm25_icms"]), len(results["vec_icms"]),
            len(teams_docs), len(results["bm25_teams"]), len(results["vec_teams"]),
        )
        return entry_docs, wr_docs, icms_docs, teams_docs

    # ── Safe wrappers ──────────────────────────────────────────────────
    def _safe_bm25(
        self, query: str, collections: List[str],
        top_k: int, date_filter: Optional[Dict[str, str]],
    ) -> List[Document]:
        """BM25 (Elasticsearch) retrieval with error handling."""
        try:
            return self.bm25.retrieve(
                query=query, collections=collections,
                top_k=top_k, date_filter=date_filter,
            )
        except Exception as e:
            logger.warning("BM25 retrieval failed for %s: %s", collections, e)
            return []

    def _safe_vector(
        self, query: str, collections: List[str],
        top_k: int, date_filter: Optional[Dict[str, str]],
    ) -> List[Document]:
        """Qdrant vector retrieval with error handling."""
        try:
            return self.vector.retrieve(
                query=query, collections=collections,
                top_k=top_k, date_filter=date_filter,
            )
        except Exception as e:
            logger.warning("Vector retrieval failed for %s: %s", collections, e)
            return []

    @staticmethod
    def _dedup_docs(docs: List[Document]) -> List[Document]:
        """Deduplicate documents by ID or content prefix, keeping highest-score copy."""
        seen: Dict[Any, Document] = {}
        for doc in docs:
            key = (
                doc.metadata.get("id")
                or doc.metadata.get("doc_id")
                or doc.metadata.get("file_id")
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

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    def _format_entries(self, docs: List[Document], max_entries: int = 50) -> str:
        """Format logbook / ICMS / Teams entries with collection-specific citation tags.

        Uses prefixed tags so the LLM can directly reference entries:
          - the logbook  → [the system-{file_id}]
          - ICMS docs     → [{file_id}]  (natural ID e.g. DOC_2295284)
          - Teams messages → no individual tag (shared file_id)
        """
        lines = []
        for doc in docs[:max_entries]:
            doc_id = doc.metadata.get("file_id", "")
            date = str(doc.metadata.get("creation_datetime", ""))[:16]
            author = doc.metadata.get("author", "")
            collection = doc.metadata.get("collection", "")
            content = (doc.page_content or '')[:500]
            # Collection-specific citation tag prefix
            if collection == "entry_logbook":
                tag = f"[the system-{doc_id}]"
            elif collection == "teams_messages":
                tag = ""  # Teams entries have shared file_id; no individual tag
            else:
                tag = f"[{doc_id}]"  # ICMS uses natural ID (e.g. DOC_2295284)
            header = f"{tag} {date} ({author})" if tag else f"{date} ({author}, {collection})"
            lines.append(f"{header}\n{content}\n")
        return "\n".join(lines)

    def _format_wr_entries(self, docs: List[Document], max_entries: int = 30) -> str:
        """Format work request entries with WR-specific fields."""
        lines = []
        for doc in docs[:max_entries]:
            raw = doc.metadata.get("_raw", {}) if isinstance(doc.metadata.get("_raw"), dict) else {}
            wrq_no = (
                doc.metadata.get("WRQ_NO")
                or raw.get("WRQ_NO")
                or doc.metadata.get("file_id", "")
            )
            date = str(
                doc.metadata.get("SUBMISSION_DATE")
                or raw.get("SUBMISSION_DATE")
                or doc.metadata.get("creation_datetime", "")
            )[:16]
            submitter = (
                doc.metadata.get("SUBMITTED_BY")
                or raw.get("SUBMITTED_BY")
                or doc.metadata.get("author", "")
            )
            status = doc.metadata.get("STATUS") or raw.get("STATUS", "")
            content = (doc.page_content or '')[:400]
            lines.append(
                f"[WRQ_{wrq_no}] {date} (by {submitter}, status: {status})\n{content}\n"
            )
        return "\n".join(lines)
