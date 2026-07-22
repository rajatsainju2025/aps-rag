"""
Post-retrieval temporal filtering with escalating expansion fallback.

Mirrors entry_rag_app's temporal expansion logic (enhanced_query_pipeline.py):
  Stage 1 — apply strict ISO date filter to returned docs
  Stage 2 — if 0 docs, expand by ±30 days and re-retrieve
  Stage 3 — if still 0, drop all temporal constraints and return unfiltered

Usage:
    from src.utils.temporal_filter import apply_temporal_filter, run_with_temporal_expansion

    # Simple post-retrieval filter
    filtered = apply_temporal_filter(docs, start_iso="2026-03-01T00:00:00",
                                           end_iso="2026-03-23T23:59:59")

    # Full 3-stage fallback
    final_docs, stage = run_with_temporal_expansion(
        retrieve_fn=lambda: vector_retriever.retrieve(query, collections, top_k),
        initial_docs=retrieved_docs,
        start_iso=temporal_context["start"],
        end_iso=temporal_context["end"],
    )
"""
import logging
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Metadata field names checked (in priority order) for document datetime
_DATE_FIELDS = (
    "creation_datetime",
    "created_datetime",
    "date",
    "timestamp",
    "created_at",
    "datetime",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_iso(dt_str: str) -> Optional[datetime]:
    """
    Parse an ISO 8601 datetime string into a timezone-naive datetime object.
    Handles both 'Z' suffix and '+00:00' suffix gracefully.
    Returns None on any parse failure.
    """
    if not dt_str:
        return None
    try:
        cleaned = str(dt_str).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        # Normalize to naive UTC for consistent comparison
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, TypeError, AttributeError):
        return None


def _get_doc_datetime(doc: Document) -> Optional[datetime]:
    """
    Extract the creation datetime from a document's metadata.
    Tries multiple field names in priority order.
    """
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    for field in _DATE_FIELDS:
        raw = meta.get(field)
        if raw:
            dt = _parse_iso(str(raw))
            if dt is not None:
                return dt
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_temporal_filter(
    docs: List[Document],
    start_iso: str,
    end_iso: str,
) -> List[Document]:
    """
    Filter documents whose creation datetime falls within [start_iso, end_iso].

    Documents that have no parseable date metadata are excluded from the result
    (they cannot be confirmed to be within the intended range).

    Args:
        docs:       Retrieved LangChain Documents.
        start_iso:  ISO 8601 start datetime string (inclusive).
        end_iso:    ISO 8601 end datetime string   (inclusive).

    Returns:
        Subset of docs within the date range.  Returns all docs unchanged if
        the range strings cannot be parsed.
    """
    start_dt = _parse_iso(start_iso)
    end_dt = _parse_iso(end_iso)

    if start_dt is None or end_dt is None:
        logger.debug(
            "Temporal filter: skipping — could not parse range %r → %r",
            start_iso, end_iso,
        )
        return docs

    filtered: List[Document] = []
    no_date_count = 0

    for doc in docs:
        doc_dt = _get_doc_datetime(doc)
        if doc_dt is None:
            no_date_count += 1
            continue
        if start_dt <= doc_dt <= end_dt:
            filtered.append(doc)

    logger.info(
        "Temporal filter [%s → %s]: %d/%d docs kept  (%d had no date metadata)",
        start_iso[:10], end_iso[:10],
        len(filtered), len(docs), no_date_count,
    )
    return filtered


def expand_iso_range(
    start_iso: str,
    end_iso: str,
    days: int = 30,
) -> Tuple[str, str]:
    """
    Expand a date range by ±`days` on each side.

    Args:
        start_iso:  Original ISO start datetime string.
        end_iso:    Original ISO end datetime string.
        days:       Number of days to extend in each direction (default 30).

    Returns:
        (new_start_iso, new_end_iso) as ISO strings.
        Returns the originals unchanged if parsing fails.
    """
    start_dt = _parse_iso(start_iso)
    end_dt = _parse_iso(end_iso)

    if start_dt is None or end_dt is None:
        return start_iso, end_iso

    new_start = start_dt - timedelta(days=days)
    new_end = end_dt + timedelta(days=days)
    return new_start.isoformat(), new_end.isoformat()


def run_with_temporal_expansion(
    retrieve_fn: Callable[[], List[Document]],
    initial_docs: List[Document],
    start_iso: str,
    end_iso: str,
    expansion_days: int = 30,
) -> Tuple[List[Document], str]:
    """
    Apply post-retrieval temporal filtering with a 3-stage escalating fallback.

    Mirrors entry_rag_app's enhanced_query_pipeline temporal expansion logic.

    Stage 1 — strict:    Filter initial_docs to [start_iso, end_iso].
    Stage 2 — expanded:  If 0 docs, expand ±expansion_days, call retrieve_fn(),
                         then filter the new results.
    Stage 3 — unfiltered: If still 0, call retrieve_fn() with no temporal
                          constraint and return unfiltered results.

    Args:
        retrieve_fn:     Zero-argument callable that performs a fresh retrieval
                         (used in stages 2 & 3 — should NOT apply date filter).
        initial_docs:    Docs already retrieved in the normal flow.
        start_iso:       ISO start datetime string.
        end_iso:         ISO end datetime string.
        expansion_days:  Days to expand on each side in stage 2 (default 30).

    Returns:
        (docs, stage_used) where stage_used is "strict" | "expanded" | "unfiltered".
    """
    # ── Stage 1: strict filter ────────────────────────────────────────────
    filtered = apply_temporal_filter(initial_docs, start_iso, end_iso)
    if filtered:
        return filtered, "strict"

    logger.warning(
        "⚠️  Temporal filter: 0/%d docs in strict range %s → %s"
        " — expanding ±%d days",
        len(initial_docs), start_iso[:10], end_iso[:10], expansion_days,
    )

    # ── Stage 2: ±30-day expansion ────────────────────────────────────────
    try:
        exp_start, exp_end = expand_iso_range(start_iso, end_iso, days=expansion_days)
        logger.info(
            "   ├─ Original range: %s → %s",
            start_iso[:10], end_iso[:10],
        )
        logger.info(
            "   └─ Expanded range: %s → %s  (+%d days each direction)",
            exp_start[:10], exp_end[:10], expansion_days,
        )
        expanded_docs = retrieve_fn()
        re_filtered = apply_temporal_filter(expanded_docs, exp_start, exp_end)
        if re_filtered:
            logger.info("✅ Found %d docs in expanded range", len(re_filtered))
            return re_filtered, "expanded"
    except Exception as exc:
        logger.error("Temporal expansion stage 2 failed: %s", exc)

    # ── Stage 3: remove all temporal constraints ──────────────────────────
    logger.warning(
        "⚠️  Still 0 docs after ±%d-day expansion — removing all temporal constraints",
        expansion_days,
    )
    try:
        unfiltered = retrieve_fn()
        if unfiltered:
            logger.info(
                "✅ Found %d docs without temporal constraints", len(unfiltered)
            )
            return unfiltered, "unfiltered"
    except Exception as exc:
        logger.error("Temporal expansion stage 3 failed: %s", exc)

    # Final fallback — return whatever we had originally
    logger.warning("All temporal expansion stages exhausted — returning initial docs")
    return initial_docs or [], "unfiltered"


def build_temporal_context_block(
    start: Optional[str],
    end: Optional[str],
    stage: str = "strict",
) -> str:
    """Render a short LLM-facing block describing which temporal-expansion stage
    the retrieved docs came from.

    Returns "" when no range was requested. Otherwise returns a multi-line string
    keyed by ``stage`` (strict | expanded | unfiltered | anything-else):

      - strict:     "exact match" — docs fell inside the requested window.
      - expanded:   the window was widened ±30 days; the docs are nearest-available
                    and the LLM is told NOT to present them as if they were in-range.
      - unfiltered: no records exist in either the requested window or the ±30d
                    widening — the LLM must explicitly say so rather than imply
                    coverage.

    Called by reasoning_agent.synthesize() and naive_rag_service so the synthesis
    LLM can disclose stage-of-evidence to the user (avoids the "off-window doc
    presented as if it were today's" failure mode).
    """
    if not start or not end:
        return ""
    s10 = str(start)[:10]
    e10 = str(end)[:10]
    label = f"{s10} to {e10}"
    stage_norm = (stage or "strict").lower()

    if stage_norm == "expanded":
        return (
            f"Requested window: {label}.\n"
            f"Retrieval widened by ±30 days because the strict window returned no "
            f"documents. The evidence below is the NEAREST AVAILABLE — when reporting "
            f"dates, use each document's own date and explicitly note when evidence "
            f"falls outside the requested window. Do not present off-window docs "
            f"as if they fell in-range."
        )
    if stage_norm == "unfiltered":
        return (
            f"Requested window: {label}.\n"
            f"No documents matched the strict window or the ±30-day widening. "
            f"The evidence below has NO temporal filter — assume no entries exist "
            f"within the requested period and say so explicitly in the answer."
        )
    # Default + unknown stages: treat as strict
    return (
        f"Requested window: {label} (exact match — strict window).\n"
        f"All evidence below falls within the requested period."
    )
