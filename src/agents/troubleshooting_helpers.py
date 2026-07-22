"""
Troubleshooting-only helpers (scoped to AI Help / troubleshooting service mode).

Nothing in this module is imported by other agents. Keeping these utilities
isolated avoids regressing naive_rag / agentic_rag / daily_summary / studies_copilot
when we tighten the diagnostic flow.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recency window
# ---------------------------------------------------------------------------

# Default live window: 7 days. A 24h window misses the bulk of fault history —
# routine ops alone produce <10 incidents/day — so any "what is failing / what's
# trending" question would have ≤1 data point. 7 days gives ~30-50 incidents,
# enough to surface real patterns without dragging in stale evidence.
_DEFAULT_LIVE_WINDOW_HOURS = 7 * 24
# Trend / predictive queries need monthly view to see real recurrence patterns.
_TREND_ANALYSIS_WINDOW_HOURS = 30 * 24
_POST_EVENT_WINDOW_HOURS = {
    "access": 7 * 24,        # post-access recovery often spans a few days
    "maintenance": 7 * 24,
    "reboot": 12,
    "none": _DEFAULT_LIVE_WINDOW_HOURS,
}


def resolve_recency_window(
    temporal_ctx: Optional[Dict[str, Any]],
    state_temporal: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    intent_flags: Optional[Dict[str, bool]] = None,
) -> Tuple[str, str, str]:
    """Pick the (since_iso, until_iso, human_label) window for the recency probe.

    Precedence:
      1. Explicit state["temporal_context"] start/end if present.
      2. Predictive intent (intent_flags["trend_analysis"]) → 30-day window.
      3. symptom-extraction temporal_context.post_event window.
      4. Default 7-day live window.

    Returns ISO 8601 strings without timezone (matches retriever date_filter shape).
    """
    now = now or datetime.utcnow()

    if state_temporal and state_temporal.get("start") and state_temporal.get("end"):
        start = str(state_temporal["start"])
        end = str(state_temporal["end"])
        return start, end, f"between {start[:10]} and {end[:10]}"

    # Predictive / trend analysis queries need ~30 days of fault history to
    # surface real recurrence patterns. Beats both default (7d) and post-event
    # windows.
    if intent_flags and intent_flags.get("trend_analysis"):
        since = now - timedelta(hours=_TREND_ANALYSIS_WINDOW_HOURS)
        return _iso(since), _iso(now), f"last {_TREND_ANALYSIS_WINDOW_HOURS // 24}d (trend window)"

    post_event = "none"
    if temporal_ctx:
        post_event = str(temporal_ctx.get("post_event") or "none").lower()
    hours = _POST_EVENT_WINDOW_HOURS.get(post_event, _DEFAULT_LIVE_WINDOW_HOURS)
    since = now - timedelta(hours=hours)

    if post_event == "none":
        days = hours // 24
        label = f"last {days}d" if days >= 1 else f"last {hours}h"
    else:
        label = f"since {post_event} ({hours}h window)"
    return _iso(since), _iso(now), label


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Work-request status filtering
# ---------------------------------------------------------------------------

_OPEN_STATUS_TOKENS = {
    "open", "active", "in progress", "in-progress", "inprogress",
    "assigned", "new", "pending", "submitted", "approved", "scheduled",
    "ready", "work in progress",
}
_CLOSED_STATUS_TOKENS = {
    "closed", "completed", "done", "cancelled", "canceled", "rejected",
    "withdrawn", "void", "abandoned",
}


def is_open_wr(doc: Document) -> bool:
    """True iff the WR document's STATUS metadata reads as still-open.

    Unknown / missing status → treated as open (operators would rather see an
    extra WR than miss an active one)."""
    status = (doc.metadata.get("STATUS") or doc.metadata.get("status") or "").strip().lower()
    if not status:
        return True
    if status in _CLOSED_STATUS_TOKENS:
        return False
    if status in _OPEN_STATUS_TOKENS:
        return True
    # Heuristic for compound statuses like "open - awaiting parts"
    return not any(tok in status for tok in _CLOSED_STATUS_TOKENS)


def filter_open_wrs(docs: Iterable[Document]) -> List[Document]:
    return [d for d in docs if is_open_wr(d)]


def tag_wr_openness(docs: Iterable[Document]) -> List[Document]:
    """Annotate metadata with is_open boolean; useful for the historical WR block."""
    out: List[Document] = []
    for d in docs:
        d.metadata["is_open"] = is_open_wr(d)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Recency-decayed re-scoring (troubleshooting only)
# ---------------------------------------------------------------------------

_DECAY_HALF_LIFE_DAYS = 14
_DECAY_DATE_FIELDS = (
    "creation_datetime", "created_datetime", "date",
    "SUBMISSION_DATE", "timestamp", "created_at",
)


def _doc_age_days(doc: Document, now: datetime) -> Optional[float]:
    """Pull a date from the doc's metadata and return age in days. None on miss."""
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    for field in _DECAY_DATE_FIELDS:
        raw = meta.get(field)
        if not raw:
            continue
        try:
            cleaned = str(raw).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return max(0.0, (now - dt).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            continue
    return None


def rerank_by_recency(
    docs: List[Document],
    half_life_days: float = _DECAY_HALF_LIFE_DAYS,
    now: Optional[datetime] = None,
    score_field: str = "score",
    keep_undated: bool = True,
) -> List[Document]:
    """Stable-sort docs by score * exp(-age_days / half_life).

    - Docs with no parseable date keep their original score if keep_undated, else
      get score 0 and sink to the bottom.
    - The recency-weighted score is written back to metadata['recency_score'] for
      observability; the original score is preserved.
    - Returns a NEW list; input is not mutated except for the recency_score tag.
    """
    if not docs:
        return docs
    now = now or datetime.utcnow()
    import math

    decorated = []
    for d in docs:
        base = float((d.metadata or {}).get(score_field, 0.0) or 0.0)
        age = _doc_age_days(d, now)
        if age is None:
            weighted = base if keep_undated else 0.0
        else:
            weighted = base * math.exp(-age / max(1.0, half_life_days))
        d.metadata["recency_score"] = weighted
        d.metadata["age_days"] = age
        decorated.append((weighted, d))

    decorated.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in decorated]


# ---------------------------------------------------------------------------
# Recent-activity formatting (for the <recent_activity> prompt block)
# ---------------------------------------------------------------------------

def format_recent_activity(
    entry_recent: List[Document],
    wr_open: List[Document],
    icms_recent: List[Document],
    window_label: str,
) -> str:
    if not (entry_recent or wr_open or icms_recent):
        return f"No recent activity matching these symptoms {window_label}."

    lines = [f"Window: {window_label}"]

    if entry_recent:
        lines.append("the system (recent entries):")
        for d in entry_recent[:5]:
            bid = d.metadata.get("entry_id") or d.metadata.get("doc_id") or d.metadata.get("file_id") or "?"
            date = str(d.metadata.get("creation_datetime", ""))[:10]
            author = d.metadata.get("author", "")
            snippet = (d.page_content or "")[:160].replace("\n", " ")
            lines.append(f"  - [{bid}] ({date}, {author}) {snippet}")
    else:
        lines.append("the system: no recent entries in window.")

    if wr_open:
        lines.append("Open Work Requests:")
        for d in wr_open[:3]:
            wrq = d.metadata.get("WRQ_NO") or d.metadata.get("doc_id") or "?"
            status = d.metadata.get("STATUS", "unknown")
            date = str(d.metadata.get("SUBMISSION_DATE") or d.metadata.get("creation_datetime", ""))[:10]
            snippet = (d.page_content or "")[:140].replace("\n", " ")
            lines.append(f"  - [WR-{wrq}] ({date}, status: {status}) {snippet}")
    else:
        lines.append("Open WRs: none matching in window.")

    if icms_recent:
        lines.append("ICMS / procedure activity:")
        for d in icms_recent[:2]:
            icms_id = d.metadata.get("file_id") or d.metadata.get("doc_id") or "?"
            date = str(d.metadata.get("creation_datetime", ""))[:10]
            snippet = (d.page_content or "")[:140].replace("\n", " ")
            lines.append(f"  - [{icms_id}] ({date}) {snippet}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expert formatting (for recency-aware Cypher results)
# ---------------------------------------------------------------------------

def experts_from_doc_authors(
    docs: Iterable[Document],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Fallback when the KG returns no experts: count authors in retrieved docs.

    Used by the troubleshooting agent when both
    `expert_query_by_recent_authorship` and the legacy `expert_query` produce
    nothing. The docs passed here should be the *recent* the system entries (the
    ones the agent already trusts) so the recommendation tracks who is
    actually touching this system right now.

    Returns rows with the same shape as expert_query_by_recent_authorship:
      {"person", "recent_count", "sample_ids", "last_entry_date"}
    """
    from collections import Counter
    by_author: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        meta = d.metadata or {}
        author = (meta.get("author") or "").strip()
        if not author:
            continue
        entry_id = (
            meta.get("entry_id") or meta.get("doc_id")
            or meta.get("file_id") or meta.get("parent_id") or ""
        )
        date = str(meta.get("creation_datetime") or "")[:10]
        slot = by_author.setdefault(author, {
            "person": author,
            "recent_count": 0,
            "sample_ids": [],
            "last_entry_date": "",
        })
        slot["recent_count"] += 1
        if entry_id and entry_id not in slot["sample_ids"]:
            slot["sample_ids"].append(str(entry_id))
        if date and date > slot["last_entry_date"]:
            slot["last_entry_date"] = date

    ranked = sorted(
        by_author.values(),
        key=lambda r: (r["recent_count"], r["last_entry_date"]),
        reverse=True,
    )
    for r in ranked:
        r["sample_ids"] = r["sample_ids"][:3]
    return ranked[:top_n]


def format_experts_with_evidence(experts: List[Dict[str, Any]]) -> str:
    if not experts:
        return "No domain experts identified with recent activity on these systems."
    lines = []
    for ex in experts:
        name = ex.get("person") or ex.get("name") or "unknown"
        count = ex.get("recent_count") or ex.get("doc_count") or 0
        samples = ex.get("sample_ids") or []
        sample_str = f" — sample: {', '.join(samples[:3])}" if samples else ""
        lines.append(f"- {name} ({count} recent related entries){sample_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Citation validator — strips fabricated IDs from the final runbook
# ---------------------------------------------------------------------------

# Token shapes seen in the (bad) wild + the runbook prompt's canonical forms
_ID_PATTERNS = (
    re.compile(r"\bENTRY[_-]?(\d{2,6})\b", re.IGNORECASE),
    re.compile(r"\bWR[Q]?[_-]?(\d{4,8})\b", re.IGNORECASE),
    re.compile(r"\bICMS[ _-]?([A-Z]{2,4}[_-]?\d{3,8})\b", re.IGNORECASE),
    re.compile(r"\bAPS[_-](\d{4,8})\b", re.IGNORECASE),
)
_SOURCE_TAG = re.compile(r"\[source_(\d+)\]")

# "Bare" = the id is NOT immediately preceded by `[` and not followed by `]`.
# We need a lookbehind/lookahead-friendly version per ID family.
_BARE_ENTRY = re.compile(r"(?<!\[)\bENTRY[_-]?\d{2,6}\b(?!\])", re.IGNORECASE)
_BARE_WR = re.compile(r"(?<!\[)\bWR[Q]?[_-]?\d{4,8}\b(?!\])", re.IGNORECASE)
_BARE_ICMS = re.compile(
    r"(?<!\[)\bICMS[ _-]?[A-Z]{2,4}[_-]?\d{3,8}\b(?!\])", re.IGNORECASE
)

# Lines that look like the LLM-generated trailing legend:
#   "ENTRY_10756: recent_activity WR-62374 (2026-05-22)"
#   "ENTRY_10726: kg_fault_chains — 'no beam in linac'"
# These mirror the prompt's evidence-block names back at us. Any line whose
# first token is an ID followed by ":" and one of the known evidence-block
# slugs is treated as legend noise and excised.
_EVIDENCE_SLUGS = (
    "recent_activity",
    "kg_fault_chains",
    "kg_related_entities",
    "kg_entities",
    "domain_experts",
    "entry_logbook_entries",
    "icms_technical_docs",
    "work_request_records",
    "entry_docs",
    "icms_docs",
    "wr_docs",
    "reported_symptoms",
    "affected_systems",
)
_LEGEND_LINE = re.compile(
    r"^\s*\[?(?:the system|WR[Q]?|ICMS|the facility)[_\- ]?\w*\]?\s*[:\-—]\s*(?:"
    + "|".join(re.escape(s) for s in _EVIDENCE_SLUGS)
    + r")\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _legitimate_ids(docs: List[Document]) -> set[str]:
    """Build the allow-list of citation IDs from retrieved doc metadata."""
    allowed: set[str] = set()
    for d in docs:
        meta = d.metadata or {}
        for key in ("entry_id", "ENTRY_ID", "WRQ_NO", "wrq_no", "file_id", "doc_id", "parent_id"):
            v = meta.get(key)
            if not v:
                continue
            s = str(v).strip()
            # Strip __chunk_N suffix (granular chunking)
            s = re.sub(r"__chunk_\d+$", "", s)
            allowed.add(s.lower())
            # Numeric tail
            for m in re.finditer(r"\d{2,8}", s):
                allowed.add(m.group(0))
    return allowed


def _excise_trailing_legend(text: str) -> Tuple[str, int]:
    """Remove LLM-emitted legend lines that mirror evidence-block names.

    Returns (cleaned_text, lines_removed).
    """
    new, n = _LEGEND_LINE.subn("", text)
    if n:
        # Collapse the orphan blank lines the removal leaves behind
        new = re.sub(r"\n{3,}", "\n\n", new).rstrip() + "\n"
    return new, n


def _enforce_brackets(text: str, allowed_ids: set[str], stripped: List[str]) -> str:
    """Pass over `text` rewriting bare IDs: legitimate ones get wrapped in
    `[...]`, illegitimate ones get replaced with `[unverified]`.

    Bracketed IDs are intentionally untouched by these patterns (the regex
    uses negative lookbehind/lookahead for `[` / `]`).
    """
    def _check_and_wrap(m: re.Match) -> str:
        token = m.group(0)
        # Pull numeric tail for the allow-list comparison
        tail = re.search(r"\d{2,8}", token)
        tail_str = tail.group(0).lower() if tail else ""
        if tail_str and (tail_str in allowed_ids or token.lower() in allowed_ids):
            return f"[{token}]"
        stripped.append(token)
        return "[unverified]"

    out = text
    for pat in (_BARE_ENTRY, _BARE_WR, _BARE_ICMS):
        out = pat.sub(_check_and_wrap, out)
    return out


def validate_citations(
    runbook_text: str,
    retrieved_docs: List[Document],
    max_source_n: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Clean up citations in the runbook before returning to the operator.

    Three passes (all troubleshooting-only):

    1. Excise the trailing "legend" block the LLM sometimes appends, where
       each line maps an ID to one of the prompt's evidence-block slugs
       (`recent_activity`, `kg_fault_chains`, etc.). These mirror the prompt
       structure back and have caused misattribution in production.
    2. For any IDs *inside* `[...]` brackets: replace illegitimate ones
       (not in any retrieved doc) with `[unverified]`. Legitimate brackets
       are preserved untouched.
    3. For any IDs OUTSIDE brackets (bare): if the ID is legitimate, wrap it
       in `[...]` so the downstream renderer can hyperlink it; if not, replace
       with `[unverified]`. Bare IDs are a prompt-rule violation regardless.

    Returns (cleaned_text, list_of_stripped_or_invalid_tokens).
    """
    if not runbook_text:
        return runbook_text, []

    allowed_ids = _legitimate_ids(retrieved_docs)
    stripped: List[str] = []

    # Pass 1: drop the trailing legend block
    out, legend_lines = _excise_trailing_legend(runbook_text)
    if legend_lines:
        logger.warning("validate_citations excised %d legend line(s)", legend_lines)
        stripped.append(f"<legend:{legend_lines}_lines>")

    # Pass 2: validate IDs that are already bracketed
    def _replace_bracketed(m: re.Match) -> str:
        token = m.group(0)
        tail = m.group(1).lower()
        if tail in allowed_ids or token.lower() in allowed_ids:
            return token
        stripped.append(token)
        return "[unverified]"
    for pat in _ID_PATTERNS:
        out = pat.sub(_replace_bracketed, out)

    # Pass 3: enforce that bare IDs are either wrapped or stripped
    out = _enforce_brackets(out, allowed_ids, stripped)

    # Pass 4: clamp out-of-range [source_N] markers
    if max_source_n is not None and max_source_n >= 0:
        def _replace_src(m: re.Match) -> str:
            try:
                n = int(m.group(1))
            except (ValueError, TypeError):
                return m.group(0)
            if n > max_source_n or n <= 0:
                stripped.append(m.group(0))
                return "[unverified]"
            return m.group(0)
        out = _SOURCE_TAG.sub(_replace_src, out)

    if stripped:
        logger.warning(
            "validate_citations stripped %d item(s): %s",
            len(stripped), stripped[:8],
        )

    return out, stripped
