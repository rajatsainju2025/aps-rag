"""
Citation formatter for Agentic GraphRAG responses.

Mirrors the entry_rag_app citation system:
  1. LLM is prompted to emit [source_N] markers inline
  2. Post-processing replaces markers with collection-prefixed IDs
  3. A formatted source section is appended with clickable links

Supported collections and their URL patterns:
  - entry_logbook   → https://your-host.example.com/entry/views/itemDomainLogbook/view?id={entry_id}
  - icms_collection → https://your-docs-host.example.com{file_url}
  - oag_collection  → SVN link from metadata or constructed URL
  - work_requests   → in-process /api/entries/work_requests/{id}.html
  - teams_messages  → in-process /api/entries/teams_messages/{id}.html
  - sdds_collection → in-process /api/entries/sdds_collection/{id}.html
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Base URL for the in-process /api/entries route. Overridable via env so the
# formatter can produce absolute URLs even when called outside an HTTP request
# context (e.g. background eval scripts).
_API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8003").rstrip("/")

# ─────────────────────────────────────────────────────────────────────────────
# URL patterns per collection
# ─────────────────────────────────────────────────────────────────────────────
_URL_PATTERNS = {
    "entry_logbook": "https://your-host.example.com/entry/views/itemDomainLogbook/view?id={entry_id}",
    "icms_collection": "https://your-docs-host.example.com{file_url}",
    "oag_collection": "https://svn.example.com/repo/apps/{app_name}",
    "work_requests": "{api_base}/api/entries/work_requests/{entry_id}.html",
    "teams_messages": "{api_base}/api/entries/teams_messages/{entry_id}.html",
    "sdds_collection": "{api_base}/api/entries/sdds_collection/{entry_id}.html",
}

# Collection display names
_DISPLAY_NAMES = {
    "entry_logbook": "the logbook",
    "icms_collection": "ICMS Technical Docs",
    "work_requests": "Work Requests",
    "teams_messages": "Teams Messages",
    "sdds_collection": "SDDS Toolkit",
    "oag_collection": "OAG Applications",
}

# ID prefix per collection
_ID_PREFIXES = {
    "entry_logbook": "the system",
    "icms_collection": "ICMS",
    "work_requests": "WRQ",
    "teams_messages": "Teams",
    "sdds_collection": "SDDS",
    "oag_collection": "OAG",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_source_map(docs: List[Document]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Build a deduplicated list of citation entries and a source_N → display_id map.

    Returns:
        (citations, source_map)
        - citations: list of dicts with id, display_id, collection, title, date,
          author, url, content_snippet
        - source_map: {"source_1": "ENTRY_3714", "source_2": "ICMS_DOC_123", ...}
    """
    citations: List[Dict[str, Any]] = []
    source_map: Dict[str, str] = {}
    seen: set = set()

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        collection = meta.get("collection", "unknown")
        raw_id = (
            meta.get("file_id")
            or meta.get("entry_id")
            or meta.get("id")
            or f"doc_{i}"
        )

        # ── Fix: strip __chunk_N suffix from granular-chunked IDs ──────
        # e.g. "515__chunk_0" → "515" so URL becomes ?id=515 (valid)
        if "__chunk" in str(raw_id):
            raw_id = str(raw_id).split("__chunk")[0]

        # ── Fix: Teams docs all share file_id="teams_messages" ─────────
        # Use author + datetime as dedup key so each message gets its own citation
        if collection == "teams_messages":
            dedup_key = (
                f"teams_{meta.get('author', '')}_{meta.get('creation_datetime', '')}"
            )
        else:
            dedup_key = raw_id

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        collection = meta.get("collection", "unknown")
        display_id = _make_display_id(raw_id, collection, meta)
        url = _make_url(raw_id, collection, meta)

        idx = len(citations) + 1
        source_key = f"source_{idx}"
        source_map[source_key] = display_id

        content = doc.page_content if hasattr(doc, "page_content") else ""

        entry: Dict[str, Any] = {
            "index": idx,
            "id": raw_id,
            "display_id": display_id,
            "collection": collection,
            "collection_display": _DISPLAY_NAMES.get(collection, collection),
            "title": meta.get("title", meta.get("entry_filename", "")),
            "date": meta.get("creation_datetime", meta.get("entry_date", "")),
            "author": meta.get("author", ""),
            "url": url,
            "content_snippet": content[:500] if content else "",
        }

        if collection == "work_requests":
            raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
            entry["status"] = raw.get("STATUS", meta.get("status", ""))
            entry["machine"] = raw.get("MACHINE", meta.get("system", ""))
            entry["wrq_no"] = raw.get("WRQ_NO", raw_id)

        citations.append(entry)

    return citations, source_map


def format_context_for_llm(docs: List[Document], max_chars: int = 12_000) -> str:
    """Format retrieved documents into numbered context for the LLM.

    Each document is labelled as [Source N (COLLECTION: PREFIX_ID)] so the
    LLM can reference them with [source_N] in its response.
    """
    parts = []
    total = 0
    seen: set = set()

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        raw_id = (
            meta.get("file_id")
            or meta.get("entry_id")
            or meta.get("id")
            or f"doc_{i}"
        )

        # ── Strip __chunk_N suffix to align with build_source_map() ──
        if "__chunk" in str(raw_id):
            raw_id = str(raw_id).split("__chunk")[0]

        # ── Teams dedup: use author+datetime (same as build_source_map) ──
        collection = meta.get("collection", "unknown")
        if collection == "teams_messages":
            dedup_key = (
                f"teams_{meta.get('author', '')}_{meta.get('creation_datetime', '')}"
            )
        else:
            dedup_key = raw_id

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        display_id = _make_display_id(raw_id, collection, meta)
        coll_display = _DISPLAY_NAMES.get(collection, collection)

        date_str = meta.get("creation_datetime", meta.get("entry_date", ""))
        date_part = f", Date: {date_str}" if date_str else ""
        author = meta.get("author", "")
        author_part = f", Author: {author}" if author else ""

        snippet = (doc.page_content or "")[:1500] if hasattr(doc, "page_content") else ""

        entry = (
            f"[Source {i} ({coll_display}: {display_id}{date_part}{author_part})]\n"
            f"{snippet}"
        )
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n\n".join(parts)


def replace_source_markers(
    answer: str,
    source_map: Dict[str, str],
    url_map: Optional[Dict[str, Optional[str]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Replace [source_N] *and* [DISPLAY_ID] markers with clickable markdown links.

    Despite explicit prompt instructions, LLMs sometimes emit inline
    citations using the raw display ID (e.g. ``[DOC_2295284]``,
    ``[ENTRY_10069]``) instead of the ``[source_N]`` format.  This
    function handles both:

      [source_1]            → [ENTRY_3714](https://your-host.example.com/...?id=3714)
      [DOC_2295284]         → [DOC_2295284](https://your-docs-host.example.com/...)
      [ENTRY_10069]          → [ENTRY_10069](https://your-host.example.com/...?id=10069)
      [source_2]  (no URL)  → [ICMS_DOC_123]

    Handles comma-separated groups as well:
      [source_1, source_2]  → [ENTRY_3714](url1) [WRQ_5021]
    """
    if not source_map and not citations:
        return answer

    def _render_one(key: str) -> str:
        display_id = source_map.get(key, key)
        url = (url_map or {}).get(key)
        if url:
            return f"[{display_id}]({url})"
        return f"[{display_id}]"

    # Handle comma-separated groups: [source_1, source_2, source_3]
    def _replace_group(match: re.Match) -> str:
        inner = match.group(1)
        keys = [k.strip() for k in inner.split(",")]
        return " ".join(_render_one(k) for k in keys)

    # ── Pass 1: Replace [source_N] markers ──────────────────────────────
    result = re.sub(
        r"\[(source_\d+(?:\s*,\s*source_\d+)*)\]",
        _replace_group,
        answer,
    )

    # ── Build id-number → citation lookup (shared by Passes 2–4) ────────
    id_to_citation: Dict[str, Dict[str, Any]] = {}
    if citations:
        for c in citations:
            cid = str(c.get("id", ""))
            if cid:
                id_to_citation[cid] = c
            did = c.get("display_id", "")
            if did:
                id_to_citation[did] = c

    # ── Pass 2: Replace [the system-xxx], [WR-xxx], [ICMS-xxx] markers ───────
    #   Daily Summary agent uses [the system-xxx], [WR-xxx] format.
    #   Map these to the canonical ENTRY_xxx, WRQ_xxx, ICMS_xxx display IDs.
    if citations:
        def _replace_short_ref(m: re.Match) -> str:
            prefix = m.group(1)  # the system, WR, ICMS
            num = m.group(2)    # numeric ID
            c = id_to_citation.get(num)
            if c:
                display = c.get("display_id", f"{prefix}-{num}")
                url = c.get("url")
                if url:
                    return f"[{display}]({url})"
                return f"[{display}]"
            return m.group(0)  # no match — leave as-is

        result = re.sub(
            r"\[(the system|WR|ICMS)-(\d+)\]",
            _replace_short_ref,
            result,
        )

        # ── Pass 2b: Replace [WRQ_xxxxx] markers (daily summary WR format) ─
        def _replace_wrq_ref(m: re.Match) -> str:
            num = m.group(1)  # WRQ number
            c = id_to_citation.get(num)
            if c:
                display = c.get("display_id", f"WRQ_{num}")
                url = c.get("url")
                if url:
                    return f"[{display}]({url})"
                return f"[{display}]"
            return m.group(0)  # no match — leave as-is

        result = re.sub(
            r"\[WRQ_(\d+)\]",
            _replace_wrq_ref,
            result,
        )

    # ── Pass 3: Replace [DISPLAY_ID] markers (e.g. [DOC_2295284]) ──────
    #   Build a lookup from display_id → URL using the citation list.
    #   Only match IDs that are NOT already inside a markdown link.
    if citations:
        display_url: Dict[str, Optional[str]] = {}
        for c in citations:
            did = c.get("display_id", "")
            if did:
                display_url[did] = c.get("url")
            # Also map by raw id (e.g. DOC_2295284 without prefix)
            raw = c.get("id", "")
            if raw and raw != did:
                display_url[raw] = c.get("url")

        if display_url:
            # Escape IDs for safe regex, longest first to avoid partial matches
            sorted_ids = sorted(display_url.keys(), key=len, reverse=True)
            escaped = [re.escape(did) for did in sorted_ids]
            pattern = r"(?<!\()\[(" + "|".join(escaped) + r")\](?!\()"

            def _replace_display_id(match: re.Match) -> str:
                did = match.group(1)
                url = display_url.get(did)
                if url:
                    return f"[{did}]({url})"
                return f"[{did}]"

            result = re.sub(pattern, _replace_display_id, result)

    # ── Pass 4: Bare ENTRY_NNNNN (no brackets) → clickable link ────────
    #   LLMs sometimes emit "ENTRY_10527" as plain text without brackets.
    if id_to_citation:
        def _replace_bare_entry(m: re.Match) -> str:
            num = m.group(1)
            c = id_to_citation.get(num) or id_to_citation.get(f"ENTRY_{num}")
            if c:
                display = c.get("display_id", f"ENTRY_{num}")
                url = c.get("url")
                if url:
                    return f"[{display}]({url})"
                return f"[{display}]"
            return m.group(0)

        result = re.sub(
            r"(?<!\[)(?<!\()ENTRY_(\d+)(?!\]|\))",
            _replace_bare_entry,
            result,
        )

    # ── Pass 5: Bare DOC_NNNNN / DOCU_NNNNN (no brackets) → link ──────
    #   Agentic/Naive RAG over ICMS docs frequently emits run-on prose
    #   like "...time DOC_2190417." or "...DOCU_1430567DOC_2286939." —
    #   Pass 3 only catches bracketed forms.
    if id_to_citation:
        def _replace_bare_doc(m: re.Match) -> str:
            full = m.group(0)
            c = id_to_citation.get(full)
            if c:
                display = c.get("display_id", full)
                url = c.get("url")
                if url:
                    return f"[{display}]({url})"
                return f"[{display}]"
            return full

        result = re.sub(
            r"(?<!\[)(?<!\()DOCU?_\d+(?!\]|\))",
            _replace_bare_doc,
            result,
        )

    # ── Pass 6: Bare WRQ_NNNNN (no brackets) → clickable link ─────────
    if id_to_citation:
        def _replace_bare_wrq(m: re.Match) -> str:
            num = m.group(1)
            c = id_to_citation.get(num) or id_to_citation.get(f"WRQ_{num}")
            if c:
                display = c.get("display_id", f"WRQ_{num}")
                url = c.get("url")
                if url:
                    return f"[{display}]({url})"
                return f"[{display}]"
            return m.group(0)

        result = re.sub(
            r"(?<!\[)(?<!\()WRQ_(\d+)(?!\]|\))",
            _replace_bare_wrq,
            result,
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool-attribution post-processing
# ─────────────────────────────────────────────────────────────────────────────
#
# Counting/aggregation queries get routed through the `data_analysis` tool,
# and the synthesis prompt allows the LLM to attribute results with
# `[tool: data_analysis]` instead of `[source_N]`. That marker is opaque to
# `replace_source_markers`, so when the LLM uses it next to a specific entry
# reference the inline link never renders.
#
# `replace_tool_attribution_markers` runs after the source-marker passes and:
#   - swaps `[tool: data_analysis]` for the matching `[source_N]`-style link
#     when a recognizable entry ID appears in the SAME SENTENCE
#   - otherwise renders the marker as `_[⚙ analysis]_` so aggregate-only
#     attribution stays visually distinct from clickable doc citations
_TOOL_MARKER_PATTERN = re.compile(r"\[tool:\s*data_analysis\s*\]", re.IGNORECASE)
_DATA_ANALYSIS_BADGE = "_[⚙ analysis]_"

# Tolerant prose-form ID patterns. Anchors on the known collection prefix and
# accepts space, underscore, hyphen, or "ID" between the prefix and the number.
_PROSE_ID_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bENTRY(?:\s+ID)?[_\s-]+(\d+)\b", re.IGNORECASE), "the system"),
    (re.compile(r"\bWRQ?(?:\s+ID)?[_\s-]+(\d+)\b", re.IGNORECASE), "WRQ"),
    (re.compile(r"\bSDDS(?:\s+ID)?[_\s-]+([A-Za-z0-9_]+)\b", re.IGNORECASE), "SDDS"),
    (re.compile(r"\bICMS(?:[_\s-]+the facility)?[_\s-]+([A-Za-z0-9_]+)\b", re.IGNORECASE), "ICMS"),
    (re.compile(r"\bOAG(?:\s+ID)?[_\s-]+([A-Za-z0-9_]+)\b", re.IGNORECASE), "OAG"),
]


def replace_tool_attribution_markers(
    answer: str,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Replace `[tool: data_analysis]` markers contextually.

    Same-sentence entry ID → swap for that citation's clickable link.
    Otherwise → italic muted badge (`_[⚙ analysis]_`).
    """
    if not _TOOL_MARKER_PATTERN.search(answer):
        return answer

    id_lookup: Dict[str, Dict[str, Any]] = {}
    if citations:
        for c in citations:
            did = c.get("display_id", "")
            if did:
                id_lookup[did.upper()] = c
            cid = str(c.get("id", ""))
            if cid:
                id_lookup[cid.upper()] = c

    # Preserve sentence terminators by using a capturing split.
    parts = re.split(r"(?<=[.!?])(\s+|$)", answer)
    out: List[str] = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        separator = parts[i + 1] if i + 1 < len(parts) else ""
        out.append(_swap_tool_marker_in_sentence(sentence, id_lookup) + separator)
    return "".join(out)


def _swap_tool_marker_in_sentence(
    sentence: str,
    id_lookup: Dict[str, Dict[str, Any]],
) -> str:
    if not _TOOL_MARKER_PATTERN.search(sentence):
        return sentence

    found: Optional[Dict[str, Any]] = None
    for pattern, prefix in _PROSE_ID_PATTERNS:
        for m in pattern.finditer(sentence):
            num = m.group(1)
            candidate = id_lookup.get(f"{prefix}_{num}".upper()) or id_lookup.get(num.upper())
            if candidate:
                found = candidate
                break
        if found:
            break

    if found:
        display = found.get("display_id", "")
        url = found.get("url", "")
        replacement = f"[{display}]({url})" if url else f"[{display}]"
    else:
        replacement = _DATA_ANALYSIS_BADGE

    return _TOOL_MARKER_PATTERN.sub(replacement, sentence)


def _summarize_snippet(snippet: str, max_len: int = 120) -> str:
    """Extract a 1-sentence summary from a content snippet.

    Takes the first sentence (or first ``max_len`` characters) and ensures
    it ends cleanly — no trailing fragments.
    """
    if not snippet:
        return ""
    # Clean up: collapse whitespace, strip leading timestamps / metadata
    text = " ".join(snippet.split())
    # Try to grab the first sentence
    for sep in (". ", ".\n", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < max_len:
            return text[: idx + 1]
    # Fallback: truncate at max_len on a word boundary
    if len(text) <= max_len:
        return text.rstrip(".")  + "."
    truncated = text[:max_len].rsplit(" ", 1)[0]
    return truncated.rstrip(",.;:—-") + "…"


def format_citation_footer(citations: List[Dict[str, Any]]) -> str:
    """Format a markdown source section to append after the answer.

    Creates a numbered list with clickable links and a 1-sentence content
    summary so the user knows what each source document contains.
    """
    if not citations:
        return ""

    lines = ["\n\n---", "**📚 Sources:**\n"]
    for c in citations[:15]:  # Limit to 15 sources
        idx = c["index"]
        display_id = c["display_id"]
        date = c.get("date", "")
        author = c.get("author", "")
        url = c.get("url", "")
        snippet = c.get("content_snippet", "")

        # Build the header: ID · date · author
        parts = [f"**{display_id}**"]
        if date:
            parts.append(f"📅 {date[:10]}")
        if author:
            parts.append(f"👤 {author}")

        line = f"{idx}. " + " · ".join(parts)

        # Add clickable link if available
        if url:
            line += f" — [🔗 View]({url})"

        lines.append(line)

    return "\n".join(lines)


def _normalize_think_tags(answer: str) -> str:
    """Normalize any <scratchpad>, <thinking>, <routing_analysis> to <think>.

    OpenWebUI renders <think>…</think> as a collapsible dropdown.
    LLMs sometimes emit other CoT tag names — this ensures they all collapse.
    """
    for tag in ("scratchpad", "thinking", "routing_analysis"):
        answer = re.sub(
            rf"<{tag}>(.*?)</{tag}>",
            r"<think>\1</think>",
            answer,
            flags=re.DOTALL,
        )
    return answer


def format_answer_with_citations(
    answer: str,
    docs: List[Document],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Full citation pipeline: normalize CoT tags, replace markers with clickable links, append footer.

    Processing order:
      1. Normalize <scratchpad>/<thinking>/<routing_analysis> → <think> (OpenWebUI dropdown)
      2. Replace [source_N] markers with clickable markdown links
      3. Replace [DISPLAY_ID] markers with clickable markdown links
      4. Replace bare ENTRY_NNNNN with clickable markdown links
      5. Replace [tool: data_analysis] markers — link if adjacent entry ID, badge otherwise
      6. Append formatted source footer

    Args:
        answer: Raw LLM answer with [source_N] markers
        docs: Retrieved documents used for context

    Returns:
        (formatted_answer, citations_list, openwebui_citations)
    """
    # Step 1: Normalize CoT tags so OpenWebUI renders them as collapsible
    answer = _normalize_think_tags(answer)

    citations, source_map = build_source_map(docs)
    # Build source_key → URL map so inline markers become clickable links
    url_map: Dict[str, Optional[str]] = {
        f"source_{c['index']}": c.get("url") for c in citations
    }
    formatted = replace_source_markers(answer, source_map, url_map, citations)
    formatted = replace_tool_attribution_markers(formatted, citations)

    # Separate adjacent rendered citation links with a single space so
    # run-on markers like "DOCU_1430567DOC_2286939" don't read as one
    # visual chunk after Passes 5/6 turn them into back-to-back links.
    _cite_id = r"(?:the system|WRQ|the facility|DOCU|ICMS|OAG|SDDS|TEAMS|SRC)[\w-]+"
    formatted = re.sub(
        rf"(\[{_cite_id}\](?:\([^)]*\))?)(?=\[{_cite_id}\])",
        r"\1 ",
        formatted,
    )

    footer = format_citation_footer(citations)
    owui_citations = build_openwebui_citations(citations)
    return formatted + footer, citations, owui_citations


def build_openwebui_citations(
    citations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build citation metadata in OpenWebUI sidebar format.

    OpenWebUI renders clickable numbered citations in the sidebar when
    message metadata contains a ``citations`` array.  Each entry follows
    the structure::

        {
            "source": {"name": "Display Name"},
            "document": ["Text preview for the sidebar panel"],
            "metadata": [{"source": "url_or_identifier"}]
        }

    Args:
        citations: Citation list from :func:`build_source_map`.

    Returns:
        List of OpenWebUI-compatible citation dicts.
    """
    owui: List[Dict[str, Any]] = []
    for c in citations:
        display_id = c.get("display_id", "")
        coll_display = c.get("collection_display", "")
        date = c.get("date", "")
        author = c.get("author", "")
        url = c.get("url")
        snippet = c.get("content_snippet", "")

        # Header line for sidebar preview
        header_parts = []
        if coll_display:
            header_parts.append(coll_display)
        if date:
            header_parts.append(f"📅 {str(date)[:10]}")
        if author:
            header_parts.append(f"👤 {author}")
        header = " · ".join(header_parts)

        preview = (
            f"**{display_id}** — {header}\n\n{snippet}"
            if snippet
            else f"**{display_id}** — {header}"
        )

        owui.append({
            "source": {"name": display_id or f"Source {c.get('index', '')}"},
            "document": [preview],
            "metadata": [{"source": url or display_id}],
        })
    return owui


def _format_numbered_footer(citations: List[Dict[str, Any]]) -> str:
    """Format 📚 Sources footer with 0-based indices matching inline [N] markers.

    Used by format_daily_summary_citations so the footer numbering aligns
    with the [0], [1], … superscripts in the answer body.
    """
    if not citations:
        return ""

    lines = ["\n\n---", "**📚 Sources:**\n"]
    for c in citations[:15]:
        idx = c["index"]  # 0-based
        display_id = c.get("display_id", f"source_{idx}")
        date = c.get("date", "")
        author = c.get("author", "")
        url = c.get("url", "")
        snippet = c.get("content_snippet", "")

        parts = [f"**{display_id}**"]
        if date:
            parts.append(f"📅 {str(date)[:10]}")
        if author:
            parts.append(f"👤 {author}")

        line = f"[{idx}] " + " · ".join(parts)
        if url:
            line += f" — [🔗 View]({url})"

        lines.append(line)

    return "\n".join(lines)


def format_daily_summary_citations(
    answer: str,
    docs: List[Document],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Citation pipeline for Daily Summary: converts named markers to [N] indices.

    Unlike format_answer_with_citations (which preserves named [source_N]
    markers for Agentic/Naive RAG), this function converts all citation markers
    emitted by the Daily Summary LLM to 0-indexed [N] so OpenWebUI renders
    them as clickable superscripts that open the sidebar panel.

    Marker formats handled::

      [the system-12345]   → [N]  (the logbook entry; URL in citations[N])
      [WRQ_12345]    → [N]  (work request; no external URL; sidebar shows content)
      [DOC_2295284]  → [N]  (ICMS document; URL in citations[N])

    Returns:
        (formatted_answer, ordered_citations, openwebui_citations)

        - formatted_answer:    answer with [N] inline markers + 📚 Sources footer
        - ordered_citations:   citation dicts numbered 0-N in first-appearance order
        - openwebui_citations: list in OpenWebUI sidebar format
    """
    answer = _normalize_think_tags(answer)

    # Build full citation map from all retrieved docs
    all_citations, _ = build_source_map(docs)

    # id / display_id → citation dict
    id_to_citation: Dict[str, Dict[str, Any]] = {}
    for c in all_citations:
        id_to_citation[str(c["id"])] = c
        if c.get("display_id"):
            id_to_citation[c["display_id"]] = c

    # WRQ_NO aliases: work request docs store the WR number in WRQ_NO metadata
    # which may differ from the file_id used as raw_id in build_source_map.
    for doc in docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        if meta.get("collection") == "work_requests":
            raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
            wrq_no = str(meta.get("WRQ_NO") or raw.get("WRQ_NO") or "")
            if wrq_no and wrq_no not in id_to_citation:
                file_id = str(meta.get("file_id", ""))
                existing = id_to_citation.get(file_id)
                if existing:
                    id_to_citation[wrq_no] = existing

    # Track citations in order of first appearance in the answer text
    ordered_citations: List[Dict[str, Any]] = []
    seen_ids: Dict[str, int] = {}  # canonical_id → 0-based index

    def _assign(cid: str):
        """Return 0-based index for this citation ID, registering it if new."""
        if cid in seen_ids:
            return seen_ids[cid]
        c = id_to_citation.get(cid)
        if c is None:
            return None
        canonical = c.get("display_id") or str(c.get("id", cid))
        if canonical in seen_ids:
            # Same citation already registered under a different alias
            seen_ids[cid] = seen_ids[canonical]
            return seen_ids[canonical]
        idx = len(ordered_citations)
        ordered_citations.append({**c, "index": idx})
        seen_ids[canonical] = idx
        seen_ids[cid] = idx
        return idx

    def _render_link(cid: str, fallback: str) -> str:
        """Convert a citation ID to a clickable markdown link or plain bracket."""
        idx = _assign(cid)
        if idx is None:
            return fallback
        c = ordered_citations[idx]
        display = c.get("display_id", fallback)
        url = c.get("url")
        if url:
            return f"[{display}]({url})"
        # No URL (e.g. work requests) — render a plain bracket. Sidebar still
        # shows the snippet via the OpenWebUI citations payload.
        return f"[{display}]"

    # Pass A: [the system-xxx] / [WR-xxx] / [ICMS-xxx]
    def _sub_short(m: re.Match) -> str:
        return _render_link(m.group(2), m.group(0))

    result = re.sub(r"\[(the system|WR|ICMS)-(\d+)\]", _sub_short, answer)

    # Pass B: [WRQ_xxxxx]
    def _sub_wrq(m: re.Match) -> str:
        return _render_link(m.group(1), m.group(0))

    result = re.sub(r"\[WRQ_(\d+)\]", _sub_wrq, result)

    # Pass C: ICMS display IDs like [DOC_2295284]
    icms_ids = [
        did for did in id_to_citation
        if re.match(r"^DOC_\d+$|^ICMS_", did)
    ]
    if icms_ids:
        sorted_icms = sorted(icms_ids, key=len, reverse=True)
        icms_pattern = r"(?<!\()\[(" + "|".join(re.escape(d) for d in sorted_icms) + r")\](?!\()"

        def _sub_icms(m: re.Match) -> str:
            return _render_link(m.group(1), m.group(0))

        result = re.sub(icms_pattern, _sub_icms, result)

    # Pass D: Bare ENTRY_NNNNN (no brackets) → clickable link
    def _sub_bare_entry(m: re.Match) -> str:
        return _render_link(m.group(1), m.group(0))

    result = re.sub(
        r"(?<!\[)(?<!\()ENTRY_(\d+)(?!\]|\))",
        _sub_bare_entry,
        result,
    )

    # Pass E: Bare WRQ_NNNNN (no brackets) → [WRQ_NNNNN] link.
    # The LLM sometimes emits run-on work-request citations like
    # "ENTRY_10726WRQ_62356WRQ_62359"; Pass D only catches the system, leaving the
    # trailing WRQs as unrendered plaintext.
    def _sub_bare_wrq(m: re.Match) -> str:
        return _render_link(m.group(1), m.group(0))

    result = re.sub(
        r"(?<!\[)(?<!\()WRQ_(\d+)(?!\]|\))",
        _sub_bare_wrq,
        result,
    )

    # Pass F: Bare DOC_NNNNN (no brackets) → [DOC_NNNNN](url) link.
    # The LLM emits run-on ICMS citations like "DOC_2295284ENTRY_10163DOC_2322827";
    # Pass C only catches bracketed [DOC_NNNNN], so bare the facility IDs in mixed run-ons
    # stay as plaintext. Render the same way as Pass D/E.
    def _sub_bare_doc(m: re.Match) -> str:
        return _render_link(m.group(0), m.group(0))

    result = re.sub(
        r"(?<!\[)(?<!\()DOC_\d+(?!\]|\))",
        _sub_bare_doc,
        result,
    )

    # Final pass: separate adjacent inline citations with a single space.
    # When run-on markers above are rendered they end up shoulder-to-shoulder
    # ("[ENTRY_10726](url)[WRQ_62356]"), which reads as one visual chunk.
    # Scoped to known citation-ID prefixes so we don't touch markdown
    # reference-style links or things like `arr[0][1]`.
    _cite_id = r"(?:the system|WRQ|the facility|ICMS|OAG|SDDS|TEAMS|SRC)[\w-]+"
    result = re.sub(
        rf"(\[{_cite_id}\](?:\([^)]*\))?)(?=\[{_cite_id}\])",
        r"\1 ",
        result,
    )

    footer = _format_numbered_footer(ordered_citations)
    owui_citations = build_openwebui_citations(ordered_citations)

    return result + footer, ordered_citations, owui_citations


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_display_id(raw_id: str, collection: str, meta: Dict[str, Any]) -> str:
    """Create a collection-prefixed display ID.

    Examples: ENTRY_3714, ICMS_DOC_2306981, WRQ_5021, OAG_SROrbitCorrection
    """
    prefix = _ID_PREFIXES.get(collection, "SRC")

    # Strip __chunk_N suffix from granular-chunked IDs
    if "__chunk" in str(raw_id):
        raw_id = str(raw_id).split("__chunk")[0]

    if collection == "oag_collection":
        app_name = meta.get("app_name", meta.get("title", raw_id))
        return f"{prefix}_{app_name}"

    if collection == "sdds_collection":
        title = meta.get("title", raw_id)
        return f"{prefix}_{title}"

    if collection == "icms_collection":
        # Use raw ID (e.g., DOC_2306981)
        return f"{raw_id}" if raw_id.startswith("DOC") else f"{prefix}_{raw_id}"

    if collection == "teams_messages":
        # Teams messages all share file_id="teams_messages", so construct
        # a unique display ID from author + creation_datetime (matching
        # the dedup pattern in build_source_map lines 87-98).
        author = meta.get("author", "unknown")
        # Clean author for use in ID: "Doe, Jane A." → "Doe_Jane_A"
        author_clean = author.replace(" ", "_").replace(",", "").replace(".", "")
        creation_dt = meta.get("creation_datetime", "")
        if creation_dt:
            # Include date + hour to avoid same-day collisions
            # "2024-02-18T14:30:00" → "2024-02-18T14"
            dt_short = creation_dt[:13] if len(creation_dt) >= 13 else creation_dt[:10]
            return f"{prefix}_{author_clean}_{dt_short}"
        return f"{prefix}_{author_clean}"

    return f"{prefix}_{raw_id}"


def _make_url(raw_id: str, collection: str, meta: Dict[str, Any]) -> Optional[str]:
    """Construct a URL for the source document, if available."""
    pattern = _URL_PATTERNS.get(collection)
    if not pattern:
        return None

    try:
        if collection == "entry_logbook":
            entry_id = str(meta.get("entry_id") or meta.get("file_id") or raw_id)
            # Strip __chunk_N suffix from granular-chunked IDs
            if "__chunk" in entry_id:
                entry_id = entry_id.split("__chunk")[0]
            return pattern.format(entry_id=entry_id)

        if collection == "icms_collection":
            file_url = (
                meta.get("file_url")
                or (meta.get("_raw") or {}).get("file_url")
                or ""
            )
            if file_url:
                return pattern.format(file_url=file_url)
            # Fallback: construct URL from file_id (e.g., DOC_2295284 → doc_2295284)
            file_id = meta.get("file_id") or meta.get("id") or raw_id
            if file_id:
                # Strip __chunk_N suffix from granular-chunked IDs so the
                # URL points at the source document, not the chunk slice.
                file_id = str(file_id)
                if "__chunk" in file_id:
                    file_id = file_id.split("__chunk")[0]
                return f"https://your-docs-host.example.com/docs/idcplg?IdcService=DISPLAY_URL&dDocName={file_id}"
            return None

        if collection == "oag_collection":
            # Prefer svn_url from metadata if available
            svn_url = meta.get("svn_url")
            if svn_url:
                return svn_url
            app_name = meta.get("app_name", meta.get("title", ""))
            if app_name:
                return pattern.format(app_name=app_name)
            return None

        if collection in {"work_requests", "teams_messages", "sdds_collection"}:
            entry_id = _entry_route_id(raw_id, collection, meta)
            if not entry_id:
                return None
            return pattern.format(api_base=_API_BASE_URL, entry_id=entry_id)

    except (KeyError, ValueError) as exc:
        logger.debug("URL construction failed for %s/%s: %s", collection, raw_id, exc)
        return None

    return None


def _entry_route_id(raw_id: str, collection: str, meta: Dict[str, Any]) -> str:
    """Pick the lookup id the /api/entries route uses for each collection."""
    if collection == "work_requests":
        raw = meta.get("_raw") or {}
        wrq_no = raw.get("WRQ_NO") if isinstance(raw, dict) else None
        return str(wrq_no or raw_id)
    if collection == "teams_messages":
        # Teams docs share file_id; use author_datetime as the unique key —
        # matches _make_display_id's dedup pattern (lines 688-700).
        author = (meta.get("author") or "unknown").replace(" ", "_").replace(",", "").replace(".", "")
        creation_dt = meta.get("creation_datetime", "")
        if creation_dt:
            dt_short = creation_dt[:13] if len(creation_dt) >= 13 else creation_dt[:10]
            return f"{author}_{dt_short}"
        return author or str(raw_id)
    # sdds_collection: raw_id is already the file_id
    return str(raw_id)
