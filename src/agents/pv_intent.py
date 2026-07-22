"""
PV intent classifier — deterministic, zero-LLM gate that detects when a query
should bypass the normal RAG path and call an EPICS live-read tool instead.

Returns one of two routes, or None:
  - Route A ("pv_read"):   query contains a literal EPICS PV address.
  - Route B ("pv_search"): query phrases describe wanting a live/current value.

Mirrors the `is_counting_query` regex-only pattern in src/retrieval/query_enhancement.py
so the gate adds no latency and is deterministic across runs.
"""
import logging
import re
from typing import List, Optional, TypedDict

logger = logging.getLogger(__name__)


class PVIntent(TypedDict):
    route: str          # "pv_read" or "pv_search"
    args: dict          # kwargs to pass to tool._run(**args)
    matched: List[str]  # what triggered the match (PV names or phrase)
    label: str          # human-readable for reasoning_trace


# ──────────────────────────────────────────────────────────────────────────
# Route A — literal EPICS PV address
# ──────────────────────────────────────────────────────────────────────────
# the facility PV naming examples we must match:
#   S35A:H1:CurrentAO, S-DCCT:CurrentM, LTS:H1:CurrentAO, BR:H1:SP
# First segment is always all-caps + digits + '_-' (so the exclusion-prefix
# check below stays meaningful). Subsequent segments allow CamelCase because
# real the facility signal suffixes are mixed case (`CurrentAO`, `Setpoint`, etc.).
_PV_ADDRESS_RE = re.compile(
    r"\b[A-Z][A-Z0-9_-]{0,15}(?::[A-Za-z][A-Za-z0-9_-]{0,30}){1,3}\b"
)

# Exclusion prefixes that look like PVs but aren't. Extend conservatively.
_PV_EXCLUSION_PREFIXES = {
    "the system", "ICMS", "SDDS", "OAG",        # internal document namespaces
    "HTTP", "HTTPS", "WWW", "URL", "URI", # protocol prefixes
    "API", "REST", "JSON", "YAML",        # tech jargon
    "TODO", "FIXME", "NOTE", "WARN",      # code markers
    "UTC", "EST", "PST",                  # timezone prefixes
    "PR", "ID", "DOC", "REF",             # ID prefixes
}

# Reject ISO date/time tokens that the PV regex would otherwise match.
_ISO_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?\b")


# ──────────────────────────────────────────────────────────────────────────
# Route B — description-based phrases
# ──────────────────────────────────────────────────────────────────────────
_PV_PHRASE_PATTERNS = [
    # "current/live/present/latest/real-time value/reading/state of X"
    r"\b(?:current|live|present|latest|real[- ]?time)\s+(?:value|reading|state)\s+of\b",
    # "what is X right now / currently / at the moment"
    r"\bwhat\s+(?:is|are|'s)\s+.+?\s+(?:right\s+now|currently|at\s+the\s+moment)\b",
    # "read PV X" / "read the PV X"
    r"\bread\s+(?:the\s+)?pv\b",
    # "EPICS value of X"
    r"\bepics\s+(?:value|reading)\s+of\b",
]
_PV_PHRASE_RE = re.compile("|".join(_PV_PHRASE_PATTERNS), flags=re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────
# Historical-query suppression for Route B
# ──────────────────────────────────────────────────────────────────────────
# If the query references a past date/time, Route B is suppressed — those are
# retrieval queries ("what was the beam current last Thursday"), not live reads.
# Route A is NOT suppressed: a literal PV address is unambiguous.
_HISTORICAL_PATTERNS = [
    r"\byesterday\b",
    r"\blast\s+(?:week|month|year|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:on|in|during|since|before|after)\s+\d",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b(?:was|were|did|had)\b",
]
_HISTORICAL_RE = re.compile("|".join(_HISTORICAL_PATTERNS), flags=re.IGNORECASE)

_MAX_PV_NAMES = 5


def _looks_historical(query: str) -> bool:
    return bool(_HISTORICAL_RE.search(query or ""))


def _find_pv_addresses(query: str) -> List[str]:
    """Extract candidate PV addresses, filtering excluded prefixes and timestamps."""
    if not query:
        return []
    cleaned = _ISO_TIMESTAMP_RE.sub(" ", query)
    candidates = _PV_ADDRESS_RE.findall(cleaned)
    out: List[str] = []
    seen: set = set()
    for cand in candidates:
        head = cand.split(":", 1)[0]
        if head in _PV_EXCLUSION_PREFIXES:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
        if len(out) >= _MAX_PV_NAMES:
            break
    return out


def classify_pv_intent(query: str) -> Optional[PVIntent]:
    """Classify a query for direct PV-tool dispatch.

    Order matters: Route A (literal PV) wins over Route B (description) so
    "current value of S35A:H1:CurrentAO" goes straight to pv_read instead of
    triggering an unnecessary channel_finder lookup via pv_search.

    Returns None to indicate fall-through to the normal RAG service.
    """
    if not query or not query.strip():
        return None

    # Route A — literal PV address
    pvs = _find_pv_addresses(query)
    if pvs:
        return PVIntent(
            route="pv_read",
            args={"pv_names": ",".join(pvs)},
            matched=pvs,
            label="literal PV address",
        )

    # Route B — description-based phrase, suppressed for historical queries
    if _PV_PHRASE_RE.search(query):
        if _looks_historical(query):
            logger.info(
                "PV intent: phrase matched but historical date detected — suppressed"
            )
            return None
        return PVIntent(
            route="pv_search",
            args={"query": query.strip()},
            matched=[query.strip()],
            label="description phrase",
        )

    return None
