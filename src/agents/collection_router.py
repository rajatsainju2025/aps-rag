"""
CollectionRouter — LLM-based first-pass query classifier.

Uses Argo claude-3-5-haiku (fast_llm) to decide which knowledge-base collections
are most relevant for a given query before retrieval begins.

This replaces the previous keyword-signal approach with a small, cheap LLM call
that understands natural language — e.g. it can infer "the system" from phrases like
"latest entry", "experiment", "studies", "injection study", etc. even without
the word "logbook" appearing explicitly.

Collections understood:
  entry_logbook     — shift/run logbook entries, experiments, studies, beam ops
  icms_collection  — technical documents, reports, forms, policies, procedures
  work_requests    — maintenance work requests, work orders, WRQs
  teams_messages   — MS Teams discussion messages
  sdds_collection  — SDDS toolkit documentation and APIs
  oag_collection   — OAG / Tcl-Tk application documentation

Returns a list of matching collection names, or an empty list meaning
"search all collections" (used as the fallback by NaiveRAGService).
"""
import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a query-routing classifier for the your facility "
    "accelerator knowledge base. Your ONLY job is to identify which data "
    "collections are relevant to a user query and return a JSON array of "
    "collection names. Be liberal — if unsure, include the collection."
)

_USER_TEMPLATE = """\
Classify the following user query into one or more of these collections.
Return ONLY a JSON array of collection names — no explanation, no markdown.

Collections:
- "entry_logbook"    : the facility shift/run logbook entries written by operators and
                      scientists. Contains experiment records, studies, beam
                      observations, injection performance, machine events,
                      fault reports, and day-to-day accelerator operations.
                      SIGNAL PHRASES: entry, entries, logbook, entry, study,
                      studies, experiment, injection, beam, shift, run, log,
                      SR, ASD, operations, machine, accelerator event, latest
                      entry, yesterday, today, this week, last week.

- "icms_collection" : Formal technical documents stored in ICMS — technical
                      notes, reports, presentations, forms, policy/procedure
                      documents, manuals, engineering docs.
                      SIGNAL PHRASES: technical note, technical document, report,
                      form, presentation, policy, procedure, ICMS, doc, document,
                      manual, specification, engineering note, TN, APSnote.

- "work_requests"   : Maintenance and repair work orders / work requests
                      submitted by the facility staff.
                      SIGNAL PHRASES: work request, WRQ, work order, maintenance,
                      repair, replacement, installation, PM, preventive.

- "teams_messages"  : Internal Microsoft Teams chat messages from the facility staff
                      discussions and channels.
                      SIGNAL PHRASES: Teams, Teams message, chat, discussion,
                      conversation, channel, posted, said, mentioned.

- "sdds_collection" : SDDS (Self Describing Data Sets) toolkit documentation,
                      API references, command-line tools, file format specs.
                      SIGNAL PHRASES: SDDS, sddsplot, sddsprocess, sddsconvert,
                      toolkit, SDDS file, pelegant, elegant.

- "oag_collection"  : OAG application docs,
                      Tcl/Tk scripts, internal the facility software tools.
                      SIGNAL PHRASES: OAG, Tcl, Tk, tcltkapp, application,
                      OAG app, OAG script, internal tool.

Rules:
1. Return [] (empty array) if the query is ambiguous and could apply to ALL
   collections equally — this causes the system to search everything.
2. You MAY return multiple collections if the query spans several areas.
3. If the query mentions a specific collection by name, always include it.
4. "entry" alone (without further context) → include entry_logbook.
5. Anything about experiments, studies, beam physics, or accelerator events
   that sounds like an operator log → include entry_logbook.

User query: {query}

Return ONLY a JSON array, e.g.: ["entry_logbook"] or ["icms_collection","work_requests"] or []
"""

# ---------------------------------------------------------------------------
# Valid collection names (guard rail for LLM output)
# ---------------------------------------------------------------------------
_VALID_COLLECTIONS = {
    "entry_logbook",
    "icms_collection",
    "work_requests",
    "teams_messages",
    "sdds_collection",
    "oag_collection",
}


class CollectionRouter:
    """
    LLM-based collection classifier using claude-3-5-haiku.

    Parameters
    ----------
    llm : ArgoLLMClientAG
        A fast/cheap LLM client (claude-3-5-haiku).  If None the router falls back
        to returning [] (search all collections).
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm

    def route(self, query: str) -> List[str]:
        """
        Classify *query* and return the relevant collection names.

        Returns an empty list if no specific collection is identified, which
        the caller interprets as "search all collections".

        This method is intentionally fast — it should add < 1 s to the
        request latency for a typical short query.
        """
        if not self.llm:
            logger.debug("CollectionRouter: no LLM configured, returning []")
            return []

        try:
            messages = [
                SystemMessage(content=_SYSTEM),
                HumanMessage(
                    content=_USER_TEMPLATE.format(query=query.strip())
                ),
            ]
            response = self.llm.invoke(messages)
            raw = response.content.strip() if hasattr(response, "content") else str(response).strip()
            collections = self._parse_response(raw)
            if collections:
                logger.info(
                    "CollectionRouter: query=%r → collections=%s",
                    query[:80],
                    collections,
                )
            else:
                logger.debug(
                    "CollectionRouter: query=%r → all collections (no signal)",
                    query[:80],
                )
            return collections

        except Exception as exc:
            logger.warning(
                "CollectionRouter LLM call failed (%s), falling back to all collections",
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> List[str]:
        """
        Extract a JSON array from the LLM response.

        Handles:
        - Clean JSON:        ["entry_logbook"]
        - Markdown-wrapped:  ```json\n["entry_logbook"]\n```
        - Extra whitespace / stray text before/after the array
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip()

        # Find the first [...] block
        match = re.search(r"\[.*?\]", cleaned, flags=re.DOTALL)
        if not match:
            logger.warning("CollectionRouter: could not find JSON array in: %r", raw[:200])
            return []

        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("CollectionRouter: JSON parse error (%s) in: %r", exc, raw[:200])
            return []

        if not isinstance(result, list):
            return []

        # Filter to valid collection names only
        valid = [c for c in result if isinstance(c, str) and c in _VALID_COLLECTIONS]
        invalid = [c for c in result if c not in _VALID_COLLECTIONS]
        if invalid:
            logger.warning("CollectionRouter: LLM returned unknown collections: %s", invalid)

        return valid
