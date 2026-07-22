"""
Query enhancement pipeline — re-imports from entry_rag_app (read-only).

Pipeline steps:
  1. Acronym / domain-term resolution  (AcronymResolver)
  2. the facility synonym expansion              (SynonymManager)
  3. Temporal inline expansion          (TemporalProcessor.enhance_query)
  4. Structured date-range extraction   (TemporalDateParser.parse_query)
  5. Multi-query variant generation     (LLM — only when llm kwarg is provided)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query-enhancement components are vendored under query_enhancement_lib/ and
# imported normally below. Each import is wrapped in try/except so a single
# component failure degrades to a harmless no-op fallback rather than disabling
# the whole enhancement pipeline.
# ---------------------------------------------------------------------------

try:
    from src.retrieval.query_enhancement_lib.temporal_processor import TemporalProcessor
    _TEMPORAL_PROCESSOR_OK = True
    logger.info("Loaded TemporalProcessor")
except Exception as exc:
    logger.warning("Could not import TemporalProcessor: %s — using fallback", exc)
    _TEMPORAL_PROCESSOR_OK = False

    class TemporalProcessor:  # type: ignore
        def enhance_query(self, query: str, collection_name: str = None):
            return query, {}

try:
    from src.retrieval.query_enhancement_lib.acronyms import AcronymResolver
    _ACRONYM_OK = True
    logger.info("Loaded AcronymResolver")
except Exception as exc:
    logger.warning("Could not import AcronymResolver: %s — using fallback", exc)
    _ACRONYM_OK = False

    class AcronymResolver:  # type: ignore
        def expand(self, text: str, collection=None) -> str:
            return text

try:
    from src.retrieval.query_enhancement_lib.synonym_manager import SynonymManager
    _SYNONYM_OK = True
    logger.info("Loaded SynonymManager")
except Exception as exc:
    logger.warning("Could not import SynonymManager: %s — using fallback", exc)
    _SYNONYM_OK = False

    class SynonymManager:  # type: ignore
        def expand_query(self, query: str, max_expansions: int = 3):
            return query, {}

try:
    from src.retrieval.query_enhancement_lib.temporal_date_parser import TemporalDateParser
    _DATE_PARSER_OK = True
    logger.info("Loaded TemporalDateParser")
except Exception as exc:
    logger.warning("Could not import TemporalDateParser: %s — using fallback", exc)
    _DATE_PARSER_OK = False

    class TemporalDateParser:  # type: ignore
        def __init__(self, argo_model: str = "", argo_username: str = ""):
            pass

        def parse_query(self, query: str):
            return None

        def get_date_range_filter(self, date_range) -> Dict:
            return {}

try:
    from src.retrieval.query_enhancement_lib.query_understanding import QueryIntent
except Exception:
    class QueryIntent:  # type: ignore
        FACTUAL = "factual"; TEMPORAL = "temporal"; TROUBLESHOOTING = "troubleshooting"
        COMPARATIVE = "comparative"; CODE_GENERATION = "code_generation"
        MATHEMATICAL = "mathematical"; MULTI_HOP = "multi_hop"; GENERAL = "general"

# ---------------------------------------------------------------------------
# Multi-query generation prompt
# ---------------------------------------------------------------------------

_MULTI_QUERY_PROMPT = """\
# Role
You are a **Query Variant Generator** for a scientific document retrieval system \
at your facility.

# Task
Generate {n} semantically diverse search query variants for the original question. \
Each variant must target a different angle, phrasing, or aspect of the same \
information need to maximize recall during multi-query retrieval.

# Rules (ABSOLUTE)
- If the question contains a date range in format "(date range: MM/DD/YYYY to MM/DD/YYYY)", \
preserve it EXACTLY and VERBATIM in ALL variants.
- Each variant should use different vocabulary, sentence structure, or focus.
- Variants must remain semantically equivalent to the original — do NOT change the intent.
- Return ONLY the queries, one per line — no numbering, no bullets, no extra text.
- Do NOT repeat the original question verbatim.

# Original Question
{question}

# Variants (one per line):"""


_COMBINED_PROMPT = """\
You are a query pre-processor for the your facility accelerator \
document retrieval system.

Given a user query, return ONLY a valid JSON object with exactly two fields:

"temporal": Extract a date/time range ONLY when the query contains an EXPLICIT date \
or time reference — a specific date, named month/year, or a relative phrase like \
"last week", "yesterday", "this month", "past 30 days".
  Today is {today}.
  Return: {{"start": "YYYY-MM-DDTHH:MM:SS", "end": "YYYY-MM-DDTHH:MM:SS", \
"start_date": "MM/DD/YYYY", "end_date": "MM/DD/YYYY", \
"phrase": "<original time text>", "confidence": 0.0-1.0, "inferred": true}}
  Single-day queries: start at T00:00:00, end at T23:59:59.
  IMPORTANT — set to null when the query uses vague/relative words that refer to \
"the most recent instance" without specifying a concrete date or bounded period. \
Vague words that must NOT trigger temporal extraction:
  current, now, recent, recently, latest, today, last, previous, prior, past, \
  ongoing, upcoming, next, old, older, historical, new, newer, existing, active, \
  former, preceding, available, earliest, freshest.
  EXCEPTION: "last week/month/year", "past 30 days", "next week/month", or any \
  of these words followed by a time unit (days/weeks/months/years/hours/a number) \
  ARE explicit temporal references and should be extracted normally.

"variants": Generate {n} semantically diverse search query variants.
  - Different vocabulary, angle, or emphasis to maximise retrieval recall.
  - Preserve any explicit dates verbatim in every variant.
  - Do NOT include the original query.

Return ONLY the JSON object — no markdown, no commentary.

Query: {query}"""


_COMBINED_ROUTING_PROMPT = """\
# Role
You are the **Query Pre-Processing & Routing Engine** for your facility accelerator operations RAG system at your organization. You perform \
three tasks in a single pass: (1) route the query to the right service/collections, \
(2) extract any explicit temporal date range, (3) generate diverse search variants.

# Available Collections
| Collection | Contains | Route When |
|-----------|----------|------------|
| entry_logbook | Logbook entries, beam ops, experiments, shift notes, studies | Operations, beam status, experiments |
| teams_messages | Team comms, discussions, meeting notes | Discussions, decisions, meeting context |
| icms_collection | 87k+ ICMS technical docs, procedures, design notes | Technical specs, procedures, design docs |
| work_requests | Maintenance work orders, repairs, replacements | Maintenance history, equipment status |
| sdds_collection | SDDS toolkit documentation | SDDS code/API questions |
| oag_collection | OAG Tcl/Tk application documentation | OAG app/script questions |
| all_databases | Federated search | Broad or multi-topic questions |

# Collection Routing Rules
## HARD RULE — Explicit collection mention wins (CHECK FIRST)
If the query explicitly names a source ("in ICMS documents", "the logbook", \
"Teams messages", "work request(s)", "WRQ", "SDDS toolkit", "OAG application"), \
restrict ONLY to the named collection(s).
## Topic-based routing (only when no explicit source is named)
- SDDS code → ["sdds_collection"]
- OAG code → ["oag_collection"]
- WR counting/filtering → ["work_requests"] with CODE_EXEC
- Fault diagnosis → ["entry_logbook", "icms_collection", "work_requests"]
- Daily summary → ["entry_logbook", "icms_collection", "work_requests", "teams_messages"]
- Team discussions → ["teams_messages", "entry_logbook"]
- Experiments/studies → ["entry_logbook", "icms_collection"]
- General the facility ops → ["entry_logbook", "icms_collection", "teams_messages"]
- PV read / live values / archiver history → ["entry_logbook", "work_requests"]
- Broad/unclear → ["all_databases"]

# Intent Types — pick EXACTLY ONE value (never compound, never a list)
factual | temporal | troubleshooting | comparative | multi_hop | code_generation | \
mathematical | counting | daily_summary | studies_copilot | pv_query | status_check | general

## Intent Disambiguation (read before choosing)
- **factual** — DEFAULT for "what / how / explain / describe / summarize <specific \
  topic>" questions. Use this for ANY query that names a specific entity, system, \
  document type, or topic, even if the verb is "summarize" or "list".
- **temporal** — ONLY for queries asking "what happened" in a specific time window \
  with NO other primary task (e.g. "what happened on 2025-06-12"). If the query also \
  has a topic + question type (troubleshooting / factual / comparative), pick THAT \
  intent and put the date in the `temporal` field.
- **troubleshooting** — root-cause / "why did X fail" / "how do we fix" / fault \
  diagnosis. A dated fault question ("what caused the beam loss on 2025-06-12") is \
  troubleshooting, NOT temporal.
- **code_generation** — ONLY when the user wants code WRITTEN ("write a script", \
  "generate code that"). "How does X work" / "Explain command options" / "What are \
  the flags for Y" are factual, NOT code_generation.
- **daily_summary** — RESERVED for explicit recap requests with NO specific topic: \
  "daily operations summary", "what happened today/yesterday in the SR", "give me \
  the shift report". "Summarize the latest entry about <topic>" or "summarize <topic> \
  from March 2026" are factual / temporal, NOT daily_summary.
- **counting** — query asks for a count, total, distribution ("how many", "list all").
- **pv_query** — query asks for a specific PV value / archive readout, OR references \
  an EPICS PV name (S35A:..., S-DCCT:CurrentM, etc.). "What is X now?" with X being a \
  measurable signal → pv_query.
- **status_check** — query asks "how has X been working / performing", "is X up / \
  operational / healthy", "what's the status of X", "has X been having issues", or \
  any health-check phrasing for a named piece of equipment (a specific monitor, magnet, \
  cavity, BPM, sector subsystem, etc.). Distinct from pv_query because the user wants \
  BOTH a live status snapshot AND a recap of recent log activity. NOT for whole-machine \
  questions ("how is the SR doing today" → daily_summary).
- **comparative** — "compare A vs B", "difference between".
- **multi_hop** — needs chained reasoning across multiple sources.

# Retrieval Strategies
- VECTOR_BM25: DEFAULT — vector + BM25, no graph. Use for naive_rag, agentic_rag, \
  daily_summary, studies_copilot unless query explicitly needs entity relationships.
- HYBRID_3WAY: vector + BM25 + KG graph — only when query needs fault chains, \
  expert attribution, system dependencies, or multi-hop traversal.
- KG_ONLY: graph traversal only — reserved for troubleshooting service mode.
- CODE_EXEC: math / data analysis / counting (data_analysis tool).
- PV_QUERY: live PV readout + operational history (use for intent=pv_query AND \
  intent=status_check — both need a live control-system probe alongside log search).
NEVER emit VECTOR_ONLY.

# Service Modes
naive_rag | agentic_rag | daily_summary | troubleshooting | studies_copilot

# Temporal Extraction
Today is {today}.
Extract a date/time range ONLY when the query contains an EXPLICIT date or time \
reference — a specific date, named month/year, or a bounded relative phrase like \
"last week", "yesterday", "this month", "past 30 days".
Set temporal to null when the query uses vague words WITHOUT a concrete time bound:
  current, now, recent, recently, latest, today, last, previous, prior, past, \
  ongoing, upcoming, next, old, older, historical, new, newer, existing, active, \
  former, preceding, available, earliest, freshest.
EXCEPTION: "last week/month/year", "past 30 days", "next week/month", or these \
words followed by a time unit (days/weeks/months/years/hours/a number) ARE explicit \
and must be extracted.
Temporal object shape: {{"start":"YYYY-MM-DDTHH:MM:SS","end":"YYYY-MM-DDTHH:MM:SS",\
"start_date":"MM/DD/YYYY","end_date":"MM/DD/YYYY","phrase":"<original text>",\
"confidence":0.0-1.0,"inferred":true}}
Single-day queries: start at T00:00:00, end at T23:59:59.

# Query Variants
Generate {n} semantically diverse search query variants.
- Different vocabulary, angle, or emphasis to maximise retrieval recall.
- Preserve any explicit dates verbatim in every variant.
- Do NOT include the original query.

# Output — JSON only, no prose, no markdown fences, no <think> block
{{"intent":"...","collections":[...],"retrieval_strategy":"...","service_mode":"...",\
"entities":[...],"reasoning":"one sentence","temporal":{{...}} or null,\
"variants":["...","...","..."]}}

Query: {query}"""


# ---------------------------------------------------------------------------
# Combined routing + temporal + variants + PLAN — used for agentic_rag mode.
# Adds a slim planner so we save the dedicated planner LLM call (~12s sonnet46).
# ---------------------------------------------------------------------------
_COMBINED_RPV_PROMPT = (
    _COMBINED_ROUTING_PROMPT.replace(
        '# Output — JSON only, no prose, no markdown fences, no <think> block\n'
        '{{"intent":"...","collections":[...],"retrieval_strategy":"...","service_mode":"...",'
        '"entities":[...],"reasoning":"one sentence","temporal":{{...}} or null,'
        '"variants":["...","...","..."]}}\n\nQuery: {query}',
        ""
    )
    + """
# Plan (research steps)
Produce a SLIM execution plan as a JSON array. 2–4 steps, final step MUST be \
{{"step":N,"tool":"synthesize","query":null,"collections":[],"rationale":"..."}}.

## Tools (use these EXACT names)
| Tool | When |
|------|------|
| vector_search | Default semantic retrieval (Qdrant) |
| bm25_search   | Default keyword retrieval (Elasticsearch) |
| graph_search  | KG traversal — fault chains, expert attribution, entity relations |
| data_analysis | Counting/aggregation (pandas on work_requests, entry_logbook, teams_messages) |
| pv_search     | Live EPICS PV read (intent=pv_query) |
| synthesize    | Final aggregation (always last step) |

## Plan rules by intent
- counting             → step 1: data_analysis; step 2: bm25_search (context for the count); step 3: synthesize
- mathematical         → step 1: vector_search; step 2: data_analysis (compute the metric); step 3: synthesize
- pv_query             → step 1: pv_search; step 2: vector_search on entry_logbook; step 3: synthesize
- status_check         → step 1: pv_search (live status + alarm probe for the named \
                         device — query should be the device/system name verbatim, e.g. \
                         "S26 radiation monitor"); step 2: vector_search on entry_logbook \
                         (recent incidents/maintenance for the device); step 3: \
                         bm25_search (broader keyword sweep); step 4: synthesize
- troubleshooting      → step 1: graph_search; step 2: vector_search; step 3: bm25_search; step 4: synthesize
- temporal / factual / general / multi_hop / comparative / daily_summary / studies_copilot \
                       → step 1: vector_search; step 2: bm25_search; step 3: synthesize
- code_generation      → step 1: vector_search on sdds_collection or oag_collection; \
                         step 2: bm25_search on same collection; step 3: synthesize

## Step query rules (avoid auditor rejection on narrow filters)
- data_analysis queries: describe the WIDE filter (e.g. \
  "Filter <table> where DESCRIPTION case-insensitive substring matches any of \
  ['vacuum leak','vac leak','leak'] AND date >= ..."), NOT a single exact phrase. \
  Always emit auditor-eligible JSON: \
  {{"count": int, "matched_sample": list[<=5], "near_miss_sample": list[<=5]}}.
- vector_search / bm25_search queries: one short sentence (≤ 12 words) covering \
  the primary entities/symptoms. Use SPECIFIC terminology from the query.

Step collections: pick from \
[entry_logbook, teams_messages, icms_collection, work_requests, sdds_collection, oag_collection] \
or [] for graph_search/pv_search/data_analysis/synthesize.

# Output — JSON only, no prose, no markdown fences, no <think> block
{{"intent":"...","collections":[...],"retrieval_strategy":"...","service_mode":"...",\
"entities":[...],"reasoning":"one sentence","temporal":{{...}} or null,\
"variants":["...","...","..."],\
"plan_steps":[{{"step":1,"tool":"...","query":"...","collections":[...],"rationale":"..."}},\
{{"step":2,"tool":"synthesize","query":null,"collections":[],"rationale":"..."}}]}}

Query: {query}"""
)


# ---------------------------------------------------------------------------
# Counting / aggregation intent detection
# ---------------------------------------------------------------------------
# Regex-only (no LLM) so it adds zero latency and is deterministic. The
# downstream effect is forcing data_analysis tool use; false positives are
# cheap (an extra tool call) and false negatives are the failure mode we
# want to eliminate (LLM-estimated counts that drift across runs).
_COUNTING_PATTERNS = [
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bnumber of\b",
    r"\btotal (number|count) of\b",
    r"\b(count|tally|enumerate)\b",
    r"\blist all\b",
    r"\bdistribution of\b",
    r"\baverage|mean|median|sum of\b",
    r"\bgroup(ed)? by\b",
    r"\bbreakdown (of|by)\b",
    r"\bpercentage of\b",
    r"\bfrequency of\b",
    r"\b(most|least) (common|frequent)\b",
]
import re as _counting_re
_COUNTING_REGEX = _counting_re.compile(
    "|".join(_COUNTING_PATTERNS), flags=_counting_re.IGNORECASE
)


def is_counting_query(query: str) -> bool:
    """Return True if the query asks for a count, aggregate, or distribution.

    Used by the orchestrator to set state["counting_intent"], which forces
    the data_analysis tool path instead of letting the LLM estimate counts
    from retrieved snippets (see DataAnalysisTool docstring for rationale).
    """
    if not query:
        return False
    return bool(_COUNTING_REGEX.search(query))


class QueryEnhancer:
    """
    Unified query enhancement pipeline.

    Steps:
      1. Acronym / domain-term resolution  (AcronymResolver from entry_rag_app)
      2. the facility synonym expansion              (SynonymManager from entry_rag_app)
      3. Temporal inline expansion          (TemporalProcessor.enhance_query)
      4. Structured date-range extraction   (TemporalDateParser.parse_query — LLM-based)
      5. Multi-query variant generation     (LLM, optional — requires llm kwarg)

    Args:
        llm: Optional LLM instance (e.g. claude-3-5-haiku).  When provided, step 5
             generates query_variants for multi-query retrieval.  When None the
             pipeline runs steps 1–4 only.
    """

    def __init__(self, llm=None, combined_llm=None):
        self.temporal_processor = TemporalProcessor()
        self.acronym_resolver = AcronymResolver()
        self.synonym_manager = SynonymManager()
        self._llm = llm
        # Dedicated LLM for the single combined temporal+variants call (naive_rag).
        # Falls back to self._llm when not set.
        self._combined_llm = combined_llm or llm

        # Lazy-init TemporalDateParser — it makes an Argo LLM call on parse_query()
        self._date_parser: Optional[Any] = None
        if _DATE_PARSER_OK:
            try:
                self._date_parser = TemporalDateParser(
                    argo_model="claude-3-5-sonnet",
                    argo_username="your-username",
                )
            except Exception as exc:
                logger.warning("TemporalDateParser init failed: %s", exc)

    # ------------------------------------------------------------------
    def enhance(
        self,
        query: str,
        generate_variants: bool = True,
        n_variants: int = 3,
        combined: bool = False,
        combined_routing: bool = False,
        combined_routing_with_plan: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full enhancement pipeline.

        Returns a dict with:
          original_query     — unchanged input
          enhanced_query     — acronym-resolved + synonym-expanded + temporal-expanded
          temporal_context   — None or {"start": ISO, "end": ISO, "start_date": MM/DD/YYYY, ...}
          has_temporal       — bool
          synonym_expansion  — metadata dict from SynonymManager
          query_variants     — list[str] (enhanced_query at index 0, LLM variants follow)
        """
        enhanced = query
        synonym_meta: Dict[str, Any] = {}

        # Step 1: Acronym resolution.
        # The real class (entry_rag_app) exposes `.expand(query, collection=None)`.
        # The fallback no-op class above also exposes `.expand` for API parity.
        try:
            expanded_acronyms = self.acronym_resolver.expand(enhanced)
            if expanded_acronyms:
                enhanced = expanded_acronyms
        except Exception as exc:
            logger.debug("Acronym resolution failed: %s", exc)

        # Step 2: Synonym expansion (the facility terminology — operator ↔ physicist)
        try:
            expanded, synonym_meta = self.synonym_manager.expand_query(
                enhanced, max_expansions=3
            )
            if expanded and expanded != enhanced:
                n_exp = synonym_meta.get("total_expansions", 0)
                logger.debug("Synonym expansion: %d term(s) expanded", n_exp)
                enhanced = expanded
        except Exception as exc:
            logger.debug("Synonym expansion failed: %s", exc)

        # Step 3: Inline temporal expansion
        #   e.g. "last week" → "(last week - 2026-03-16 to 2026-03-22)"
        try:
            expanded_temporal, _ = self.temporal_processor.enhance_query(enhanced)
            if expanded_temporal and expanded_temporal != enhanced:
                enhanced = expanded_temporal
        except Exception as exc:
            logger.debug("Inline temporal expansion failed: %s", exc)

        # Combined fast-path: one LLM call for temporal + variants (naive_rag)
        if combined and self._llm is not None:
            combined_result = self._enhance_combined(enhanced, query, n_variants)
            if combined_result is not None:
                return {
                    "original_query": query,
                    "enhanced_query": enhanced,
                    "temporal_context": combined_result["temporal_context"],
                    "has_temporal": combined_result["temporal_context"] is not None,
                    "synonym_expansion": synonym_meta,
                    "query_variants": combined_result["query_variants"],
                    "counting_intent": is_counting_query(query),
                }
            logger.warning("Combined enhancement failed — falling back to sequential path")

        # Combined routing+planning fast-path: ONE LLM call for router + temporal +
        # variants + plan. Used by agentic_rag to avoid the dedicated PlannerAgent
        # call (~12s sonnet46). Falls through to combined_routing on failure.
        if combined_routing_with_plan and (self._llm is not None or self._combined_llm is not None):
            rpv = self._enhance_route_plan_combined(enhanced, query, n_variants)
            if rpv is not None:
                return {
                    "original_query": query,
                    "enhanced_query": enhanced,
                    "temporal_context": rpv["temporal_context"],
                    "has_temporal": rpv["temporal_context"] is not None,
                    "synonym_expansion": synonym_meta,
                    "query_variants": rpv["query_variants"],
                    "counting_intent": (
                        rpv["query_intent"] == "counting" or is_counting_query(query)
                    ),
                    "_query_intent": rpv["query_intent"],
                    "_selected_databases": rpv["selected_databases"],
                    "_retrieval_strategy": rpv["retrieval_strategy"],
                    "_service_mode": rpv["service_mode"],
                    "_extracted_entities": rpv["extracted_entities"],
                    "_plan_steps": rpv["plan_steps"],
                    "_plan_raw": rpv["plan"],
                    "_routing_done": True,
                    "_plan_done": True,
                }
            logger.warning("Combined route+plan failed — falling back to combined routing")

        # Combined routing fast-path: one LLM call for router + temporal + variants.
        # Uses self._llm (sonnet46) for routing accuracy.
        if combined_routing and (self._llm is not None or self._combined_llm is not None):
            routing_result = self._enhance_and_route_combined(enhanced, query, n_variants)
            if routing_result is not None:
                return {
                    "original_query": query,
                    "enhanced_query": enhanced,
                    "temporal_context": routing_result["temporal_context"],
                    "has_temporal": routing_result["temporal_context"] is not None,
                    "synonym_expansion": synonym_meta,
                    "query_variants": routing_result["query_variants"],
                    "counting_intent": (
                        routing_result["query_intent"] == "counting"
                        or is_counting_query(query)
                    ),
                    "_query_intent": routing_result["query_intent"],
                    "_selected_databases": routing_result["selected_databases"],
                    "_retrieval_strategy": routing_result["retrieval_strategy"],
                    "_service_mode": routing_result["service_mode"],
                    "_extracted_entities": routing_result["extracted_entities"],
                    "_routing_done": True,
                }
            logger.warning("Combined routing failed — falling back to sequential path")

        # Step 4: Structured date-range extraction
        #   Primary:  TemporalDateParser (LLM-based, entry_rag_app)
        #   Fallback: parse YYYY-MM-DD patterns already embedded by step 3
        temporal_context: Optional[Dict[str, Any]] = None
        try:
            if self._date_parser is not None:
                # Refresh the parser's cached "today" before every call. The
                # upstream TemporalDateParser sets self.today in __init__ only
                # and never refreshes it, so a long-running FastAPI process
                # would resolve "yesterday" against the stale start-of-process
                # date. We rebind it here per-call so wall-clock drift never
                # silently shifts the temporal window.
                _now = datetime.now()
                self._date_parser.today = _now
                self._date_parser.today_str = _now.strftime("%m/%d/%Y")
                date_range = self._date_parser.parse_query(query)  # use raw query
                if date_range is not None:
                    filter_dict = self._date_parser.get_date_range_filter(date_range)
                    meta = filter_dict.get("metadata", {})
                    temporal_context = {
                        "start": meta.get("start_datetime"),   # ISO string (no Z)
                        "end": meta.get("end_datetime"),
                        "start_date": meta.get("start_date"),  # MM/DD/YYYY
                        "end_date": meta.get("end_date"),
                        "phrase": getattr(date_range, "temporal_phrase", ""),
                        "confidence": getattr(date_range, "confidence", 0.5),
                        "inferred": getattr(date_range, "inferred", True),
                    }
                    logger.info(
                        "Temporal context (LLM parser): %s → %s (phrase=%r)",
                        meta.get("start_date"),
                        meta.get("end_date"),
                        getattr(date_range, "temporal_phrase", ""),
                    )
        except Exception as exc:
            logger.debug("LLM date-range parsing failed: %s", exc)

        # Regex fallback — parse YYYY-MM-DD to YYYY-MM-DD from the inline-expanded query
        # (TemporalProcessor.enhance_query already computed these correctly)
        if temporal_context is None:
            temporal_context = self._parse_temporal_from_enhanced(enhanced)
            if temporal_context:
                logger.info(
                    "Temporal context (regex fallback): %s → %s",
                    temporal_context.get("start_date"),
                    temporal_context.get("end_date"),
                )

        # Step 5: Multi-query variant generation (LLM required)
        query_variants: List[str] = [enhanced]
        if generate_variants and self._llm is not None:
            try:
                variants = self._generate_variants(enhanced, n=n_variants)
                seen = {enhanced}
                for v in variants:
                    if v and v not in seen:
                        query_variants.append(v)
                        seen.add(v)
                if len(query_variants) > 1:
                    logger.info(
                        "Multi-query: %d variants generated", len(query_variants) - 1
                    )
            except Exception as exc:
                logger.warning("Multi-query generation failed: %s", exc)

        return {
            "original_query": query,
            "enhanced_query": enhanced,
            "temporal_context": temporal_context,
            "has_temporal": temporal_context is not None,
            "synonym_expansion": synonym_meta,
            "query_variants": query_variants,
            # Detect on the RAW query, not the enhanced one — synonym expansion
            # can introduce words like "number" or "list" that aren't in the
            # user's actual intent.
            "counting_intent": is_counting_query(query),
        }

    # ------------------------------------------------------------------
    def _generate_variants(self, query: str, n: int = 3) -> List[str]:
        """Call the LLM to generate n alternative phrasings of query."""
        prompt = _MULTI_QUERY_PROMPT.format(n=n, question=query)
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            lines = [
                line.strip()
                for line in text.strip().splitlines()
                if line.strip()
            ]
            # Filter out header-like lines and copies of the original
            variants = [
                ln for ln in lines
                if ln.lower() != query.lower()
                and not ln.lower().startswith(("original", "variant", "#", "-"))
                and len(ln) > 5
            ]
            return variants[:n]
        except Exception as exc:
            logger.warning("LLM variant generation failed: %s", exc)
            return []

    def _enhance_combined(
        self, enhanced_query: str, original_query: str, n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Single LLM call combining TemporalDateParser + query variant generation.

        Returns {"temporal_context": ..., "query_variants": [...]} or None on failure
        (caller falls back to the sequential two-call path).
        """
        from datetime import date as _date
        from src.utils.json_parser import parse_json_with_repair

        today = _date.today().isoformat()
        prompt = _COMBINED_PROMPT.format(today=today, n=n, query=original_query)
        try:
            response = self._combined_llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("Combined enhancement LLM call failed: %s", exc)
            return None

        parsed = parse_json_with_repair(raw, llm=self._combined_llm, context="combined_enhance")
        if not isinstance(parsed, dict):
            logger.warning("Combined enhancement returned non-dict: %s", type(parsed))
            return None

        tc_raw = parsed.get("temporal")
        temporal_context = None
        if isinstance(tc_raw, dict):
            temporal_context = {
                "start": tc_raw.get("start"),
                "end": tc_raw.get("end"),
                "start_date": tc_raw.get("start_date"),
                "end_date": tc_raw.get("end_date"),
                "phrase": tc_raw.get("phrase", ""),
                "confidence": tc_raw.get("confidence", 0.9),
                "inferred": tc_raw.get("inferred", True),
            }
            logger.info(
                "Temporal context (combined): %s → %s",
                tc_raw.get("start_date"), tc_raw.get("end_date"),
            )

        raw_variants = parsed.get("variants", [])
        seen = {enhanced_query.lower()}
        variants = [enhanced_query]
        for v in (raw_variants if isinstance(raw_variants, list) else []):
            if isinstance(v, str) and v.strip() and v.strip().lower() not in seen:
                variants.append(v.strip())
                seen.add(v.strip().lower())
        if len(variants) > 1:
            logger.info("Combined multi-query: %d variants", len(variants) - 1)

        return {"temporal_context": temporal_context, "query_variants": variants}

    def _enhance_and_route_combined(
        self, enhanced_query: str, original_query: str, n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Single LLM call combining Router + TemporalDateParser + query variants.

        Returns a dict with both enhancement fields (temporal_context, query_variants)
        AND routing fields (query_intent, selected_databases, retrieval_strategy,
        service_mode, extracted_entities), or None on failure (caller falls back
        to the separate router + sequential enhancement path).
        """
        from datetime import date as _date
        from src.utils.json_parser import parse_json_with_repair
        from src.vectorstore.collection_registry import COLLECTION_OPTIONS

        today = _date.today().isoformat()
        prompt = _COMBINED_ROUTING_PROMPT.format(today=today, n=n, query=original_query)
        # Use self._llm (sonnet46) for combined routing — router accuracy matters
        # more than latency here. _combined_llm (haiku45) is for naive_rag's
        # lighter temporal+variants-only combined call.
        routing_llm = self._llm or self._combined_llm
        try:
            response = routing_llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("Combined routing LLM call failed: %s", exc)
            return None

        parsed = parse_json_with_repair(raw, llm=routing_llm, context="combined_routing")
        if not isinstance(parsed, dict):
            logger.warning("Combined routing returned non-dict: %s", type(parsed))
            return None

        # temporal_context
        tc_raw = parsed.get("temporal")
        temporal_context = None
        if isinstance(tc_raw, dict):
            temporal_context = {
                "start": tc_raw.get("start"),
                "end": tc_raw.get("end"),
                "start_date": tc_raw.get("start_date"),
                "end_date": tc_raw.get("end_date"),
                "phrase": tc_raw.get("phrase", ""),
                "confidence": tc_raw.get("confidence", 0.9),
                "inferred": tc_raw.get("inferred", True),
            }

        # query_variants
        seen = {enhanced_query.lower()}
        variants = [enhanced_query]
        for v in (parsed.get("variants") or []):
            if isinstance(v, str) and v.strip() and v.strip().lower() not in seen:
                variants.append(v.strip())
                seen.add(v.strip().lower())

        # routing fields — validate collections
        raw_collections = parsed.get("collections") or ["all_databases"]
        valid_collections = [c for c in raw_collections if c in COLLECTION_OPTIONS]
        if not valid_collections:
            valid_collections = ["all_databases"]

        intent = parsed.get("intent", "general")
        strategy = parsed.get("retrieval_strategy", "VECTOR_BM25")
        service_mode = parsed.get("service_mode", "agentic_rag")
        entities = parsed.get("entities") or []

        logger.info(
            "Combined routing: intent=%s strategy=%s collections=%s",
            intent, strategy, valid_collections,
        )
        if temporal_context:
            logger.info(
                "Combined routing temporal: %s → %s",
                temporal_context.get("start_date"), temporal_context.get("end_date"),
            )
        if len(variants) > 1:
            logger.info("Combined routing variants: %d", len(variants) - 1)

        return {
            "temporal_context": temporal_context,
            "query_variants": variants,
            "query_intent": intent,
            "selected_databases": valid_collections,
            "retrieval_strategy": strategy,
            "service_mode": service_mode,
            "extracted_entities": entities,
        }

    def _enhance_route_plan_combined(
        self, enhanced_query: str, original_query: str, n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Single LLM call: Router + TemporalDateParser + variants + slim Planner.

        Returns the same fields as _enhance_and_route_combined PLUS:
          - plan_steps: list of dicts (matches PlannerAgent output shape)
          - plan: raw text representation of the plan

        Falls back to None on failure so caller can route to the legacy path.
        """
        from datetime import date as _date
        from src.utils.json_parser import parse_json_with_repair
        from src.vectorstore.collection_registry import COLLECTION_OPTIONS

        today = _date.today().isoformat()
        prompt = _COMBINED_RPV_PROMPT.format(today=today, n=n, query=original_query)
        routing_llm = self._llm or self._combined_llm
        try:
            response = routing_llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("Combined route+plan LLM call failed: %s", exc)
            return None

        parsed = parse_json_with_repair(raw, llm=routing_llm, context="combined_route_plan")
        if not isinstance(parsed, dict):
            logger.warning("Combined route+plan returned non-dict: %s", type(parsed))
            return None

        # temporal_context (same logic as _enhance_and_route_combined)
        tc_raw = parsed.get("temporal")
        temporal_context = None
        if isinstance(tc_raw, dict):
            temporal_context = {
                "start": tc_raw.get("start"),
                "end": tc_raw.get("end"),
                "start_date": tc_raw.get("start_date"),
                "end_date": tc_raw.get("end_date"),
                "phrase": tc_raw.get("phrase", ""),
                "confidence": tc_raw.get("confidence", 0.9),
                "inferred": tc_raw.get("inferred", True),
            }

        # variants
        seen = {enhanced_query.lower()}
        variants = [enhanced_query]
        for v in (parsed.get("variants") or []):
            if isinstance(v, str) and v.strip() and v.strip().lower() not in seen:
                variants.append(v.strip())
                seen.add(v.strip().lower())

        # routing
        raw_collections = parsed.get("collections") or ["all_databases"]
        valid_collections = [c for c in raw_collections if c in COLLECTION_OPTIONS]
        if not valid_collections:
            valid_collections = ["all_databases"]
        intent = parsed.get("intent", "general")
        strategy = parsed.get("retrieval_strategy", "VECTOR_BM25")
        service_mode = parsed.get("service_mode", "agentic_rag")
        entities = parsed.get("entities") or []

        # plan_steps — must end with synthesize; if missing/bad, return None so
        # caller falls back to the dedicated PlannerAgent (no silent bad plans).
        raw_steps = parsed.get("plan_steps")
        if not isinstance(raw_steps, list) or len(raw_steps) < 2:
            logger.warning("Combined route+plan returned invalid plan_steps — falling back")
            return None
        plan_steps: List[Dict[str, Any]] = []
        for i, s in enumerate(raw_steps, 1):
            if isinstance(s, dict):
                plan_steps.append({
                    "step": s.get("step", i),
                    "tool": s.get("tool", "synthesize"),
                    "query": s.get("query"),
                    "collections": s.get("collections", []) if isinstance(s.get("collections", []), list) else [],
                    "rationale": s.get("rationale", ""),
                })
            elif isinstance(s, str):
                plan_steps.append({
                    "step": i, "tool": "vector_search", "query": s[:200],
                    "collections": [], "rationale": "Coerced from string",
                })
        if not plan_steps or plan_steps[-1].get("tool") != "synthesize":
            plan_steps.append({
                "step": len(plan_steps) + 1, "tool": "synthesize",
                "query": None, "collections": [],
                "rationale": "Auto-appended terminal synthesize step",
            })

        logger.info(
            "Combined route+plan: intent=%s strategy=%s collections=%s steps=%d",
            intent, strategy, valid_collections, len(plan_steps),
        )

        return {
            "temporal_context": temporal_context,
            "query_variants": variants,
            "query_intent": intent,
            "selected_databases": valid_collections,
            "retrieval_strategy": strategy,
            "service_mode": service_mode,
            "extracted_entities": entities,
            "plan_steps": plan_steps,
            "plan": json.dumps(plan_steps, indent=2),
        }

    @staticmethod
    def _parse_temporal_from_enhanced(enhanced_query: str) -> Optional[Dict[str, Any]]:
        """
        Regex fallback: extract a structured date range from an inline-expanded
        query string produced by TemporalProcessor.enhance_query().

        TemporalProcessor embeds patterns like:
            "(this week - 2026-03-16 to 2026-03-22)"
            "(last month - February 2026)"   ← no range, ignored
            "(recently - past 30 days (2026-02-21 to 2026-03-23))"

        Looks for the first "YYYY-MM-DD to YYYY-MM-DD" occurrence.
        Returns None if no such pattern is found.
        """
        import re
        from datetime import datetime

        match = re.search(
            r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
            enhanced_query,
        )
        if not match:
            return None

        start_str, end_str = match.group(1), match.group(2)
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59
            )
            return {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "start_date": start_dt.strftime("%m/%d/%Y"),
                "end_date": end_dt.strftime("%m/%d/%Y"),
                "phrase": "",
                "confidence": 0.8,
                "inferred": True,
            }
        except ValueError:
            return None
