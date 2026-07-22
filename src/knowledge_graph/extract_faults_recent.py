"""
Nightly Fault / Procedure enrichment for the troubleshooting agent.

Pulls recent troubleshooting-flagged LogEntry nodes from Neo4j, asks an LLM to
extract structured fault + procedure metadata (matching the sr_rag_kg notebook
schema), and MERGEs Fault / Procedure / PRECEDES nodes/edges. Additive only:
existing KG schema and queries are untouched. Idempotent — re-running on the
same LogEntry skips already-extracted faults (keyed by source entry_id).

Usage:
    python -m src.knowledge_graph.extract_faults_recent --days 60
    python -m src.knowledge_graph.extract_faults_recent --days 7 --dry-run
    python -m src.knowledge_graph.extract_faults_recent --limit 50 --since 2026-04-01

CRITICAL: this script consumes LLM credits and writes to the production KG.
Run interactively first with --dry-run to inspect the planned changes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on sys.path when run as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.json_parser import parse_json_with_repair  # noqa: E402

logger = logging.getLogger(__name__)


FAULT_EXTRACTION_PROMPT = """\
# Role
You are an expert the facility accelerator physicist. Given a logbook troubleshooting \
entry, extract structured fault metadata using Chain-of-Thought reasoning.

# Schema (return JSON ONLY, after <think>)
{{"fault_category": "<one of: beam_loss|injection_issue|orbit_drift|rf_fault|\
vacuum_fault|magnet_fault|controls_fault|bpm_fault|kicker_fault|undulator_fault|\
timing_fault|interlock_fault|other>",
"fault_summary": "<one sentence>",
"symptoms": ["<symptom 1>", "<symptom 2>"],
"systems": ["<system canonical name, e.g. SR, BTS, Booster, Linac>"],
"components": ["<component, e.g. K2, BPM, IK3 PSU>"],
"root_cause": "<root cause if identified, else 'unknown'>",
"actions_taken": ["<short imperative phrase>"],
"preventative_measures": ["<measure>"],
"severity": "CRITICAL|HIGH|MEDIUM|LOW"}}

# Input entry
<entry_id>{entry_id}</entry_id>
<created>{created}</created>
<author>{author}</author>
<content>{content}</content>

# Constraints
- Skip generic terms. Use the facility canonical names (SR not Storage Ring).
- If the entry is not actually about a fault, return fault_category="other" \
and severity="LOW" so the caller can drop it."""


def fetch_recent_log_entries(
    neo4j_client,
    since_iso: str,
    limit: int,
    skip_already_extracted: bool = True,
) -> List[Dict[str, Any]]:
    """Pull recent LogEntry rows that haven't yet had Fault extraction.

    Looks for any LogEntry with creation_datetime >= since_iso. The
    skip_already_extracted flag joins on Fault.source_entry_id to avoid
    re-processing.
    """
    cypher = """
    MATCH (e:LogEntry)
    WHERE e.creation_datetime >= datetime($since_iso)
      AND coalesce(e.is_troubleshooting, true) = true
    OPTIONAL MATCH (f:Fault) WHERE f.source_entry_id = coalesce(e.entry_id, e.id)
    WITH e, count(f) AS already
    WHERE NOT $skip OR already = 0
    RETURN e.id            AS id,
           coalesce(e.entry_id, e.id) AS entry_id,
           e.title          AS title,
           coalesce(e.text, e.content) AS content,
           e.author         AS author,
           toString(e.creation_datetime) AS created
    ORDER BY e.creation_datetime DESC
    LIMIT $limit
    """
    return neo4j_client.run_query(
        cypher,
        {"since_iso": since_iso, "limit": limit, "skip": skip_already_extracted},
    )


def extract_fault_from_entry(
    llm, entry: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Call the LLM on one entry; return parsed dict or None on failure."""
    from langchain_core.messages import HumanMessage, SystemMessage

    content = (entry.get("content") or "")[:3500]
    if not content.strip():
        return None

    messages = [
        SystemMessage(content=(
            "You are a deterministic fault-extraction engine. Respond with a "
            "<think> block then JSON only."
        )),
        HumanMessage(content=FAULT_EXTRACTION_PROMPT.format(
            entry_id=entry.get("entry_id", "?"),
            created=entry.get("created", "?"),
            author=entry.get("author", "?"),
            content=content,
        )),
    ]
    try:
        resp = llm.invoke(messages)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = parse_json_with_repair(raw, llm=llm, context="fault_extraction")
        if not isinstance(data, dict):
            return None
        # Tag with provenance so re-runs are idempotent
        data["_source_entry_id"] = entry.get("entry_id")
        data["_source_log_id"] = entry.get("id")
        data["_source_created"] = entry.get("created")
        data["_source_author"] = entry.get("author")
        return data
    except Exception as e:
        logger.warning("Fault extraction failed for %s: %s", entry.get("entry_id"), e)
        return None


def write_fault_to_kg(neo4j_client, fault: Dict[str, Any]) -> None:
    """Idempotent write of a Fault node + EXTRACTED_FROM + RESOLVED_BY edges."""
    entry_id = fault.get("_source_entry_id") or fault.get("_source_log_id")
    if not entry_id:
        return
    fault_id = f"fault:{entry_id}"

    # 1. Fault node
    neo4j_client.upsert_node("Fault", fault_id, {
        "name": fault.get("fault_summary", "")[:200],
        "description": fault.get("fault_summary", "")[:1000],
        "category": fault.get("fault_category", "other"),
        "symptoms": json.dumps(fault.get("symptoms", []) or []),
        "root_cause": fault.get("root_cause", "unknown"),
        "severity": fault.get("severity", "MEDIUM"),
        "source_entry_id": entry_id,
        "source_author": fault.get("_source_author", ""),
        "extracted_at": datetime.utcnow().isoformat(),
    })

    # 2. EXTRACTED_FROM edge → original LogEntry
    log_id = fault.get("_source_log_id") or entry_id
    try:
        neo4j_client.upsert_edge(fault_id, log_id, "EXTRACTED_FROM", {})
    except Exception as e:
        logger.debug("EXTRACTED_FROM edge skipped (%s): %s", log_id, e)

    # 3. Procedure node + RESOLVED_BY edge for each action group
    actions = fault.get("actions_taken", []) or []
    if actions:
        proc_id = f"proc:{entry_id}"
        neo4j_client.upsert_node("Procedure", proc_id, {
            "name": f"Recovery for {fault.get('fault_summary', '')[:120]}",
            "steps": " | ".join(str(a) for a in actions[:10]),
            "preventative": json.dumps(fault.get("preventative_measures", []) or []),
            "source_entry_id": entry_id,
        })
        neo4j_client.upsert_edge(fault_id, proc_id, "RESOLVED_BY", {})


def add_precedes_edges(
    neo4j_client, system: str, within_hours: int = 6
) -> int:
    """Add temporal PRECEDES edges between faults on the same system whose
    source LogEntries are within `within_hours` of each other. Idempotent
    via MERGE. Returns count of edges written."""
    cypher = """
    MATCH (f1:Fault), (f2:Fault)
    MATCH (e1:LogEntry {id: f1.source_entry_id})
    MATCH (e2:LogEntry {id: f2.source_entry_id})
    WHERE f1 <> f2
      AND $system IN coalesce(f1.systems, [])
      AND $system IN coalesce(f2.systems, [])
      AND e1.creation_datetime < e2.creation_datetime
      AND duration.inHours(e1.creation_datetime, e2.creation_datetime).hours <= $within
    MERGE (f1)-[:PRECEDES]->(f2)
    RETURN count(*) AS edges
    """
    try:
        rows = neo4j_client.run_query(cypher, {"system": system, "within": within_hours})
        return int(rows[0]["edges"]) if rows else 0
    except Exception as e:
        logger.warning("PRECEDES edges failed for system %s: %s", system, e)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--days", type=int, default=60,
                        help="Look back this many days (default 60).")
    parser.add_argument("--since", type=str, default=None,
                        help="ISO date to look back from (overrides --days).")
    parser.add_argument("--limit", type=int, default=200,
                        help="Maximum entries to process this run (default 200).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned writes without touching Neo4j.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-extract even for entries that already have a Fault node.")
    parser.add_argument("--no-precedes", action="store_true",
                        help="Skip the PRECEDES temporal edge pass.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.since:
        since_iso = args.since
    else:
        since_iso = (datetime.utcnow() - timedelta(days=args.days)).isoformat()

    # Lazy imports keep `--help` working without an Argo / Neo4j env.
    from src.integrations.neo4j_client import Neo4jClient
    from src.integrations.argo_llm import build_argo_llm  # type: ignore

    neo4j_client = Neo4jClient()
    llm = build_argo_llm(role="fast")  # haiku-tier ok for structured extraction

    logger.info("Fetching LogEntries since %s (limit=%d, reprocess=%s)",
                since_iso, args.limit, args.reprocess)
    entries = fetch_recent_log_entries(
        neo4j_client, since_iso, args.limit,
        skip_already_extracted=not args.reprocess,
    )
    logger.info("Got %d candidate entries", len(entries))

    written = 0
    skipped = 0
    systems_touched: set = set()
    for entry in entries:
        fault = extract_fault_from_entry(llm, entry)
        if not fault or fault.get("fault_category") in (None, "other"):
            skipped += 1
            continue

        if args.dry_run:
            logger.info("[DRY] would write fault for %s: %s [%s] systems=%s",
                        entry.get("entry_id"),
                        fault.get("fault_summary", "")[:80],
                        fault.get("severity"),
                        fault.get("systems"))
        else:
            write_fault_to_kg(neo4j_client, fault)
            for s in (fault.get("systems") or []):
                systems_touched.add(s)
        written += 1

    logger.info("Extraction done. wrote=%d skipped=%d", written, skipped)

    if not args.dry_run and not args.no_precedes:
        for system in systems_touched:
            n = add_precedes_edges(neo4j_client, system)
            if n:
                logger.info("PRECEDES edges added for %s: %d", system, n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
