#!/usr/bin/env python3
"""
generate_inpars.py — synthesize evaluation QA pairs (question + nuggets) from your
source documents, InPARS-style.

InPARS (Bonifacio et al., SIGIR 2022) prompts an LLM to read a source document and
generate a realistic question that is answerable from it. This tool extends that to
also emit the **expected answer** and the **nuggets** — the atomic, extractive
supporting facts the answer depends on. Nuggets are the unit a retrieval/RAG
evaluator scores for recall, so a nugget-annotated set lets you measure whether a
system surfaced the facts a correct answer requires.

It reads the same folder-per-database corpus you ingest (data/raw/<db>/), samples
documents, and writes one JSONL row per generated item:

    {
      "id": "...", "source_database": "...", "source_doc_id": "...",
      "question": "...", "answer": "...", "nuggets": ["...", "..."],
      "answerable": true, "generator_model": "...", "llm_provider": "..."
    }

Requires an LLM backend (config `llm.*`). Examples:

    python scripts/generate_inpars.py --database example_database --num 5 --output qa.jsonl
    python scripts/generate_inpars.py --num 3 --output qa.jsonl        # sample across all databases
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("generate_inpars")

# Generic, institution-neutral InPARS prompt. Requests question + answer + nuggets.
PROMPT_TEMPLATE = """You are building an evaluation benchmark for a retrieval-augmented \
question-answering (RAG) system. Read the SOURCE DOCUMENT and produce ONE realistic \
question that a user of this corpus would plausibly ask AND that is answerable using \
ONLY this document.

Return STRICT JSON and nothing else:
{{
  "question": "a natural, specific question answerable from this document",
  "answer": "the correct answer, grounded in the document",
  "nuggets": ["atomic fact 1", "atomic fact 2"],
  "answerable": true
}}

Rules:
- Each NUGGET is a single, self-contained, verifiable fact taken directly from the
  document text (extractive — do NOT infer or add outside knowledge). Provide 2-6.
- Nuggets are the unit an evaluator checks for recall, so keep each to ONE fact.
- The answer must be fully supported by the nuggets.
- If the document is too sparse to yield a clean question, return exactly:
  {{"question": null, "answerable": false, "reason": "why"}}

SOURCE DOCUMENT
collection: {source_database}
id: {doc_id}
---
{content}
"""


def _sample_documents(databases, num_per_db, seed):
    from src.utils.document_loader import DocumentLoader
    from src.utils.schema_loader import load_schema

    rng = random.Random(seed)
    loader = DocumentLoader()
    sampled = []
    for db in databases:
        docs = loader.load_database(db, schema=load_schema(db))
        if not docs:
            logger.warning("No documents in database '%s' — skipping", db)
            continue
        pick = docs if num_per_db <= 0 or num_per_db >= len(docs) else rng.sample(docs, num_per_db)
        sampled.extend(pick)
    return sampled


def main() -> int:
    from config.settings import require_valid_settings
    from src.vectorstore.collection_registry import discover_raw_databases
    from src.integrations.argo.llm_client import LLMFactory
    from src.utils.json_parser import parse_json_with_repair
    from langchain_core.messages import HumanMessage

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", default=None, help="Database to sample from (default: all under data/raw/).")
    p.add_argument("--num", type=int, default=5, help="Documents to sample per database (0 = all).")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    p.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature for diversity.")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed (deterministic).")
    args = p.parse_args()

    settings = require_valid_settings(require_llm=True)

    databases = [args.database] if args.database else discover_raw_databases()
    if not databases:
        logger.warning("No databases found under data/raw/. Nothing to do.")
        return 0

    docs = _sample_documents(databases, args.num, args.seed)
    if not docs:
        logger.warning("No documents sampled. Nothing to do.")
        return 0

    llm = LLMFactory.get_synthesis_llm(settings.argo.username, settings.argo.base_url)
    try:
        llm.temperature = args.temperature  # diversity across drafts
    except Exception:
        pass
    model_name = getattr(llm, "model", None) or getattr(settings.llm, "model", "unknown")

    rows = []
    for i, doc in enumerate(docs, 1):
        prompt = PROMPT_TEMPLATE.format(
            source_database=doc.get("source_database", ""),
            doc_id=doc.get("id", ""),
            content=(doc.get("content", "") or "")[:6000],
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            parsed = parse_json_with_repair(raw, context="generate_inpars") or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generation failed for doc %s: %s", doc.get("id"), exc)
            parsed = {"question": None, "answerable": False, "reason": f"generation_error: {exc}"}

        rows.append({
            "id": f"INPARS-{i:04d}",
            "source_database": doc.get("source_database", ""),
            "source_doc_id": doc.get("id", ""),
            "question": parsed.get("question"),
            "answer": parsed.get("answer"),
            "nuggets": parsed.get("nuggets", []),
            "answerable": parsed.get("answerable", parsed.get("question") is not None),
            "generator_model": model_name,
            "llm_provider": settings.llm.provider,
        })
        logger.info("[%d/%d] %s → %s", i, len(docs), doc.get("id"),
                    "OK" if parsed.get("question") else "no-question")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in rows if r["question"])
    n_nug = sum(len(r["nuggets"]) for r in rows)
    logger.info("Wrote %d rows (%d with a question, %d nuggets total) → %s",
                len(rows), n_ok, n_nug, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
