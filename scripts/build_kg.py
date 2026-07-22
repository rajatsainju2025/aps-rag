#!/usr/bin/env python3
"""
build_kg.py — build/update the Neo4j knowledge graph from your ingested data.

Loads the documents from data/raw/<database>/ folders, extracts entities +
relations with the configured LLM, and persists nodes/triplets to Neo4j.

Requires an LLM backend (see config `llm.*`) and a running Neo4j (see config
`graph.*` / .env). This is also runnable as `scripts/ingest.py --build-kg`.

Examples:
    python scripts/build_kg.py                    # all databases (incremental)
    python scripts/build_kg.py --databases my_docs
    python scripts/build_kg.py --full             # full rebuild (ignore delta)
"""
import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_kg")


def main() -> int:
    from config.settings import require_valid_settings
    from src.vectorstore.collection_registry import discover_raw_databases
    from src.utils.document_loader import DocumentLoader
    from src.utils.schema_loader import load_schema, SchemaError
    from src.knowledge_graph.build import build_kg

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--databases", nargs="+", default=None, help="Database(s) to include (default: all).")
    p.add_argument("--full", action="store_true", help="Full rebuild (ignore incremental delta).")
    args = p.parse_args()

    settings = require_valid_settings(require_llm=True)  # KG extraction needs an LLM

    loader = DocumentLoader()
    databases = args.databases or discover_raw_databases()
    if not databases:
        logger.warning("No databases found under data/raw/. Nothing to build.")
        return 0

    docs = []
    for db in databases:
        try:
            schema = load_schema(db)
        except SchemaError as exc:
            sys.exit(f"ERROR: {exc}")
        docs.extend(loader.load_database(db, schema=schema))

    if not docs:
        logger.warning("No documents loaded. Nothing to build.")
        return 0

    logger.info("Building knowledge graph from %d documents across %d database(s)...", len(docs), len(databases))
    try:
        stats = build_kg(settings, docs, incremental=not args.full)
    except Exception as exc:  # noqa: BLE001
        logger.error("KG build failed: %s", exc, exc_info=True)
        return 1
    logger.info("KG build complete: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
