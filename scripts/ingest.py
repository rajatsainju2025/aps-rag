#!/usr/bin/env python3
"""
ingest.py — build/refresh the RAG stores from your documents.

This is the single, plug-and-play ingestion entry point. It auto-discovers every
subfolder of ``data/raw/`` as a separate database, chunks + embeds each one, and
writes:

  • dense vectors  → Qdrant   (semantic search)
  • a BM25 index   → Elasticsearch (keyword search)
  • (optional)     → Neo4j knowledge graph  (--build-kg, needs an LLM backend)

Folder-per-database convention (no code changes to add a database):

    data/raw/
    ├── my_docs/            ← database name = folder name  (→ source_database metadata)
    │   ├── a.json          ← .json (list / {"entries":[...]} / single record), .txt, .md
    │   └── b.txt
    └── another_db/
        └── ...

Optionally place a metadata schema at ``schemas/<db>.yaml`` (or
``data/raw/<db>/schema.yaml``). With no schema, a generic fallback embeds
auto-captured metadata into each chunk. See schemas/README.md.

Usage:
    python scripts/ingest.py                     # ingest ALL databases under data/raw/
    python scripts/ingest.py --databases my_docs # ingest specific database(s)
    python scripts/ingest.py --list              # list discovered databases and exit
    python scripts/ingest.py --strategy whole-doc
    python scripts/ingest.py --build-kg          # also build the Neo4j knowledge graph
"""
import argparse
import logging
import sys
from pathlib import Path

# Make `config.*` / `src.*` importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from src.vectorstore.collection_registry import (  # noqa: E402
    discover_raw_databases,
    get_collection_names,
    RAW_DATA_ROOT,
)
from src.utils.document_loader import DocumentLoader  # noqa: E402
from src.utils.schema_loader import load_schema, SchemaError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def _init_clients(settings, strategy):
    """Construct the store clients + embedding model, with clear failure msgs."""
    from src.integrations.qdrant_client import AgRAGQdrantClient
    from src.integrations.elasticsearch_client import AgRAGElasticsearchClient
    from src.vectorstore.builder import AgRAGVectorStoreBuilder
    from src.integrations.argo.llm_client import LLMFactory  # local e5 embeddings

    qdrant = AgRAGQdrantClient(
        host=settings.vector.qdrant_host, port=settings.vector.qdrant_port
    )
    es = AgRAGElasticsearchClient(
        host=settings.vector.es_host,
        port=settings.vector.es_port,
        username=settings.vector.es_username,
        password=settings.vector.es_password,
    )

    if not qdrant.health_check():
        sys.exit(
            f"ERROR: cannot reach Qdrant at {settings.vector.qdrant_host}:{settings.vector.qdrant_port}.\n"
            f"       Start it (see infrastructure/docker-compose.yml) or fix `vector.qdrant_*` in your config."
        )
    if not es.health_check():
        sys.exit(
            f"ERROR: cannot reach Elasticsearch at {settings.vector.es_host}:{settings.vector.es_port}.\n"
            f"       Start it (see infrastructure/docker-compose.yml) or fix `vector.es_*` in your config."
        )

    logger.info("Loading embedding model '%s' (device=%s)...",
                settings.vector.embedding_model, settings.vector.embedding_device)
    embedding_model = LLMFactory.create_embeddings(settings)

    builder = AgRAGVectorStoreBuilder(
        settings, qdrant, es, embedding_model, chunking_strategy=strategy
    )
    return builder


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest data/raw/<database>/ folders into the RAG stores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--databases", nargs="+", default=None,
                        help="Database folder name(s) to ingest (default: all under data/raw/).")
    parser.add_argument("--strategy", choices=["granular-chunking", "whole-doc"], default=None,
                        help="Chunking strategy (default: config chunking.strategy).")
    parser.add_argument("--build-kg", action="store_true",
                        help="Also build the Neo4j knowledge graph (requires an LLM backend).")
    parser.add_argument("--list", action="store_true",
                        help="List discovered databases and exit.")
    parser.add_argument("--data-root", default=None,
                        help="Override the data/raw root directory.")
    args = parser.parse_args()

    raw_root = Path(args.data_root) if args.data_root else RAW_DATA_ROOT
    discovered = discover_raw_databases(raw_root)

    if args.list:
        if discovered:
            print("Discovered databases under %s:" % raw_root)
            for db in discovered:
                print(f"  - {db}")
        else:
            print(f"No databases found under {raw_root}. Create data/raw/<name>/ and drop files in.")
        return 0

    settings = get_settings()
    strategy = args.strategy or getattr(settings.chunking, "strategy", "granular-chunking")

    databases = args.databases or discovered
    if not databases:
        logger.warning("No databases to ingest under %s. Create data/raw/<name>/ and drop files in.", raw_root)
        return 0

    builder = _init_clients(settings, strategy)
    loader = DocumentLoader()

    summary = []
    all_docs = []  # collected for optional KG build
    for db in databases:
        logger.info("=" * 68)
        logger.info("DATABASE: %s   (strategy=%s)", db, strategy)
        try:
            schema = load_schema(db)
        except SchemaError as exc:
            sys.exit(f"ERROR: {exc}")

        docs = loader.load_database(db, schema=schema, raw_root=raw_root)
        if not docs:
            logger.warning("  No documents found for '%s' — skipping.", db)
            summary.append((db, 0, 0, "empty"))
            continue

        result = builder.build_collection(db, documents=docs)
        names = get_collection_names(db, strategy)
        logger.info("  → %d vectors indexed into '%s' (Qdrant + Elasticsearch)",
                    result.get("indexed", 0), names["qdrant_collection"])
        summary.append((db, len(docs), result.get("indexed", 0), names["qdrant_collection"]))
        all_docs.extend(docs)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 68)
    logger.info("INGEST SUMMARY")
    logger.info("%-24s %8s %10s   %s", "database", "docs", "vectors", "collection")
    for db, ndocs, nvec, coll in summary:
        logger.info("%-24s %8d %10d   %s", db, ndocs, nvec, coll)

    # ── Optional: knowledge graph ─────────────────────────────────────────────
    if args.build_kg:
        if not all_docs:
            logger.warning("No documents ingested — skipping knowledge-graph build.")
        else:
            logger.info("=" * 68)
            logger.info("Building knowledge graph (Neo4j) from %d documents...", len(all_docs))
            try:
                from src.knowledge_graph.build import build_kg
                kg_stats = build_kg(settings, all_docs, incremental=True)
                logger.info("KG build complete: %s", kg_stats)
            except Exception as exc:  # noqa: BLE001
                logger.error("Knowledge-graph build failed: %s", exc, exc_info=True)
                logger.error("Vectors + BM25 were still indexed successfully. "
                             "Check Neo4j (graph.*) and your LLM backend (llm.*) config.")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
