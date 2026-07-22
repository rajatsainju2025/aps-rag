#!/usr/bin/env python3
"""
query.py — ask a question against your ingested corpus.

Two modes:

  --retrieve-only   Run hybrid retrieval (vector + BM25) locally and print the
                    top chunks. Requires NO LLM — ideal for verifying that
                    ingestion worked and retrieval returns relevant results.

  (default)         POST the question to a running APS-RAG API for a full,
                    generated, cited answer. Start the API first:
                        bash run_api.sh        (or: uvicorn api.main:app --port 8000)

Examples:
    python scripts/query.py --retrieve-only "how is torque drift fixed?"
    python scripts/query.py "how is torque drift fixed?"            # full answer via API
    python scripts/query.py --collection example_database "throughput?"
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _retrieve_only(args) -> int:
    from config.settings import get_settings
    from src.integrations.qdrant_client import AgRAGQdrantClient
    from src.integrations.elasticsearch_client import AgRAGElasticsearchClient
    from src.integrations.argo.llm_client import LLMFactory
    from src.retrieval.vector_retriever import VectorRetriever
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_fusion import AdaptiveHybridFusion

    s = get_settings()
    strategy = args.strategy or getattr(s.chunking, "strategy", "granular-chunking")
    qdrant = AgRAGQdrantClient(host=s.vector.qdrant_host, port=s.vector.qdrant_port)
    es = AgRAGElasticsearchClient(
        host=s.vector.es_host, port=s.vector.es_port,
        username=s.vector.es_username, password=s.vector.es_password,
    )
    if not qdrant.health_check() or not es.health_check():
        sys.exit("ERROR: Qdrant/Elasticsearch not reachable — start them and check your config.")

    embed = LLMFactory.create_embeddings(s)
    vr = VectorRetriever(qdrant, embed, top_k=args.top_k, chunking_strategy=strategy)
    br = BM25Retriever(es, top_k=args.top_k, chunking_strategy=strategy)

    collections = [args.collection]
    vdocs = vr.retrieve(args.query, collections, top_k=args.top_k)
    bdocs = br.retrieve(args.query, collections, top_k=args.top_k)
    fused = AdaptiveHybridFusion().fuse(vdocs, bdocs, [], query_intent="general")

    top = fused[: args.top_k]
    print(f"\nTop {len(top)} chunks for: {args.query!r}\n" + "=" * 70)
    for i, d in enumerate(top, 1):
        md = d.metadata
        print(f"[{i}] db={md.get('source_database','?')} id={md.get('id','?')} "
              f"score={md.get('rrf_score', md.get('score','?'))}")
        print(f"    {(d.page_content or '')[:240].strip()}\n")
    if not top:
        print("(no results — did you run scripts/ingest.py first?)")
    return 0


def _via_api(args) -> int:
    import httpx
    payload = {
        "query": args.query,
        "service_mode": args.service_mode,
        "collection": args.collection,
        "top_k": args.top_k,
        "retrieval_strategy": args.retrieval_strategy,
    }
    url = args.api_url.rstrip("/") + "/query"
    try:
        resp = httpx.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"ERROR: could not reach the APS-RAG API at {url}: {exc}\n"
            f"       Start it with `bash run_api.sh` (or use --retrieve-only for a local, LLM-free check)."
        )
    data = resp.json()
    print("\nANSWER\n" + "=" * 70)
    print(data.get("answer", "(no answer field in response)"))
    srcs = data.get("sources") or data.get("citations") or []
    if srcs:
        print("\nSOURCES\n" + "-" * 70)
        print(json.dumps(srcs, indent=2)[:2000])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Query the ingested corpus (retrieval-only locally, or full answer via the API).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument("query", help="Your question.")
    p.add_argument("--retrieve-only", action="store_true",
                   help="Run hybrid retrieval locally and print top chunks (no LLM).")
    p.add_argument("--collection", default="all_databases",
                   help="Database to search, or 'all_databases' (default).")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--strategy", choices=["granular-chunking", "whole-doc"], default=None)
    # API-mode options
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--service-mode", default="agentic_rag", choices=["naive_rag", "agentic_rag"])
    p.add_argument("--retrieval-strategy", default="HYBRID_3WAY")
    args = p.parse_args()

    return _retrieve_only(args) if args.retrieve_only else _via_api(args)


if __name__ == "__main__":
    sys.exit(main())
