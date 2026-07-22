---
name: aps-rag
description: >-
  Query the APS-RAG knowledge base — hybrid (vector + BM25 + knowledge-graph)
  retrieval with reranking and cited, agentic answers over the ingested corpus.
  Use when the user asks a question that should be answered from the documents
  ingested into APS-RAG. Invoke as /aps-rag <question>.
allowed-tools: Bash, Read
---

# APS-RAG skill (SKELETON — customize for your deployment)

This skill lets Claude Code answer a question by querying a running **APS-RAG**
system (see the repo README for setup/ingestion). It's a skeleton: fill in the
`CONFIGURE` values below for your deployment.

## When to use
- The user asks a factual/operational question answerable from the ingested corpus.
- The user explicitly types `/aps-rag <question>`.

## When NOT to use
- General questions unrelated to the ingested documents.
- Anything requiring write access — APS-RAG is read-only retrieval + generation.

## CONFIGURE (edit these)
- `APS_RAG_API_URL` — base URL of your running APS-RAG API (default `http://localhost:8000`).
- `APS_RAG_HOME`    — path to your APS-RAG checkout (for the local-CLI option).
- Default `collection` (`all_databases`) and `service_mode` (`agentic_rag`).

## How to run

### Option A — running API (full agentic, cited answer) [preferred]
```bash
curl -s -X POST "${APS_RAG_API_URL:-http://localhost:8000}/query" \
  -H 'Content-Type: application/json' \
  -d '{
        "query": "<QUESTION>",
        "service_mode": "agentic_rag",
        "collection": "all_databases",
        "top_k": 10,
        "retrieval_strategy": "HYBRID_3WAY"
      }'
```
The response JSON has `answer`, `citations`/`sources`, `latency_ms`, and a
`reasoning_trace`.

### Option B — local CLI (from the APS-RAG checkout)
```bash
cd "${APS_RAG_HOME:-.}"
python scripts/query.py "<QUESTION>"                 # full answer via the API
python scripts/query.py --retrieve-only "<QUESTION>" # local retrieval only (no LLM)
```

## What to return
1. The synthesized answer, preserving its `[source_N]` citations.
2. A short "Sources" list mapping each cited source to its `source_database` +
   document id from the response.
3. If the system abstains or returns nothing, say so plainly — do not fabricate.

## Notes
- Requires APS-RAG to be set up and data ingested (`scripts/ingest.py`).
- Retrieval-only mode needs no LLM and is a good connectivity check.
