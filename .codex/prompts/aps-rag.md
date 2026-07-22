# APS-RAG (Codex custom prompt — SKELETON)
#
# Install: copy this file to your Codex prompts dir so it becomes the `/aps-rag`
# command:
#     mkdir -p ~/.codex/prompts && cp .codex/prompts/aps-rag.md ~/.codex/prompts/
# Then in Codex:  /aps-rag <your question>
#
# CONFIGURE: set APS_RAG_API_URL (default http://localhost:8000) and, for the
# local-CLI option, run from your APS-RAG checkout.

Answer the following question by querying the APS-RAG knowledge base
(hybrid vector + BM25 + knowledge-graph retrieval with reranking and cited,
agentic generation over the ingested corpus):

QUESTION: $ARGUMENTS

Steps:
1. Query APS-RAG. Prefer the running API:
   curl -s -X POST "${APS_RAG_API_URL:-http://localhost:8000}/query" \
     -H 'Content-Type: application/json' \
     -d '{"query": "$ARGUMENTS", "service_mode": "agentic_rag",
          "collection": "all_databases", "top_k": 10,
          "retrieval_strategy": "HYBRID_3WAY"}'
   Or, from the APS-RAG checkout, use the CLI:
     python scripts/query.py "$ARGUMENTS"                 # full answer via API
     python scripts/query.py --retrieve-only "$ARGUMENTS" # local retrieval only
2. Return the synthesized answer, preserving its [source_N] citations.
3. Add a short "Sources" list (source_database + document id per citation).
4. If APS-RAG abstains or returns nothing, say so — do not fabricate an answer.

Notes: APS-RAG is read-only retrieval + generation. It must be set up and data
ingested first (see the repo README and scripts/ingest.py).
