#!/usr/bin/env bash
# Manual full-or-incremental knowledge-graph refresh.
#
# Re-ingests vectors + BM25 from data/raw/, then (re)builds the Neo4j graph.
# Requires an LLM backend (config llm.*) and a running Neo4j (config graph.*).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PYTHON:-python}"

cd "${PROJECT_ROOT}"
[ -d "${PROJECT_ROOT}/.venv" ] && source "${PROJECT_ROOT}/.venv/bin/activate" || true

echo "Step 1/2: Ingest vectors + BM25 from data/raw/ ..."
"$PY" scripts/ingest.py

echo "Step 2/2: Build knowledge graph (Neo4j) ..."
"$PY" scripts/build_kg.py --full

echo "KG refresh complete."
