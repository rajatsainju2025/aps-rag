#!/usr/bin/env bash
# Install Python dependencies into the agentic_aps_retrieval conda environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="agentic_aps_retrieval"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
CONDA_PIP="${CONDA_BASE}/envs/${CONDA_ENV}/bin/pip"

echo "Installing Agentic GraphRAG dependencies into conda env: $CONDA_ENV"

if [ ! -f "$CONDA_PIP" ]; then
    echo "Conda env '$CONDA_ENV' not found — creating it (Python 3.11)..."
    conda create -y -n "$CONDA_ENV" python=3.11
fi

REQUIREMENTS="${PROJECT_ROOT}/requirements.txt"
if [ -f "${REQUIREMENTS}" ]; then
    "$CONDA_PIP" install --upgrade pip
    "$CONDA_PIP" install -r "${REQUIREMENTS}"
else
    echo "WARNING: requirements.txt not found at ${REQUIREMENTS}"
fi

echo ""
echo "Dependencies installed successfully in '$CONDA_ENV'."
echo "Key packages:"
"$CONDA_PIP" list 2>/dev/null | grep -E "langgraph|langchain-core|qdrant|neo4j|fastapi|pydantic" || true
echo ""
echo "To activate: conda activate $CONDA_ENV"
echo "To run API:  cd $PROJECT_ROOT && ./run_api.sh"
