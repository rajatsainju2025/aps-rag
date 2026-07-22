#!/bin/bash
# Start the APS-RAG FastAPI server.
#
# Config comes from config/config.yaml + .env (see config.example.yaml / .env.example).
# Override the port with API_PORT, or edit `api.port` in your config.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${API_PORT:-8000}"
PY="${PYTHON:-python}"

echo "Starting APS-RAG API on port ${PORT} ..."
exec "$PY" -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-level info
