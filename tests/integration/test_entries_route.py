"""
Integration tests for /api/entries/{collection}/{entry_id}.html.

Uses FastAPI TestClient against an isolated FastAPI app — does NOT spin up
the full agentic-graph-rag api/main.py because that does heavy startup
(LLM clients, embeddings, Neo4j, Qdrant). The router is tested standalone
with the document cache populated by hand to avoid touching the data dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.routes import entries as entries_route  # noqa: E402


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(entries_route.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _seed_cache(monkeypatch):
    """Replace DocumentLoader.load_collection with deterministic fixtures."""
    fixtures = {
        "work_requests": [
            {
                "id": "13230",
                "_raw": {
                    "WRQ_NO": "13230",
                    "STATUS": "CLOSED",
                    "DATE_REQUESTED": "2024-02-18",
                    "REQUESTOR": "Roe, ALEX",
                    "MACHINE": "STORAGE_RING",
                    "PRIORITY": "HIGH",
                    "DESCRIPTION": "Cathode voltage exceeded nominal during ramp-up.",
                },
            },
        ],
        "teams_messages": [
            {
                "id": "teams_messages",
                "author": "Doe, Jane A.",
                "creation_datetime": "2024-02-18T14:30:00",
                "_raw": {
                    "channel": "RF Operations",
                    "content": "Anyone seeing oscillation on 1MV?",
                    "author": "Doe, Jane A.",
                    "creation_datetime": "2024-02-18T14:30:00",
                },
            },
        ],
        "sdds_collection": [
            {
                "id": "sddsplot_spec",
                "title": "sddsplot column reference",
                "_raw": {
                    "file_name": "sddsplot.json",
                    "title": "sddsplot column reference",
                    "content": "sddsplot accepts -columnNames=<x>,<y> ...",
                },
            },
        ],
    }

    entries_route._doc_cache.clear()
    for key, docs in fixtures.items():
        entries_route._doc_cache[key] = docs

    yield
    entries_route._doc_cache.clear()


def test_wrq_html_renders_description(client):
    resp = client.get("/api/entries/work_requests/13230.html")
    assert resp.status_code == 200
    body = resp.text
    assert "Cathode voltage exceeded nominal" in body
    assert "Roe, ALEX" in body
    assert "13230" in body
    assert resp.headers["content-type"].startswith("text/html")


def test_teams_html_renders_message_body(client):
    resp = client.get("/api/entries/teams_messages/Doe_Jane_A_2024-02-18T14.html")
    assert resp.status_code == 200
    body = resp.text
    assert "oscillation on 1MV" in body
    assert "Doe, Jane A." in body


def test_sdds_html_renders_spec(client):
    resp = client.get("/api/entries/sdds_collection/sddsplot_spec.html")
    assert resp.status_code == 200
    body = resp.text
    assert "sddsplot accepts" in body
    assert "sddsplot column reference" in body


def test_unknown_collection_404(client):
    resp = client.get("/api/entries/entry_logbook/9999.html")
    assert resp.status_code == 404


def test_missing_entry_404(client):
    resp = client.get("/api/entries/work_requests/00000.html")
    assert resp.status_code == 404


def test_html_escape_prevents_injection(client):
    """Render-time escaping must keep raw HTML in the body from being executed."""
    entries_route._doc_cache["work_requests"] = [
        {
            "id": "evil",
            "_raw": {
                "WRQ_NO": "evil",
                "STATUS": "OPEN",
                "DESCRIPTION": "<script>alert('xss')</script>",
                "REQUESTOR": "evil",
            },
        },
    ]
    resp = client.get("/api/entries/work_requests/evil.html")
    assert resp.status_code == 200
    assert "<script>alert" not in resp.text
    assert "&lt;script&gt;" in resp.text
