"""Unit tests for GraphSearchTool — Neo4j entity + fulltext search wrapper."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tools.graph_search import GraphSearchTool


@pytest.fixture
def neo4j():
    n = MagicMock()
    n.fulltext_search.return_value = []
    n.entity_search.return_value = []
    return n


def test_no_neo4j_returns_error_payload():
    tool = GraphSearchTool(neo4j_client=None)
    out = json.loads(tool._run("query"))
    assert "error" in out
    assert out["results"] == []


def test_fulltext_results_returned_directly(neo4j):
    neo4j.fulltext_search.return_value = [{"name": "Linac", "type": "system"}]
    tool = GraphSearchTool(neo4j_client=neo4j)
    out = json.loads(tool._run("Linac"))
    assert len(out["results"]) == 1
    assert out["query"] == "Linac"
    # entity_search should NOT be called when fulltext returned hits
    neo4j.entity_search.assert_not_called()


def test_falls_back_to_entity_search_when_fulltext_empty(neo4j):
    neo4j.fulltext_search.return_value = []
    neo4j.entity_search.return_value = [{"name": "BPM", "type": "device"}]
    tool = GraphSearchTool(neo4j_client=neo4j)
    out = json.loads(tool._run("BPM"))
    assert len(out["results"]) == 1
    neo4j.entity_search.assert_called_once()


def test_neo4j_exception_returns_error_payload(neo4j):
    neo4j.fulltext_search.side_effect = RuntimeError("Bolt connection refused")
    tool = GraphSearchTool(neo4j_client=neo4j)
    out = json.loads(tool._run("anything"))
    assert "error" in out
    assert "refused" in out["error"]


def test_args_schema_contract():
    """The schema should expose a single required `query: str` field."""
    schema = GraphSearchTool().args_schema.model_json_schema()
    assert "query" in schema["properties"]
    assert schema["required"] == ["query"]


def test_run_passes_query_through(neo4j):
    tool = GraphSearchTool(neo4j_client=neo4j)
    tool._run("S-DCCT:CurrentM")
    args, kwargs = neo4j.fulltext_search.call_args
    assert "S-DCCT:CurrentM" in args
