"""
Unit tests for the granular-chunked-presence gate in icms_daily_pipeline.

Covers list_files_missing_from_gc: Qdrant-unreachable fallback, missing-collection
fallback, parent_id filtering, file_id stem fallback, and scroll pagination.
"""
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.vectorstore import icms_daily_pipeline as pipeline

pytestmark = pytest.mark.unit


def _write_json(path: Path, payload):
    """Write JSON payload to path. If payload is bytes, write raw bytes (malformed test)."""
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload))


def _make_collections_response(names):
    """Mimic qc.get_collections().collections — a list of objects with .name."""
    return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])


def _make_point(parent_id):
    """Mimic a Qdrant scroll point with a .payload dict."""
    return SimpleNamespace(payload={"parent_id": parent_id})


def test_returns_empty_when_qdrant_unreachable(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pipeline, "ICMS_DEST", tmp_path)
    caplog.set_level(logging.WARNING)

    with patch("qdrant_client.QdrantClient", side_effect=ConnectionError("boom")):
        result = pipeline.list_files_missing_from_gc(logging.getLogger("test"))

    assert result == []
    assert any("Qdrant unreachable" in r.message for r in caplog.records)


def test_returns_all_local_when_collection_missing(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pipeline, "ICMS_DEST", tmp_path)
    caplog.set_level(logging.INFO)

    _write_json(tmp_path / "DOC_111.json", {"file_id": "DOC_111"})
    _write_json(tmp_path / "DOC_222.json", {"file_id": "DOC_222"})

    mock_qc = MagicMock()
    mock_qc.get_collections.return_value = _make_collections_response([])

    with patch("qdrant_client.QdrantClient", return_value=mock_qc):
        result = pipeline.list_files_missing_from_gc(logging.getLogger("test"))

    assert [p.name for p in result] == ["DOC_111.json", "DOC_222.json"]
    assert any(
        "not found; treating all local files as missing" in r.message
        for r in caplog.records
    )


def test_excludes_files_present_as_parent_id(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pipeline, "ICMS_DEST", tmp_path)
    caplog.set_level(logging.INFO)

    _write_json(tmp_path / "DOC_111.json", {"file_id": "DOC_111"})
    _write_json(tmp_path / "DOC_222.json", {"file_id": "DOC_222"})

    mock_qc = MagicMock()
    mock_qc.get_collections.return_value = _make_collections_response(["agrag_gc_icms"])
    mock_qc.scroll.return_value = ([_make_point("DOC_111")], None)

    with patch("qdrant_client.QdrantClient", return_value=mock_qc):
        result = pipeline.list_files_missing_from_gc(logging.getLogger("test"))

    assert [p.name for p in result] == ["DOC_222.json"]


def test_falls_back_to_stem_when_no_file_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ICMS_DEST", tmp_path)

    _write_json(tmp_path / "DOC_222.json", {})
    _write_json(tmp_path / "DOC_333.json", b"\x00\x01not valid json")

    mock_qc = MagicMock()
    mock_qc.get_collections.return_value = _make_collections_response(["agrag_gc_icms"])
    mock_qc.scroll.return_value = ([_make_point("DOC_111")], None)

    with patch("qdrant_client.QdrantClient", return_value=mock_qc):
        result = pipeline.list_files_missing_from_gc(logging.getLogger("test"))

    assert [p.name for p in result] == ["DOC_222.json", "DOC_333.json"]


def test_paginates_scroll(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ICMS_DEST", tmp_path)

    _write_json(tmp_path / "DOC_111.json", {"file_id": "DOC_111"})
    _write_json(tmp_path / "DOC_222.json", {"file_id": "DOC_222"})
    _write_json(tmp_path / "DOC_333.json", {"file_id": "DOC_333"})

    mock_qc = MagicMock()
    mock_qc.get_collections.return_value = _make_collections_response(["agrag_gc_icms"])
    mock_qc.scroll.side_effect = [
        ([_make_point("DOC_111")], "next_token"),
        ([_make_point("DOC_222")], None),
    ]

    with patch("qdrant_client.QdrantClient", return_value=mock_qc):
        result = pipeline.list_files_missing_from_gc(logging.getLogger("test"))

    assert [p.name for p in result] == ["DOC_333.json"]
    assert mock_qc.scroll.call_count == 2
