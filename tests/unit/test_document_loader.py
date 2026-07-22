"""
Unit tests for DocumentLoader and its helpers (_extract_content, _normalize).
Uses tmp_path fixtures with synthetic JSON data — no real data files required.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------

class TestExtractContent:
    def test_direct_content_field(self):
        from src.utils.document_loader import _extract_content
        doc = {"content": "RF cavity trip at sector 35"}
        assert _extract_content(doc, "icms_collection") == "RF cavity trip at sector 35"

    def test_oag_rag_summary(self):
        from src.utils.document_loader import _extract_content
        doc = {"rag_summary": "Tool analyzes beam data", "main_actions": "Plot, Export"}
        result = _extract_content(doc, "oag_collection")
        assert "Tool analyzes beam data" in result
        assert "Plot, Export" in result

    def test_work_request_description(self):
        from src.utils.document_loader import _extract_content
        doc = {"DESCRIPTION": "Link down on switch XY"}
        assert _extract_content(doc, "work_requests") == "Link down on switch XY"

    def test_logbook_topic_and_summary(self):
        from src.utils.document_loader import _extract_content
        doc = {"topic": "Beam current", "studies_summary": "Measured at 100mA"}
        result = _extract_content(doc, "entry_logbook")
        assert "Beam current" in result
        assert "Measured at 100mA" in result

    def test_logbook_with_sub_entries(self):
        from src.utils.document_loader import _extract_content
        doc = {
            "topic": "RF test",
            "entry_sub_entries": [{"body": "Sub-entry body text here"}],
        }
        result = _extract_content(doc, "entry_logbook")
        assert "RF test" in result
        assert "Sub-entry body text here" in result

    def test_fallback_joins_string_values(self):
        from src.utils.document_loader import _extract_content
        doc = {"field_a": "alpha", "field_b": "beta", "field_c": 42}
        result = _extract_content(doc, "unknown_collection")
        assert "alpha" in result
        assert "beta" in result
        assert "42" not in result  # int values skipped

    def test_fallback_truncates_at_2000(self):
        from src.utils.document_loader import _extract_content
        doc = {"big_field": "x" * 5000}
        result = _extract_content(doc, "unknown_collection")
        assert len(result) <= 2000


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_standard_schema_keys(self):
        from src.utils.document_loader import _normalize
        doc = {"content": "test", "file_id": "ABC123", "author": "Doe"}
        result = _normalize(doc, "icms_collection", "/some/path.json")
        required_keys = {"id", "content", "title", "author", "creation_datetime",
                         "system", "collection", "_source_file", "_raw"}
        assert required_keys.issubset(set(result.keys()))

    def test_id_from_file_id(self):
        from src.utils.document_loader import _normalize
        doc = {"content": "test", "file_id": "DOCU_12345"}
        result = _normalize(doc, "icms_collection")
        assert result["id"] == "DOCU_12345"

    def test_id_from_entry_id(self):
        from src.utils.document_loader import _normalize
        doc = {"content": "test", "ENTRY_ID": 9294}
        result = _normalize(doc, "entry_logbook")
        assert result["id"] == "9294"

    def test_id_from_wrq_no(self):
        from src.utils.document_loader import _normalize
        doc = {"DESCRIPTION": "fix switch", "WRQ_NO": 35789}
        result = _normalize(doc, "work_requests")
        assert result["id"] == "35789"

    def test_id_fallback_to_hash(self):
        from src.utils.document_loader import _normalize
        doc = {"content": "some content without id fields"}
        result = _normalize(doc, "icms_collection")
        assert result["id"]  # should be non-empty hash string

    def test_collection_field_set(self):
        from src.utils.document_loader import _normalize
        doc = {"content": "test"}
        result = _normalize(doc, "sdds_collection")
        assert result["collection"] == "sdds_collection"

    def test_raw_preserved(self):
        from src.utils.document_loader import _normalize
        doc = {"content": "test", "custom_field": "custom_value"}
        result = _normalize(doc, "oag_collection")
        assert result["_raw"]["custom_field"] == "custom_value"

    def test_author_from_requestor(self):
        from src.utils.document_loader import _normalize
        doc = {"DESCRIPTION": "test", "REQUESTOR": "Smith"}
        result = _normalize(doc, "work_requests")
        assert result["author"] == "Smith"


# ---------------------------------------------------------------------------
# DocumentLoader — list_file format
# ---------------------------------------------------------------------------

class TestDocumentLoaderListFile:
    def test_loads_list_file(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data = [
            {"content": "Message 1", "author": "Alice"},
            {"content": "Message 2", "author": "Bob"},
        ]
        json_path = tmp_path / "messages.json"
        json_path.write_text(json.dumps(data))

        # Patch COLLECTIONS to use our tmp_path
        mock_collections = {
            "test_list": {
                "data_path": str(json_path),
                "doc_format": "list_file",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_list")
        assert len(docs) == 2
        assert docs[0]["content"] == "Message 1"
        assert docs[1]["author"] == "Bob"

    def test_loads_dict_as_single_item(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data = {"content": "Single message"}
        json_path = tmp_path / "single.json"
        json_path.write_text(json.dumps(data))

        mock_collections = {
            "test_single": {
                "data_path": str(json_path),
                "doc_format": "list_file",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_single")
        assert len(docs) == 1

    def test_limit_parameter(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data = [{"content": f"Item {i}"} for i in range(20)]
        json_path = tmp_path / "many.json"
        json_path.write_text(json.dumps(data))

        mock_collections = {
            "test_limit": {
                "data_path": str(json_path),
                "doc_format": "list_file",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_limit", limit=5)
        assert len(docs) == 5


# ---------------------------------------------------------------------------
# DocumentLoader — logbook_entries format
# ---------------------------------------------------------------------------

class TestDocumentLoaderLogbook:
    def test_loads_logbook_entries(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data = {
            "subfolder_name": "2025-01-15",
            "total_entries": 2,
            "entries": [
                {"ENTRY_ID": 100, "topic": "RF test", "studies_summary": "Good results"},
                {"ENTRY_ID": 101, "topic": "Vacuum issue"},
            ],
        }
        data_dir = tmp_path / "logbook"
        data_dir.mkdir()
        (data_dir / "log_001.json").write_text(json.dumps(data))

        mock_collections = {
            "test_logbook": {
                "data_path": str(data_dir),
                "doc_format": "logbook_entries",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_logbook")
        assert len(docs) == 2
        assert docs[0]["id"] == "100"
        assert "RF test" in docs[0]["content"]

    def test_limit_across_multiple_files(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data_dir = tmp_path / "logbook2"
        data_dir.mkdir()

        for i in range(3):
            data = {"entries": [{"ENTRY_ID": j, "topic": f"Topic {j}"} for j in range(i * 10, i * 10 + 10)]}
            (data_dir / f"log_{i:03d}.json").write_text(json.dumps(data))

        mock_collections = {
            "test_logbook2": {
                "data_path": str(data_dir),
                "doc_format": "logbook_entries",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_logbook2", limit=5)
        assert len(docs) == 5

    def test_skips_non_dict_entries(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data = {"entries": [{"topic": "Valid"}, "not a dict", None, {"topic": "Also valid"}]}
        data_dir = tmp_path / "logbook3"
        data_dir.mkdir()
        (data_dir / "log.json").write_text(json.dumps(data))

        mock_collections = {
            "test_logbook3": {
                "data_path": str(data_dir),
                "doc_format": "logbook_entries",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_logbook3")
        assert len(docs) == 2


# ---------------------------------------------------------------------------
# DocumentLoader — single_doc_files format
# ---------------------------------------------------------------------------

class TestDocumentLoaderSingleDoc:
    def test_loads_single_doc_files(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data_dir = tmp_path / "single_docs"
        data_dir.mkdir()
        for i in range(3):
            (data_dir / f"doc_{i}.json").write_text(
                json.dumps({"file_id": f"DOC_{i}", "content": f"Document {i}"})
            )

        mock_collections = {
            "test_single_docs": {
                "data_path": str(data_dir),
                "doc_format": "single_doc_files",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_single_docs")
        assert len(docs) == 3

    def test_skips_conversion_summary(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data_dir = tmp_path / "with_summary"
        data_dir.mkdir()
        (data_dir / "doc_1.json").write_text(json.dumps({"file_id": "D1", "content": "Real doc"}))
        (data_dir / "conversion_summary.json").write_text(json.dumps({"total": 100}))

        mock_collections = {
            "test_skip": {
                "data_path": str(data_dir),
                "doc_format": "single_doc_files",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_skip")
        assert len(docs) == 1
        assert docs[0]["id"] == "D1"

    def test_handles_invalid_json(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        data_dir = tmp_path / "bad_json"
        data_dir.mkdir()
        (data_dir / "good.json").write_text(json.dumps({"file_id": "G1", "content": "OK"}))
        (data_dir / "bad.json").write_text("NOT JSON {{{")

        mock_collections = {
            "test_bad": {
                "data_path": str(data_dir),
                "doc_format": "single_doc_files",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_bad")
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestDocumentLoaderEdgeCases:
    def test_unknown_collection_raises(self):
        from src.utils.document_loader import DocumentLoader
        loader = DocumentLoader()
        with pytest.raises(KeyError, match="Unknown collection"):
            loader.load_collection("nonexistent")

    def test_missing_data_path_returns_empty(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        mock_collections = {
            "test_missing": {
                "data_path": str(tmp_path / "does_not_exist"),
                "doc_format": "list_file",
            }
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            docs = loader.load_collection("test_missing")
        assert docs == []

    def test_load_all_calls_each_collection(self, tmp_path):
        from src.utils.document_loader import DocumentLoader
        # Create minimal data for 2 collections
        f1 = tmp_path / "coll1.json"
        f1.write_text(json.dumps([{"content": "A"}]))
        f2 = tmp_path / "coll2.json"
        f2.write_text(json.dumps([{"content": "B"}]))

        mock_collections = {
            "c1": {"data_path": str(f1), "doc_format": "list_file"},
            "c2": {"data_path": str(f2), "doc_format": "list_file"},
        }
        with patch("src.utils.document_loader.COLLECTIONS", mock_collections):
            loader = DocumentLoader()
            result = loader.load_all()
        assert set(result.keys()) == {"c1", "c2"}
        assert len(result["c1"]) == 1
        assert len(result["c2"]) == 1
