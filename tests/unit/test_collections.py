"""
Unit tests for the vectorstore collections registry.
Covers: COLLECTIONS structure, get_collection_config, naming conventions,
        path constants, COLLECTION_OPTIONS.
No live services required.
"""
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------

class TestCollectionsRegistry:
    def test_six_collections_defined(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        assert len(COLLECTIONS) == 6

    def test_expected_collection_keys(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        expected = {
            "entry_logbook", "teams_messages", "icms_collection",
            "work_requests", "sdds_collection", "oag_collection",
        }
        assert set(COLLECTIONS.keys()) == expected

    def test_each_collection_has_required_fields(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        required = {"qdrant_collection", "es_index", "data_path", "doc_format"}
        for key, cfg in COLLECTIONS.items():
            missing = required - set(cfg.keys())
            assert not missing, f"Collection '{key}' missing: {missing}"


# ---------------------------------------------------------------------------
# agrag_ naming convention
# ---------------------------------------------------------------------------

class TestNamingConvention:
    def test_qdrant_names_start_with_agrag(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        for key, cfg in COLLECTIONS.items():
            name = cfg["qdrant_collection"]
            assert name.startswith("agrag_"), (
                f"Collection '{key}' qdrant name '{name}' should start with 'agrag_'"
            )

    def test_es_indices_start_with_agrag(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        for key, cfg in COLLECTIONS.items():
            name = cfg["es_index"]
            assert name.startswith("agrag_"), (
                f"Collection '{key}' ES index '{name}' should start with 'agrag_'"
            )

    def test_qdrant_and_es_names_match(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        for key, cfg in COLLECTIONS.items():
            assert cfg["qdrant_collection"] == cfg["es_index"], (
                f"Collection '{key}': qdrant and ES names should match"
            )


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

class TestPathConstants:
    def test_ag_rag_root_is_project_root(self):
        from src.vectorstore.collection_registry import AG_RAG_ROOT
        assert AG_RAG_ROOT.name == "agentic-graph-rag"
        assert AG_RAG_ROOT.is_dir()

    def test_local_data_root_under_project(self):
        from src.vectorstore.collection_registry import LOCAL_DATA_ROOT, AG_RAG_ROOT
        assert LOCAL_DATA_ROOT == AG_RAG_ROOT / "0_raw_data"

    def test_entry_rag_app_path_is_sibling(self):
        from src.vectorstore.collection_registry import ENTRY_RAG_APP_PATH, AG_RAG_ROOT
        expected = str(AG_RAG_ROOT.parent / "entry_rag_app")
        assert ENTRY_RAG_APP_PATH == expected

    def test_data_paths_are_absolute(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        for key, cfg in COLLECTIONS.items():
            p = cfg["data_path"]
            assert p.startswith("/"), (
                f"Collection '{key}' data_path should be absolute, got: {p}"
            )


# ---------------------------------------------------------------------------
# doc_format values
# ---------------------------------------------------------------------------

class TestDocFormats:
    def test_only_valid_doc_formats(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        valid = {"logbook_entries", "list_file", "single_doc_files"}
        for key, cfg in COLLECTIONS.items():
            assert cfg["doc_format"] in valid, (
                f"Collection '{key}' has invalid doc_format: {cfg['doc_format']}"
            )

    def test_entry_logbook_is_logbook_entries(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        assert COLLECTIONS["entry_logbook"]["doc_format"] == "logbook_entries"

    def test_list_file_collections(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        for key in ("teams_messages", "work_requests"):
            assert COLLECTIONS[key]["doc_format"] == "list_file"

    def test_single_doc_collections(self):
        from src.vectorstore.collection_registry import COLLECTIONS
        for key in ("icms_collection", "sdds_collection", "oag_collection"):
            assert COLLECTIONS[key]["doc_format"] == "single_doc_files"


# ---------------------------------------------------------------------------
# get_collection_config
# ---------------------------------------------------------------------------

class TestGetCollectionConfig:
    def test_valid_key_returns_config(self):
        from src.vectorstore.collection_registry import get_collection_config
        cfg = get_collection_config("entry_logbook")
        assert cfg["qdrant_collection"] == "agrag_entry_logbook"

    def test_unknown_key_raises_key_error(self):
        from src.vectorstore.collection_registry import get_collection_config
        with pytest.raises(KeyError, match="Unknown collection"):
            get_collection_config("nonexistent_db")

    def test_all_databases_raises_value_error(self):
        from src.vectorstore.collection_registry import get_collection_config
        with pytest.raises(ValueError, match="meta-option"):
            get_collection_config("all_databases")


# ---------------------------------------------------------------------------
# COLLECTION_OPTIONS
# ---------------------------------------------------------------------------

class TestCollectionOptions:
    def test_includes_all_databases(self):
        from src.vectorstore.collection_registry import COLLECTION_OPTIONS
        assert "all_databases" in COLLECTION_OPTIONS

    def test_includes_all_six_collections(self):
        from src.vectorstore.collection_registry import COLLECTION_OPTIONS, COLLECTIONS
        for key in COLLECTIONS:
            assert key in COLLECTION_OPTIONS

    def test_length_is_seven(self):
        from src.vectorstore.collection_registry import COLLECTION_OPTIONS
        assert len(COLLECTION_OPTIONS) == 7  # 6 collections + all_databases
