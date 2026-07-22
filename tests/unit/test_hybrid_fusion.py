"""
Unit tests for HybridFusion — specifically _get_doc_id, which determines
how docs are deduplicated across retrievers.

Regression context: Teams chunks share doc_id="teams_messages" as a legacy
single-file-collection sentinel. If _get_doc_id prefers doc_id over the
chunk-unique id, every Teams hit collapses into one fused result. The
cascade must put chunk-id first; doc_id is the fallback.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.retrieval.hybrid_fusion import HybridFusion


def _make_fusion():
    return HybridFusion()


class TestGetDocIdCascade:
    def test_prefers_chunk_id_over_doc_id(self):
        """Chunk-unique id MUST win over semantic doc_id to keep distinct
        chunks distinct under RRF fusion."""
        fusion = _make_fusion()
        doc = Document(
            page_content="hello",
            metadata={
                "id": "teams_borland_abc__chunk_0",  # unique
                "doc_id": "teams_messages",          # shared sentinel
            },
        )
        assert fusion._get_doc_id(doc) == "teams_borland_abc__chunk_0"

    def test_teams_chunks_do_not_collapse(self):
        """Three Teams docs sharing doc_id='teams_messages' must remain three
        distinct fusion entries — regression test for the May-26 collapse bug."""
        fusion = _make_fusion()
        docs = [
            Document(
                page_content=f"message {i}",
                metadata={
                    "id": f"teams_author_{i}_hash__chunk_0",
                    "doc_id": "teams_messages",
                    "parent_id": "teams_messages",
                },
            )
            for i in range(3)
        ]
        ids = {fusion._get_doc_id(d) for d in docs}
        assert len(ids) == 3, (
            f"Teams chunks collapsed to {len(ids)} bucket(s) — fusion will "
            f"merge them into one result. Cascade in _get_doc_id is wrong."
        )

    def test_falls_back_to_doc_id_when_no_chunk_id(self):
        """If chunk id is absent, doc_id is the next-best stable key."""
        fusion = _make_fusion()
        doc = Document(
            page_content="hello",
            metadata={"doc_id": "10750"},
        )
        assert fusion._get_doc_id(doc) == "10750"

    def test_falls_back_to_file_id_and_parent_id(self):
        fusion = _make_fusion()
        d1 = Document(page_content="x", metadata={"file_id": "FID-1"})
        d2 = Document(page_content="y", metadata={"parent_id": "PID-1"})
        assert fusion._get_doc_id(d1) == "FID-1"
        assert fusion._get_doc_id(d2) == "PID-1"

    def test_content_hash_when_no_id(self):
        fusion = _make_fusion()
        doc = Document(page_content="some content", metadata={})
        result = fusion._get_doc_id(doc)
        assert result == str(hash("some content"))
