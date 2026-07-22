"""
Granular chunking strategy — mirrors entry_rag_app's 3-stage pipeline.

Stages:
  1. Pre-chunking (JSONPreChunker style): split large content fields
     - ICMS: RecursiveCharacterTextSplitter on content (chunk_size=1000, overlap=200)
     - the system: split sub-entries into groups that fit within chunk_size
     - Teams/WR: one message/request per chunk (already atomic)
  2. Atomic chunking (AtomicChunkingStrategy style): build Document objects
     with full metadata, RecursiveCharacterTextSplitter for ICMS (max_chunk_size=4000)
  3. Return List[Document] ready for embedding + indexing

This produces ~100x more chunks than the default "atomic" (whole-doc) strategy,
enabling fine-grained retrieval at the cost of longer build times.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Defaults matching entry_rag_app config
DEFAULT_PRE_CHUNK_SIZE = 1000
DEFAULT_PRE_CHUNK_OVERLAP = 200
DEFAULT_MAX_CHUNK_SIZE = 4000
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class GranularChunker:
    """
    Recursively splits documents into fine-grained chunks,
    mirroring entry_rag_app's JSONPreChunker + AtomicChunkingStrategy pipeline.
    """

    def __init__(
        self,
        pre_chunk_size: int = DEFAULT_PRE_CHUNK_SIZE,
        pre_chunk_overlap: int = DEFAULT_PRE_CHUNK_OVERLAP,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.pre_chunk_size = pre_chunk_size
        self.pre_chunk_overlap = pre_chunk_overlap
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Stage-1 splitter (pre-chunking for very large content)
        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.pre_chunk_size,
            chunk_overlap=self.pre_chunk_overlap,
            length_function=len,
            separators=_SEPARATORS,
        )

        # Stage-2 splitter (atomic chunking)
        self.atomic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=_SEPARATORS,
        )

        self.stats: Dict[str, int] = {}

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_key: str,
    ) -> List[Document]:
        """
        Apply granular chunking to a list of normalized documents.

        Args:
            documents: list of dicts from DocumentLoader (id, content, title, ...)
            collection_key: one of the 6 collection keys

        Returns:
            List[Document] with page_content and metadata, ready for embedding.
        """
        handler = {
            "entry_logbook": self._chunk_entry,
            "icms_collection": self._chunk_icms,
            "teams_messages": self._chunk_teams,
            "work_requests": self._chunk_work_requests,
            "sdds_collection": self._chunk_sdds_oag,
            "oag_collection": self._chunk_sdds_oag,
        }.get(collection_key, self._chunk_generic)

        all_chunks: List[Document] = []
        for doc in documents:
            chunks = handler(doc, collection_key)
            all_chunks.extend(chunks)

        self.stats[collection_key] = len(all_chunks)
        logger.info(
            "GranularChunker: %s → %d docs → %d chunks",
            collection_key,
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Collection-specific chunking
    # ------------------------------------------------------------------

    def _chunk_entry(self, doc: Dict[str, Any], collection_key: str) -> List[Document]:
        """
        the logbook: build rich text from header + studies + sub-entries,
        then split with atomic_splitter if content exceeds max_chunk_size.
        """
        raw = doc.get("_raw", doc)
        content = doc.get("content", "")
        if not content or len(content) < MIN_CHUNK_SIZE:
            return []

        base_meta = self._base_metadata(doc, collection_key, "entry_entry")

        # Build enriched content (header + studies + sub-entries) like entry_rag_app
        parts: List[str] = []
        if raw.get("ENTRY_ID") or raw.get("entry_id"):
            parts.append(f"Entry ID: {raw.get('ENTRY_ID') or raw.get('entry_id')}")
        if raw.get("entry_creation_datetime"):
            parts.append(f"Date: {raw['entry_creation_datetime']}")
        if raw.get("machine/system"):
            parts.append(f"System: {raw['machine/system']}")
        if raw.get("created_by"):
            parts.append(f"Author: {raw['created_by']}")
        if raw.get("logbook"):
            parts.append(f"Logbook: {raw['logbook']}")
        if raw.get("entry_filename"):
            parts.append(f"Title: {raw['entry_filename']}")
        if raw.get("topic"):
            parts.append(f"Topic: {raw['topic']}")
        if raw.get("keywords"):
            parts.append("Keywords: " + ", ".join(map(str, raw["keywords"])))
        if raw.get("personnel_involved"):
            parts.append("Personnel: " + ", ".join(map(str, raw["personnel_involved"])))

        header = "\n".join(parts)

        # Studies sections
        body_parts: List[str] = []
        if raw.get("studies_plan") and str(raw["studies_plan"]).strip():
            body_parts.append("Studies Plan:\n" + str(raw["studies_plan"]).strip())
        if raw.get("studies_summary") and str(raw["studies_summary"]).strip():
            body_parts.append("Studies Summary:\n" + str(raw["studies_summary"]).strip())

        # Sub-entries
        sub_entries = raw.get("entry_sub_entries", [])
        if sub_entries and isinstance(sub_entries, list):
            sub_texts: List[str] = []
            for sub in sub_entries:
                if isinstance(sub, dict):
                    text = (sub.get("entry_text") or sub.get("body") or "").strip()
                    if text:
                        sub_texts.append(text)
            if sub_texts:
                body_parts.append("Entries:\n" + "\n\n".join(sub_texts))

        full_content = header + ("\n\n" if header else "") + "\n\n".join(body_parts)
        if not full_content.strip():
            full_content = content  # fallback to normalized content

        return self._split_and_wrap(full_content, base_meta)

    def _chunk_icms(self, doc: Dict[str, Any], collection_key: str) -> List[Document]:
        """
        ICMS: 2-stage splitting mirroring entry_rag_app's pipeline.
        Stage 1: pre_splitter (1000 chars) on raw content
        Stage 2: atomic_splitter (4000 chars) on each pre-chunk with header prepended
        """
        content = doc.get("content", "")
        if not content or len(content) < MIN_CHUNK_SIZE:
            return []

        raw = doc.get("_raw", doc)
        base_meta = self._base_metadata(doc, collection_key, "icms_chunk")

        # Header metadata like entry_rag_app
        file_id = raw.get("file_id", doc.get("id", ""))
        filename = raw.get("filename", doc.get("title", ""))
        author = raw.get("author", doc.get("author", ""))
        date = raw.get("creation_datetime", doc.get("creation_datetime", ""))

        header = (
            f"ICMS Document ID: {file_id}\n"
            f"Title: {filename}\n"
            f"Date: {date}\n"
            f"Author: {author}\n"
            f"---\n\n"
        )

        # Stage 1: pre-split long content (mirrors JSONPreChunker)
        if len(content) > self.pre_chunk_size:
            pre_chunks = self.pre_splitter.split_text(content)
        else:
            pre_chunks = [content]

        # Stage 2: atomic split each pre-chunk with header prepended
        all_docs: List[Document] = []
        global_idx = 0
        for pre_chunk in pre_chunks:
            full_text = header + pre_chunk
            atomic_chunks = self.atomic_splitter.split_text(full_text)
            for chunk in atomic_chunks:
                if len(chunk.strip()) < MIN_CHUNK_SIZE:
                    continue
                meta = {
                    **base_meta,
                    "chunk_index": global_idx,
                    "id": f"{base_meta.get('parent_id', '')}__chunk_{global_idx}",
                }
                all_docs.append(Document(page_content=chunk, metadata=meta))
                global_idx += 1

        # Backfill total_chunks
        for d in all_docs:
            d.metadata["total_chunks"] = len(all_docs)

        return all_docs

    def _chunk_teams(self, doc: Dict[str, Any], collection_key: str) -> List[Document]:
        """Teams: one message per chunk (already atomic), just wrap."""
        content = doc.get("content", "")
        if not content or len(content) < MIN_CHUNK_SIZE:
            return []
        meta = self._base_metadata(doc, collection_key, "teams_message")
        return self._split_and_wrap(content, meta)

    def _chunk_work_requests(self, doc: Dict[str, Any], collection_key: str) -> List[Document]:
        """Work requests: one request per chunk using all available fields.

        Many work requests have very short DESCRIPTION fields (<50 chars).
        When content is below MIN_CHUNK_SIZE we fall back to serializing all
        raw key:value pairs — mirrors entry_rag_app's AtomicChunkingStrategy.
        """
        content = doc.get("content", "")
        raw = doc.get("_raw", doc)

        # If DESCRIPTION is too short, build rich content from all raw fields
        if len(content) < MIN_CHUNK_SIZE:
            parts = []
            for k, v in raw.items():
                if v and str(v).strip() and not str(k).startswith("_"):
                    parts.append(f"{k}: {v}")
            content = "\n".join(parts)

        if not content or len(content) < MIN_CHUNK_SIZE:
            return []
        meta = self._base_metadata(doc, collection_key, "work_request")
        return self._split_and_wrap(content, meta)

    def _chunk_sdds_oag(self, doc: Dict[str, Any], collection_key: str) -> List[Document]:
        """SDDS/OAG: 2-stage splitting for large docs, single-pass for small."""
        content = doc.get("content", "")
        if not content or len(content) < MIN_CHUNK_SIZE:
            return []
        meta = self._base_metadata(doc, collection_key, "sdds_oag_chunk")

        # Stage 1: pre-split if content is large
        if len(content) > self.pre_chunk_size:
            pre_chunks = self.pre_splitter.split_text(content)
            all_docs: List[Document] = []
            global_idx = 0
            for pre_chunk in pre_chunks:
                atomic_chunks = self.atomic_splitter.split_text(pre_chunk)
                for chunk in atomic_chunks:
                    if len(chunk.strip()) < MIN_CHUNK_SIZE:
                        continue
                    m = {
                        **meta,
                        "chunk_index": global_idx,
                        "id": f"{meta.get('parent_id', '')}__chunk_{global_idx}",
                    }
                    all_docs.append(Document(page_content=chunk, metadata=m))
                    global_idx += 1
            for d in all_docs:
                d.metadata["total_chunks"] = len(all_docs)
            return all_docs

        return self._split_and_wrap(content, meta)

    def _chunk_generic(self, doc: Dict[str, Any], collection_key: str) -> List[Document]:
        """Fallback for unknown collections."""
        content = doc.get("content", "")
        if not content or len(content) < MIN_CHUNK_SIZE:
            return []
        meta = self._base_metadata(doc, collection_key, "generic_chunk")
        return self._split_and_wrap(content, meta)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_and_wrap(
        self, text: str, base_meta: Dict[str, Any]
    ) -> List[Document]:
        """Split text and wrap each piece as a Document with chunk_index metadata."""
        chunks = self.atomic_splitter.split_text(text)
        docs: List[Document] = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < MIN_CHUNK_SIZE:
                continue
            meta = {**base_meta, "chunk_index": i, "total_chunks": len(chunks)}
            # Stable chunk ID: parent_id + chunk_index
            meta["id"] = f"{base_meta.get('parent_id', '')}__chunk_{i}"
            docs.append(Document(page_content=chunk, metadata=meta))
        return docs

    def _base_metadata(
        self, doc: Dict[str, Any], collection_key: str, chunk_type: str
    ) -> Dict[str, Any]:
        """Build base metadata dict from a normalized document."""
        meta = {
            "parent_id": doc.get("id", ""),
            "collection": collection_key,
            # Folder-per-database identity — stamped on EVERY chunk so retrieval
            # and citations can attribute a chunk back to its source database.
            "source_database": doc.get("source_database", collection_key),
            "chunk_type": chunk_type,
            "title": doc.get("title", ""),
            "author": doc.get("author", ""),
            "creation_datetime": doc.get("creation_datetime", ""),
            "system": doc.get("system", ""),
            "source": doc.get("source") or doc.get("_source_file", ""),
        }
        # Preserve URL-relevant fields so citation links resolve to the
        # source document instead of a __chunk_N-suffixed fallback URL.
        if doc.get("file_id"):
            meta["file_id"] = doc["file_id"]
        if doc.get("file_url"):
            meta["file_url"] = doc["file_url"]
        # Carry any schema-declared metadata fields onto every chunk so they are
        # queryable/filterable downstream.
        for k, v in (doc.get("_schema_fields") or {}).items():
            if k not in meta and v not in (None, ""):
                meta[k] = v
        return meta

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)
