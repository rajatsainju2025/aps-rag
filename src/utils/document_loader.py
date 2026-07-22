"""
DocumentLoader — folder-per-database loader.

Loads documents from `data/raw/<database>/` and normalizes them into a common
record schema. The database (folder) name is stamped onto every record as
`source_database`, so it propagates to every chunk's metadata downstream.

Supported on-disk formats (nothing the original handled is dropped):
  - "generic" (default)     : a folder of mixed files, one record per file —
                              *.json (list / {"entries":[...]} / single dict),
                              *.txt, *.md/*.markdown. Other extensions are
                              skipped with a clear log message (never a crash).
  - "logbook_entries"       : each *.json file has {"entries": [...]}
  - "list_file"             : a single *.json file that IS a list of records
  - "single_doc_files"      : a directory of *.json files, each one record

A `schema` (optional, per-database — see schemas/README.md) can be passed to
`load_database`; when present its declared fields are extracted as structured
metadata. For the generic fallback (no schema) the auto-captured metadata is
embedded into each chunk so retrieval stays metadata-aware.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.vectorstore.collection_registry import (
    RAW_DATA_ROOT,
    get_collection_profile,
    discover_raw_databases,
)

logger = logging.getLogger(__name__)

# File extensions the generic loader reads directly.
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
_JSON_SUFFIXES = {".json"}
# Reserved filenames inside a database folder (not treated as documents).
_RESERVED_FILES = {"schema.yaml", "schema.yml", "schema.json", ".gitkeep", "README.md"}


def _extract_content(doc: Dict[str, Any]) -> str:
    """Return a single text field from a raw doc dict regardless of schema."""
    for key in ("content", "text", "body"):
        if doc.get(key):
            return str(doc[key])
    # Common summary/description shapes
    if doc.get("rag_summary"):
        parts = [str(doc["rag_summary"])]
        if doc.get("main_actions"):
            parts.append(str(doc["main_actions"]))
        return "\n".join(parts)
    for key in ("description", "DESCRIPTION", "summary"):
        if doc.get(key):
            return str(doc[key])
    # Fallback: join all string values (bounded)
    return " ".join(str(v) for v in doc.values() if isinstance(v, str))[:4000]


def _apply_schema(doc: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the schema-declared metadata fields from a raw record.

    Returns {field_name: value} for the fields declared under the schema's
    `fields:` mapping. A field may map to a different source key via
    `source:`; missing fields fall back to a declared `default:` or are
    omitted. Kept intentionally simple — see schemas/README.md.
    """
    if not schema:
        return {}
    out: Dict[str, Any] = {}
    fields = (schema or {}).get("fields", {}) or {}
    for name, spec in fields.items():
        spec = spec if isinstance(spec, dict) else {}
        source_key = spec.get("source", name)
        if source_key in doc and doc[source_key] not in (None, ""):
            out[name] = doc[source_key]
        elif "default" in spec:
            out[name] = spec["default"]
    return out


def _metadata_header(meta: Dict[str, Any]) -> str:
    """Render a compact metadata header to prepend to content so it is embedded
    (and BM25-searchable) alongside the body — the generic fallback's way of
    keeping retrieval metadata-aware without a user-authored schema."""
    lines = []
    for key in ("source_database", "title", "author", "creation_datetime", "source"):
        val = meta.get(key)
        if val:
            lines.append(f"{key.replace('_', ' ').title()}: {val}")
    for k, v in (meta.get("_schema_fields") or {}).items():
        if v:
            lines.append(f"{k.replace('_', ' ').title()}: {v}")
    return "\n".join(lines)


def _normalize(
    doc: Dict[str, Any],
    collection_key: str,
    source_file: str = "",
    source_database: str = "",
    schema: Optional[Dict[str, Any]] = None,
    embed_metadata: bool = False,
) -> Dict[str, Any]:
    """Normalize a raw doc dict into the common record schema.

    Always stamps `source_database`. When `embed_metadata` is True (generic
    fallback), a compact metadata header is prepended to the content so it is
    embedded into the resulting chunks; schema fields are also stored as
    structured metadata.
    """
    content = _extract_content(doc)

    source_stem = Path(source_file).stem if source_file else ""
    doc_id = (
        doc.get("id")
        or doc.get("doc_id")
        or doc.get("file_id")
        or source_stem
        or doc.get("filename")
        or hashlib.md5(content[:200].encode()).hexdigest()[:16]
    )

    schema_fields = _apply_schema(doc, schema)

    title = doc.get("title") or doc.get("filename") or doc.get("name") or source_stem or ""
    author = str(doc.get("author") or doc.get("created_by") or doc.get("REQUESTOR") or "")
    created = str(doc.get("creation_datetime") or doc.get("timestamp") or doc.get("date") or "")

    record: Dict[str, Any] = {
        "id": str(doc_id),
        "content": content,
        "title": str(title)[:200],
        "author": author,
        "creation_datetime": created,
        "system": str(doc.get("system") or ""),
        "collection": collection_key,
        "source_database": source_database or collection_key,
        "file_id": str(doc_id),
        "file_url": doc.get("file_url", ""),
        "source": source_file,
        "_source_file": source_file,
        "_schema_fields": schema_fields,
        "_raw": doc,
    }
    # Surface schema fields at top level so they also land in chunk metadata.
    for k, v in schema_fields.items():
        record.setdefault(k, v)

    if embed_metadata:
        header = _metadata_header(record)
        if header:
            record["content"] = f"{header}\n---\n{content}"

    return record


class DocumentLoader:
    """Loads and normalizes documents from data/raw/<database>/ folders."""

    def load_all(self, raw_root: Path = RAW_DATA_ROOT) -> Dict[str, List[Dict[str, Any]]]:
        """Discover and load every database under data/raw/. Returns a dict
        keyed by database name."""
        return {
            db: self.load_database(db, raw_root=raw_root)
            for db in discover_raw_databases(raw_root)
        }

    def load_database(
        self,
        db_name: str,
        schema: Optional[Dict[str, Any]] = None,
        raw_root: Path = RAW_DATA_ROOT,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """Load one database folder into normalized records.

        Args:
            db_name: the folder name under data/raw/ (the database id).
            schema: optional per-database schema. When None, the generic
                    fallback embeds auto-captured metadata into content.
            limit: cap the number of records returned (0 = all).
        """
        profile = get_collection_profile(db_name, raw_root=raw_root)
        data_path = Path(profile["data_path"])
        fmt = profile.get("doc_format", "generic")

        if not data_path.exists():
            logger.warning("Database path does not exist: %s", data_path)
            return []

        # Generic fallback embeds metadata into content; known structured
        # formats already carry rich fields, so we don't double-embed there.
        embed_metadata = fmt == "generic"

        if fmt == "list_file":
            docs = self._load_list_files(data_path, db_name, schema)
        elif fmt == "logbook_entries":
            docs = self._load_logbook_files(data_path, db_name, schema)
        elif fmt == "single_doc_files":
            docs = self._load_single_doc_files(data_path, db_name, schema)
        else:  # generic
            docs = self._load_generic_folder(data_path, db_name, schema, embed_metadata)

        if limit:
            docs = docs[:limit]
        logger.info("Loaded %d docs from database '%s'", len(docs), db_name)
        return docs

    # Back-compat alias (older callers used load_collection).
    def load_collection(self, collection_key: str, limit: int = 0) -> List[Dict[str, Any]]:
        return self.load_database(collection_key, limit=limit)

    # ------------------------------------------------------------------
    # Generic folder loader (default path)
    # ------------------------------------------------------------------
    def _load_generic_folder(
        self,
        folder: Path,
        db_name: str,
        schema: Optional[Dict[str, Any]],
        embed_metadata: bool,
    ) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name in _RESERVED_FILES or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            try:
                if suffix in _JSON_SUFFIXES:
                    docs.extend(self._records_from_json(path, db_name, schema, embed_metadata))
                elif suffix in _TEXT_SUFFIXES:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        docs.append(_normalize(
                            {"content": text, "filename": path.name},
                            db_name, str(path), db_name, schema, embed_metadata,
                        ))
                else:
                    logger.warning(
                        "Unsupported file type '%s' — skipping %s "
                        "(supported: .json/.txt/.md)", suffix or "<none>", path.name,
                    )
            except Exception as exc:  # noqa: BLE001 — never let one bad file crash ingest
                logger.warning("Failed to load %s: %s", path, exc)
        return docs

    def _records_from_json(
        self, path: Path, db_name: str, schema, embed_metadata: bool,
    ) -> List[Dict[str, Any]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            items = [x for x in raw if isinstance(x, dict)]
        elif isinstance(raw, dict) and isinstance(raw.get("entries"), list):
            items = [x for x in raw["entries"] if isinstance(x, dict)]
        elif isinstance(raw, dict):
            items = [raw]
        else:
            return []
        return [
            _normalize(item, db_name, str(path), db_name, schema, embed_metadata)
            for item in items
        ]

    # ------------------------------------------------------------------
    # Known structured formats (used by BUILTIN_PROFILES)
    # ------------------------------------------------------------------
    def _load_list_files(self, data_path: Path, db_name: str, schema) -> List[Dict[str, Any]]:
        files = [data_path] if data_path.is_file() else sorted(data_path.glob("*.json"))
        docs: List[Dict[str, Any]] = []
        for jf in files:
            try:
                raw = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", jf, exc)
                continue
            items = raw if isinstance(raw, list) else [raw]
            docs.extend(
                _normalize(it, db_name, str(jf), db_name, schema)
                for it in items if isinstance(it, dict)
            )
        return docs

    def _load_logbook_files(self, data_path: Path, db_name: str, schema) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for jf in sorted(data_path.glob("*.json")):
            try:
                raw = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", jf, exc)
                continue
            entries = raw.get("entries", []) if isinstance(raw, dict) else []
            docs.extend(
                _normalize(e, db_name, str(jf), db_name, schema)
                for e in entries if isinstance(e, dict)
            )
        return docs

    def _load_single_doc_files(self, data_path: Path, db_name: str, schema) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for jf in sorted(data_path.glob("**/*.json")):
            if jf.name in _RESERVED_FILES:
                continue
            try:
                raw = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", jf, exc)
                continue
            if isinstance(raw, dict):
                docs.append(_normalize(raw, db_name, str(jf), db_name, schema))
        return docs
