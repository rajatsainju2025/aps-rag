"""
Per-database schema loader (optional).

A schema lets a user declare which metadata fields to extract from a database's
records. It is OPTIONAL — with no schema, ingestion uses the generic fallback
(existing chunker + auto-captured metadata embedded into each chunk).

Resolution order (first match wins), for a database named ``<db>``:
  1. schemas/<db>.yaml        ← primary convention (recommended)
  2. schemas/<db>.yml
  3. schemas/<db>.json
  4. data/raw/<db>/schema.yaml
  5. data/raw/<db>/schema.yml
  6. data/raw/<db>/schema.json

Schema shape (see schemas/README.md for the annotated example):

    database: my_database          # optional, for documentation
    description: "..."             # optional
    fields:
      title:                       # metadata field name (as stored on chunks)
        source: headline           # OPTIONAL: raw key to read from (default: field name)
        type: string               # OPTIONAL: string|integer|float|date|list|bool
        description: "..."         # OPTIONAL
        default: ""                # OPTIONAL: value when the raw key is absent
      author:
        type: string

On a malformed schema this raises SchemaError naming the file and the problem.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.vectorstore.collection_registry import RAW_DATA_ROOT, SCHEMAS_ROOT

logger = logging.getLogger(__name__)

_VALID_TYPES = {"string", "str", "integer", "int", "float", "number", "date", "datetime", "list", "array", "bool", "boolean"}


class SchemaError(ValueError):
    """Raised when a per-database schema file is present but malformed."""


def _candidate_paths(db_name: str, schemas_root: Path, raw_root: Path) -> List[Path]:
    return [
        schemas_root / f"{db_name}.yaml",
        schemas_root / f"{db_name}.yml",
        schemas_root / f"{db_name}.json",
        raw_root / db_name / "schema.yaml",
        raw_root / db_name / "schema.yml",
        raw_root / db_name / "schema.json",
    ]


def find_schema_path(
    db_name: str,
    schemas_root: Path = SCHEMAS_ROOT,
    raw_root: Path = RAW_DATA_ROOT,
) -> Optional[Path]:
    """Return the first existing schema path for a database, or None."""
    for p in _candidate_paths(db_name, schemas_root, raw_root):
        if p.exists():
            return p
    return None


def _parse(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text) or {}
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise SchemaError(f"Could not parse schema file '{path}': {exc}") from exc


def validate_schema(schema: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Validate a parsed schema; raise SchemaError with an actionable message."""
    if not isinstance(schema, dict):
        raise SchemaError(f"Schema '{path}' must be a mapping at the top level.")
    fields = schema.get("fields")
    if fields is None:
        raise SchemaError(f"Schema '{path}' must declare a 'fields:' mapping.")
    if not isinstance(fields, dict) or not fields:
        raise SchemaError(f"Schema '{path}': 'fields' must be a non-empty mapping of field-name -> spec.")
    for name, spec in fields.items():
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise SchemaError(
                f"Schema '{path}': field '{name}' must be a mapping "
                f"(e.g. `{name}:\\n    type: string`), got {type(spec).__name__}."
            )
        ftype = spec.get("type")
        if ftype is not None and str(ftype).lower() not in _VALID_TYPES:
            raise SchemaError(
                f"Schema '{path}': field '{name}' has unknown type '{ftype}'. "
                f"Valid types: {sorted(_VALID_TYPES)}."
            )
    return schema


def load_schema(
    db_name: str,
    schemas_root: Path = SCHEMAS_ROOT,
    raw_root: Path = RAW_DATA_ROOT,
) -> Optional[Dict[str, Any]]:
    """Load + validate the schema for a database, or return None if none exists.

    Raises SchemaError if a schema file exists but is malformed (fail-fast with
    a clear message rather than silently ingesting with wrong metadata).
    """
    path = find_schema_path(db_name, schemas_root, raw_root)
    if path is None:
        logger.info("No schema for database '%s' — using generic fallback", db_name)
        return None
    schema = validate_schema(_parse(path), path)
    logger.info("Loaded schema for '%s' from %s (%d fields)", db_name, path, len(schema.get("fields", {})))
    return schema
