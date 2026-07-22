"""
Collection registry — folder-per-database model.

The public release uses a **folder-per-database** convention: every subfolder of
`data/raw/<database_name>/` is treated as one source database ("collection").
The folder name IS the database identifier and is stamped onto every chunk as
`source_database` metadata.

Auto-discovery: adding a new database = create `data/raw/<name>/` and drop files
in. No code change is required — `discover_raw_databases()` finds it, the loader
ingests it, and retrieval enumerates it from the live stores.

Store naming: each database maps to a Qdrant collection and an Elasticsearch
index named `<prefix><database>`, where the prefix encodes the chunking strategy:
  - "whole-doc"          → prefix "agrag_"     (1 vector per record)
  - "granular-chunking"  → prefix "agrag_gc_"  (default, fine-grained chunks)

`BUILTIN_PROFILES` gives optional format hints for a few well-known on-disk JSON
shapes so the example dataset ingests with rich metadata; any folder NOT listed
there is ingested with the generic loader + generic chunker (nothing is dropped).
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
AG_RAG_ROOT = _THIS_FILE.parent.parent.parent          # project root
RAW_DATA_ROOT = AG_RAG_ROOT / "data" / "raw"           # user drops <db>/ folders here
SCHEMAS_ROOT = AG_RAG_ROOT / "schemas"                 # optional per-db schemas

# ---------------------------------------------------------------------------
# Store naming
# ---------------------------------------------------------------------------
DEFAULT_STRATEGY = "granular-chunking"
STRATEGY_PREFIXES = {
    "whole-doc": "agrag_",
    "granular-chunking": "agrag_gc_",
}


def sanitize_name(name: str) -> str:
    """Normalize a database/folder name into a safe store identifier.

    Lower-cases, replaces any run of non-alphanumeric chars with a single
    underscore, and trims leading/trailing underscores. Keeps Qdrant/ES names
    predictable regardless of how the user named the folder.
    """
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return s or "unnamed"


def get_collection_names(collection_key: str, strategy: str = DEFAULT_STRATEGY) -> Dict[str, str]:
    """Return the Qdrant collection + ES index names for a database + strategy.

    Works for ANY database name (built-in or user-added) — the name is derived
    purely from prefix + sanitized key, so no registry entry is required.
    """
    prefix = STRATEGY_PREFIXES.get(strategy, STRATEGY_PREFIXES[DEFAULT_STRATEGY])
    short = sanitize_name(collection_key)
    name = f"{prefix}{short}"
    return {"qdrant_collection": name, "es_index": name}


# ---------------------------------------------------------------------------
# Built-in format hints (OPTIONAL)
# ---------------------------------------------------------------------------
# These describe a few well-known on-disk JSON shapes so a bundled dataset can
# ingest with rich, structured metadata. A folder whose name is NOT listed here
# is still ingested — via the generic loader + generic chunker. `data_path`
# defaults to data/raw/<key>.
#
# doc_format values (see src/utils/document_loader.py):
#   "logbook_entries"  : each *.json file has {"entries": [...]}
#   "list_file"        : a single *.json file that IS a list of records
#   "single_doc_files" : a directory of *.json files, each one record
#   "generic"          : mixed files (*.json/*.txt/*.md), one record per file
# Empty by default: every database uses the generic loader + generic chunker,
# which handles .json/.txt/.md and preserves the chunking strategy. Add an entry
# here only if a folder has a specialized on-disk JSON shape you want parsed with
# a dedicated handler, e.g.:
#     "my_logbook": {"doc_format": "logbook_entries"},
BUILTIN_PROFILES: Dict[str, Dict[str, Any]] = {}

# Backward-compatible alias. Discovery is the source of truth at runtime; this
# dict only carries optional format hints for known folder names.
COLLECTIONS: Dict[str, Dict[str, Any]] = BUILTIN_PROFILES


def discover_raw_databases(raw_root: Path = RAW_DATA_ROOT) -> List[str]:
    """Return the names of every database folder under data/raw/.

    A database is any immediate subdirectory of `raw_root` that contains at
    least one file (recursively). Hidden folders and the .gitkeep placeholder
    are ignored. Returns a sorted list of folder names (the database ids).
    """
    root = Path(raw_root)
    if not root.exists():
        logger.warning("Raw data root does not exist: %s", root)
        return []
    dbs: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if any(p.is_file() for p in child.rglob("*")):
            dbs.append(child.name)
        else:
            logger.info("Skipping empty database folder: %s", child.name)
    return dbs


def get_collection_profile(db_name: str, raw_root: Path = RAW_DATA_ROOT) -> Dict[str, Any]:
    """Resolve a database name to a profile {data_path, doc_format, ...}.

    Uses a BUILTIN_PROFILES hint when the folder name matches a known shape;
    otherwise returns a generic profile pointing at data/raw/<db_name>.
    """
    profile = dict(BUILTIN_PROFILES.get(db_name, {}))
    profile.setdefault("doc_format", "generic")
    profile.setdefault("display_name", db_name)
    profile["data_path"] = str(Path(raw_root) / db_name)
    profile["collection_key"] = db_name
    return profile


def get_collection_config(collection_key: str) -> Dict[str, Any]:
    """Back-compat accessor returning a database profile (never raises for
    discovered databases)."""
    if collection_key == "all_databases":
        raise ValueError("'all_databases' is a meta-option — iterate discovered databases instead")
    return get_collection_profile(collection_key)


def collection_options(raw_root: Path = RAW_DATA_ROOT) -> List[str]:
    """Selectable options: every discovered database + the federated meta-option."""
    return discover_raw_databases(raw_root) + ["all_databases"]


# Static fallback for callers that import a constant; prefer collection_options().
COLLECTION_OPTIONS = ["all_databases"]


def list_all_strategy_collections(strategy: str = DEFAULT_STRATEGY) -> List[Dict[str, Any]]:
    """Return per-database configs (names adjusted for the strategy) for every
    discovered database. Used by build scripts that iterate all databases."""
    result = []
    for db in discover_raw_databases():
        names = get_collection_names(db, strategy)
        profile = get_collection_profile(db)
        result.append({**profile, **names, "collection_key": db})
    return result
