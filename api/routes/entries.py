"""
Entry-view router — /api/entries/{collection}/{entry_id}.html.

Renders a single entry from one of the in-memory collections (work_requests,
teams_messages, sdds_collection) as HTML, so citation links from those
collections become clickable. The entry_logbook / icms_collection / oag_collection
collections have their own external URLs (handled in citation_formatter._make_url).

The first request per collection populates a process-local cache via
DocumentLoader.load_collection(); subsequent requests are O(N) lookups in the
cached list. For the planned collection sizes (≤17k entries) this is trivial.
"""
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.utils.document_loader import DocumentLoader

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/entries", tags=["entries"])

# Collections this route is responsible for. The rest already have external
# URLs and don't need an in-process renderer.
SUPPORTED_COLLECTIONS = {"work_requests", "teams_messages", "sdds_collection"}

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)

_doc_cache: Dict[str, List[Dict[str, Any]]] = {}
_cache_lock = Lock()


def _load_collection_cached(collection: str) -> List[Dict[str, Any]]:
    """Return the cached collection, loading from disk on first access."""
    with _cache_lock:
        cached = _doc_cache.get(collection)
        if cached is not None:
            return cached
        loader = DocumentLoader()
        docs = loader.load_collection(collection, limit=0)
        _doc_cache[collection] = docs
        logger.info("entries route cached %d docs for %s", len(docs), collection)
        return docs


def _teams_synthetic_id(doc: Dict[str, Any]) -> str:
    """Author + truncated timestamp — mirrors citation_formatter._entry_route_id."""
    raw = doc.get("_raw") or {}
    author = doc.get("author") or (raw.get("author") if isinstance(raw, dict) else "") or "unknown"
    author_clean = author.replace(" ", "_").replace(",", "").replace(".", "")
    creation_dt = doc.get("creation_datetime") or (raw.get("creation_datetime") if isinstance(raw, dict) else "") or ""
    if creation_dt:
        dt_short = creation_dt[:13] if len(creation_dt) >= 13 else creation_dt[:10]
        return f"{author_clean}_{dt_short}"
    return author_clean


def _find_entry(collection: str, entry_id: str) -> Optional[Dict[str, Any]]:
    """Locate an entry by id across the common id-bearing fields."""
    docs = _load_collection_cached(collection)
    target = entry_id.strip()
    for doc in docs:
        if str(doc.get("id", "")) == target:
            return doc
        raw = doc.get("_raw") or {}
        if isinstance(raw, dict):
            if str(raw.get("WRQ_NO", "")) == target:
                return doc
            if str(raw.get("file_id", "")) == target:
                return doc
        if collection == "teams_messages" and _teams_synthetic_id(doc) == target:
            return doc
    return None


def _wrq_metadata(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pick the fields a Work Request reviewer expects to see at the top."""
    return [
        {"label": "WRQ #",       "value": str(raw.get("WRQ_NO", ""))},
        {"label": "Status",      "value": str(raw.get("STATUS", ""))},
        {"label": "Submitted",   "value": str(raw.get("SUBMISSION_DATE", ""))},
        {"label": "Requestor",   "value": str(raw.get("REQUESTOR", ""))},
        {"label": "Group",       "value": str(raw.get("GROUP", ""))},
        {"label": "Machine",     "value": str(raw.get("MACHINE", ""))},
        {"label": "Schedule",    "value": str(raw.get("SCHEDULE_PERIOD", ""))},
        {"label": "Duration",    "value": str(raw.get("ESTIMATED_DURATION", ""))},
        {"label": "Start",       "value": str(raw.get("REQUEST_START", ""))},
        {"label": "End",         "value": str(raw.get("REQUEST_END", ""))},
    ]


def _teams_metadata(doc: Dict[str, Any], raw: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"label": "Author", "value": str(doc.get("author", raw.get("author", "")))},
        {"label": "Sent",   "value": str(doc.get("creation_datetime", raw.get("creation_datetime", "")))},
        {"label": "Channel", "value": str(raw.get("channel", raw.get("thread_id", "")))},
    ]


def _sdds_metadata(doc: Dict[str, Any], raw: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"label": "Spec",  "value": str(doc.get("title", raw.get("title", "")))},
        {"label": "File",  "value": str(raw.get("file_name", ""))},
    ]


def _extract_body(doc: Dict[str, Any], raw: Dict[str, Any]) -> str:
    """Pick the most descriptive prose field available for the body block."""
    for key in ("DESCRIPTION", "description", "content", "entry_text", "text", "body"):
        val = raw.get(key) or doc.get(key)
        if val:
            return str(val)
    return ""


def _render(collection: str, entry_id: str, doc: Dict[str, Any]) -> str:
    raw = doc.get("_raw") or {}
    if collection == "work_requests":
        rows = _wrq_metadata(raw)
        title = f"Work Request {raw.get('WRQ_NO', entry_id)}"
    elif collection == "teams_messages":
        rows = _teams_metadata(doc, raw)
        title = f"Teams Message — {doc.get('author', 'unknown')}"
    else:  # sdds_collection
        rows = _sdds_metadata(doc, raw)
        title = f"SDDS Spec — {doc.get('title', entry_id)}"

    template = _jinja_env.get_template("entry_view.html")
    return template.render(
        collection=collection,
        entry_id=entry_id,
        title=title,
        metadata_rows=[r for r in rows if r["value"]],
        body=_extract_body(doc, raw),
    )


@router.get("/{collection}/{entry_id}.html", response_class=HTMLResponse)
async def get_entry_html(collection: str, entry_id: str) -> HTMLResponse:
    if collection not in SUPPORTED_COLLECTIONS:
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection}' has no in-process entry view",
        )
    doc = _find_entry(collection, entry_id)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"entry '{entry_id}' not found in '{collection}'",
        )
    return HTMLResponse(content=_render(collection, entry_id, doc))
