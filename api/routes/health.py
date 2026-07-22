"""
Health router — /health endpoints.
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

# PV-related tool names; /health/pv reports on these specifically.
_PV_TOOL_NAMES = ("channel_finder", "pv_read", "archiver_read", "pv_search")


@router.get("")
async def health() -> Dict[str, Any]:
    from api.main import components, settings
    checks: Dict[str, str] = {}

    # Check Qdrant
    try:
        qdrant = components.get("qdrant_client")
        if qdrant:
            qdrant.client.get_collections()
            checks["qdrant"] = "ok"
        else:
            checks["qdrant"] = "not_configured"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"

    # Check Neo4j
    try:
        neo4j = components.get("neo4j_client")
        if neo4j:
            if neo4j.health_check():
                checks["neo4j"] = "ok"
            else:
                checks["neo4j"] = "error: health_check returned False"
        else:
            checks["neo4j"] = "not_configured"
    except Exception as exc:
        checks["neo4j"] = f"error: {exc}"

    overall = "ok" if all(v in ("ok", "not_configured") for v in checks.values()) else "degraded"
    return {
        "status": overall,
        "services": checks,
        "version": settings.api.version,
    }


@router.get("/pv")
async def pv_health() -> Dict[str, Any]:
    """Per-PV-tool readiness. Independent from / so general health is
    unaffected by PV-specific outages."""
    from api.main import components
    tool_status = components.get("tool_status", {}) or {}
    pv_status = {name: tool_status.get(name, {"status": "unknown"})
                 for name in _PV_TOOL_NAMES}
    overall = "ok" if all(s.get("status") == "ok" for s in pv_status.values()) else "degraded"
    return {"status": overall, "tools": pv_status}
