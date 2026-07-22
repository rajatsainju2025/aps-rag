"""
Graph router — /graph endpoints.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/nodes")
async def get_nodes(
    limit: int = Query(default=50, ge=1, le=500),
    label: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    from api.main import components
    neo4j = components.get("neo4j_client")
    if not neo4j:
        raise HTTPException(status_code=503, detail="Neo4j not configured")
    try:
        cypher = f"MATCH (n{':' + label if label else ''}) RETURN n LIMIT {limit}"
        nodes = neo4j.run_query(cypher)
        return {"nodes": nodes, "count": len(nodes)}
    except Exception as exc:
        logger.error("Graph nodes error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/paths")
async def get_paths(
    start: str = Query(..., description="Start entity"),
    end: Optional[str] = Query(default=None, description="End entity"),
    max_hops: int = Query(default=3, ge=1, le=6),
) -> Dict[str, Any]:
    from api.main import components
    neo4j = components.get("neo4j_client")

    if not neo4j:
        raise HTTPException(status_code=503, detail="No graph client configured")

    try:
        if end:
            result = neo4j.shortest_path(start, end)
            return {"path": result, "algorithm": "neo4j_shortest_path"}

        result = neo4j.bfs_neighbors(start, max_hops=max_hops)
        return {"neighbors": result, "start": start, "max_hops": max_hops}
    except Exception as exc:
        logger.error("Graph paths error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
