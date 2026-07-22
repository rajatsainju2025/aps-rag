"""MCP Server: Graph — Neo4j knowledge-graph search.

Provides one tool:
  - graph_search: Fulltext + entity search over the facility knowledge graph

The Neo4j driver is opened lazily on first call and reused for the lifetime
of the server subprocess (the connection pool stays warm). Credentials are
injected via env vars at server startup (see .mcp.json):
  NEO4J_URI=bolt://host:7687
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=...

Run: python -m mcp_servers.graph
"""
from __future__ import annotations

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_servers._common import add_project_root_to_path, env_with_default, err, mcp_tool_safe, ok

logger = logging.getLogger(__name__)
add_project_root_to_path()

mcp = FastMCP(
    "graph",
    instructions=(
        "the facility knowledge-graph search via Neo4j. Use for entity/relationship "
        "queries: 'What components are in sector 32?', 'Which fault entities "
        "share trigger conditions?', etc. Returns a list of nodes with "
        "properties + relationships."
    ),
)

# ---------------------------------------------------------------------------
# Lazy-loaded Neo4j client (one per server process)
# ---------------------------------------------------------------------------
_neo4j_client = None


def _get_neo4j_client():
    """Lazy-init the Neo4j driver. Returns None if NEO4J creds are missing or
    connection fails — the tool function then returns a structured error so
    the LLM sees an actionable message instead of a stack trace."""
    global _neo4j_client
    if _neo4j_client is not None:
        return _neo4j_client
    from src.integrations.neo4j_client import Neo4jClient
    uri = env_with_default("NEO4J_URI", "bolt://localhost:7687")
    user = env_with_default("NEO4J_USERNAME", "neo4j")
    password = env_with_default("NEO4J_PASSWORD", "")
    if not password:
        logger.warning("NEO4J_PASSWORD env var not set — graph_search will fail")
        return None
    client = Neo4jClient(uri=uri, user=user, password=password)
    try:
        client.connect()
    except Exception as exc:  # noqa: BLE001
        logger.error("Neo4j connect failed (%s): %s", uri, exc)
        return None
    _neo4j_client = client
    logger.info("Neo4j connection ready: %s", uri)
    return _neo4j_client


# ---------------------------------------------------------------------------
# graph_search
# ---------------------------------------------------------------------------
@mcp.tool()
@mcp_tool_safe
def graph_search(query: str, limit: int = 10) -> str:
    """Search the facility knowledge graph for entities matching the query.

    Tries fulltext search first; falls back to entity name search if no
    fulltext hits. Returns up to ``limit`` matching nodes with properties.

    Args:
      query: Natural-language search string (entity name, system, fault type).
      limit: Max number of results (default 10, max 50).

    Examples:
      graph_search("kicker magnet")
      graph_search("BPM signal anomaly")
      graph_search("sector 32 vacuum", limit=20)
    """
    client = _get_neo4j_client()
    if client is None:
        return err(
            "Neo4j unavailable (connection failed or credentials missing).",
            tool="graph_search",
            hint="Check NEO4J_URI / NEO4J_PASSWORD env in .mcp.json.",
        )
    limit = max(1, min(int(limit), 50))
    try:
        results = client.fulltext_search(query, limit=limit)
        if not results:
            results = client.entity_search(query, limit=limit) if hasattr(client, "entity_search") else []
    except Exception as exc:  # noqa: BLE001 — surface as structured error
        return err(
            f"Neo4j query failed: {exc}",
            tool="graph_search",
            query=query,
        )
    return ok(result=results, query=query, count=len(results))


if __name__ == "__main__":
    mcp.run()
