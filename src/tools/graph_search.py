"""
LangChain Tool: GraphSearchTool — Neo4j entity and full-text search.
"""
import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _GraphSearchInput(BaseModel):
    query: str = Field(description="Search query for the knowledge graph")


class GraphSearchTool(BaseTool):
    name: str = "graph_search"
    description: str = "Search Neo4j knowledge graph for entities and relationships"
    args_schema: Type[BaseModel] = _GraphSearchInput
    neo4j_client: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, neo4j_client: Any = None, **kwargs: Any) -> None:
        # Stage C: neo4j_client constructor arg is kept for backward compat but
        # ignored — the MCP `graph` server holds its own connection. api/main.py
        # still passes it; future cleanup can drop the wiring.
        super().__init__(**kwargs)
        object.__setattr__(self, "neo4j_client", neo4j_client)

    def _run(self, query: str) -> str:
        # Stage C: route through MCP `graph` server.
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync(
                "graph", "graph_search", {"query": query}
            )
        )
