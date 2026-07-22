"""
LangChain Tool: RelationshipTraversalTool — KG path traversal via Neo4j.
"""
import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _TraversalInput(BaseModel):
    start_entity: str = Field(description="Starting entity name or ID")
    end_entity: Optional[str] = Field(default=None, description="Target entity for path finding")
    max_hops: int = Field(default=3, description="Maximum relationship hops")


class RelationshipTraversalTool(BaseTool):
    name: str = "relationship_traversal"
    description: str = "Traverse knowledge graph relationships between entities"
    args_schema: Type[BaseModel] = _TraversalInput
    neo4j_client: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        neo4j_client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "neo4j_client", neo4j_client)

    def _run(
        self,
        start_entity: str,
        end_entity: Optional[str] = None,
        max_hops: int = 3,
    ) -> str:
        try:
            if end_entity and self.neo4j_client:
                # Use Neo4j shortest path
                result = self.neo4j_client.shortest_path(start_entity, end_entity)
                return json.dumps({"path": result, "algorithm": "neo4j_shortest_path"}, default=str)

            if self.neo4j_client:
                # BFS neighbourhood
                result = self.neo4j_client.bfs_neighbors(start_entity, max_hops=max_hops)
                return json.dumps(
                    {"neighbors": result, "start": start_entity, "max_hops": max_hops},
                    default=str,
                )

            return json.dumps({"error": "No graph client configured", "results": []})
        except Exception as exc:
            logger.error("Relationship traversal failed: %s", exc)
            return json.dumps({"error": str(exc), "results": []})
