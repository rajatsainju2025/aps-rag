"""
Long-term memory — persistent entity memory graph in Neo4j.
Stores: (:User)-[:DISCUSSED]->(:Topic) subgraph.
Enables cross-session entity tracking.
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Persistent entity memory backed by Neo4j.
    Tracks what topics each user/session has discussed.
    """

    def __init__(self, neo4j_client):
        self.neo4j = neo4j_client
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        """Create Neo4j constraints for memory graph."""
        try:
            self.neo4j.run_query(
                "CREATE CONSTRAINT user_id IF NOT EXISTS "
                "FOR (u:User) REQUIRE u.id IS UNIQUE"
            )
            self.neo4j.run_query(
                "CREATE CONSTRAINT topic_name IF NOT EXISTS "
                "FOR (t:MemoryTopic) REQUIRE t.name IS UNIQUE"
            )
        except Exception as e:
            logger.warning(f"Memory constraints setup: {e}")

    def store_discussion(
        self,
        session_id: str,
        entities: List[str],
        query: str,
        answer_summary: str,
    ) -> None:
        """
        Store a discussion in the entity memory graph.
        Creates User node and DISCUSSED edges to topic entities.
        """
        try:
            # Upsert User node
            self.neo4j.run_query(
                "MERGE (u:User {id: $session_id}) "
                "SET u.last_active = datetime()",
                {"session_id": session_id},
            )

            # Create topic nodes and edges
            for entity in entities:
                self.neo4j.run_query(
                    "MERGE (t:MemoryTopic {name: $name}) "
                    "WITH t "
                    "MATCH (u:User {id: $session_id}) "
                    "MERGE (u)-[r:DISCUSSED]->(t) "
                    "ON CREATE SET r.count = 1, r.last_query = $query "
                    "ON MATCH SET r.count = r.count + 1, r.last_query = $query",
                    {
                        "name": entity.lower(),
                        "session_id": session_id,
                        "query": query[:200],
                    },
                )
        except Exception as e:
            logger.warning(f"Long-term memory store failed: {e}")

    def get_user_topics(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get topics this user has previously discussed."""
        try:
            return self.neo4j.run_query(
                "MATCH (u:User {id: $session_id})-[r:DISCUSSED]->(t:MemoryTopic) "
                "RETURN t.name AS topic, r.count AS count, r.last_query AS last_query "
                "ORDER BY r.count DESC LIMIT $limit",
                {"session_id": session_id, "limit": limit},
            )
        except Exception as e:
            logger.warning(f"Long-term memory retrieval failed: {e}")
            return []

    def get_relevant_context(self, session_id: str, query: str) -> str:
        """Get relevant past context for a new query."""
        topics = self.get_user_topics(session_id, limit=5)
        if not topics:
            return ""
        topic_str = ", ".join([t["topic"] for t in topics])
        return f"User has previously discussed: {topic_str}"
