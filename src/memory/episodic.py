"""
Episodic memory — session summaries stored in Neo4j.
Enables "last time we discussed X" cross-session recall.
"""
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Session summary store using Neo4j.
    Each session gets a :Session node with summary and topic properties.
    """

    def __init__(self, neo4j_client):
        self.neo4j = neo4j_client

    def store_session_summary(
        self,
        session_id: str,
        summary: str,
        topics: List[str],
    ) -> None:
        """Store or update episodic summary for a session."""
        if not self.neo4j:
            return
        try:
            self.neo4j.store_episode(session_id, summary, topics)
            logger.debug(f"Episodic memory stored for session {session_id}")
        except Exception as e:
            logger.warning(f"Episodic memory store failed: {e}")

    def recall(self, session_id: str) -> Optional[str]:
        """Recall episodic summary for a session."""
        if not self.neo4j:
            return None
        try:
            episode = self.neo4j.get_episode(session_id)
            if episode:
                return episode.get("summary", "")
        except Exception as e:
            logger.warning(f"Episodic memory recall failed: {e}")
        return None

    def find_related_sessions(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Find sessions that discussed similar topics."""
        if not self.neo4j:
            return []
        try:
            cypher = """
            MATCH (s:Session)
            WHERE any(t IN $topics WHERE t IN s.topics)
            RETURN s.id AS session_id, s.last_summary AS summary,
                   s.topics AS topics, s.updated_at AS date
            ORDER BY s.updated_at DESC LIMIT 5
            """
            return self.neo4j.run_query(cypher, {"topics": topics})
        except Exception as e:
            logger.warning(f"Related sessions query failed: {e}")
            return []
