"""
Memory manager — unified read/write interface across all 3 memory tiers.
"""
import logging
from typing import Dict, Any, List, Optional

from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified memory interface:
    - Short-term: 10-turn sliding window (in-session)
    - Long-term: Neo4j entity graph (cross-session persistent)
    - Episodic: Neo4j session summaries (cross-session summaries)
    """

    def __init__(self, neo4j_client=None):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(neo4j_client) if neo4j_client else None
        self.episodic = EpisodicMemory(neo4j_client) if neo4j_client else None
        # Per-session key/value metadata (in-memory only). Used by the clarification
        # gate to remember pending clarifications across turns.
        self._session_metadata: Dict[str, Dict[str, Any]] = {}

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Add a conversation turn to short-term memory."""
        self.short_term.add_turn(session_id, role, content)

    def set_session_state(self, session_id: str, key: str, value: Any) -> None:
        """Store a per-session piece of metadata (e.g. clarification_pending)."""
        self._session_metadata.setdefault(session_id, {})[key] = value

    def get_session_state(self, session_id: str, key: str, default: Any = None) -> Any:
        """Read per-session metadata; returns default when unset."""
        return self._session_metadata.get(session_id, {}).get(key, default)

    def get_short_term_context(self, session_id: str) -> str:
        """Get formatted short-term conversation history."""
        return self.short_term.get_formatted_context(session_id)

    def update_long_term(
        self,
        session_id: str,
        entities: List[str],
        query: str,
        answer_summary: str,
    ) -> None:
        """Update long-term entity memory after a successful interaction."""
        if self.long_term:
            self.long_term.store_discussion(session_id, entities, query, answer_summary)

    def get_long_term_context(self, session_id: str) -> str:
        """Get relevant long-term context for the session."""
        if not self.long_term:
            return ""
        return self.long_term.get_relevant_context(session_id, "")

    def store_episode(self, session_id: str, summary: str, topics: List[str]) -> None:
        """Store episodic session summary."""
        if self.episodic:
            self.episodic.store_session_summary(session_id, summary, topics)

    def recall_episode(self, session_id: str) -> Optional[str]:
        """Recall episodic memory for a session."""
        if not self.episodic:
            return None
        return self.episodic.recall(session_id)

    def get_full_context(self, session_id: str) -> Dict[str, Any]:
        """Get all available memory context for a session."""
        short = self.get_short_term_context(session_id)
        long = self.get_long_term_context(session_id)
        episodic = self.recall_episode(session_id)

        return {
            "short_term": short,
            "long_term": long,
            "episodic": episodic,
            "has_context": bool(short or long or episodic),
        }
