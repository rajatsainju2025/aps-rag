"""
Short-term memory — in-session sliding window conversation buffer.
Per-session isolation via session_id. Max 10 turns.
"""
import logging
from collections import deque
from typing import Dict, List, Any, Deque

logger = logging.getLogger(__name__)

MAX_TURNS = 10


class ShortTermMemory:
    """
    Session-scoped conversation buffer.
    Each session_id has its own independent sliding window.
    """

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        self._sessions: Dict[str, Deque[Dict[str, Any]]] = {}

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Add a turn (user or assistant) to the session buffer."""
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_turns * 2)  # *2 for user+assistant

        self._sessions[session_id].append({"role": role, "content": content})
        logger.debug(
            f"Short-term memory [{session_id}]: {len(self._sessions[session_id])} turns"
        )

    def get_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all stored turns for a session as message list."""
        return list(self._sessions.get(session_id, []))

    def get_formatted_context(self, session_id: str) -> str:
        """Format conversation history as a string for LLM context."""
        turns = self.get_context(session_id)
        if not turns:
            return ""
        lines = []
        for turn in turns:
            role = turn["role"].capitalize()
            content = turn["content"][:500]  # Truncate long turns
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        """Clear session buffer."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def session_count(self) -> int:
        return len(self._sessions)
