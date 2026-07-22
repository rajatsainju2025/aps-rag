"""
Unit tests for the 3-tier memory system.
Covers: ShortTermMemory (sliding window, session isolation),
        LongTermMemory (Neo4j mocked), EpisodicMemory (Neo4j mocked),
        MemoryManager (unified interface, graceful degradation).
No live Neo4j required.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

class TestShortTermMemory:
    @pytest.fixture
    def stm(self):
        from src.memory.short_term import ShortTermMemory
        return ShortTermMemory(max_turns=3)

    def test_add_and_get_turn(self, stm):
        stm.add_turn("s1", "user", "Hello")
        turns = stm.get_context("s1")
        assert len(turns) == 1
        assert turns[0] == {"role": "user", "content": "Hello"}

    def test_session_isolation(self, stm):
        stm.add_turn("s1", "user", "Q1 in session 1")
        stm.add_turn("s2", "user", "Q1 in session 2")
        assert len(stm.get_context("s1")) == 1
        assert len(stm.get_context("s2")) == 1
        assert stm.get_context("s1")[0]["content"] != stm.get_context("s2")[0]["content"]

    def test_sliding_window_evicts_old_turns(self, stm):
        # max_turns=3, buffer = maxlen=6 (user+assistant pairs)
        for i in range(10):
            stm.add_turn("s1", "user", f"msg {i}")
        turns = stm.get_context("s1")
        assert len(turns) == 6  # 3 * 2
        assert turns[0]["content"] == "msg 4"  # oldest surviving

    def test_get_context_unknown_session(self, stm):
        assert stm.get_context("nonexistent") == []

    def test_get_formatted_context_empty(self, stm):
        assert stm.get_formatted_context("s1") == ""

    def test_get_formatted_context(self, stm):
        stm.add_turn("s1", "user", "What is the beam current?")
        stm.add_turn("s1", "assistant", "The beam current is 100mA.")
        formatted = stm.get_formatted_context("s1")
        assert "User: What is the beam current?" in formatted
        assert "Assistant: The beam current is 100mA." in formatted

    def test_formatted_context_truncates_long_turns(self, stm):
        stm.add_turn("s1", "user", "x" * 1000)
        formatted = stm.get_formatted_context("s1")
        # Content truncated to 500 chars
        assert len(formatted) < 600

    def test_clear_removes_session(self, stm):
        stm.add_turn("s1", "user", "msg")
        stm.clear("s1")
        assert stm.get_context("s1") == []

    def test_clear_nonexistent_session_no_error(self, stm):
        stm.clear("nonexistent")  # Should not raise

    def test_session_count(self, stm):
        assert stm.session_count() == 0
        stm.add_turn("s1", "user", "a")
        stm.add_turn("s2", "user", "b")
        assert stm.session_count() == 2


# ---------------------------------------------------------------------------
# LongTermMemory (mocked Neo4j)
# ---------------------------------------------------------------------------

class TestLongTermMemory:
    @pytest.fixture
    def mock_neo4j(self):
        client = MagicMock()
        client.run_query.return_value = []
        return client

    @pytest.fixture
    def ltm(self, mock_neo4j):
        from src.memory.long_term import LongTermMemory
        return LongTermMemory(neo4j_client=mock_neo4j)

    def test_ensure_constraints_called_on_init(self, mock_neo4j):
        from src.memory.long_term import LongTermMemory
        LongTermMemory(neo4j_client=mock_neo4j)
        # Should have called run_query at least twice (2 constraints)
        assert mock_neo4j.run_query.call_count >= 2

    def test_store_discussion_creates_user_and_topics(self, ltm, mock_neo4j):
        mock_neo4j.run_query.reset_mock()
        ltm.store_discussion("sess_1", ["rf cavity", "beam loss"], "What failed?", "RF fault")
        # 1 call for User + 2 calls for topics = 3
        assert mock_neo4j.run_query.call_count == 3

    def test_store_discussion_truncates_query(self, ltm, mock_neo4j):
        mock_neo4j.run_query.reset_mock()
        long_query = "x" * 500
        ltm.store_discussion("sess_1", ["topic"], long_query, "answer")
        # Check that the query param is truncated to 200
        topic_call = mock_neo4j.run_query.call_args_list[-1]
        assert len(topic_call[0][1]["query"]) <= 200

    def test_store_discussion_lowercases_entities(self, ltm, mock_neo4j):
        mock_neo4j.run_query.reset_mock()
        ltm.store_discussion("sess_1", ["RF Cavity"], "query", "answer")
        topic_call = mock_neo4j.run_query.call_args_list[-1]
        assert topic_call[0][1]["name"] == "rf cavity"

    def test_get_user_topics(self, ltm, mock_neo4j):
        mock_neo4j.run_query.return_value = [
            {"topic": "beam loss", "count": 5, "last_query": "What happened?"}
        ]
        topics = ltm.get_user_topics("sess_1")
        assert len(topics) == 1
        assert topics[0]["topic"] == "beam loss"

    def test_get_relevant_context_with_topics(self, ltm, mock_neo4j):
        mock_neo4j.run_query.return_value = [
            {"topic": "beam loss", "count": 3, "last_query": "q1"},
            {"topic": "rf cavity", "count": 2, "last_query": "q2"},
        ]
        ctx = ltm.get_relevant_context("sess_1", "new query")
        assert "beam loss" in ctx
        assert "rf cavity" in ctx

    def test_get_relevant_context_empty(self, ltm, mock_neo4j):
        mock_neo4j.run_query.return_value = []
        assert ltm.get_relevant_context("sess_1", "query") == ""

    def test_neo4j_error_handled_in_store(self, ltm, mock_neo4j):
        mock_neo4j.run_query.side_effect = RuntimeError("Neo4j down")
        # Should not raise
        ltm.store_discussion("sess_1", ["topic"], "q", "a")

    def test_neo4j_error_handled_in_get(self, ltm, mock_neo4j):
        mock_neo4j.run_query.side_effect = RuntimeError("Neo4j down")
        assert ltm.get_user_topics("sess_1") == []


# ---------------------------------------------------------------------------
# EpisodicMemory (mocked Neo4j)
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    @pytest.fixture
    def mock_neo4j(self):
        client = MagicMock()
        client.store_episode.return_value = None
        client.get_episode.return_value = {"summary": "Discussed beam dynamics"}
        client.run_query.return_value = []
        return client

    @pytest.fixture
    def episodic(self, mock_neo4j):
        from src.memory.episodic import EpisodicMemory
        return EpisodicMemory(neo4j_client=mock_neo4j)

    def test_store_session_summary(self, episodic, mock_neo4j):
        episodic.store_session_summary("s1", "Summary text", ["beam", "rf"])
        mock_neo4j.store_episode.assert_called_once_with("s1", "Summary text", ["beam", "rf"])

    def test_store_no_client(self):
        from src.memory.episodic import EpisodicMemory
        ep = EpisodicMemory(neo4j_client=None)
        ep.store_session_summary("s1", "text", [])  # Should not raise

    def test_recall(self, episodic, mock_neo4j):
        result = episodic.recall("s1")
        assert result == "Discussed beam dynamics"

    def test_recall_no_client(self):
        from src.memory.episodic import EpisodicMemory
        ep = EpisodicMemory(neo4j_client=None)
        assert ep.recall("s1") is None

    def test_recall_no_episode(self, episodic, mock_neo4j):
        mock_neo4j.get_episode.return_value = None
        assert episodic.recall("s1") is None

    def test_find_related_sessions(self, episodic, mock_neo4j):
        mock_neo4j.run_query.return_value = [
            {"session_id": "s2", "summary": "Related session", "topics": ["beam"]}
        ]
        results = episodic.find_related_sessions(["beam"])
        assert len(results) == 1

    def test_find_related_no_client(self):
        from src.memory.episodic import EpisodicMemory
        ep = EpisodicMemory(neo4j_client=None)
        assert ep.find_related_sessions(["beam"]) == []

    def test_error_handling_in_store(self, episodic, mock_neo4j):
        mock_neo4j.store_episode.side_effect = RuntimeError("Neo4j down")
        episodic.store_session_summary("s1", "text", [])  # Should not raise

    def test_error_handling_in_recall(self, episodic, mock_neo4j):
        mock_neo4j.get_episode.side_effect = RuntimeError("Neo4j down")
        assert episodic.recall("s1") is None


# ---------------------------------------------------------------------------
# MemoryManager (unified interface)
# ---------------------------------------------------------------------------

class TestMemoryManager:
    @pytest.fixture
    def mm_no_backends(self):
        from src.memory.memory_manager import MemoryManager
        return MemoryManager(neo4j_client=None)

    def test_short_term_always_available(self, mm_no_backends):
        mm_no_backends.add_turn("s1", "user", "hello")
        ctx = mm_no_backends.get_short_term_context("s1")
        assert "hello" in ctx

    def test_long_term_none_without_neo4j(self, mm_no_backends):
        assert mm_no_backends.long_term is None

    def test_episodic_none_without_neo4j(self, mm_no_backends):
        assert mm_no_backends.episodic is None

    def test_get_long_term_context_without_neo4j(self, mm_no_backends):
        assert mm_no_backends.get_long_term_context("s1") == ""

    def test_recall_episode_without_neo4j(self, mm_no_backends):
        assert mm_no_backends.recall_episode("s1") is None

    def test_update_long_term_noop_without_neo4j(self, mm_no_backends):
        # Should not raise
        mm_no_backends.update_long_term("s1", ["entity"], "query", "answer")

    def test_store_episode_noop_without_neo4j(self, mm_no_backends):
        mm_no_backends.store_episode("s1", "summary", ["topic"])

    def test_get_full_context_structure(self, mm_no_backends):
        mm_no_backends.add_turn("s1", "user", "test question")
        ctx = mm_no_backends.get_full_context("s1")
        assert "short_term" in ctx
        assert "long_term" in ctx
        assert "episodic" in ctx
        assert "has_context" in ctx
        assert ctx["has_context"] is True  # short_term has content

    def test_get_full_context_no_context(self, mm_no_backends):
        ctx = mm_no_backends.get_full_context("empty_session")
        assert ctx["has_context"] is False

    def test_with_mocked_neo4j(self):
        from src.memory.memory_manager import MemoryManager
        mock_neo4j = MagicMock()
        mock_neo4j.run_query.return_value = []
        mock_neo4j.get_episode.return_value = {"summary": "Prior session"}

        mm = MemoryManager(neo4j_client=mock_neo4j)
        assert mm.long_term is not None
        assert mm.episodic is not None
