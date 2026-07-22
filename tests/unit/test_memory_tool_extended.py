"""Extended tests for MemoryTool — read/write/summarize contracts."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tools.memory_tool import MemoryTool


@pytest.fixture
def mm():
    m = MagicMock()
    m.get_full_context.return_value = {"short_term": "abc", "long_term": "xyz"}
    m.summarize.return_value = "rolling summary"
    return m


def test_no_memory_manager_returns_unconfigured_message():
    tool = MemoryTool(memory_manager=None)
    out = tool._run("read")
    assert "not configured" in out


def test_read_without_key_returns_full_context(mm):
    tool = MemoryTool(memory_manager=mm, session_id="sess1")
    out = tool._run("read")
    assert "abc" in out
    mm.get_full_context.assert_called_once_with("sess1")


def test_read_with_key_returns_specific_value(mm):
    tool = MemoryTool(memory_manager=mm, session_id="sess1")
    out = tool._run("read", key="long_term")
    assert out == "xyz"


def test_read_with_unknown_key_returns_not_found(mm):
    tool = MemoryTool(memory_manager=mm, session_id="sess1")
    out = tool._run("read", key="missing")
    assert "not found" in out


def test_write_requires_key(mm):
    tool = MemoryTool(memory_manager=mm)
    out = tool._run("write", key="", value="data")
    assert "Error" in out
    mm.store.assert_not_called()


def test_write_persists_value(mm):
    tool = MemoryTool(memory_manager=mm, session_id="sess1")
    out = tool._run("write", key="k", value="v")
    mm.store.assert_called_once_with("sess1", "k", "v")
    assert "Stored" in out


def test_summarize_calls_manager(mm):
    tool = MemoryTool(memory_manager=mm, session_id="sess1")
    out = tool._run("summarize")
    mm.summarize.assert_called_once_with("sess1")
    assert out == "rolling summary"


def test_unknown_action_returns_help_message(mm):
    tool = MemoryTool(memory_manager=mm)
    out = tool._run("delete")
    assert "Unknown action" in out


def test_manager_exception_returns_error_string(mm):
    mm.get_full_context.side_effect = RuntimeError("redis down")
    tool = MemoryTool(memory_manager=mm)
    out = tool._run("read")
    assert "error" in out.lower()
    assert "redis down" in out
