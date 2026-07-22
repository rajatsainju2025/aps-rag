"""Tests for src/agents/_tool_schema_render.py."""
import json
import re

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.agents._tool_schema_render import render_tool_schemas


class _SingleArgInput(BaseModel):
    query: str = Field(..., description="Search query")


class _MultiArgInput(BaseModel):
    pv_names: str = Field(..., description="Comma-separated PV names")
    start: str = Field(..., description="ISO8601 or relative time (1h, 24h)")


class _SingleArgTool(BaseTool):
    name: str = "single_tool"
    description: str = "A tool that takes one argument"
    args_schema: type = _SingleArgInput

    def _run(self, query: str) -> str:
        return query


class _MultiArgTool(BaseTool):
    name: str = "multi_tool"
    description: str = "A tool that takes two arguments"
    args_schema: type = _MultiArgInput

    def _run(self, pv_names: str, start: str) -> str:
        return f"{pv_names}/{start}"


def test_render_empty_returns_empty_string():
    assert render_tool_schemas({}) == ""


def test_render_single_arg_tool():
    out = render_tool_schemas({"single_tool": _SingleArgTool()})
    assert "## Tool: `single_tool`" in out
    assert "A tool that takes one argument" in out
    assert "TOOL_CALL: single_tool(<query>)" in out
    # Schema block must contain valid JSON for the args
    json_block = re.search(r"```json\n(.+?)\n```", out, re.DOTALL).group(1)
    parsed = json.loads(json_block)
    assert parsed["required"] == ["query"]
    assert "query" in parsed["properties"]


def test_render_multi_arg_tool_uses_pipe_separator():
    out = render_tool_schemas({"multi_tool": _MultiArgTool()})
    # Multi-arg tools use `||` to match the existing convention in
    # data_analysis and archiver_read.
    assert "TOOL_CALL: multi_tool(<pv_names>||<start>)" in out


def test_render_drops_pydantic_titles_from_schema():
    out = render_tool_schemas({"single_tool": _SingleArgTool()})
    json_block = re.search(r"```json\n(.+?)\n```", out, re.DOTALL).group(1)
    parsed = json.loads(json_block)
    # Pydantic emits "title" for each field by default; we strip them to
    # keep the prompt compact.
    assert "title" not in parsed
    for field in parsed["properties"].values():
        assert "title" not in field


def test_render_size_stays_compact_for_realistic_tool_count():
    """8 tools × ~250 chars/tool should land well under 3 KB."""
    tools = {f"tool_{i}": _SingleArgTool() for i in range(8)}
    out = render_tool_schemas(tools)
    assert len(out) < 3000, f"Schema block too large: {len(out)} chars"
