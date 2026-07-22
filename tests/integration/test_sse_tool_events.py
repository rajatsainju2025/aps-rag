"""Real-time SSE tool-event streaming (R5).

Verifies that `_stream_with_events` yields tool_call events as the worker
thread emits them — not after the worker thread completes.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.routes.pipelines import _stream_with_events
from src.agents.event_emitter import QueueEventEmitter


def _parse_sse_chunk(chunk: str) -> List[Dict[str, Any]]:
    """Parse a raw SSE chunk into a list of message dicts.

    A single chunk can contain multiple `data: ...\\n\\n` messages
    (e.g. `_sse_done` emits the final delta + `[DONE]` together).
    """
    out: List[Dict[str, Any]] = []
    for raw in chunk.split("\n\n"):
        raw = raw.strip()
        if not raw or not raw.startswith("data: "):
            continue
        body = raw[len("data: "):]
        if body == "[DONE]":
            out.append({"_done": True})
            continue
        out.append(json.loads(body))
    return out


def _delta_content(parsed: Dict[str, Any]) -> str:
    try:
        return parsed["choices"][0]["delta"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _flatten(chunks: List[str]) -> str:
    out = ""
    for c in chunks:
        for p in _parse_sse_chunk(c):
            out += _delta_content(p)
    return out


async def _test_tool_events_arrive_before_final_answer():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    emitter = QueueEventEmitter(queue, loop)

    # Track timing: when did the first tool event hit the stream consumer?
    first_event_at: List[float] = []
    completion_at: List[float] = []

    def _slow_query():
        # Emit two tool events early, then "synthesize" for 0.3 s, then return.
        emitter.emit("tool_call_start", {"tool": "channel_finder", "args": "beam current"})
        time.sleep(0.05)
        emitter.emit("tool_call_result", {"tool": "channel_finder", "duration_ms": 50})
        time.sleep(0.3)  # synthesis time
        completion_at.append(time.monotonic())
        return {
            "answer": "The beam current is 100.2 mA.",
            "reasoning_trace": ["fake trace step"],
            "openwebui_citations": [],
        }

    query_task = asyncio.create_task(asyncio.to_thread(_slow_query))

    chunks: List[str] = []
    async for chunk in _stream_with_events(
        pipeline_id="agentic-rag",
        pipeline_name="Agentic-RAG",
        completion_id="cmpl-test",
        query_task=query_task,
        event_queue=queue,
    ):
        if not first_event_at:
            # Record the time when the first chunk reaches the consumer
            first_event_at.append(time.monotonic())
        chunks.append(chunk)

    # ── Assertion 1: at least one chunk arrived BEFORE the slow synthesis finished
    assert first_event_at, "stream produced no chunks"
    assert completion_at, "query task never recorded completion"
    assert first_event_at[0] < completion_at[0], (
        f"First chunk at {first_event_at[0]:.3f}s arrived AFTER query completion at "
        f"{completion_at[0]:.3f}s — events were not streamed in real time"
    )

    # ── Assertion 2: tool_call_start text and the final answer both made it through
    flat = _flatten(chunks)
    assert "channel_finder" in flat
    assert "beam current is 100.2 mA" in flat


async def _test_stream_handles_query_with_no_events():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _no_tools_query():
        return {
            "answer": "No tools needed for this answer.",
            "reasoning_trace": [],
            "openwebui_citations": [],
        }

    query_task = asyncio.create_task(asyncio.to_thread(_no_tools_query))

    chunks: List[str] = []
    async for chunk in _stream_with_events(
        pipeline_id="naive-rag",
        pipeline_name="Naive-RAG",
        completion_id="cmpl-test2",
        query_task=query_task,
        event_queue=queue,
    ):
        chunks.append(chunk)

    flat = _flatten(chunks)
    assert "Thinking..." in flat
    assert "No tools needed" in flat
    # Should end with [DONE]
    assert any("[DONE]" in c for c in chunks)


async def _test_stream_surfaces_query_exception_as_error_message():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _broken_query():
        raise RuntimeError("simulated failure")

    query_task = asyncio.create_task(asyncio.to_thread(_broken_query))

    chunks: List[str] = []
    async for chunk in _stream_with_events(
        pipeline_id="agentic-rag",
        pipeline_name="Agentic-RAG",
        completion_id="cmpl-test3",
        query_task=query_task,
        event_queue=queue,
    ):
        chunks.append(chunk)

    flat = _flatten(chunks)
    assert "Pipeline error" in flat
    assert "simulated failure" in flat


# ── pytest entry points (sync wrappers around the async tests) ────────────────
def test_tool_events_arrive_before_final_answer():
    asyncio.run(_test_tool_events_arrive_before_final_answer())


def test_stream_handles_query_with_no_events():
    asyncio.run(_test_stream_handles_query_with_no_events())


def test_stream_surfaces_query_exception_as_error_message():
    asyncio.run(_test_stream_surfaces_query_exception_as_error_message())
