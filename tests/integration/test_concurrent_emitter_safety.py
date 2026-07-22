"""Concurrency safety tests for `event_emitter`'s ContextVar approach.

The R5 streaming design relies on `contextvars.ContextVar` so that each
in-flight request has its own emitter, even when many requests share the
same orchestrator/services in parallel threads. These tests verify:

- Two concurrent ContextVar bindings stay isolated
- ContextVar resets cleanly after a request
- QueueEventEmitter.emit is safe to call from a worker thread
- get_emitter() returns NullEventEmitter when nothing is bound
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.event_emitter import (
    NullEventEmitter,
    QueueEventEmitter,
    current_event_emitter,
    get_emitter,
)


def test_get_emitter_returns_null_when_unbound():
    # Force a fresh ContextVar state
    try:
        current_event_emitter.set(None)
    except LookupError:
        pass
    em = get_emitter()
    assert isinstance(em, NullEventEmitter)


def test_two_concurrent_contexts_stay_isolated():
    """Two separate `contextvars.copy_context()` runs must not see each
    other's bindings — proves per-request isolation."""
    import contextvars

    captured = {}

    def _bind_and_check(label: str, value: str):
        token = current_event_emitter.set(value)  # type: ignore[arg-type]
        captured[label] = current_event_emitter.get()
        current_event_emitter.reset(token)

    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()
    ctx_a.run(_bind_and_check, "a", "EMITTER_A")
    ctx_b.run(_bind_and_check, "b", "EMITTER_B")

    assert captured["a"] == "EMITTER_A"
    assert captured["b"] == "EMITTER_B"


def test_contextvar_resets_after_request():
    """After `reset(token)`, the parent context's binding must be restored."""
    try:
        current_event_emitter.set(None)
    except LookupError:
        pass

    sentinel = object()
    token = current_event_emitter.set(sentinel)  # type: ignore[arg-type]
    assert current_event_emitter.get() is sentinel
    current_event_emitter.reset(token)
    # Reset returns us to the prior value (None per the var default)
    assert current_event_emitter.get() is None


def test_queue_emitter_thread_safe_put():
    """QueueEventEmitter.emit must be safe to call from a worker thread.
    The emit uses loop.call_soon_threadsafe internally."""
    async def _runner():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        emitter = QueueEventEmitter(queue, loop)

        # Fire 50 emits from 5 worker threads
        def _worker(i: int):
            for j in range(10):
                emitter.emit("tool_call_start", {"tool": f"t{i}", "step": j})

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Drain the queue and count
        events = []
        while not queue.empty() or len(events) < 50:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=2.0)
                events.append(ev)
                if len(events) >= 50:
                    break
            except asyncio.TimeoutError:
                break

        assert len(events) == 50
        # Every event has the right type
        assert all(e["type"] == "tool_call_start" for e in events)

    asyncio.run(_runner())


def test_queue_emitter_after_loop_closed_does_not_raise():
    """If the event loop is gone (e.g. request cancelled), emit() must not crash."""
    async def _setup_emitter():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        return QueueEventEmitter(queue, loop)

    emitter = asyncio.run(_setup_emitter())
    # The loop has now closed — emit should fail-silent, not raise
    emitter.emit("post_close", {"k": "v"})
