"""Thread-safe event emitter for streaming tool / synthesis events to the API.

The orchestrator runs `service.query()` in a worker thread via
`asyncio.to_thread`, so events emitted deep inside `tool_executor.execute()`
need to cross back to the FastAPI event loop. We use:

  - A `contextvars.ContextVar` to give every node access to the current
    emitter without threading it through every function signature.
  - A `QueueEventEmitter` backed by `asyncio.Queue` whose `put_nowait` is
    scheduled on the loop via `loop.call_soon_threadsafe`.

Default emitter is `NullEventEmitter` — no-op, zero overhead for callers
(tests, batch eval, CLI) that don't need streaming.
"""
from __future__ import annotations

import asyncio
import contextvars
from typing import Any, Dict, Optional, Protocol

# ContextVar so deep call sites can find the active emitter without plumbing.
# Reset via `current_event_emitter.reset(token)` in a finally block.
current_event_emitter: "contextvars.ContextVar[Optional[EventEmitter]]" = (
    contextvars.ContextVar("current_event_emitter", default=None)
)


class EventEmitter(Protocol):
    def emit(self, event_type: str, payload: Dict[str, Any]) -> None: ...


class NullEventEmitter:
    """No-op emitter — default when no streaming consumer is attached."""

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:  # noqa: D401
        return None


class QueueEventEmitter:
    """Thread-safe emitter that hands events to an asyncio.Queue.

    The loop reference is captured at construction (must be the loop that
    owns the queue). Workers call `emit()` from any thread; the loop's
    `call_soon_threadsafe` schedules the put on the correct event loop.
    """

    def __init__(self, queue: "asyncio.Queue[Dict[str, Any]]", loop: asyncio.AbstractEventLoop):
        self._queue = queue
        self._loop = loop

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {"type": event_type, **payload}
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:
            # Loop closed mid-flight — drop the event silently. The query
            # will finish and the task wrapper will surface any real error.
            return


def get_emitter() -> EventEmitter:
    """Return the active emitter or a NullEventEmitter when nothing is bound."""
    em = current_event_emitter.get()
    return em if em is not None else NullEventEmitter()
