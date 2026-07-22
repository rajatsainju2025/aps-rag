"""
title: APS-RAG Phase Status
author: agentic-graph-rag
description: Surface pipeline phase events (Planning, Vector search, Reranking, …) on the inline status indicator (the rotating wheel) instead of leaving it stuck on "Thinking…". Stops updating once the answer body starts.
version: 1.1
license: MIT
"""
# The agentic-graph-rag backend emits phase events as plain `<think>`-internal
# content lines like "Planning...", "Vector search...", "Reranking...",
# "Synthesizing answer...". OpenWebUI renders them inside the collapsible
# reasoning dropdown but the rotating-wheel indicator above the chat bubble
# stays on the hardcoded "Thinking..." until completion.
#
# This Filter intercepts each streaming chunk, detects phase markers, and
# fires a `status` event so the inline indicator updates in real time.
#
# Once `</think>` arrives (= the reasoning block has closed and the answer
# body is about to stream), we stop emitting — otherwise any incidental
# mention of "Planning" or "Vector search" inside the answer prose would
# spuriously bump the indicator while the user is reading the response.

from collections import deque

from pydantic import BaseModel


# Match longer strings first so e.g. "Lexical (BM25) search" beats "search".
_PHASES = (
    "Synthesizing answer",
    "Lexical (BM25) search",
    "Vector search",
    "Graph search",
    "Reranking",
    "Planning",
    "Thinking",
)

# Bound the per-completion-id set so a long-lived OpenWebUI process doesn't
# accumulate forever. 1000 chats is comfortably larger than any single user
# session's working set.
_MAX_DONE = 1000


class Filter:
    class Valves(BaseModel):
        priority: int = 0

    def __init__(self):
        self.valves = self.Valves()
        self._done_ids: set = set()
        self._done_order: deque = deque()

    def _mark_done(self, cid: str) -> None:
        if cid in self._done_ids:
            return
        self._done_ids.add(cid)
        self._done_order.append(cid)
        while len(self._done_order) > _MAX_DONE:
            old = self._done_order.popleft()
            self._done_ids.discard(old)

    async def stream(self, event: dict, __event_emitter__=None) -> dict:
        """Called per upstream SSE chunk. Return the event unchanged."""
        if __event_emitter__ is None:
            return event
        try:
            cid = event.get("id") or ""
            # Gate: once </think> has been seen for this completion, stop.
            if cid in self._done_ids:
                return event

            choices = event.get("choices") or []
            if not choices:
                return event
            content = (choices[0].get("delta") or {}).get("content", "")
            if not content:
                return event

            # </think> means the reasoning block closed — answer body follows.
            # Clear the indicator (done=True) and lock this completion id so we
            # ignore later chunks that may incidentally contain phase words.
            if "</think>" in content:
                self._mark_done(cid)
                await __event_emitter__(
                    {"type": "status", "data": {"description": "", "done": True}}
                )
                return event

            for phase in _PHASES:
                if phase in content:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": phase, "done": False},
                        }
                    )
                    break
        except Exception:
            # Never let an indicator update break a chat
            pass
        return event
