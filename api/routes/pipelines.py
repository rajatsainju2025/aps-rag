"""
OpenWebUI Pipelines endpoints for all Agentic GraphRAG service modes.

OpenWebUI discovers pipelines via:
  GET  /api/pipelines              — list available pipelines
  POST /api/pipelines/{id}/run     — execute a pipeline

Available pipelines:
  - naive-rag          : Single-pass RAG (entry_rag_app workflow)
  - agentic-rag        : Multi-agent reasoning with corrective loop
  - daily-summary      : Multi-database daily operations report
  - troubleshooting    : Fault diagnosis with KG fault chains
  - studies-copilot    : Study comparison and live experiment assistance
"""
import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["pipelines"])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline definitions
# ─────────────────────────────────────────────────────────────────────────────
PIPELINES = {
    "naive-rag": {
        "id": "naive-rag",
        "name": "APS-RAG (Fast)",
        "description": "Single-pass Retrieval-Augmented Generation for the facility beamline knowledge",
        "service_mode": "naive_rag",
    },
    "agentic-rag": {
        "id": "agentic-rag",
        "name": "APS-RAG (Advanced)",
        "description": "Multi-agent reasoning pipeline with corrective RAG loop, code interpreter, and tool execution",
        "service_mode": "agentic_rag",
    },
    "daily-summary": {
        "id": "daily-summary",
        "name": "Daily Brief",
        "description": "Multi-database daily operations report from the system, ICMS, and Work Requests",
        "service_mode": "daily_summary",
    },
    "troubleshooting": {
        "id": "troubleshooting",
        "name": "AI Help",
        "description": "Fault diagnosis with KG fault chains, multi-database search, and runbook generation",
        "service_mode": "troubleshooting",
    },
    "studies-copilot": {
        "id": "studies-copilot",
        "name": "Studies Copilot",
        "description": "Live study assistant — compare experiments, analyze parameters, suggest improvements",
        "service_mode": "studies_copilot",
    },
    "image-search": {
        "id": "image-search",
        "name": "Image Search",
        "description": "Text-to-Image (SigLIP) and Image-to-Image (DINOv2) retrieval over the logbook images",
        "service_mode": "image_search",
    },
    "pv-search": {
        "id": "pv-search",
        "name": "PV Search",
        "description": "Live EPICS PV readout, archiver history, and operational context from logbook",
        "service_mode": "agentic_rag",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SSE helpers (reuse the same format as openai_compat)
# ─────────────────────────────────────────────────────────────────────────────
def _sse_chunk(content: str, model: str, cid: str) -> str:
    return "data: " + json.dumps({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }) + "\n\n"


def _sse_done(model: str, cid: str) -> str:
    return "data: " + json.dumps({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }) + "\n\ndata: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# PV short-circuit response shapers — used when classify_pv_intent matched
# ─────────────────────────────────────────────────────────────────────────────
async def _stream_pv_short_circuit(
    pv_result: Dict[str, Any], pipeline_id: str, completion_id: str,
) -> AsyncIterator[str]:
    """One-shot SSE response for a PV-intent match: <think> trace + answer."""
    trace = pv_result.get("reasoning_trace", []) or []
    answer = pv_result.get("answer", "")

    yield _sse_chunk("<think>\n", pipeline_id, completion_id)
    for step in trace:
        yield _sse_chunk(f"• {step}\n", pipeline_id, completion_id)
    yield _sse_chunk("</think>\n\n", pipeline_id, completion_id)

    chunk_size = 80
    for i in range(0, len(answer), chunk_size):
        yield _sse_chunk(answer[i:i + chunk_size], pipeline_id, completion_id)
    yield _sse_done(pipeline_id, completion_id)


def _build_owui_response(
    pv_result: Dict[str, Any], pipeline_id: str, completion_id: str,
) -> Dict[str, Any]:
    """Non-streaming OpenWebUI response shape for a PV-intent match."""
    trace = pv_result.get("reasoning_trace", []) or []
    answer = pv_result.get("answer", "")
    think_block = ""
    if trace:
        think_block = "<think>\n" + "\n".join(f"• {s}" for s in trace) + "\n</think>\n\n"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": pipeline_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": think_block + answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ─────────────────────────────────────────────────────────────────────────────
# OpenWebUI /v1/models — pipeline entries (merged with Argo models)
# ─────────────────────────────────────────────────────────────────────────────
def get_pipeline_model_entries() -> List[Dict[str, Any]]:
    """Return model-list entries for the RAG pipelines.

    These appear as selectable models in OpenWebUI's model dropdown.
    They route through our /v1/chat/completions endpoint (OpenAI compat),
    which dispatches to the correct service based on model id.

    NOTE: Do NOT include a ``pipeline`` field here.  That field tells
    OpenWebUI to route through its separate Pipelines system (PIPELINES_URLS),
    which is a different integration from the OpenAI API connection.  Since
    we serve everything via the OpenAI-compatible /v1/ endpoints, the
    ``pipeline`` field causes a 404 because no Pipelines server is configured.

    Pipeline names are prefixed with an emoji to visually distinguish them
    from raw Argo LLM models in the dropdown.
    """
    now = int(time.time())
    return [
        {
            "id": pid,
            "object": "model",
            "created": now,
            "owned_by": "agrag-pipeline",
            "name": f"\U0001f9e0 {p['name']}",
            "description": p.get("description", ""),
        }
        for pid, p in PIPELINES.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────
class PipelineMessage(BaseModel):
    role: str
    content: Any


class PipelineChatRequest(BaseModel):
    model: str = Field(..., description="Pipeline id (naive-rag or agentic-rag)")
    messages: List[PipelineMessage]
    stream: bool = False
    temperature: float = 0.1
    # The upstream Argo model to use for LLM calls within the pipeline
    argo_model: Optional[str] = Field(default=None, description="Argo model for LLM calls")


async def run_pipeline(
    pipeline_id: str,
    messages: List[PipelineMessage],
    stream: bool = False,
    argo_model: Optional[str] = None,
) -> Any:
    """Execute a RAG pipeline and return the answer.

    When streaming, wraps the pipeline's reasoning_trace in ``<think>``
    tags so OpenWebUI renders them as a collapsible "Thought" dropdown.
    OpenWebUI recognises ``<think>``, ``<thinking>``, ``<reason>``, etc.
    (see ``DEFAULT_REASONING_TAGS`` in openwebui/utils/middleware.py).
    """
    pipeline_def = PIPELINES.get(pipeline_id)
    if not pipeline_def:
        raise HTTPException(404, f"Pipeline '{pipeline_id}' not found")

    service_mode = pipeline_def["service_mode"]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Extract query from last user message
    user_msgs = [m for m in messages if m.role == "user"]
    raw_content = user_msgs[-1].content if user_msgs else ""
    query = raw_content if isinstance(raw_content, str) else ""

    if not query.strip() and pipeline_id != "image-search":
        raise HTTPException(400, "No user message provided")

    # ── Image Search pipeline — standalone, not routed through ServiceSelector
    if pipeline_id == "image-search":
        return await _run_image_search(messages, stream, completion_id)

    # ── PV intent pre-route — bypass retrieval for live-PV questions.
    #    Runs before service dispatch so any pipeline (naive_rag, agentic_rag,
    #    rag, etc.) handles "what is the current value of S35A:H1:CurrentAO?"
    #    or "what is the beam current right now?" uniformly via pv_read / pv_search.
    from api.main import components as _pv_components
    from src.services.pv_short_circuit import try_pv_short_circuit
    pv_result = try_pv_short_circuit(
        query, _pv_components, service_mode, f"owui-{completion_id}",
    )
    if pv_result is not None:
        if stream:
            return StreamingResponse(
                _stream_pv_short_circuit(pv_result, pipeline_id, completion_id),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )
        return _build_owui_response(pv_result, pipeline_id, completion_id)

    result: Dict[str, Any] = {}
    error_answer: Optional[str] = None

    # ── Streaming-enabled path: wire a real-time event emitter so tool
    #    starts/results stream BEFORE the final synthesis completes.
    if stream:
        from api.main import service_selector, settings, components
        from src.agents.event_emitter import QueueEventEmitter

        if argo_model:
            settings.argo.primary_model = argo_model
        service = service_selector.get_service(service_mode, settings, components)

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue()
        emitter = QueueEventEmitter(event_queue, loop)

        query_task = asyncio.create_task(
            asyncio.to_thread(
                service.query,
                query=query,
                session_id=f"owui-{completion_id}",
                selected_databases=["all_databases"],
                top_k=10,
                event_emitter=emitter,
            )
        )

        return StreamingResponse(
            _stream_with_events(
                pipeline_id=pipeline_id,
                pipeline_name=pipeline_def["name"],
                completion_id=completion_id,
                query_task=query_task,
                event_queue=event_queue,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # ── Non-streaming path (no event emitter — tool events would have nowhere to go)
    try:
        from api.main import service_selector, settings, components

        # Override the Argo model if the user selected one in OpenWebUI
        if argo_model:
            settings.argo.primary_model = argo_model

        service = service_selector.get_service(service_mode, settings, components)

        # ── Run blocking service.query() in a thread so the uvicorn
        #    event loop stays responsive for health-checks, model listing,
        #    and concurrent requests from other users.
        result = await asyncio.to_thread(
            service.query,
            query=query,
            session_id=f"owui-{completion_id}",
            selected_databases=["all_databases"],
            top_k=10,
        )

    except Exception as exc:
        logger.error("Pipeline %s error: %s", pipeline_id, exc, exc_info=True)
        error_answer = (
            f"⚠️ Pipeline error ({pipeline_def['name']}): {exc}\n\n"
            "Please check that Qdrant, Neo4j, and the Argo API are running."
        )

    answer = error_answer or result.get("answer", "No answer generated.")
    reasoning_trace: List[str] = result.get("reasoning_trace", [])
    owui_citations: list = result.get("openwebui_citations", [])

    # ── Merge <think> blocks: the answer from the orchestrator may already
    #    contain <think>…</think> blocks (LLM CoT reasoning).  We extract
    #    them and merge with reasoning_trace into a single <think> block so
    #    the user sees exactly ONE collapsible "Thought" dropdown.
    import re
    embedded_thinks: List[str] = []
    def _extract_think(m: re.Match) -> str:
        embedded_thinks.append(m.group(1).strip())
        return ""  # remove from answer body
    clean_answer = re.sub(
        r"<think>(.*?)</think>",
        _extract_think,
        answer,
        flags=re.DOTALL,
    ).strip()
    # Collapse excessive blank lines left after removal
    clean_answer = re.sub(r"\n{3,}", "\n\n", clean_answer)

    # Build unified think content: trace steps first, then LLM CoT
    think_parts: List[str] = []
    for step in reasoning_trace:
        think_parts.append(f"• {step}")
    for block in embedded_thinks:
        think_parts.append(block)
    has_think = bool(think_parts)

    # Non-streaming: embed single merged <think> block
    if has_think:
        think_block = "<think>\n" + "\n".join(think_parts) + "\n</think>\n\n"
        clean_answer = think_block + clean_answer

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": pipeline_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": clean_answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    # Inject OpenWebUI-compatible citation metadata for sidebar
    if owui_citations:
        response["citations"] = owui_citations

    return response


# ─────────────────────────────────────────────────────────────────────────────
# SSE streaming with real-time tool events
# ─────────────────────────────────────────────────────────────────────────────

# Max seconds the SSE stream may go without yielding bytes before we emit a
# heartbeat. Must be < the strictest upstream proxy idle timeout on the path.
# The your organization off-box proxy fronting your-rag-host.example.com idles ~30 s; we pick 8 s
# to leave a wide margin for jitter.
_SSE_KEEPALIVE_INTERVAL = 8.0


def _format_tool_event(event: Dict[str, Any]) -> str:
    """Render a tool event as a one-line markdown string for the <think> dropdown.

    OpenWebUI does not have a first-class tool-event channel, so we render
    events as `<think>` lines that show up in the collapsible dropdown the
    user already sees for reasoning traces.

    Returns an empty string for events that have nothing useful to display;
    the caller skips empty lines so they don't pollute the dropdown.
    """
    et = event.get("type", "event")
    tool = event.get("tool", "?")
    if et == "thinking_start":
        # Skip the event entirely if there's no status — emitting "Thinking..."
        # would become the dropdown's first content line and OpenWebUI promotes
        # that to the persistent collapsed-state label.
        return event.get("status") or ""
    if et == "tool_call_start":
        args = event.get("args", "")
        args_str = args if isinstance(args, str) else json.dumps(args, default=str)
        if len(args_str) > 80:
            args_str = args_str[:77] + "..."
        return f"▶ {tool}({args_str})"
    if et == "tool_call_result":
        ms = event.get("duration_ms")
        return f"✓ {tool} ({ms} ms)" if ms is not None else f"✓ {tool}"
    if et == "tool_call_error":
        return f"✗ {tool} — {event.get('error', 'error')}"
    return f"• {et}"


async def _stream_with_events(
    pipeline_id: str,
    pipeline_name: str,
    completion_id: str,
    query_task: asyncio.Task,
    event_queue: asyncio.Queue,
) -> AsyncIterator[str]:
    """Multiplex tool events from `event_queue` with the final answer.

    While `query_task` runs in a worker thread, the orchestrator emits
    `tool_call_start/result/error` events into the queue. This generator
    races a queue-get against the task to yield events as they arrive,
    then renders the final <think>/answer/citations block after the task
    completes.
    """
    sent_open_think = True

    # Open the reasoning block immediately. The first phase event (planner /
    # vector_search / ...) typically streams within ~100ms and becomes the
    # dropdown's first content line. We used to seed a literal "Thinking..."
    # placeholder here, but OpenWebUI promotes the dropdown's first line to
    # the collapsed-state label — so the placeholder visibly persisted as
    # "Thinking..." for the entire response. Phase events convey the same
    # information and don't have that side effect.
    yield _sse_chunk("<think>\n", pipeline_id, completion_id)

    async def _next_event() -> Any:
        """Wait for next event, query completion, or keepalive timeout.

        Returns the event dict, None on completion, or the string sentinel
        "__keepalive__" when the orchestrator has been silent for longer than
        _SSE_KEEPALIVE_INTERVAL — the caller emits a heartbeat to keep the
        upstream proxy from idle-timing-out mid-pipeline.
        """
        get_task = asyncio.create_task(event_queue.get())
        done, _pending = await asyncio.wait(
            {get_task, query_task},
            timeout=_SSE_KEEPALIVE_INTERVAL,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            get_task.cancel()
            return "__keepalive__"
        if get_task in done:
            return get_task.result()
        # query_task finished — drain any remaining events without blocking
        get_task.cancel()
        try:
            return event_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # ── Phase 1: stream events as they arrive
    while not query_task.done() or not event_queue.empty():
        evt = await _next_event()
        if evt == "__keepalive__":
            # No-op chat-completion chunk (empty delta). We used to emit
            # an SSE comment here, but OpenWebUI's streaming middleware
            # filters lines that don't start with "data:" (middleware.py
            # `if not data.startswith('data:'): continue`), so comments
            # never reached the browser and the upstream proxy still
            # idle-timed-out. An empty-delta data chunk is a valid
            # OpenAI-format event that OpenWebUI forwards as bytes
            # without adding any visible content.
            yield "data: " + json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": pipeline_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            }) + "\n\n"
            continue
        if evt is None:
            break
        formatted = _format_tool_event(evt)
        if not formatted:
            continue  # skip empty events so they don't seed the dropdown label
        yield _sse_chunk(formatted + "\n", pipeline_id, completion_id)

    # ── Phase 2: render the final answer
    try:
        result = await query_task
    except Exception as exc:
        logger.error("Pipeline %s error: %s", pipeline_id, exc, exc_info=True)
        result = {
            "answer": (
                f"⚠️ Pipeline error ({pipeline_name}): {exc}\n\n"
                "Please check that Qdrant, Neo4j, and the Argo API are running."
            ),
            "reasoning_trace": [],
            "openwebui_citations": [],
        }

    answer = result.get("answer", "No answer generated.")
    reasoning_trace: List[str] = result.get("reasoning_trace", [])
    owui_citations: list = result.get("openwebui_citations", [])

    # Strip embedded <think> blocks from the answer body and merge them
    # with the reasoning_trace into the single open <think> block.
    import re as _re
    embedded_thinks: List[str] = []
    def _take_think(m: _re.Match) -> str:
        embedded_thinks.append(m.group(1).strip())
        return ""
    clean_answer = _re.sub(r"<think>(.*?)</think>", _take_think, answer, flags=_re.DOTALL).strip()
    clean_answer = _re.sub(r"\n{3,}", "\n\n", clean_answer)

    extra_think_parts: List[str] = []
    for step in reasoning_trace:
        extra_think_parts.append(f"• {step}")
    for block in embedded_thinks:
        extra_think_parts.append(block)

    if extra_think_parts:
        if not sent_open_think:
            yield _sse_chunk("<think>\n", pipeline_id, completion_id)
            sent_open_think = True
        for part in extra_think_parts:
            yield _sse_chunk(part + "\n", pipeline_id, completion_id)

    if sent_open_think:
        yield _sse_chunk("</think>\n\n", pipeline_id, completion_id)

    chunk_size = 80
    for i in range(0, len(clean_answer), chunk_size):
        yield _sse_chunk(clean_answer[i:i + chunk_size], pipeline_id, completion_id)

    if owui_citations:
        yield "data: " + json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": pipeline_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "citations": owui_citations,
        }) + "\n\ndata: [DONE]\n\n"
    else:
        yield _sse_done(pipeline_id, completion_id)


# ─────────────────────────────────────────────────────────────────────────────
# Image Search pipeline (standalone — SigLIP + DINOv2)
# ─────────────────────────────────────────────────────────────────────────────
_image_pipeline = None
_image_pipeline_lock = threading.Lock()


def _get_image_pipeline():
    global _image_pipeline
    if _image_pipeline is None:
        with _image_pipeline_lock:
            if _image_pipeline is None:
                from pipelines.image_search.pipeline import Pipeline
                _image_pipeline = Pipeline()
                logger.info("Image Search pipeline initialized")
    return _image_pipeline


async def _run_image_search(
    messages: List[PipelineMessage],
    stream: bool,
    completion_id: str,
) -> Any:
    body = {"messages": [{"role": m.role, "content": m.content} for m in messages]}
    pipe = _get_image_pipeline()
    answer = await asyncio.to_thread(pipe.pipe, body)
    if not isinstance(answer, str):
        answer = "".join(answer)

    pipeline_id = "image-search"
    if stream:
        async def _stream() -> AsyncIterator[str]:
            chunk_size = 200
            for i in range(0, len(answer), chunk_size):
                yield _sse_chunk(answer[i:i + chunk_size], pipeline_id, completion_id)
            yield _sse_done(pipeline_id, completion_id)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": pipeline_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
