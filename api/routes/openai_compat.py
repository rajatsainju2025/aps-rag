"""
OpenAI-compatible proxy routes for Argo LLM Gateway.

Exposes:
  GET  /v1/models              — lists available Argo models (OpenAI format)
  POST /v1/chat/completions    — proxies to Argo streamchat, supports streaming

OpenWebUI (configured with OPENAI_API_BASE_URL=http://host.docker.internal:8003)
auto-discovers these endpoints and makes all Argo models available in the UI.
"""
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai-compat"])

# ─────────────────────────────────────────────────────────────────────────────
# Argo configuration
# ─────────────────────────────────────────────────────────────────────────────
ARGO_STREAM_URL = "https://your-argo-gateway.example.com/api/v1/resource/streamchat/"
ARGO_TIMEOUT = 120.0

ARGO_MODELS: List[str] = [
    "claude-3-opus",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-1.5-pro",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_argo_username() -> str:
    try:
        from config.settings import get_settings
        return get_settings().argo.username
    except Exception:
        return "your-username"


def _make_sse_chunk(content: str, model: str, completion_id: str) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _make_sse_done(model: str, completion_id: str) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o")
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    stop: Optional[Any] = None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/v1/models")
async def list_models():
    """Return available Argo models + RAG pipelines in OpenAI format."""
    from api.routes.pipelines import get_pipeline_model_entries
    now = int(time.time())
    argo_entries = [
        {"id": m, "object": "model", "created": now, "owned_by": "argo"}
        for m in ARGO_MODELS
    ]
    pipeline_entries = get_pipeline_model_entries()
    return {"object": "list", "data": pipeline_entries + argo_entries}


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Proxy chat completions to Argo or RAG pipelines."""
    from api.routes.pipelines import PIPELINES, run_pipeline, PipelineMessage

    argo_user = _get_argo_username()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Route pipeline models (naive-rag, agentic-rag) through pipeline executor
    if request.model in PIPELINES:
        pipeline_msgs = [PipelineMessage(role=m.role, content=m.content) for m in request.messages]
        return await run_pipeline(
            pipeline_id=request.model,
            messages=pipeline_msgs,
            stream=request.stream,
        )

    model = request.model if request.model in ARGO_MODELS else "gpt-4o"

    payload = {
        "user": argo_user,
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stop": request.stop if isinstance(request.stop, list) else [],
        "temperature": request.temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/plain",
        "Cache-Control": "no-cache",
        "Accept-Encoding": "identity",
    }

    if request.stream:
        return StreamingResponse(
            _stream_argo(payload, headers, model, completion_id),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # ── Non-streaming ────────────────────────────────────────────────────────
    full_text = ""
    try:
        async with httpx.AsyncClient(timeout=ARGO_TIMEOUT) as client:
            async with client.stream("POST", ARGO_STREAM_URL, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise HTTPException(status_code=resp.status_code, detail=body.decode())
                async for chunk in resp.aiter_text():
                    if chunk:
                        full_text += chunk
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Argo request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Argo gateway error: {exc}")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text.strip()},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_argo(
    payload: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
    completion_id: str,
) -> AsyncIterator[str]:
    """Yield SSE chunks from the Argo stream endpoint."""
    try:
        async with httpx.AsyncClient(timeout=ARGO_TIMEOUT) as client:
            async with client.stream("POST", ARGO_STREAM_URL, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    error_chunk = {
                        "error": {"message": body.decode(), "type": "upstream_error", "code": resp.status_code}
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\ndata: [DONE]\n\n"
                    return
                async for chunk in resp.aiter_text():
                    if chunk:
                        yield _make_sse_chunk(chunk, model, completion_id)
        yield _make_sse_done(model, completion_id)
    except Exception as exc:
        logger.error("Argo streaming error: %s", exc)
        error_chunk = {"error": {"message": str(exc), "type": "streaming_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\ndata: [DONE]\n\n"
