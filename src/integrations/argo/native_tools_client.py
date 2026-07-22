"""Argo `/chat/` client with native tool calling for OpenAI / Gemini / Anthropic.

Argo's `/chat/` endpoint supports a `tools` parameter (per the gateway docs in
`argo_tool_call_notes.md`); each vendor wants a different request shape AND
returns a different response shape. This module hides that variance behind a
single LangChain-compatible chat model that the orchestrator's R2 ReAct branch
and the planner can use.

Why a separate client (and not extend `ArgoLLMClientAG`):
  - Endpoint differs: `/chat/` here vs `/streamchat/` for the existing client.
  - Response is a single JSON envelope `{"response": {"content", "tool_calls"}}`,
    not the streamed text the parent accumulates.
  - When `bound_tools` is empty we still hit `/chat/` (per user directive:
    "/chat/ for tool calling and planning").

The R2 structured-output adapter (`ToolCapableArgoLLM`) stays in the tree as a
fallback for models without native tool support.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from src.integrations.argo.llm_client import ArgoLLMClientAG

logger = logging.getLogger(__name__)


# Vendor detection — keep in sync with `LLMFactory` model names.
_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4")
_GEMINI_PREFIXES = ("gemini",)
_ANTHROPIC_PREFIXES = ("claude",)


def _vendor_for_model(model: str) -> str:
    """Return one of: 'openai', 'gemini', 'anthropic', or 'unknown'."""
    m = model.lower()
    if any(m.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    if any(m.startswith(p) for p in _GEMINI_PREFIXES):
        return "gemini"
    if any(m.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Schema rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tool_json_schema(tool: BaseTool) -> Dict[str, Any]:
    """Pull a JSON Schema (object-typed) from a LangChain tool's args_schema."""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return {"type": "object", "properties": {}, "required": []}
    if hasattr(args_schema, "model_json_schema"):
        raw = args_schema.model_json_schema()
    else:
        raw = args_schema.schema()  # pydantic v1 fallback
    # Pydantic adds 'title' fields and possibly '$defs'; drop them for prompt economy.
    out: Dict[str, Any] = {"type": "object"}
    props = raw.get("properties") or {}
    out["properties"] = {
        name: {k: v for k, v in field.items() if k != "title"}
        for name, field in props.items()
    }
    if raw.get("required"):
        out["required"] = raw["required"]
    return out


def _gemini_schema(tool: BaseTool) -> Dict[str, Any]:
    """Same as `_tool_json_schema` but with Gemini's uppercase type names."""
    base = _tool_json_schema(tool)
    type_map = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    def _upper(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: (type_map.get(v.lower(), v) if k == "type" and isinstance(v, str) else _upper(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [_upper(x) for x in node]
        return node

    return _upper(base)


# ─────────────────────────────────────────────────────────────────────────────
# Vendor request shapes (BaseTool list → Argo `tools` field)
# ─────────────────────────────────────────────────────────────────────────────
def _to_openai_tools(tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "").strip(),
                "parameters": _tool_json_schema(t),
            },
        }
        for t in tools
    ]


def _to_anthropic_tools(tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": (t.description or "").strip(),
            "input_schema": _tool_json_schema(t),
        }
        for t in tools
    ]


def _to_gemini_tools(tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tools:
        schema = _gemini_schema(t)
        # Gemini wants `required` at the top level, NOT inside `parameters`.
        required = schema.pop("required", [])
        out.append({
            "name": t.name,
            "description": (t.description or "").strip(),
            "parameters": schema,
            "required": required,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Vendor response shapes (Argo response.tool_calls → unified ToolCall list)
# ─────────────────────────────────────────────────────────────────────────────
def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _from_openai_response(raw: Any) -> List[Dict[str, Any]]:
    """OpenAI returns: [{"id", "function": {"arguments": JSON_STR, "name"}, "type": "function"}]"""
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    out: List[Dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name")
        if not name:
            continue
        args_raw = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        out.append({
            "id": tc.get("id") or _new_call_id(),
            "name": name,
            "args": args,
            "type": "tool_call",
        })
    return out


def _from_anthropic_response(raw: Any) -> List[Dict[str, Any]]:
    """Anthropic returns: [{"id", "input": dict, "name", "type": "tool_use"}]"""
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    out: List[Dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name")
        if not name:
            continue
        args = tc.get("input")
        if not isinstance(args, dict):
            args = {}
        out.append({
            "id": tc.get("id") or _new_call_id(),
            "name": name,
            "args": args,
            "type": "tool_call",
        })
    return out


def _from_gemini_response(raw: Any) -> List[Dict[str, Any]]:
    """Gemini returns either a single object OR a list of {"id": null, "args": dict, "name"}."""
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    out: List[Dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name")
        if not name:
            continue
        args = tc.get("args")
        if not isinstance(args, dict):
            args = {}
        out.append({
            "id": tc.get("id") or _new_call_id(),
            "name": name,
            "args": args,
            "type": "tool_call",
        })
    return out


_VENDOR_TOOLS_MAPPER = {
    "openai": _to_openai_tools,
    "anthropic": _to_anthropic_tools,
    "gemini": _to_gemini_tools,
}

_VENDOR_RESPONSE_MAPPER = {
    "openai": _from_openai_response,
    "anthropic": _from_anthropic_response,
    "gemini": _from_gemini_response,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main client
# ─────────────────────────────────────────────────────────────────────────────
class NativeToolsArgoLLM(ArgoLLMClientAG):
    """Argo `/chat/` client with native tool calling for the supported vendors."""

    supports_tool_calls: ClassVar[bool] = True

    bound_tools: List[Any] = []

    model_config = {"arbitrary_types_allowed": True}

    # ── Endpoint override ─────────────────────────────────────────────────
    def _get_chat_url(self) -> str:  # type: ignore[override]
        # Parent returns `…/streamchat/`; we need `…/chat/` for native tool support.
        # `base_url` is set by the parent and already strips a trailing slash.
        return f"{self.base_url}/chat/"

    # ── bind_tools (LangChain tool-binding contract) ──────────────────────
    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[BaseTool],
        **kwargs: Any,
    ) -> "NativeToolsArgoLLM":
        """Return a new instance with `tools` bound for the next `_generate` call."""
        copy = self.model_copy(deep=False)
        copy.bound_tools = list(tools)
        return copy

    # ── Argo payload extension for native tools ───────────────────────────
    def _attach_tools(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.bound_tools:
            return payload
        vendor = _vendor_for_model(self.model)
        mapper = _VENDOR_TOOLS_MAPPER.get(vendor)
        if mapper is None:
            logger.warning(
                "NativeToolsArgoLLM: model %r has no vendor mapping — skipping tools",
                self.model,
            )
            return payload
        payload["tools"] = mapper(self.bound_tools)
        # tool_choice: OpenAI uses "auto"; Anthropic uses {"type": "auto"}; Gemini omits.
        if vendor == "openai":
            payload["tool_choice"] = "auto"
        elif vendor == "anthropic":
            payload["tool_choice"] = {"type": "auto"}
        return payload

    # ── _generate: single POST + JSON parse (no streaming) ────────────────
    def _generate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        chat_url = self._get_chat_url()
        argo_messages = self._convert_messages_to_argo_format(messages)
        payload = self._build_payload(argo_messages, stop)
        payload = self._attach_tools(payload)

        logger.info(
            "ARGO /chat/ CALL | model=%s tools=%d msgs=%d",
            self.model, len(self.bound_tools), len(messages),
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(chat_url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            logger.error("Argo /chat/ timeout after %ss: %s", self.timeout, e)
            raise Exception(f"Argo /chat/ timeout: {e}")
        except Exception as e:
            logger.error("Argo /chat/ request failed: %s", e)
            raise

        if resp.status_code != 200:
            body = resp.text[:500]
            logger.error("Argo /chat/ %d: %s", resp.status_code, body)
            raise Exception(f"Argo /chat/ HTTP {resp.status_code}: {body}")

        try:
            envelope = resp.json()
        except json.JSONDecodeError as e:
            logger.error("Argo /chat/ returned non-JSON body: %s", resp.text[:500])
            raise Exception(f"Argo /chat/ JSON decode error: {e}")

        # Argo /chat/ has THREE envelope shapes in the wild:
        #   1. {"response": "<plain text>"}          ← when no tools bound (the
        #      planner case; missed before because we only handled dict shape)
        #   2. {"response": {"content": str|null, "tool_calls": [...]}}
        #   3. {"content": ..., "tool_calls": ...}   ← rare top-level variant
        # Silently dropping shape #1 turned every planner call into an empty
        # string → "Could not parse plan JSON" → synthesize-only fallback,
        # which is what produced 9/10 AR queries with "1-step plan → synthesize"
        # in the 50-q eval.
        response = envelope.get("response") if isinstance(envelope, dict) else None
        if isinstance(response, str):
            content = response
            raw_tool_calls = None
        elif isinstance(response, dict):
            content = response.get("content") or ""
            raw_tool_calls = response.get("tool_calls")
        else:
            top = envelope if isinstance(envelope, dict) else {}
            content = top.get("content") or ""
            raw_tool_calls = top.get("tool_calls")
        if not isinstance(content, str):
            content = str(content)
        vendor = _vendor_for_model(self.model)
        parser = _VENDOR_RESPONSE_MAPPER.get(vendor)
        tool_calls = parser(raw_tool_calls) if parser else []

        message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])
