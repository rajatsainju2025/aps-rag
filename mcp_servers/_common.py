"""Shared utilities for MCP servers.

Provides:
  - ``ok(payload, **extras)`` / ``err(msg, **extras)`` — uniform JSON response
    shape returned across the stdio boundary. Every tool returns
    ``{"status": "ok"|"error", "result": <payload>, "error": <msg|null>, ...}``.
  - ``mcp_tool_safe`` — decorator that catches any exception in a tool
    function and returns a structured error envelope instead of letting the
    exception propagate across stdio (which would crash the server subprocess
    and leave the client with an opaque broken-pipe error). The agent's
    ``ToolExecutor`` and the LLM's ReAct loop then see the structured error
    and can choose to retry with corrected arguments.
  - ``add_project_root_to_path()`` — convenience for servers that import
    project modules (most do, since tool logic lives in ``src/tools/``).
"""
from __future__ import annotations

import functools
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def add_project_root_to_path() -> Path:
    """Insert the agentic-graph-rag root onto sys.path so MCP servers can
    ``import src.tools.<x>`` (the actual tool logic lives there). Called
    once per server module at import time."""
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def ok(result: Any = None, **extras: Any) -> str:
    """Build an OK response envelope and serialize as JSON."""
    payload: Dict[str, Any] = {"status": "ok", "result": result, "error": None}
    payload.update(extras)
    return json.dumps(payload, default=_json_default)


def err(message: str, **extras: Any) -> str:
    """Build an error response envelope and serialize as JSON.

    ``message`` should be human-readable and ideally actionable (so the LLM
    can self-correct on the next ReAct iteration). Include ``hint=`` in
    ``extras`` if you can suggest how to fix the call.
    """
    payload: Dict[str, Any] = {"status": "error", "result": None, "error": message}
    payload.update(extras)
    return json.dumps(payload, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Best-effort serializer for objects that aren't natively JSON-friendly."""
    try:
        return str(obj)
    except Exception:  # noqa: BLE001
        return repr(obj)


def mcp_tool_safe(func: Callable[..., str]) -> Callable[..., str]:
    """Wrap an MCP tool function so any exception becomes a structured error
    response rather than propagating across the stdio boundary.

    Why this matters:
      - FastMCP serializes tool return values to JSON over stdio. If the tool
        raises, FastMCP forwards the exception to the client, but the client
        sees only a generic "tool call failed" with no actionable detail.
      - The LLM's ReAct loop needs the error MESSAGE, not just a failure flag,
        so it can retry with fixed arguments (e.g. wrong column name in a
        pandas call → error tells it the valid columns → retry with correction).
      - If the exception is severe enough to crash the server subprocess,
        every subsequent call hangs until the supervisor restarts it. Catching
        keeps the subprocess alive.

    The decorated function may still call ``ok(...)``/``err(...)`` directly to
    return tool-level errors (e.g. "channel not found"); this wrapper only
    catches *unexpected* exceptions.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — explicit catch-all is the point
            tb = traceback.format_exc(limit=4)
            logger.error(
                "MCP tool %s raised: %s\n%s", func.__name__, exc, tb,
            )
            return err(
                f"{type(exc).__name__}: {exc}",
                tool=func.__name__,
                traceback_excerpt=tb.splitlines()[-3:],
            )
    return wrapper


def env_with_default(key: str, default: str) -> str:
    """Read ``os.environ[key]``; return ``default`` if missing or empty."""
    val = os.environ.get(key, "").strip()
    return val if val else default
