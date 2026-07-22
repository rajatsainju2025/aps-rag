"""
Tool executor node — dispatches tool calls from reasoning_agent to tool implementations.
"""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple

from .event_emitter import get_emitter
from .state import AgentState

logger = logging.getLogger(__name__)


def _parallel_enabled() -> bool:
    """Feature flag — set ENTRY_PARALLEL_RETRIEVAL=0 to roll back to serial."""
    return os.environ.get("ENTRY_PARALLEL_RETRIEVAL", "1") != "0"


class ToolExecutor:
    """Routes tool calls to the appropriate tool implementation."""

    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools  # {"calculator": CalculatorTool(), ...}

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute pending tool calls and return results.

        Emits `tool_call_start` / `tool_call_result` / `tool_call_error`
        events through the contextvar-bound emitter (no-op by default), and
        appends a one-line summary per call to `state["reasoning_trace"]` so
        downstream observers (the OpenWebUI <think> rendering, the counting
        verifier's tool-use detection) can see which tools fired.

        When ENTRY_PARALLEL_RETRIEVAL is unset or "1" (default ON) and the
        batch has ≥2 calls, calls fan out across a ThreadPoolExecutor to cut
        agentic_rag latency on multi-tool plans (SPEC §10). Order in the
        returned tool_results matches the input tool_calls order. Per-tool
        failures are isolated — one tool raising does not abort the others.
        """
        tool_calls: List[Dict] = state.get("tool_calls", [])
        if not tool_calls:
            return {"tool_results": []}

        emitter = get_emitter()
        # Take a copy so we don't mutate the caller's list in place.
        reasoning_trace: List[str] = list(state.get("reasoning_trace") or [])

        if _parallel_enabled() and len(tool_calls) > 1:
            results, trace_lines = self._dispatch_parallel(tool_calls, emitter)
            logger.info(
                "Parallel tool dispatch: %d calls (tools=%s)",
                len(tool_calls),
                ",".join(sorted({c.get("tool", "?") for c in tool_calls})),
            )
        else:
            results, trace_lines = self._dispatch_serial(tool_calls, emitter)
        reasoning_trace.extend(trace_lines)

        # Merge with any pre-existing tool_results in state so an upstream
        # dispatch (e.g. PV tools fired from agentic_rag_node) is not wiped
        # when this node runs later in the graph.
        prior = state.get("tool_results") or []
        return {
            "tool_results": list(prior) + results,
            "tool_calls": [],
            "reasoning_trace": reasoning_trace,
        }

    # ------------------------------------------------------------------
    # Single-call dispatch — used by both serial and parallel paths
    # ------------------------------------------------------------------
    def _execute_call(self, call: Dict) -> Dict[str, Any]:
        """Run one tool call. Returns a result dict; never raises.

        Carries `duration_ms` and a `missing` flag for the post-pass that
        emits trace lines + events; both are stripped before the result is
        returned to the caller.
        """
        tool_name = call.get("tool", "")
        args = call.get("args", "")
        tool = self.tools.get(tool_name)
        if not tool:
            return {
                "tool": tool_name,
                "args": args,
                "result": f"Tool '{tool_name}' not available",
                "error": True,
                "duration_ms": 0,
                "missing": True,
            }
        t0 = time.monotonic()
        try:
            result = tool.run(tool_input=args) if isinstance(args, dict) else tool.run(args)
            duration_ms = int((time.monotonic() - t0) * 1000)
            return {"tool": tool_name, "args": args, "result": result, "error": False, "duration_ms": duration_ms}
        except Exception as exc:  # noqa: BLE001 — defensive isolation per spec
            duration_ms = int((time.monotonic() - t0) * 1000)
            return {"tool": tool_name, "args": args, "result": str(exc), "error": True, "duration_ms": duration_ms}

    def _emit_and_trace(self, result: Dict, emitter) -> str:
        """Emit the appropriate event for one result; return its trace line.

        Mutates `result` in place to strip the internal `duration_ms` and
        `missing` fields so callers see the same shape the original
        serial loop produced.
        """
        tool_name = result["tool"]
        duration_ms = result.get("duration_ms", 0)
        if result.get("missing"):
            line = f"Tool {tool_name} skipped: not registered"
            emitter.emit("tool_call_error", {"tool": tool_name, "error": "tool not registered"})
            logger.warning(f"Tool {tool_name} not registered")
        elif result.get("error"):
            line = f"Tool {tool_name} failed ({duration_ms} ms): {result['result']}"
            emitter.emit("tool_call_error", {"tool": tool_name, "duration_ms": duration_ms, "error": result["result"]})
            logger.error(f"Tool {tool_name} failed: {result['result']}")
        else:
            line = f"Tool {tool_name} executed ({duration_ms} ms)"
            emitter.emit("tool_call_result", {"tool": tool_name, "duration_ms": duration_ms})
            logger.info(f"Tool {tool_name} executed successfully")
        result.pop("duration_ms", None)
        result.pop("missing", None)
        return line

    def _dispatch_serial(self, tool_calls: List[Dict], emitter) -> Tuple[List[Dict], List[str]]:
        """Serial path — preserves the legacy event ordering for a single call."""
        results: List[Dict] = []
        trace_lines: List[str] = []
        for call in tool_calls:
            emitter.emit("tool_call_start", {"tool": call.get("tool", ""), "args": call.get("args", "")})
            r = self._execute_call(call)
            trace_lines.append(self._emit_and_trace(r, emitter))
            results.append(r)
        return results, trace_lines

    def _dispatch_parallel(self, tool_calls: List[Dict], emitter) -> Tuple[List[Dict], List[str]]:
        """Fan out across a thread pool. Order is preserved via pool.map()."""
        for call in tool_calls:
            emitter.emit("tool_call_start", {"tool": call.get("tool", ""), "args": call.get("args", "")})
        with ThreadPoolExecutor(max_workers=min(8, len(tool_calls))) as pool:
            results = list(pool.map(self._execute_call, tool_calls))
        trace_lines = [self._emit_and_trace(r, emitter) for r in results]
        return results, trace_lines
