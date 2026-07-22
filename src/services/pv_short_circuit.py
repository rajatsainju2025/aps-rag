"""
PV short-circuit handler — runs the PV intent classifier and, on a hit,
invokes pv_read or pv_search directly and returns a service-shaped response
without touching the normal RAG path.

Used by all three API call sites (api/routes/query.py and the two
service.query() entry points in api/routes/pipelines.py).
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from src.agents.pv_intent import PVIntent, classify_pv_intent

logger = logging.getLogger(__name__)

_TRACE_HEADER = "PV intent detected, bypassed retrieval"


def try_pv_short_circuit(
    query: str,
    components: Dict[str, Any],
    selected_mode: str,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Return a service-result dict if PV intent matched, else None.

    The returned dict matches the contract of BaseService.query() so callers
    can hand it back to the client (REST or OpenWebUI) without reshaping.
    """
    intent = classify_pv_intent(query)
    if intent is None:
        return None

    start = time.time()
    tool_name = intent["route"]
    tools = components.get("tools") or {}
    tool = tools.get(tool_name)

    base_trace = [
        f"{_TRACE_HEADER} ({intent['label']})",
        f"User selected mode: {selected_mode}",
    ]

    if tool is None:
        msg = (
            f"PV intent detected ({intent['label']}) but the live-read tool "
            f"`{tool_name}` is currently unavailable. Retrieval would not "
            "return live values, so this request was not answered. Check "
            "that the MCP `controls` server is running."
        )
        logger.warning("PV short-circuit: tool %s unavailable", tool_name)
        return _build_response(
            answer=msg,
            trace=base_trace + [f"Tool unavailable: {tool_name}", "Aborted"],
            mode=selected_mode,
            start=start,
        )

    args_repr = _format_args(intent["args"])
    trace = base_trace + [f"Called {tool_name}({args_repr})"]

    try:
        raw = tool._run(**intent["args"])
    except Exception as exc:
        logger.error("PV short-circuit: tool %s raised: %s", tool_name, exc)
        return _build_response(
            answer=(
                f"PV tool `{tool_name}` failed: {exc}. The PV may not exist, "
                "the IOC may be down, or the gateway may have timed out."
            ),
            trace=trace + [f"Tool raised: {exc}"],
            mode=selected_mode,
            start=start,
        )

    answer, n_readings, status = _format_tool_output(tool_name, raw, intent)
    if status == "error":
        trace.append("Tool returned an error envelope")
    elif n_readings == 0:
        trace.append("Tool returned 0 readings")
    else:
        trace.append(f"Tool returned {n_readings} reading(s)")
    return _build_response(answer=answer, trace=trace, mode=selected_mode, start=start)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _format_args(args: Dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def _build_response(
    answer: str,
    trace: List[str],
    mode: str,
    start: float,
) -> Dict[str, Any]:
    return {
        "answer": answer,
        "citations": [],
        "openwebui_citations": [],
        "sources": [],
        "service_mode": mode,
        "reasoning_trace": trace,
        "latency_ms": round((time.time() - start) * 1000, 2),
    }


def _format_tool_output(tool_name: str, raw: Any, intent: PVIntent) -> tuple[str, int, str]:
    """Turn raw tool JSON into a natural-language answer.

    Returns (answer, n_readings, status) where status is "ok" or "error".

    pv_read returns:   {"readings": [{"pv","value","timestamp","status",...}], "count": N}
                  or:  {"error": "...", "readings": []}

    pv_search returns: {"resolved_pvs": [...], "readings": {pv_name: {value: ...}}, "errors": null|{pv: msg}}
                  i.e. readings is a DICT keyed by PV name, not a list.
    """
    data = _safe_json(raw)
    if not isinstance(data, dict):
        return (f"Tool returned an unparseable response:\n```\n{raw}\n```", 0, "error")

    # pv_read uses "error" (singular), pv_search uses "errors" (plural, dict or None)
    err = data.get("error")
    if err:
        targets = ", ".join(f"`{n}`" for n in intent.get("matched", [])) or "the requested PV(s)"
        return (
            f"Could not read {targets}: {err}. "
            "The PV may not exist, the IOC may be down, or the gateway may have timed out.",
            0, "error",
        )

    # Normalise readings to a list of {"pv", "value", ...} dicts regardless of shape.
    norm_readings = _normalise_readings(data.get("readings"))

    if not norm_readings:
        targets = ", ".join(f"`{n}`" for n in intent.get("matched", [])) or "the requested PV(s)"
        # pv_search may report per-PV errors here
        errs = data.get("errors")
        if isinstance(errs, dict) and errs:
            err_lines = "; ".join(f"`{k}`: {v}" for k, v in errs.items())
            return (f"No readings for {targets}. Errors: {err_lines}.", 0, "error")
        return (f"No readings returned for {targets}.", 0, "ok")

    lines: List[str] = []
    if len(norm_readings) == 1:
        lines.append(_format_one_reading(norm_readings[0], prefix="The current value of "))
    else:
        lines.append(f"Live values for {len(norm_readings)} PV(s):")
        lines.append("")
        for r in norm_readings:
            lines.append("- " + _format_one_reading(r, prefix=""))

    # Surface any per-PV errors that came alongside successful readings
    errs = data.get("errors")
    if isinstance(errs, dict) and errs:
        lines.append("")
        lines.append(f"_Errors on other PVs: {', '.join(f'`{k}`' for k in errs)}._")

    # Surface resolved-PV provenance from pv_search
    if tool_name == "pv_search":
        resolved = data.get("resolved_pvs") or data.get("pv_addresses")
        if isinstance(resolved, list) and resolved:
            names = [
                item["pv"] if isinstance(item, dict) and item.get("pv") else item
                for item in resolved if isinstance(item, (dict, str))
            ]
            if names:
                lines.append("")
                lines.append(f"_Resolved from query: {', '.join(f'`{n}`' for n in names[:10])}_")

    return ("\n".join(lines), len(norm_readings), "ok")


def _normalise_readings(readings: Any) -> List[Dict[str, Any]]:
    """Flatten pv_read's list-of-dicts OR pv_search's dict-of-dicts to a list."""
    if isinstance(readings, list):
        return [r for r in readings if isinstance(r, dict)]
    if isinstance(readings, dict):
        out: List[Dict[str, Any]] = []
        for pv_name, payload in readings.items():
            if isinstance(payload, dict):
                merged = {"pv": pv_name, **payload}
                out.append(merged)
            else:
                # value-only (e.g. raw number) — wrap it
                out.append({"pv": pv_name, "value": payload})
        return out
    return []


def _format_one_reading(r: Dict[str, Any], prefix: str) -> str:
    pv = r.get("pv") or r.get("name") or "?"
    val = r.get("value")
    val_str = f"**{val}**" if val is not None else "_(no value)_"
    ts = r.get("timestamp") or r.get("time") or ""
    status = r.get("status") or ""
    units = r.get("units") or r.get("egu") or ""
    units_str = f" {units}" if units else ""
    ts_str = f" (read at {ts}" if ts else ""
    status_str = f", status: {status}" if status else ""
    close = ")" if ts_str else ""
    return f"{prefix}`{pv}` is {val_str}{units_str}{ts_str}{status_str}{close}".rstrip()


def _safe_json(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None
