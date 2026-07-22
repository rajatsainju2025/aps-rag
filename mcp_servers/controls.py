"""MCP Server: Controls — EPICS controls + composite PV search.

Provides three tools:
  - channel_read: Read current values of EPICS PVs (read-only via caget)
  - channel_find: Resolve device descriptions to PV addresses
  - pv_search:    Composite — channel_find then channel_read (no cross-server IPC)

All three share the EPICS gateway env (EPICS_CA_ADDR_LIST) and the channel
database (loaded once per server process).

Run: python -m mcp_servers.controls
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_servers._common import add_project_root_to_path, err, mcp_tool_safe, ok

logger = logging.getLogger(__name__)
_PROJECT_ROOT = add_project_root_to_path()

mcp = FastMCP(
    "controls",
    instructions=(
        "the facility EPICS control system tools. READ-ONLY access via the your-epics-gateway "
        "read-only gateway. Provides channel discovery (channel_find), live PV "
        "reads (channel_read), and a composite query (pv_search) that resolves "
        "a natural-language device description and reads the resulting PVs in "
        "one call."
    ),
)

# ---------------------------------------------------------------------------
# EPICS setup (env mutation lives in this subprocess only — safe)
# ---------------------------------------------------------------------------
_epics_configured = False


def _ensure_epics_env():
    global _epics_configured
    if _epics_configured:
        return
    addr = os.environ.get("EPICS_CA_ADDR_LIST", "your-epics-gateway.example.com:5064")
    os.environ["EPICS_CA_ADDR_LIST"] = addr
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    _epics_configured = True
    logger.info("EPICS env: ADDR_LIST=%s (read-only)", addr)


try:
    import epics as _epics  # type: ignore
    _EPICS_AVAILABLE = True
except ImportError:
    _epics = None
    _EPICS_AVAILABLE = False
    logger.warning("pyepics not installed; channel_read will be unavailable")


# ---------------------------------------------------------------------------
# Channel database (loaded once)
# ---------------------------------------------------------------------------
_channel_index = None


def _load_channel_index():
    global _channel_index
    if _channel_index is not None:
        return _channel_index

    from src.tools.channel_finder import ChannelFinderTool

    db_path = str(_PROJECT_ROOT / "data" / "channel_databases" / "hierarchical.json")
    ctx_path = str(_PROJECT_ROOT / "data" / "channel_databases" / "its_pv_in_context.json")

    tool = ChannelFinderTool(db_path=db_path, in_context_db_path=ctx_path)
    tool._load_databases()
    _channel_index = tool
    logger.info(
        "Channel DB loaded: %d hierarchical + %d in-context entries",
        len(tool._db_index), len(tool._in_context_index),
    )
    return _channel_index


# ---------------------------------------------------------------------------
# channel_read — internal helper + tool
# ---------------------------------------------------------------------------
def _read_channels_payload(channels: list[str], include_metadata: bool, timeout: float) -> dict:
    """Inner read implementation, returns a dict (not a JSON envelope) so
    pv_search can consume it directly without re-decoding."""
    if not _EPICS_AVAILABLE:
        return {"readings": {}, "errors": {"_pyepics": "not installed"}}
    _ensure_epics_env()
    if len(channels) > 50:
        return {"readings": {}, "errors": {"_cap": f"too many channels ({len(channels)}), max 50"}}
    readings: dict = {}
    errors: dict = {}
    for ch in channels:
        ch = ch.strip()
        if not ch:
            continue
        try:
            value = _epics.caget(ch, timeout=timeout)
            if value is not None:
                entry: dict = {
                    "value": float(value) if isinstance(value, (int, float)) else str(value)
                }
                if include_metadata:
                    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
                    entry["status"] = "ok"
                readings[ch] = entry
            else:
                errors[ch] = "Timeout or PV not found"
        except Exception as e:  # noqa: BLE001 — per-PV error surfaced
            errors[ch] = str(e)
    return {"readings": readings, "errors": errors}


@mcp.tool()
@mcp_tool_safe
def channel_read(
    channels: list[str],
    include_metadata: bool = False,
    timeout: float = 5.0,
) -> str:
    """Read current values of EPICS channels from the facility control system.

    READ-ONLY access only. Provide a list of PV/channel names.

    Examples:
      channel_read(["S-DCCT:CurrentM"])
      channel_read(["S-DCCT:CurrentM", "LTS:CM1:measCurrentCM"], include_metadata=True)
    """
    payload = _read_channels_payload(channels, include_metadata, timeout)
    if not payload["readings"] and payload["errors"]:
        return err(
            f"All {len(channels)} channel reads failed",
            tool="channel_read",
            errors=payload["errors"],
        )
    return ok(
        result={"readings": payload["readings"]},
        channels_read=len(payload["readings"]),
        errors=payload["errors"] or None,
    )


# ---------------------------------------------------------------------------
# channel_find
# ---------------------------------------------------------------------------
@mcp.tool()
@mcp_tool_safe
def channel_find(
    query: str,
    ring: Optional[str] = None,
    system: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """Find EPICS PV addresses for accelerator devices.

    Searches the facility channel database to resolve device descriptions to
    concrete PV addresses.

    Examples:
      channel_find("beam current")
      channel_find("BPM positions sector 5")
      channel_find("corrector", ring="SR", system="MAG")
    """
    tool = _load_channel_index()
    results = tool._search(query, ring=ring, system=system, max_results=max_results)
    return ok(
        result={"channels": results},
        count=len(results),
        query=query,
    )


# ---------------------------------------------------------------------------
# pv_search — composite (channel_find → channel_read)
# ---------------------------------------------------------------------------
@mcp.tool()
@mcp_tool_safe
def pv_search(query: str, max_pvs: int = 3) -> str:
    """One-shot device-name → PV-readings query.

    Resolves a natural-language description to PV addresses (channel_find),
    then reads the top ``max_pvs`` of them (channel_read). Useful when the
    LLM doesn't know the exact PV name and just wants the current value.

    Args:
      query: device description, e.g. "PAR kicker current".
      max_pvs: maximum number of PVs to read (default 3, capped at 10).

    Returns: ``{"resolved_pvs": [...], "readings": {pv: {value, ...}}, "errors": {...}}``
    """
    max_pvs = max(1, min(int(max_pvs), 10))
    finder = _load_channel_index()
    candidates = finder._search(query, max_results=max_pvs)
    if not candidates:
        return err(
            f"No PVs found for query: {query!r}",
            tool="pv_search",
            query=query,
            hint="Try broader keywords or include 'sector N' / system name.",
        )
    pv_names: list[str] = []
    for c in candidates:
        if isinstance(c, dict):
            for key in ("pv", "pv_name", "name", "address"):
                if key in c and c[key]:
                    pv_names.append(str(c[key]))
                    break
    pv_names = pv_names[:max_pvs]
    if not pv_names:
        return err(
            f"channel_find returned candidates but no PV addresses extractable for {query!r}",
            tool="pv_search",
            candidates=candidates,
        )
    read_payload = _read_channels_payload(pv_names, include_metadata=False, timeout=5.0)
    return ok(
        result={
            "resolved_pvs": pv_names,
            "readings": read_payload["readings"],
            "errors": read_payload["errors"] or None,
        },
        query=query,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
