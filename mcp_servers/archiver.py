"""MCP Server: Archiver — historical PV data from the facility Archiver Appliance.

Provides one tool:
  - archiver_read: Fetch historical PV data with summary statistics

Run: python -m mcp_servers.archiver
"""
from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from mcp_servers._common import add_project_root_to_path, err, mcp_tool_safe, ok

logger = logging.getLogger(__name__)
add_project_root_to_path()

mcp = FastMCP(
    "archiver",
    instructions=(
        "the facility Archiver Appliance tools. Fetch historical PV data from "
        "https://your-archiver.example.com. Returns summary statistics."
    ),
)


@mcp.tool()
@mcp_tool_safe
def archiver_read(
    pv_names: list[str],
    start: str,
    end: str = "",
    processing: str = "",
) -> str:
    """Fetch historical PV data from the facility EPICS Archiver Appliance.

    Returns summary statistics (min, max, mean, count) for each PV over a time range.

    Args:
      pv_names: List of PV names to query
      start: Start time — ISO8601 (2025-01-01T00:00:00Z) or relative (1h, 24h, 7d, 30m)
      end: End time — ISO8601 or empty string for 'now'
      processing: Server-side processing, e.g. 'mean_600' for 10-min bins

    Examples:
      archiver_read(["S-DCCT:CurrentM"], "1h")
      archiver_read(["S-DCCT:CurrentM", "S-DCCT:LifetimeM"], "24h")
      archiver_read(["S-DCCT:CurrentM"], "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z")
    """
    if not pv_names:
        return err("pv_names is empty", tool="archiver_read", hint="Pass at least one PV name.")
    from src.tools.archiver_reader import ArchiverReaderTool
    tool = ArchiverReaderTool()
    pv_str = ", ".join(pv_names)
    # Call _run_local — _run is now an MCP shim that would recurse.
    raw = tool._run_local(pv_names=pv_str, start=start, end=end, processing=processing)
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        return err(f"unexpected archiver response shape: {type(parsed).__name__}", tool="archiver_read")
    if not parsed.get("data"):
        # Strip keys that would collide with err()'s named args (Python would
        # raise "got multiple values for argument 'message'" otherwise).
        _reserved = {"data", "message", "tool", "status", "result", "error"}
        extras = {k: v for k, v in parsed.items() if k not in _reserved}
        return err(
            f"Archiver returned no data for {len(pv_names)} PV(s) over {start}",
            tool="archiver_read",
            pv_names=pv_names,
            start=start,
            end=end,
            **extras,
        )
    return ok(
        result=parsed,
        pv_count=len(pv_names),
        start=start,
        end=end or "now",
    )


if __name__ == "__main__":
    mcp.run()
