"""PV utility helpers shared across tools and orchestrator."""
import json
from typing import Any, List


def resolve_pvs_from_channel_finder(raw: Any) -> str:
    """Extract a comma-separated PV list from a channel_finder JSON result.

    channel_finder returns either:
      - {"pv_addresses": [{"pv": "S-DCCT:CurrentM", ...}, ...], "count": N}  (local)
      - {"pv_addresses": ["S-DCCT:CurrentM", ...], "count": N}               (MCP)
    Returns "" if the payload cannot be parsed or no PVs are present.
    """
    if not raw:
        return ""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return ""
    addrs = data.get("pv_addresses") if isinstance(data, dict) else None
    if not isinstance(addrs, list):
        return ""
    pvs: List[str] = []
    for item in addrs:
        if isinstance(item, dict):
            pv = item.get("pv")
            if isinstance(pv, str) and pv:
                pvs.append(pv)
        elif isinstance(item, str) and item:
            pvs.append(item)
    return ",".join(pvs)
