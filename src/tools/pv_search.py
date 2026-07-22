"""
LangChain Tool: PVSearchTool — one-shot composite of channel_finder + pv_read.

Resolves a natural-language device/signal name to EPICS PV addresses and reads
their current live values in a single call. Designed for the planner to emit a
single step instead of chaining channel_finder → pv_read manually.
"""
import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.pv_utils import resolve_pvs_from_channel_finder

logger = logging.getLogger(__name__)


class _PVSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Device or signal name in natural language "
            "(e.g., 'beam current', 'BPM positions sector 5', 'H1 corrector setpoint')."
        )
    )
    max_pvs: int = Field(
        default=3,
        description="Maximum number of resolved PVs to read (caps CA load).",
    )


class PVSearchTool(BaseTool):
    """One-shot device-to-live-value lookup. Composes channel_finder + pv_read."""

    name: str = "pv_search"
    description: str = (
        "Resolve a device/signal name to EPICS PV addresses and read their "
        "current live values in one call. Use for queries like 'what is the "
        "beam current right now' or 'BPM positions sector 5'. Returns JSON "
        "with resolved PVs and live readings."
    )
    args_schema: Type[BaseModel] = _PVSearchInput

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        channel_finder: Optional[BaseTool] = None,
        pv_reader: Optional[BaseTool] = None,
        **kwargs: Any,
    ) -> None:
        # Stage C: channel_finder + pv_reader constructor args are kept for
        # backward compat (api/main.py wires them at startup) but no longer
        # used — the MCP `controls/pv_search` tool does the chaining inside
        # one server process. Defaulting to None lets unit tests + external
        # callers instantiate without the legacy dependencies.
        super().__init__(**kwargs)
        object.__setattr__(self, "_channel_finder", channel_finder)
        object.__setattr__(self, "_pv_reader", pv_reader)

    def _run(self, query: str, max_pvs: int = 3) -> str:
        # Stage C: route through MCP `controls/pv_search` tool, which does
        # the channel_find → channel_read chain inside ONE server process —
        # no cross-server IPC. The constructor args (channel_finder, pv_reader)
        # are kept for backward compat but no longer used.
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync(
                "controls", "pv_search", {"query": query, "max_pvs": max_pvs}
            )
        )

    @staticmethod
    def _safe_load(raw: Any) -> Any:
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}
