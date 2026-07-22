"""
LangChain Tool: PVReaderTool — read current EPICS PV values.

READ-ONLY access to the facility EPICS control system via the your-epics-gateway
read-only gateway. Prefers MCP server when available, falls back to
direct pyepics caget().
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    import epics
    _EPICS_AVAILABLE = True
except ImportError:
    _EPICS_AVAILABLE = False
    logger.warning("pyepics not installed; PV reader will use MCP or be unavailable")

_MAX_PVS_DEFAULT = 50


class _PVReaderInput(BaseModel):
    pv_names: str = Field(
        description=(
            "Comma-separated EPICS PV names to read "
            "(e.g., 'S-DCCT:CurrentM' or 'S-DCCT:CurrentM, LTS:H1:CurrentAO')"
        )
    )
    timeout: float = Field(default=5.0, description="Timeout in seconds per PV read")


class PVReaderTool(BaseTool):
    """Read current EPICS PV values (READ-ONLY)."""

    name: str = "pv_read"
    description: str = (
        "Read current values of EPICS process variables from the facility control system. "
        "READ-ONLY access only. Provide comma-separated PV names. "
        "Returns JSON with current values, timestamps, and status."
    )
    args_schema: Type[BaseModel] = _PVReaderInput
    gateway_address: str = Field(default="your-epics-gateway.example.com", exclude=True)
    gateway_port: int = Field(default=5064, exclude=True)
    default_timeout: float = Field(default=5.0, exclude=True)
    max_pvs: int = Field(default=_MAX_PVS_DEFAULT, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        gateway_address: str = "your-epics-gateway.example.com",
        gateway_port: int = 5064,
        default_timeout: float = 5.0,
        max_pvs: int = _MAX_PVS_DEFAULT,
        mcp_client: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "gateway_address", gateway_address)
        object.__setattr__(self, "gateway_port", gateway_port)
        object.__setattr__(self, "default_timeout", default_timeout)
        object.__setattr__(self, "max_pvs", max_pvs)
        object.__setattr__(self, "_mcp_client", mcp_client)
        object.__setattr__(self, "_env_configured", False)

    def _configure_epics_env(self) -> None:
        if not self._env_configured:
            os.environ["EPICS_CA_ADDR_LIST"] = f"{self.gateway_address}:{self.gateway_port}"
            os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
            object.__setattr__(self, "_env_configured", True)
            logger.info(
                "EPICS env configured: gateway=%s:%d (read-only)",
                self.gateway_address,
                self.gateway_port,
            )

    def _run(self, pv_names: str, timeout: float = 5.0) -> str:
        # Stage C: route through MCP `controls` server (channel_read tool).
        # Local pyepics fallback is gone — the MCP server holds the EPICS env
        # and is the single source of truth.
        pvs = [pv.strip() for pv in pv_names.split(",") if pv.strip()]
        if not pvs:
            return json.dumps({"error": "No PV names provided", "readings": []})
        if len(pvs) > self.max_pvs:
            return json.dumps({
                "error": f"Too many PVs ({len(pvs)}), max {self.max_pvs}",
                "readings": [],
            })
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync(
                "controls", "channel_read",
                {"channels": pvs, "include_metadata": True, "timeout": timeout},
            )
        )
