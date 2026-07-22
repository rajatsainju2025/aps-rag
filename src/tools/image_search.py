"""
LangChain Tool: ImageSearchTool — text-to-image retrieval over the the system index.

Stage D: ``_run`` dispatches through MCP (``image/image_search``) so the
multi-GB SigLIP index lives in ONE persistent subprocess (not reloaded per
call). The schema + description here are what the LLM sees natively — the
actual search logic lives in [mcp_servers/image.py](../../mcp_servers/image.py).

Image-to-image search remains on the OpenWebUI pipeline path; the planner
contract is string-args only.
"""
import logging
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _ImageSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Text description of the image to find "
            "(e.g., 'beam profile plot', 'scope screenshot of injection waveform')."
        )
    )
    top_k: int = Field(default=5, description="Maximum number of matches to return.")


class ImageSearchTool(BaseTool):
    """Search the the system image index by text description (SigLIP). MCP-backed."""

    name: str = "image_search"
    description: str = (
        "Find images in the the logbook image index that match a text "
        "description. Uses SigLIP text-to-image similarity. Returns top-K "
        "matches as markdown with image references and similarity scores."
    )
    args_schema: Type[BaseModel] = _ImageSearchInput

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, top_k: int = 5) -> str:
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync(
                "image", "image_search", {"query": query, "top_k": top_k}
            )
        )
