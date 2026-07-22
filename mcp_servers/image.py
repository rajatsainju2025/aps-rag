"""MCP Server: Image Search — text-to-image retrieval over the the system index.

Provides one tool:
  - image_search: SigLIP text-to-image similarity over the the logbook
    attachment index

Why a dedicated server (Stage D): the underlying index (``ImageIndex``) holds
multi-GB SigLIP embeddings + a FAISS index in memory. Per-call subprocess
spawn would re-load this every time. With persistent MCP sessions, the
``Pipeline`` is constructed **once** inside this server subprocess; all
subsequent calls reuse the warm index (subsequent search ~tens of ms vs.
seconds of cold load).

Image-to-image search remains on the OpenWebUI pipeline path; the MCP tool
contract is text-only.

Run: python -m mcp_servers.image
"""
from __future__ import annotations

import logging
import threading

from mcp.server.fastmcp import FastMCP

from mcp_servers._common import add_project_root_to_path, err, mcp_tool_safe, ok

logger = logging.getLogger(__name__)
add_project_root_to_path()

mcp = FastMCP(
    "image",
    instructions=(
        "Search the the logbook image index by text description. "
        "SigLIP text-to-image similarity. Returns top-K matches as markdown "
        "with image references and similarity scores."
    ),
)


_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()


def _get_pipeline():
    """Lazy-init the image_search Pipeline. Cached for the lifetime of
    this MCP server subprocess (the Stage D win — index loads once, not per
    call). The Pipeline starts a background indexer thread on construction so
    the first search may need to wait for ``_index.ready``."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        from pipelines.image_search.pipeline import Pipeline
        _PIPELINE = Pipeline()
        _PIPELINE._trigger_build()
        logger.info("image_search: Pipeline initialized; index build started")
    return _PIPELINE


@mcp.tool()
@mcp_tool_safe
def image_search(query: str, top_k: int = 5) -> str:
    """Find images in the the logbook image index that match a text description.

    Args:
      query: Text description of the image to find (e.g., "beam profile plot",
        "scope screenshot of injection waveform").
      top_k: Maximum number of matches to return (default 5).

    Returns a JSON envelope whose ``result`` is ``{"query", "results_markdown",
    "count"}`` — the markdown is the renderable surface that the synthesis
    prompt can pass through to the user.

    Examples:
      image_search("beam profile plot")
      image_search("scope screenshot of injection waveform", top_k=3)
    """
    if not query or not query.strip():
        return err("query is empty", tool="image_search",
                   hint="Pass a non-empty text description.")
    pipe = _get_pipeline()
    if not pipe._index.ready:
        return err(
            "image index is still building — try again shortly",
            tool="image_search",
            hint="Index build runs in a background thread on first init; "
                 "first call may need to wait ~tens of seconds.",
        )
    threshold = pipe.valves.similarity_threshold
    parts = pipe._search_text(query, top_k=top_k, threshold=threshold)
    markdown = "\n\n".join(parts)
    # Approximate count from the rendered markdown (one block per result).
    # A precise count would require duplicating _collect logic; the markdown
    # is the canonical surface so we pass it through.
    count = markdown.count("\n---\n") + 1 if markdown else 0
    return ok(
        result={
            "query": query,
            "results_markdown": markdown,
            "count": count,
        },
        top_k=top_k,
    )


if __name__ == "__main__":
    mcp.run()
