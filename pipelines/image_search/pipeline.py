"""
OpenWebUI Pipeline: the system Image Search

Provides Text-to-Image (SigLIP) and Image-to-Image (DINOv2) retrieval
over the the logbook attachment images.

This pipeline is fully isolated and does NOT interfere with any
other pipelines or models registered in OpenWebUI.
"""
import logging
import threading
from typing import Any, Generator

from .config import SIMILARITY_THRESH, TOP_K
from .indexer import ImageIndex
from .models import get_dinov2, get_siglip
from .utils import (
    decode_user_image,
    detect_query_image,
    extract_text_query,
    render_results,
)

log = logging.getLogger("image_search.pipeline")

_META_COMMANDS = frozenset({"", "hi", "hello", "hey", "help", "/help"})


class Pipeline:
    """
    OpenWebUI pipeline contract:
      - class must be named `Pipeline`
      - must expose `name`, `pipe()`, and optionally `on_startup()` / `on_shutdown()`
    """

    class Valves:
        """User-tunable parameters exposed in the OpenWebUI UI."""

        top_k: int = TOP_K
        similarity_threshold: float = SIMILARITY_THRESH

    def __init__(self) -> None:
        self.name = "Image Search"
        self.valves = self.Valves()
        self._index = ImageIndex()
        self._build_lock = threading.Lock()
        self._build_started = False

    # -- Lifecycle hooks ---------------------------------------------------

    async def on_startup(self) -> None:
        """Trigger background index build so first query is fast."""
        self._trigger_build()

    async def on_shutdown(self) -> None:
        """Cleanup hook (no-op)."""

    def _trigger_build(self) -> None:
        """Start index build in a background thread (non-blocking)."""
        if self._build_started:
            return
        with self._build_lock:
            if self._build_started:
                return
            self._build_started = True
            t = threading.Thread(target=self._index.build_or_load, daemon=True)
            t.start()

    # -- Main entry point --------------------------------------------------

    def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
    ) -> str | Generator[str, None, None]:
        """
        Called by OpenWebUI for every user message routed to this pipeline.

        Routing logic:
          - Image attachment detected  ->  Image-to-Image (DINOv2)
          - Text only                  ->  Text-to-Image  (SigLIP)
          - Both                       ->  Run both, concatenate results
        """
        self._trigger_build()

        if not self._index.ready:
            return "**Image index is still building.** Please try again in a moment."

        if not self._index.records:
            return "**No images were indexed.** Check the JSON data path and image URLs."

        messages = body.get("messages", [])
        image_data = detect_query_image(messages)
        text_query = extract_text_query(messages)
        top_k = self.valves.top_k
        thresh = self.valves.similarity_threshold

        parts: list[str] = []

        if text_query and text_query.strip().lower() not in _META_COMMANDS:
            parts.extend(self._search_text(text_query, top_k, thresh))

        if image_data:
            parts.extend(self._search_image(image_data, top_k, thresh))

        if not parts:
            return (
                "**the system Image Search**\n\n"
                "I can help you find images from the the logbook:\n\n"
                "- **Text search** — describe what you're looking for "
                '(e.g., *"beam profile plot"*)\n'
                "- **Image search** — upload an image to find visually similar ones\n\n"
                "_Powered by SigLIP (text-to-image) and DINOv2 (image-to-image)._"
            )

        return "\n\n---\n\n".join(parts)

    # -- Search helpers ----------------------------------------------------

    def _search_text(
        self, text_query: str, top_k: int, threshold: float
    ) -> list[str]:
        """Run SigLIP text-to-image search."""
        try:
            siglip = get_siglip()
            q_emb = siglip.embed_text([text_query])
            scores, indices = self._index.siglip_index.search(q_emb, top_k)
            results = self._collect(scores[0], indices[0], threshold)
            return [render_results(results, mode="text")]
        except Exception as e:
            log.exception("SigLIP search failed")
            return [f"Text search error: {e}"]

    def _search_image(
        self, image_data: str, top_k: int, threshold: float
    ) -> list[str]:
        """Run DINOv2 image-to-image search."""
        try:
            query_img = decode_user_image(image_data)
            dinov2 = get_dinov2()
            q_emb = dinov2.embed_images([query_img])
            scores, indices = self._index.dinov2_index.search(q_emb, top_k)
            results = self._collect(scores[0], indices[0], threshold)
            return [render_results(results, mode="image")]
        except Exception as e:
            log.exception("DINOv2 search failed")
            return [f"Image search error: {e}"]

    def _collect(
        self,
        scores: Any,
        indices: Any,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Gather results above similarity threshold."""
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or score < threshold:
                continue
            rec = self._index.records[idx]
            results.append(
                {
                    "url": rec["url"],
                    "score": float(score),
                    "metadata": rec.get("metadata", {}),
                }
            )
        return results
