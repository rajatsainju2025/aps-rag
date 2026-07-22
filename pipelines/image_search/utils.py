"""Shared helpers: image decoding, query detection, result rendering."""
import base64
import re
from io import BytesIO
from typing import Any

from PIL import Image


def decode_user_image(data_uri_or_path: str) -> Image.Image:
    """Decode a base64 data URI or local file path to PIL Image."""
    if data_uri_or_path.startswith("data:image"):
        _header, b64 = data_uri_or_path.split(",", 1)
        raw = base64.b64decode(b64)
        return Image.open(BytesIO(raw)).convert("RGB")
    return Image.open(data_uri_or_path).convert("RGB")


def detect_query_image(messages: list[dict[str, Any]]) -> str | None:
    """
    Inspect the last user message for an image attachment.
    OpenWebUI sends images as content blocks:
      {"type": "image_url", "image_url": {"url": "data:…"}}
    Returns the data URI string or None.
    """
    if not messages:
        return None
    last = messages[-1]
    content = last.get("content", "")

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                url_obj = block.get("image_url", {})
                if isinstance(url_obj, dict):
                    return url_obj.get("url")
                return url_obj
            if block.get("type") == "image":
                return block.get("url") or block.get("data")

    match = re.search(r"(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)", str(content))
    if match:
        return match.group(1)
    return None


def extract_text_query(messages: list[dict[str, Any]]) -> str:
    """Extract the plain-text portion of the last user message."""
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content", "")
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return str(content).strip()


_DISPLAY_KEYS = ("logbook", "topic", "created_by", "entry_creation_datetime", "source_logbook")


def render_results(results: list[dict[str, Any]], mode: str) -> str:
    """
    Format search results as Markdown with inline images.
    Each result dict has: url, score, metadata.
    """
    if not results:
        hint = "different description" if mode == "text" else "different image"
        return f"**No matching images found.** Try a {hint}."

    if mode == "text":
        header = "## Text-to-Image Results (SigLIP)"
    else:
        header = "## Image-to-Image Results (DINOv2)"

    lines: list[str] = [header, ""]
    for i, r in enumerate(results, 1):
        score_pct = f"{r['score'] * 100:.1f}%"
        meta = r.get("metadata", {})

        entry_id = meta.get("ENTRY_ID", "")
        web_url = meta.get("web_url", "")
        id_part = f"[the system #{entry_id}]({web_url})" if entry_id and web_url else ""

        detail_parts = [
            f"**{k}**: {meta[k]}" for k in _DISPLAY_KEYS if meta.get(k)
        ]
        detail_str = " | ".join(detail_parts)

        lines.append(f"### Result {i}  —  similarity {score_pct}")
        if id_part:
            lines.append(id_part)
        if detail_str:
            lines.append(f"_{detail_str}_")
        lines.append(f"![result {i}]({r['url']})")
        lines.append("")
    return "\n".join(lines)
