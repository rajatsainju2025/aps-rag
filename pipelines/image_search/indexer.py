"""
Image index builder: JSON parsing, async download, embedding, FAISS indexing.

1. Parse all JSON files in JSON_DIR  ->  list of {url, metadata}
2. Download images async  ->  PIL Images
3. Embed with SigLIP AND DINOv2
4. Build two FAISS IndexFlatIP indices
5. Persist to disk; reload if cache valid
"""
import asyncio
import hashlib
import json
import logging
import pickle
import ssl
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
import faiss
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .config import (
    WEB_BASE_URL,
    CACHE_DIR,
    DOWNLOAD_CONCURRENCY,
    DOWNLOAD_TIMEOUT,
    IMAGE_EXTENSIONS,
    JSON_DIR,
    SSL_VERIFY,
)
from .models import get_dinov2, get_siglip

log = logging.getLogger("image_search.indexer")

_METADATA_KEYS = (
    "id",
    "entry_id",
    "ENTRY_ID",
    "title",
    "name",
    "topic",
    "timestamp",
    "date",
    "entry_creation_datetime",
    "author",
    "created_by",
    "category",
    "logbook",
)


# -- JSON Parsing ----------------------------------------------------------


def _extract_urls(
    data: Any, parent_meta: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Recursively extract every attachment URL with metadata context."""
    results: list[dict[str, Any]] = []
    if parent_meta is None:
        parent_meta = {}

    if isinstance(data, dict):
        local_meta = {**parent_meta}
        for key in _METADATA_KEYS:
            if key in data and data[key] is not None:
                local_meta[key] = str(data[key])

        if "attachment_urls" in data and isinstance(data["attachment_urls"], list):
            for url in data["attachment_urls"]:
                if not url or not isinstance(url, str):
                    continue
                ext = Path(url.rsplit("?", 1)[0]).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    record_meta = {**local_meta}
                    entry_id = record_meta.get("ENTRY_ID")
                    if entry_id:
                        record_meta["web_url"] = f"{WEB_BASE_URL}{entry_id}"
                    results.append({"url": url, "metadata": record_meta})

        for v in data.values():
            results.extend(_extract_urls(v, local_meta))

    elif isinstance(data, list):
        for item in data:
            results.extend(_extract_urls(item, parent_meta))

    return results


def load_image_records() -> list[dict[str, Any]]:
    """Return deduplicated list of {url, metadata} from all the system JSON files in JSON_DIR."""
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        log.error("No JSON files found in %s", JSON_DIR)
        return []

    log.info("Found %d JSON files in %s", len(json_files), JSON_DIR)
    all_records: list[dict[str, Any]] = []

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Skipping %s: %s", jf.name, e)
            continue

        source_logbook = raw.get("subfolder_name", jf.stem)
        records = _extract_urls(raw, {"source_logbook": source_logbook})
        log.info("  %s: %d image URLs", jf.name, len(records))
        all_records.extend(records)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in all_records:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    log.info(
        "Extracted %d unique image URLs from %d files",
        len(unique),
        len(json_files),
    )
    return unique


# -- Async Image Downloading -----------------------------------------------


async def _download_one(
    session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore
) -> Image.Image | None:
    """Download a single image. Returns None on failure."""
    async with sem:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    log.warning("HTTP %d for %s", resp.status, url)
                    return None
                raw = await resp.read()
                return Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            log.warning("Failed to download %s: %s", url, e)
            return None


async def download_images(urls: list[str]) -> list[Image.Image | None]:
    """Download images with bounded concurrency. Returns list aligned with input URLs."""
    ssl_ctx = ssl.create_default_context()
    if not SSL_VERIFY:
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    conn = aiohttp.TCPConnector(ssl=ssl_ctx, limit=DOWNLOAD_CONCURRENCY)
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [_download_one(session, url, sem) for url in urls]
        return await asyncio.gather(*tasks)


# -- Embedding & Index Build -----------------------------------------------


def _cache_fingerprint() -> str:
    """Hash all JSON files' names + sizes for cache invalidation.

    Uses size (not mtime) so daily rsyncs that don't change content
    won't trigger a costly full rebuild of the image index.
    """
    parts: list[str] = []
    for jf in sorted(JSON_DIR.glob("*.json")):
        parts.append(f"{jf.name}:{jf.stat().st_size}")
    key = "|".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _cache_paths(fingerprint: str) -> dict[str, Path]:
    """Return dict of cache file paths for a given fingerprint."""
    d = CACHE_DIR / fingerprint
    return {
        "dir": d,
        "siglip_index": d / "siglip.faiss",
        "dinov2_index": d / "dinov2.faiss",
        "metadata": d / "metadata.pkl",
    }


class ImageIndex:
    """Holds both FAISS indices + aligned metadata."""

    def __init__(self) -> None:
        self.siglip_index: faiss.IndexFlatIP | None = None
        self.dinov2_index: faiss.IndexFlatIP | None = None
        self.records: list[dict[str, Any]] = []
        self._ready = False

    @property
    def ready(self) -> bool:
        """Whether the index has been built or loaded."""
        return self._ready

    def build_or_load(self) -> None:
        """Build indices from scratch or load from cache."""
        fp = _cache_fingerprint()
        paths = _cache_paths(fp)

        if all(
            paths[k].exists() for k in ("siglip_index", "dinov2_index", "metadata")
        ):
            log.info("Loading indices from cache …")
            self.siglip_index = faiss.read_index(str(paths["siglip_index"]))
            self.dinov2_index = faiss.read_index(str(paths["dinov2_index"]))
            with open(paths["metadata"], "rb") as f:
                self.records = pickle.load(f)  # noqa: S301
            self._ready = True
            log.info("Loaded %d image records from cache.", len(self.records))
            return

        log.info("Cache miss — building indices from scratch …")
        all_records = load_image_records()
        urls = [r["url"] for r in all_records]

        loop = asyncio.new_event_loop()
        try:
            images = loop.run_until_complete(download_images(urls))
        finally:
            loop.close()

        valid = [
            (rec, img)
            for rec, img in zip(all_records, images)
            if img is not None
        ]
        if not valid:
            log.error("No images downloaded successfully — index will be empty.")
            self._ready = True
            return

        self.records, pil_images = map(list, zip(*valid))
        log.info(
            "Downloaded %d / %d images successfully.", len(pil_images), len(urls)
        )

        siglip_embs, dinov2_embs = self._embed_batches(pil_images)

        siglip_matrix = np.vstack(siglip_embs).astype("float32")
        dinov2_matrix = np.vstack(dinov2_embs).astype("float32")

        self.siglip_index = faiss.IndexFlatIP(siglip_matrix.shape[1])
        self.siglip_index.add(siglip_matrix)

        self.dinov2_index = faiss.IndexFlatIP(dinov2_matrix.shape[1])
        self.dinov2_index.add(dinov2_matrix)

        paths["dir"].mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.siglip_index, str(paths["siglip_index"]))
        faiss.write_index(self.dinov2_index, str(paths["dinov2_index"]))
        with open(paths["metadata"], "wb") as f:
            pickle.dump(self.records, f)

        self._ready = True
        log.info("Index built: %d vectors in each index.", self.siglip_index.ntotal)

    @staticmethod
    def _embed_batches(
        pil_images: list[Image.Image], batch_size: int = 32
    ) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
        """Embed images in batches with both models. Returns (siglip_list, dinov2_list)."""
        siglip_embs: list[NDArray[np.float32]] = []
        dinov2_embs: list[NDArray[np.float32]] = []
        siglip_model = get_siglip()
        dinov2_model = get_dinov2()
        total_batches = (len(pil_images) - 1) // batch_size + 1

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            siglip_embs.append(siglip_model.embed_images(batch))
            dinov2_embs.append(dinov2_model.embed_images(batch))
            log.info(
                "  Embedded batch %d/%d", i // batch_size + 1, total_batches
            )

        return siglip_embs, dinov2_embs
