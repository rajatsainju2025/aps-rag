"""
End-to-end test: picks 10 random indexed images, tests both search modes.

Test B: Text-to-Image  — query with parent entry's topic, check if image appears in top-K
Test C: Image-to-Image — download image, feed as query, check self-retrieval at rank 1
Test D: Metadata verification — check ENTRY_ID, web_url, logbook present
"""
import asyncio
import logging
import random
import ssl
from io import BytesIO

import aiohttp
from PIL import Image

from pipelines.image_search.config import DOWNLOAD_TIMEOUT, SSL_VERIFY
from pipelines.image_search.indexer import ImageIndex
from pipelines.image_search.models import get_dinov2, get_siglip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("test_e2e")

TOP_K = 50


async def download_image(url: str) -> Image.Image | None:
    """Download a single image for testing."""
    ssl_ctx = ssl.create_default_context()
    if not SSL_VERIFY:
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(connector=conn) as session:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    return None
                raw = await resp.read()
                return Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            log.warning("Download failed %s: %s", url, e)
            return None


def main() -> None:
    # Load index
    log.info("Loading index from cache...")
    idx = ImageIndex()
    idx.build_or_load()
    assert idx.ready, "Index not ready"
    assert len(idx.records) > 0, "No records in index"
    log.info("Index loaded: %d records", len(idx.records))

    # Pick 10 random records that have topic metadata (for text search)
    candidates = [
        r for r in idx.records
        if r["metadata"].get("topic") and len(r["metadata"]["topic"]) > 5
    ]
    random.seed(42)
    sample = random.sample(candidates, min(10, len(candidates)))
    log.info("Selected %d test images from %d candidates with topics", len(sample), len(candidates))

    # Load models
    siglip = get_siglip()
    dinov2 = get_dinov2()

    # Download the 10 test images
    async def _download_all() -> list[Image.Image | None]:
        return await asyncio.gather(*[download_image(r["url"]) for r in sample])

    loop = asyncio.new_event_loop()
    test_images = loop.run_until_complete(_download_all())
    loop.close()

    # ── Test D: Metadata Verification ──
    log.info("\n=== TEST D: Metadata Verification ===")
    required_keys = {"ENTRY_ID", "web_url", "logbook"}
    missing_count = 0
    for i, r in enumerate(sample):
        meta = r["metadata"]
        present = {k for k in required_keys if meta.get(k)}
        missing = required_keys - present
        if missing:
            log.warning("  Record %d (%s): MISSING %s", i, r["url"][-40:], missing)
            missing_count += 1
        else:
            log.info(
                "  Record %d: ENTRY_ID=%s, web_url=%s, logbook=%s",
                i, meta["ENTRY_ID"], meta["web_url"], meta["logbook"],
            )
    log.info("Metadata check: %d/%d records have all required keys", len(sample) - missing_count, len(sample))

    # ── Test B: Text-to-Image (SigLIP) ──
    log.info("\n=== TEST B: Text-to-Image Search (SigLIP) ===")
    text_hits = 0
    for i, r in enumerate(sample):
        topic = r["metadata"]["topic"]
        q_emb = siglip.embed_text([topic])
        scores, indices = idx.siglip_index.search(q_emb, TOP_K)
        result_urls = [idx.records[int(j)]["url"] for j in indices[0] if j >= 0]
        found = r["url"] in result_urls
        rank = result_urls.index(r["url"]) + 1 if found else -1
        status = f"FOUND at rank {rank}" if found else "NOT FOUND in top-{TOP_K}"
        log.info(
            "  [%d] query=%r -> %s (score=%.4f)",
            i, topic[:50], status, scores[0][0],
        )
        if found:
            text_hits += 1
    log.info("Text-to-Image: %d/%d found in top-%d", text_hits, len(sample), TOP_K)

    # ── Test C: Image-to-Image (DINOv2) ──
    log.info("\n=== TEST C: Image-to-Image Search (DINOv2) ===")
    img_hits = 0
    img_rank1 = 0
    for i, (r, img) in enumerate(zip(sample, test_images)):
        if img is None:
            log.warning("  [%d] Skipped — download failed", i)
            continue
        q_emb = dinov2.embed_images([img])
        scores, indices = idx.dinov2_index.search(q_emb, TOP_K)
        result_urls = [idx.records[int(j)]["url"] for j in indices[0] if j >= 0]
        found = r["url"] in result_urls
        rank = result_urls.index(r["url"]) + 1 if found else -1
        status = f"FOUND at rank {rank}" if found else "NOT FOUND"
        log.info(
            "  [%d] self-retrieval -> %s (score=%.4f)",
            i, status, scores[0][0],
        )
        if found:
            img_hits += 1
        if rank == 1:
            img_rank1 += 1
    log.info(
        "Image-to-Image: %d/%d found, %d/%d at rank 1",
        img_hits, len(sample), img_rank1, len(sample),
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("E2E TEST SUMMARY")
    print("=" * 60)
    print(f"Index size:          {len(idx.records)} records")
    print(f"SigLIP dim:          {idx.siglip_index.d}")
    print(f"DINOv2 dim:          {idx.dinov2_index.d}")
    print(f"Metadata complete:   {len(sample) - missing_count}/{len(sample)}")
    print(f"Text-to-Image hits:  {text_hits}/{len(sample)} in top-{TOP_K}")
    print(f"Image-to-Image hits: {img_hits}/{len(sample)} in top-{TOP_K}")
    print(f"Image-to-Image @1:   {img_rank1}/{len(sample)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
