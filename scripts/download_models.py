#!/usr/bin/env python3
"""
download_models.py — fetch + cache the embedding and reranker models into models/.

Defaults (overridable in config or via flags):
  • Embedding: intfloat/e5-large-v2         (1024-dim; query/passage prefixing is
                                             NOT used by default — raw text in)
  • Reranker:  jinaai/jina-reranker-v3      (cross-encoder, loaded with
                                             trust_remote_code)
    ┌─ LICENSE ───────────────────────────────────────────────────────────────┐
    │ jinaai/jina-reranker-v3 is CC BY-NC 4.0 (NON-COMMERCIAL). For commercial  │
    │ use, set a permissive reranker in your config, e.g.                        │
    │   agent.reranker_model_name: BAAI/bge-reranker-v2-m3   (Apache-2.0)        │
    └───────────────────────────────────────────────────────────────────────────┘

Bring-your-own models: pass --embedding-model / --reranker-model a Hugging Face
id OR a local directory, or set `vector.embedding_model` /
`agent.reranker_model_name` in your config. A substituted embedding model must
have `embedding_dim` set to match its output dimensionality.

The downloader is idempotent (skips models already present), shows progress, and
fails gracefully with a clear message if downloads are unavailable (e.g. offline).

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --embedding-model intfloat/e5-base-v2
    python scripts/download_models.py --reranker-model BAAI/bge-reranker-v2-m3
"""
import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("download_models")

MODELS_DIR = _PROJECT_ROOT / "models"


def _is_local_path(model_id: str) -> bool:
    return Path(model_id).exists()


def _download(model_id: str, label: str) -> bool:
    """Snapshot-download a HF model into models/<sanitized-id>. Idempotent."""
    if _is_local_path(model_id):
        logger.info("%s: '%s' is a local path — nothing to download.", label, model_id)
        return True

    target = MODELS_DIR / model_id.replace("/", "__")
    if target.exists() and any(target.iterdir()):
        logger.info("%s: already present at %s — skipping.", label, target)
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Run: pip install -r requirements.txt")
        return False

    logger.info("%s: downloading '%s' → %s ...", label, model_id, target)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            # keep real files (not symlinks into the HF cache) so models/ is self-contained
            local_dir_use_symlinks=False,
        )
        logger.info("%s: done → %s", label, target)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("%s: FAILED to download '%s': %s", label, model_id, exc)
        logger.error("       Check your network / Hugging Face availability. If the model is gated, "
                     "run `huggingface-cli login` first. To run fully offline, pre-place the model "
                     "under models/ and point your config at that local path.")
        return False


def main() -> int:
    from config.settings import get_settings
    s = get_settings()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-model", default=None,
                   help="HF id or local path (default: config vector.embedding_model).")
    p.add_argument("--reranker-model", default=None,
                   help="HF id or local path (default: config agent.reranker_model_name).")
    p.add_argument("--skip-reranker", action="store_true", help="Only fetch the embedding model.")
    args = p.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    embedding_model = args.embedding_model or s.vector.embedding_model
    reranker_model = args.reranker_model or getattr(s.agent, "reranker_model_name", "jinaai/jina-reranker-v3")

    ok = _download(embedding_model, "embedding")
    if not args.skip_reranker:
        ok = _download(reranker_model, "reranker") and ok

    if ok:
        logger.info("All requested models are available.")
        logger.info("Tip: for fully-offline use, point your config at the local dirs under models/.")
        return 0
    logger.error("One or more downloads failed — see messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
