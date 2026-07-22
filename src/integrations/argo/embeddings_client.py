"""
Argo Embeddings client (OPTIONAL, self-contained backend).

Provides RetryArgoEmbeddingsClient (exponential backoff for transient errors)
on top of the vendored ArgoEmbeddingsClient base.

NOTE: the DEFAULT embedding path is the local HuggingFace e5-large-v2 model
(see LLMFactory.create_embeddings) — no external API. This gateway embedder
remains for sites that prefer a remote embedding service.
"""
import logging
import time
from typing import List

from src.integrations.argo.argo_embeddings_base import ArgoEmbeddingsClient  # vendored base

logger = logging.getLogger(__name__)


class RetryArgoEmbeddingsClient(ArgoEmbeddingsClient):
    """
    ArgoEmbeddingsClient with retry logic for transient API errors.

    Retries on 5xx, 429 (rate limit), and timeout errors with exponential backoff.
    Used by pipelines where a single transient failure must not destroy a
    multi-minute embedding job.
    """

    def __init__(
        self,
        *args,
        max_retries: int = 3,
        base_delay: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.base_delay = base_delay

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed with per-batch retry on transient errors."""
        import httpx

        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} documents in batches of {self.batch_size}")
        all_embeddings = []

        for batch_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_idx: batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            payload = {
                "user": self.username,
                "model": self.model,
                "prompt": batch_texts,
            }

            last_error = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = httpx.post(
                        self.embed_url, json=payload, timeout=self.timeout
                    )
                    response.raise_for_status()

                    result = response.json()
                    batch_embeddings = result.get("embedding", [])
                    all_embeddings.extend(batch_embeddings)
                    last_error = None
                    break  # success

                except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                    last_error = exc
                    is_retryable = isinstance(exc, httpx.TimeoutException) or (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code in (429, 500, 502, 503, 504)
                    )

                    if is_retryable and attempt < self.max_retries:
                        delay = self.base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"Batch {batch_num}/{total_batches} failed (attempt {attempt}): "
                            f"{exc} — retrying in {delay:.0f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Batch {batch_num}/{total_batches} failed after "
                            f"{attempt} attempt(s): {exc}"
                        )
                        raise Exception(
                            f"Argo embeddings failed at batch {batch_num}/{total_batches}: {exc}"
                        ) from exc

            if last_error is not None:
                raise Exception(f"Argo embeddings failed: {last_error}")

        logger.info(f"Successfully embedded {len(all_embeddings)} documents")
        return all_embeddings
