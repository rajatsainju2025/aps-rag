"""
Qdrant client for _ag_rag collections.
"""
import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class AgRAGQdrantClient:
    """
    Qdrant wrapper for the agentic-graph-rag system.
    Operates exclusively on _ag_rag collections.
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if not self._client:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            return False

    def list_collections(self, prefix: str = "") -> List[str]:
        """Return existing Qdrant collection names, optionally filtered by prefix.

        Used to enumerate ingested databases for federated ('all_databases')
        search so any database added under data/raw/ is queryable with no code
        change once it has been ingested.
        """
        try:
            names = [c.name for c in self.client.get_collections().collections]
            return sorted(n for n in names if n.startswith(prefix)) if prefix else sorted(names)
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        date_range: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search in a _ag_rag collection.

        Args:
            filters: Exact-match payload filters  {key: value}.
            date_range: Date-window filter applied to ``creation_datetime``.
                        Expects ``{"gte": "ISO-datetime", "lte": "ISO-datetime"}``.
        """
        conditions: List[Any] = []

        if filters:
            for key, value in filters.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        if date_range:
            conditions.append(
                models.FieldCondition(
                    key="creation_datetime",
                    range=models.DatetimeRange(
                        gte=date_range.get("gte"),
                        lte=date_range.get("lte"),
                    ),
                )
            )

        qdrant_filter = models.Filter(must=conditions) if conditions else None

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload or {},
            }
            for r in results
        ]

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
