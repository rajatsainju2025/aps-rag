"""
Elasticsearch client for _ag_rag indices.
"""
import logging
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class AgRAGElasticsearchClient:
    """
    Elasticsearch wrapper for the agentic-graph-rag system.
    Operates on _ag_rag indices using BM25 + dense retrieval.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9210,
        username: str = "elastic",
        password: str = "change-me",
    ):
        self.host = host
        self.port = port
        self._client: Optional[Elasticsearch] = None
        self._username = username
        self._password = password

    @property
    def client(self) -> Elasticsearch:
        if not self._client:
            self._client = Elasticsearch(
                f"http://{self.host}:{self.port}",
                basic_auth=(self._username, self._password),
                verify_certs=False,
                ssl_show_warn=False,
                request_timeout=240,
            )
        return self._client

    def bm25_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 100,
        date_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 full-text search with optional date filter.

        When *query* is ``"*"`` or empty the text-match clause is replaced by
        ``match_all`` so that date-only filtering (e.g. daily-summary sweeps)
        returns all documents in the date window instead of zero hits.
        """
        must_clauses: List[Dict] = []

        # Text clause — skip or replace with match_all for wildcard / empty
        if query and query.strip() not in ("*", ""):
            must_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "title^1.5", "author", "system"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            })
        else:
            must_clauses.append({"match_all": {}})

        if date_filter:
            must_clauses.append(
                {"range": {"creation_datetime": date_filter}}
            )

        body = {
            "query": {"bool": {"must": must_clauses}},
            "size": top_k,
            "_source": True,
        }
        try:
            resp = self.client.search(index=index_name, body=body)
            return [
                {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "payload": hit["_source"],
                }
                for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"ES BM25 search failed on {index_name}: {e}")
            return []

    def index_exists(self, index_name: str) -> bool:
        try:
            return self.client.indices.exists(index=index_name).body
        except Exception:
            return False

    def list_indices(self, prefix: str = "") -> List[str]:
        """Return existing index names, optionally filtered by prefix.

        Used to enumerate ingested databases for federated ('all_databases')
        search so any database added under data/raw/ is queryable with no code
        change once it has been ingested.
        """
        try:
            pattern = f"{prefix}*" if prefix else "*"
            resp = self.client.indices.get(index=pattern)
            names = list(getattr(resp, "body", resp).keys())
            # Drop hidden/system indices (leading dot).
            return sorted(n for n in names if not n.startswith("."))
        except Exception as e:
            logger.error(f"Error listing indices: {e}")
            return []

    def health_check(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False
