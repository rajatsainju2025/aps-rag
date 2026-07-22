"""
Knowledge-graph build helper.

Populates the Neo4j knowledge graph from a list of normalized documents by
running the entity + relation extractors and persisting nodes/triplets. This is
the single entry point shared by `scripts/ingest.py --build-kg` and
`scripts/build_kg.py`.

Requires an LLM backend (entity/relation extraction). The LLM is selected via
the `llm.provider` config key (see src/integrations/llm/ and config).
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def build_kg(settings, documents: List[Dict[str, Any]], incremental: bool = True) -> Dict[str, Any]:
    """Build/update the Neo4j knowledge graph from `documents`.

    Args:
        settings: AppSettings (provides graph.* connection + llm.* backend).
        documents: normalized doc dicts (id, content, collection, ...).
        incremental: skip documents whose content is unchanged since last build.

    Returns:
        Summary dict with node/edge/document counts.
    """
    from src.integrations.neo4j_client import Neo4jClient
    from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder

    neo4j = Neo4jClient(
        uri=settings.graph.neo4j_uri,
        user=settings.graph.neo4j_user,
        password=settings.graph.neo4j_password,
    )
    builder = KnowledgeGraphBuilder(settings, neo4j)
    logger.info("Building knowledge graph from %d documents (incremental=%s)", len(documents), incremental)
    return builder.build_from_documents(documents, incremental=incremental)
