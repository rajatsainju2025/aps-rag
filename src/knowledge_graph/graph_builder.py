"""
KG builder — orchestrates full knowledge graph construction from all 6 collections.
Reads documents from _ag_rag collections, extracts entities + relations,
persists to Neo4j.
"""
import logging
from typing import List, Dict, Any, Optional

from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor
from .graph_store import GraphStore
from .incremental_updater import IncrementalKGUpdater
from .property_graph_index import PropertyGraphIndexBuilder

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Orchestrates KG construction pipeline:
    1. Load documents from _ag_rag collections (Qdrant payloads)
    2. Delta check via IncrementalKGUpdater
    3. Entity extraction (claude-3-5-sonnet, batch=20)
    4. Relation extraction (claude-3-opus, batch=5)
    5. Store to Neo4j via GraphStore
    6. Build/update LlamaIndex PropertyGraphIndex
    """

    def __init__(self, settings, neo4j_client):
        self.settings = settings

        from src.integrations.argo.llm_client import LLMFactory
        fast_llm = LLMFactory.get_extraction_llm(
            settings.argo.username, settings.argo.base_url
        )
        synthesis_llm = LLMFactory.get_synthesis_llm(
            settings.argo.username, settings.argo.base_url
        )

        self.entity_extractor = EntityExtractor(llm=fast_llm, batch_size=20)
        self.relation_extractor = RelationExtractor(llm=synthesis_llm, batch_size=5)
        self.graph_store = GraphStore(neo4j_client)
        self.updater = IncrementalKGUpdater()
        self.pg_builder = PropertyGraphIndexBuilder(settings)

    def build_from_documents(
        self, documents: List[Dict[str, Any]], incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Build or update KG from a list of documents.

        Args:
            documents: List of document dicts with 'id', 'content', 'collection' keys
            incremental: If True, skip unchanged documents

        Returns:
            Summary dict with node/edge counts
        """
        if incremental:
            to_process, unchanged = self.updater.get_delta(documents)
            logger.info(f"Processing {len(to_process)} new/changed documents")
        else:
            to_process = documents
            logger.info(f"Full rebuild: processing {len(to_process)} documents")

        if not to_process:
            logger.info("No changes detected — KG is up to date")
            return {"nodes": 0, "edges": 0, "documents_processed": 0}

        # Step 1: Extract entities
        logger.info("Step 1/3: Entity extraction...")
        nodes = self.entity_extractor.extract_from_batch(to_process)

        # Step 2: Extract relations
        logger.info("Step 2/3: Relation extraction...")
        triplets = self.relation_extractor.extract_from_batch(to_process)

        # Step 3: Store to Neo4j
        logger.info("Step 3/3: Persisting to Neo4j...")
        self.graph_store.store_nodes(nodes)
        triplet_stats = self.graph_store.store_triplets(triplets)

        # Persistence succeeded — promote pending hashes so the next
        # incremental run skips these docs. (Done here, not in get_delta,
        # so an interrupted build doesn't orphan docs in state.)
        if incremental:
            committed = self.updater.commit_pending()
            logger.info(f"Committed {committed} hashes to sync state")

        # Create full-text indexes after build
        try:
            self.graph_store.neo4j.create_fulltext_index()
        except Exception as e:
            logger.warning(f"Full-text index creation: {e}")

        stats = self.graph_store.graph_stats()
        result = {
            "nodes_extracted": len(nodes),
            "triplets_extracted": len(triplets),
            "neo4j_nodes_stored": triplet_stats["nodes"],
            "neo4j_edges_stored": triplet_stats["edges"],
            "documents_processed": len(to_process),
            "total_kg_nodes": sum(stats.get("nodes", {}).values()),
            "total_kg_edges": sum(stats.get("edges", {}).values()),
        }
        logger.info(f"KG build complete: {result}")
        return result

    def build_all_collections(
        self, collection_docs: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Build KG from all 6 collections."""
        all_documents = []
        for collection_name, docs in collection_docs.items():
            for doc in docs:
                doc["collection"] = collection_name
            all_documents.extend(docs)

        logger.info(
            f"Building unified KG from {len(all_documents)} documents "
            f"across {len(collection_docs)} collections"
        )
        result = self.build_from_documents(all_documents, incremental=True)
        return {**result, **self.finalize_build(all_documents)}

    def finalize_build(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize the KG build with PropertyGraphIndex."""
        result: Dict[str, Any] = {
            "property_graph_index_built": False,
            "property_graph_index_documents": 0,
        }

        if documents:
            try:
                index = self.pg_builder.build(documents)
                result["property_graph_index_built"] = index is not None
                result["property_graph_index_documents"] = len(documents)
            except Exception as e:
                logger.warning(f"PropertyGraphIndex build failed: {e}")
                result["property_graph_index_error"] = str(e)

        return result

    def get_stats(self) -> Dict[str, Any]:
        return self.graph_store.graph_stats()
