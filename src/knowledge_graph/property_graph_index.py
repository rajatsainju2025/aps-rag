"""
LlamaIndex PropertyGraphIndex wrapper.
Builds a PropertyGraphIndex from _ag_rag collections and persists to Neo4j.
"""
import logging
from typing import List, Optional, Any, Dict

logger = logging.getLogger(__name__)


class PropertyGraphIndexBuilder:
    """
    Builds LlamaIndex PropertyGraphIndex backed by Neo4j.
    Uses SimpleLLMPathExtractor with Argo LLM for triplet extraction.
    """

    def __init__(self, settings):
        self.settings = settings
        self._index: Optional[Any] = None

    def _prepare_documents(self, documents: List[Any]) -> List[Any]:
        """Convert normalized raw docs into LlamaIndex Document objects."""
        if not documents:
            return []

        try:
            from llama_index.core import Document
        except ImportError as e:
            logger.error(f"LlamaIndex not installed: {e}")
            raise

        prepared: List[Any] = []
        for doc in documents:
            if hasattr(doc, "text") and hasattr(doc, "metadata"):
                prepared.append(doc)
                continue

            if not isinstance(doc, dict):
                continue

            text = doc.get("content", "") or doc.get("page_content", "")
            if not text:
                continue

            metadata: Dict[str, Any] = {
                "collection": doc.get("collection", ""),
                "title": doc.get("title", ""),
                "author": doc.get("author", ""),
                "creation_datetime": doc.get("creation_datetime", ""),
                "system": doc.get("system", ""),
                "source_file": doc.get("_source_file", ""),
            }
            metadata = {k: v for k, v in metadata.items() if v not in (None, "")}

            doc_id = str(doc.get("id", "")).strip()
            if doc_id:
                prepared.append(Document(text=text, metadata=metadata, id_=doc_id))
            else:
                prepared.append(Document(text=text, metadata=metadata))

        return prepared

    def build(self, documents: List[Any]) -> Any:
        """
        Build PropertyGraphIndex from a list of LlamaIndex Document objects.
        Persists to Neo4j via Neo4jPropertyGraphStore.
        """
        try:
            from llama_index.core import PropertyGraphIndex
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
        except ImportError as e:
            logger.error(f"LlamaIndex not installed: {e}")
            raise

        prepared_documents = self._prepare_documents(documents)
        if not prepared_documents:
            logger.info("No documents available for PropertyGraphIndex build")
            return None

        neo4j_store = Neo4jPropertyGraphStore(
            username=self.settings.graph.neo4j_user,
            password=self.settings.graph.neo4j_password,
            url=self.settings.graph.neo4j_uri,
        )

        # Import Argo LLM for LlamaIndex
        from src.integrations.argo.llm_client import LLMFactory
        argo_llm = LLMFactory.get_synthesis_llm(
            self.settings.argo.username, self.settings.argo.base_url
        )

        extractor = SimpleLLMPathExtractor(
            llm=argo_llm,
            possible_entities=[
                "Person", "System", "Fault", "Component",
                "Procedure", "Concept", "Study", "SDDSTool",
            ],
            possible_relations=[
                "AUTHORED_BY", "WORKS_ON", "ABOUT_SYSTEM", "HAS_FAULT",
                "HAS_COMPONENT", "RESOLVED_BY", "CAUSED_BY", "RELATED_TO",
                "USES_TOOL", "PART_OF_STUDY", "FOLLOWS_PROCEDURE",
            ],
            num_workers=2,
        )

        logger.info(f"Building PropertyGraphIndex from {len(prepared_documents)} documents...")
        self._index = PropertyGraphIndex.from_documents(
            prepared_documents,
            property_graph_store=neo4j_store,
            kg_extractors=[extractor],
            show_progress=True,
        )
        logger.info("PropertyGraphIndex built and persisted to Neo4j")
        return self._index

    def load_existing(self) -> Any:
        """Load existing PropertyGraphIndex from Neo4j."""
        try:
            from llama_index.core import PropertyGraphIndex
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
        except ImportError as e:
            raise ImportError(f"LlamaIndex not installed: {e}")

        neo4j_store = Neo4jPropertyGraphStore(
            username=self.settings.graph.neo4j_user,
            password=self.settings.graph.neo4j_password,
            url=self.settings.graph.neo4j_uri,
        )
        self._index = PropertyGraphIndex.from_existing(
            property_graph_store=neo4j_store
        )
        logger.info("Loaded existing PropertyGraphIndex from Neo4j")
        return self._index

    @property
    def index(self) -> Optional[Any]:
        return self._index
