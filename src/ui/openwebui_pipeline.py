"""
OpenWebUI Pipe plugin for Agentic GraphRAG.
"""
import logging
from typing import Any, Dict, Generator, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Pipeline:
    """OpenWebUI Pipe — routes queries through the Agentic GraphRAG system."""

    class Valves(BaseModel):
        service_mode: str = Field(default="naive_rag", description="Service mode")
        collection: str = Field(default="all_databases", description="Collection to search")
        top_k: int = Field(default=10, description="Number of documents to retrieve")
        enable_web_search: bool = Field(default=False, description="Enable web search tool")
        enable_code_execution: bool = Field(default=False, description="Enable code execution tool")
        argo_username: str = Field(default="", description="Argo API username")
        neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
        neo4j_password: str = Field(default="change-me", description="Neo4j password")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._components: Optional[Dict[str, Any]] = None
        self._settings = None
        self._service_selector = None

    def _get_settings(self):
        if self._settings is None:
            from config.settings import get_settings
            self._settings = get_settings()
            if self.valves.argo_username:
                self._settings.argo.username = self.valves.argo_username
            if self.valves.neo4j_uri:
                self._settings.graph.neo4j_uri = self.valves.neo4j_uri
            if self.valves.neo4j_password:
                self._settings.graph.neo4j_password = self.valves.neo4j_password
        return self._settings

    def _build_components(self) -> Dict[str, Any]:
        """Lazily initialize all integration components."""
        settings = self._get_settings()
        components: Dict[str, Any] = {}

        try:
            from src.integrations.argo.llm_client import LLMFactory
            components["fast_llm"] = LLMFactory.get_routing_llm(
                settings.argo.username, settings.argo.base_url
            )
            components["synthesis_llm"] = LLMFactory.get_synthesis_llm(
                settings.argo.username, settings.argo.base_url
            )
        except Exception as exc:
            logger.warning("LLM init failed: %s", exc)

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            import torch
            device = settings.vector.embedding_device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available — falling back to CPU")
                device = "cpu"
            components["embeddings"] = HuggingFaceEmbeddings(
                model_name=settings.vector.embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32 if device.startswith("cuda") else 16},
            )
            logger.info("HuggingFaceEmbeddings: model=%s, device=%s", settings.vector.embedding_model, device)
        except Exception as exc:
            logger.warning("Embeddings init failed: %s", exc)

        try:
            from src.integrations.qdrant_client import AgRAGQdrantClient
            components["qdrant_client"] = AgRAGQdrantClient(
                host=settings.vector.qdrant_host,
                port=settings.vector.qdrant_port,
            )
        except Exception as exc:
            logger.warning("Qdrant init failed: %s", exc)

        try:
            from src.integrations.neo4j_client import Neo4jClient
            components["neo4j_client"] = Neo4jClient(
                uri=settings.graph.neo4j_uri,
                user=settings.graph.neo4j_user,
                password=settings.graph.neo4j_password,
            )
        except Exception as exc:
            logger.warning("Neo4j init failed: %s", exc)

        try:
            from src.retrieval.vector_retriever import VectorRetriever
            components["vector_retriever"] = VectorRetriever(
                qdrant_client=components.get("qdrant_client"),
                embedding_model=components.get("embeddings"),
                top_k=settings.agent.top_k,
                chunking_strategy="granular-chunking",
            )
        except Exception as exc:
            logger.warning("VectorRetriever init failed: %s", exc)

        try:
            from elasticsearch import Elasticsearch
            from src.retrieval.bm25_retriever import BM25Retriever
            es_client = Elasticsearch(
                f"http://{settings.vector.es_host}:{settings.vector.es_port}",
                basic_auth=(settings.vector.es_username, settings.vector.es_password),
            )
            components["bm25_retriever"] = BM25Retriever(
                es_client=es_client,
                top_k=settings.agent.top_k,
                chunking_strategy="granular-chunking",
            )
        except Exception as exc:
            logger.warning("BM25Retriever init failed: %s", exc)

        try:
            from src.retrieval.graph_retriever import GraphRetriever
            components["graph_retriever"] = GraphRetriever(
                neo4j_client=components.get("neo4j_client"),
            )
        except Exception as exc:
            logger.warning("GraphRetriever init failed: %s", exc)

        try:
            from src.retrieval.hybrid_fusion import HybridFusion
            components["fusion"] = HybridFusion()
        except Exception as exc:
            logger.warning("HybridFusion init failed: %s", exc)

        try:
            from src.retrieval.reranker import GraphProximityReranker
            components["reranker"] = GraphProximityReranker(
                llm=components.get("fast_llm"),
                top_k=settings.agent.reranker_top_k,
            )
        except Exception as exc:
            logger.warning("Reranker init failed: %s", exc)

        try:
            from src.retrieval.query_enhancement import QueryEnhancer
            components["query_enhancer"] = QueryEnhancer()
        except Exception as exc:
            logger.warning("QueryEnhancer init failed: %s", exc)

        try:
            from src.memory.memory_manager import MemoryManager
            components["memory_manager"] = MemoryManager()
        except Exception as exc:
            logger.warning("Memory init failed: %s", exc)

        components["tools"] = {}
        return components

    def _get_service(self):
        if self._components is None:
            self._components = self._build_components()
        if self._service_selector is None:
            from src.ui.service_selector import ServiceSelector
            self._service_selector = ServiceSelector()
        return self._service_selector.get_service(
            self.valves.service_mode,
            self._get_settings(),
            self._components,
        )

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
    ) -> Generator[str, None, None]:
        session_id = body.get("session_id", "default")
        try:
            service = self._get_service()
            result = service.query(
                query=user_message,
                session_id=session_id,
                selected_databases=[self.valves.collection],
                top_k=self.valves.top_k,
            )
            answer = result.get("answer", "No answer generated.")
            # Yield in chunks for streaming experience
            chunk_size = 100
            for i in range(0, len(answer), chunk_size):
                yield answer[i : i + chunk_size]
        except Exception as exc:
            logger.error("Pipeline error: %s", exc)
            yield f"Error processing query: {exc}"
