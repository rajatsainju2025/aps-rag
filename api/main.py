"""
FastAPI application entry point for Agentic GraphRAG API.
Port: 8003
"""
import logging
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import get_settings
from src.utils.logging import setup_logging
from src.ui.service_selector import ServiceSelector

setup_logging()
logger = logging.getLogger(__name__)

# Module-level singletons (populated at startup)
settings = get_settings()
components: Dict[str, Any] = {}
service_selector = ServiceSelector()


def _build_components() -> Dict[str, Any]:
    comps: Dict[str, Any] = {}

    try:
        from src.integrations.argo.llm_client import LLMFactory
        comps["fast_llm"] = LLMFactory.get_routing_llm(settings.argo.username, settings.argo.base_url)
        comps["synthesis_llm"] = LLMFactory.get_synthesis_llm(settings.argo.username, settings.argo.base_url)
        comps["critique_llm"] = LLMFactory.get_critique_llm(settings.argo.username, settings.argo.base_url)
        comps["judge_llm"] = LLMFactory.get_judge_llm(settings.argo.username, settings.argo.base_url)
        # Native-tools clients use Argo /chat/ (vs /streamchat/ for the others).
        # Tools-LLM drives the create_react_agent path; planner-LLM is shared
        # /chat/ infra without bound tools. Init each independently so a failure
        # in one is surfaced loudly and doesn't mask the other.
        try:
            comps["tools_llm"] = LLMFactory.get_tools_llm(
                settings.argo.username,
                settings.argo.base_url,
                model=settings.agent.tools_llm_model,
            )
            logger.info(
                "tools_llm initialized: model=%s endpoint=/chat/",
                settings.agent.tools_llm_model,
            )
        except Exception as exc:
            logger.error(
                "tools_llm INIT FAILED — create_react_agent path disabled. "
                "Cause: %s",
                exc,
            )

        try:
            comps["planner_llm"] = LLMFactory.get_planner_llm(
                settings.argo.username,
                settings.argo.base_url,
                model=settings.agent.planner_llm_model,
            )
            logger.info(
                "planner_llm initialized: model=%s endpoint=/chat/",
                settings.agent.planner_llm_model,
            )
        except Exception as exc:
            logger.error(
                "planner_llm INIT FAILED — orchestrator will fall back to "
                "synthesis_llm (gpt-4o /streamchat/ text-completion). JSON "
                "parsing reliability will degrade. Cause: %s",
                exc,
            )
        logger.info("LLM clients initialized (synthesis=gpt-4o, fast=claude-3-5-haiku, critique=gpt-4o-mini, judge=claude-3-opus)")
    except Exception as exc:
        logger.warning("LLM init failed: %s", exc)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch
        device = settings.vector.embedding_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            device = "cpu"
        comps["embeddings"] = HuggingFaceEmbeddings(
            model_name=settings.vector.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32 if device.startswith("cuda") else 16},
        )
        logger.info("HuggingFaceEmbeddings initialized: model=%s, device=%s", settings.vector.embedding_model, device)
    except Exception as exc:
        logger.warning("Embeddings init failed: %s", exc)

    try:
        from src.integrations.qdrant_client import AgRAGQdrantClient
        comps["qdrant_client"] = AgRAGQdrantClient(
            host=settings.vector.qdrant_host,
            port=settings.vector.qdrant_port,
        )
        logger.info("Qdrant client initialized")
    except Exception as exc:
        logger.warning("Qdrant init failed: %s", exc)

    try:
        from src.integrations.neo4j_client import Neo4jClient
        comps["neo4j_client"] = Neo4jClient(
            uri=settings.graph.neo4j_uri,
            user=settings.graph.neo4j_user,
            password=settings.graph.neo4j_password,
        )
        logger.info("Neo4j client initialized")
    except Exception as exc:
        logger.warning("Neo4j init failed: %s", exc)

    # --- Retrieval components (isolated so one failure doesn't cascade) ---
    # Always use granular-chunking (agrag_gc_* indices/collections).
    _CHUNKING = "granular-chunking"

    try:
        from src.retrieval.vector_retriever import VectorRetriever
        comps["vector_retriever"] = VectorRetriever(
            qdrant_client=comps.get("qdrant_client"),
            embedding_model=comps.get("embeddings"),
            top_k=50,
            chunking_strategy=_CHUNKING,
        )
        logger.info("VectorRetriever initialized [strategy=%s, top_k=50]", _CHUNKING)
    except Exception as exc:
        logger.warning("VectorRetriever init failed: %s", exc)

    try:
        from src.integrations.elasticsearch_client import AgRAGElasticsearchClient
        from src.retrieval.bm25_retriever import BM25Retriever
        es_client = AgRAGElasticsearchClient(
            host=settings.vector.es_host,
            port=settings.vector.es_port,
            username=settings.vector.es_username,
            password=settings.vector.es_password,
        )
        comps["bm25_retriever"] = BM25Retriever(
            es_client=es_client,
            top_k=50,
            chunking_strategy=_CHUNKING,
        )
        logger.info("BM25Retriever initialized [strategy=%s, top_k=50]", _CHUNKING)
    except Exception as exc:
        logger.warning("BM25Retriever init failed: %s", exc)

    try:
        from src.retrieval.graph_retriever import GraphRetriever
        comps["graph_retriever"] = GraphRetriever(
            neo4j_client=comps.get("neo4j_client"),
        )
        logger.info("GraphRetriever initialized")
    except Exception as exc:
        logger.warning("GraphRetriever init failed: %s", exc)

    try:
        from src.retrieval.hybrid_fusion import HybridFusion
        comps["fusion"] = HybridFusion()
        logger.info("HybridFusion initialized")
    except Exception as exc:
        logger.warning("HybridFusion init failed: %s", exc)

    try:
        rtype = getattr(settings.agent, "reranker_type", "jina_v3")
        if rtype == "jina_v3":
            from src.retrieval.reranker import JinaCrossEncoderReranker
            model_name = getattr(
                settings.agent, "reranker_model_name", "jinaai/jina-reranker-v3"
            )
            # Use the same device as embeddings for memory locality
            device = getattr(settings.vector, "embedding_device", None)
            comps["reranker"] = JinaCrossEncoderReranker(
                model_name=model_name,
                top_k=settings.agent.reranker_top_k,
                device=device,
                rrf_blend=float(os.environ.get("APS-RAG_RRF_BLEND", "0.5")),
            )
            logger.info(
                "JinaCrossEncoderReranker initialized (model=%s, device=%s)",
                model_name, device or "auto",
            )
        else:  # rtype == "llm" — legacy LLM-based reranker
            from src.retrieval.reranker import GraphProximityReranker
            comps["reranker"] = GraphProximityReranker(
                llm=comps.get("fast_llm"),
                top_k=settings.agent.reranker_top_k,
            )
            logger.info("GraphProximityReranker initialized (legacy Argo LLM scoring)")
    except Exception as exc:
        logger.warning("Reranker init failed: %s", exc)

    try:
        from src.retrieval.query_enhancement import QueryEnhancer
        combined_llm = LLMFactory.get_haiku_llm(settings.argo.username, settings.argo.base_url)
        comps["query_enhancer"] = QueryEnhancer(
            llm=comps.get("fast_llm"),
            combined_llm=combined_llm,
        )
        logger.info("QueryEnhancer initialized (variants=claude-3-5-sonnet, combined=claude-3-5-haiku)")
    except Exception as exc:
        logger.warning("QueryEnhancer init failed: %s", exc)

    try:
        from src.memory.memory_manager import MemoryManager
        comps["memory_manager"] = MemoryManager()
    except Exception as exc:
        logger.warning("Memory init failed: %s", exc)

    # Register tools for Agentic-RAG tool execution.
    #
    # tool_status mirrors `tools` but tracks the outcome of every attempted
    # registration so /health/pv and operator log scraping can answer
    # "which tools are live right now?" without reading source. Each entry:
    #   {"status": "ok"|"skipped"|"failed", "reason": str?, "details": dict?}
    tools: Dict[str, Any] = {}
    tool_status: Dict[str, Dict[str, Any]] = {}

    def _record(name: str, status: str, reason: str = "", **details):
        """Record per-tool registration outcome and emit one structured log line."""
        entry: Dict[str, Any] = {"status": status}
        if reason:
            entry["reason"] = reason
        if details:
            entry["details"] = details
        tool_status[name] = entry
        detail_str = " ".join(f"{k}={v}" for k, v in details.items())
        if status == "ok":
            logger.info("tool_registered tool=%s status=ok %s", name, detail_str)
        else:
            logger.warning(
                "tool_registered tool=%s status=%s reason=%s %s",
                name, status, reason or "n/a", detail_str,
            )

    try:
        from src.tools.calculator import CalculatorTool
        tools["calculator"] = CalculatorTool()
        _record("calculator", "ok")
    except Exception as exc:
        _record("calculator", "failed", reason=str(exc))

    try:
        from src.tools.code_executor import CodeExecutorTool
        tools["code_executor"] = CodeExecutorTool(
            timeout_seconds=settings.agent.tool_timeout_seconds,
        )
        _record("code_executor", "ok")
    except Exception as exc:
        _record("code_executor", "failed", reason=str(exc))

    try:
        from src.tools.data_analysis import DataAnalysisTool
        tools["data_analysis"] = DataAnalysisTool(
            timeout_seconds=settings.agent.tool_timeout_seconds + 15,
        )
        _record("data_analysis", "ok")
    except Exception as exc:
        _record("data_analysis", "failed", reason=str(exc))

    try:
        from src.tools.graph_search import GraphSearchTool
        neo4j = comps.get("neo4j_client")
        if neo4j:
            tools["graph_search"] = GraphSearchTool(neo4j_client=neo4j)
            _record("graph_search", "ok", backend="neo4j")
        else:
            _record("graph_search", "skipped", reason="neo4j_client not initialized")
    except Exception as exc:
        _record("graph_search", "failed", reason=str(exc))

    try:
        from src.tools.math_reasoning import MathReasoningTool
        tools["math_reasoning"] = MathReasoningTool(llm=comps.get("synthesis_llm"))
        _record("math_reasoning", "ok")
    except Exception as exc:
        _record("math_reasoning", "failed", reason=str(exc))

    # PV tools (channel_finder, pv_read, archiver_read) — MCP-backed with fallback
    if settings.pv.enable_pv_tools:
        _root = Path(__file__).resolve().parent.parent
        mcp_client = None
        try:
            from src.tools.mcp_client import MCPClientManager
            mcp_config = _root / ".mcp.json"
            if mcp_config.exists():
                mcp_client = MCPClientManager(str(mcp_config))
                logger.info("MCP client initialized from %s", mcp_config)
        except Exception as exc:
            logger.warning("MCP client init failed (tools will use direct mode): %s", exc)

        try:
            from src.tools.channel_finder import ChannelFinderTool
            db_path = _root / settings.pv.channel_db_path
            tools["channel_finder"] = ChannelFinderTool(
                db_path=str(db_path),
                in_context_db_path=str(_root / settings.pv.in_context_db_path),
                mcp_client=mcp_client,
            )
            _record(
                "channel_finder", "ok",
                backend=("mcp:controls" if mcp_client else "local_json"),
                db=db_path.name,
            )
        except Exception as exc:
            _record("channel_finder", "failed", reason=str(exc))
        try:
            from src.tools.pv_reader import PVReaderTool
            tools["pv_read"] = PVReaderTool(
                gateway_address=settings.pv.epics_gateway_address,
                gateway_port=settings.pv.epics_gateway_port,
                default_timeout=settings.pv.epics_timeout,
                max_pvs=settings.pv.max_pvs_per_read,
                mcp_client=mcp_client,
            )
            _record(
                "pv_read", "ok",
                backend=("mcp:controls" if mcp_client else "direct_pyepics"),
                gateway=f"{settings.pv.epics_gateway_address}:{settings.pv.epics_gateway_port}",
            )
        except Exception as exc:
            _record("pv_read", "failed", reason=str(exc))
        try:
            from src.tools.archiver_reader import ArchiverReaderTool
            tools["archiver_read"] = ArchiverReaderTool(
                archiver_url=settings.pv.archiver_url,
                timeout=settings.pv.archiver_timeout,
                mcp_client=mcp_client,
            )
            _record(
                "archiver_read", "ok",
                backend=("mcp:archiver" if mcp_client else "direct_rest"),
            )
        except Exception as exc:
            _record("archiver_read", "failed", reason=str(exc))

        # Composite pv_search tool — depends on channel_finder + pv_read above.
        try:
            from src.tools.pv_search import PVSearchTool
            if "channel_finder" in tools and "pv_read" in tools:
                tools["pv_search"] = PVSearchTool(
                    channel_finder=tools["channel_finder"],
                    pv_reader=tools["pv_read"],
                )
                _record("pv_search", "ok", backend="channel_finder+pv_read")
            else:
                missing = [t for t in ("channel_finder", "pv_read") if t not in tools]
                _record(
                    "pv_search", "skipped",
                    reason=f"requires {', '.join(missing)}",
                )
        except Exception as exc:
            _record("pv_search", "failed", reason=str(exc))
    else:
        for name in ("channel_finder", "pv_read", "archiver_read", "pv_search"):
            _record(name, "skipped", reason="pv.enable_pv_tools=false")

    # Image search tool — wraps the standalone image_search pipeline.
    try:
        from src.tools.image_search import ImageSearchTool
        tools["image_search"] = ImageSearchTool()
        _record("image_search", "ok")
    except Exception as exc:
        _record("image_search", "failed", reason=str(exc))

    comps["tools"] = tools
    comps["tool_status"] = tool_status
    if tools:
        logger.info("Tools registered: %s", list(tools.keys()))

    return comps


@asynccontextmanager
async def lifespan(app: FastAPI):
    global components
    logger.info("Starting Agentic GraphRAG API v%s", settings.api.version)
    components = _build_components()
    logger.info("All components initialized")

    # Stage D: image search index is now owned by the persistent ``image`` MCP
    # server subprocess (see mcp_servers/image.py); no API-side pre-warm needed.
    # The OpenWebUI image-search pipeline path still uses _get_image_pipeline()
    # at request time, which lazy-inits in the API process if needed.
    yield
    logger.info("Shutting down Agentic GraphRAG API")
    # Close connections
    try:
        if "neo4j_client" in components:
            components["neo4j_client"].close()
    except Exception:
        pass


app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from api.routes.query import router as query_router
from api.routes.health import router as health_router
from api.routes.graph import router as graph_router
from api.routes.openai_compat import router as openai_compat_router
from api.routes.pipelines import router as pipelines_router
from api.routes.entries import router as entries_router

app.include_router(query_router)
app.include_router(health_router)
app.include_router(graph_router)
app.include_router(openai_compat_router)  # OpenAI-compatible proxy + model list
app.include_router(pipelines_router)       # Naive-RAG & Agentic-RAG pipelines
app.include_router(entries_router)         # WRQ/TEAMS/SDDS entry HTML view


@app.get("/")
async def root():
    return {
        "service": settings.api.title,
        "version": settings.api.version,
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )
