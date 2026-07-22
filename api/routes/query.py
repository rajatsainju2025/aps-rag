"""
Query router — /query endpoints.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    service_mode: str = Field(default="naive_rag", description="Service mode: naive_rag|agentic_rag|daily_summary|troubleshooting|studies_copilot")
    collection: str = Field(default="all_databases", description="Collection to search")
    session_id: str = Field(default="default", description="Session identifier")
    top_k: int = Field(default=10, ge=1, le=200, description="Documents to retrieve")
    retrieval_strategy: Optional[str] = Field(
        default=None,
        description="Eval override — pin retrieval strategy, bypassing the router: "
        "VECTOR_BM25 | HYBRID_3WAY | KG_ONLY | VECTOR_ONLY | BM25_ONLY",
    )


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    sources: List[Any] = []
    service_mode: str
    latency_ms: float
    reasoning_trace: Optional[List[str]] = None
    correction_rounds: Optional[int] = None
    fault_chain: Optional[List[Any]] = None
    code_output: Optional[str] = None
    # Pre-flight clarification gate. Set when the system needs the user to
    # disambiguate scope before retrieval (e.g. "is part of" → participant
    # vs mention?). Null on normal answers. Shape:
    #   {"question": str, "options": [{"id","label","interpretation"}],
    #    "original_query": str}
    clarification: Optional[Dict[str, Any]] = None


def get_service_selector():
    from api.main import service_selector
    return service_selector


def get_settings():
    from api.main import settings
    return settings


def get_components():
    from api.main import components
    return components


@router.post("", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    selector=Depends(get_service_selector),
    settings=Depends(get_settings),
    comps=Depends(get_components),
) -> QueryResponse:
    try:
        from src.services.pv_short_circuit import try_pv_short_circuit
        pv_result = try_pv_short_circuit(
            request.query, comps, request.service_mode, request.session_id,
        )
        if pv_result is not None:
            return QueryResponse(**{k: v for k, v in pv_result.items() if k in QueryResponse.model_fields})

        service = selector.get_service(request.service_mode, settings, comps)
        result = service.query(
            query=request.query,
            session_id=request.session_id,
            selected_databases=[request.collection],
            top_k=request.top_k,
            retrieval_strategy=request.retrieval_strategy,
        )
        return QueryResponse(**{k: v for k, v in result.items() if k in QueryResponse.model_fields})
    except Exception as exc:
        logger.error("Query error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
