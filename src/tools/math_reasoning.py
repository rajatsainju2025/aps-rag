"""
LangChain Tool: MathReasoningTool — step-by-step math reasoning via LLM.

Stage D: ``_run`` dispatches through MCP (``math/math_reasoning``) so the
Argo LLM client lives in ONE persistent subprocess (HTTP connection pool +
auth state warm across calls). The ``llm`` constructor argument is kept for
backward compatibility with [api/main.py](../../api/main.py) wiring but is
no longer consulted — the math MCP server reads creds from its own env
(``ARGO_USERNAME``, ``ARGO_BASE_URL``, ``ARGO_MATH_MODEL``).
"""
import logging
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _MathReasoningInput(BaseModel):
    problem: str = Field(description="Mathematical problem to solve with step-by-step reasoning")


class MathReasoningTool(BaseTool):
    name: str = "math_reasoning"
    description: str = "Step-by-step mathematical reasoning with LLM (scratchpad + boxed final answer)."
    args_schema: Type[BaseModel] = _MathReasoningInput
    llm: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: Any = None, **kwargs: Any) -> None:
        # Stage D: ``llm`` constructor arg is kept for backward compat (api/main.py
        # wires it at startup) but no longer used — the MCP math server owns the
        # LLM client lifecycle inside its persistent subprocess.
        super().__init__(**kwargs)
        object.__setattr__(self, "llm", llm)

    def _run(self, problem: str) -> str:
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync(
                "math", "math_reasoning", {"problem": problem}
            )
        )
