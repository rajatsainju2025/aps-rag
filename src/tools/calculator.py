"""
LangChain Tool: CalculatorTool — symbolic and numeric math using SymPy.
"""
import logging
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    import sympy
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False
    logger.warning("SymPy not installed; calculator will use eval() fallback")

_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
}

try:
    import math
    _SAFE_GLOBALS.update({k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
except ImportError:
    pass


class _CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")


def calculator_impl(expression: str) -> str:
    """Pure SymPy/eval logic — called by both the LangChain shim (via MCP) AND
    the compute MCP server directly. Extracting this as a module function
    lets the MCP server skip the BaseTool wrapper without infinite recursion
    (BaseTool._run now dispatches to MCP, which would call _run again, etc.).
    """
    if _SYMPY_AVAILABLE:
        try:
            result = sympy.sympify(expression)
            simplified = sympy.simplify(result)
            return str(simplified)
        except Exception as sympy_exc:
            logger.debug("SymPy failed (%s), trying eval fallback", sympy_exc)

    # Fallback: restricted eval
    try:
        result = eval(expression, _SAFE_GLOBALS)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Perform symbolic and numeric mathematical calculations"
    args_schema: Type[BaseModel] = _CalculatorInput

    def _run(self, expression: str) -> str:
        # Stage C: route through the MCP `compute` server so external clients
        # (OpenWebUI, Cursor, etc.) can invoke the same tool, and the agent
        # gets a uniform error envelope on transport failures.
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync(
                "compute", "calculator", {"expression": expression}
            )
        )
