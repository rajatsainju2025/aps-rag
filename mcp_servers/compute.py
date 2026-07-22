"""MCP Server: Compute — sandboxed computation tools.

Provides three tools:
  - calculator: SymPy-backed expression evaluation
  - code_executor: Sandboxed Python execution (subprocess-isolated)
  - data_analysis: Pandas analysis over whitelisted the facility data files

All three are stateless within the server process (calculator has no state;
code_executor and data_analysis spawn their own subprocess per call so
state lives there, not in this server). The data_analysis tool's schema
probe runs lazily on first call and caches the result for the lifetime of
the server subprocess (~80% latency reduction on warm calls).

Run: python -m mcp_servers.compute
"""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from mcp_servers._common import add_project_root_to_path, err, mcp_tool_safe, ok

logger = logging.getLogger(__name__)
add_project_root_to_path()

mcp = FastMCP(
    "compute",
    instructions=(
        "Sandboxed computation: math expressions (calculator), Python code "
        "execution (code_executor), and pandas analysis over whitelisted the facility "
        "data files (data_analysis — work_requests, entry_logbook, teams_messages)."
    ),
)


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------
@mcp.tool()
@mcp_tool_safe
def calculator(expression: str) -> str:
    """Evaluate a math expression using SymPy.

    Examples:
      calculator("2+2")
      calculator("sqrt(2) * pi")
      calculator("integrate(x**2, (x, 0, 1))")
    """
    # Call the module-level _impl to bypass BaseTool._run (which now dispatches
    # back through MCP — would recurse).
    from src.tools.calculator import calculator_impl
    return ok(result=calculator_impl(expression), expression=expression)


# ---------------------------------------------------------------------------
# code_executor
# ---------------------------------------------------------------------------
@mcp.tool()
@mcp_tool_safe
def code_executor(code: str) -> str:
    """Execute Python code in a sandboxed subprocess (30s timeout).

    Banned patterns: ``__import__``, ``os.system``, ``subprocess``, ``eval(``,
    ``exec(``, ``shutil``. Returns ``{"stdout", "stderr", "returncode"}``.

    Examples:
      code_executor("print(sum(range(100)))")
      code_executor("import math; print(math.pi)")
    """
    from src.tools.code_executor import code_executor_impl
    return ok(result=code_executor_impl(code))


# ---------------------------------------------------------------------------
# data_analysis
# ---------------------------------------------------------------------------
@mcp.tool()
@mcp_tool_safe
def data_analysis(analysis_code: str, data_source: str) -> str:
    """Run pandas analysis on whitelisted the facility data.

    Args:
      analysis_code: pandas code; the DataFrame ``df`` is pre-loaded.
        Available preamble constants: ``TODAY`` (date), ``THIS_YEAR`` (int),
        ``YESTERDAY``, ``LAST_WEEK_START``. Always ``print()`` the answer.
      data_source: one of ``work_requests``, ``teams_messages``, ``entry_logbook``.

    See the underlying tool's description for the full schema cheatsheet
    (column names, value enums, date ranges). Use this tool for any
    counting / filtering / aggregation question — the LLM is an unreliable
    counter and chunked retrieval undercounts.
    """
    from src.tools.data_analysis import data_analysis_impl, WHITELISTED_DATA_FILES
    if data_source not in WHITELISTED_DATA_FILES:
        return err(
            f"Unknown data_source {data_source!r}. "
            f"Available: {list(WHITELISTED_DATA_FILES.keys())}",
            tool="data_analysis",
            hint="Pick one of work_requests | teams_messages | entry_logbook.",
        )
    raw = data_analysis_impl(analysis_code=analysis_code, data_source=data_source)
    return ok(result=raw, data_source=data_source)


if __name__ == "__main__":
    mcp.run()
