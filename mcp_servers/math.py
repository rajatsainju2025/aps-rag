"""MCP Server: Math Reasoning — step-by-step LLM-backed problem solving.

Provides one tool:
  - math_reasoning: Argo-LLM-driven scratchpad solution + boxed final answer

Why a dedicated server (Stage D): the underlying LLM client (``ArgoLLMClientAG``
or ``NativeToolsArgoLLM``) holds Argo creds + an HTTP session. Constructing it
per call would re-open the connection pool every time. With persistent MCP
sessions (see ``src/tools/mcp_client.py``), the LLM client is built **once**
inside this server subprocess and reused across calls.

Args read from env (set by .mcp.json):
  - ARGO_USERNAME (required)
  - ARGO_BASE_URL (defaults to settings.llm.argo_base_url at startup if unset)
  - ARGO_MATH_MODEL (defaults to gpt-4o)

Run: python -m mcp_servers.math
"""
from __future__ import annotations

import logging
import os

from mcp.server.fastmcp import FastMCP

from mcp_servers._common import add_project_root_to_path, err, mcp_tool_safe, ok

logger = logging.getLogger(__name__)
add_project_root_to_path()

mcp = FastMCP(
    "math",
    instructions=(
        "Step-by-step mathematical reasoning via Argo LLM. Returns a "
        "scratchpad-style solution with intermediate steps, units check, and "
        "a clearly-boxed final answer. Use for accelerator physics, beam "
        "dynamics, and any problem that benefits from explicit chain-of-thought."
    ),
)


_MATH_PROMPT = """# Role & Persona
You are an **Expert Mathematician and Physicist** specializing in accelerator \
physics and beam dynamics at the facility.

# Task
Solve the following problem with rigorous step-by-step reasoning.

# Chain-of-Thought Solution Protocol (MANDATORY)
Work through the problem inside a <scratchpad> block:

<scratchpad>
Step 1 — Problem Parsing: Identify the known quantities, unknowns, and \
applicable physical laws or mathematical relationships.
Step 2 — Method Selection: Choose the solution approach (algebraic, numerical, \
calculus-based, etc.) and justify why.
Step 3 — Intermediate Calculations: Show each arithmetic or algebraic step \
explicitly. Use LaTeX notation for equations.
Step 4 — Units & Dimensional Analysis: Verify units at each step. Flag any \
unit mismatches.
Step 5 — Sanity Check: Is the result physically reasonable? Compare to known \
orders of magnitude if possible.
</scratchpad>

# Problem
{problem}

# Output
After the scratchpad, present the solution clearly:
- State each step with its intermediate result.
- Use LaTeX notation (e.g., $E = mc^2$) for all equations.
- Clearly box or bold the final answer.
- If the problem is ambiguous or missing information, state what assumptions you made."""


_LLM_CLIENT = None  # cached for the lifetime of this MCP server subprocess


def _get_llm():
    """Lazy-init the Argo LLM client so server start is fast and creds are
    only required when the tool is first called. Cached for reuse across calls
    within this subprocess (the Stage D win — no per-call re-init)."""
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT
    username = os.environ.get("ARGO_USERNAME", "").strip()
    base_url = os.environ.get("ARGO_BASE_URL", "").strip()
    model = os.environ.get("ARGO_MATH_MODEL", "gpt-4o").strip() or "gpt-4o"
    if not username or not base_url:
        # Fall back to settings.argo.* — may also be empty in offline test envs.
        from config.settings import settings as _s
        username = username or _s.argo.username
        base_url = base_url or _s.argo.base_url
    if not username or not base_url:
        raise RuntimeError(
            "math_reasoning requires ARGO_USERNAME + ARGO_BASE_URL "
            "(set in .mcp.json env or shell env)"
        )
    from src.integrations.argo.llm_client import ArgoLLMClientAG
    _LLM_CLIENT = ArgoLLMClientAG(
        model=model,
        username=username,
        base_url=base_url,
        temperature=0.0,
        top_p=0.9,
        max_tokens=4096,
    )
    logger.info("math_reasoning: Argo LLM client initialized (model=%s)", model)
    return _LLM_CLIENT


@mcp.tool()
@mcp_tool_safe
def math_reasoning(problem: str) -> str:
    """Step-by-step mathematical reasoning with LLM (scratchpad + boxed answer).

    Args:
      problem: Free-form math/physics problem statement. Include units and any
        known constants or constraints. Ambiguity is fine — the LLM will state
        assumptions explicitly.

    Examples:
      math_reasoning("If beam current is 200 mA and revolution period 3.68 us, how many electrons per turn?")
      math_reasoning("Compute the synchrotron tune for a ring with V_RF=9 MV, f_RF=352 MHz, alpha_p=2.7e-4, E=7 GeV")
    """
    if not problem or not problem.strip():
        return err("problem is empty", tool="math_reasoning",
                   hint="Pass a non-empty problem statement.")
    llm = _get_llm()
    prompt = _MATH_PROMPT.format(problem=problem)
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        text = response.content
    else:
        text = str(response)
    return ok(result=text, model=os.environ.get("ARGO_MATH_MODEL", "gpt-4o"))


if __name__ == "__main__":
    mcp.run()
