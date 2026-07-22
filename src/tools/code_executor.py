"""
LangChain Tool: CodeExecutorTool — safe Python code execution in subprocess.
"""
import json
import logging
import subprocess
import sys
import textwrap
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_BANNED_PATTERNS = [
    "os.system",
    "__import__",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "eval(",
    "exec(",
    "open(",
    "shutil",
]


class _CodeExecutorInput(BaseModel):
    code: str = Field(description="Python code to execute")


def _check_safety(code: str) -> None:
    """Reject code containing banned patterns. Strips comments + string
    literals first so a docstring mentioning 'subprocess' doesn't trip."""
    import re as _re
    stripped = _re.sub(r'#[^\n]*', '', code)
    stripped = _re.sub(r'""".*?"""', '', stripped, flags=_re.DOTALL)
    stripped = _re.sub(r"'''.*?'''", '', stripped, flags=_re.DOTALL)
    stripped = _re.sub(r'"[^"\n]*"', '', stripped)
    stripped = _re.sub(r"'[^'\n]*'", '', stripped)
    for pattern in _BANNED_PATTERNS:
        if pattern in stripped:
            raise ValueError(f"Forbidden pattern detected: '{pattern}'")


def code_executor_impl(code: str, timeout_seconds: int = 30) -> str:
    """Pure subprocess code-execution logic — called by both the LangChain
    shim (via MCP) AND the compute MCP server directly. Module-level so the
    MCP server can bypass the BaseTool wrapper without infinite recursion.
    """
    try:
        _check_safety(code)
    except ValueError as exc:
        return json.dumps({"stdout": "", "stderr": str(exc), "returncode": 1})

    code = textwrap.dedent(code).strip()
    wrapper = "\n".join([
        "import sys",
        "import math",
        "import json",
        "import re",
        "from datetime import datetime",
        "",
        code,
        "",
    ])
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return json.dumps({
            "stdout": proc.stdout[:4000],
            "stderr": proc.stderr[:2000],
            "returncode": proc.returncode,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "stdout": "",
            "stderr": f"Code execution timed out after {timeout_seconds}s",
            "returncode": -1,
        })
    except Exception as exc:
        logger.error("Code execution error: %s", exc)
        return json.dumps({"stdout": "", "stderr": str(exc), "returncode": 1})


class CodeExecutorTool(BaseTool):
    name: str = "code_executor"
    description: str = "Execute Python code safely"
    args_schema: Type[BaseModel] = _CodeExecutorInput
    timeout_seconds: int = 30

    def _check_safety(self, code: str) -> None:
        # Kept for backward compat — some tests / external callers reach for it.
        _check_safety(code)

    def _run(self, code: str) -> str:
        # Stage C: route through MCP `compute` server.
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync(
                "compute", "code_executor", {"code": code}
            )
        )
