"""Render LangChain BaseTool args_schemas as compact JSON for prompt injection.

Replaces the legacy free-text `# Available Tools` block in SYNTHESIS_PROMPT
with structured JSON Schema descriptions so the LLM can produce well-formed
TOOL_CALL directives even on Argo (which has no native tool-use API).

Output is a markdown block sized for the Argo context budget:
~200 chars per tool × N tools.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping


def _compact_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Strip noisy fields from a Pydantic model_json_schema() result.

    Keeps `properties`, `required`, and `description`; drops `$defs`, `title`,
    and per-field titles to keep the prompt tight.
    """
    out: Dict[str, Any] = {}
    if "description" in schema:
        out["description"] = schema["description"]
    props = schema.get("properties") or {}
    if props:
        out["properties"] = {
            name: {k: v for k, v in field.items() if k != "title"}
            for name, field in props.items()
        }
    if schema.get("required"):
        out["required"] = schema["required"]
    return out


def _tool_call_format(tool_name: str, schema: Dict[str, Any]) -> str:
    """Build a one-line example TOOL_CALL: directive from a tool's schema.

    For single-arg tools (most), shows `TOOL_CALL: name(<value>)`. For multi-arg
    tools, shows `TOOL_CALL: name(<arg1>||<arg2>)` matching the existing
    `||`-separated convention used by data_analysis and archiver_read.
    """
    props = list((schema.get("properties") or {}).keys())
    if not props:
        return f"TOOL_CALL: {tool_name}()"
    if len(props) == 1:
        return f"TOOL_CALL: {tool_name}(<{props[0]}>)"
    return f"TOOL_CALL: {tool_name}(" + "||".join(f"<{p}>" for p in props) + ")"


def render_tool_schemas(tools: Mapping[str, Any]) -> str:
    """Render every registered tool as a markdown block with JSON Schema.

    Args:
        tools: dict mapping tool_name → LangChain BaseTool instance.

    Returns:
        Markdown string with one `## Tool: <name>` section per tool, each
        containing description, JSON schema, and an example TOOL_CALL line.
        Returns an empty string when no tools are registered.
    """
    if not tools:
        return ""

    sections = ["# Available Tools (emit `TOOL_CALL: ...` on its own line)"]
    for name, tool in tools.items():
        description = getattr(tool, "description", "") or ""
        description = description.strip().splitlines()[0] if description else ""

        schema_obj: Dict[str, Any] = {}
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            try:
                if hasattr(args_schema, "model_json_schema"):
                    raw = args_schema.model_json_schema()
                else:
                    raw = args_schema.schema()  # pydantic v1 fallback
                schema_obj = _compact_schema(raw)
            except Exception:
                schema_obj = {}

        sections.append(f"\n## Tool: `{name}`")
        if description:
            sections.append(description)
        if schema_obj:
            sections.append("```json")
            sections.append(json.dumps(schema_obj, separators=(",", ": ")))
            sections.append("```")
        sections.append(f"Example: `{_tool_call_format(name, schema_obj)}`")

    return "\n".join(sections)
