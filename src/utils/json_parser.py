"""
Robust JSON parsing utility with LLM-based repair fallback.

Used by router, corrective agent, troubleshooting agent, and other
agents that require structured JSON output from LLM calls.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Tags to strip before extracting JSON
_STRIP_PATTERNS = [
    r"<scratchpad>.*?</scratchpad>",
    r"<thinking>.*?</thinking>",
    r"<routing_analysis>.*?</routing_analysis>",
    r"<plan>.*?</plan>",
    r"<analysis>.*?</analysis>",
]

JSON_REPAIR_PROMPT = """\
The following text was supposed to be valid JSON but failed to parse.
Fix ONLY the JSON syntax errors (missing quotes, trailing commas, unescaped \
characters) without changing the content.
Return ONLY the corrected JSON — no explanation, no markdown fences.

Broken JSON:
{broken_json}"""


def strip_cot_blocks(text: str) -> str:
    """Remove chain-of-thought XML blocks from LLM output."""
    for pattern in _STRIP_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    # Fix double-brace wrapping: {{ ... }} → { ... }
    text = re.sub(r"^\s*\{\{", "{", text.strip())
    text = re.sub(r"\}\}\s*$", "}", text.strip())
    return text.strip()


def extract_json_object(text: str) -> str:
    """Extract the first complete JSON object or array from text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```\w*", "", text).strip()

    # Try object first, then array
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = text.find(open_char)
        if start < 0:
            continue
        # Find matching close using depth counting
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return ""


def parse_json_with_repair(
    raw: str,
    llm=None,
    context: str = "",
) -> Optional[Union[Dict, List]]:
    """
    Parse JSON from LLM output with optional repair fallback.

    Steps:
    1. Strip CoT blocks (<scratchpad>, <thinking>, etc.)
    2. Extract JSON substring
    3. Parse with json.loads
    4. On failure: send to LLM for repair (if llm provided)
    5. On repair failure: return None

    Args:
        raw: Raw LLM output string
        llm: Optional LLM for repair fallback (uses invoke(repair_prompt))
        context: Short description for logging (e.g., "router", "corrective")

    Returns:
        Parsed dict/list, or None if all parsing attempts fail.
    """
    # Step 1: Strip CoT blocks
    cleaned = strip_cot_blocks(raw)

    # Step 2: Extract JSON substring
    json_str = extract_json_object(cleaned)
    if not json_str:
        logger.warning(f"[{context}] No JSON object/array found in response")
        if llm:
            return _repair_json(raw[:2000], llm, context)
        return None

    # Step 3: Parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[{context}] JSON parse error: {e}")
        if llm:
            return _repair_json(json_str, llm, context)
        return None


def _repair_json(
    broken: str, llm, context: str
) -> Optional[Union[Dict, List]]:
    """Ask LLM to fix malformed JSON. Returns parsed result or None."""
    try:
        from langchain_core.messages import HumanMessage
        repair_prompt = JSON_REPAIR_PROMPT.format(broken_json=broken[:3000])
        response = llm.invoke([HumanMessage(content=repair_prompt)])
        repaired_raw = response.content if hasattr(response, "content") else str(response)
        repaired_str = extract_json_object(repaired_raw.strip())
        if repaired_str:
            result = json.loads(repaired_str)
            logger.info(f"[{context}] JSON repair succeeded")
            return result
    except Exception as e:
        logger.warning(f"[{context}] JSON repair failed: {e}")
    return None
