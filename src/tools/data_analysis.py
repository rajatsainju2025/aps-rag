"""
LangChain Tool: DataAnalysisTool — safe analysis on whitelisted the facility data files.

Used by Agentic-RAG when the query requires counting, filtering, or
statistical analysis on structured data (work requests, logbook entries, etc.).

The LLM generates a pandas code snippet; this tool executes it in a
sandboxed subprocess with access ONLY to the whitelisted data files.
"""
import json
import logging
import subprocess
import sys
import textwrap
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Data paths the tool is allowed to read, derived from the folder-per-database
# layout under data/raw/. This is a SITE-SPECIFIC analysis tool (counting /
# aggregation via pandas) and is disabled by default — see config `agent.tools`.
_AG_RAG_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DATA_ROOT = _AG_RAG_ROOT / "data" / "raw"


def _discover_data_files() -> Dict[str, str]:
    """Map each discovered database name to its data/raw/<db>/ folder so the
    analysis tool can operate over structured records dropped there."""
    files: Dict[str, str] = {}
    if _RAW_DATA_ROOT.exists():
        for child in sorted(_RAW_DATA_ROOT.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                files[child.name] = str(child)
    return files


WHITELISTED_DATA_FILES: Dict[str, str] = _discover_data_files()

# Cap how many distinct values to enumerate per categorical column in the
# tool's description (kept short to avoid blowing up the LLM-visible schema).
_MAX_ENUM_VALUES = 20
# Bumped from 1500 → 3000 when per-column behavioural hints were added so the
# critical list-valued hints (e.g. `personnel_involved`) and the date-range
# section both fit even on the widest schema (work_requests, 11 cols).
_SCHEMA_SUMMARY_CHAR_CAP = 3000


@lru_cache(maxsize=8)
def _build_schema_summary(data_source: str) -> str:
    """Build a concise schema cheatsheet for `data_source`.

    Loads the file lazily (once per source per process) and returns a
    markdown-flavored summary the LLM can read in the tool description:
    column names + dtypes, value enums for low-cardinality string columns,
    date range for any date column, and total row count.

    Returns "" if the source is not on disk; the tool description falls
    back to a generic blurb in that case.
    """
    path = WHITELISTED_DATA_FILES.get(data_source)
    if not path or not Path(path).exists():
        return ""
    try:
        import pandas as pd  # local import keeps module import cheap if pandas missing
    except ImportError:
        return ""

    try:
        p = Path(path)
        if p.is_dir():
            entries: List[Any] = []
            for fp in sorted(p.glob("*.json")):
                with open(fp, "r") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict) and "entries" in raw:
                    entries.extend(raw["entries"])
                elif isinstance(raw, list):
                    entries.extend(raw)
                else:
                    entries.append(raw)
            df = pd.DataFrame(entries)
        else:
            with open(p, "r") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                df = pd.DataFrame(raw)
            elif isinstance(raw, dict) and "entries" in raw:
                df = pd.DataFrame(raw["entries"])
            else:
                df = pd.DataFrame([raw])
    except Exception as exc:  # noqa: BLE001 — never block tool init on schema probe
        logger.warning("Schema probe failed for %s: %s", data_source, exc)
        return ""

    lines: List[str] = []
    lines.append(f"### `{data_source}` — DataFrame `df` ({len(df):,} rows × {len(df.columns)} cols)")
    lines.append("")
    lines.append("Columns:")
    for col in df.columns:
        col_dtype = str(df[col].dtype)
        sample_val = ""
        non_null = df[col].dropna()
        if len(non_null) > 0:
            sv = non_null.iloc[0]
            sample_val = repr(sv)[:60]
        lines.append(f"- `{col}` ({col_dtype}) e.g. {sample_val}")

    # Per-column behavioural hints for the LLM. Four buckets:
    #   1. list-valued  — `apply` + `any('TOKEN' in str(x).lower() for x in xs)`
    #   2. unhashable (other) — treat as free text
    #   3. low-cardinality (≤ _MAX_ENUM_VALUES) — list actual values
    #   4. high-cardinality free text — substring-match directive
    # The list-valued check runs FIRST because `unique()` raises TypeError on
    # list-valued columns (the bug that previously dropped `personnel_involved`
    # from the schema entirely, leading to `==` filters that miss variants).
    # Bucket hints by priority so the schema char-cap truncates the LEAST
    # critical info first. Order:
    #   1. list-valued / unhashable (highest — these are the columns that
    #      previously vanished from the schema and caused the Doe bug)
    #   2. low-cardinality enums (concrete values are useful for filter literals)
    #   3. high-cardinality free text (just a substring directive — repetitive)
    list_lines: List[str] = []
    enum_lines: List[str] = []
    high_card_lines: List[str] = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        sample = s.iloc[0]
        if isinstance(sample, list):
            list_lines.append(
                f"- `{col}` (list-valued) — use "
                f"`df['{col}'].apply(lambda xs: any('TOKEN' in str(x).lower() "
                f"for x in (xs or [])))`. Do NOT use `==` on lists."
            )
            continue
        try:
            uniques = s.unique()
        except TypeError:
            list_lines.append(
                f"- `{col}` (unhashable, treat as free text) — use "
                f"`.astype(str).str.contains('TOKEN', case=False, na=False)`."
            )
            continue
        n = len(uniques)
        if 1 < n <= _MAX_ENUM_VALUES:
            vals = sorted(str(u) for u in uniques)
            enum_lines.append(f"- `{col}` values ({n}): {vals}")
        elif n > _MAX_ENUM_VALUES:
            high_card_lines.append(
                f"- `{col}` ({n} unique values, high-cardinality free text) — "
                f"use `.str.contains('TOKEN', case=False, na=False)`. Do NOT exact-match."
            )
    hint_lines = list_lines + enum_lines + high_card_lines
    if hint_lines:
        lines.append("")
        lines.append("Per-column hints (read these before writing a filter):")
        lines.extend(hint_lines)

    # Date-like columns: report min/max. Detector is name-hint + value-shape
    # check (must look like YYYY-MM-DD…) to avoid false positives like
    # `created_by` (a username column) being flagged as a date.
    import re as _re
    _ISO_DATE_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}")
    date_lines: List[str] = []
    for col in df.columns:
        cl = col.lower()
        if not any(tok in cl for tok in ("date", "time", "timestamp", "submission")):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        if s.dtype == object:
            sample = s.iloc[0]
            if not (isinstance(sample, str) and _ISO_DATE_RE.match(sample)):
                continue  # name hinted at a date but values aren't ISO dates
            date_lines.append(f"- `{col}` range: {s.min()} → {s.max()}  (ISO 8601 string)")
        else:
            date_lines.append(f"- `{col}` range: {s.min()} → {s.max()}")
    if date_lines:
        lines.append("")
        lines.append("Date columns (use `.str.startswith('2025')` for ISO strings; `pd.to_datetime` if you need arithmetic):")
        lines.extend(date_lines)

    out = "\n".join(lines)
    if len(out) > _SCHEMA_SUMMARY_CHAR_CAP:
        out = out[:_SCHEMA_SUMMARY_CHAR_CAP] + "\n…(truncated)"
    return out


def _build_tool_description() -> str:
    """Compose the LLM-visible tool description with live schema summaries."""
    header = (
        "Run pandas analysis over whitelisted the facility data files in a sandboxed "
        "subprocess. **Use this tool for any counting / filtering / grouping / "
        "aggregation question** instead of estimating from retrieved snippets — "
        "the model is not a reliable counter, and chunked retrieval undercounts.\n\n"
        "Args:\n"
        "  - `data_source`: one of " + str(list(WHITELISTED_DATA_FILES.keys())) + "\n"
        "  - `analysis_code`: pandas code; the DataFrame `df` is pre-loaded.\n\n"
        "Available preamble constants in the subprocess (do NOT redefine):\n"
        "  - `df`         — the loaded DataFrame for the chosen `data_source`\n"
        "  - `pd`, `np`, `json`\n"
        "  - `TODAY`      — `datetime.date` of the server's current date\n"
        "  - `THIS_YEAR`  — int, `TODAY.year` (use this for 'this year' queries)\n"
        "  - `YESTERDAY`  — `TODAY - timedelta(days=1)`\n"
        "  - `LAST_WEEK_START` — `TODAY - timedelta(days=7)`\n\n"
        "**Always `print()` the final answer.** Print intermediate counts too "
        "if the question has sub-parts ('how many in each machine?').\n\n"
        "Common pitfalls:\n"
        "  - Date columns are ISO 8601 strings, NOT pandas datetimes. Use "
        "`df['SUBMISSION_DATE'].str.startswith(str(THIS_YEAR))` for year filters.\n"
        "  - Equipment terms like 'kicker', 'monitor', 'BPM' live in free text "
        "(`DESCRIPTION`, `studies_summary`); use "
        "`.str.contains('kicker', case=False, na=False)`.\n"
        "  - Match the EXACT case of column names and enum values shown below.\n\n"
        "**Defensive filter idiom (CRITICAL for human names, IDs, and free-text columns):**\n\n"
        "Database text is messy and contains variants. When filtering by human names, "
        "identifiers, or entity mentions:\n\n"
        "  - NEVER use exact equality (`==`). The same person may appear as "
        "`'Jane Doe'`, `'Doe'`, `'M. Doe'`, or `'user010'`.\n"
        "  - ALWAYS use case-insensitive substring matching:\n"
        "        `df[df['col'].astype(str).str.contains('Doe', case=False, na=False)]`\n"
        "  - For LIST-VALUED columns (e.g. `personnel_involved`), substring-check each item:\n"
        "        `df[df['col'].apply(lambda xs: any('Doe' in str(x).lower() for x in (xs or [])))]`\n"
        "  - Before stating a final count, SAMPLE the matches:\n"
        "        `print(filtered.head(5).to_dict(orient='records'))`\n"
        "    Eyeball whether the filter caught the variants you expect.\n\n"
        "**For entity-count queries — emit a structured output instead of a bare number:**\n\n"
        "When the user asks `how many <entity>`, print a JSON dict so a downstream "
        "auditor can verify your filter caught the right rows:\n\n"
        "```python\n"
        "import json\n"
        "filtered = df[df['col'].apply(\n"
        "    lambda xs: any('Doe' in str(x).lower() for x in (xs or [])))]\n"
        "excluded = df[~df.index.isin(filtered.index)]\n"
        "near_miss = excluded.sample(min(5, len(excluded)), random_state=0) \\\n"
        "    if len(excluded) else excluded\n"
        "print(json.dumps({\n"
        "    'count': int(len(filtered)),\n"
        "    'matched_sample': filtered.head(5).to_dict(orient='records'),\n"
        "    'near_miss_sample': near_miss.to_dict(orient='records'),\n"
        "}, default=str))\n"
        "```\n\n"
        "For non-count queries (averages, groupbys, time series), the loose "
        "`print(answer)` form is fine.\n\n"
        "---\n\n"
        "## Available data sources\n\n"
    )
    parts: List[str] = [header]
    for src in WHITELISTED_DATA_FILES.keys():
        summary = _build_schema_summary(src)
        if summary:
            parts.append(summary)
            parts.append("")
        else:
            parts.append(f"### `{src}` — (file not found at import; schema unknown)\n")
    return "\n".join(parts)


class _DataAnalysisInput(BaseModel):
    analysis_code: str = Field(
        description=(
            "Python/pandas code to analyse the data. "
            "Available variables: `df` (pandas DataFrame loaded from the target file). "
            "Print your final answer with print()."
        )
    )
    data_source: str = Field(
        description="Which data source to load: work_requests | teams_messages | entry_logbook"
    )


_DEFAULT_DESCRIPTION = _build_tool_description()


class DataAnalysisTool(BaseTool):
    """Execute pandas analysis on whitelisted the facility data files."""

    name: str = "data_analysis"
    description: str = _DEFAULT_DESCRIPTION
    args_schema: Type[BaseModel] = _DataAnalysisInput
    timeout_seconds: int = 45

    def _run(self, analysis_code: str, data_source: str) -> str:
        # Stage C: route through MCP `compute` server. The actual subprocess
        # execution + safety checks + TODAY preamble live in
        # ``data_analysis_impl`` (see below) so the MCP server can call them
        # without going through this method (which would recurse).
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync(
                "compute", "data_analysis",
                {"analysis_code": analysis_code, "data_source": data_source},
            )
        )


def data_analysis_impl(
    analysis_code: str,
    data_source: str,
    timeout_seconds: int = 45,
) -> str:
    """Pure data-analysis logic — called by the compute MCP server. Module-level
    so the server can bypass DataAnalysisTool._run (which now dispatches back
    through MCP and would recurse).

    Returns a JSON string with ``{"stdout", "stderr", "returncode",
    "data_source", "data_path"}`` on success, or ``{"error", ...}`` on
    setup/timeout failures (sandbox safety, missing data file, etc.).
    """
    data_path = WHITELISTED_DATA_FILES.get(data_source)
    if not data_path:
        return json.dumps({
            "error": f"Unknown data_source '{data_source}'. "
                     f"Available: {list(WHITELISTED_DATA_FILES.keys())}",
            "stdout": "",
            "returncode": 1,
        })

    if not Path(data_path).exists():
        return json.dumps({
            "error": f"Data file not found: {data_path}",
            "stdout": "",
            "returncode": 1,
        })

    # Safety checks — strip comments + string literals to reduce false positives
    import re as _re
    _code_stripped = _re.sub(r'#[^\n]*', '', analysis_code)
    _code_stripped = _re.sub(r'""".*?"""', '', _code_stripped, flags=_re.DOTALL)
    _code_stripped = _re.sub(r"'''.*?'''", '', _code_stripped, flags=_re.DOTALL)
    _code_stripped = _re.sub(r'"[^"\n]*"', '', _code_stripped)
    _code_stripped = _re.sub(r"'[^'\n]*'", '', _code_stripped)
    forbidden = ["os.system", "__import__", "subprocess", "shutil", "exec(", "eval("]
    for pattern in forbidden:
        if pattern in _code_stripped:
            return json.dumps({
                "error": f"Forbidden pattern '{pattern}' in analysis code",
                "stdout": "",
                "returncode": 1,
            })

    # Build the subprocess loader script
    is_dir = Path(data_path).is_dir()
    if is_dir:
        loader_lines = [
            "import glob",
            "from pathlib import Path",
            f"_files = sorted(Path({data_path!r}).glob('*.json'))",
            "_all_entries = []",
            "for _f in _files:",
            "    with open(_f, 'r') as _fh:",
            "        _data = json.load(_fh)",
            "        if isinstance(_data, dict) and 'entries' in _data:",
            "            _all_entries.extend(_data['entries'])",
            "        elif isinstance(_data, list):",
            "            _all_entries.extend(_data)",
            "        else:",
            "            _all_entries.append(_data)",
            "df = pd.DataFrame(_all_entries)",
        ]
    else:
        loader_lines = [
            f"with open({data_path!r}, 'r') as _fh:",
            "    _raw = json.load(_fh)",
            "if isinstance(_raw, list):",
            "    df = pd.DataFrame(_raw)",
            "elif isinstance(_raw, dict) and 'entries' in _raw:",
            "    df = pd.DataFrame(_raw['entries'])",
            "else:",
            "    df = pd.DataFrame([_raw])",
        ]

    # Today's date is computed HERE (parent process) and injected into the
    # subprocess preamble so the LLM-generated pandas code has a stable
    # programmatic anchor for "this year" / "yesterday" / "last week" queries.
    today = date.today()
    date_preamble = [
        "from datetime import date, timedelta",
        f"TODAY = date({today.year}, {today.month}, {today.day})",
        "THIS_YEAR = TODAY.year",
        "YESTERDAY = TODAY - timedelta(days=1)",
        "LAST_WEEK_START = TODAY - timedelta(days=7)",
    ]

    # Build complete script as list-of-lines (avoids textwrap.dedent +
    # f-string indentation mismatch that previously corrupted the script).
    script = "\n".join([
        "import json",
        "import pandas as pd",
        "import numpy as np",
        "from datetime import datetime, timedelta",
        *date_preamble,
        "",
        *loader_lines,
        "",
        "# --- User analysis code ---",
        textwrap.dedent(analysis_code).strip(),
        "",
    ])

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return json.dumps({
            "stdout": proc.stdout[:8000],
            "stderr": proc.stderr[:2000],
            "returncode": proc.returncode,
            "data_source": data_source,
            "data_path": data_path,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"Analysis timed out after {timeout_seconds}s",
            "stdout": "",
            "returncode": -1,
        })
    except Exception as exc:
        logger.error("DataAnalysis execution error: %s", exc)
        return json.dumps({"error": str(exc), "stdout": "", "returncode": 1})
