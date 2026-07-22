"""
LangChain Tool: ArchiverReaderTool — historical PV data from the facility archiver.

Ported from Reference_aps-control-assistant-main/aps_archiver_connector.py.
Standalone implementation — no OSPREY dependency.

REST API: GET https://your-archiver.example.com/retrieval/data/getData.json
"""
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urljoin

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import requests as _requests
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False
    pd = None  # type: ignore[assignment]
    _requests = None  # type: ignore[assignment]
    logger.warning("pandas/requests not installed; archiver reader will be unavailable")


class _ArchiverReaderInput(BaseModel):
    pv_names: str = Field(description="Comma-separated PV names")
    start: str = Field(
        description="Start time: ISO8601 (2025-01-01T00:00:00Z) or relative (1h, 24h, 7d, 30m)"
    )
    end: str = Field(default="", description="End time: ISO8601 or empty for 'now'")
    processing: str = Field(
        default="",
        description="Server-side processing: 'mean_600' for 10-min bins, 'max_3600' for hourly max",
    )


class ArchiverReaderTool(BaseTool):
    """Retrieve historical PV data from the facility archiver."""

    name: str = "archiver_read"
    description: str = (
        "Fetch historical PV data from the facility EPICS Archiver Appliance. "
        "Returns summary statistics (min, max, mean, count) for each PV over a time range. "
        "Provide PV names and a time range (ISO8601 or relative like '1h', '24h', '7d')."
    )
    args_schema: Type[BaseModel] = _ArchiverReaderInput
    archiver_url: str = Field(default="https://your-archiver.example.com", exclude=True)
    timeout: int = Field(default=60, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        archiver_url: str = "https://your-archiver.example.com",
        timeout: int = 60,
        mcp_client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "archiver_url", archiver_url)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "_mcp_client", mcp_client)

    # --- Ported helpers (standalone, no OSPREY) ---

    @staticmethod
    def _build_retrieval_url(base_url: str) -> str:
        if "retrieval" not in base_url:
            base_url = base_url.rstrip("/") + "/retrieval"
        return urljoin(base_url.rstrip("/") + "/", "data/getData.json")

    @staticmethod
    def _to_iso8601_utc(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_relative_time(s: str) -> datetime:
        """Convert relative time strings like '1h', '24h', '7d', '30m' to absolute datetime."""
        s = s.strip().lower()
        match = re.match(r"^(\d+)\s*([mhd])$", s)
        if not match:
            raise ValueError(f"Invalid relative time format: '{s}'. Use e.g. '1h', '24h', '7d', '30m'")
        value = int(match.group(1))
        unit = match.group(2)
        delta_map = {"m": timedelta(minutes=value), "h": timedelta(hours=value), "d": timedelta(days=value)}
        return datetime.now(timezone.utc) - delta_map[unit]

    @staticmethod
    def _parse_series(payload: list) -> Dict[str, Any]:
        """Convert Archiver Appliance JSON response to {pv_name: pd.Series}."""
        series: Dict[str, Any] = {}
        for entry in payload or []:
            pv_name = entry.get("meta", {}).get("name")
            data = entry.get("data", [])
            if not pv_name or not data:
                continue

            index: List[datetime] = []
            values: list = []
            for point in data:
                secs = point.get("secs")
                nanos = point.get("nanos", 0)
                val = point.get("val")
                if secs is None:
                    continue
                if isinstance(val, list) and len(val) == 1:
                    val = val[0]
                ts = datetime.fromtimestamp(secs + nanos / 1e9, tz=timezone.utc)
                index.append(ts)
                values.append(val)

            if index:
                series[pv_name] = pd.Series(values, index=index, dtype="object")

        return series

    def _fetch_data(
        self,
        pv_list: List[str],
        start_date: datetime,
        end_date: datetime,
        processing: str = "",
        bin_size: Optional[int] = None,
    ) -> Any:
        """Fetch data from the Archiver Appliance REST API.

        Tries batch request first, then falls back to individual requests.
        """
        retrieval_url = self._build_retrieval_url(self.archiver_url)
        start_iso = self._to_iso8601_utc(start_date)
        end_iso = self._to_iso8601_utc(end_date)

        session = _requests.Session()
        all_series: Dict[str, Any] = {}

        def _pv_with_processing(pv: str) -> str:
            if processing and processing != "raw" and bin_size:
                return f"{processing}_{bin_size}({pv})"
            return pv

        # Batch request
        params: List[tuple] = [("from", start_iso), ("to", end_iso)]
        for pv in pv_list:
            params.append(("pv", _pv_with_processing(pv)))

        try:
            resp = session.get(retrieval_url, params=params, timeout=self.timeout, verify=False)
            resp.raise_for_status()
            payload = resp.json()
            all_series = self._parse_series(payload)
            logger.debug("Batch request: %d/%d PVs returned data", len(all_series), len(pv_list))
        except Exception as e:
            logger.warning("Batch archiver request failed: %s", e)

        # Per-PV fallback
        for pv in pv_list:
            if pv in all_series:
                continue
            try:
                resp = session.get(
                    retrieval_url,
                    params={"from": start_iso, "to": end_iso, "pv": _pv_with_processing(pv)},
                    timeout=self.timeout,
                    verify=False,
                )
                resp.raise_for_status()
                payload = resp.json()
                parsed = self._parse_series(payload)
                all_series.update(parsed)
            except Exception as e:
                logger.warning("Individual request for '%s' failed: %s", pv, e)

        if not all_series:
            return pd.DataFrame()

        df = pd.DataFrame(all_series)
        df.sort_index(inplace=True)
        return df

    def _run(
        self,
        pv_names: str,
        start: str,
        end: str = "",
        processing: str = "",
    ) -> str:
        # Stage C: route through MCP `archiver` server. The local REST-call
        # logic lives in ``_run_local`` below — the MCP server (mcp_servers/
        # archiver.py) calls that to avoid recursing back through this method.
        pvs = [pv.strip() for pv in pv_names.split(",") if pv.strip()]
        if not pvs:
            return json.dumps({"error": "No PV names provided", "data": []})
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        args = {"pv_names": pvs, "start": start}
        if end:
            args["end"] = end
        if processing:
            args["processing"] = processing
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync("archiver", "archiver_read", args)
        )

    def _run_local(
        self,
        pv_names: str,
        start: str,
        end: str = "",
        processing: str = "",
    ) -> str:
        """Pre-Stage-C body: direct REST call to the facility Archiver Appliance.
        Called by the MCP `archiver` server (mcp_servers/archiver.py) to
        bypass the MCP shim in ``_run`` (which would recurse). Not a public
        API — do not call from agent code."""
        pvs = [pv.strip() for pv in pv_names.split(",") if pv.strip()]
        if not pvs:
            return json.dumps({"error": "No PV names provided", "data": []})

        if not _DEPS_AVAILABLE:
            return json.dumps({"error": "pandas/requests not installed", "data": []})

        # Parse start time
        try:
            if re.match(r"^\d+[mhd]$", start.strip().lower()):
                start_dt = self._parse_relative_time(start)
            else:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except Exception as exc:
            return json.dumps({"error": f"Invalid start time: {exc}", "data": []})

        # Parse end time
        if end and end.strip():
            try:
                if re.match(r"^\d+[mhd]$", end.strip().lower()):
                    end_dt = self._parse_relative_time(end)
                else:
                    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            except Exception as exc:
                return json.dumps({"error": f"Invalid end time: {exc}", "data": []})
        else:
            end_dt = datetime.now(timezone.utc)

        # Parse processing
        proc_type = ""
        bin_sz: Optional[int] = None
        if processing:
            proc_match = re.match(r"^(\w+?)_(\d+)$", processing.strip())
            if proc_match:
                proc_type = proc_match.group(1)
                bin_sz = int(proc_match.group(2))

        # Fetch data
        try:
            df = self._fetch_data(pvs, start_dt, end_dt, processing=proc_type, bin_size=bin_sz)
        except Exception as exc:
            return json.dumps({"error": f"Archiver fetch failed: {exc}", "data": []})

        if df.empty:
            return json.dumps({"data": [], "message": "No data returned from archiver"})

        # Compute summary statistics per PV
        summaries = []
        errors = []
        for pv in pvs:
            if pv not in df.columns:
                errors.append({"pv": pv, "error": "No data returned"})
                continue
            col = pd.to_numeric(df[pv], errors="coerce").dropna()
            if col.empty:
                errors.append({"pv": pv, "error": "No numeric data"})
                continue
            summaries.append({
                "pv": pv,
                "count": int(len(col)),
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": round(float(col.mean()), 6),
                "std": round(float(col.std()), 6) if len(col) > 1 else 0.0,
                "first_value": float(col.iloc[0]),
                "last_value": float(col.iloc[-1]),
                "start": col.index[0].isoformat(),
                "end": col.index[-1].isoformat(),
            })

        return json.dumps({"data": summaries, "errors": errors})
