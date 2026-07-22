"""
LangChain Tool: ChannelFinderTool — resolve device names to EPICS PV addresses.

Uses the facility hierarchical channel database (hierarchical.json) and the
in-context database (its_pv_in_context.json) to map natural language device
descriptions to concrete PV addresses.

READ-ONLY: no control system access, purely database lookup.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _ChannelFinderInput(BaseModel):
    query: str = Field(
        description=(
            "Device or signal name to find (e.g., 'beam current', "
            "'BPM sector 2 positions', 'H1 corrector setpoint')"
        )
    )
    ring: Optional[str] = Field(default=None, description="Filter by ring: SR, BR, or BTS")
    system: Optional[str] = Field(default=None, description="Filter by system: MAG, DIAG, VAC, RF")
    max_results: int = Field(default=20, description="Maximum number of PV addresses to return")


class ChannelFinderTool(BaseTool):
    """Resolve device names to EPICS PV addresses from channel database."""

    name: str = "channel_finder"
    description: str = (
        "Find EPICS PV addresses for accelerator devices. "
        "Use when the user asks about control system values but doesn't give exact PV names. "
        "Examples: 'beam current', 'BPM positions sector 5', 'H1 corrector setpoint'."
    )
    args_schema: Type[BaseModel] = _ChannelFinderInput
    db_path: str = Field(default="", exclude=True)
    in_context_db_path: str = Field(default="", exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, db_path: str = "", in_context_db_path: str = "", mcp_client: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "db_path", db_path)
        object.__setattr__(self, "in_context_db_path", in_context_db_path)
        object.__setattr__(self, "_mcp_client", mcp_client)
        object.__setattr__(self, "_db_index", [])
        object.__setattr__(self, "_in_context_index", [])
        object.__setattr__(self, "_loaded", False)

    def _load_databases(self) -> None:
        if self._loaded:
            return

        if self.in_context_db_path and Path(self.in_context_db_path).exists():
            try:
                with open(self.in_context_db_path) as f:
                    data = json.load(f)
                channels = data.get("channels", [])
                object.__setattr__(self, "_in_context_index", self._build_in_context_index(channels))
                logger.info("In-context DB loaded: %d entries", len(self._in_context_index))
            except Exception as exc:
                logger.warning("Failed to load in-context DB: %s", exc)

        if self.db_path and Path(self.db_path).exists():
            try:
                with open(self.db_path) as f:
                    data = json.load(f)
                tree = data.get("tree", {})
                object.__setattr__(self, "_db_index", self._build_hierarchical_index(tree))
                logger.info("Hierarchical DB loaded: %d entries", len(self._db_index))
            except Exception as exc:
                logger.warning("Failed to load hierarchical DB: %s", exc)

        object.__setattr__(self, "_loaded", True)

    @staticmethod
    def _build_in_context_index(channels: List[Dict]) -> List[Dict]:
        index = []
        for ch in channels:
            address = ch.get("address", "")
            if not address:
                continue
            description = ch.get("description", "")
            channel_name = ch.get("channel", "")
            tags = ch.get("tags", [])
            search_text = " ".join([channel_name, description, address] + tags).lower()
            index.append({
                "pv": address,
                "description": description or channel_name,
                "path": "ITS (in-context)",
                "ring": "ITS",
                "system": "",
                "family": "",
                "search_text": search_text,
                "source": "in_context",
            })
        return index

    @staticmethod
    def _build_hierarchical_index(tree: Dict) -> List[Dict]:
        """Recursively walk the hierarchical tree and build a flat search index.

        Structure: tree[ring][system][family] -> DEVICE -> {_expansion, field -> subfield}
        Full PV = device_instance + subfield_key (e.g., S01A:FH1 + :PS:SetCurrentC)
        """
        index = []

        for ring_key, ring_node in tree.items():
            if ring_key.startswith("_"):
                continue
            ring_desc = ring_node.get("_description", ring_key)

            for sys_key, sys_node in ring_node.items():
                if sys_key.startswith("_") or not isinstance(sys_node, dict):
                    continue
                sys_desc = sys_node.get("_description", sys_key)

                for fam_key, fam_node in sys_node.items():
                    if fam_key.startswith("_") or not isinstance(fam_node, dict):
                        continue
                    fam_desc = fam_node.get("_description", fam_key)

                    device_node = fam_node.get("DEVICE")
                    if not isinstance(device_node, dict):
                        continue

                    # Get device instances
                    instances = ChannelFinderTool._expand_devices(device_node.get("_expansion", {}))

                    # Get field→subfield structure (keys in DEVICE that aren't _expansion)
                    for field_key, field_node in device_node.items():
                        if field_key.startswith("_") or not isinstance(field_node, dict):
                            continue
                        field_desc = field_node.get("_description", field_key)

                        for subfield_key, subfield_node in field_node.items():
                            if subfield_key.startswith("_") or not isinstance(subfield_node, dict):
                                continue
                            subfield_desc = subfield_node.get("_description", subfield_key)

                            combined_desc = " ".join(filter(None, [
                                fam_desc, field_desc, subfield_desc,
                            ]))
                            path = f"{ring_key} > {sys_key} > {fam_key} > {field_key} > {subfield_key}"

                            for device in instances:
                                pv = f"{device}{subfield_key}"
                                search_text = " ".join([
                                    ring_key, sys_key, fam_key, field_key,
                                    subfield_key, device, pv,
                                    ring_desc, combined_desc,
                                ]).lower()

                                index.append({
                                    "pv": pv,
                                    "description": combined_desc.strip(),
                                    "path": path,
                                    "ring": ring_key,
                                    "system": sys_key,
                                    "family": fam_key,
                                    "search_text": search_text,
                                    "source": "hierarchical",
                                })

        return index

    @staticmethod
    def _expand_devices(expansion: Dict) -> List[str]:
        exp_type = expansion.get("_type", "")
        if exp_type == "list":
            return expansion.get("_instances", [])
        elif exp_type == "range":
            pattern = expansion.get("_pattern", "{}")
            rng = expansion.get("_range", [0, 0])
            if len(rng) >= 2:
                return [pattern.format(i) for i in range(rng[0], rng[1] + 1)]
        return []

    def _search(
        self,
        query: str,
        ring: Optional[str] = None,
        system: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict]:
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []

        scored: List[tuple] = []
        all_entries = self._in_context_index + self._db_index

        for entry in all_entries:
            if ring and entry["ring"].upper() != ring.upper():
                continue
            if system and entry.get("system", "").upper() != system.upper():
                continue

            text = entry["search_text"]
            pv_lower = entry["pv"].lower()

            score = 0
            matched_tokens = 0
            for token in tokens:
                if token in text:
                    score += 1
                    matched_tokens += 1
                if token in pv_lower:
                    score += 0.5

            if matched_tokens == 0:
                continue

            # Bonus for matching all tokens
            if matched_tokens == len(tokens):
                score += 2

            # Bonus for in-context entries (curated, higher quality descriptions)
            if entry["source"] == "in_context":
                score += 1

            scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])

        # Deduplicate by PV name (keep highest-scored)
        seen_pvs = set()
        results = []
        for _, entry in scored:
            if entry["pv"] in seen_pvs:
                continue
            seen_pvs.add(entry["pv"])
            results.append({
                "pv": entry["pv"],
                "description": entry["description"],
                "path": entry["path"],
            })
            if len(results) >= max_results:
                break

        return results

    def _run(
        self,
        query: str,
        ring: Optional[str] = None,
        system: Optional[str] = None,
        max_results: int = 20,
    ) -> str:
        # Stage C: route through MCP `controls` server. The local-DB fallback
        # is gone — the MCP server (mcp_servers/controls.py) is the single
        # source of truth for the channel index, so the index is loaded once
        # in that subprocess and reused across calls.
        from src.tools.mcp_client import get_default_client, unwrap_mcp_envelope
        args = {"query": query, "max_results": max_results}
        if ring:
            args["ring"] = ring
        if system:
            args["system"] = system
        return unwrap_mcp_envelope(
            get_default_client().call_tool_sync("controls", "channel_find", args)
        )
