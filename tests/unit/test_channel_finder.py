"""
Unit tests for ChannelFinderTool.
Covers: index building, keyword search, filtering, JSON output.
No external services required.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tools.channel_finder import ChannelFinderTool


# --- Minimal fixtures ---

MINIMAL_HIERARCHICAL = {
    "tree": {
        "SR": {
            "_description": "Storage Ring",
            "DIAG": {
                "_description": "Diagnostics",
                "DCCT": {
                    "_description": "DC Current Transformer",
                    "DEVICE": {
                        "_expansion": {"_type": "list", "_instances": ["S-DCCT"]},
                        "CURRENT": {
                            "_description": "Beam Current",
                            ":CurrentM": {"_description": "Current Monitor"},
                            ":LifetimeM": {"_description": "Lifetime Monitor"},
                        },
                    },
                },
            },
            "MAG": {
                "_description": "Magnets",
                "H1": {
                    "_description": "Horizontal Corrector 1",
                    "DEVICE": {
                        "_expansion": {
                            "_type": "range",
                            "_pattern": "S{:02d}A:FH1",
                            "_range": [1, 3],
                        },
                        "PS": {
                            "_description": "Power Supply",
                            ":PS:SetCurrentC": {"_description": "Set Current Command"},
                            ":PS:CurrentM": {"_description": "Current Monitor"},
                        },
                    },
                },
            },
        },
        "BTS": {
            "_description": "Booster-to-Storage Ring",
            "MAG": {
                "_description": "Magnets",
                "BM": {
                    "_description": "Bend Magnet",
                    "DEVICE": {
                        "_expansion": {"_type": "list", "_instances": ["BTS:BM1"]},
                        "PS": {
                            "_description": "Power Supply",
                            ":PS:CurrentM": {"_description": "Current Monitor"},
                        },
                    },
                },
            },
        },
    }
}

MINIMAL_IN_CONTEXT = {
    "channels": [
        {
            "channel": "SR Beam Current",
            "address": "S-DCCT:CurrentM",
            "description": "Storage ring beam current from DCCT",
            "tags": ["beam", "current", "dcct"],
        },
        {
            "channel": "ITS Temperature",
            "address": "ITS:Temp1:ValueM",
            "description": "ITS cryogenic temperature sensor 1",
            "tags": ["its", "temperature", "cryo"],
        },
    ]
}


@pytest.fixture
def tool_with_data(tmp_path):
    hier_file = tmp_path / "hierarchical.json"
    ctx_file = tmp_path / "in_context.json"
    hier_file.write_text(json.dumps(MINIMAL_HIERARCHICAL))
    ctx_file.write_text(json.dumps(MINIMAL_IN_CONTEXT))
    return ChannelFinderTool(
        db_path=str(hier_file),
        in_context_db_path=str(ctx_file),
    )


@pytest.fixture
def tool_empty():
    return ChannelFinderTool()


class TestBuildHierarchicalIndex:
    def test_index_has_entries(self, tool_with_data):
        tool_with_data._load_databases()
        assert len(tool_with_data._db_index) > 0

    def test_dcct_pv_in_index(self, tool_with_data):
        tool_with_data._load_databases()
        pvs = [e["pv"] for e in tool_with_data._db_index]
        assert "S-DCCT:CurrentM" in pvs
        assert "S-DCCT:LifetimeM" in pvs

    def test_range_expansion(self, tool_with_data):
        tool_with_data._load_databases()
        pvs = [e["pv"] for e in tool_with_data._db_index]
        assert "S01A:FH1:PS:SetCurrentC" in pvs
        assert "S02A:FH1:PS:SetCurrentC" in pvs
        assert "S03A:FH1:PS:SetCurrentC" in pvs

    def test_hierarchy_path(self, tool_with_data):
        tool_with_data._load_databases()
        dcct_entry = next(
            e for e in tool_with_data._db_index if e["pv"] == "S-DCCT:CurrentM"
        )
        assert "SR" in dcct_entry["path"]
        assert "DIAG" in dcct_entry["path"]


class TestSearch:
    def test_search_beam_current(self, tool_with_data):
        result = json.loads(tool_with_data._run("beam current"))
        pvs = [r["pv"] for r in result["pv_addresses"]]
        assert "S-DCCT:CurrentM" in pvs

    def test_search_corrector(self, tool_with_data):
        result = json.loads(tool_with_data._run("corrector"))
        pvs = [r["pv"] for r in result["pv_addresses"]]
        corrector_pvs = [p for p in pvs if "FH1" in p]
        assert len(corrector_pvs) > 0

    def test_search_ring_filter(self, tool_with_data):
        result = json.loads(tool_with_data._run("magnets", ring="BTS"))
        pvs = [r["pv"] for r in result["pv_addresses"]]
        for pv in pvs:
            assert "BTS" in pv or "BTS" in str(result)

    def test_search_system_filter(self, tool_with_data):
        result = json.loads(tool_with_data._run("current", system="MAG"))
        pvs = [r["pv"] for r in result["pv_addresses"]]
        for entry in result["pv_addresses"]:
            assert "MAG" in entry.get("path", "") or "Corrector" in entry.get("description", "")

    def test_search_max_results(self, tool_with_data):
        result = json.loads(tool_with_data._run("current", max_results=2))
        assert len(result["pv_addresses"]) <= 2

    def test_search_no_results(self, tool_with_data):
        result = json.loads(tool_with_data._run("xyzzy123nonexistent"))
        assert result["pv_addresses"] == []
        assert result["count"] == 0


class TestInContextDB:
    def test_in_context_entries_loaded(self, tool_with_data):
        tool_with_data._load_databases()
        assert len(tool_with_data._in_context_index) == 2

    def test_in_context_searched(self, tool_with_data):
        result = json.loads(tool_with_data._run("ITS temperature"))
        pvs = [r["pv"] for r in result["pv_addresses"]]
        assert "ITS:Temp1:ValueM" in pvs


class TestRunOutput:
    def test_run_returns_valid_json(self, tool_with_data):
        result = json.loads(tool_with_data._run("beam current"))
        assert "pv_addresses" in result
        assert "count" in result
        assert isinstance(result["pv_addresses"], list)

    def test_no_databases_loaded(self, tool_empty):
        result = json.loads(tool_empty._run("anything"))
        assert "error" in result
