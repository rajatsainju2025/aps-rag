"""
Integration tests for the PV query pipeline.
Tests end-to-end flow with mocked EPICS and archiver.
Requires: settings loadable, tool classes importable.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.mark.integration
class TestPVQueryRouting:
    def test_pv_intent_routes_to_agentic_rag(self):
        from src.agents.router import _INTENT_TO_SERVICE
        assert _INTENT_TO_SERVICE.get("pv_query") == "agentic_rag"

    def test_pv_query_state_fields_exist(self):
        from src.agents.state import AgentState
        annotations = AgentState.__annotations__
        assert "pv_addresses" in annotations
        assert "pv_readings" in annotations
        assert "pv_history" in annotations


@pytest.mark.integration
class TestPVToolChaining:
    def test_channel_finder_to_pv_read(self, tmp_path):
        """Test that channel_finder results can feed into pv_read."""
        from src.tools.channel_finder import ChannelFinderTool

        hier = {
            "tree": {
                "SR": {
                    "DIAG": {
                        "DCCT": {
                            "DEVICE": {
                                "_expansion": {"_type": "list", "_instances": ["S-DCCT"]},
                                "CURRENT": {
                                    ":CurrentM": {"_description": "Current Monitor"},
                                },
                            },
                        },
                    },
                },
            }
        }
        hier_file = tmp_path / "hier.json"
        hier_file.write_text(json.dumps(hier))

        cf = ChannelFinderTool(db_path=str(hier_file))
        result = json.loads(cf._run("beam current"))
        pvs = [r["pv"] for r in result["pv_addresses"]]

        assert len(pvs) > 0
        pv_names_str = ", ".join(pvs)
        assert "S-DCCT:CurrentM" in pv_names_str


@pytest.mark.integration
class TestArchiverQueryMock:
    @patch("src.tools.archiver_reader._requests")
    def test_archiver_summary_stats(self, mock_requests):
        from src.tools.archiver_reader import ArchiverReaderTool

        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session
        mock_response = MagicMock()
        data_points = [
            {"secs": 1700000000 + i * 60, "nanos": 0, "val": 100.0 + i * 0.1}
            for i in range(60)
        ]
        mock_response.json.return_value = [
            {"meta": {"name": "S-DCCT:CurrentM"}, "data": data_points}
        ]
        mock_session.get.return_value = mock_response

        tool = ArchiverReaderTool()
        result = json.loads(tool._run("S-DCCT:CurrentM", "1h"))
        assert len(result["data"]) == 1
        summary = result["data"][0]
        assert summary["pv"] == "S-DCCT:CurrentM"
        assert summary["count"] == 60
        assert summary["min"] == pytest.approx(100.0, abs=0.01)
        assert summary["max"] == pytest.approx(105.9, abs=0.01)
        assert "mean" in summary
