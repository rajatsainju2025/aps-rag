"""
Unit tests for PVReaderTool.
Covers: EPICS reads, mocking, safety, env config.
No live EPICS connection required — all pyepics calls are mocked.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import ModuleType

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(autouse=True)
def _inject_mock_epics():
    """Inject a mock epics module into pv_reader so @patch can find it."""
    import src.tools.pv_reader as mod
    if not hasattr(mod, "epics") or mod.epics is None:
        mock_epics = MagicMock(spec=ModuleType)
        mod.epics = mock_epics
    yield


from src.tools.pv_reader import PVReaderTool


@pytest.fixture
def tool():
    return PVReaderTool(
        gateway_address="test-gateway.example.com",
        gateway_port=5064,
        default_timeout=2.0,
        max_pvs=50,
    )


class TestEpicsUnavailable:
    def test_graceful_degradation(self, tool):
        with patch("src.tools.pv_reader._EPICS_AVAILABLE", False):
            result = json.loads(tool._run("S-DCCT:CurrentM"))
            assert "error" in result
            assert "pyepics" in result["error"]
            assert result["readings"] == []


class TestSuccessfulRead:
    @patch("src.tools.pv_reader._EPICS_AVAILABLE", True)
    @patch("src.tools.pv_reader.epics")
    def test_single_pv_read(self, mock_epics, tool):
        mock_epics.caget.return_value = 100.23
        result = json.loads(tool._run("S-DCCT:CurrentM"))
        assert result["count"] == 1
        assert result["readings"][0]["value"] == 100.23
        assert result["readings"][0]["status"] == "ok"
        assert result["readings"][0]["pv"] == "S-DCCT:CurrentM"

    @patch("src.tools.pv_reader._EPICS_AVAILABLE", True)
    @patch("src.tools.pv_reader.epics")
    def test_multiple_pvs(self, mock_epics, tool):
        mock_epics.caget.side_effect = [100.23, 45.6]
        result = json.loads(tool._run("S-DCCT:CurrentM, LTS:H1:CurrentAO"))
        assert result["count"] == 2
        assert len(result["readings"]) == 2

    @patch("src.tools.pv_reader._EPICS_AVAILABLE", True)
    @patch("src.tools.pv_reader.epics")
    def test_string_value(self, mock_epics, tool):
        mock_epics.caget.return_value = "Running"
        result = json.loads(tool._run("SR:Mode:StatusM"))
        assert result["readings"][0]["value"] == "Running"


class TestTimeoutPV:
    @patch("src.tools.pv_reader._EPICS_AVAILABLE", True)
    @patch("src.tools.pv_reader.epics")
    def test_timeout_returns_error(self, mock_epics, tool):
        mock_epics.caget.return_value = None
        result = json.loads(tool._run("NONEXIST:PV"))
        assert result["count"] == 0
        assert len(result["errors"]) == 1
        assert "Timeout" in result["errors"][0]["error"]


class TestMaxPVs:
    def test_max_pvs_exceeded(self, tool):
        with patch("src.tools.pv_reader._EPICS_AVAILABLE", True):
            pvs = ", ".join([f"PV{i}" for i in range(51)])
            result = json.loads(tool._run(pvs))
            assert "error" in result
            assert "Too many" in result["error"]


class TestEpicsEnvConfig:
    @patch("src.tools.pv_reader._EPICS_AVAILABLE", True)
    @patch("src.tools.pv_reader.epics")
    def test_env_configured(self, mock_epics, tool):
        mock_epics.caget.return_value = 1.0
        tool._run("TEST:PV")
        assert os.environ.get("EPICS_CA_ADDR_LIST") == "test-gateway.example.com:5064"
        assert os.environ.get("EPICS_CA_AUTO_ADDR_LIST") == "NO"


class TestCagetException:
    @patch("src.tools.pv_reader._EPICS_AVAILABLE", True)
    @patch("src.tools.pv_reader.epics")
    def test_exception_captured(self, mock_epics, tool):
        mock_epics.caget.side_effect = Exception("Connection refused")
        result = json.loads(tool._run("BAD:PV"))
        assert result["count"] == 0
        assert len(result["errors"]) == 1
        assert "Connection refused" in result["errors"][0]["error"]


class TestEmptyInput:
    def test_no_pv_names(self, tool):
        with patch("src.tools.pv_reader._EPICS_AVAILABLE", True):
            result = json.loads(tool._run(""))
            assert "error" in result
            assert "No PV" in result["error"]


class TestNoWriteImports:
    def test_no_write_functions(self):
        source_file = Path(__file__).parent.parent.parent / "src" / "tools" / "pv_reader.py"
        source = source_file.read_text()
        assert "caput" not in source
        assert "ca.put" not in source
        assert "PV.put" not in source
        assert ".put(" not in source
