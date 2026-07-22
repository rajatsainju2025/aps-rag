"""
Unit tests for ArchiverReaderTool.
Covers: time parsing, series parsing, fetch logic, summary stats.
No live archiver connection — all HTTP calls are mocked.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tools.archiver_reader import ArchiverReaderTool


@pytest.fixture
def tool():
    return ArchiverReaderTool(
        archiver_url="https://your-archiver.example.com",
        timeout=30,
    )


class TestParseRelativeTime:
    def test_hours(self, tool):
        result = tool._parse_relative_time("1h")
        expected = datetime.now(timezone.utc) - timedelta(hours=1)
        assert abs((result - expected).total_seconds()) < 5

    def test_days(self, tool):
        result = tool._parse_relative_time("7d")
        expected = datetime.now(timezone.utc) - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 5

    def test_minutes(self, tool):
        result = tool._parse_relative_time("30m")
        expected = datetime.now(timezone.utc) - timedelta(minutes=30)
        assert abs((result - expected).total_seconds()) < 5

    def test_invalid_format(self, tool):
        with pytest.raises(ValueError):
            tool._parse_relative_time("invalid")


class TestParseSeries:
    def test_basic_parse(self, tool):
        payload = [
            {
                "meta": {"name": "PV1"},
                "data": [{"secs": 1700000000, "nanos": 0, "val": 42.0}],
            }
        ]
        series = tool._parse_series(payload)
        assert "PV1" in series
        assert len(series["PV1"]) == 1
        assert series["PV1"].iloc[0] == 42.0

    def test_list_val_unwrap(self, tool):
        payload = [
            {
                "meta": {"name": "PV2"},
                "data": [{"secs": 1700000000, "nanos": 0, "val": [42.0]}],
            }
        ]
        series = tool._parse_series(payload)
        assert series["PV2"].iloc[0] == 42.0

    def test_empty_payload(self, tool):
        assert tool._parse_series([]) == {}
        assert tool._parse_series(None) == {}

    def test_multiple_points(self, tool):
        payload = [
            {
                "meta": {"name": "PV3"},
                "data": [
                    {"secs": 1700000000, "nanos": 0, "val": 1.0},
                    {"secs": 1700000001, "nanos": 0, "val": 2.0},
                    {"secs": 1700000002, "nanos": 0, "val": 3.0},
                ],
            }
        ]
        series = tool._parse_series(payload)
        assert len(series["PV3"]) == 3


class TestBuildRetrievalUrl:
    def test_base_url(self, tool):
        url = tool._build_retrieval_url("https://your-archiver.example.com")
        assert url == "https://your-archiver.example.com/retrieval/data/getData.json"

    def test_url_with_trailing_slash(self, tool):
        url = tool._build_retrieval_url("https://your-archiver.example.com/")
        assert url == "https://your-archiver.example.com/retrieval/data/getData.json"

    def test_url_already_has_retrieval(self, tool):
        url = tool._build_retrieval_url("https://your-archiver.example.com/retrieval")
        assert "retrieval/data/getData.json" in url


class TestToISO8601UTC:
    def test_naive_datetime(self, tool):
        dt = datetime(2025, 1, 1, 12, 0, 0)
        result = tool._to_iso8601_utc(dt)
        assert result.endswith("Z")

    def test_aware_datetime(self, tool):
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = tool._to_iso8601_utc(dt)
        assert result.endswith("Z")
        assert "2025-01-01" in result


class TestFetchData:
    @patch("src.tools.archiver_reader._requests")
    def test_batch_success(self, mock_requests, tool):
        import pandas as pd

        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "meta": {"name": "PV1"},
                "data": [{"secs": 1700000000, "nanos": 0, "val": 10.0}],
            },
            {
                "meta": {"name": "PV2"},
                "data": [{"secs": 1700000000, "nanos": 0, "val": 20.0}],
            },
        ]
        mock_session.get.return_value = mock_response

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        df = tool._fetch_data(["PV1", "PV2"], start, end)
        assert "PV1" in df.columns
        assert "PV2" in df.columns

    @patch("src.tools.archiver_reader._requests")
    def test_batch_fallback(self, mock_requests, tool):
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session

        batch_response = MagicMock()
        batch_response.raise_for_status.side_effect = Exception("Batch failed")

        individual_response = MagicMock()
        individual_response.json.return_value = [
            {
                "meta": {"name": "PV1"},
                "data": [{"secs": 1700000000, "nanos": 0, "val": 10.0}],
            },
        ]

        mock_session.get.side_effect = [batch_response, individual_response]

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        df = tool._fetch_data(["PV1"], start, end)
        assert "PV1" in df.columns

    @patch("src.tools.archiver_reader._requests")
    def test_all_fail(self, mock_requests, tool):
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Failed")
        mock_session.get.return_value = mock_response

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        df = tool._fetch_data(["PV1"], start, end)
        assert df.empty


class TestRunSummaryStats:
    @patch("src.tools.archiver_reader._requests")
    def test_summary_output(self, mock_requests, tool):
        # Stage C: ``tool._run`` now dispatches over MCP to a fresh subprocess,
        # where this in-process mock can't reach. Test ``_run_local`` directly —
        # that's the function the MCP server calls, so it's the right contract
        # to mock.
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session
        mock_response = MagicMock()

        data_points = [
            {"secs": 1700000000 + i, "nanos": 0, "val": float(i)}
            for i in range(100)
        ]
        mock_response.json.return_value = [
            {"meta": {"name": "PV1"}, "data": data_points}
        ]
        mock_session.get.return_value = mock_response

        result = json.loads(tool._run_local("PV1", "1h"))
        assert len(result["data"]) == 1
        summary = result["data"][0]
        assert summary["pv"] == "PV1"
        assert summary["count"] == 100
        assert "min" in summary
        assert "max" in summary
        assert "mean" in summary
        assert summary["min"] == 0.0
        assert summary["max"] == 99.0


class TestRunErrors:
    def test_deps_unavailable(self, tool):
        with patch("src.tools.archiver_reader._DEPS_AVAILABLE", False):
            result = json.loads(tool._run("PV1", "1h"))
            assert "error" in result

    def test_no_pv_names(self, tool):
        result = json.loads(tool._run("", "1h"))
        assert "error" in result
        assert "No PV" in result["error"]

    def test_invalid_start_time(self, tool):
        result = json.loads(tool._run("PV1", "invalid_time"))
        assert "error" in result
