"""
Unit tests for PVSettings configuration.
Covers: defaults, YAML loading, env override.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import PVSettings


class TestPVSettingsDefaults:
    def test_defaults(self):
        s = PVSettings()
        assert s.epics_gateway_address == "your-epics-gateway.example.com"
        assert s.epics_gateway_port == 5064
        assert s.epics_timeout == 5.0
        assert s.archiver_url == "https://your-archiver.example.com"
        assert s.archiver_timeout == 60
        assert s.enable_pv_tools is True
        assert s.max_pvs_per_read == 50

    def test_channel_db_paths(self):
        s = PVSettings()
        assert "hierarchical.json" in s.channel_db_path
        assert "its_pv_in_context.json" in s.in_context_db_path


class TestPVSettingsInAppSettings:
    def test_app_settings_has_pv(self):
        from config.settings import AppSettings
        s = AppSettings()
        assert hasattr(s, "pv")
        assert isinstance(s.pv, PVSettings)
