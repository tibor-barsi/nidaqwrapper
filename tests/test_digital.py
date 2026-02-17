"""Tests for nidaqwrapper.digital module (DigitalInput / DigitalOutput).

Tests are organized by task group from the OpenSpec change, with separate
test classes for DigitalInput and DigitalOutput.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: create a mock nidaqmx module for patching into digital.py
# ---------------------------------------------------------------------------

def _make_mock_nidaqmx(
    device_names: list[str] | None = None,
    task_names: list[str] | None = None,
):
    """Build a mock nidaqmx module suitable for ``digital.py`` imports.

    Returns (mock_nidaqmx, mock_task_instance) so tests can inspect the
    nidaqmx.Task() that ``initiate()`` would create.
    """
    if device_names is None:
        device_names = ["Dev1", "Dev2"]
    if task_names is None:
        task_names = []

    mock_nidaqmx = MagicMock()

    # System
    devices = []
    for name in device_names:
        dev = MagicMock()
        dev.name = name
        devices.append(dev)

    system = MagicMock()
    system.devices = devices

    # Tasks in NI MAX
    tasks_collection = MagicMock()
    tasks_collection.task_names = task_names
    system.tasks = tasks_collection

    mock_nidaqmx.system.System.local.return_value = system

    # Task constructor returns a mock task instance
    mock_task_instance = MagicMock()
    mock_task_instance.di_channels = MagicMock()
    mock_task_instance.do_channels = MagicMock()
    mock_task_instance.timing = MagicMock()
    mock_task_instance.number_of_channels = 1
    mock_nidaqmx.Task.return_value = mock_task_instance

    # Constants
    mock_nidaqmx.constants.AcquisitionType.CONTINUOUS = "CONTINUOUS"
    mock_nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES = "CHAN_FOR_ALL_LINES"
    mock_nidaqmx.constants.READ_ALL_AVAILABLE = -1

    return mock_nidaqmx, mock_task_instance


# ═══════════════════════════════════════════════════════════════════════════
# DigitalInput Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDigitalInputConstructor:
    """Task group 1: DigitalInput constructor tests."""

    def test_on_demand_mode_defaults(self):
        """DigitalInput without sample_rate has mode='on_demand', sample_rate=None."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_read")
            assert di.task_name == "di_read"
            assert di.sample_rate is None
            assert di.mode == "on_demand"
            assert di.channels == {}

    def test_clocked_mode(self):
        """DigitalInput with sample_rate=1000 has mode='clocked'."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_clocked", sample_rate=1000)
            assert di.sample_rate == 1000
            assert di.mode == "clocked"

    def test_device_discovery(self):
        """DigitalInput discovers connected NI-DAQmx devices."""
        mock_nidaqmx, _ = _make_mock_nidaqmx(device_names=["cDAQ1Mod1", "cDAQ1Mod2"])
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_dev")
            assert "cDAQ1Mod1" in di.device_list
            assert "cDAQ1Mod2" in di.device_list

    def test_logger_name(self):
        """DigitalInput logger is named 'nidaqwrapper.digital'."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_log")
            assert di.logger.name == "nidaqwrapper.digital"

    def test_reject_duplicate_task_name(self):
        """DigitalInput raises ValueError if task_name already exists in NI MAX."""
        mock_nidaqmx, _ = _make_mock_nidaqmx(task_names=["existing_task"])
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            with pytest.raises(ValueError, match="existing_task"):
                DigitalInput(task_name="existing_task")


class TestDigitalInputAddChannel:
    """Task group 2: DigitalInput.add_channel() tests."""

    def _make_di(self):
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        patcher = patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system})
        patcher.start()
        from nidaqwrapper.digital import DigitalInput

        di = DigitalInput(task_name="di_ch")
        return di, patcher

    def test_add_single_line(self):
        """add_channel stores single line specification."""
        di, patcher = self._make_di()
        try:
            di.add_channel("btn_1", lines="Dev1/port0/line0")
            assert "btn_1" in di.channels
            assert di.channels["btn_1"]["lines"] == "Dev1/port0/line0"
        finally:
            patcher.stop()

    def test_add_line_range(self):
        """add_channel stores line range specification."""
        di, patcher = self._make_di()
        try:
            di.add_channel("switches", lines="Dev1/port0/line0:3")
            assert di.channels["switches"]["lines"] == "Dev1/port0/line0:3"
        finally:
            patcher.stop()

    def test_add_full_port(self):
        """add_channel stores full port specification."""
        di, patcher = self._make_di()
        try:
            di.add_channel("port0", lines="Dev1/port0")
            assert di.channels["port0"]["lines"] == "Dev1/port0"
        finally:
            patcher.stop()

    def test_reject_duplicate_channel_name(self):
        """add_channel raises ValueError on duplicate channel name."""
        di, patcher = self._make_di()
        try:
            di.add_channel("btn_1", lines="Dev1/port0/line0")
            with pytest.raises(ValueError, match="btn_1"):
                di.add_channel("btn_1", lines="Dev1/port0/line1")
        finally:
            patcher.stop()

    def test_reject_duplicate_lines(self):
        """add_channel raises ValueError when lines are already in use."""
        di, patcher = self._make_di()
        try:
            di.add_channel("ch1", lines="Dev1/port0/line0")
            with pytest.raises(ValueError, match="Dev1/port0/line0"):
                di.add_channel("ch2", lines="Dev1/port0/line0")
        finally:
            patcher.stop()
