"""Tests for nidaqwrapper.task_input module (NITask class).

Tests are organized by task group following openspec/changes/task-input/tasks.md.
All tests use mocked nidaqmx — no hardware required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — mock UNITS dict with __objclass__ for channel dispatch
# ---------------------------------------------------------------------------

def _make_mock_units():
    """Create a mock UNITS dict whose values have __objclass__ for dispatch."""
    units = {}

    def _make_const(objclass_name, value):
        c = MagicMock()
        c.__objclass__ = type(objclass_name, (), {"__name__": objclass_name})
        c._value_ = value
        c.__repr__ = lambda self: value
        return c

    # AccelSensitivityUnits
    units["mV/g"] = _make_const("AccelSensitivityUnits", "MILLIVOLTS_PER_G")
    units["mV/m/s**2"] = _make_const(
        "AccelSensitivityUnits", "MILLIVOLTS_PER_METERS_PER_SECOND_SQUARED"
    )

    # AccelUnits
    units["g"] = _make_const("AccelUnits", "ACCEL_G")
    units["m/s**2"] = _make_const("AccelUnits", "METERS_PER_SECOND_SQUARED")

    # ForceIEPESensorSensitivityUnits
    units["mV/N"] = _make_const(
        "ForceIEPESensorSensitivityUnits", "MILLIVOLTS_PER_NEWTON"
    )

    # ForceUnits
    units["N"] = _make_const("ForceUnits", "NEWTONS")

    # VoltageUnits
    units["V"] = _make_const("VoltageUnits", "VOLTS")

    return units


MOCK_UNITS = _make_mock_units()


# ---------------------------------------------------------------------------
# 1. NITask Constructor
# ---------------------------------------------------------------------------


class TestNITaskConstructor:
    """Tests for NITask.__init__()."""

    def test_task_name_stored(self, mock_system):
        """task_name is stored on the instance."""
        system = mock_system(task_names=[])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("vibration_test", sample_rate=25600)
            assert task.task_name == "vibration_test"

    def test_sample_rate_stored(self, mock_system):
        """sample_rate is stored on the instance."""
        system = mock_system(task_names=[])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task.sample_rate == 25600

    def test_channels_empty_dict(self, mock_system):
        """channels is an empty dict on a new task."""
        system = mock_system(task_names=[])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task.channels == {}

    def test_channel_list_empty(self, mock_system):
        """channel_list is an empty list on a new task."""
        system = mock_system(task_names=[])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task.channel_list == []

    def test_number_of_ch_zero(self, mock_system):
        """number_of_ch is 0 on a new task."""
        system = mock_system(task_names=[])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task.number_of_ch == 0

    def test_sample_mode_continuous(self, mock_system, mock_constants):
        """sample_mode defaults to AcquisitionType.CONTINUOUS."""
        system = mock_system(task_names=[])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.constants", mock_constants),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task.sample_mode == mock_constants.AcquisitionType.CONTINUOUS

    def test_device_list_populated(self, mock_system):
        """device_list contains device name strings from the system."""
        system = mock_system(
            devices=[("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        )
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task.device_list == ["cDAQ1Mod1", "cDAQ1Mod2"]

    def test_device_product_type_populated(self, mock_system):
        """device_product_type contains product type strings."""
        system = mock_system(
            devices=[("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        )
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task.device_product_type == ["NI 9234", "NI 9263"]

    def test_logger_name(self, mock_system):
        """Logger is named 'nidaqwrapper.task'."""
        system = mock_system(task_names=[])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task._logger.name == "nidaqwrapper.task"

    def test_duplicate_task_name_raises_valueerror(self, mock_system):
        """Constructor raises ValueError when task_name already exists in NI MAX."""
        system = mock_system(task_names=["existing_task"])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            with pytest.raises(ValueError, match="already"):
                NITask("existing_task", sample_rate=25600)

    def test_settings_none_by_default(self, mock_system):
        """settings is None when no settings_file is provided."""
        system = mock_system(task_names=[])
        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test", sample_rate=25600)
            assert task.settings is None
