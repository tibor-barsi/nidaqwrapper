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


# ---------------------------------------------------------------------------
# Helper — build a fresh NITask with two devices, no existing MAX tasks
# ---------------------------------------------------------------------------

def _make_task(mock_system, sample_rate: float = 25600) -> "NITask":
    """Return a ready-to-use NITask instance backed by two mock devices.

    Uses the default ``mock_system`` devices (cDAQ1Mod1 + cDAQ1Mod2) so
    device indices 0 and 1 are always valid.  Import must happen inside the
    patch context; callers receive the already-constructed object.
    """
    system = mock_system(task_names=[])
    with (
        patch("nidaqwrapper.task_input.nidaqmx.system.System.local", return_value=system),
        patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
    ):
        from nidaqwrapper.task_input import NITask

        return NITask("test", sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# 2. Accelerometer channels
# ---------------------------------------------------------------------------


class TestAccelChannel:
    """Tests for add_channel() with accelerometer-type sensors."""

    def _task(self, mock_system):
        """Return a fresh NITask with two mock devices."""
        return _make_task(mock_system)

    def test_accel_channel_stored_in_channels(self, mock_system):
        """Accelerometer channel config is stored under its name in task.channels."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )

        assert "accel_x" in task.channels
        cfg = task.channels["accel_x"]
        assert cfg["device_ind"] == 0
        assert cfg["channel_ind"] == 0
        assert cfg["sensitivity"] == 100.0
        assert cfg["sensitivity_units"] == MOCK_UNITS["mV/g"]
        assert cfg["units"] == MOCK_UNITS["g"]

    def test_accel_channel_updates_channel_list(self, mock_system):
        """channel_list grows by one after adding an accelerometer channel."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )

        assert task.channel_list == ["accel_x"]

    def test_accel_channel_increments_number_of_ch(self, mock_system):
        """number_of_ch increments for each channel added."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )
            assert task.number_of_ch == 1

            task.add_channel(
                "accel_y",
                device_ind=0,
                channel_ind=1,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )
            assert task.number_of_ch == 2

    def test_accel_channel_ms2_units(self, mock_system):
        """Accelerometer channel accepts mV/m/s**2 sensitivity units and m/s**2 units."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_z",
                device_ind=0,
                channel_ind=2,
                sensitivity=10.204,
                sensitivity_units="mV/m/s**2",
                units="m/s**2",
            )

        cfg = task.channels["accel_z"]
        assert cfg["sensitivity_units"] == MOCK_UNITS["mV/m/s**2"]
        assert cfg["units"] == MOCK_UNITS["m/s**2"]

    def test_accel_channel_custom_min_max(self, mock_system):
        """min_val and max_val are stored when explicitly provided."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
                min_val=-50.0,
                max_val=50.0,
            )

        cfg = task.channels["accel_x"]
        assert cfg["min_val"] == -50.0
        assert cfg["max_val"] == 50.0

    def test_accel_channel_min_val_zero_stored(self, mock_system):
        """min_val=0.0 is stored correctly and not silently dropped (LDAQ bug fix).

        In the original LDAQ implementation, ``if min_val:`` treated 0.0 as
        falsy and fell back to a default.  This test ensures that 0.0 is an
        intentional, valid value that must be preserved.
        """
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
                min_val=0.0,
                max_val=100.0,
            )

        # Explicitly check identity/equality — must be 0.0, not None or default
        assert task.channels["accel_x"]["min_val"] == 0.0
        assert task.channels["accel_x"]["min_val"] is not None

    def test_accel_channel_serial_nr_stored(self, mock_system):
        """serial_nr is stored in the channel config when provided."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
                serial_nr="SN123456",
            )

        assert task.channels["accel_x"]["serial_nr"] == "SN123456"

    def test_accel_channel_config_has_all_required_keys(self, mock_system):
        """Channel config dict always contains the full set of expected keys."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )

        cfg = task.channels["accel_x"]
        expected_keys = {
            "device_ind",
            "channel_ind",
            "sensitivity",
            "sensitivity_units",
            "units",
            "serial_nr",
            "scale",
            "min_val",
            "max_val",
            "custom_scale_name",
        }
        assert expected_keys.issubset(cfg.keys())


# ---------------------------------------------------------------------------
# 3. Force channels
# ---------------------------------------------------------------------------


class TestForceChannel:
    """Tests for add_channel() with force (IEPE) sensors."""

    def _task(self, mock_system):
        """Return a fresh NITask with two mock devices."""
        return _make_task(mock_system)

    def test_force_channel_stored_in_channels(self, mock_system):
        """Force channel config is stored under its name in task.channels."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "force_1",
                device_ind=0,
                channel_ind=0,
                sensitivity=22.5,
                sensitivity_units="mV/N",
                units="N",
            )

        assert "force_1" in task.channels
        cfg = task.channels["force_1"]
        assert cfg["device_ind"] == 0
        assert cfg["channel_ind"] == 0
        assert cfg["sensitivity"] == 22.5
        assert cfg["sensitivity_units"] == MOCK_UNITS["mV/N"]
        assert cfg["units"] == MOCK_UNITS["N"]

    def test_force_channel_updates_channel_list(self, mock_system):
        """channel_list contains the force channel name after adding."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "force_1",
                device_ind=0,
                channel_ind=0,
                sensitivity=22.5,
                sensitivity_units="mV/N",
                units="N",
            )

        assert "force_1" in task.channel_list

    def test_force_channel_custom_min_max(self, mock_system):
        """min_val and max_val are stored correctly for a force channel."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "force_1",
                device_ind=0,
                channel_ind=0,
                sensitivity=22.5,
                sensitivity_units="mV/N",
                units="N",
                min_val=-500.0,
                max_val=500.0,
            )

        cfg = task.channels["force_1"]
        assert cfg["min_val"] == -500.0
        assert cfg["max_val"] == 500.0

    def test_force_channel_min_val_zero_stored(self, mock_system):
        """min_val=0.0 is stored correctly for a force channel (not treated as falsy)."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "force_1",
                device_ind=0,
                channel_ind=0,
                sensitivity=22.5,
                sensitivity_units="mV/N",
                units="N",
                min_val=0.0,
                max_val=1000.0,
            )

        assert task.channels["force_1"]["min_val"] == 0.0
        assert task.channels["force_1"]["min_val"] is not None


# ---------------------------------------------------------------------------
# 4. Voltage channels
# ---------------------------------------------------------------------------


class TestVoltageChannel:
    """Tests for add_channel() with voltage-type channels."""

    def _task(self, mock_system):
        """Return a fresh NITask with two mock devices."""
        return _make_task(mock_system)

    def test_voltage_channel_no_sensitivity_required(self, mock_system):
        """Voltage channel (units='V') can be added without sensitivity args."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=0,
                units="V",
            )

        assert "voltage_1" in task.channels
        assert task.channels["voltage_1"]["units"] == MOCK_UNITS["V"]

    def test_voltage_channel_float_scale_stored(self, mock_system):
        """A float scale is stored in the channel config."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=0,
                units="V",
                scale=2500.0,
            )

        assert task.channels["voltage_1"]["scale"] == 2500.0

    def test_voltage_channel_tuple_scale_stored(self, mock_system):
        """A tuple scale (gain, offset) is stored in the channel config."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=0,
                units="V",
                scale=(2500.0, -100.0),
            )

        assert task.channels["voltage_1"]["scale"] == (2500.0, -100.0)

    def test_voltage_channel_invalid_scale_type_raises_typeerror(self, mock_system):
        """A scale of unsupported type raises TypeError."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises(TypeError, match="scale"):
                task.add_channel(
                    "voltage_1",
                    device_ind=0,
                    channel_ind=0,
                    units="V",
                    scale="2500",  # string is not a valid scale type
                )

    def test_voltage_channel_scale_skips_sensitivity_validation(self, mock_system):
        """When scale is provided, missing sensitivity does not raise an error."""
        task = self._task(mock_system)
        # No sensitivity or sensitivity_units — should succeed because scale is given
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=0,
                units="V",
                scale=2500.0,
            )

        assert "voltage_1" in task.channels


# ---------------------------------------------------------------------------
# 5. Channel validation
# ---------------------------------------------------------------------------


class TestChannelValidation:
    """Tests for add_channel() input validation and error handling."""

    def _task(self, mock_system):
        """Return a fresh NITask with two mock devices."""
        return _make_task(mock_system)

    def test_duplicate_channel_name_raises_valueerror(self, mock_system):
        """Adding a second channel with the same name raises ValueError."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )
            with pytest.raises(ValueError, match="accel_x"):
                task.add_channel(
                    "accel_x",
                    device_ind=0,
                    channel_ind=1,
                    sensitivity=100.0,
                    sensitivity_units="mV/g",
                    units="g",
                )

    def test_duplicate_device_channel_index_raises_valueerror(self, mock_system):
        """Adding two channels with the same (device_ind, channel_ind) raises ValueError."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )
            with pytest.raises(ValueError, match="already"):
                task.add_channel(
                    "accel_y",
                    device_ind=0,
                    channel_ind=0,  # same slot
                    sensitivity=100.0,
                    sensitivity_units="mV/g",
                    units="g",
                )

    def test_same_channel_ind_on_different_devices_is_allowed(self, mock_system):
        """The same channel_ind on different devices is a distinct physical channel."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )
            # device_ind=1, channel_ind=0 — different device, must not raise
            task.add_channel(
                "accel_y",
                device_ind=1,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )

        assert "accel_y" in task.channels

    def test_out_of_range_device_ind_raises_valueerror(self, mock_system):
        """device_ind beyond the available device list raises ValueError."""
        task = self._task(mock_system)
        # default mock_system has 2 devices: indices 0 and 1 are valid; 99 is not
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises(ValueError, match="device"):
                task.add_channel(
                    "accel_x",
                    device_ind=99,
                    channel_ind=0,
                    sensitivity=100.0,
                    sensitivity_units="mV/g",
                    units="g",
                )

    def test_missing_units_raises_valueerror(self, mock_system):
        """Calling add_channel() without units raises ValueError."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises((ValueError, TypeError)):
                task.add_channel(
                    "accel_x",
                    device_ind=0,
                    channel_ind=0,
                    sensitivity=100.0,
                    sensitivity_units="mV/g",
                    # units intentionally omitted
                )

    def test_invalid_units_without_scale_raises_valueerror(self, mock_system):
        """An unrecognised units string raises ValueError when no scale is given."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises(ValueError, match="units"):
                task.add_channel(
                    "accel_x",
                    device_ind=0,
                    channel_ind=0,
                    sensitivity=100.0,
                    sensitivity_units="mV/g",
                    units="furlongs_per_fortnight",  # not in UNITS
                )

    def test_invalid_sensitivity_units_without_scale_raises_valueerror(self, mock_system):
        """An unrecognised sensitivity_units string raises ValueError when no scale is provided."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises(ValueError, match="sensitivity_units"):
                task.add_channel(
                    "accel_x",
                    device_ind=0,
                    channel_ind=0,
                    sensitivity=100.0,
                    sensitivity_units="mV/parsec",  # not in UNITS
                    units="g",
                )

    def test_invalid_units_with_scale_does_not_raise(self, mock_system):
        """When scale is provided, units validation is skipped."""
        task = self._task(mock_system)
        # With a scale, units can be anything recognisable by NI as a custom scale
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            # Should not raise — scale bypasses the UNITS lookup
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=0,
                units="V",
                scale=2500.0,
            )

        assert "voltage_1" in task.channels

    def test_missing_sensitivity_for_non_voltage_non_scale_raises_valueerror(
        self, mock_system
    ):
        """Non-voltage channel without scale and without sensitivity raises ValueError."""
        task = self._task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises(ValueError, match="sensitivity"):
                task.add_channel(
                    "accel_x",
                    device_ind=0,
                    channel_ind=0,
                    # sensitivity intentionally omitted
                    sensitivity_units="mV/g",
                    units="g",
                )
