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


# ---------------------------------------------------------------------------
# Helpers — build a mock nidaqmx task for initiate() tests
# ---------------------------------------------------------------------------


def _make_nidaqmx_task(samp_clk_rate: float = 25600) -> MagicMock:
    """Return a mock nidaqmx.task.Task pre-configured for initiate() tests.

    Parameters
    ----------
    samp_clk_rate : float
        Value returned by ``task._timing.samp_clk_rate`` after timing is
        configured.  Defaults to 25600 to match the standard test rate.

    Returns
    -------
    MagicMock
        Mock nidaqmx Task with ai_channels, timing, and lifecycle methods.
    """
    mock = MagicMock()
    mock._timing.samp_clk_rate = samp_clk_rate
    return mock


# ---------------------------------------------------------------------------
# 7. initiate() method
# ---------------------------------------------------------------------------


class TestInitiate:
    """Tests for NITask.initiate() — Task 7.

    All tests mock nidaqmx.task.Task and related system calls so that no
    hardware is required.  The mock pattern is:

    1. Build NITask via ``_make_task()`` (patches System.local at construction).
    2. Add channels via ``add_channel()`` inside a UNITS patch.
    3. Call ``initiate()`` inside patches for Task constructor, System.local,
       UNITS, and constants.
    """

    def _task_with_accel(self, mock_system, sample_rate: float = 25600) -> "NITask":
        """Return an NITask with a single accelerometer channel."""
        task = _make_task(mock_system, sample_rate=sample_rate)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )
        return task

    def _task_with_force(self, mock_system) -> "NITask":
        """Return an NITask with a single force channel."""
        task = _make_task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "force_1",
                device_ind=0,
                channel_ind=0,
                sensitivity=22.5,
                sensitivity_units="mV/N",
                units="N",
            )
        return task

    def _task_with_voltage(self, mock_system) -> "NITask":
        """Return an NITask with a single plain voltage channel."""
        task = _make_task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=0,
                units="V",
            )
        return task

    # ------------------------------------------------------------------
    # nidaqmx.task.Task construction
    # ------------------------------------------------------------------

    def test_initiate_creates_nidaqmx_task(self, mock_system, mock_constants):
        """initiate() creates a nidaqmx.task.Task with the correct task name."""
        nitask = self._task_with_accel(mock_system)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ) as mock_task_cls,
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate()

        mock_task_cls.assert_called_once_with(new_task_name="test")

    # ------------------------------------------------------------------
    # Channel dispatch — each sensor type calls the correct add_ai_* method
    # ------------------------------------------------------------------

    def test_initiate_adds_accel_channel(self, mock_system, mock_constants):
        """initiate() calls add_ai_accel_chan() with correct args for an accel channel.

        Verified args: physical channel string (includes device name),
        sensitivity value, sensitivity_units constant, and units constant.
        """
        nitask = self._task_with_accel(mock_system)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate()

        call_info = mock_ni_task.ai_channels.add_ai_accel_chan.call_args
        assert call_info is not None, "add_ai_accel_chan() was never called"

        args, kwargs = call_info
        all_values = list(args) + list(kwargs.values())

        # Physical channel string must reference the correct device
        assert any("cDAQ1Mod1" in str(v) for v in all_values), (
            "Physical channel string must reference device cDAQ1Mod1"
        )
        # Sensitivity, sensitivity_units, and units must be forwarded
        assert 100.0 in all_values or kwargs.get("sensitivity") == 100.0
        assert (
            MOCK_UNITS["mV/g"] in all_values
            or kwargs.get("sensitivity_units") == MOCK_UNITS["mV/g"]
        )
        assert (
            MOCK_UNITS["g"] in all_values
            or kwargs.get("units") == MOCK_UNITS["g"]
        )

    def test_initiate_adds_force_channel(self, mock_system, mock_constants):
        """initiate() calls add_ai_force_iepe_chan() with correct args for a force channel."""
        nitask = self._task_with_force(mock_system)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate()

        call_info = mock_ni_task.ai_channels.add_ai_force_iepe_chan.call_args
        assert call_info is not None, "add_ai_force_iepe_chan() was never called"

        args, kwargs = call_info
        all_values = list(args) + list(kwargs.values())
        assert any("cDAQ1Mod1" in str(v) for v in all_values)
        assert (
            MOCK_UNITS["mV/N"] in all_values
            or kwargs.get("sensitivity_units") == MOCK_UNITS["mV/N"]
        )
        assert (
            MOCK_UNITS["N"] in all_values
            or kwargs.get("units") == MOCK_UNITS["N"]
        )

    def test_initiate_adds_voltage_channel(self, mock_system, mock_constants):
        """initiate() calls add_ai_voltage_chan() for a plain voltage channel.

        Sensitivity args must NOT be forwarded — voltage channels carry no
        sensor sensitivity.
        """
        nitask = self._task_with_voltage(mock_system)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate()

        assert mock_ni_task.ai_channels.add_ai_voltage_chan.called, (
            "add_ai_voltage_chan() was never called"
        )
        _, kwargs = mock_ni_task.ai_channels.add_ai_voltage_chan.call_args
        assert "sensitivity" not in kwargs
        assert "sensitivity_units" not in kwargs

    # ------------------------------------------------------------------
    # Custom scale — voltage channel with float or tuple scale
    # ------------------------------------------------------------------

    def test_initiate_voltage_with_float_custom_scale(self, mock_system, mock_constants):
        """initiate() calls create_lin_scale() and uses FROM_CUSTOM_SCALE for float scale.

        A float scale value (e.g. 2500.0) means slope=value, y_intercept=0.
        The voltage channel must be created with units=FROM_CUSTOM_SCALE.
        """
        task = _make_task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=0,
                units="V",
                scale=2500.0,
            )
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale"
            ) as mock_scale,
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            task.initiate()

        mock_scale.assert_called_once()
        scale_args, scale_kwargs = mock_scale.call_args
        all_scale_values = list(scale_args) + list(scale_kwargs.values())

        # slope must be 2500.0
        assert 2500.0 in all_scale_values or scale_kwargs.get("slope") == 2500.0, (
            "slope=2500.0 must be passed to create_lin_scale()"
        )
        # y_intercept must be 0 for a float (no-offset) scale
        assert (
            0 in all_scale_values
            or 0.0 in all_scale_values
            or scale_kwargs.get("y_intercept") == 0
        ), "y_intercept=0 must be passed to create_lin_scale() for a float scale"

        # Voltage channel must use FROM_CUSTOM_SCALE
        assert mock_ni_task.ai_channels.add_ai_voltage_chan.called
        chan_args, chan_kwargs = mock_ni_task.ai_channels.add_ai_voltage_chan.call_args
        all_chan_values = list(chan_args) + list(chan_kwargs.values())
        assert (
            mock_constants.VoltageUnits.FROM_CUSTOM_SCALE in all_chan_values
            or chan_kwargs.get("units") == mock_constants.VoltageUnits.FROM_CUSTOM_SCALE
        ), "add_ai_voltage_chan() must use FROM_CUSTOM_SCALE when a custom scale is set"

    def test_initiate_voltage_with_tuple_custom_scale(self, mock_system, mock_constants):
        """initiate() passes both slope and y_intercept to create_lin_scale() for tuple scale."""
        task = _make_task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=0,
                units="V",
                scale=(2500.0, -100.0),
            )
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale"
            ) as mock_scale,
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            task.initiate()

        mock_scale.assert_called_once()
        scale_args, scale_kwargs = mock_scale.call_args
        all_scale_values = list(scale_args) + list(scale_kwargs.values())

        assert 2500.0 in all_scale_values or scale_kwargs.get("slope") == 2500.0, (
            "slope=2500.0 must be passed to create_lin_scale()"
        )
        assert -100.0 in all_scale_values or scale_kwargs.get("y_intercept") == -100.0, (
            "y_intercept=-100.0 must be passed to create_lin_scale()"
        )

    # ------------------------------------------------------------------
    # Timing configuration
    # ------------------------------------------------------------------

    def test_initiate_configures_timing(self, mock_system, mock_constants):
        """initiate() calls cfg_samp_clk_timing() with correct rate and CONTINUOUS mode."""
        nitask = self._task_with_accel(mock_system, sample_rate=25600)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task(samp_clk_rate=25600)

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate()

        assert mock_ni_task.timing.cfg_samp_clk_timing.called, (
            "cfg_samp_clk_timing() was never called"
        )
        timing_args, timing_kwargs = mock_ni_task.timing.cfg_samp_clk_timing.call_args
        all_timing_values = list(timing_args) + list(timing_kwargs.values())

        assert 25600 in all_timing_values or timing_kwargs.get("rate") == 25600, (
            "rate=25600 must be passed to cfg_samp_clk_timing()"
        )
        continuous = mock_constants.AcquisitionType.CONTINUOUS
        assert continuous in all_timing_values or timing_kwargs.get("sample_mode") == continuous, (
            "sample_mode=CONTINUOUS must be passed to cfg_samp_clk_timing()"
        )

    # ------------------------------------------------------------------
    # Sample rate validation
    # ------------------------------------------------------------------

    def test_initiate_validates_sample_rate_pass(self, mock_system, mock_constants):
        """initiate() does not raise when the driver echoes back the requested rate."""
        nitask = self._task_with_accel(mock_system, sample_rate=25600)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task(samp_clk_rate=25600)

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            # Exact match — must complete without raising
            nitask.initiate()

    def test_initiate_validates_sample_rate_fail(self, mock_system, mock_constants):
        """initiate() raises ValueError when the driver coerces the rate to a different value.

        Some NI devices only support discrete sample rates.  When the actual
        committed rate differs from the requested rate, NITask must raise
        ValueError to prevent silent data corruption.
        """
        nitask = self._task_with_accel(mock_system, sample_rate=25600)
        system = mock_system(task_names=[])
        # Simulate hardware coercing to a different supported rate
        mock_ni_task = _make_nidaqmx_task(samp_clk_rate=25000)

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            with pytest.raises(ValueError, match="[Ss]ample.?[Rr]ate|rate"):
                nitask.initiate()

    # ------------------------------------------------------------------
    # Task start / no-start
    # ------------------------------------------------------------------

    def test_initiate_starts_task_when_true(self, mock_system, mock_constants):
        """initiate(start_task=True) calls task.start() after configuration."""
        nitask = self._task_with_accel(mock_system)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate(start_task=True)

        mock_ni_task.start.assert_called_once()

    def test_initiate_does_not_start_when_false(self, mock_system, mock_constants):
        """initiate(start_task=False) configures the task but does NOT call task.start()."""
        nitask = self._task_with_accel(mock_system)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate(start_task=False)

        mock_ni_task.start.assert_not_called()

    def test_initiate_default_does_not_start_task(self, mock_system, mock_constants):
        """initiate() with no arguments does NOT auto-start (start_task defaults to False)."""
        nitask = self._task_with_accel(mock_system)
        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate()

        mock_ni_task.start.assert_not_called()

    # ------------------------------------------------------------------
    # Re-initiation — existing MAX task deleted first
    # ------------------------------------------------------------------

    def test_initiate_deletes_existing_task_before_reinitiation(
        self, mock_system, mock_constants
    ):
        """When task_name already exists in NI MAX, the old task is deleted before creating a new one.

        On re-initiation the implementation must look up the existing saved task
        in NI MAX and remove it (e.g. via ``.delete()``), then create the fresh
        nidaqmx.task.Task.  Without this step nidaqmx raises an error on
        duplicate task names.
        """
        nitask = self._task_with_accel(mock_system)
        # Simulate NI MAX already containing a saved task with the same name
        old_max_task = MagicMock()
        system = mock_system(task_names=["test"])
        # Attach the mock task object so it can be inspected after initiation
        system.tasks.__iter__ = MagicMock(
            side_effect=lambda: iter([old_max_task])
        )
        old_max_task._name = "test"
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            # Must NOT raise despite the colliding task name in NI MAX
            nitask.initiate()

        # The old saved task must have been deleted
        old_max_task.delete.assert_called_once()

    # ------------------------------------------------------------------
    # Mixed channel types in a single task
    # ------------------------------------------------------------------

    def test_initiate_mixed_channel_types(self, mock_system, mock_constants):
        """initiate() correctly dispatches all three channel types in a single task."""
        task = _make_task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )
            task.add_channel(
                "force_1",
                device_ind=0,
                channel_ind=1,
                sensitivity=22.5,
                sensitivity_units="mV/N",
                units="N",
            )
            task.add_channel(
                "voltage_1",
                device_ind=0,
                channel_ind=2,
                units="V",
            )

        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            task.initiate()

        assert mock_ni_task.ai_channels.add_ai_accel_chan.called, (
            "add_ai_accel_chan() not called for mixed task"
        )
        assert mock_ni_task.ai_channels.add_ai_force_iepe_chan.called, (
            "add_ai_force_iepe_chan() not called for mixed task"
        )
        assert mock_ni_task.ai_channels.add_ai_voltage_chan.called, (
            "add_ai_voltage_chan() not called for mixed task"
        )

    # ------------------------------------------------------------------
    # min_val=0.0 is not silently dropped (LDAQ bug fix)
    # ------------------------------------------------------------------

    def test_initiate_min_val_zero_passed_to_nidaqmx(self, mock_system, mock_constants):
        """min_val=0.0 is forwarded to add_ai_accel_chan() and not silently dropped.

        The original LDAQ implementation used ``if min_val:`` which evaluates
        0.0 as falsy, silently omitting the value and defaulting to the
        nidaqmx default.  nidaqwrapper must use ``is not None`` so that 0.0
        is recognised as an intentional, valid minimum value.
        """
        task = _make_task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "accel_x",
                device_ind=0,
                channel_ind=0,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
                min_val=0.0,
                max_val=50.0,
            )

        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            task.initiate()

        assert mock_ni_task.ai_channels.add_ai_accel_chan.called
        chan_args, chan_kwargs = mock_ni_task.ai_channels.add_ai_accel_chan.call_args
        all_chan_values = list(chan_args) + list(chan_kwargs.values())

        assert 0.0 in all_chan_values or chan_kwargs.get("min_val") == 0.0, (
            "min_val=0.0 was not forwarded to add_ai_accel_chan(); "
            "likely caused by 'if min_val:' truthiness bug in LDAQ"
        )


# ---------------------------------------------------------------------------
# 7b. Channel dispatch — internal _add_channel() routing
# ---------------------------------------------------------------------------


class TestInitiateChannelDispatch:
    """Tests for the internal channel-type dispatch logic in initiate() — Task 7.

    Verifies that the ``__objclass__.__name__`` inspection correctly routes
    each units constant to the matching ``add_ai_*`` nidaqmx method.
    """

    def _run_initiate_with_channel(
        self,
        mock_system,
        mock_constants,
        channel_kwargs: dict,
    ) -> MagicMock:
        """Construct an NITask, add one channel, run initiate(), return mock nidaqmx task.

        Parameters
        ----------
        mock_system : fixture
            The mock_system factory fixture.
        mock_constants : fixture
            The mock_constants fixture.
        channel_kwargs : dict
            Keyword arguments forwarded to ``add_channel()`` (excluding
            ``channel_name``, ``device_ind``, and ``channel_ind`` which are
            always ``'ch'``, ``0``, ``0``).

        Returns
        -------
        MagicMock
            The mock nidaqmx task on which assertions can be made.
        """
        nitask = _make_task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            nitask.add_channel("ch", device_ind=0, channel_ind=0, **channel_kwargs)

        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate()

        return mock_ni_task

    def test_accel_dispatch_uses_objclass(self, mock_system, mock_constants):
        """Units constant with AccelUnits __objclass__ dispatches to add_ai_accel_chan()."""
        mock_ni_task = self._run_initiate_with_channel(
            mock_system,
            mock_constants,
            {
                "sensitivity": 100.0,
                "sensitivity_units": "mV/g",
                "units": "g",  # MOCK_UNITS['g'].__objclass__.__name__ == 'AccelUnits'
            },
        )

        assert mock_ni_task.ai_channels.add_ai_accel_chan.called, (
            "AccelUnits must dispatch to add_ai_accel_chan()"
        )
        assert not mock_ni_task.ai_channels.add_ai_force_iepe_chan.called
        assert not mock_ni_task.ai_channels.add_ai_voltage_chan.called

    def test_force_dispatch_uses_objclass(self, mock_system, mock_constants):
        """Units constant with ForceUnits __objclass__ dispatches to add_ai_force_iepe_chan()."""
        mock_ni_task = self._run_initiate_with_channel(
            mock_system,
            mock_constants,
            {
                "sensitivity": 22.5,
                "sensitivity_units": "mV/N",
                "units": "N",  # MOCK_UNITS['N'].__objclass__.__name__ == 'ForceUnits'
            },
        )

        assert mock_ni_task.ai_channels.add_ai_force_iepe_chan.called, (
            "ForceUnits must dispatch to add_ai_force_iepe_chan()"
        )
        assert not mock_ni_task.ai_channels.add_ai_accel_chan.called
        assert not mock_ni_task.ai_channels.add_ai_voltage_chan.called

    def test_voltage_dispatch_uses_objclass(self, mock_system, mock_constants):
        """Units constant with VoltageUnits __objclass__ dispatches to add_ai_voltage_chan()."""
        mock_ni_task = self._run_initiate_with_channel(
            mock_system,
            mock_constants,
            {
                "units": "V",  # MOCK_UNITS['V'].__objclass__.__name__ == 'VoltageUnits'
            },
        )

        assert mock_ni_task.ai_channels.add_ai_voltage_chan.called, (
            "VoltageUnits must dispatch to add_ai_voltage_chan()"
        )
        assert not mock_ni_task.ai_channels.add_ai_accel_chan.called
        assert not mock_ni_task.ai_channels.add_ai_force_iepe_chan.called

    def test_custom_scale_dispatch_to_voltage(self, mock_system, mock_constants):
        """Channels with a custom scale dispatch to add_ai_voltage_chan().

        When a float or tuple scale is specified the channel always goes through
        the voltage path regardless of the units string, because nidaqmx
        requires ``FROM_CUSTOM_SCALE`` on a ``VoltageAIChannel``.
        """
        nitask = _make_task(mock_system)
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            nitask.add_channel(
                "ch",
                device_ind=0,
                channel_ind=0,
                units="V",
                scale=2500.0,
            )

        system = mock_system(task_names=[])
        mock_ni_task = _make_nidaqmx_task()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale"),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            nitask.initiate()

        assert mock_ni_task.ai_channels.add_ai_voltage_chan.called, (
            "Custom scale channels must use add_ai_voltage_chan()"
        )
        assert not mock_ni_task.ai_channels.add_ai_accel_chan.called
        assert not mock_ni_task.ai_channels.add_ai_force_iepe_chan.called


# ---------------------------------------------------------------------------
# 8. acquire_base()
# ---------------------------------------------------------------------------


class TestAcquireBase:
    """Tests for NITask.acquire_base() — reads all available samples from hardware."""

    def test_acquire_base_multi_channel(self, mock_system):
        """Multi-channel read returns a (n_channels, n_samples) numpy ndarray.

        nidaqmx returns a list-of-lists for multi-channel tasks:
        [[ch0_s0, ch0_s1, ...], [ch1_s0, ch1_s1, ...], ...].
        acquire_base() must convert this to a 2D numpy array.
        """
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task
        mock_hw_task.read.return_value = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]

        result = task.acquire_base()

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[1], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result[2], [7.0, 8.0, 9.0])

    def test_acquire_base_single_channel_reshaped(self, mock_system):
        """Single-channel read (nidaqmx returns 1D list) is reshaped to (1, n_samples).

        nidaqmx returns a flat list for single-channel tasks rather than a
        list-of-lists.  acquire_base() must reshape this 1D result to 2D so
        that callers always receive a consistent (n_channels, n_samples) shape.
        """
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task
        mock_hw_task.read.return_value = [1.0, 2.0, 3.0, 4.0]

        result = task.acquire_base()

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (1, 4)
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0, 4.0])

    def test_acquire_base_empty_buffer(self, mock_system):
        """Empty buffer returns an empty numpy array without raising."""
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task
        mock_hw_task.read.return_value = []

        result = task.acquire_base()

        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_acquire_base_calls_read_all_available(self, mock_system):
        """task.read() is called with number_of_samples_per_channel=READ_ALL_AVAILABLE.

        READ_ALL_AVAILABLE (-1) tells nidaqmx to drain every sample currently
        in the on-board buffer, which is the correct semantics for continuous
        acquisition polling.
        """
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task
        mock_hw_task.read.return_value = [[0.0], [0.0]]

        task.acquire_base()

        mock_hw_task.read.assert_called_once_with(
            number_of_samples_per_channel=-1  # nidaqmx.constants.READ_ALL_AVAILABLE
        )


# ---------------------------------------------------------------------------
# 9. clear_task()
# ---------------------------------------------------------------------------


class TestClearTask:
    """Tests for NITask.clear_task() — releases hardware resources."""

    def test_clear_task_calls_close(self, mock_system):
        """clear_task() calls task.close() on an initiated task."""
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task

        task.clear_task()

        mock_hw_task.close.assert_called_once()

    def test_clear_task_multiple_calls_no_error(self, mock_system):
        """Calling clear_task() twice raises no exception on the second call.

        After the first call the underlying task handle is gone; a second call
        must be a safe no-op rather than raising AttributeError or similar.
        """
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task

        task.clear_task()
        # Second call must not raise
        task.clear_task()

    def test_clear_task_never_initiated_no_error(self, mock_system):
        """clear_task() on a never-initiated task is a safe no-op.

        NITask does not set self.task in __init__, so clear_task() must
        guard against the attribute being absent entirely.
        """
        task = _make_task(mock_system)
        # Do NOT assign task.task — task was never initiated.
        # The call below must complete without raising.
        task.clear_task()

    def test_clear_task_exception_warns_not_propagated(self, mock_system):
        """clear_task() emits warning when task.close() raises."""
        import warnings

        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        mock_hw_task.close.side_effect = RuntimeError("close failed")
        task.task = mock_hw_task

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.clear_task()

        assert len(w) >= 1
        assert "close failed" in str(w[0].message)
        assert task.task is None


# ---------------------------------------------------------------------------
# 10. save()
# ---------------------------------------------------------------------------


class TestSave:
    """Tests for NITask.save() — saves task to NI MAX."""

    def test_save_calls_nidaqmx_save(self, mock_system):
        """save() calls nidaqmx task.save() with overwrite_existing_task=True."""
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task

        task.save()

        mock_hw_task.save.assert_called_once_with(
            overwrite_existing_task=True
        )

    def test_save_clears_task_by_default(self, mock_system):
        """save() calls clear_task() after saving when clear_task=True (default)."""
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task

        # Patch clear_task to observe whether it is invoked
        task.clear_task = MagicMock()

        task.save()

        task.clear_task.assert_called_once()

    def test_save_does_not_clear_when_false(self, mock_system):
        """save(clear_task=False) saves the task but leaves it open."""
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task

        task.clear_task = MagicMock()

        task.save(clear_task=False)

        # Task saved
        mock_hw_task.save.assert_called_once()
        # But NOT closed
        task.clear_task.assert_not_called()

    def test_save_auto_initiates(self, mock_system):
        """save() on an un-initiated task calls initiate(start_task=False) first.

        This is the LDAQ bug-fix verification: the original LDAQ code checked
        ``self.Task`` (capital T), which is always falsy for NITask and caused
        the auto-initiation branch to always trigger.  The correct attribute is
        ``self.task`` (lowercase).
        """
        task = _make_task(mock_system)
        # Simulate un-initiated state: self.task must be absent or None
        if hasattr(task, "task"):
            task.task = None

        task.initiate = MagicMock()

        # After initiate is called, save() will try to call self.task.save().
        # We simulate this by having initiate set task.task to a mock.
        mock_hw_task = MagicMock()

        def _fake_initiate(start_task: bool = True) -> None:
            task.task = mock_hw_task

        task.initiate.side_effect = _fake_initiate
        task.clear_task = MagicMock()

        task.save()

        task.initiate.assert_called_once_with(start_task=False)
        mock_hw_task.save.assert_called_once()


# ---------------------------------------------------------------------------
# 11. Context Manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for NITask.__enter__ and __exit__ (context manager protocol)."""

    def test_context_manager_enter_returns_self(self, mock_system):
        """__enter__ returns the NITask instance itself."""
        task = _make_task(mock_system)

        result = task.__enter__()

        assert result is task

    def test_context_manager_exit_calls_clear_task(self, mock_system):
        """__exit__ calls clear_task() to release hardware resources."""
        task = _make_task(mock_system)
        task.clear_task = MagicMock()

        task.__exit__(None, None, None)

        task.clear_task.assert_called_once()

    def test_context_manager_exception_still_clears(self, mock_system):
        """An exception raised inside the with-block still triggers __exit__.

        The context manager must guarantee cleanup even when the body raises,
        which is the primary reason for using the protocol in the first place.
        """
        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task

        cleared = []

        original_clear = task.clear_task

        def _tracking_clear():
            cleared.append(True)
            if hasattr(task, "task") and task.task is not None:
                task.task.close()

        task.clear_task = _tracking_clear

        with pytest.raises(ValueError):
            with task:
                raise ValueError("body error")

        assert cleared, "clear_task() must be called even when the body raises"

    def test_context_manager_cleanup_exception_warns_not_propagated(
        self, mock_system
    ):
        """A cleanup exception from clear_task() emits a warning, not raised.

        If the with-block body completed normally and clear_task() itself
        raises, that secondary exception must be swallowed and a
        warnings.warn() emitted so the user is notified without needing
        logging configuration.
        """
        import warnings

        task = _make_task(mock_system)
        mock_hw_task = MagicMock()
        task.task = mock_hw_task
        mock_hw_task.close.side_effect = RuntimeError("cleanup error")

        # The body raises nothing; the cleanup raises RuntimeError.
        # __exit__ must not let the RuntimeError propagate, but should warn.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.__exit__(None, None, None)  # must not raise

        assert len(w) >= 1
        assert "cleanup error" in str(w[0].message)


# ---------------------------------------------------------------------------
# 12. Introspection Properties
# ---------------------------------------------------------------------------


class TestIntrospection:
    """Tests for NITask introspection properties: channel_list, channel_info,
    device_list, number_of_ch."""

    def _add_accel(
        self, task: "NITask", name: str, device_ind: int = 0, channel_ind: int = 0
    ) -> None:
        """Add a minimal accelerometer channel to *task* for test setup."""
        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                name,
                device_ind=device_ind,
                channel_ind=channel_ind,
                sensitivity=100.0,
                sensitivity_units="mV/g",
                units="g",
            )

    def test_channel_list_returns_names_in_order(self, mock_system):
        """channel_list returns channel names in the order they were added."""
        task = _make_task(mock_system)
        self._add_accel(task, "ch_a", channel_ind=0)
        self._add_accel(task, "ch_b", channel_ind=1)
        self._add_accel(task, "ch_c", channel_ind=2)

        assert task.channel_list == ["ch_a", "ch_b", "ch_c"]

    def test_channel_info_returns_copy(self, mock_system):
        """channel_info returns a copy; mutating it does not affect internal state.

        Returning a shallow copy protects the internal channel configuration
        from accidental external modification.
        """
        task = _make_task(mock_system)
        self._add_accel(task, "ch_a", channel_ind=0)

        info = task.channel_info
        # Mutate the returned copy
        info["ch_a"]["sensitivity"] = 999.0

        # Internal state must be unchanged
        assert task.channels["ch_a"]["sensitivity"] == 100.0

    def test_channel_info_has_full_config(self, mock_system):
        """channel_info values contain all required configuration keys."""
        task = _make_task(mock_system)
        self._add_accel(task, "ch_a", channel_ind=0)

        info = task.channel_info
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
        assert expected_keys.issubset(info["ch_a"].keys())

    def test_device_list_returns_device_names(self, mock_system):
        """device_list returns the names of all devices discovered at construction."""
        task = _make_task(mock_system)

        # The default mock_system provides two devices
        assert task.device_list == ["cDAQ1Mod1", "cDAQ1Mod2"]

    def test_number_of_ch_returns_count(self, mock_system):
        """number_of_ch returns the exact count of configured channels."""
        task = _make_task(mock_system)
        assert task.number_of_ch == 0

        self._add_accel(task, "ch_a", channel_ind=0)
        assert task.number_of_ch == 1

        self._add_accel(task, "ch_b", channel_ind=1)
        assert task.number_of_ch == 2

    def test_introspection_empty_task(self, mock_system):
        """All introspection properties return empty/zero on a fresh task."""
        task = _make_task(mock_system)

        assert task.channel_list == []
        assert task.number_of_ch == 0
        assert task.channel_info == {}


# ---------------------------------------------------------------------------
# 13. Settings File Loading
# ---------------------------------------------------------------------------


class TestSettingsFile:
    """Tests for NITask settings file loading and serial_nr lookup — Task 13.

    Covers _read_settings_file() (called from __init__) and the serial_nr
    lookup path inside add_channel().
    """

    # Build a reusable DataFrame that exercises multiple sensor types.
    _MOCK_DF_DATA = {
        "serial_nr": ["SN001", "SN002"],
        "sensitivity": [100.0, 22.5],
        "sensitivity_units": ["mV/g", "mV/N"],
        "units": ["g", "N"],
    }

    def _mock_df(self):
        """Return a fresh mock settings DataFrame."""
        import pandas as pd

        return pd.DataFrame(self._MOCK_DF_DATA)

    # ------------------------------------------------------------------
    # File loading — xlsx and csv
    # ------------------------------------------------------------------

    def test_xlsx_settings_file_loads(self, mock_system):
        """Constructor loads an .xlsx settings file via pd.read_excel().

        When settings_file='sensors.xlsx' is supplied, NITask must call
        pd.read_excel() and store the result in self.settings.
        """
        system = mock_system(task_names=[])
        mock_df = self._mock_df()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("pandas.read_excel", return_value=mock_df) as mock_read,
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test_xlsx", sample_rate=25600, settings_file="sensors.xlsx")

        mock_read.assert_called_once_with("sensors.xlsx")
        assert task.settings is not None
        assert list(task.settings.columns) == list(mock_df.columns)

    def test_csv_settings_file_loads(self, mock_system):
        """Constructor loads a .csv settings file via pd.read_csv().

        When settings_file='sensors.csv' is supplied, NITask must call
        pd.read_csv() and store the result in self.settings.
        """
        system = mock_system(task_names=[])
        mock_df = self._mock_df()

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("pandas.read_csv", return_value=mock_df) as mock_read,
        ):
            from nidaqwrapper.task_input import NITask

            task = NITask("test_csv", sample_rate=25600, settings_file="sensors.csv")

        mock_read.assert_called_once_with("sensors.csv")
        assert task.settings is not None

    # ------------------------------------------------------------------
    # Invalid input validation
    # ------------------------------------------------------------------

    def test_invalid_extension_raises_valueerror(self, mock_system):
        """A .txt settings file raises ValueError.

        Only .xlsx and .csv extensions are supported; any other extension
        must raise ValueError at construction time.
        """
        system = mock_system(task_names=[])

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            with pytest.raises(ValueError, match=r"\.xlsx|\.csv|extension|Unsupported"):
                NITask("test_ext", sample_rate=25600, settings_file="sensors.txt")

    def test_non_string_filename_raises_typeerror(self, mock_system):
        """An integer settings_file argument raises TypeError.

        The constructor must validate that settings_file is a string before
        attempting any file operations.
        """
        system = mock_system(task_names=[])

        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
        ):
            from nidaqwrapper.task_input import NITask

            with pytest.raises(TypeError, match="string|str"):
                NITask("test_type", sample_rate=25600, settings_file=42)

    # ------------------------------------------------------------------
    # serial_nr lookup — happy path
    # ------------------------------------------------------------------

    def test_serial_nr_lookup_returns_correct_values(self, mock_system):
        """add_channel() with serial_nr='SN001' uses calibration from settings.

        The lookup returns the raw strings 'mV/g' and 'g'; add_channel()
        resolves them through UNITS to the mock constants.  We verify the
        resolved constants are stored, not the raw strings.
        """
        task = _make_task(mock_system)
        mock_df = self._mock_df()
        task.settings = mock_df  # inject settings DataFrame directly

        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "sensor_1",
                device_ind=0,
                channel_ind=0,
                serial_nr="SN001",
            )

        cfg = task.channels["sensor_1"]
        # sensitivity is the raw float from the DataFrame
        assert cfg["sensitivity"] == 100.0
        # sensitivity_units and units are resolved through MOCK_UNITS
        assert cfg["sensitivity_units"] == MOCK_UNITS["mV/g"]
        assert cfg["units"] == MOCK_UNITS["g"]

    def test_serial_nr_sn002_force_sensor(self, mock_system):
        """add_channel() with serial_nr='SN002' resolves force sensor calibration.

        SN002 is a force sensor (mV/N / N); verify the resolved constants
        match the ForceIEPESensorSensitivityUnits and ForceUnits mocks.
        """
        task = _make_task(mock_system)
        mock_df = self._mock_df()
        task.settings = mock_df

        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            task.add_channel(
                "force_sensor",
                device_ind=0,
                channel_ind=0,
                serial_nr="SN002",
            )

        cfg = task.channels["force_sensor"]
        assert cfg["sensitivity"] == 22.5
        assert cfg["sensitivity_units"] == MOCK_UNITS["mV/N"]
        assert cfg["units"] == MOCK_UNITS["N"]

    # ------------------------------------------------------------------
    # serial_nr lookup — error cases
    # ------------------------------------------------------------------

    def test_missing_serial_nr_raises_valueerror(self, mock_system):
        """add_channel() with an unknown serial_nr raises ValueError.

        When the settings DataFrame contains no row matching the given
        serial_nr, _lookup_serial_nr() must raise ValueError with a
        message identifying the missing serial number.
        """
        task = _make_task(mock_system)
        task.settings = self._mock_df()

        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises(ValueError, match="MISSING|not found|serial"):
                task.add_channel(
                    "sensor_x",
                    device_ind=0,
                    channel_ind=0,
                    serial_nr="MISSING",
                )

    def test_settings_missing_required_columns_raises_valueerror(self, mock_system):
        """Settings DataFrame missing 'sensitivity' column raises ValueError.

        _lookup_serial_nr() must validate that required columns are present
        before attempting the row lookup.
        """
        import pandas as pd

        task = _make_task(mock_system)
        # DataFrame is intentionally missing the 'sensitivity' column
        task.settings = pd.DataFrame({
            "serial_nr": ["SN001"],
            "sensitivity_units": ["mV/g"],
            "units": ["g"],
            # 'sensitivity' deliberately omitted
        })

        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises(ValueError, match="sensitivity|missing|column"):
                task.add_channel(
                    "sensor_1",
                    device_ind=0,
                    channel_ind=0,
                    serial_nr="SN001",
                )

    def test_serial_nr_without_settings_file_raises_valueerror(self, mock_system):
        """add_channel() with serial_nr but no settings file loaded raises ValueError.

        NITask.settings is None by default; attempting a serial_nr lookup
        without first loading a settings file must raise ValueError with a
        clear message directing the user to provide settings_file.
        """
        task = _make_task(mock_system)
        # settings is None (no file loaded) — this is the default state

        with patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS):
            with pytest.raises(ValueError, match="settings|settings_file"):
                task.add_channel(
                    "sensor_1",
                    device_ind=0,
                    channel_ind=0,
                    serial_nr="SN001",
                )

    def test_pandas_not_installed_raises_importerror(self, mock_system):
        """settings_file with pandas unavailable raises ImportError.

        When pandas is not installed, _read_settings_file() must raise
        ImportError with instructions to install nidaqwrapper[settings].
        """
        import builtins

        real_import = builtins.__import__

        def _no_pandas(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)

        system = mock_system(task_names=[])
        with (
            patch(
                "nidaqwrapper.task_input.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("builtins.__import__", side_effect=_no_pandas),
        ):
            from nidaqwrapper.task_input import NITask

            with pytest.raises(ImportError, match="pandas|settings"):
                NITask("test_nopd", sample_rate=25600, settings_file="sensors.xlsx")
