"""Tests for nidaqwrapper.task_input module (NITask class).

Architecture: Direct Delegation
-------------------------------
Constructor creates nidaqmx.Task immediately.
add_channel() delegates to nidaqmx task.ai_channels.add_ai_*() directly.
start() configures timing and optionally starts the task.
Getters read from nidaqmx task properties.

All tests use mocked nidaqmx — no hardware required.
"""

from __future__ import annotations

import warnings
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

    units["mV/g"] = _make_const("AccelSensitivityUnits", "MILLIVOLTS_PER_G")
    units["mV/m/s**2"] = _make_const(
        "AccelSensitivityUnits", "MILLIVOLTS_PER_METERS_PER_SECOND_SQUARED"
    )
    units["g"] = _make_const("AccelUnits", "ACCEL_G")
    units["m/s**2"] = _make_const("AccelUnits", "METERS_PER_SECOND_SQUARED")
    units["mV/N"] = _make_const(
        "ForceIEPESensorSensitivityUnits", "MILLIVOLTS_PER_NEWTON"
    )
    units["N"] = _make_const("ForceUnits", "NEWTONS")
    units["V"] = _make_const("VoltageUnits", "VOLTS")
    return units


MOCK_UNITS = _make_mock_units()


# ---------------------------------------------------------------------------
# Helpers — mock nidaqmx.Task that tracks channel additions
# ---------------------------------------------------------------------------

def _make_mock_ni_task(samp_clk_rate=25600):
    """Create a mock nidaqmx.Task that tracks channel additions.

    The mock records all add_ai_*_chan() calls and maintains a channel list
    so that duplicate detection (via task.channel_names and ai_channels
    iteration) works correctly in the implementation under test.
    """
    task = MagicMock()
    _channel_names = []
    _channel_objects = []

    def _make_handler():
        def handler(**kwargs):
            name = kwargs.get("name_to_assign_to_channel", "")
            phys = kwargs.get("physical_channel", "")
            _channel_names.append(name)
            ch = MagicMock()
            ch.name = name
            ch.physical_channel = MagicMock()
            ch.physical_channel.name = phys
            _channel_objects.append(ch)
        return handler

    task.ai_channels.add_ai_accel_chan.side_effect = _make_handler()
    task.ai_channels.add_ai_force_iepe_chan.side_effect = _make_handler()
    task.ai_channels.add_ai_voltage_chan.side_effect = _make_handler()

    # channel_names: same list object, stays in sync as channels are added
    task.channel_names = _channel_names

    # ai_channels iteration (for physical channel duplicate detection)
    task.ai_channels.__iter__ = MagicMock(
        side_effect=lambda: iter(_channel_objects)
    )

    # Timing (for start() tests)
    task._timing.samp_clk_rate = samp_clk_rate
    task.timing.samp_clk_rate = samp_clk_rate

    return task


def _build(mock_system, mock_constants, sample_rate=25600, samp_clk_rate=None,
           task_names=None, task_name="test"):
    """Construct an NITask inside a fully-patched context.

    Returns (ni_task_instance, mock_nidaqmx_task, patch_context_manager).
    Use inside a ``with`` block — patches stay active so that add_channel()
    and start() can also run under mocking.

    Example::

        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(...)
    """
    from contextlib import ExitStack

    if task_names is None:
        task_names = []
    if samp_clk_rate is None:
        samp_clk_rate = sample_rate

    system = mock_system(task_names=task_names)
    mock_ni_task = _make_mock_ni_task(samp_clk_rate=samp_clk_rate)

    stack = ExitStack()
    stack.enter_context(
        patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
              return_value=system)
    )
    stack.enter_context(
        patch("nidaqwrapper.task_input.nidaqmx.task.Task",
              return_value=mock_ni_task)
    )
    stack.enter_context(
        patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS)
    )
    stack.enter_context(
        patch("nidaqwrapper.task_input.constants", mock_constants)
    )

    from nidaqwrapper.task_input import NITask
    ni_task = NITask(task_name, sample_rate=sample_rate)

    return stack, ni_task, mock_ni_task


# ===========================================================================
# Task Group 1: NITask Constructor
# ===========================================================================

class TestNITaskConstructor:
    """Constructor creates nidaqmx.Task immediately (direct delegation)."""

    def test_creates_nidaqmx_task_with_name(self, mock_system, mock_constants):
        """Constructor calls nidaqmx.task.Task(new_task_name=task_name)."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task) as mock_cls,
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            NITask("vibration_test", sample_rate=25600)

        mock_cls.assert_called_once_with(new_task_name="vibration_test")

    def test_task_attribute_set_immediately(self, mock_system, mock_constants):
        """self.task is set to the nidaqmx.Task in the constructor."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            task = NITask("test", sample_rate=25600)

        assert task.task is mock_ni_task

    def test_task_name_stored(self, mock_system, mock_constants):
        """task_name is stored on the instance."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.task_name == "test"

    def test_sample_rate_stored(self, mock_system, mock_constants):
        """sample_rate is stored on the instance."""
        ctx, task, _ = _build(mock_system, mock_constants, sample_rate=51200)
        with ctx:
            pass
        assert task.sample_rate == 51200

    def test_device_list_populated(self, mock_system, mock_constants):
        """device_list contains device name strings from the system."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.device_list == ["cDAQ1Mod1", "cDAQ1Mod2"]

    def test_device_product_type_populated(self, mock_system, mock_constants):
        """device_product_type contains product type strings."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.device_product_type == ["NI 9234", "NI 9263"]

    def test_duplicate_task_name_raises_valueerror(self, mock_system, mock_constants):
        """Constructor raises ValueError when task_name already exists in NI MAX."""
        system = mock_system(task_names=["existing_task"])

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            with pytest.raises(ValueError, match="already"):
                NITask("existing_task", sample_rate=25600)

    def test_no_channels_dict(self, mock_system, mock_constants):
        """The old self.channels dict no longer exists."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "channels")

    def test_no_settings_file_parameter(self, mock_system, mock_constants):
        """Constructor no longer accepts settings_file parameter."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            with pytest.raises(TypeError):
                NITask("test", sample_rate=25600, settings_file="foo.xlsx")


# ===========================================================================
# Task Group 1: add_channel() — Accelerometer channels
# ===========================================================================

class TestAddChannelAccel:
    """add_channel() with accelerometer-type sensors delegates to nidaqmx."""

    def test_calls_add_ai_accel_chan(self, mock_system, mock_constants):
        """Accelerometer channel calls add_ai_accel_chan() on the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        mt.ai_channels.add_ai_accel_chan.assert_called_once()

    def test_passes_physical_channel(self, mock_system, mock_constants):
        """Physical channel string includes device name and ai index."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=2,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod1/ai2"

    def test_passes_channel_name(self, mock_system, mock_constants):
        """Channel name is forwarded as name_to_assign_to_channel."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["name_to_assign_to_channel"] == "accel_x"

    def test_passes_sensitivity(self, mock_system, mock_constants):
        """Sensitivity value and units are forwarded to nidaqmx."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["sensitivity"] == 100.0
        assert kwargs["sensitivity_units"] == MOCK_UNITS["mV/g"]
        assert kwargs["units"] == MOCK_UNITS["g"]

    def test_passes_terminal_config_default(self, mock_system, mock_constants):
        """Terminal config is set to DEFAULT."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["terminal_config"] == mock_constants.TerminalConfiguration.DEFAULT

    def test_ms2_units(self, mock_system, mock_constants):
        """Accelerometer channel accepts mV/m/s**2 sensitivity and m/s**2 units."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_z", device_ind=0, channel_ind=2,
                sensitivity=10.204, sensitivity_units="mV/m/s**2", units="m/s**2",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["sensitivity_units"] == MOCK_UNITS["mV/m/s**2"]
        assert kwargs["units"] == MOCK_UNITS["m/s**2"]

    def test_min_val_zero_forwarded(self, mock_system, mock_constants):
        """min_val=0.0 is forwarded (not treated as falsy — LDAQ bug fix)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=0.0, max_val=50.0,
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["min_val"] == 0.0
        assert kwargs["max_val"] == 50.0

    def test_custom_min_max(self, mock_system, mock_constants):
        """Custom min_val and max_val are forwarded to nidaqmx."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=-50.0, max_val=50.0,
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["min_val"] == -50.0
        assert kwargs["max_val"] == 50.0

    def test_min_max_omitted_when_none(self, mock_system, mock_constants):
        """When min_val/max_val are None, they are NOT in the kwargs."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert "min_val" not in kwargs
        assert "max_val" not in kwargs

    def test_second_device(self, mock_system, mock_constants):
        """Channel on device_ind=1 uses the second device name."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=1, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod2/ai0"


# ===========================================================================
# Task Group 1: add_channel() — Force channels
# ===========================================================================

class TestAddChannelForce:
    """add_channel() with force (IEPE) sensors delegates to add_ai_force_iepe_chan."""

    def test_calls_add_ai_force_iepe_chan(self, mock_system, mock_constants):
        """Force channel calls add_ai_force_iepe_chan()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "force_1", device_ind=0, channel_ind=0,
                sensitivity=22.5, sensitivity_units="mV/N", units="N",
            )

        mt.ai_channels.add_ai_force_iepe_chan.assert_called_once()
        assert not mt.ai_channels.add_ai_accel_chan.called
        assert not mt.ai_channels.add_ai_voltage_chan.called

    def test_passes_sensitivity(self, mock_system, mock_constants):
        """Force channel sensitivity and units are forwarded."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "force_1", device_ind=0, channel_ind=0,
                sensitivity=22.5, sensitivity_units="mV/N", units="N",
            )

        kwargs = mt.ai_channels.add_ai_force_iepe_chan.call_args.kwargs
        assert kwargs["sensitivity"] == 22.5
        assert kwargs["sensitivity_units"] == MOCK_UNITS["mV/N"]
        assert kwargs["units"] == MOCK_UNITS["N"]


# ===========================================================================
# Task Group 1: add_channel() — Voltage channels
# ===========================================================================

class TestAddChannelVoltage:
    """add_channel() with voltage channels delegates to add_ai_voltage_chan."""

    def test_calls_add_ai_voltage_chan(self, mock_system, mock_constants):
        """Plain voltage channel calls add_ai_voltage_chan()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "voltage_1", device_ind=0, channel_ind=0, units="V",
            )

        mt.ai_channels.add_ai_voltage_chan.assert_called_once()
        assert not mt.ai_channels.add_ai_accel_chan.called
        assert not mt.ai_channels.add_ai_force_iepe_chan.called

    def test_no_sensitivity_required(self, mock_system, mock_constants):
        """Voltage channels don't need sensitivity or sensitivity_units."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "voltage_1", device_ind=0, channel_ind=0, units="V",
            )

        kwargs = mt.ai_channels.add_ai_voltage_chan.call_args.kwargs
        assert "sensitivity" not in kwargs
        assert "sensitivity_units" not in kwargs

    def test_float_custom_scale(self, mock_system, mock_constants):
        """Float scale creates a linear custom scale and uses FROM_CUSTOM_SCALE."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale") as mock_scale:
                mock_scale.return_value.name = "voltage_1_scale"
                task.add_channel(
                    "voltage_1", device_ind=0, channel_ind=0,
                    units="V", scale=2500.0,
                )

        mock_scale.assert_called_once()
        scale_kwargs = mock_scale.call_args.kwargs
        assert scale_kwargs["slope"] == 2500.0
        assert scale_kwargs["y_intercept"] == 0

        chan_kwargs = mt.ai_channels.add_ai_voltage_chan.call_args.kwargs
        assert chan_kwargs["units"] == mock_constants.VoltageUnits.FROM_CUSTOM_SCALE
        assert chan_kwargs["custom_scale_name"] == "voltage_1_scale"

    def test_tuple_custom_scale(self, mock_system, mock_constants):
        """Tuple scale passes both slope and y_intercept."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale") as mock_scale:
                mock_scale.return_value.name = "voltage_1_scale"
                task.add_channel(
                    "voltage_1", device_ind=0, channel_ind=0,
                    units="V", scale=(2500.0, -100.0),
                )

        scale_kwargs = mock_scale.call_args.kwargs
        assert scale_kwargs["slope"] == 2500.0
        assert scale_kwargs["y_intercept"] == -100.0

    def test_custom_scale_dispatches_to_voltage(self, mock_system, mock_constants):
        """Channel with scale always uses add_ai_voltage_chan regardless of units."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale") as mock_scale:
                mock_scale.return_value.name = "ch_scale"
                task.add_channel(
                    "ch", device_ind=0, channel_ind=0,
                    units="V", scale=2500.0,
                )

        mt.ai_channels.add_ai_voltage_chan.assert_called_once()
        assert not mt.ai_channels.add_ai_accel_chan.called


# ===========================================================================
# Task Group 1: add_channel() — Validation
# ===========================================================================

class TestChannelValidation:
    """add_channel() input validation and error handling."""

    def test_duplicate_channel_name_raises(self, mock_system, mock_constants):
        """Adding a second channel with the same name raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            with pytest.raises(ValueError, match="accel_x"):
                task.add_channel(
                    "accel_x", device_ind=0, channel_ind=1,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )

    def test_duplicate_physical_channel_raises(self, mock_system, mock_constants):
        """Adding two channels for the same (device, channel) raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            with pytest.raises(ValueError, match="already"):
                task.add_channel(
                    "accel_y", device_ind=0, channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )

    def test_same_channel_on_different_device_ok(self, mock_system, mock_constants):
        """Same channel_ind on different devices is allowed."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.add_channel(
                "accel_y", device_ind=1, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        assert mt.ai_channels.add_ai_accel_chan.call_count == 2

    def test_out_of_range_device_raises(self, mock_system, mock_constants):
        """device_ind beyond available device list raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="device"):
                task.add_channel(
                    "accel_x", device_ind=99, channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )

    def test_missing_units_raises(self, mock_system, mock_constants):
        """add_channel() without units raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises((ValueError, TypeError)):
                task.add_channel(
                    "accel_x", device_ind=0, channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g",
                )

    def test_invalid_units_raises(self, mock_system, mock_constants):
        """Unrecognised units string raises ValueError when no scale given."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="units"):
                task.add_channel(
                    "accel_x", device_ind=0, channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g",
                    units="furlongs_per_fortnight",
                )

    def test_invalid_sensitivity_units_raises(self, mock_system, mock_constants):
        """Unrecognised sensitivity_units raises ValueError when no scale given."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="sensitivity_units"):
                task.add_channel(
                    "accel_x", device_ind=0, channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/parsec",
                    units="g",
                )

    def test_missing_sensitivity_for_accel_raises(self, mock_system, mock_constants):
        """Non-voltage channel without sensitivity raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="sensitivity"):
                task.add_channel(
                    "accel_x", device_ind=0, channel_ind=0,
                    sensitivity_units="mV/g", units="g",
                )

    def test_invalid_scale_type_raises(self, mock_system, mock_constants):
        """Scale of unsupported type raises TypeError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(TypeError, match="scale"):
                task.add_channel(
                    "voltage_1", device_ind=0, channel_ind=0,
                    units="V", scale="2500",
                )

    def test_multiple_channels_added(self, mock_system, mock_constants):
        """Multiple channels of different types can be added to one task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale") as mock_scale:
                mock_scale.return_value.name = "v_scale"
                task.add_channel(
                    "accel_x", device_ind=0, channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )
                task.add_channel(
                    "force_1", device_ind=0, channel_ind=1,
                    sensitivity=22.5, sensitivity_units="mV/N", units="N",
                )
                task.add_channel(
                    "voltage_1", device_ind=0, channel_ind=2,
                    units="V",
                )

        assert mt.ai_channels.add_ai_accel_chan.called
        assert mt.ai_channels.add_ai_force_iepe_chan.called
        assert mt.ai_channels.add_ai_voltage_chan.called


# ===========================================================================
# Task Group 2: start() — replaces initiate()
# ===========================================================================

class TestStart:
    """start() configures timing and optionally starts the task."""

    def test_configures_timing(self, mock_system, mock_constants):
        """start() calls cfg_samp_clk_timing with correct rate and CONTINUOUS mode."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=25600)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.start()

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["rate"] == 25600
        assert kwargs["sample_mode"] == mock_constants.AcquisitionType.CONTINUOUS

    def test_validates_sample_rate_pass(self, mock_system, mock_constants):
        """start() succeeds when actual rate matches requested rate."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25600)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.start()  # Should not raise

    def test_validates_sample_rate_fail(self, mock_system, mock_constants):
        """start() raises ValueError when driver coerces the rate."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25000)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            with pytest.raises(ValueError, match="[Ss]ample.?[Rr]ate|rate"):
                task.start()

    def test_starts_task_when_true(self, mock_system, mock_constants):
        """start(start_task=True) calls task.start()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.start(start_task=True)

        mt.start.assert_called_once()

    def test_does_not_start_by_default(self, mock_system, mock_constants):
        """start() with no args does NOT call task.start() (default is False)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.start()

        mt.start.assert_not_called()

    def test_start_task_false(self, mock_system, mock_constants):
        """start(start_task=False) configures timing but does NOT start."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.start(start_task=False)

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        mt.start.assert_not_called()

    def test_start_no_channels_raises(self, mock_system, mock_constants):
        """start() raises ValueError when no channels have been added."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="[Nn]o channels"):
                task.start()

    def test_rate_validation_failure_does_not_close_task(self, mock_system, mock_constants):
        """On rate mismatch, task handle remains valid (clear_task() not called).

        Unlike old initiate() which created then destroyed the task on failure,
        start() merely configures timing on an existing task. The task handle
        survives a rate validation failure.
        """
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25000)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            with pytest.raises(ValueError):
                task.start()

        # task.close() should NOT have been called — the task still exists
        mt.close.assert_not_called()


# ===========================================================================
# Task Group 2: Getters — read from nidaqmx
# ===========================================================================

class TestGetters:
    """Getters delegate to nidaqmx task properties."""

    def test_channel_list_from_nidaqmx(self, mock_system, mock_constants):
        """channel_list returns names from the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.add_channel(
                "accel_y", device_ind=0, channel_ind=1,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        assert task.channel_list == ["accel_x", "accel_y"]

    def test_number_of_ch(self, mock_system, mock_constants):
        """number_of_ch returns count from the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            assert task.number_of_ch == 0

            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            assert task.number_of_ch == 1

            task.add_channel(
                "accel_y", device_ind=0, channel_ind=1,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            assert task.number_of_ch == 2

    def test_channel_list_empty_initially(self, mock_system, mock_constants):
        """channel_list is empty on a new task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.channel_list == []

    def test_sample_rate_from_attribute(self, mock_system, mock_constants):
        """sample_rate property returns the stored sample rate."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=51200)
        with ctx:
            pass
        assert task.sample_rate == 51200


# ===========================================================================
# Task Group 2: from_settings() removed
# ===========================================================================

class TestSettingsRemoved:
    """All CSV/Excel settings file support is removed."""

    def test_no_from_settings_method(self, mock_system, mock_constants):
        """from_settings() method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "from_settings")

    def test_no_read_settings_file_method(self, mock_system, mock_constants):
        """_read_settings_file() method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_read_settings_file")

    def test_no_lookup_serial_nr_method(self, mock_system, mock_constants):
        """_lookup_serial_nr() method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_lookup_serial_nr")

    def test_no_settings_attribute(self, mock_system, mock_constants):
        """settings attribute no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "settings")


# ===========================================================================
# Existing: acquire_base() — minimal changes
# ===========================================================================

class TestAcquireBase:
    """acquire_base() reads all available samples from the hardware buffer."""

    def test_multi_channel(self, mock_system, mock_constants):
        """Multi-channel read returns (n_channels, n_samples) numpy array."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]

        result = task.acquire_base()

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    def test_single_channel_reshaped(self, mock_system, mock_constants):
        """Single-channel 1D result is reshaped to (1, n_samples)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = [1.0, 2.0, 3.0, 4.0]

        result = task.acquire_base()

        assert result.ndim == 2
        assert result.shape == (1, 4)

    def test_empty_buffer(self, mock_system, mock_constants):
        """Empty buffer returns an empty array without raising."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = []

        result = task.acquire_base()
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_calls_read_all_available(self, mock_system, mock_constants):
        """task.read() is called with number_of_samples_per_channel=-1."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = [[0.0], [0.0]]

        task.acquire_base()

        mt.read.assert_called_once_with(number_of_samples_per_channel=-1)


# ===========================================================================
# Existing: clear_task()
# ===========================================================================

class TestClearTask:
    """clear_task() releases hardware resources."""

    def test_calls_close(self, mock_system, mock_constants):
        """clear_task() calls task.close()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.clear_task()
        mt.close.assert_called_once()

    def test_sets_task_none(self, mock_system, mock_constants):
        """clear_task() sets self.task to None."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.clear_task()
        assert task.task is None

    def test_multiple_calls_no_error(self, mock_system, mock_constants):
        """Calling clear_task() twice raises no exception."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.clear_task()
        task.clear_task()  # Must not raise

    def test_exception_warns_not_propagated(self, mock_system, mock_constants):
        """clear_task() emits warning when task.close() raises."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.close.side_effect = RuntimeError("close failed")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.clear_task()

        assert len(w) >= 1
        assert "close failed" in str(w[0].message)
        assert task.task is None


# ===========================================================================
# Existing: save()
# ===========================================================================

class TestSave:
    """save() persists the task to NI MAX."""

    def test_calls_nidaqmx_save(self, mock_system, mock_constants):
        """save() calls task.save(overwrite_existing_task=True)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.save()
        mt.save.assert_called_once_with(overwrite_existing_task=True)

    def test_clears_task_by_default(self, mock_system, mock_constants):
        """save() calls clear_task() after saving when clear_task=True."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        task.clear_task = MagicMock()

        task.save()
        task.clear_task.assert_called_once()

    def test_does_not_clear_when_false(self, mock_system, mock_constants):
        """save(clear_task=False) saves but does not close."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        task.clear_task = MagicMock()

        task.save(clear_task=False)
        mt.save.assert_called_once()
        task.clear_task.assert_not_called()

    def test_no_auto_initiate(self, mock_system, mock_constants):
        """save() does NOT auto-initiate (task always exists in new arch)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        # Task always exists — save should work directly
        task.save()
        mt.save.assert_called_once()


# ===========================================================================
# Existing: Context Manager
# ===========================================================================

class TestContextManager:
    """NITask __enter__/__exit__ (context manager protocol)."""

    def test_enter_returns_self(self, mock_system, mock_constants):
        """__enter__ returns the NITask instance."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        result = task.__enter__()
        assert result is task

    def test_exit_calls_clear_task(self, mock_system, mock_constants):
        """__exit__ calls clear_task()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        task.clear_task = MagicMock()

        task.__exit__(None, None, None)
        task.clear_task.assert_called_once()

    def test_exception_still_clears(self, mock_system, mock_constants):
        """Exception in with-block still triggers cleanup."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        cleared = []
        original_clear = task.clear_task

        def _tracking_clear():
            cleared.append(True)
            if task.task is not None:
                task.task.close()

        task.clear_task = _tracking_clear

        with pytest.raises(ValueError):
            with task:
                raise ValueError("body error")

        assert cleared

    def test_cleanup_exception_warns(self, mock_system, mock_constants):
        """Cleanup exception emits warning, not raised."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.close.side_effect = RuntimeError("cleanup error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.__exit__(None, None, None)

        assert len(w) >= 1
        assert "cleanup error" in str(w[0].message)


# ===========================================================================
# Task Group 2: initiate() removed
# ===========================================================================

class TestInitiateRemoved:
    """initiate() method no longer exists."""

    def test_no_initiate_method(self, mock_system, mock_constants):
        """initiate() method does not exist on NITask."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "initiate")

    def test_no_add_channels_method(self, mock_system, mock_constants):
        """_add_channels() internal method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_add_channels")

    def test_no_add_channel_private_method(self, mock_system, mock_constants):
        """_add_channel() private method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_add_channel")

    def test_no_setup_task_method(self, mock_system, mock_constants):
        """_setup_task() internal method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_setup_task")


# ===========================================================================
# Task Group 3: TOML config save/load
# ===========================================================================

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


class TestSaveConfig:
    """save_config() serialises the task configuration to TOML."""

    def test_writes_toml_file(self, mock_system, mock_constants, tmp_path):
        """save_config() creates a file that can be parsed as TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        path = tmp_path / "config.toml"
        task.save_config(path)
        assert path.exists()

        with open(path, "rb") as f:
            data = tomllib.load(f)
        assert "task" in data
        assert "devices" in data
        assert "channels" in data

    def test_task_section(self, mock_system, mock_constants, tmp_path):
        """[task] section contains name, sample_rate, and type='input'."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=51200)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["name"] == "test"
        assert data["task"]["sample_rate"] == 51200
        assert data["task"]["type"] == "input"

    def test_devices_section(self, mock_system, mock_constants, tmp_path):
        """[devices] section contains unique device aliases for used devices."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.add_channel(
                "accel_y", device_ind=1, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Each used device gets an alias in the [devices] section
        devices = data["devices"]
        assert len(devices) == 2
        device_names = set(devices.values())
        assert "cDAQ1Mod1" in device_names
        assert "cDAQ1Mod2" in device_names

    def test_channel_entries(self, mock_system, mock_constants, tmp_path):
        """[[channels]] entries contain name, device alias, channel, units, etc."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=2,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=-50.0, max_val=50.0,
            )

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        channels = data["channels"]
        assert len(channels) == 1
        ch = channels[0]
        assert ch["name"] == "accel_x"
        assert ch["channel"] == 2
        assert ch["sensitivity"] == 100.0
        assert ch["sensitivity_units"] == "mV/g"
        assert ch["units"] == "g"
        assert ch["min_val"] == -50.0
        assert ch["max_val"] == 50.0
        # Device alias must reference a key in [devices]
        assert ch["device"] in data["devices"]

    def test_force_channel(self, mock_system, mock_constants, tmp_path):
        """Force/IEPE channel is saved correctly in TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "force_1", device_ind=0, channel_ind=0,
                sensitivity=22.5, sensitivity_units="mV/N", units="N",
            )

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["name"] == "force_1"
        assert ch["sensitivity"] == 22.5
        assert ch["sensitivity_units"] == "mV/N"
        assert ch["units"] == "N"

    def test_min_val_zero_preserved(self, mock_system, mock_constants, tmp_path):
        """min_val=0.0 is saved in TOML (not treated as falsy)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=0.0, max_val=50.0,
            )

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["min_val"] == 0.0
        assert ch["max_val"] == 50.0

    def test_voltage_channel_no_sensitivity(self, mock_system, mock_constants, tmp_path):
        """Voltage channels omit sensitivity/sensitivity_units from TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("v1", device_ind=0, channel_ind=0, units="V")

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["units"] == "V"
        assert "sensitivity" not in ch
        assert "sensitivity_units" not in ch

    def test_scale_channel(self, mock_system, mock_constants, tmp_path):
        """Channel with custom scale saves scale value in TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale") as ms:
                ms.return_value.name = "v1_scale"
                task.add_channel(
                    "v1", device_ind=0, channel_ind=0,
                    units="V", scale=(2500.0, -100.0),
                )

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["scale"] == [2500.0, -100.0]

    def test_min_max_omitted_when_none(self, mock_system, mock_constants, tmp_path):
        """min_val/max_val are not in TOML when they were None."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert "min_val" not in ch
        assert "max_val" not in ch


class TestFromConfig:
    """from_config() creates an NITask from a TOML file."""

    def _write_config(self, tmp_path, content):
        """Write a TOML string to a temp file and return the path."""
        path = tmp_path / "config.toml"
        path.write_text(content)
        return path

    def test_creates_task_with_name(self, mock_system, mock_constants, tmp_path):
        """from_config() creates a task with the name from [task] section."""
        path = self._write_config(tmp_path, """\
[task]
name = "vibration_test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task) as mock_cls,
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            task = NITask.from_config(path)

        mock_cls.assert_called_once_with(new_task_name="vibration_test")
        assert task.sample_rate == 25600

    def test_resolves_device_alias(self, mock_system, mock_constants, tmp_path):
        """from_config() resolves device alias to device index."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod2 = "cDAQ1Mod2"

[[channels]]
name = "accel_x"
device = "mod2"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            task = NITask.from_config(path)

        # cDAQ1Mod2 is device_ind=1, so physical channel should be cDAQ1Mod2/ai0
        kwargs = mock_ni_task.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod2/ai0"

    def test_multi_device_channels(self, mock_system, mock_constants, tmp_path):
        """from_config() handles channels on different devices."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"
mod2 = "cDAQ1Mod2"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"

[[channels]]
name = "accel_y"
device = "mod2"
channel = 1
sensitivity = 50.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            task = NITask.from_config(path)

        assert mock_ni_task.ai_channels.add_ai_accel_chan.call_count == 2
        calls = mock_ni_task.ai_channels.add_ai_accel_chan.call_args_list
        phys_channels = {c.kwargs["physical_channel"] for c in calls}
        assert "cDAQ1Mod1/ai0" in phys_channels
        assert "cDAQ1Mod2/ai1" in phys_channels

    def test_force_channel_from_config(self, mock_system, mock_constants, tmp_path):
        """from_config() handles force/IEPE channels."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "force_1"
device = "mod1"
channel = 0
sensitivity = 22.5
sensitivity_units = "mV/N"
units = "N"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            task = NITask.from_config(path)

        mock_ni_task.ai_channels.add_ai_force_iepe_chan.assert_called_once()
        kwargs = mock_ni_task.ai_channels.add_ai_force_iepe_chan.call_args.kwargs
        assert kwargs["sensitivity"] == 22.5

    def test_min_val_zero_from_config(self, mock_system, mock_constants, tmp_path):
        """from_config() preserves min_val=0.0 (not treated as falsy)."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
min_val = 0.0
max_val = 50.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            task = NITask.from_config(path)

        kwargs = mock_ni_task.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["min_val"] == 0.0
        assert kwargs["max_val"] == 50.0

    def test_voltage_channel_from_config(self, mock_system, mock_constants, tmp_path):
        """from_config() handles voltage channels (no sensitivity)."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "v1"
device = "mod1"
channel = 0
units = "V"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            task = NITask.from_config(path)

        mock_ni_task.ai_channels.add_ai_voltage_chan.assert_called_once()

    def test_channel_with_scale(self, mock_system, mock_constants, tmp_path):
        """from_config() handles channels with scale parameter."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "v1"
device = "mod1"
channel = 0
units = "V"
scale = [2500.0, -100.0]
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
            patch("nidaqwrapper.task_input.nidaqmx.Scale.create_lin_scale") as ms,
        ):
            ms.return_value.name = "v1_scale"
            from nidaqwrapper.task_input import NITask
            task = NITask.from_config(path)

        ms.assert_called_once()
        scale_kwargs = ms.call_args.kwargs
        assert scale_kwargs["slope"] == 2500.0
        assert scale_kwargs["y_intercept"] == -100.0

    def test_channel_with_min_max(self, mock_system, mock_constants, tmp_path):
        """from_config() forwards min_val/max_val from TOML."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
min_val = -50.0
max_val = 50.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            task = NITask.from_config(path)

        kwargs = mock_ni_task.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["min_val"] == -50.0
        assert kwargs["max_val"] == 50.0

    def test_invalid_device_alias_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when channel references unknown alias."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "nonexistent_module"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            with pytest.raises(ValueError, match="alias|device"):
                NITask.from_config(path)

    def test_device_not_in_system_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when device name not found in system."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "NonExistentDevice"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            with pytest.raises(ValueError, match="device|not found"):
                NITask.from_config(path)

    def test_missing_task_section_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when [task] section is missing."""
        path = self._write_config(tmp_path, """\
[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
units = "g"
""")
        system = mock_system(task_names=[])

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            with pytest.raises(ValueError, match="task"):
                NITask.from_config(path)

    def test_missing_devices_section_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when [devices] section is missing."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
units = "g"
""")
        system = mock_system(task_names=[])

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            with pytest.raises(ValueError, match="devices"):
                NITask.from_config(path)

    def test_malformed_toml_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises an error on syntactically invalid TOML."""
        path = self._write_config(tmp_path, "not = valid [ toml {\n")

        from nidaqwrapper.task_input import NITask
        with pytest.raises(Exception):  # tomllib.TOMLDecodeError
            NITask.from_config(path)


class TestConfigRoundtrip:
    """save_config → from_config produces equivalent task configuration."""

    def test_roundtrip_accel(self, mock_system, mock_constants, tmp_path):
        """Accel channel survives a save/load roundtrip."""
        # Create and configure original task
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device_ind=0, channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=-50.0, max_val=50.0,
            )

        path = tmp_path / "config.toml"
        task.save_config(path)

        # Load back via from_config
        system = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.task_input.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.task_input.nidaqmx.task.Task",
                  return_value=mock_ni_task2) as mock_cls,
            patch("nidaqwrapper.task_input.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.task_input.constants", mock_constants),
        ):
            from nidaqwrapper.task_input import NITask
            loaded = NITask.from_config(path)

        # Verify the loaded task matches the original
        mock_cls.assert_called_once_with(new_task_name="test")
        assert loaded.sample_rate == 25600
        mock_ni_task2.ai_channels.add_ai_accel_chan.assert_called_once()
        kwargs = mock_ni_task2.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["name_to_assign_to_channel"] == "accel_x"
        assert kwargs["physical_channel"] == "cDAQ1Mod1/ai0"
        assert kwargs["sensitivity"] == 100.0
        assert kwargs["min_val"] == -50.0
        assert kwargs["max_val"] == 50.0
