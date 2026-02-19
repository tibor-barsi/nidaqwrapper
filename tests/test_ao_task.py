"""Tests for nidaqwrapper.ao_task module (AOTask class).

Architecture: Direct Delegation
--------------------------------
Constructor creates nidaqmx.Task immediately.
add_channel() delegates to nidaqmx task.ao_channels.add_ao_voltage_chan() directly.
start() configures timing, regen mode, and optionally starts the task.
Getters read from nidaqmx task properties.

All tests use mocked nidaqmx — no hardware required.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — mock nidaqmx.Task that tracks AO channel additions
# ---------------------------------------------------------------------------

def _make_mock_ni_task(samp_clk_rate: float = 10000) -> MagicMock:
    """Create a mock nidaqmx.Task that tracks AO channel additions.

    The mock records all add_ao_voltage_chan() calls and maintains a channel
    list so that duplicate detection (via task.channel_names iteration) works
    correctly in the implementation under test.
    """
    task = MagicMock()
    _channel_names: list[str] = []
    _channel_objects: list[MagicMock] = []

    def _ao_handler(**kwargs):
        name = kwargs.get("name_to_assign_to_channel", "")
        phys = kwargs.get("physical_channel", "")
        _channel_names.append(name)
        ch = MagicMock()
        ch.name = name
        ch.physical_channel = MagicMock()
        ch.physical_channel.name = phys
        _channel_objects.append(ch)

    task.ao_channels.add_ao_voltage_chan.side_effect = _ao_handler

    # channel_names: same list object, stays in sync as channels are added
    task.channel_names = _channel_names

    # ao_channels iteration (for physical channel duplicate detection)
    task.ao_channels.__iter__ = MagicMock(
        side_effect=lambda: iter(_channel_objects)
    )

    # Timing (for start() tests)
    task._timing.samp_clk_rate = samp_clk_rate
    task.timing.samp_clk_rate = samp_clk_rate

    return task


def _build(
    mock_system,
    mock_constants,
    sample_rate: float = 10000,
    samp_clk_rate: float | None = None,
    task_names: list[str] | None = None,
    task_name: str = "signal_gen",
    samples_per_channel: int | None = None,
):
    """Construct an AOTask inside a fully-patched context.

    Returns (exit_stack, ao_task_instance, mock_nidaqmx_task).
    Use inside a ``with`` block — patches stay active so that add_channel()
    and start() can also run under mocking.

    Example::

        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(...)
            task.start()
    """
    if task_names is None:
        task_names = []
    if samp_clk_rate is None:
        samp_clk_rate = sample_rate

    system = mock_system(task_names=task_names)
    mock_ni_task = _make_mock_ni_task(samp_clk_rate=samp_clk_rate)

    stack = ExitStack()
    stack.enter_context(
        patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
              return_value=system)
    )
    stack.enter_context(
        patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
              return_value=mock_ni_task)
    )
    stack.enter_context(
        patch("nidaqwrapper.ao_task.constants", mock_constants)
    )

    from nidaqwrapper.ao_task import AOTask

    kwargs = {}
    if samples_per_channel is not None:
        kwargs["samples_per_channel"] = samples_per_channel

    ni_task = AOTask(task_name, sample_rate=sample_rate, **kwargs)

    return stack, ni_task, mock_ni_task


# ===========================================================================
# Task Group 4.1: AOTask Constructor
# ===========================================================================

class TestAOTaskConstructor:
    """Constructor creates nidaqmx.Task immediately (direct delegation)."""

    def test_creates_nidaqmx_task_with_name(self, mock_system, mock_constants):
        """Constructor calls nidaqmx.task.Task(new_task_name=task_name)."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task) as mock_cls,
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            AOTask("signal_gen", sample_rate=10000)

        mock_cls.assert_called_once_with(new_task_name="signal_gen")

    def test_task_attribute_set_immediately(self, mock_system, mock_constants):
        """self.task is set to the nidaqmx.Task in the constructor."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask("signal_gen", sample_rate=10000)

        assert task.task is mock_ni_task

    def test_task_name_stored(self, mock_system, mock_constants):
        """task_name is stored on the instance."""
        ctx, task, _ = _build(mock_system, mock_constants, task_name="signal_gen")
        with ctx:
            pass
        assert task.task_name == "signal_gen"

    def test_sample_rate_stored(self, mock_system, mock_constants):
        """sample_rate is stored on the instance."""
        ctx, task, _ = _build(mock_system, mock_constants, sample_rate=20000)
        with ctx:
            pass
        assert task.sample_rate == 20000

    def test_samples_per_channel_default(self, mock_system, mock_constants):
        """Default samples_per_channel is 5 * int(sample_rate)."""
        ctx, task, _ = _build(mock_system, mock_constants, sample_rate=10000)
        with ctx:
            pass
        assert task.samples_per_channel == 50000  # 5 * 10000

    def test_samples_per_channel_explicit_override(self, mock_system, mock_constants):
        """Explicit samples_per_channel overrides the default."""
        ctx, task, _ = _build(
            mock_system, mock_constants, sample_rate=10000, samples_per_channel=20000
        )
        with ctx:
            pass
        assert task.samples_per_channel == 20000

    def test_samples_per_channel_is_int(self, mock_system, mock_constants):
        """samples_per_channel is always stored as int (float sample_rate coerced)."""
        ctx, task, _ = _build(mock_system, mock_constants, sample_rate=10000.0)
        with ctx:
            pass
        assert isinstance(task.samples_per_channel, int)
        assert task.samples_per_channel == 50000

    def test_device_list_populated(self, mock_system, mock_constants):
        """device_list contains device name strings from the system."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.device_list == ["cDAQ1Mod1", "cDAQ1Mod2"]

    def test_device_list_empty_when_no_devices(self, mock_system, mock_constants):
        """device_list is empty when no devices are present."""
        system = mock_system(devices=[], task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask("signal_gen", sample_rate=10000)

        assert task.device_list == []

    def test_duplicate_task_name_raises_valueerror(self, mock_system, mock_constants):
        """Constructor raises ValueError when task_name already exists in NI MAX."""
        system = mock_system(task_names=["existing_task"])

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            with pytest.raises(ValueError, match="already"):
                AOTask("existing_task", sample_rate=10000)

    def test_no_channels_dict(self, mock_system, mock_constants):
        """The old self.channels dict no longer exists."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "channels")

    def test_no_settings_attribute(self, mock_system, mock_constants):
        """settings attribute no longer exists."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "settings")


# ===========================================================================
# Task Group 4.2: add_channel() — delegates to nidaqmx directly
# ===========================================================================

class TestAddChannel:
    """add_channel() delegates to nidaqmx ao_channels.add_ao_voltage_chan()."""

    def test_calls_add_ao_voltage_chan(self, mock_system, mock_constants):
        """add_channel() calls add_ao_voltage_chan() on the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)

        mt.ao_channels.add_ao_voltage_chan.assert_called_once()

    def test_passes_physical_channel_ao_prefix(self, mock_system, mock_constants):
        """Physical channel string uses 'ao' prefix (not 'ai')."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)

        kwargs = mt.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod1/ao0"

    def test_passes_physical_channel_with_index(self, mock_system, mock_constants):
        """Physical channel index is incorporated correctly."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_2", device_ind=0, channel_ind=2)

        kwargs = mt.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod1/ao2"

    def test_passes_channel_name(self, mock_system, mock_constants):
        """Channel name is forwarded as name_to_assign_to_channel."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("my_output", device_ind=0, channel_ind=0)

        kwargs = mt.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["name_to_assign_to_channel"] == "my_output"

    def test_default_min_val(self, mock_system, mock_constants):
        """Default min_val is -10.0."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)

        kwargs = mt.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["min_val"] == -10.0

    def test_default_max_val(self, mock_system, mock_constants):
        """Default max_val is 10.0."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)

        kwargs = mt.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["max_val"] == 10.0

    def test_custom_min_max(self, mock_system, mock_constants):
        """Custom min_val and max_val are forwarded to nidaqmx."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0,
                             min_val=-5.0, max_val=5.0)

        kwargs = mt.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["min_val"] == -5.0
        assert kwargs["max_val"] == 5.0

    def test_second_device(self, mock_system, mock_constants):
        """Channel on device_ind=1 uses the second device name."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=1, channel_ind=0)

        kwargs = mt.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod2/ao0"

    def test_duplicate_channel_name_raises(self, mock_system, mock_constants):
        """Adding a second channel with the same name raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            with pytest.raises(ValueError, match="ao_0"):
                task.add_channel("ao_0", device_ind=0, channel_ind=1)

    def test_duplicate_physical_channel_raises(self, mock_system, mock_constants):
        """Adding two channels for the same physical (device, channel) raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            with pytest.raises(ValueError, match="already"):
                task.add_channel("ao_1", device_ind=0, channel_ind=0)

    def test_same_channel_ind_on_different_device_ok(self, mock_system, mock_constants):
        """Same channel_ind on different devices is allowed."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.add_channel("ao_1", device_ind=1, channel_ind=0)

        assert mt.ao_channels.add_ao_voltage_chan.call_count == 2

    def test_min_val_zero_forwarded(self, mock_system, mock_constants):
        """min_val=0.0 is forwarded to nidaqmx (not treated as falsy)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0,
                             min_val=0.0, max_val=5.0)

        kwargs = mt.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["min_val"] == 0.0
        assert kwargs["max_val"] == 5.0

    def test_out_of_range_device_ind_raises(self, mock_system, mock_constants):
        """device_ind beyond available device list raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="device"):
                task.add_channel("ao_0", device_ind=99, channel_ind=0)

    def test_multiple_channels_added(self, mock_system, mock_constants):
        """Multiple channels can be added to one task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.add_channel("ao_1", device_ind=0, channel_ind=1)
            task.add_channel("ao_2", device_ind=0, channel_ind=2)

        assert mt.ao_channels.add_ao_voltage_chan.call_count == 3


# ===========================================================================
# Task Group 4.4: start() — replaces initiate()
# ===========================================================================

class TestStart:
    """start() configures timing, regen mode, and optionally starts the task."""

    def test_configures_timing_rate(self, mock_system, mock_constants):
        """start() calls cfg_samp_clk_timing with the configured sample rate."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=10000)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start()

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["rate"] == 10000

    def test_configures_timing_continuous_mode(self, mock_system, mock_constants):
        """start() passes CONTINUOUS sample mode to cfg_samp_clk_timing."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start()

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["sample_mode"] == mock_constants.AcquisitionType.CONTINUOUS

    def test_configures_timing_samps_per_chan(self, mock_system, mock_constants):
        """start() passes samples_per_channel to cfg_samp_clk_timing."""
        ctx, task, mt = _build(
            mock_system, mock_constants, sample_rate=10000, samples_per_channel=20000
        )
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start()

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["samps_per_chan"] == 20000

    def test_sets_regen_mode(self, mock_system, mock_constants):
        """start() sets _out_stream.regen_mode to ALLOW_REGENERATION."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start()

        assert mt._out_stream.regen_mode == mock_constants.RegenerationMode.ALLOW_REGENERATION

    def test_validates_sample_rate_pass(self, mock_system, mock_constants):
        """start() succeeds when actual rate matches requested rate."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=10000, samp_clk_rate=10000)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start()  # Should not raise

    def test_validates_sample_rate_fail(self, mock_system, mock_constants):
        """start() raises ValueError when driver coerces the rate."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=10000, samp_clk_rate=10240)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            with pytest.raises(ValueError, match="[Ss]ample.?[Rr]ate|rate"):
                task.start()

    def test_starts_task_when_true(self, mock_system, mock_constants):
        """start(start_task=True) calls task.start() on the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start(start_task=True)

        mt.start.assert_called_once()

    def test_does_not_start_by_default(self, mock_system, mock_constants):
        """start() with no args does NOT call task.start() (default is False)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start()

        mt.start.assert_not_called()

    def test_start_task_false_explicit(self, mock_system, mock_constants):
        """start(start_task=False) configures timing but does NOT start the task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start(start_task=False)

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        mt.start.assert_not_called()

    def test_no_channels_guard(self, mock_system, mock_constants):
        """start() raises ValueError when no channels have been added."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="[Nn]o channels|channel"):
                task.start()

    def test_rate_mismatch_does_not_close_task(self, mock_system, mock_constants):
        """On rate mismatch, the task handle remains valid — task.close() is NOT called.

        Unlike old initiate() which destroyed the task on failure, start()
        only configures timing on an existing task. The task handle survives.
        """
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=10000, samp_clk_rate=10240)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            with pytest.raises(ValueError):
                task.start()

        # task.close() should NOT have been called
        mt.close.assert_not_called()
        # task handle should still be set
        assert task.task is mt


# ===========================================================================
# Task Group 4.5: Getters — read from nidaqmx task
# ===========================================================================

class TestGetters:
    """Getters delegate to nidaqmx task properties."""

    def test_channel_list_from_nidaqmx(self, mock_system, mock_constants):
        """channel_list returns names from the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.add_channel("ao_1", device_ind=0, channel_ind=1)

        assert task.channel_list == ["ao_0", "ao_1"]

    def test_channel_list_empty_initially(self, mock_system, mock_constants):
        """channel_list is empty on a new task before any channels are added."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.channel_list == []

    def test_number_of_ch(self, mock_system, mock_constants):
        """number_of_ch returns the count of channels from the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            assert task.number_of_ch == 0

            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            assert task.number_of_ch == 1

            task.add_channel("ao_1", device_ind=0, channel_ind=1)
            assert task.number_of_ch == 2

    def test_sample_rate_from_attribute(self, mock_system, mock_constants):
        """sample_rate property returns the stored sample rate."""
        ctx, task, _ = _build(mock_system, mock_constants, sample_rate=20000)
        with ctx:
            pass
        assert task.sample_rate == 20000


# ===========================================================================
# Task Group 4.3: generate() — writes signal to output buffer
# ===========================================================================

class TestGenerate:
    """generate() writes signal data using np.ascontiguousarray(signal.T)."""

    def test_2d_multi_channel_transposed(self, mock_system, mock_constants):
        """2D (n_samples, n_channels) input is transposed to (n_channels, n_samples)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.add_channel("ao_1", device_ind=0, channel_ind=1)
            signal = np.random.rand(1000, 2)
            task.generate(signal)

        written_data = mt.write.call_args[0][0]
        assert written_data.shape == (2, 1000)

    def test_1d_single_channel_passed_directly(self, mock_system, mock_constants):
        """1D (n_samples,) input is passed directly (no transpose)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            signal = np.random.rand(1000)
            task.generate(signal)

        written_data = mt.write.call_args[0][0]
        assert written_data.ndim == 1
        assert written_data.shape == (1000,)

    def test_2d_single_channel_squeezed_to_1d(self, mock_system, mock_constants):
        """2D (n_samples, 1) input is squeezed to 1D (n_samples,).

        nidaqmx requires a 1-D array for single-channel tasks — a (1, N)
        array triggers a channel-count validation error.
        """
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            signal = np.random.rand(1000, 1)
            task.generate(signal)

        written_data = mt.write.call_args[0][0]
        assert written_data.ndim == 1
        assert written_data.shape == (1000,)

    def test_auto_start_true(self, mock_system, mock_constants):
        """generate() calls write() with auto_start=True."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            signal = np.random.rand(1000)
            task.generate(signal)

        mt.write.assert_called_once()
        assert mt.write.call_args.kwargs["auto_start"] is True

    def test_uses_ascontiguousarray(self, mock_system, mock_constants):
        """generate() uses np.ascontiguousarray to ensure C-contiguous layout.

        data.T returns Fortran-order. The nidaqmx C layer requires C-contiguous.
        """
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.add_channel("ao_1", device_ind=0, channel_ind=1)
            signal = np.random.rand(1000, 2)
            task.generate(signal)

        written_data = mt.write.call_args[0][0]
        # np.ascontiguousarray(signal.T) must produce C-contiguous array
        assert written_data.flags["C_CONTIGUOUS"]

    def test_2d_transposed_values_correct(self, mock_system, mock_constants):
        """Transposed 2D data contains the same values in the correct layout."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.add_channel("ao_1", device_ind=0, channel_ind=1)
            signal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
            task.generate(signal)

        written_data = mt.write.call_args[0][0]
        # After transpose: row 0 = channel 0 samples, row 1 = channel 1 samples
        np.testing.assert_array_equal(written_data[0], [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(written_data[1], [2.0, 4.0, 6.0])


# ===========================================================================
# Task Group: clear_task()
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
        """clear_task() emits a warning when task.close() raises."""
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
# Task Group: Context Manager
# ===========================================================================

class TestContextManager:
    """AOTask __enter__/__exit__ (context manager protocol)."""

    def test_enter_returns_self(self, mock_system, mock_constants):
        """__enter__ returns the AOTask instance."""
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

    def test_exception_in_body_still_cleans_up(self, mock_system, mock_constants):
        """Exception in with-block still triggers cleanup."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        cleared = []
        original_close = mt.close

        def _tracking_clear():
            cleared.append(True)
            if task.task is not None:
                task.task.close()

        task.clear_task = _tracking_clear

        with pytest.raises(RuntimeError):
            with task:
                raise RuntimeError("body error")

        assert cleared

    def test_cleanup_exception_warns_not_propagated(self, mock_system, mock_constants):
        """Cleanup exception emits a warning and is not raised."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.close.side_effect = OSError("hardware error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.__exit__(None, None, None)

        assert len(w) >= 1
        assert "hardware error" in str(w[0].message)


# ===========================================================================
# Task Group 4.4: initiate() removed — architecture guard tests
# ===========================================================================

class TestInitiateRemoved:
    """Old initiate() and its private helpers no longer exist."""

    def test_no_initiate_method(self, mock_system, mock_constants):
        """initiate() method does not exist on AOTask."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "initiate")

    def test_no_create_task_method(self, mock_system, mock_constants):
        """_create_task() internal method no longer exists."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_create_task")

    def test_no_add_channels_method(self, mock_system, mock_constants):
        """_add_channels() internal method no longer exists."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_add_channels")

    def test_no_setup_task_method(self, mock_system, mock_constants):
        """_setup_task() internal method no longer exists."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_setup_task")


# ===========================================================================
# Task Group 4.6: TOML config save/load
# ===========================================================================

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


class TestSaveConfig:
    """save_config() serialises the AOTask configuration to TOML."""

    def test_writes_toml_file(self, mock_system, mock_constants, tmp_path):
        """save_config() creates a file that can be parsed as TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)

        path = tmp_path / "config.toml"
        task.save_config(path)
        assert path.exists()

        with open(path, "rb") as f:
            data = tomllib.load(f)
        assert "task" in data
        assert "devices" in data
        assert "channels" in data

    def test_task_section_type_is_output(self, mock_system, mock_constants, tmp_path):
        """[task] section contains type='output' (not 'input')."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=10000)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["type"] == "output"

    def test_task_section_name_and_rate(self, mock_system, mock_constants, tmp_path):
        """[task] section contains the task name and sample_rate."""
        ctx, task, mt = _build(
            mock_system, mock_constants,
            task_name="signal_gen", sample_rate=20000
        )
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["name"] == "signal_gen"
        assert data["task"]["sample_rate"] == 20000

    def test_devices_section(self, mock_system, mock_constants, tmp_path):
        """[devices] section contains unique device aliases for used devices."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.add_channel("ao_1", device_ind=1, channel_ind=0)

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        devices = data["devices"]
        assert len(devices) == 2
        device_names = set(devices.values())
        assert "cDAQ1Mod1" in device_names
        assert "cDAQ1Mod2" in device_names

    def test_channel_entries(self, mock_system, mock_constants, tmp_path):
        """[[channels]] entries contain name, device alias, channel, min_val, max_val."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=2,
                             min_val=-5.0, max_val=5.0)

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        channels = data["channels"]
        assert len(channels) == 1
        ch = channels[0]
        assert ch["name"] == "ao_0"
        assert ch["channel"] == 2
        assert ch["min_val"] == -5.0
        assert ch["max_val"] == 5.0
        # Device alias must reference a key in [devices]
        assert ch["device"] in data["devices"]

    def test_default_min_max_in_toml(self, mock_system, mock_constants, tmp_path):
        """Default min_val=-10.0 and max_val=10.0 are saved in TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["min_val"] == -10.0
        assert ch["max_val"] == 10.0

    def test_multiple_channels_saved(self, mock_system, mock_constants, tmp_path):
        """Multiple channels are all serialised to [[channels]] entries."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.add_channel("ao_1", device_ind=0, channel_ind=1)

        path = tmp_path / "config.toml"
        task.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert len(data["channels"]) == 2


class TestFromConfig:
    """from_config() creates an AOTask from a TOML file."""

    def _write_config(self, tmp_path, content: str):
        """Write a TOML string to a temp file and return the path."""
        path = tmp_path / "config.toml"
        path.write_text(content)
        return path

    def test_creates_task_with_name(self, mock_system, mock_constants, tmp_path):
        """from_config() creates a task with the name from [task] section."""
        path = self._write_config(tmp_path, """\
[task]
name = "signal_gen"
sample_rate = 10000
type = "output"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "ao_0"
device = "mod1"
channel = 0
min_val = -10.0
max_val = 10.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task) as mock_cls,
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_config(path)

        mock_cls.assert_called_once_with(new_task_name="signal_gen")
        assert task.sample_rate == 10000

    def test_resolves_device_alias(self, mock_system, mock_constants, tmp_path):
        """from_config() resolves device alias to the correct physical channel."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 10000
type = "output"

[devices]
mod2 = "cDAQ1Mod2"

[[channels]]
name = "ao_0"
device = "mod2"
channel = 0
min_val = -10.0
max_val = 10.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            AOTask.from_config(path)

        # cDAQ1Mod2 is device_ind=1, so physical channel should be cDAQ1Mod2/ao0
        kwargs = mock_ni_task.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod2/ao0"

    def test_forwards_min_max_from_toml(self, mock_system, mock_constants, tmp_path):
        """from_config() forwards min_val and max_val to add_channel()."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 10000
type = "output"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "ao_0"
device = "mod1"
channel = 0
min_val = -5.0
max_val = 5.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            AOTask.from_config(path)

        kwargs = mock_ni_task.ao_channels.add_ao_voltage_chan.call_args.kwargs
        assert kwargs["min_val"] == -5.0
        assert kwargs["max_val"] == 5.0

    def test_multi_device_channels(self, mock_system, mock_constants, tmp_path):
        """from_config() handles channels on different devices."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 10000
type = "output"

[devices]
mod1 = "cDAQ1Mod1"
mod2 = "cDAQ1Mod2"

[[channels]]
name = "ao_0"
device = "mod1"
channel = 0
min_val = -10.0
max_val = 10.0

[[channels]]
name = "ao_1"
device = "mod2"
channel = 1
min_val = -10.0
max_val = 10.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            AOTask.from_config(path)

        assert mock_ni_task.ao_channels.add_ao_voltage_chan.call_count == 2
        calls = mock_ni_task.ao_channels.add_ao_voltage_chan.call_args_list
        phys_channels = {c.kwargs["physical_channel"] for c in calls}
        assert "cDAQ1Mod1/ao0" in phys_channels
        assert "cDAQ1Mod2/ao1" in phys_channels

    def test_invalid_device_alias_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when channel references unknown alias."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 10000
type = "output"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "ao_0"
device = "nonexistent_module"
channel = 0
min_val = -10.0
max_val = 10.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            with pytest.raises(ValueError, match="alias|device"):
                AOTask.from_config(path)

    def test_device_not_in_system_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when device name is not in the system."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 10000
type = "output"

[devices]
mod1 = "NonExistentDevice"

[[channels]]
name = "ao_0"
device = "mod1"
channel = 0
min_val = -10.0
max_val = 10.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            with pytest.raises(ValueError, match="device|not found"):
                AOTask.from_config(path)

    def test_missing_task_section_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when [task] section is missing."""
        path = self._write_config(tmp_path, """\
[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "ao_0"
device = "mod1"
channel = 0
min_val = -10.0
max_val = 10.0
""")
        system = mock_system(task_names=[])

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            with pytest.raises(ValueError, match="task"):
                AOTask.from_config(path)

    def test_missing_devices_section_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when [devices] section is missing."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 10000
type = "output"

[[channels]]
name = "ao_0"
device = "mod1"
channel = 0
min_val = -10.0
max_val = 10.0
""")
        system = mock_system(task_names=[])

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            with pytest.raises(ValueError, match="devices"):
                AOTask.from_config(path)

    def test_malformed_toml_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises an error on syntactically invalid TOML."""
        path = self._write_config(tmp_path, "not = valid [ toml {\n")

        from nidaqwrapper.ao_task import AOTask
        with pytest.raises(Exception):  # tomllib.TOMLDecodeError
            AOTask.from_config(path)


# ===========================================================================
# Task Group: from_task() — external task injection
# ===========================================================================

class TestFromTask:
    """from_task() wraps a pre-created nidaqmx.Task object."""

    def _make_external_task(
        self,
        mock_system,
        mock_constants,
        channel_count: int = 1,
        is_running: bool = False,
        sample_rate: float = 10000,
        samples_per_channel: int = 50000,
    ) -> MagicMock:
        """Create a mock nidaqmx.Task with AO channels configured."""
        task = MagicMock()
        task.name = "external_task"

        # Configure AO channels
        channel_objects = []
        channel_names = []
        for i in range(channel_count):
            ch = MagicMock()
            ch.name = f"ao_{i}"
            ch.physical_channel = MagicMock()
            ch.physical_channel.name = f"cDAQ1Mod1/ao{i}"
            channel_objects.append(ch)
            channel_names.append(f"ao_{i}")

        task.ao_channels = channel_objects
        task.channel_names = channel_names

        # Configure timing
        task.timing = MagicMock()
        task.timing.samp_clk_rate = sample_rate
        task.timing.samp_quant_samp_per_chan = samples_per_channel
        task.timing.samp_quant_samp_mode = mock_constants.AcquisitionType.CONTINUOUS

        # Task state (running or not)
        if is_running:
            # Mock task._saved_name or task.is_task_done() to indicate running state
            task.is_task_done = MagicMock(return_value=False)
        else:
            task.is_task_done = MagicMock(return_value=True)

        return task

    def test_creates_instance_without_calling_init(self, mock_system, mock_constants):
        """from_task() creates an AOTask without calling __init__."""
        # We verify this by checking that nidaqmx.Task() is NOT called
        # (which would happen in __init__ to create a new task)
        external = self._make_external_task(mock_system, mock_constants)
        system = mock_system(task_names=[])

        with (
            patch("nidaqwrapper.ao_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ao_task.nidaqmx.task.Task") as mock_task_cls,
            patch("nidaqwrapper.ao_task.constants", mock_constants),
        ):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

            # nidaqmx.Task() should NOT be called (no new task created)
            mock_task_cls.assert_not_called()
            assert task.task is external

    def test_populates_task_attribute(self, mock_system, mock_constants):
        """from_task() sets self.task to the provided task object."""
        external = self._make_external_task(mock_system, mock_constants)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        assert task.task is external

    def test_populates_task_name(self, mock_system, mock_constants):
        """from_task() reads task_name from task.name."""
        external = self._make_external_task(mock_system, mock_constants)
        external.name = "my_external_task"

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        assert task.task_name == "my_external_task"

    def test_populates_sample_rate(self, mock_system, mock_constants):
        """from_task() reads sample_rate from task.timing.samp_clk_rate."""
        external = self._make_external_task(
            mock_system, mock_constants, sample_rate=20000
        )

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        assert task.sample_rate == 20000

    def test_populates_channel_list(self, mock_system, mock_constants):
        """from_task() reads channel names from task.ao_channels."""
        external = self._make_external_task(mock_system, mock_constants, channel_count=3)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        assert task.channel_list == ["ao_0", "ao_1", "ao_2"]

    def test_populates_samples_per_channel(self, mock_system, mock_constants):
        """from_task() reads samples_per_channel from task.timing.samp_quant_samp_per_chan."""
        external = self._make_external_task(
            mock_system, mock_constants, samples_per_channel=100000
        )

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        assert task.samples_per_channel == 100000

    def test_populates_sample_mode(self, mock_system, mock_constants):
        """from_task() reads sample_mode from task.timing.samp_quant_samp_mode."""
        external = self._make_external_task(mock_system, mock_constants)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        assert task.sample_mode == mock_constants.AcquisitionType.CONTINUOUS

    def test_sets_owns_task_false(self, mock_system, mock_constants):
        """from_task() sets _owns_task to False."""
        external = self._make_external_task(mock_system, mock_constants)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        assert task._owns_task is False

    def test_constructor_sets_owns_task_true(self, mock_system, mock_constants):
        """Normal __init__() sets _owns_task to True."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass

        assert task._owns_task is True

    def test_validates_no_ao_channels_raises(self, mock_system, mock_constants):
        """from_task() raises ValueError when task has no AO channels."""
        task = MagicMock()
        task.name = "empty_task"
        task.ao_channels = []  # No channels

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            with pytest.raises(ValueError, match="no AO channels"):
                AOTask.from_task(task)

    def test_warns_when_task_already_running(self, mock_system, mock_constants):
        """from_task() warns when the task is already running."""
        external = self._make_external_task(
            mock_system, mock_constants, is_running=True
        )

        with (
            patch("nidaqwrapper.ao_task.constants", mock_constants),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            from nidaqwrapper.ao_task import AOTask
            AOTask.from_task(external)

        assert len(w) >= 1
        assert "running" in str(w[0].message).lower()

    def test_no_warning_when_task_not_running(self, mock_system, mock_constants):
        """from_task() does NOT warn when the task is not running."""
        external = self._make_external_task(
            mock_system, mock_constants, is_running=False
        )

        with (
            patch("nidaqwrapper.ao_task.constants", mock_constants),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            from nidaqwrapper.ao_task import AOTask
            AOTask.from_task(external)

        # Should be no warnings
        assert len(w) == 0

    def test_add_channel_blocked_raises(self, mock_system, mock_constants):
        """add_channel() raises RuntimeError when _owns_task is False."""
        external = self._make_external_task(mock_system, mock_constants)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

            with pytest.raises(RuntimeError, match="Cannot add channels"):
                task.add_channel("new_channel", device_ind=0, channel_ind=1)

    def test_add_channel_allowed_when_owns_task(self, mock_system, mock_constants):
        """add_channel() succeeds when _owns_task is True (normal constructor)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)  # Should not raise

        assert task.number_of_ch == 1

    def test_start_blocked_raises(self, mock_system, mock_constants):
        """start() raises RuntimeError when _owns_task is False."""
        external = self._make_external_task(mock_system, mock_constants)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

            with pytest.raises(RuntimeError, match="Cannot start"):
                task.start()

    def test_start_allowed_when_owns_task(self, mock_system, mock_constants):
        """start() succeeds when _owns_task is True (normal constructor)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("ao_0", device_ind=0, channel_ind=0)
            task.start()  # Should not raise

    def test_clear_task_does_not_close_external(self, mock_system, mock_constants):
        """clear_task() does NOT call task.close() when _owns_task is False."""
        external = self._make_external_task(mock_system, mock_constants)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)
            task.clear_task()

        external.close.assert_not_called()
        assert task.task is external  # Task reference remains

    def test_clear_task_warns_external(self, mock_system, mock_constants):
        """clear_task() warns when _owns_task is False."""
        external = self._make_external_task(mock_system, mock_constants)

        with (
            patch("nidaqwrapper.ao_task.constants", mock_constants),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)
            task.clear_task()

        assert len(w) >= 1
        assert "externally" in str(w[0].message).lower()

    def test_clear_task_closes_owned(self, mock_system, mock_constants):
        """clear_task() calls task.close() when _owns_task is True."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.clear_task()
        mt.close.assert_called_once()

    def test_exit_does_not_close_external(self, mock_system, mock_constants):
        """__exit__ does NOT close external task when _owns_task is False."""
        external = self._make_external_task(mock_system, mock_constants)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)
            task.__exit__(None, None, None)

        external.close.assert_not_called()

    def test_exit_warns_external(self, mock_system, mock_constants):
        """__exit__ warns when _owns_task is False."""
        external = self._make_external_task(mock_system, mock_constants)

        with (
            patch("nidaqwrapper.ao_task.constants", mock_constants),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)
            task.__exit__(None, None, None)

        assert len(w) >= 1
        assert "externally" in str(w[0].message).lower()

    def test_exit_closes_owned(self, mock_system, mock_constants):
        """__exit__ closes task when _owns_task is True."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.__exit__(None, None, None)
        mt.close.assert_called_once()

    def test_generate_works_with_external_task(self, mock_system, mock_constants):
        """generate() works correctly with an external task."""
        external = self._make_external_task(mock_system, mock_constants, channel_count=2)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

            signal = np.random.rand(1000, 2)
            task.generate(signal)

        external.write.assert_called_once()

    def test_channel_list_property_external(self, mock_system, mock_constants):
        """channel_list property reads from external task correctly."""
        external = self._make_external_task(mock_system, mock_constants, channel_count=2)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        # The property should delegate to task.channel_names
        assert task.channel_list == ["ao_0", "ao_1"]

    def test_number_of_ch_property_external(self, mock_system, mock_constants):
        """number_of_ch property reads from external task correctly."""
        external = self._make_external_task(mock_system, mock_constants, channel_count=3)

        with patch("nidaqwrapper.ao_task.constants", mock_constants):
            from nidaqwrapper.ao_task import AOTask
            task = AOTask.from_task(external)

        assert task.number_of_ch == 3
