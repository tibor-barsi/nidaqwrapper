"""Tests for NITaskOutput — analog output task configuration.

TDD tests written BEFORE implementation, following the OpenSpec
task-output change specification.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — create mock nidaqmx module hierarchy
# ---------------------------------------------------------------------------

def _build_mock_nidaqmx(
    device_names: list[str] | None = None,
    task_names: list[str] | None = None,
):
    """Build a mock nidaqmx module with system, constants, and task."""
    if device_names is None:
        device_names = ["cDAQ1Mod1", "cDAQ1Mod2"]
    if task_names is None:
        task_names = []

    mock_nidaqmx = MagicMock()

    # --- constants ---
    mock_nidaqmx.constants.AcquisitionType.CONTINUOUS = "CONTINUOUS"
    mock_nidaqmx.constants.RegenerationMode.ALLOW_REGENERATION = "ALLOW_REGENERATION"

    # --- system ---
    devices = []
    for name in device_names:
        dev = MagicMock()
        dev.name = name
        devices.append(dev)

    system = MagicMock()
    system.devices = devices
    system.tasks.task_names = task_names
    mock_nidaqmx.system.System.local.return_value = system

    return mock_nidaqmx


def _make_task_output(mock_nidaqmx, task_name="signal_gen", sample_rate=10000, **kwargs):
    """Import and instantiate NITaskOutput with patched nidaqmx."""
    with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.errors": mock_nidaqmx.errors}):
        # Force reimport to pick up the mock
        import importlib
        import nidaqwrapper.task_output as mod
        importlib.reload(mod)
        return mod.NITaskOutput(task_name=task_name, sample_rate=sample_rate, **kwargs)


# ===========================================================================
# 1. Constructor Tests
# ===========================================================================

class TestNITaskOutputConstructor:
    """Tests for NITaskOutput.__init__."""

    def test_task_name_stored(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx, task_name="signal_gen")
        assert task.task_name == "signal_gen"

    def test_sample_rate_stored(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx, sample_rate=10000)
        assert task.sample_rate == 10000

    def test_samples_per_channel_default(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx, sample_rate=10000)
        assert task.samples_per_channel == 50000  # 5 * 10000

    def test_samples_per_channel_explicit_override(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx, sample_rate=10000, samples_per_channel=20000)
        assert task.samples_per_channel == 20000

    def test_samples_per_channel_is_int(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx, sample_rate=10000.0)
        assert isinstance(task.samples_per_channel, int)
        assert task.samples_per_channel == 50000

    def test_channels_empty_dict(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        assert task.channels == {}

    def test_sample_mode_continuous(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        assert task.sample_mode == "CONTINUOUS"

    def test_device_list_populated(self):
        mock_nidaqmx = _build_mock_nidaqmx(device_names=["cDAQ1Mod1", "cDAQ1Mod2"])
        task = _make_task_output(mock_nidaqmx)
        assert task.device_list == ["cDAQ1Mod1", "cDAQ1Mod2"]

    def test_device_list_empty_when_no_devices(self):
        mock_nidaqmx = _build_mock_nidaqmx(device_names=[])
        task = _make_task_output(mock_nidaqmx)
        assert task.device_list == []

    def test_reject_duplicate_task_name(self):
        mock_nidaqmx = _build_mock_nidaqmx(task_names=["existing_task"])
        with pytest.raises(ValueError, match="already"):
            _make_task_output(mock_nidaqmx, task_name="existing_task")

    def test_nidaqmx_task_not_created_at_init(self):
        """Constructor should NOT create the nidaqmx task — that happens at initiate()."""
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        assert task.task is None


# ===========================================================================
# 2. Add Channel Tests
# ===========================================================================

class TestAddChannel:
    """Tests for NITaskOutput.add_channel."""

    def test_add_channel_defaults(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        task.add_channel("ao_0", device_ind=0, channel_ind=0)

        assert "ao_0" in task.channels
        assert task.channels["ao_0"]["min_val"] == -10.0
        assert task.channels["ao_0"]["max_val"] == 10.0

    def test_add_channel_custom_range(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        task.add_channel("ao_0", device_ind=0, channel_ind=0, min_val=-5.0, max_val=5.0)

        assert task.channels["ao_0"]["min_val"] == -5.0
        assert task.channels["ao_0"]["max_val"] == 5.0

    def test_add_channel_stores_device_and_channel_ind(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        task.add_channel("ao_0", device_ind=1, channel_ind=3)

        assert task.channels["ao_0"]["device_ind"] == 1
        assert task.channels["ao_0"]["channel_ind"] == 3

    def test_reject_duplicate_channel_name(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        task.add_channel("ao_0", device_ind=0, channel_ind=0)

        with pytest.raises(ValueError, match="duplicate.*name|already exists"):
            task.add_channel("ao_0", device_ind=0, channel_ind=1)

    def test_reject_duplicate_device_channel_ind(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        task.add_channel("ao_0", device_ind=0, channel_ind=0)

        with pytest.raises(ValueError, match="device_ind.*channel_ind|physical channel"):
            task.add_channel("ao_1", device_ind=0, channel_ind=0)

    def test_reject_out_of_range_device_ind(self):
        mock_nidaqmx = _build_mock_nidaqmx(device_names=["cDAQ1Mod1"])
        task = _make_task_output(mock_nidaqmx)

        with pytest.raises(ValueError, match="device_ind.*out of range|available"):
            task.add_channel("ao_0", device_ind=99, channel_ind=0)

    def test_multiple_channels(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task = _make_task_output(mock_nidaqmx)
        task.add_channel("ao_0", device_ind=0, channel_ind=0)
        task.add_channel("ao_1", device_ind=0, channel_ind=1)

        assert len(task.channels) == 2


# ===========================================================================
# 3. Initiate Tests
# ===========================================================================

class TestInitiate:
    """Tests for NITaskOutput.initiate."""

    def _setup_task_for_initiate(self, mock_nidaqmx, sample_rate=10000):
        """Create an NITaskOutput with one channel, ready for initiate()."""
        task_out = _make_task_output(mock_nidaqmx, sample_rate=sample_rate)
        task_out.add_channel("ao_0", device_ind=0, channel_ind=0)

        # Mock the nidaqmx.task.Task that will be created
        mock_daq_task = MagicMock()
        mock_daq_task.timing = MagicMock()
        mock_daq_task._timing = MagicMock()
        mock_daq_task._timing.samp_clk_rate = float(sample_rate)
        mock_daq_task._out_stream = MagicMock()
        mock_daq_task.ao_channels = MagicMock()
        mock_nidaqmx.task.Task.return_value = mock_daq_task

        return task_out, mock_daq_task

    def test_creates_nidaqmx_task_with_name(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_task_for_initiate(mock_nidaqmx)
        task_out.initiate()

        mock_nidaqmx.task.Task.assert_called_once_with(new_task_name="signal_gen")

    def test_adds_ao_voltage_chan(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_task_for_initiate(mock_nidaqmx)
        task_out.initiate()

        mock_daq_task.ao_channels.add_ao_voltage_chan.assert_called_once_with(
            physical_channel="cDAQ1Mod1/ao0",
            name_to_assign_to_channel="ao_0",
            min_val=-10.0,
            max_val=10.0,
        )

    def test_configures_timing(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_task_for_initiate(mock_nidaqmx)
        task_out.initiate()

        mock_daq_task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=10000,
            sample_mode="CONTINUOUS",
            samps_per_chan=50000,
        )

    def test_sets_regeneration_mode(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_task_for_initiate(mock_nidaqmx)
        task_out.initiate()

        assert mock_daq_task._out_stream.regen_mode == "ALLOW_REGENERATION"

    def test_sample_rate_validation_passes(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_task_for_initiate(mock_nidaqmx, sample_rate=10000)
        mock_daq_task._timing.samp_clk_rate = 10000.0
        # Should not raise
        task_out.initiate()

    def test_sample_rate_validation_fails(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_task_for_initiate(mock_nidaqmx, sample_rate=10000)
        mock_daq_task._timing.samp_clk_rate = 10240.0  # Hardware chose different rate

        with pytest.raises(ValueError, match="(?i)sample rate.*not available|mismatch"):
            task_out.initiate()

    def test_sample_rate_validation_failure_cleans_up_task(self):
        """Hardware bug fix: initiate() must close nidaqmx task on rate mismatch."""
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_task_for_initiate(mock_nidaqmx, sample_rate=10000)
        mock_daq_task._timing.samp_clk_rate = 10240.0

        with pytest.raises(ValueError):
            task_out.initiate()

        mock_daq_task.close.assert_called_once()
        assert task_out.task is None

    def test_multiple_channels_added(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_task_for_initiate(mock_nidaqmx)
        task_out.add_channel("ao_1", device_ind=0, channel_ind=1, min_val=-5.0, max_val=5.0)
        task_out.initiate()

        assert mock_daq_task.ao_channels.add_ao_voltage_chan.call_count == 2


# ===========================================================================
# 4. Generate Tests
# ===========================================================================

class TestGenerate:
    """Tests for NITaskOutput.generate."""

    def _setup_initiated_task(self, mock_nidaqmx, sample_rate=10000):
        """Create and initiate an NITaskOutput with one channel."""
        task_out = _make_task_output(mock_nidaqmx, sample_rate=sample_rate)
        task_out.add_channel("ao_0", device_ind=0, channel_ind=0)

        mock_daq_task = MagicMock()
        mock_daq_task.timing = MagicMock()
        mock_daq_task._timing = MagicMock()
        mock_daq_task._timing.samp_clk_rate = float(sample_rate)
        mock_daq_task._out_stream = MagicMock()
        mock_daq_task.ao_channels = MagicMock()
        mock_nidaqmx.task.Task.return_value = mock_daq_task

        task_out.initiate()
        return task_out, mock_daq_task

    def test_2d_multi_channel_transposed(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_initiated_task(mock_nidaqmx)
        task_out.add_channel("ao_1", device_ind=0, channel_ind=1)

        signal = np.random.rand(1000, 2)
        task_out.generate(signal)

        written_data = mock_daq_task.write.call_args[0][0]
        assert written_data.shape == (2, 1000)

    def test_1d_single_channel_passed_directly(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_initiated_task(mock_nidaqmx)

        signal = np.random.rand(1000)
        task_out.generate(signal)

        written_data = mock_daq_task.write.call_args[0][0]
        # 1D array should be passed as-is (or as list)
        assert written_data.ndim == 1
        assert written_data.shape == (1000,)

    def test_2d_single_channel_transposed(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_initiated_task(mock_nidaqmx)

        signal = np.random.rand(1000, 1)
        task_out.generate(signal)

        written_data = mock_daq_task.write.call_args[0][0]
        # Single-channel 2D input (n_samples, 1) is squeezed to 1D (n_samples,)
        # because nidaqmx requires a 1-D array for single-channel tasks.
        assert written_data.shape == (1000,)

    def test_auto_start_true(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out, mock_daq_task = self._setup_initiated_task(mock_nidaqmx)

        signal = np.random.rand(1000)
        task_out.generate(signal)

        mock_daq_task.write.assert_called_once()
        assert mock_daq_task.write.call_args[1]["auto_start"] is True


# ===========================================================================
# 5. Clear Task Tests
# ===========================================================================

class TestClearTask:
    """Tests for NITaskOutput.clear_task."""

    def test_clear_initiated_task_closes(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out = _make_task_output(mock_nidaqmx)
        task_out.add_channel("ao_0", device_ind=0, channel_ind=0)

        mock_daq_task = MagicMock()
        mock_daq_task.timing = MagicMock()
        mock_daq_task._timing = MagicMock()
        mock_daq_task._timing.samp_clk_rate = 10000.0
        mock_daq_task._out_stream = MagicMock()
        mock_daq_task.ao_channels = MagicMock()
        mock_nidaqmx.task.Task.return_value = mock_daq_task

        task_out.initiate()
        task_out.clear_task()

        mock_daq_task.close.assert_called_once()

    def test_clear_sets_task_to_none(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out = _make_task_output(mock_nidaqmx)
        task_out.add_channel("ao_0", device_ind=0, channel_ind=0)

        mock_daq_task = MagicMock()
        mock_daq_task.timing = MagicMock()
        mock_daq_task._timing = MagicMock()
        mock_daq_task._timing.samp_clk_rate = 10000.0
        mock_daq_task._out_stream = MagicMock()
        mock_daq_task.ao_channels = MagicMock()
        mock_nidaqmx.task.Task.return_value = mock_daq_task

        task_out.initiate()
        task_out.clear_task()

        assert task_out.task is None

    def test_clear_multiple_calls_no_error(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out = _make_task_output(mock_nidaqmx)
        task_out.add_channel("ao_0", device_ind=0, channel_ind=0)

        mock_daq_task = MagicMock()
        mock_daq_task.timing = MagicMock()
        mock_daq_task._timing = MagicMock()
        mock_daq_task._timing.samp_clk_rate = 10000.0
        mock_daq_task._out_stream = MagicMock()
        mock_daq_task.ao_channels = MagicMock()
        mock_nidaqmx.task.Task.return_value = mock_daq_task

        task_out.initiate()
        task_out.clear_task()
        task_out.clear_task()  # Second call should not raise

    def test_clear_never_initiated_no_error(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out = _make_task_output(mock_nidaqmx)
        task_out.clear_task()  # Never initiated, should not raise


# ===========================================================================
# 6. Context Manager Tests
# ===========================================================================

class TestContextManager:
    """Tests for NITaskOutput context manager protocol."""

    def test_enter_returns_self(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out = _make_task_output(mock_nidaqmx)
        assert task_out.__enter__() is task_out

    def test_exit_calls_clear_task(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out = _make_task_output(mock_nidaqmx)
        task_out.add_channel("ao_0", device_ind=0, channel_ind=0)

        mock_daq_task = MagicMock()
        mock_daq_task.timing = MagicMock()
        mock_daq_task._timing = MagicMock()
        mock_daq_task._timing.samp_clk_rate = 10000.0
        mock_daq_task._out_stream = MagicMock()
        mock_daq_task.ao_channels = MagicMock()
        mock_nidaqmx.task.Task.return_value = mock_daq_task

        task_out.initiate()
        task_out.__exit__(None, None, None)

        mock_daq_task.close.assert_called_once()

    def test_exception_in_body_still_cleans_up(self):
        mock_nidaqmx = _build_mock_nidaqmx()
        task_out = _make_task_output(mock_nidaqmx)
        task_out.add_channel("ao_0", device_ind=0, channel_ind=0)

        mock_daq_task = MagicMock()
        mock_daq_task.timing = MagicMock()
        mock_daq_task._timing = MagicMock()
        mock_daq_task._timing.samp_clk_rate = 10000.0
        mock_daq_task._out_stream = MagicMock()
        mock_daq_task.ao_channels = MagicMock()
        mock_nidaqmx.task.Task.return_value = mock_daq_task

        task_out.initiate()

        with pytest.raises(RuntimeError):
            with task_out:
                raise RuntimeError("test error")

        mock_daq_task.close.assert_called_once()

    def test_cleanup_exception_warns_not_propagated(self):
        import warnings

        mock_nidaqmx = _build_mock_nidaqmx()
        task_out = _make_task_output(mock_nidaqmx)
        task_out.add_channel("ao_0", device_ind=0, channel_ind=0)

        mock_daq_task = MagicMock()
        mock_daq_task.timing = MagicMock()
        mock_daq_task._timing = MagicMock()
        mock_daq_task._timing.samp_clk_rate = 10000.0
        mock_daq_task._out_stream = MagicMock()
        mock_daq_task.ao_channels = MagicMock()
        mock_daq_task.close.side_effect = OSError("hardware error")
        mock_nidaqmx.task.Task.return_value = mock_daq_task

        task_out.initiate()
        # __exit__ should not propagate the cleanup exception but should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task_out.__exit__(None, None, None)

        assert len(w) >= 1
        assert "hardware error" in str(w[0].message)
