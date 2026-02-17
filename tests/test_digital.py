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
    mock_nidaqmx.constants.LineGrouping.CHAN_PER_LINE = "CHAN_PER_LINE"
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


class TestDigitalInputInitiate:
    """Task group 3: DigitalInput.initiate() tests."""

    def test_on_demand_creates_task_and_adds_channels(self):
        """On-demand initiate creates nidaqmx.Task and adds DI channels."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_init")
            di.add_channel("btn", lines="Dev1/port0/line0")
            di.initiate()

            mock_nidaqmx.Task.assert_called_once()
            mock_task.di_channels.add_di_chan.assert_called_once_with(
                lines="Dev1/port0/line0",
                name_to_assign_to_lines="btn",
                line_grouping="CHAN_PER_LINE",
            )

    def test_on_demand_no_timing(self):
        """On-demand initiate does NOT configure timing."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_notimed")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()

            mock_task.timing.cfg_samp_clk_timing.assert_not_called()

    def test_on_demand_does_not_start_task(self):
        """On-demand initiate does NOT call task.start() regardless of flag."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_nostart")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate(start_task=True)

            mock_task.start.assert_not_called()

    def test_clocked_configures_timing(self):
        """Clocked initiate configures timing with sample_rate and CONTINUOUS."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_clk", sample_rate=1000)
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()

            mock_task.timing.cfg_samp_clk_timing.assert_called_once_with(
                rate=1000,
                sample_mode="CONTINUOUS",
            )

    def test_clocked_start_task_true(self):
        """Clocked initiate with start_task=True calls task.start()."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_start", sample_rate=1000)
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate(start_task=True)

            mock_task.start.assert_called_once()

    def test_clocked_start_task_false(self):
        """Clocked initiate with start_task=False does NOT call task.start()."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_nostart2", sample_rate=1000)
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate(start_task=False)

            mock_task.start.assert_not_called()


class TestDigitalInputRead:
    """Task group 4: DigitalInput.read() (on-demand) tests."""

    def test_read_single_line_bool(self):
        """read() with single-line returns numpy array with one bool."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        mock_task.read.return_value = True
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_rd1")
            di.add_channel("btn", lines="Dev1/port0/line0")
            di.initiate()
            data = di.read()

            assert isinstance(data, np.ndarray)
            assert len(data) == 1
            assert data[0] == True

    def test_read_multi_line_list(self):
        """read() with multi-line returns numpy array with one value per line."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        mock_task.read.return_value = [True, False, True, False]
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_rdm")
            di.add_channel("sw", lines="Dev1/port0/line0:3")
            di.initiate()
            data = di.read()

            assert isinstance(data, np.ndarray)
            assert len(data) == 4
            np.testing.assert_array_equal(data, [True, False, True, False])

    def test_read_returns_numpy_array(self):
        """read() always returns a numpy array."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        mock_task.read.return_value = False
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_np")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()
            data = di.read()

            assert isinstance(data, np.ndarray)

    def test_read_on_demand_without_explicit_start(self):
        """read() works in on-demand mode without explicit task start."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        mock_task.read.return_value = True
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_impl")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()

            mock_task.start.assert_not_called()
            data = di.read()
            assert data[0] == True


class TestDigitalInputReadAllAvailable:
    """Task group 5: DigitalInput.read_all_available() (clocked) tests."""

    def test_returns_n_samples_n_lines_shape(self):
        """read_all_available() returns (n_samples, n_lines) array."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        # nidaqmx returns (n_lines, n_samples) — 4 lines, 500 samples
        mock_task.read.return_value = [
            [True] * 500,
            [False] * 500,
            [True] * 500,
            [False] * 500,
        ]
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_cont", sample_rate=1000)
            di.add_channel("ch", lines="Dev1/port0/line0:3")
            di.initiate()
            data = di.read_all_available()

            assert data.shape == (500, 4)

    def test_empty_buffer_returns_empty(self):
        """read_all_available() returns empty array when buffer is empty."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        mock_task.read.return_value = []
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_empty", sample_rate=1000)
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()
            data = di.read_all_available()

            assert data.size == 0

    def test_uses_read_all_available_constant(self):
        """read_all_available() passes READ_ALL_AVAILABLE to task.read()."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        mock_task.read.return_value = [True, False]
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_raa", sample_rate=1000)
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()
            di.read_all_available()

            mock_task.read.assert_called_with(number_of_samples_per_channel=-1)

    def test_raises_in_on_demand_mode(self):
        """read_all_available() raises RuntimeError in on-demand mode."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_od")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()

            with pytest.raises(RuntimeError, match="clocked mode"):
                di.read_all_available()

    def test_single_line_reshaped(self):
        """read_all_available() reshapes single-line data to (n_samples, 1)."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        # Single line: nidaqmx returns flat list
        mock_task.read.return_value = [True, False, True]
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_1line", sample_rate=1000)
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()
            data = di.read_all_available()

            assert data.shape == (3, 1)


class TestDigitalInputClearTask:
    """Task group 6: DigitalInput.clear_task() and context manager tests."""

    def test_clear_task_closes_initiated_task(self):
        """clear_task() calls task.close() on an initiated task."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_clear")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()
            di.clear_task()

            mock_task.close.assert_called_once()
            assert di.task is None

    def test_clear_task_multiple_calls(self):
        """clear_task() called twice does not raise."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_multi")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()
            di.clear_task()
            di.clear_task()  # Should not raise

    def test_clear_task_never_initiated(self):
        """clear_task() on a never-initiated task does not raise."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_noinit")
            di.clear_task()  # Should not raise

    def test_context_manager_enter_returns_self(self):
        """__enter__ returns the DigitalInput instance."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_ctx")
            with di as ctx:
                assert ctx is di

    def test_context_manager_exit_calls_clear_task(self):
        """__exit__ calls clear_task()."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_exit")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()

            with di:
                pass

            mock_task.close.assert_called_once()

    def test_context_manager_cleanup_on_exception(self):
        """__exit__ still calls clear_task() when exception occurs."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalInput

            di = DigitalInput(task_name="di_exc")
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.initiate()

            with pytest.raises(RuntimeError):
                with di:
                    raise RuntimeError("test error")

            mock_task.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# DigitalOutput Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDigitalOutputConstructor:
    """Task group 7: DigitalOutput constructor tests."""

    def test_on_demand_mode_defaults(self):
        """DigitalOutput without sample_rate has mode='on_demand', sample_rate=None."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_ctrl")
            assert do.task_name == "do_ctrl"
            assert do.sample_rate is None
            assert do.mode == "on_demand"
            assert do.channels == {}

    def test_clocked_mode(self):
        """DigitalOutput with sample_rate=1000 has mode='clocked'."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_clocked", sample_rate=1000)
            assert do.sample_rate == 1000
            assert do.mode == "clocked"

    def test_device_discovery(self):
        """DigitalOutput discovers connected devices."""
        mock_nidaqmx, _ = _make_mock_nidaqmx(device_names=["cDAQ1Mod1", "cDAQ1Mod2"])
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_dev")
            assert "cDAQ1Mod1" in do.device_list
            assert "cDAQ1Mod2" in do.device_list

    def test_reject_duplicate_task_name(self):
        """DigitalOutput raises ValueError if task_name exists in NI MAX."""
        mock_nidaqmx, _ = _make_mock_nidaqmx(task_names=["existing_task"])
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            with pytest.raises(ValueError, match="existing_task"):
                DigitalOutput(task_name="existing_task")


class TestDigitalOutputAddChannel:
    """Task group 8: DigitalOutput.add_channel() tests."""

    def _make_do(self):
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        patcher = patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system})
        patcher.start()
        from nidaqwrapper.digital import DigitalOutput

        do = DigitalOutput(task_name="do_ch")
        return do, patcher

    def test_add_single_line(self):
        """add_channel stores single line specification."""
        do, patcher = self._make_do()
        try:
            do.add_channel("led_1", lines="Dev1/port1/line0")
            assert "led_1" in do.channels
            assert do.channels["led_1"]["lines"] == "Dev1/port1/line0"
        finally:
            patcher.stop()

    def test_add_line_range(self):
        """add_channel stores line range specification."""
        do, patcher = self._make_do()
        try:
            do.add_channel("leds", lines="Dev1/port1/line0:7")
            assert do.channels["leds"]["lines"] == "Dev1/port1/line0:7"
        finally:
            patcher.stop()

    def test_add_full_port(self):
        """add_channel stores full port specification."""
        do, patcher = self._make_do()
        try:
            do.add_channel("port1", lines="Dev1/port1")
            assert do.channels["port1"]["lines"] == "Dev1/port1"
        finally:
            patcher.stop()

    def test_reject_duplicate_name(self):
        """add_channel raises ValueError on duplicate channel name."""
        do, patcher = self._make_do()
        try:
            do.add_channel("led_1", lines="Dev1/port1/line0")
            with pytest.raises(ValueError, match="led_1"):
                do.add_channel("led_1", lines="Dev1/port1/line1")
        finally:
            patcher.stop()

    def test_reject_duplicate_lines(self):
        """add_channel raises ValueError when lines are already in use."""
        do, patcher = self._make_do()
        try:
            do.add_channel("ch1", lines="Dev1/port1/line0")
            with pytest.raises(ValueError, match="Dev1/port1/line0"):
                do.add_channel("ch2", lines="Dev1/port1/line0")
        finally:
            patcher.stop()


class TestDigitalOutputInitiate:
    """Task group 9: DigitalOutput.initiate() tests."""

    def test_on_demand_creates_task_and_adds_do_channels(self):
        """On-demand initiate creates task and adds DO channels."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_init")
            do.add_channel("led", lines="Dev1/port1/line0")
            do.initiate()

            mock_nidaqmx.Task.assert_called_once()
            mock_task.do_channels.add_do_chan.assert_called_once_with(
                lines="Dev1/port1/line0",
                name_to_assign_to_lines="led",
                line_grouping="CHAN_PER_LINE",
            )

    def test_on_demand_no_timing(self):
        """On-demand initiate does NOT configure timing."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_nt")
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()

            mock_task.timing.cfg_samp_clk_timing.assert_not_called()

    def test_clocked_configures_timing(self):
        """Clocked initiate configures timing."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_clk", sample_rate=1000)
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()

            mock_task.timing.cfg_samp_clk_timing.assert_called_once_with(
                rate=1000,
                sample_mode="CONTINUOUS",
            )

    def test_clocked_start_task(self):
        """Clocked initiate with start_task=True starts the task."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_st", sample_rate=1000)
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate(start_task=True)

            mock_task.start.assert_called_once()


class TestDigitalOutputWrite:
    """Task group 10: DigitalOutput.write() (on-demand) tests."""

    def test_write_single_bool(self):
        """write(True) passes True to task.write()."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wb")
            do.add_channel("led", lines="Dev1/port1/line0")
            do.initiate()
            do.write(True)

            mock_task.write.assert_called_once_with(True)

    def test_write_single_int(self):
        """write(1) converts to True and passes to task.write()."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wi")
            do.add_channel("led", lines="Dev1/port1/line0")
            do.initiate()
            do.write(1)

            mock_task.write.assert_called_once_with(True)

    def test_write_list(self):
        """write([True, False, True, False]) converts to bools and passes."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wl")
            do.add_channel("leds", lines="Dev1/port1/line0:3")
            do.initiate()
            do.write([True, False, True, False])

            mock_task.write.assert_called_once_with([True, False, True, False])

    def test_write_numpy_array(self):
        """write(np.array([1, 0, 1, 0])) converts to list of bools."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wn")
            do.add_channel("leds", lines="Dev1/port1/line0:3")
            do.initiate()
            do.write(np.array([1, 0, 1, 0]))

            mock_task.write.assert_called_once_with([True, False, True, False])

    def test_write_calls_task_write(self):
        """write() delegates to task.write() with the data."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wd")
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()
            do.write(False)

            mock_task.write.assert_called_once()


class TestDigitalOutputWriteContinuous:
    """Task group 11: DigitalOutput.write_continuous() tests."""

    def test_multi_line_2d_transposed(self):
        """write_continuous() transposes (n_samples, n_lines) to (n_lines, n_samples)."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wc2d", sample_rate=1000)
            do.add_channel("ch", lines="Dev1/port1/line0:3")
            do.initiate()

            data = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])  # (3, 4)
            do.write_continuous(data)

            call_args = mock_task.write.call_args
            written_data = call_args[0][0]
            assert len(written_data) == 4  # n_lines
            assert len(written_data[0]) == 3  # n_samples
            assert call_args[1]["auto_start"] is True

    def test_single_line_1d_written_directly(self):
        """write_continuous() writes 1D array directly for single line."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wc1d", sample_rate=1000)
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()

            data = np.array([1, 0, 1, 0, 1])
            do.write_continuous(data)

            call_args = mock_task.write.call_args
            written_data = call_args[0][0]
            assert written_data == [1, 0, 1, 0, 1]

    def test_raises_in_on_demand_mode(self):
        """write_continuous() raises RuntimeError in on-demand mode."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wcod")
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()

            with pytest.raises(RuntimeError, match="clocked mode"):
                do.write_continuous(np.array([1, 0, 1]))

    def test_auto_start_true(self):
        """write_continuous() calls task.write() with auto_start=True."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_wcas", sample_rate=1000)
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()

            do.write_continuous(np.array([1, 0, 1]))

            call_args = mock_task.write.call_args
            assert call_args[1]["auto_start"] is True


class TestDigitalOutputClearTask:
    """Task group 12: DigitalOutput.clear_task() and context manager tests."""

    def test_clear_task_closes_initiated_task(self):
        """clear_task() calls task.close() on an initiated task."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_clear")
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()
            do.clear_task()

            mock_task.close.assert_called_once()
            assert do.task is None

    def test_clear_task_multiple_calls(self):
        """clear_task() called twice does not raise."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_multi")
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()
            do.clear_task()
            do.clear_task()  # Should not raise

    def test_clear_task_never_initiated(self):
        """clear_task() on a never-initiated task does not raise."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_noinit")
            do.clear_task()  # Should not raise

    def test_context_manager_enter_returns_self(self):
        """__enter__ returns the DigitalOutput instance."""
        mock_nidaqmx, _ = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_ctx")
            with do as ctx:
                assert ctx is do

    def test_context_manager_exit_calls_clear_task(self):
        """__exit__ calls clear_task()."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_exit")
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()

            with do:
                pass

            mock_task.close.assert_called_once()

    def test_context_manager_cleanup_on_exception(self):
        """__exit__ still calls clear_task() when exception occurs."""
        mock_nidaqmx, mock_task = _make_mock_nidaqmx()
        with patch.dict("sys.modules", {"nidaqmx": mock_nidaqmx, "nidaqmx.constants": mock_nidaqmx.constants, "nidaqmx.system": mock_nidaqmx.system}):
            from nidaqwrapper.digital import DigitalOutput

            do = DigitalOutput(task_name="do_exc")
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.initiate()

            with pytest.raises(RuntimeError):
                with do:
                    raise RuntimeError("test error")

            mock_task.close.assert_called_once()
