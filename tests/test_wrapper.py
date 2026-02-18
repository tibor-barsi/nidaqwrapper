"""Unit tests for NIDAQWrapper — high-level single-task NI-DAQmx interface.

Tests are fully mocked — nidaqmx is NOT required. Each test patches
only the specific dependencies needed, keeping tests isolated and fast.

Test groups mirror the task groups in openspec/changes/wrapper/tasks.md.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_ni_task(
    task_name="TestInput",
    sample_rate=25600.0,
    channels=None,
    settings_file=None,
):
    """Create a mock NITask object that behaves like the real one."""
    task = MagicMock()
    task.task_name = task_name
    task.sample_rate = sample_rate
    task.settings_file = settings_file
    task.channels = channels or {
        "ch0": {"device_ind": 0, "channel_ind": 0, "sensitivity": 100.0,
                "sensitivity_units": "mV/g", "units": "g", "units_str": "g",
                "serial_nr": None, "scale": None, "min_val": None,
                "max_val": None, "custom_scale_name": ""},
        "ch1": {"device_ind": 0, "channel_ind": 1, "sensitivity": 100.0,
                "sensitivity_units": "mV/g", "units": "g", "units_str": "g",
                "serial_nr": None, "scale": None, "min_val": None,
                "max_val": None, "custom_scale_name": ""},
    }
    task.channel_list = list(task.channels.keys())
    task.number_of_ch = len(task.channels)
    task.device_list = ["cDAQ1Mod1"]

    # Mock the underlying nidaqmx task (set after initiate)
    inner_task = MagicMock()
    inner_task.channel_names = list(task.channels.keys())
    inner_task.number_of_channels = len(task.channels)
    inner_task.devices = [MagicMock(name="cDAQ1Mod1")]
    inner_task.devices[0].name = "cDAQ1Mod1"
    inner_task.timing = MagicMock()
    inner_task.timing.samp_clk_rate = sample_rate
    task.task = inner_task

    return task


def _make_mock_ni_task_output(
    task_name="TestOutput",
    sample_rate=10000.0,
    channels=None,
):
    """Create a mock NITaskOutput object."""
    task = MagicMock()
    task.task_name = task_name
    task.sample_rate = sample_rate
    task.channels = channels or {
        "ao0": {"device_ind": 0, "channel_ind": 0, "min_val": -10.0, "max_val": 10.0},
    }
    task.channel_list = list(task.channels.keys())
    task.number_of_channels = len(task.channels)
    task.device_list = ["cDAQ1Mod2"]

    inner_task = MagicMock()
    inner_task.channel_names = list(task.channels.keys())
    inner_task.number_of_channels = len(task.channels)
    inner_task.devices = [MagicMock(name="cDAQ1Mod2")]
    inner_task.devices[0].name = "cDAQ1Mod2"
    inner_task.timing = MagicMock()
    inner_task.timing.samp_clk_rate = sample_rate
    task.task = inner_task

    return task


def _make_mock_nidaqmx_task(channel_names=None, sample_rate=25600.0, device_names=None):
    """Create a mock nidaqmx.task.Task (loaded from NI MAX)."""
    task = MagicMock()
    task.channel_names = channel_names or ["ch0", "ch1"]
    task.number_of_channels = len(task.channel_names)

    devices = []
    for name in (device_names or ["cDAQ1Mod1"]):
        d = MagicMock()
        d.name = name
        devices.append(d)
    task.devices = devices

    task.timing = MagicMock()
    task.timing.samp_clk_rate = sample_rate
    task.is_task_done.return_value = True

    return task


# ---------------------------------------------------------------------------
# Module-level patches
# ---------------------------------------------------------------------------

@pytest.fixture
def wrapper_module():
    """Import the wrapper module with nidaqmx mocked.

    Carefully saves and restores ALL affected sys.modules entries so
    that other test files (test_utils, test_digital, etc.) are not
    polluted by the mock nidaqmx references.
    """
    import sys

    # Modules that will be mocked or indirectly affected
    _MOCK_TARGETS = [
        "nidaqmx", "nidaqmx.constants", "nidaqmx.system",
        "nidaqmx.task", "nidaqmx.errors",
        "pyTrigger",
    ]
    _NIDAQWRAPPER_MODULES = [
        "nidaqwrapper", "nidaqwrapper.wrapper", "nidaqwrapper.utils",
        "nidaqwrapper.task_input", "nidaqwrapper.task_output",
        "nidaqwrapper.digital",
    ]

    # Save current state
    saved = {}
    for mod_name in _MOCK_TARGETS + _NIDAQWRAPPER_MODULES:
        saved[mod_name] = sys.modules.get(mod_name)

    # Mock nidaqmx
    mock_nidaqmx = MagicMock()
    mock_nidaqmx.constants.READ_ALL_AVAILABLE = -1
    mock_nidaqmx.constants.AcquisitionType.CONTINUOUS = "CONTINUOUS"

    for mod_name in _MOCK_TARGETS:
        if mod_name == "nidaqmx":
            sys.modules[mod_name] = mock_nidaqmx
        elif mod_name == "pyTrigger":
            sys.modules[mod_name] = MagicMock()
        else:
            sys.modules[mod_name] = MagicMock()

    # Remove cached nidaqwrapper modules so they reimport with mocked nidaqmx
    for mod_name in _NIDAQWRAPPER_MODULES:
        sys.modules.pop(mod_name, None)

    import nidaqwrapper.wrapper as wrapper_mod

    yield wrapper_mod

    # Restore ALL modules to their pre-fixture state
    for mod_name, original in saved.items():
        if original is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original


@pytest.fixture
def NIDAQWrapper(wrapper_module):
    """Return the NIDAQWrapper class."""
    return wrapper_module.NIDAQWrapper


# ===================================================================
# 1. Constructor and Dual Initialization
# ===================================================================

class TestConstructor:
    """Task group 1: Constructor and dual initialization."""

    def test_empty_constructor_state(self, NIDAQWrapper):
        """1.1 Empty constructor sets correct default state."""
        w = NIDAQWrapper()
        assert w._configured is False
        assert w._connected is False
        assert w._trigger_is_set is False
        assert isinstance(w._lock, type(threading.RLock()))

    def test_empty_constructor_timing_defaults(self, NIDAQWrapper):
        """1.1 Default timing parameters."""
        w = NIDAQWrapper()
        assert w.acquisition_sleep == 0.01
        assert w.post_trigger_delay == 0.05

    def test_constructor_with_task_in_calls_configure(self, NIDAQWrapper):
        """1.2 Constructor with task_in kwarg calls configure."""
        w = NIDAQWrapper(task_in="MyTask")
        assert w._configured is True
        assert w._task_in_name == "MyTask"

    def test_constructor_with_task_out_calls_configure(self, NIDAQWrapper):
        """1.2 Constructor with task_out kwarg calls configure."""
        w = NIDAQWrapper(task_out="MyOutput")
        assert w._configured is True

    def test_constructor_with_both_tasks(self, NIDAQWrapper):
        """1.2 Constructor with both task_in and task_out."""
        w = NIDAQWrapper(task_in="Input", task_out="Output")
        assert w._configured is True
        assert w._task_in_name == "Input"
        assert w._task_out_name == "Output"

    def test_constructor_custom_timing(self, NIDAQWrapper):
        """1.3 Constructor with custom timing kwargs."""
        w = NIDAQWrapper(acquisition_sleep=0.005, post_trigger_delay=0.1)
        assert w.acquisition_sleep == 0.005
        assert w.post_trigger_delay == 0.1

    def test_constructor_does_not_call_connect(self, NIDAQWrapper):
        """Constructor should NOT call connect() even with kwargs."""
        w = NIDAQWrapper(task_in="MyTask")
        assert w._connected is False
        assert w._connect_called is False


# ===================================================================
# 2. configure() with NI MAX Task Name Strings
# ===================================================================

class TestConfigureStrings:
    """Task group 2: configure() with NI MAX task name strings."""

    def test_configure_input_string(self, NIDAQWrapper):
        """2.1 configure(task_in=string) stores name and sets flag."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        assert w._task_in_name == "InputTask"
        assert w._task_in_is_str is True
        assert w._task_in_is_obj is False
        assert w._configured is True

    def test_configure_output_string(self, NIDAQWrapper):
        """2.2 configure(task_out=string) stores output task name."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        assert w._task_out_name == "OutputTask"
        assert w._task_out_is_str is True

    def test_configure_both_strings(self, NIDAQWrapper):
        """2.3 configure() with both input and output strings."""
        w = NIDAQWrapper()
        w.configure(task_in="Input", task_out="Output")
        assert w._task_in_name == "Input"
        assert w._task_out_name == "Output"
        assert w._configured is True

    def test_configure_resets_state(self, NIDAQWrapper):
        """2.4 Calling configure() twice resets previous state."""
        w = NIDAQWrapper()
        w.configure(task_in="First")
        w._trigger_is_set = True
        w.configure(task_in="Second")
        assert w._task_in_name == "Second"
        assert w._trigger_is_set is False

    def test_configure_timing_kwargs(self, NIDAQWrapper):
        """2.5 configure() with timing kwargs overrides defaults."""
        w = NIDAQWrapper()
        w.configure(task_in="Task", acquisition_sleep=0.02, post_trigger_delay=0.1)
        assert w.acquisition_sleep == 0.02
        assert w.post_trigger_delay == 0.1


# ===================================================================
# 3. configure() with NITask/NITaskOutput Objects
# ===================================================================

class TestConfigureObjects:
    """Task group 3: configure() with NITask/NITaskOutput objects."""

    def test_configure_ni_task_object(self, NIDAQWrapper, wrapper_module):
        """3.1 configure(task_in=NITask) stores object and extracts config."""
        mock_task = _make_mock_ni_task()

        with patch.object(wrapper_module, "NITask", new=type(mock_task)):
            w = NIDAQWrapper()
            w.configure(task_in=mock_task)

        assert w._task_in_obj is mock_task
        assert w._task_in_is_obj is True
        assert w._task_in_is_str is False
        assert w._task_in_name_str == "TestInput"
        assert w._task_in_sample_rate == 25600.0

    def test_configure_ni_task_output_object(self, NIDAQWrapper, wrapper_module):
        """3.2 configure(task_out=NITaskOutput) stores output object."""
        mock_task_out = _make_mock_ni_task_output()

        with patch.object(wrapper_module, "NITaskOutput", new=type(mock_task_out)):
            w = NIDAQWrapper()
            w.configure(task_out=mock_task_out)

        assert w._task_out_obj is mock_task_out
        assert w._task_out_is_obj is True

    def test_configure_mixed_types(self, NIDAQWrapper, wrapper_module):
        """3.3 configure() with mixed types: NITask + string output."""
        mock_task = _make_mock_ni_task()

        with patch.object(wrapper_module, "NITask", new=type(mock_task)):
            w = NIDAQWrapper()
            w.configure(task_in=mock_task, task_out="OutputTask")

        assert w._task_in_is_obj is True
        assert w._task_out_is_str is True

    def test_configure_deep_copies_channels(self, NIDAQWrapper, wrapper_module):
        """3.4 configure() deep-copies channel config for reconnection."""
        mock_task = _make_mock_ni_task()

        with patch.object(wrapper_module, "NITask", new=type(mock_task)):
            w = NIDAQWrapper()
            w.configure(task_in=mock_task)

        # Should be a deep copy, not the same dict
        assert w._task_in_channels is not mock_task.channels
        assert w._task_in_channels == mock_task.channels


# ===================================================================
# 4. connect() and disconnect() Lifecycle
# ===================================================================

class TestConnectDisconnect:
    """Task group 4: connect() and disconnect() lifecycle."""

    def test_connect_with_string_loads_task(self, NIDAQWrapper, wrapper_module):
        """4.1 connect() with NI MAX string calls get_task_by_name."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")

        mock_loaded_task = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded_task) as mock_get, \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            result = w.connect()

        mock_get.assert_called_once_with("InputTask")
        assert result is True
        assert w._connected is True

    def test_connect_extracts_metadata_from_ni_max_task(self, NIDAQWrapper, wrapper_module):
        """4.1 connect() extracts channel names, sample rate, devices."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")

        mock_loaded_task = _make_mock_nidaqmx_task(
            channel_names=["accel_x", "accel_y"],
            sample_rate=51200.0,
            device_names=["cDAQ1Mod1"],
        )
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded_task), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        assert w._channel_names_in == ["accel_x", "accel_y"]
        assert w._sample_rate_in == 51200.0
        assert "cDAQ1Mod1" in w._required_devices

    def test_connect_with_ni_task_object_calls_initiate(self, NIDAQWrapper, wrapper_module):
        """4.2 connect() with NITask object calls initiate() exactly once.

        The wrapper calls initiate() without a start_task kwarg, relying
        on NITask's default (start_task=False). We verify the call was made
        but do NOT inspect kwargs — that is NITask's responsibility.
        """
        mock_task = _make_mock_ni_task()
        # The recreated task that connect() will build
        recreated_task = _make_mock_ni_task()

        with patch.object(wrapper_module, "NITask", new=type(mock_task)):
            w = NIDAQWrapper()
            w.configure(task_in=mock_task)

        # Patch NITask constructor so _recreate_ni_task_in returns our mock
        with patch.object(wrapper_module, "NITask", return_value=recreated_task) as mock_cls, \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        # Verify initiate() was called exactly once — do NOT assert on start_task kwarg.
        recreated_task.initiate.assert_called_once()

    def test_connect_closes_previous_tasks(self, NIDAQWrapper, wrapper_module):
        """4.3 connect() closes previous tasks before opening new ones."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")

        mock_first_task = _make_mock_nidaqmx_task()
        mock_second_task = _make_mock_nidaqmx_task()

        with patch.object(wrapper_module, "get_task_by_name", side_effect=[mock_first_task, mock_second_task]), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()
            # First task is now the active _task_in
            w.connect()  # Second connect should close first task

        mock_first_task.close.assert_called_once()

    def test_connect_returns_false_on_failure(self, NIDAQWrapper, wrapper_module):
        """4.4 connect() returns False when task load fails."""
        w = NIDAQWrapper()
        w.configure(task_in="BadTask")

        with patch.object(wrapper_module, "get_task_by_name", side_effect=KeyError("Not found")):
            result = w.connect()

        assert result is False
        assert w._connected is False

    def test_connect_returns_false_when_ping_fails(self, NIDAQWrapper, wrapper_module):
        """4.4 connect() returns False when ping fails."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")

        mock_loaded_task = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded_task), \
             patch.object(wrapper_module, "get_connected_devices", return_value=set()):
            result = w.connect()

        assert result is False

    def test_connect_ni_task_recreation_on_second_connect(self, NIDAQWrapper, wrapper_module):
        """4.5 Second connect() recreates NITask from stored channel config."""
        mock_task = _make_mock_ni_task()
        recreated = _make_mock_ni_task()
        mock_ni_task_class = MagicMock(return_value=recreated)

        w = NIDAQWrapper()
        # Manually set up as if configured with an NITask object
        w._configured = True
        w._task_in_is_obj = True
        w._task_in_is_str = False
        w._task_in_obj = mock_task
        w._task_in_channels = {"ch0": mock_task.channels["ch0"], "ch1": mock_task.channels["ch1"]}
        w._task_in_name_str = "TestInput"
        w._task_in_sample_rate = 25600.0
        w._task_in_settings_file = None
        w._task_out_is_str = False
        w._task_out_is_obj = False

        with patch.object(wrapper_module, "NITask", mock_ni_task_class) as mock_cls, \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        # NITask constructor should be called for re-creation
        mock_cls.assert_called_once()

    def test_disconnect_closes_input_task(self, NIDAQWrapper, wrapper_module):
        """4.6 disconnect() closes input task."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")

        mock_loaded = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        w.disconnect()
        mock_loaded.close.assert_called()
        assert w._connected is False

    def test_disconnect_closes_output_task(self, NIDAQWrapper, wrapper_module):
        """4.7 disconnect() closes output task."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")

        mock_loaded = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        w.disconnect()
        mock_loaded.close.assert_called()

    def test_connect_with_both_input_and_output_tasks(self, NIDAQWrapper, wrapper_module):
        """4.x connect() loads both input and output NI MAX tasks simultaneously."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask", task_out="OutputTask")

        mock_input = _make_mock_nidaqmx_task(
            channel_names=["ch0", "ch1"],
            sample_rate=25600.0,
            device_names=["cDAQ1Mod1"],
        )
        mock_output = _make_mock_nidaqmx_task(
            channel_names=["ao0"],
            sample_rate=10000.0,
            device_names=["cDAQ1Mod2"],
        )

        def _side_effect(name):
            return mock_input if name == "InputTask" else mock_output

        with patch.object(wrapper_module, "get_task_by_name", side_effect=_side_effect), \
             patch.object(wrapper_module, "get_connected_devices",
                          return_value={"cDAQ1Mod1", "cDAQ1Mod2"}):
            result = w.connect()

        assert result is True
        assert w._connected is True
        assert w._task_in is mock_input
        assert w._task_out is mock_output
        assert w._channel_names_in == ["ch0", "ch1"]
        assert w._channel_names_out == ["ao0"]

    def test_disconnect_idempotent(self, NIDAQWrapper):
        """4.8 Calling disconnect() twice does not raise."""
        w = NIDAQWrapper()
        result1 = w.disconnect()
        result2 = w.disconnect()
        assert result1 is True
        assert result2 is True

    def test_disconnect_when_never_connected(self, NIDAQWrapper):
        """4.9 disconnect() when no tasks were ever connected."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        result = w.disconnect()
        assert result is True


# ===================================================================
# 5. set_trigger() with pyTrigger
# ===================================================================

class TestSetTrigger:
    """Task group 5: set_trigger() with pyTrigger."""

    def test_set_trigger_creates_pytrigger(self, NIDAQWrapper, wrapper_module):
        """5.1 set_trigger() creates pyTrigger with correct params."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")

        # Simulate being connected with 2 channels
        w._connected = True
        w._n_channels_in = 2
        w._channel_names_in = ["ch0", "ch1"]

        mock_pytrigger_cls = MagicMock()
        with patch.object(wrapper_module, "pyTrigger", mock_pytrigger_cls):
            w.set_trigger(
                n_samples=1000,
                trigger_channel=0,
                trigger_level=0.5,
                trigger_type="abs",
                presamples=100,
            )

        mock_pytrigger_cls.assert_called_once_with(
            rows=1000,
            channels=2,
            trigger_channel=0,
            trigger_level=0.5,
            trigger_type="abs",
            presamples=100,
        )
        assert w._trigger_is_set is True

    def test_set_trigger_defaults(self, NIDAQWrapper, wrapper_module):
        """5.2 set_trigger() with defaults: trigger_type='abs', presamples=0."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2
        w._channel_names_in = ["ch0", "ch1"]

        mock_pytrigger_cls = MagicMock()
        with patch.object(wrapper_module, "pyTrigger", mock_pytrigger_cls):
            w.set_trigger(n_samples=500, trigger_channel=0, trigger_level=1.0)

        mock_pytrigger_cls.assert_called_once_with(
            rows=500,
            channels=2,
            trigger_channel=0,
            trigger_level=1.0,
            trigger_type="abs",
            presamples=0,
        )

    def test_set_trigger_no_input_task_raises(self, NIDAQWrapper):
        """5.3 set_trigger() raises ValueError with no input task."""
        w = NIDAQWrapper()
        with pytest.raises(ValueError, match="input task"):
            w.set_trigger(n_samples=100, trigger_channel=0, trigger_level=0.5)

    def test_set_trigger_sets_flag(self, NIDAQWrapper, wrapper_module):
        """5.4 set_trigger() sets _trigger_is_set=True."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 1
        w._channel_names_in = ["ch0"]

        with patch.object(wrapper_module, "pyTrigger", MagicMock()):
            w.set_trigger(n_samples=100, trigger_channel=0, trigger_level=0.5)

        assert w._trigger_is_set is True


# ===================================================================
# 6. acquire() RuntimeError Guard
# ===================================================================

class TestAcquireGuard:
    """Task group 6: acquire() RuntimeError guard."""

    def test_acquire_raises_without_trigger(self, NIDAQWrapper):
        """6.1 acquire() raises RuntimeError when trigger not set."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2

        with pytest.raises(RuntimeError, match="set_trigger"):
            w.acquire()

    def test_acquire_raises_without_input_task(self, NIDAQWrapper):
        """6.2 acquire() raises ValueError when no input task configured."""
        w = NIDAQWrapper()
        with pytest.raises(ValueError, match="input task"):
            w.acquire()


# ===================================================================
# 7. acquire() Triggered Loop
# ===================================================================

class TestAcquireLoop:
    """Task group 7: acquire() triggered acquisition loop."""

    def _setup_acquire(self, NIDAQWrapper, wrapper_module, n_channels=2):
        """Helper: set up a wrapper ready for acquire().

        Uses add_data side effect to set trigger.finished = True on first call,
        simulating immediate trigger completion.
        """
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._connect_called = True
        w._n_channels_in = n_channels
        w._channel_names_in = [f"ch{i}" for i in range(n_channels)]
        w._sample_rate_in = 25600.0
        w.acquisition_sleep = 0  # speed up tests
        w.post_trigger_delay = 0

        # Mock the underlying nidaqmx task
        mock_inner_task = MagicMock()
        if n_channels == 1:
            mock_inner_task.read.return_value = [0.0] * 100
        else:
            mock_inner_task.read.return_value = [[0.0] * 100] * n_channels
        w._task_in = mock_inner_task

        # Mock trigger — add_data side effect finishes trigger on first call
        mock_trigger = MagicMock()
        mock_trigger.finished = False
        mock_trigger.rows = 1000

        def _add_data_finishes(data):
            mock_trigger.finished = True

        mock_trigger.add_data.side_effect = _add_data_finishes
        mock_trigger.get_data.return_value = np.zeros((1000, n_channels))
        w.trigger = mock_trigger
        w._trigger_is_set = True

        return w, mock_inner_task, mock_trigger

    def test_acquire_starts_task(self, NIDAQWrapper, wrapper_module):
        """7.1 acquire() starts the input task."""
        w, mock_task, _ = self._setup_acquire(NIDAQWrapper, wrapper_module)
        w.acquire()
        mock_task.start.assert_called_once()

    def test_acquire_flushes_buffer(self, NIDAQWrapper, wrapper_module):
        """7.2 acquire() flushes buffer (first read discarded)."""
        w, mock_task, _ = self._setup_acquire(NIDAQWrapper, wrapper_module)
        w.acquire()
        # First read is the flush + at least one loop read = >= 2 total reads
        assert mock_task.read.call_count >= 2

    def test_acquire_loop_until_trigger_finished(self, NIDAQWrapper, wrapper_module):
        """7.3 acquire() reads in loop until trigger.finished."""
        w, mock_task, mock_trigger = self._setup_acquire(NIDAQWrapper, wrapper_module)

        # Override: trigger finishes after 3 add_data calls
        call_count = [0]
        def side_effect(data):
            call_count[0] += 1
            if call_count[0] >= 3:
                mock_trigger.finished = True

        mock_trigger.add_data.side_effect = side_effect
        w.acquire()
        assert call_count[0] >= 3

    def test_acquire_single_channel_reshape(self, NIDAQWrapper, wrapper_module):
        """7.4 acquire() reshapes single-channel 1D data to 2D."""
        w, _, mock_trigger = self._setup_acquire(NIDAQWrapper, wrapper_module, n_channels=1)
        mock_trigger.get_data.return_value = np.zeros((1000, 1))

        # Capture what add_data receives
        added_data = []
        def capture(data):
            added_data.append(data.copy())
            mock_trigger.finished = True
        mock_trigger.add_data.side_effect = capture

        result = w.acquire()
        assert result.shape == (1000, 1)
        # add_data should have received (n_samples, 1) shaped data
        assert added_data[0].ndim == 2
        assert added_data[0].shape[1] == 1

    def test_acquire_multi_channel_transpose(self, NIDAQWrapper, wrapper_module):
        """7.5 acquire() transposes multi-channel 2D data for pyTrigger."""
        w, mock_task, mock_trigger = self._setup_acquire(NIDAQWrapper, wrapper_module)

        added_data = []
        def capture(data):
            added_data.append(data.copy())
            mock_trigger.finished = True
        mock_trigger.add_data.side_effect = capture
        mock_task.read.return_value = [list(range(100)), list(range(100, 200))]

        w.acquire()

        # add_data should have received (n_samples, n_channels) format
        assert len(added_data) > 0
        assert added_data[0].shape == (100, 2)

    def test_acquire_post_trigger_delay(self, NIDAQWrapper, wrapper_module):
        """7.6 acquire() sleeps for post_trigger_delay after trigger finishes."""
        w, _, _ = self._setup_acquire(NIDAQWrapper, wrapper_module)
        w.post_trigger_delay = 0.123

        with patch.object(wrapper_module.time, "sleep") as mock_sleep:
            w.acquire()

        # The post-trigger sleep must be called with the configured delay.
        # (acquisition_sleep may also be called, so check any call matches)
        sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
        assert 0.123 in sleep_values, (
            f"Expected time.sleep(0.123) for post_trigger_delay; "
            f"actual sleep calls: {sleep_values}"
        )

    def test_acquire_stops_task(self, NIDAQWrapper, wrapper_module):
        """7.7 acquire() stops the input task after acquisition."""
        w, mock_task, _ = self._setup_acquire(NIDAQWrapper, wrapper_module)
        w.acquire()
        mock_task.stop.assert_called_once()

    def test_acquire_returns_correct_shape(self, NIDAQWrapper, wrapper_module):
        """7.8 acquire() returns (n_samples, n_channels) array."""
        w, _, mock_trigger = self._setup_acquire(NIDAQWrapper, wrapper_module)
        expected = np.random.randn(1000, 2)
        mock_trigger.get_data.return_value = expected

        result = w.acquire()
        assert result.shape == (1000, 2)
        np.testing.assert_array_equal(result, expected)

    def test_acquire_resets_trigger_on_second_call(self, NIDAQWrapper, wrapper_module):
        """7.9 acquire() resets trigger on second call."""
        w, _, mock_trigger = self._setup_acquire(NIDAQWrapper, wrapper_module)

        # First acquire
        w.acquire()
        # Reset side effect for second call
        mock_trigger.finished = False
        mock_trigger.add_data.side_effect = lambda data: setattr(mock_trigger, 'finished', True)

        w.acquire()
        # ringbuff.clear should be called at least twice (once per _reset_trigger)
        assert mock_trigger.ringbuff.clear.call_count >= 2


# ===================================================================
# 8. acquire() with return_dict=True
# ===================================================================

class TestAcquireReturnDict:
    """Task group 8: acquire() with return_dict=True."""

    def _setup_acquire_dict(self, NIDAQWrapper, wrapper_module):
        """Helper: set up wrapper for dict-return acquire."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._connect_called = True
        w._n_channels_in = 2
        w._channel_names_in = ["accel_x", "accel_y"]
        w._sample_rate_in = 1000.0
        w.acquisition_sleep = 0
        w.post_trigger_delay = 0

        mock_inner_task = MagicMock()
        mock_inner_task.read.return_value = [[0.0] * 50, [0.0] * 50]
        w._task_in = mock_inner_task

        mock_trigger = MagicMock()
        mock_trigger.finished = False
        mock_trigger.rows = 100
        mock_trigger.add_data.side_effect = lambda data: setattr(mock_trigger, 'finished', True)
        w.trigger = mock_trigger
        w._trigger_is_set = True

        return w, mock_inner_task, mock_trigger

    def test_return_dict_has_channel_keys(self, NIDAQWrapper, wrapper_module):
        """8.1 acquire(return_dict=True) returns dict with channel names."""
        w, _, mock_trigger = self._setup_acquire_dict(NIDAQWrapper, wrapper_module)
        mock_trigger.get_data.return_value = np.random.randn(100, 2)

        result = w.acquire(return_dict=True)

        assert isinstance(result, dict)
        assert "accel_x" in result
        assert "accel_y" in result

    def test_return_dict_has_time_key(self, NIDAQWrapper, wrapper_module):
        """8.2 Dict includes 'time' key with correct values."""
        w, _, mock_trigger = self._setup_acquire_dict(NIDAQWrapper, wrapper_module)
        mock_trigger.get_data.return_value = np.random.randn(100, 2)

        result = w.acquire(return_dict=True)

        assert "time" in result
        expected_time = np.arange(100) / 1000.0
        np.testing.assert_array_almost_equal(result["time"], expected_time)

    def test_return_dict_channel_values_1d(self, NIDAQWrapper, wrapper_module):
        """8.3 Each channel value is a 1D numpy array."""
        w, _, mock_trigger = self._setup_acquire_dict(NIDAQWrapper, wrapper_module)
        data = np.random.randn(100, 2)
        mock_trigger.get_data.return_value = data

        result = w.acquire(return_dict=True)

        assert result["accel_x"].ndim == 1
        assert len(result["accel_x"]) == 100
        np.testing.assert_array_equal(result["accel_x"], data[:, 0])


# ===================================================================
# 9. acquire() with blocking=False (Future)
# ===================================================================

class TestAcquireNonBlocking:
    """Task group 9: acquire() with blocking=False."""

    def _setup_nb(self, NIDAQWrapper, wrapper_module):
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._connect_called = True
        w._n_channels_in = 2
        w._channel_names_in = ["ch0", "ch1"]
        w._sample_rate_in = 25600.0
        w.acquisition_sleep = 0
        w.post_trigger_delay = 0

        mock_inner_task = MagicMock()
        mock_inner_task.read.return_value = [[0.0] * 50, [0.0] * 50]
        w._task_in = mock_inner_task

        mock_trigger = MagicMock()
        mock_trigger.finished = False
        mock_trigger.rows = 100
        mock_trigger.add_data.side_effect = lambda data: setattr(mock_trigger, 'finished', True)
        mock_trigger.get_data.return_value = np.zeros((100, 2))
        w.trigger = mock_trigger
        w._trigger_is_set = True

        return w, mock_inner_task, mock_trigger

    def test_blocking_true_returns_data(self, NIDAQWrapper, wrapper_module):
        """9.1 acquire(blocking=True) returns data directly."""
        w, _, _ = self._setup_nb(NIDAQWrapper, wrapper_module)
        result = w.acquire(blocking=True)
        assert isinstance(result, np.ndarray)

    def test_blocking_false_returns_future(self, NIDAQWrapper, wrapper_module):
        """9.2 acquire(blocking=False) returns Future."""
        w, _, _ = self._setup_nb(NIDAQWrapper, wrapper_module)
        result = w.acquire(blocking=False)
        assert isinstance(result, Future)

    def test_future_result_returns_data(self, NIDAQWrapper, wrapper_module):
        """9.3 future.result() returns acquired data."""
        w, _, _ = self._setup_nb(NIDAQWrapper, wrapper_module)
        future = w.acquire(blocking=False)
        data = future.result(timeout=5)
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 2)

    def test_future_done_after_completion(self, NIDAQWrapper, wrapper_module):
        """9.4 future.done() returns True after acquisition completes."""
        w, _, _ = self._setup_nb(NIDAQWrapper, wrapper_module)
        future = w.acquire(blocking=False)
        # Wait for acquisition to finish before checking done()
        future.result(timeout=5)
        assert future.done() is True


# ===================================================================
# 10. read_all_available() for LDAQ Integration
# ===================================================================

class TestReadAllAvailable:
    """Task group 10: read_all_available() for LDAQ integration."""

    def test_read_all_available_transposes(self, NIDAQWrapper, wrapper_module):
        """10.1 read_all_available() returns (n_samples, n_channels)."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True

        # Mock the NITask's acquire_base returning (n_channels, n_samples)
        mock_ni_task = MagicMock()
        mock_ni_task.acquire_base.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        w._task_in_obj_active = mock_ni_task

        result = w.read_all_available()
        assert result.shape == (3, 2)  # (n_samples, n_channels)

    def test_read_all_available_empty(self, NIDAQWrapper, wrapper_module):
        """10.2 read_all_available() returns empty array when no data."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2

        mock_ni_task = MagicMock()
        mock_ni_task.acquire_base.return_value = np.empty((2, 0))
        w._task_in_obj_active = mock_ni_task

        result = w.read_all_available()
        assert result.shape == (0, 2)

    def test_read_all_available_no_trigger(self, NIDAQWrapper, wrapper_module):
        """10.3 read_all_available() does NOT use pyTrigger."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True

        mock_ni_task = MagicMock()
        mock_ni_task.acquire_base.return_value = np.array([[1, 2], [3, 4]])
        w._task_in_obj_active = mock_ni_task

        # No trigger should be involved
        w._trigger_is_set = False
        result = w.read_all_available()
        assert result.shape == (2, 2)

    def test_read_all_available_single_channel(self, NIDAQWrapper, wrapper_module):
        """10.4 read_all_available() reshapes single-channel data."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 1

        mock_ni_task = MagicMock()
        # acquire_base returns (1, n_samples) for single channel
        mock_ni_task.acquire_base.return_value = np.array([[1, 2, 3, 4]])
        w._task_in_obj_active = mock_ni_task

        result = w.read_all_available()
        assert result.shape == (4, 1)


# ===================================================================
# 11. read() Single Sample
# ===================================================================

class TestRead:
    """Task group 11: read() single sample."""

    def test_read_returns_single_sample(self, NIDAQWrapper, wrapper_module):
        """11.1 read() returns (n_channels,) array."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2

        mock_task = MagicMock()
        mock_task.read.return_value = [1.5, 2.5]
        w._task_in = mock_task

        result = w.read()
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [1.5, 2.5])

    def test_read_raises_during_continuous_acquisition(self, NIDAQWrapper, wrapper_module):
        """11.2 read() raises ValueError when continuous acquisition is running."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2

        mock_task = MagicMock()
        mock_task.read.return_value = [1.5, 2.5]
        w._task_in = mock_task

        # Simulate a continuous acquisition in progress
        w._acquire_running = True

        with pytest.raises(ValueError, match="acquisition"):
            w.read()

    def test_read_no_input_task_raises(self, NIDAQWrapper):
        """11.3 read() raises ValueError when no input task configured."""
        w = NIDAQWrapper()
        with pytest.raises(ValueError, match="input task"):
            w.read()

    def test_read_single_channel_scalar_returns_1d(self, NIDAQWrapper, wrapper_module):
        """11.4 read() reshapes 0-D scalar result to (1,) for single-channel tasks.

        nidaqmx may return a bare float for a single-channel task when 1 sample
        is requested. np.array(scalar) produces a 0-D array; this must be
        reshaped to (1,) so callers always receive a 1-D array.
        """
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 1

        mock_task = MagicMock()
        # Simulate nidaqmx returning a bare float (0-D after np.array())
        mock_task.read.return_value = 3.14
        w._task_in = mock_task

        result = w.read()
        assert result.ndim == 1
        assert result.shape == (1,)
        np.testing.assert_almost_equal(result[0], 3.14)


# ===================================================================
# 12. generate() Continuous with Buffer Management
# ===================================================================

class TestGenerate:
    """Task group 12: generate() continuous generation."""

    def test_generate_2d_data(self, NIDAQWrapper, wrapper_module):
        """12.1 generate() with 2D data: transposes, writes, and is C-contiguous.

        Pitfall #1: data.T produces a Fortran-order view; nidaqmx C layer
        requires C-contiguous memory. np.ascontiguousarray() is mandatory.
        """
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._task_out_is_str = True
        w._n_channels_out = 2
        w._generation_running = False

        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        w.generate(data)

        mock_task.write.assert_called_once()
        written = mock_task.write.call_args[0][0]
        assert written.shape == (2, 3)  # transposed to (n_channels, n_samples)
        assert written.flags["C_CONTIGUOUS"], (
            "Written array must be C-contiguous (np.ascontiguousarray required "
            "after .T to avoid nidaqmx C layer memory errors — see Pitfall #1)"
        )

    def test_generate_1d_single_channel(self, NIDAQWrapper, wrapper_module):
        """12.2 generate() with 1D data for single channel."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._task_out_is_str = True
        w._n_channels_out = 1
        w._generation_running = False

        data = np.array([1.0, 2.0, 3.0])
        w.generate(data)

        mock_task.write.assert_called_once()

    def test_generate_overwrite_stops_first(self, NIDAQWrapper, wrapper_module):
        """12.3 generate(overwrite=True) stops current output first."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._task_out_is_str = True
        w._n_channels_out = 1
        w._generation_running = True

        data = np.array([1.0, 2.0, 3.0])
        w.generate(data, overwrite=True)

        mock_task.stop.assert_called()

    def test_generate_no_overwrite_raises_when_running(self, NIDAQWrapper, wrapper_module):
        """12.4 generate(overwrite=False) raises when output is running."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = False
        w._task_out = mock_task
        w._n_channels_out = 1
        w._generation_running = True

        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="running"):
            w.generate(data, overwrite=False)

    def test_generate_no_output_task_raises(self, NIDAQWrapper):
        """12.5 generate() raises ValueError with no output task."""
        w = NIDAQWrapper()
        with pytest.raises(ValueError, match="output task"):
            w.generate(np.array([1.0]))

    def test_generate_shape_mismatch_raises(self, NIDAQWrapper, wrapper_module):
        """12.6 generate() validates data shape vs channel count."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = False

        # 3 columns but only 2 output channels
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="channels"):
            w.generate(data)


# ===================================================================
# 13. write() Single Sample with 2-Sample Minimum
# ===================================================================

class TestWrite:
    """Task group 13: write() single sample."""

    def test_write_single_float(self, NIDAQWrapper, wrapper_module):
        """13.1 write(float) for single-channel: duplicated to 2 samples."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._n_channels_out = 1
        w._generation_running = False

        w.write(2.5)

        mock_task.write.assert_called_once()
        written = mock_task.write.call_args[0][0]
        # Should be 2 samples for 1 channel
        assert len(written) == 2

    def test_write_multi_channel(self, NIDAQWrapper, wrapper_module):
        """13.2 write(array) for multi-channel: duplicated to 2 rows, C-contiguous.

        Pitfall #1: stacking + .T produces a Fortran-order view; the written
        array must be C-contiguous for the nidaqmx C layer.
        """
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = False

        w.write(np.array([1.0, 2.0]))

        mock_task.write.assert_called_once()
        written = mock_task.write.call_args[0][0]
        assert written.shape == (2, 2)  # (n_channels, 2_samples)
        assert written.flags["C_CONTIGUOUS"], (
            "Written array must be C-contiguous (np.ascontiguousarray required "
            "after .T — see Pitfall #1)"
        )

    def test_write_overwrite_stops_first(self, NIDAQWrapper, wrapper_module):
        """13.3 write(overwrite=True) stops current generation."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._n_channels_out = 1
        w._generation_running = True

        w.write(1.0, overwrite=True)
        mock_task.stop.assert_called()

    def test_write_shape_mismatch_raises(self, NIDAQWrapper, wrapper_module):
        """13.4 write() raises ValueError for shape mismatch."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = False

        with pytest.raises(ValueError, match="channels"):
            w.write(np.array([1.0, 2.0, 3.0]))  # 3 values, 2 channels

    def test_write_no_overwrite_raises_when_running(self, NIDAQWrapper, wrapper_module):
        """13.5 write() raises ValueError when running and overwrite=False."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = False
        w._task_out = mock_task
        w._n_channels_out = 1
        w._generation_running = True

        with pytest.raises(ValueError, match="running"):
            w.write(1.0, overwrite=False)


# ===================================================================
# 14. stop_generation() with Safe Zeros
# ===================================================================

class TestStopGeneration:
    """Task group 14: stop_generation() with safe zeros."""

    def test_stop_generation_stops_task(self, NIDAQWrapper, wrapper_module):
        """14.1 stop_generation() stops the output task."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = True

        w.stop_generation()
        mock_task.stop.assert_called()

    def test_stop_generation_writes_zeros(self, NIDAQWrapper, wrapper_module):
        """14.2 stop_generation() writes zeros to all channels before stopping."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = True

        w.stop_generation()

        mock_task.write.assert_called()
        written = mock_task.write.call_args[0][0]
        # The safe-shutdown zeros write must contain only zero values.
        assert np.all(written == 0.0), (
            f"stop_generation() must write all-zero data; got: {written}"
        )

    def test_stop_generation_no_output_task(self, NIDAQWrapper):
        """14.3 stop_generation() with no output task is no-op."""
        w = NIDAQWrapper()
        # Should not raise
        w.stop_generation()


# ===================================================================
# 15. ping() and check_state()
# ===================================================================

class TestPingAndCheckState:
    """Task group 15: ping() and check_state() with auto-reconnection."""

    def test_ping_true_when_devices_present(self, NIDAQWrapper, wrapper_module):
        """15.1 ping() returns True when required devices are connected."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._required_devices = {"cDAQ1Mod1"}

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1", "cDAQ1Mod2"}):
            assert w.ping() is True

    def test_ping_false_when_device_missing(self, NIDAQWrapper, wrapper_module):
        """15.2 ping() returns False when device is missing."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._required_devices = {"cDAQ1Mod1"}

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod2"}):
            assert w.ping() is False

    def test_ping_false_when_not_configured(self, NIDAQWrapper):
        """15.3 ping() returns False when not configured."""
        w = NIDAQWrapper()
        assert w.ping() is False

    def test_check_state_connected(self, NIDAQWrapper, wrapper_module):
        """15.4 check_state() returns 'connected' when all OK."""
        w = NIDAQWrapper()
        w._connected = True
        w._configured = True
        w._required_devices = {"cDAQ1Mod1"}

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            assert w.check_state() == "connected"

    def test_check_state_reconnected(self, NIDAQWrapper, wrapper_module):
        """15.5 check_state() returns 'reconnected' on successful reconnect."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = False
        w._connect_called = True

        # Make connect() succeed
        mock_loaded = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            result = w.check_state()

        assert result == "reconnected"

    def test_check_state_connection_lost(self, NIDAQWrapper, wrapper_module):
        """15.6 check_state() returns 'connection lost' on failed reconnect."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = False
        w._connect_called = True

        # Make connect() fail
        with patch.object(wrapper_module, "get_task_by_name", side_effect=KeyError("No")):
            result = w.check_state()

        assert result == "connection lost"

    def test_check_state_disconnected(self, NIDAQWrapper):
        """15.7 check_state() returns 'disconnected' when never connected."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        assert w.check_state() == "disconnected"


# ===================================================================
# 16. Introspection Methods
# ===================================================================

class TestIntrospection:
    """Task group 16: get_device_info, get_sample_rate, get_channel_names."""

    def test_get_device_info_input(self, NIDAQWrapper):
        """16.1 get_device_info() with input task."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True
        w._channel_names_in = ["ch0", "ch1"]
        w._sample_rate_in = 25600.0

        info = w.get_device_info()
        assert "input" in info
        assert info["input"]["channel_names"] == ["ch0", "ch1"]
        assert info["input"]["sample_rate"] == 25600.0

    def test_get_device_info_both_tasks(self, NIDAQWrapper):
        """16.2 get_device_info() with both input and output."""
        w = NIDAQWrapper()
        w.configure(task_in="Input", task_out="Output")
        w._connected = True
        w._channel_names_in = ["ch0"]
        w._sample_rate_in = 25600.0
        w._channel_names_out = ["ao0"]
        w._sample_rate_out = 10000.0

        info = w.get_device_info()
        assert "input" in info
        assert "output" in info

    def test_get_device_info_no_tasks(self, NIDAQWrapper):
        """16.3 get_device_info() with no tasks returns empty dict."""
        w = NIDAQWrapper()
        info = w.get_device_info()
        assert info == {}

    def test_get_sample_rate(self, NIDAQWrapper):
        """16.4 get_sample_rate() returns input sample rate."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._sample_rate_in = 51200.0
        assert w.get_sample_rate() == 51200.0

    def test_get_sample_rate_no_input_raises(self, NIDAQWrapper):
        """16.5 get_sample_rate() raises ValueError with no input task."""
        w = NIDAQWrapper()
        with pytest.raises(ValueError, match="input"):
            w.get_sample_rate()

    def test_get_channel_names(self, NIDAQWrapper):
        """16.6 get_channel_names() returns channel name list."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._channel_names_in = ["accel_x", "accel_y"]
        assert w.get_channel_names() == ["accel_x", "accel_y"]

    def test_get_channel_names_no_input(self, NIDAQWrapper):
        """16.7 get_channel_names() with no input returns empty list."""
        w = NIDAQWrapper()
        assert w.get_channel_names() == []


# ===================================================================
# 17. clear_buffer()
# ===================================================================

class TestClearBuffer:
    """Task group 17: clear_buffer()."""

    def test_clear_buffer_reads_and_discards(self, NIDAQWrapper, wrapper_module):
        """17.1 clear_buffer() reads and discards all buffered data."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        w._connected = True

        mock_ni_task = MagicMock()
        w._task_in_obj_active = mock_ni_task

        w.clear_buffer()
        mock_ni_task.acquire_base.assert_called_once()

    def test_clear_buffer_no_input_task(self, NIDAQWrapper):
        """17.3 clear_buffer() with no input task is no-op."""
        w = NIDAQWrapper()
        # Should not raise
        w.clear_buffer()


# ===================================================================
# 18. Context Manager
# ===================================================================

class TestContextManager:
    """Task group 18: Context manager protocol."""

    def test_enter_returns_self(self, NIDAQWrapper):
        """18.1 __enter__() returns self."""
        w = NIDAQWrapper()
        assert w.__enter__() is w

    def test_exit_calls_disconnect(self, NIDAQWrapper):
        """18.2 __exit__() calls disconnect()."""
        w = NIDAQWrapper()
        w.disconnect = MagicMock()
        w.__exit__(None, None, None)
        w.disconnect.assert_called_once()

    def test_exit_does_not_suppress_exceptions(self, NIDAQWrapper):
        """18.3 __exit__() returns None (does not suppress exceptions)."""
        w = NIDAQWrapper()
        result = w.__exit__(ValueError, ValueError("test"), None)
        assert result is None or result is False

    def test_context_manager_cleanup(self, NIDAQWrapper):
        """18.4 Context manager calls disconnect on exit."""
        w = NIDAQWrapper()
        w.disconnect = MagicMock()
        with w:
            pass
        w.disconnect.assert_called_once()

    def test_context_manager_cleanup_on_exception(self, NIDAQWrapper):
        """18.4 Context manager calls disconnect even on exception."""
        w = NIDAQWrapper()
        w.disconnect = MagicMock()
        with pytest.raises(RuntimeError):
            with w:
                raise RuntimeError("test")
        w.disconnect.assert_called_once()

    def test_del_best_effort(self, NIDAQWrapper):
        """18.5 __del__() suppresses exceptions."""
        w = NIDAQWrapper()
        w.disconnect = MagicMock(side_effect=RuntimeError("fail"))
        # __del__ should not raise
        w.__del__()


# ===================================================================
# 19. RLock Thread Safety
# ===================================================================

class TestRLockThreadSafety:
    """Task group 19: RLock thread safety."""

    def test_rlock_created_in_init(self, NIDAQWrapper):
        """19.1 RLock is created in __init__."""
        w = NIDAQWrapper()
        assert hasattr(w, "_lock")
        assert isinstance(w._lock, type(threading.RLock()))

    def test_connect_acquires_lock(self, NIDAQWrapper, wrapper_module):
        """19.4 connect() acquires lock."""
        w = NIDAQWrapper()
        w.configure(task_in="InputTask")
        original_lock = w._lock
        w._lock = MagicMock(wraps=original_lock)

        with patch.object(wrapper_module, "get_task_by_name", side_effect=KeyError("No")):
            w.connect()

        w._lock.__enter__.assert_called()

    def test_acquire_acquires_lock(self, NIDAQWrapper, wrapper_module):
        """19.2 acquire() acquires the RLock during execution."""
        # Use the same _setup_acquire pattern from TestAcquireLoop to avoid hanging.
        setup = TestAcquireLoop()
        w, _, _ = setup._setup_acquire(NIDAQWrapper, wrapper_module)

        original_lock = w._lock
        w._lock = MagicMock(wraps=original_lock)

        w.acquire()

        w._lock.__enter__.assert_called()

    def test_generate_acquires_lock(self, NIDAQWrapper, wrapper_module):
        """19.3 generate() acquires the RLock during execution."""
        w = NIDAQWrapper()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._task_out_is_str = True
        w._n_channels_out = 1
        w._generation_running = False

        original_lock = w._lock
        w._lock = MagicMock(wraps=original_lock)

        w.generate(np.array([1.0, 2.0, 3.0]))

        w._lock.__enter__.assert_called()

    def test_rlock_reentrancy(self, NIDAQWrapper):
        """19.5 RLock allows reentrant acquisition (no deadlock)."""
        w = NIDAQWrapper()
        # Simulate nested lock usage
        with w._lock:
            with w._lock:
                pass  # Should not deadlock


# ===================================================================
# 20. Configurable Timing Parameters
# ===================================================================

class TestTimingParameters:
    """Task group 20: Configurable timing parameters."""

    def test_default_timing(self, NIDAQWrapper):
        """20.1 Default timing values."""
        w = NIDAQWrapper()
        assert w.acquisition_sleep == 0.01
        assert w.post_trigger_delay == 0.05

    def test_timing_via_configure(self, NIDAQWrapper):
        """20.2 Custom timing via configure()."""
        w = NIDAQWrapper()
        w.configure(acquisition_sleep=0.002, post_trigger_delay=0.2)
        assert w.acquisition_sleep == 0.002
        assert w.post_trigger_delay == 0.2

    def test_timing_via_constructor(self, NIDAQWrapper):
        """20.3 Custom timing via constructor kwargs."""
        w = NIDAQWrapper(acquisition_sleep=0.003, post_trigger_delay=0.15)
        assert w.acquisition_sleep == 0.003
        assert w.post_trigger_delay == 0.15
