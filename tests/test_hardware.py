"""Hardware integration tests for nidaqwrapper.

All tests in this module require real NI-DAQmx hardware and are marked with
``@pytest.mark.hardware``.  Run them with::

    uv run pytest tests/test_hardware.py -v -m hardware

Or exclude them from a normal run with::

    uv run pytest -m "not hardware"

Hardware configuration
----------------------
- cDAQ2       : NI cDAQ-9174 USB chassis (device index 0)
- cDAQ2Mod3   : NI 9260 (BNC) — 2 AO channels (device index 1)
- cDAQ2Mod4   : NI 9215 (BNC) — 4 AI voltage channels (device index 2)
- Dev1        : PCIe-6320 — 16 AI, 24 DI, 24 DO, 0 AO (device index 3)

Note: cDAQ USB chassis may re-enumerate after system restart (cDAQ1 → cDAQ2).
Update the constants below if device names change.

NI MAX tasks
------------
- ``IM3`` : Pre-existing saved task (do NOT delete)

Notes
-----
Each test uses a unique task name to prevent NI MAX collisions.  All tests
clean up after themselves via try/finally or context managers so that a test
failure never leaves stale hardware tasks behind.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.hardware

# ---------------------------------------------------------------------------
# Hardware constants — update if the rig changes
# Set to None to skip tests that require that hardware.
# ---------------------------------------------------------------------------

# Analog input: cDAQ2Mod4 (NI 9215)
AI_DEVICE_NAME = "cDAQ2Mod4"
AI_DEVICE_INDEX = 2
AI_SAMPLE_RATE = 25600  # exact rate supported by NI 9215
AI_VOLTAGE_MIN = -10.0
AI_VOLTAGE_MAX = 10.0

# Analog output: cDAQ2Mod3 (NI 9260)
AO_DEVICE_NAME = "cDAQ2Mod3"
AO_DEVICE_INDEX = 1
AO_SAMPLE_RATE = 25600  # exact rate supported by NI 9260
AO_VOLTAGE_RANGE = 4.242  # NI 9260 max output ±4.242641V (use slightly under)

# Digital I/O: Dev1 (PCIe-6320) — set to None if no DI/DO hardware
DI_LINES = "Dev1/port0/line0"
DO_LINES = "Dev1/port1/line0"

# Second AI device for multi-task: Dev1 (PCIe-6320)
AI2_DEVICE_NAME = "Dev1"
AI2_DEVICE_INDEX = 3

# NI MAX task — set to None if no saved tasks exist
NI_MAX_TASK_NAME = "IM3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def skip_if_no_device():
    """Skip all tests in this module if no NI devices are connected."""
    from nidaqwrapper import list_devices

    devices = list_devices()
    if not devices:
        pytest.skip("No NI-DAQmx devices connected")


# ===========================================================================
# Task Group 3: Device Discovery Smoke Tests
# ===========================================================================


class TestDeviceDiscovery:
    """Verify list_devices() and list_tasks() against real hardware."""

    def test_list_devices_returns_real_devices(self) -> None:
        """list_devices() returns a non-empty list with correct structure.

        Each entry must have 'name' and 'product_type' keys.  Prints
        discovered devices for documentation.
        """
        from nidaqwrapper import list_devices

        devices = list_devices()

        assert isinstance(devices, list)
        assert len(devices) > 0, "Expected at least one NI device connected"

        for dev in devices:
            assert "name" in dev, f"Device entry missing 'name': {dev}"
            assert "product_type" in dev, f"Device entry missing 'product_type': {dev}"

        # Log discovered devices for documentation
        print("\n--- Discovered NI Devices ---")
        for dev in devices:
            print(f"  {dev['name']}: {dev['product_type']}")
        print("---")

        # Verify expected devices are present
        device_names = [d["name"] for d in devices]
        assert AI_DEVICE_NAME in device_names, (
            f"Expected {AI_DEVICE_NAME} in device list, got: {device_names}"
        )

    def test_list_tasks_returns_list(self) -> None:
        """list_tasks() returns a list (may contain saved NI MAX tasks).

        Prints discovered tasks for documentation.
        """
        from nidaqwrapper import list_tasks

        tasks = list_tasks()

        assert isinstance(tasks, list)

        # Log discovered tasks
        print(f"\n--- Discovered NI MAX Tasks: {tasks} ---")

        if tasks:
            for t in tasks:
                assert isinstance(t, str), f"Expected string task name, got: {type(t)}"
            assert NI_MAX_TASK_NAME in tasks, (
                f"Expected '{NI_MAX_TASK_NAME}' in task list, got: {tasks}"
            )


# ===========================================================================
# Task Group 4: NITask Voltage Channel Acquisition Tests
# ===========================================================================


class TestNITaskHardware:
    """Validate NITask voltage acquisition against real NI 9215."""

    def test_nitask_voltage_channel_acquisition(self) -> None:
        """Create a voltage channel, initiate, acquire, verify data shape.

        Uses the priming read pattern: first read(-1) may return 0 samples,
        so we discard it, sleep, and assert on the second read.
        """
        from nidaqwrapper import NITask

        task = NITask("hw_acq_voltage", sample_rate=AI_SAMPLE_RATE)
        try:
            task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
            task.initiate(start_task=True)

            # Priming read — discard
            time.sleep(0.1)
            task.acquire_base()

            # Real acquisition
            time.sleep(0.2)
            data = task.acquire_base()

            assert isinstance(data, np.ndarray), f"Expected ndarray, got {type(data)}"
            assert data.ndim == 2, f"Expected 2-D array, got shape {data.shape}"
            assert data.shape[0] == 1, f"Expected 1 channel row, got shape {data.shape}"
            assert data.shape[1] > 0, "Expected at least one sample"
        finally:
            task.clear_task()

    def test_nitask_sample_rate_accuracy(self) -> None:
        """Acquired sample count matches expected rate within 20% tolerance.

        Acquires for 0.5s at 25600 Hz. Expected ~12800 samples.
        """
        from nidaqwrapper import NITask

        task = NITask("hw_rate_check", sample_rate=AI_SAMPLE_RATE)
        try:
            task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
            task.initiate(start_task=True)

            # Priming read
            time.sleep(0.1)
            task.acquire_base()

            # Timed acquisition
            time.sleep(0.5)
            data = task.acquire_base()

            expected_samples = int(AI_SAMPLE_RATE * 0.5)
            actual_samples = data.shape[1]
            tolerance = 0.20
            assert abs(actual_samples - expected_samples) / expected_samples < tolerance, (
                f"Sample count {actual_samples} deviates >20% from expected "
                f"{expected_samples} (rate={AI_SAMPLE_RATE} Hz, duration=0.5s)"
            )
        finally:
            task.clear_task()

    def test_nitask_context_manager(self) -> None:
        """NITask context manager cleans up properly."""
        from nidaqwrapper import NITask

        task_name = "hw_ctx_nitask"
        with NITask(task_name, sample_rate=AI_SAMPLE_RATE) as task:
            task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
            task.initiate(start_task=True)

            time.sleep(0.1)
            task.acquire_base()
            time.sleep(0.1)
            data = task.acquire_base()
            assert data.shape[1] > 0

        # After exit, task handle is released
        assert task.task is None

        # Can re-create with same name (proves cleanup)
        with NITask(task_name, sample_rate=AI_SAMPLE_RATE) as task2:
            task2.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
            task2.initiate(start_task=False)


# ===========================================================================
# Task Group 5: NITaskOutput AO Generation Tests
# ===========================================================================


class TestNITaskOutputHardware:
    """Validate NITaskOutput analog output against real NI 9260."""

    def test_nitaskoutput_ao_generation(self) -> None:
        """Create AO channel, initiate, generate sine wave, clear task."""
        if AO_DEVICE_NAME is None:
            pytest.skip("No AO device available")

        from nidaqwrapper import NITaskOutput

        task = NITaskOutput("hw_ao_gen", sample_rate=AO_SAMPLE_RATE)
        try:
            task.add_channel(
                "ao0", device_ind=AO_DEVICE_INDEX, channel_ind=0,
                min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
            )
            task.initiate()

            # Generate a short sine wave (1 second, within NI 9260 range)
            t = np.linspace(0, 1, AO_SAMPLE_RATE, endpoint=False)
            signal = (2.0 * np.sin(2 * np.pi * 10 * t)).reshape(-1, 1)
            task.generate(signal)

            # Let it run briefly
            time.sleep(0.2)
        finally:
            task.clear_task()

    def test_nitaskoutput_context_manager(self) -> None:
        """NITaskOutput context manager cleans up properly."""
        if AO_DEVICE_NAME is None:
            pytest.skip("No AO device available")

        from nidaqwrapper import NITaskOutput

        task_name = "hw_ctx_ao"
        with NITaskOutput(task_name, sample_rate=AO_SAMPLE_RATE) as task:
            task.add_channel(
                "ao0", device_ind=AO_DEVICE_INDEX, channel_ind=0,
                min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
            )
            task.initiate()

            t = np.linspace(0, 1, AO_SAMPLE_RATE, endpoint=False)
            signal = (1.0 * np.sin(2 * np.pi * 10 * t)).reshape(-1, 1)
            task.generate(signal)
            time.sleep(0.1)

        assert task.task is None


# ===========================================================================
# Task Group 6: NIDAQWrapper NI MAX Task Test
# ===========================================================================


class TestWrapperNIMaxTask:
    """Validate NIDAQWrapper with a pre-existing NI MAX task."""

    def test_wrapper_ni_max_task(self) -> None:
        """Configure, connect, introspect, and disconnect with NI MAX task.

        Uses the pre-existing 'IM3' task saved in NI MAX.
        """
        if NI_MAX_TASK_NAME is None:
            pytest.skip("No NI MAX task available")

        from nidaqwrapper import NIDAQWrapper

        wrapper = NIDAQWrapper()
        try:
            wrapper.configure(task_in=NI_MAX_TASK_NAME)
            result = wrapper.connect()
            assert result is True, "connect() should return True for NI MAX task"

            # Introspect
            ch_names = wrapper.get_channel_names()
            assert len(ch_names) > 0, "Expected at least one channel from NI MAX task"

            sample_rate = wrapper.get_sample_rate()
            assert sample_rate > 0, f"Expected positive sample rate, got {sample_rate}"
        finally:
            wrapper.disconnect()


# ===========================================================================
# Task Group 7: NIDAQWrapper Programmatic Task Test
# ===========================================================================


class TestWrapperProgrammatic:
    """Validate NIDAQWrapper full lifecycle with programmatic NITask."""

    def test_wrapper_programmatic_full_lifecycle(self) -> None:
        """Configure, connect, set trigger, acquire, disconnect.

        Uses a very low trigger level so noise triggers it quickly.
        """
        from nidaqwrapper import NIDAQWrapper, NITask

        n_samples = 5000

        task = NITask("hw_wrap_prog", sample_rate=AI_SAMPLE_RATE)
        task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")

        wrapper = NIDAQWrapper()
        try:
            wrapper.configure(task_in=task)
            result = wrapper.connect()
            assert result is True, "connect() should return True"

            assert wrapper.get_channel_names() == ["ch0"]
            assert wrapper.get_sample_rate() == AI_SAMPLE_RATE

            # Set trigger with very low level so noise triggers immediately
            wrapper.set_trigger(
                n_samples=n_samples,
                trigger_channel=0,
                trigger_level=0.001,
                trigger_type="abs",
                presamples=100,
            )

            data = wrapper.acquire()

            assert isinstance(data, np.ndarray)
            assert data.shape == (n_samples, 1), (
                f"Expected shape ({n_samples}, 1), got {data.shape}"
            )
        finally:
            wrapper.disconnect()

    def test_wrapper_read_all_available(self) -> None:
        """read_all_available() returns (n_samples, n_channels) data."""
        from nidaqwrapper import NIDAQWrapper, NITask

        task = NITask("hw_wrap_raa", sample_rate=AI_SAMPLE_RATE)
        task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")

        wrapper = NIDAQWrapper()
        try:
            wrapper.configure(task_in=task)
            wrapper.connect()

            # Start the task manually
            wrapper._task_in.start()

            # Priming read
            time.sleep(0.1)
            wrapper.read_all_available()

            # Real read
            time.sleep(0.2)
            data = wrapper.read_all_available()

            assert isinstance(data, np.ndarray)
            assert data.ndim == 2
            assert data.shape[1] == 1, f"Expected 1 channel, got shape {data.shape}"
            assert data.shape[0] > 0, "Expected at least one sample"

            # Verify voltage range
            assert np.all(data >= AI_VOLTAGE_MIN)
            assert np.all(data <= AI_VOLTAGE_MAX)
        finally:
            wrapper.disconnect()


# ===========================================================================
# Task Group 8: Single-Sample Read/Write Tests
# ===========================================================================


class TestSingleSample:
    """Validate single-sample read() and write() on real hardware."""

    def test_single_sample_read(self) -> None:
        """read() returns (n_channels,) array with reasonable voltages."""
        from nidaqwrapper import NIDAQWrapper, NITask

        task = NITask("hw_ss_read", sample_rate=AI_SAMPLE_RATE)
        task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")

        wrapper = NIDAQWrapper()
        try:
            wrapper.configure(task_in=task)
            wrapper.connect()

            # Start the task for single-sample reads
            wrapper._task_in.start()

            data = wrapper.read()

            assert isinstance(data, np.ndarray)
            assert data.shape == (1,), f"Expected shape (1,), got {data.shape}"
            assert AI_VOLTAGE_MIN <= data[0] <= AI_VOLTAGE_MAX, (
                f"Voltage {data[0]:.4f} outside expected range"
            )
        finally:
            wrapper.disconnect()

    def test_single_sample_write(self) -> None:
        """write() sets output voltage on NI 9260 without error."""
        if AO_DEVICE_NAME is None:
            pytest.skip("No AO device available")

        from nidaqwrapper import NIDAQWrapper, NITaskOutput

        task_out = NITaskOutput("hw_ss_write", sample_rate=AO_SAMPLE_RATE)
        task_out.add_channel(
            "ao0", device_ind=AO_DEVICE_INDEX, channel_ind=0,
            min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
        )

        wrapper = NIDAQWrapper()
        try:
            wrapper.configure(task_out=task_out)
            wrapper.connect()

            # Write various values — should not raise
            wrapper.write(0.0)
            wrapper.write(1.0)
            wrapper.write(0.0)  # Reset to zero
        finally:
            wrapper.disconnect()


# ===========================================================================
# Task Group 9: Context Manager Cleanup Tests
# ===========================================================================


class TestWrapperContextManager:
    """Validate NIDAQWrapper context manager releases hardware resources."""

    def test_context_manager_normal_exit(self) -> None:
        """Resources released on normal with-block exit."""
        from nidaqwrapper import NIDAQWrapper, NITask

        task_name = "hw_ctx_norm"

        with NIDAQWrapper() as wrapper:
            task = NITask(task_name, sample_rate=AI_SAMPLE_RATE)
            task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
            wrapper.configure(task_in=task)
            wrapper.connect()

        # After context exit, should be disconnected
        # Verify resources released: can create a new task with the same name
        new_task = NITask(task_name, sample_rate=AI_SAMPLE_RATE)
        try:
            new_task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
            new_task.initiate(start_task=False)
        finally:
            new_task.clear_task()

    def test_context_manager_exception_cleanup(self) -> None:
        """Resources released even when exception occurs in with-block."""
        from nidaqwrapper import NIDAQWrapper, NITask

        task_name = "hw_ctx_exc"

        with pytest.raises(RuntimeError, match="deliberate"):
            with NIDAQWrapper() as wrapper:
                task = NITask(task_name, sample_rate=AI_SAMPLE_RATE)
                task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
                wrapper.configure(task_in=task)
                wrapper.connect()
                raise RuntimeError("deliberate test exception")

        # After exception, resources should still be released
        new_task = NITask(task_name, sample_rate=AI_SAMPLE_RATE)
        try:
            new_task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
            new_task.initiate(start_task=False)
        finally:
            new_task.clear_task()


# ===========================================================================
# Task Group 10: Digital I/O Hardware Tests
# ===========================================================================


class TestDigitalIO:
    """Validate standalone DigitalInput/DigitalOutput on PCIe-6320.

    Note: wrapper digital integration (read_digital/write_digital) is
    skipped — depends on Phase 6b (digital-wrapper-integration).
    """

    def test_digital_input_read(self) -> None:
        """DigitalInput reads a boolean value from Dev1 DI line."""
        if DI_LINES is None:
            pytest.skip("No digital input hardware available")

        from nidaqwrapper import DigitalInput

        di = DigitalInput("hw_di_read")
        try:
            di.add_channel("di_ch", lines=DI_LINES)
            di.initiate()
            data = di.read()

            assert isinstance(data, np.ndarray)
            assert data.size >= 1, "Expected at least one value from DI read"
            # Each value should be boolean-like (0 or 1)
            for val in data.flat:
                assert val in (True, False, 0, 1), f"Unexpected DI value: {val}"
        finally:
            di.clear_task()

    def test_digital_output_write(self) -> None:
        """DigitalOutput writes True/False to Dev1 DO line without error."""
        if DO_LINES is None:
            pytest.skip("No digital output hardware available")

        from nidaqwrapper import DigitalOutput

        do = DigitalOutput("hw_do_write")
        try:
            do.add_channel("do_ch", lines=DO_LINES)
            do.initiate()

            do.write(True)
            do.write(False)
        finally:
            do.clear_task()

    def test_digital_context_manager(self) -> None:
        """Digital I/O context managers clean up properly."""
        if DI_LINES is None:
            pytest.skip("No digital input hardware available")

        from nidaqwrapper import DigitalInput

        task_name = "hw_di_ctx"
        with DigitalInput(task_name) as di:
            di.add_channel("di_ch", lines=DI_LINES)
            di.initiate()
            data = di.read()
            assert data.size >= 1

        assert di.task is None

    def test_wrapper_digital_integration(self) -> None:
        """NIDAQWrapper.read_digital() and write_digital() work with real hardware.

        Configures a wrapper with digital input (Dev1 port0/line0) and digital
        output (Dev1 port1/line0), connects, writes True/False, reads, and
        disconnects.
        """
        if DI_LINES is None or DO_LINES is None:
            pytest.skip("Both DI and DO hardware required for wrapper digital test")

        from nidaqwrapper import DigitalInput, DigitalOutput, NIDAQWrapper

        di = DigitalInput("hw_wrap_di")
        di.add_channel("di_ch", lines=DI_LINES)

        do = DigitalOutput("hw_wrap_do")
        do.add_channel("do_ch", lines=DO_LINES)

        wrapper = NIDAQWrapper()
        wrapper.configure(task_digital_in=di, task_digital_out=do)

        try:
            result = wrapper.connect()
            assert result is True

            # Write True, then read digital input
            wrapper.write_digital(True)
            data = wrapper.read_digital()
            assert isinstance(data, np.ndarray)
            assert data.size >= 1

            # Write False
            wrapper.write_digital(False)
        finally:
            wrapper.disconnect()


# ===========================================================================
# Task Group 11: NIAdvanced Multi-Task Test
# ===========================================================================


class TestNIAdvancedHardware:
    """Validate NIAdvanced on real hardware.

    Tests both single-task software trigger and (if 2+ modules are available)
    multi-task hardware trigger modes.
    """

    def test_niadvanced_single_task_software_trigger(self) -> None:
        """NIAdvanced with a single input task and software trigger."""
        from nidaqwrapper import NIAdvanced, NITask

        n_samples = 5000

        task = NITask("hw_adv_st", sample_rate=AI_SAMPLE_RATE)
        try:
            task.add_channel("ch0", device_ind=AI_DEVICE_INDEX, channel_ind=0, units="V")
            task.initiate(start_task=False)

            adv = NIAdvanced()
            try:
                result = adv.configure(input_tasks=[task.task])
                assert result is True, "configure() should return True"

                result = adv.connect()
                assert result is True, "connect() should return True"

                # Set trigger with very low level
                adv.set_trigger(
                    n_samples=n_samples,
                    trigger_channel=0,
                    trigger_level=0.001,
                    trigger_type="abs",
                )

                data = adv.acquire()

                # Software trigger returns dict with channel names + 'time'
                assert isinstance(data, dict)
                assert "time" in data
                non_time_keys = [k for k in data if k != "time"]
                assert len(non_time_keys) >= 1

                # Verify data shape
                for key in non_time_keys:
                    assert isinstance(data[key], np.ndarray)
                    assert len(data[key]) == n_samples, (
                        f"Expected {n_samples} samples for '{key}', got {len(data[key])}"
                    )
            finally:
                adv.disconnect()
        finally:
            task.clear_task()
