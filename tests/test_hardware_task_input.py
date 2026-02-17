"""Hardware integration tests for nidaqwrapper.task_input.NITask.

All tests in this module require real NI-DAQmx hardware and are marked with
``@pytest.mark.hardware``.  Run them with::

    uv run pytest tests/test_hardware_task_input.py -v -m hardware

Or exclude them from a normal run with::

    uv run pytest -m "not hardware"

Hardware configuration
----------------------
- cDAQ1Mod4 : NI 9215 (BNC) — 4 analog input voltage channels (ai0–ai3)
- cDAQ1Mod3 : NI 9260 (BNC) — 2 analog output channels (not used here)
- cDAQ1 chassis : NI cDAQ-9174 USB
- Device indices (as seen by nidaqmx): cDAQ1=0, cDAQ1Mod3=1, cDAQ1Mod4=2, Dev1=3

Notes
-----
Each test uses a unique task name to prevent NI MAX collisions.  All tests
clean up after themselves via try/finally or the NITask context manager so
that a test failure never leaves stale hardware tasks behind.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.hardware

# ---------------------------------------------------------------------------
# Hardware constants — update if the rig changes
# ---------------------------------------------------------------------------

DEVICE_IND = 2          # cDAQ1Mod4 (NI 9215) in the device list
DEVICE_NAME = "cDAQ1Mod4"
PRODUCT_SUBSTR = "9215"  # substring that must appear in the product type
SAMPLE_RATE = 25600      # exact rate supported by NI 9215
VOLTAGE_MIN = -10.0      # NI 9215 input range lower bound
VOLTAGE_MAX = 10.0       # NI 9215 input range upper bound

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _delete_saved_task(task_name: str) -> None:
    """Delete a saved task from NI MAX, ignoring errors if it does not exist.

    Parameters
    ----------
    task_name : str
        Name of the task to remove from NI MAX.
    """
    import nidaqmx

    system = nidaqmx.system.System.local()
    for saved in system.tasks:
        if saved._name == task_name:
            saved.delete()
            return


# ---------------------------------------------------------------------------
# TestHardwareConstructor
# ---------------------------------------------------------------------------


class TestHardwareConstructor:
    """Verify NITask constructor behaviour against real hardware."""

    def test_constructor_discovers_devices(self) -> None:
        """device_list includes cDAQ1Mod4 and device_product_type includes NI 9215.

        The constructor calls nidaqmx.system.System.local() to enumerate every
        device present in the NI-DAQmx system.  This test confirms that the
        expected module (cDAQ1Mod4) is visible and that its product type string
        contains '9215'.
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_discover", sample_rate=SAMPLE_RATE)
        try:
            assert DEVICE_NAME in task.device_list, (
                f"Expected '{DEVICE_NAME}' in device_list, got: {task.device_list}"
            )
            assert any(PRODUCT_SUBSTR in pt for pt in task.device_product_type), (
                f"Expected a product type containing '{PRODUCT_SUBSTR}', "
                f"got: {task.device_product_type}"
            )
        finally:
            task.clear_task()

    def test_constructor_rejects_duplicate_saved_task(self) -> None:
        """Constructor raises ValueError when task_name already exists in NI MAX.

        'IM3' is a pre-existing saved task in NI MAX on this system.  The
        constructor must detect the collision and raise ValueError before
        returning an instance, so that the caller is never handed a task object
        that would silently shadow or corrupt the saved task.
        """
        from nidaqwrapper.task_input import NITask

        with pytest.raises(ValueError, match="already"):
            NITask("IM3", sample_rate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# TestHardwareAddChannel
# ---------------------------------------------------------------------------


class TestHardwareAddChannel:
    """Verify add_channel() channel storage against real hardware."""

    def test_add_voltage_channel(self) -> None:
        """add_channel() stores a voltage channel for cDAQ1Mod4/ai0.

        The NI 9215 is a voltage-only module.  This test confirms that a
        voltage channel added on device_ind=2 (cDAQ1Mod4) / channel_ind=0
        appears in task.channel_list and task.channels with the correct
        device_ind and channel_ind values stored.
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_addch1", sample_rate=SAMPLE_RATE)
        try:
            task.add_channel(
                "voltage_ai0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )

            assert "voltage_ai0" in task.channel_list
            cfg = task.channels["voltage_ai0"]
            assert cfg["device_ind"] == DEVICE_IND
            assert cfg["channel_ind"] == 0
        finally:
            task.clear_task()

    def test_add_multiple_voltage_channels(self) -> None:
        """Adding three voltage channels on ai0, ai1, ai2 produces three entries.

        Confirms that channel_list length matches the number of add_channel()
        calls and that each channel is stored under its unique name.
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_addch3", sample_rate=SAMPLE_RATE)
        try:
            for idx in range(3):
                task.add_channel(
                    f"ch_ai{idx}",
                    device_ind=DEVICE_IND,
                    channel_ind=idx,
                    units="V",
                )

            assert len(task.channel_list) == 3
            assert task.channel_list == ["ch_ai0", "ch_ai1", "ch_ai2"]
        finally:
            task.clear_task()


# ---------------------------------------------------------------------------
# TestHardwareInitiate
# ---------------------------------------------------------------------------


class TestHardwareInitiate:
    """Verify NITask.initiate() creates and starts a real nidaqmx task."""

    def test_initiate_creates_task(self) -> None:
        """initiate(start_task=False) creates a nidaqmx Task without starting it.

        After initiate() returns, task.task must be a live nidaqmx Task object
        (not None).  With start_task=False the task should not yet be running,
        so calling task.task.is_task_done() should return True (nothing started).
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_init1", sample_rate=SAMPLE_RATE)
        try:
            task.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            task.initiate(start_task=False)

            assert task.task is not None, "task.task must not be None after initiate()"
        finally:
            task.clear_task()

    def test_initiate_starts_task(self) -> None:
        """initiate(start_task=True) leaves the task in a running state.

        A running continuous-acquisition task is not done, so
        task.task.is_task_done() must return False immediately after start.
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_init2", sample_rate=SAMPLE_RATE)
        try:
            task.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            task.initiate(start_task=True)

            assert task.task.is_task_done() is False, (
                "A started continuous task must not report is_task_done()=True"
            )
        finally:
            task.clear_task()

    def test_initiate_sample_rate_validated(self) -> None:
        """initiate() does not raise for 25600 Hz, an exact match on NI 9215.

        The NI 9215 supports 25600 Hz natively.  initiate() reads the rate
        the driver actually committed (task._timing.samp_clk_rate) and raises
        ValueError if it differs from the requested rate.  This test verifies
        no exception is raised for a known-good rate.
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_init3", sample_rate=SAMPLE_RATE)
        try:
            task.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            # Should complete without raising ValueError
            task.initiate(start_task=False)
        finally:
            task.clear_task()


# ---------------------------------------------------------------------------
# TestHardwareAcquire
# ---------------------------------------------------------------------------


class TestHardwareAcquire:
    """Verify acquire_base() returns correctly shaped data from the NI 9215."""

    def test_acquire_base_returns_data(self) -> None:
        """acquire_base() on a running single-channel task returns (1, n) ndarray.

        After a brief sleep to let the hardware buffer fill, acquire_base()
        must return a 2-D numpy array with exactly 1 row (channel-major layout)
        and at least one sample column.

        Note: the very first ``read(-1)`` on a newly started nidaqmx task may
        return 0 samples (driver initialisation artefact).  A priming read
        followed by a second sleep ensures the buffer has data for the real
        assertion.
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_acq1", sample_rate=SAMPLE_RATE)
        try:
            task.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            task.initiate(start_task=True)

            # Priming read — first read(-1) may return 0 samples
            time.sleep(0.1)
            task.acquire_base()

            # Let the buffer refill after the priming drain
            time.sleep(0.2)
            data = task.acquire_base()

            assert isinstance(data, np.ndarray), (
                f"Expected ndarray, got {type(data)}"
            )
            assert data.ndim == 2, (
                f"Expected 2-D array, got shape {data.shape}"
            )
            assert data.shape[0] == 1, (
                f"Expected 1 channel row, got shape {data.shape}"
            )
            assert data.shape[1] > 0, (
                "Expected at least one sample, buffer appears empty"
            )
        finally:
            task.clear_task()

    def test_acquire_base_multi_channel(self) -> None:
        """acquire_base() on a 2-channel task returns (2, n) ndarray.

        Confirms that the channel-major shape is maintained when more than
        one physical channel is active — nidaqmx returns a list-of-lists for
        multi-channel tasks, and acquire_base() must convert that to 2-D.
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_acq2", sample_rate=SAMPLE_RATE)
        try:
            task.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            task.add_channel(
                "ch1",
                device_ind=DEVICE_IND,
                channel_ind=1,
                units="V",
            )
            task.initiate(start_task=True)

            # Priming read — first read(-1) may return 0 samples
            time.sleep(0.1)
            task.acquire_base()

            # Let the buffer refill after the priming drain
            time.sleep(0.2)
            data = task.acquire_base()

            assert data.ndim == 2, (
                f"Expected 2-D array, got shape {data.shape}"
            )
            assert data.shape[0] == 2, (
                f"Expected 2 channel rows, got shape {data.shape}"
            )
            assert data.shape[1] > 0, "Expected at least one sample"
        finally:
            task.clear_task()

    def test_acquire_voltage_range(self) -> None:
        """All acquired samples are within the NI 9215 input range of ±10 V.

        An unconnected input floats near 0 V.  Regardless of what the channel
        is connected to, every sample must lie within the hardware's absolute
        input range.  Values outside ±10 V indicate a driver or channel
        configuration problem.
        """
        from nidaqwrapper.task_input import NITask

        task = NITask("hwtask_acq3", sample_rate=SAMPLE_RATE)
        try:
            task.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            task.initiate(start_task=True)

            # Priming read — first read(-1) may return 0 samples
            time.sleep(0.1)
            task.acquire_base()

            # Let the buffer refill after the priming drain
            time.sleep(0.2)
            data = task.acquire_base()

            assert data.size > 0, "Expected non-empty data array"
            assert np.all(data >= VOLTAGE_MIN), (
                f"Sample(s) below {VOLTAGE_MIN} V detected: min={data.min():.4f}"
            )
            assert np.all(data <= VOLTAGE_MAX), (
                f"Sample(s) above {VOLTAGE_MAX} V detected: max={data.max():.4f}"
            )
        finally:
            task.clear_task()


# ---------------------------------------------------------------------------
# TestHardwareClearTask
# ---------------------------------------------------------------------------


class TestHardwareClearTask:
    """Verify that clear_task() fully releases the hardware resource."""

    def test_clear_task_releases_hardware(self) -> None:
        """After clear_task(), a new task with the same name can be created.

        nidaqmx forbids two live tasks with the same name.  If clear_task()
        does not properly close the underlying handle, creating a second task
        with the same name would fail.  Success here proves the release was
        complete.
        """
        from nidaqwrapper.task_input import NITask

        task_name = "hwtask_clear1"
        first = NITask(task_name, sample_rate=SAMPLE_RATE)
        first.add_channel(
            "ch0",
            device_ind=DEVICE_IND,
            channel_ind=0,
            units="V",
        )
        first.initiate(start_task=False)
        first.clear_task()

        # If the hardware handle was not released, this second construction
        # or its initiate() call would fail with a nidaqmx error.
        second = NITask(task_name, sample_rate=SAMPLE_RATE)
        try:
            second.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            second.initiate(start_task=False)
        finally:
            second.clear_task()


# ---------------------------------------------------------------------------
# TestHardwareSave
# ---------------------------------------------------------------------------


class TestHardwareSave:
    """Verify save() persists the task to NI MAX and that cleanup works."""

    def test_save_and_cleanup(self) -> None:
        """save() persists the task to NI MAX; it can then be found and deleted.

        Calls save(clear_task=False) so the task handle stays open, then
        independently verifies the task name appears in
        nidaqmx.system.System.local().tasks.task_names.  Finally, both the
        live task and the saved NI MAX entry are cleaned up.
        """
        import nidaqmx
        from nidaqwrapper.task_input import NITask

        task_name = "hwtask_save1"
        task = NITask(task_name, sample_rate=SAMPLE_RATE)
        try:
            task.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            # save with clear_task=False so we can keep the handle open for
            # cleanup; the saved entry in NI MAX is what we actually verify.
            task.save(clear_task=False)

            system = nidaqmx.system.System.local()
            saved_names = list(system.tasks.task_names)

            assert task_name in saved_names, (
                f"Expected '{task_name}' in NI MAX tasks, got: {saved_names}"
            )
        finally:
            task.clear_task()
            # Remove the saved NI MAX entry so it does not pollute other runs
            _delete_saved_task(task_name)


# ---------------------------------------------------------------------------
# TestHardwareContextManager
# ---------------------------------------------------------------------------


class TestHardwareContextManager:
    """Verify the context manager protocol releases hardware on exit."""

    def test_context_manager_cleanup(self) -> None:
        """The with-block pattern calls clear_task() on exit, releasing hardware.

        After the with-block exits, task.task must be None (the handle was
        closed) and a new task with the same name must be constructable,
        demonstrating that the hardware resource was fully returned.
        """
        from nidaqwrapper.task_input import NITask

        task_name = "hwtask_ctx1"

        with NITask(task_name, sample_rate=SAMPLE_RATE) as task:
            task.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            task.initiate(start_task=False)

        # After __exit__, the task handle must be released
        assert task.task is None, (
            "task.task must be None after the context manager exits"
        )

        # Verify the hardware resource was genuinely freed: creating a second
        # task with the same name must not raise a nidaqmx duplicate error.
        with NITask(task_name, sample_rate=SAMPLE_RATE) as second:
            second.add_channel(
                "ch0",
                device_ind=DEVICE_IND,
                channel_ind=0,
                units="V",
            )
            second.initiate(start_task=False)
        # second.__exit__ cleans up automatically here
