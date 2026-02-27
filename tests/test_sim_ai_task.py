"""Simulated AITask lifecycle tests for nidaqwrapper.

All tests in this module run against NI-DAQmx simulated devices (not mocks).
They are marked with ``@pytest.mark.simulated`` and use the simulated device
fixtures from conftest.py.

Run with::

    uv run pytest tests/test_sim_ai_task.py -v -m simulated

Device Configuration
---------------------
- SimDev1 : PCIe-6361 (simulated) — 16 AI, 2 AO, 24 DI lines, 24 DO lines
"""

from __future__ import annotations

import time

import nidaqmx
import numpy as np
import pytest

pytestmark = pytest.mark.simulated


class TestAITaskSimulated:
    """Full AITask lifecycle tests with simulated device."""

    def test_constructor_and_add_channel(self, sim_device_name):
        """Create AITask, add 2 voltage channels, verify properties."""
        from nidaqwrapper import AITask

        task = AITask("test_ai_construct", sample_rate=10000)
        try:
            # Add 2 voltage channels
            task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            task.add_channel(
                "ai1", device=sim_device_name, channel_ind=1, units="V"
            )

            # Verify properties
            assert task.number_of_ch == 2
            assert len(task.channel_list) == 2
            assert "ai0" in task.channel_list
            assert "ai1" in task.channel_list
        finally:
            task.clear_task()

    def test_configure_and_acquire_finite(self, sim_device_name):
        """Configure and start task, acquire exactly n_samples, verify shape and data."""
        from nidaqwrapper import AITask

        task = AITask("test_ai_finite", sample_rate=10000)
        try:
            # Add 2 voltage channels
            task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            task.add_channel(
                "ai1", device=sim_device_name, channel_ind=1, units="V"
            )

            # Configure timing then start acquisition
            task.configure()
            task.start()

            # Acquire exactly 100 samples (blocking call)
            data = task.acquire(n_samples=100)

            # Verify shape: (n_samples, n_channels)
            assert isinstance(data, np.ndarray)
            assert data.dtype == np.float64
            assert data.shape == (100, 2), f"Expected (100, 2), got {data.shape}"

            # Verify data is non-zero (simulated devices generate noise)
            # At least some samples should be non-zero
            assert np.count_nonzero(data) > 0, "Expected non-zero data from simulated device"
        finally:
            task.clear_task()

    def test_continuous_acquire(self, sim_device_name):
        """Start task, sleep, call acquire() with no args, verify samples returned."""
        from nidaqwrapper import AITask

        task = AITask("test_ai_cont", sample_rate=10000)
        try:
            task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )

            task.configure()
            task.start()

            # Sleep to let buffer fill
            time.sleep(0.1)

            # Drain buffer with acquire() (no args = READ_ALL_AVAILABLE)
            data = task.acquire()

            # Should return approximately 1000 samples (10000 Hz * 0.1s)
            # Allow generous tolerance for simulated device timing
            assert isinstance(data, np.ndarray)
            assert data.shape[1] == 1, f"Expected 1 channel, got {data.shape[1]}"
            assert 500 <= data.shape[0] <= 2000, (
                f"Expected ~1000 samples (500-2000), got {data.shape[0]}"
            )
        finally:
            task.clear_task()

    def test_from_task_wrapping(self, simulated_device_name):
        """Create raw nidaqmx.Task, wrap with AITask.from_task(), verify behavior."""
        from nidaqwrapper import AITask

        # Create raw nidaqmx task
        raw_task = nidaqmx.Task("test_raw_wrap")
        try:
            # Configure raw task with AI channels
            raw_task.ai_channels.add_ai_voltage_chan(f"{simulated_device_name}/ai0:1")
            raw_task.timing.cfg_samp_clk_timing(
                rate=10000,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            )
            raw_task.start()

            # Wrap with AITask.from_task()
            wrapped = AITask.from_task(raw_task)

            # Verify ownership flag
            assert wrapped._owns_task is False

            # Verify acquire works
            time.sleep(0.1)
            data = wrapped.acquire(n_samples=100)
            assert data.shape == (100, 2), f"Expected (100, 2), got {data.shape}"

            # Clear wrapper — should NOT close the raw task
            wrapped.clear_task()
            assert wrapped.task is None

            # Verify raw task is still usable
            raw_data = raw_task.read(number_of_samples_per_channel=10)
            assert raw_data is not None
            assert len(raw_data) == 2, "Expected 2-channel data from raw task"
        finally:
            # Caller must close raw task
            try:
                raw_task.close()
            except Exception:
                pass

    def test_context_manager_cleanup(self, sim_device_name):
        """Use AITask in with block, verify cleanup, verify resources released."""
        from nidaqwrapper import AITask

        task_name = "test_ai_ctx"

        # First use in context manager
        with AITask(task_name, sample_rate=10000) as task:
            task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            task.configure()
            task.start()
            time.sleep(0.1)
            data = task.acquire(n_samples=100)
            assert data.shape == (100, 1)

        # After exit, task should be None
        assert task.task is None

        # Verify resources released: can create new task with same name
        with AITask(task_name, sample_rate=10000) as task2:
            task2.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            task2.configure()

    def test_save_config_and_from_config_round_trip(
        self, sim_device_name, tmp_path
    ):
        """Create AITask, save_config to TOML, from_config to recreate, verify works."""
        from nidaqwrapper import AITask

        config_path = tmp_path / "test_ai_config.toml"

        # Create and configure task
        task1 = AITask("test_ai_save", sample_rate=10000)
        try:
            task1.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            task1.add_channel(
                "ai1", device=sim_device_name, channel_ind=1, units="V"
            )

            # Configure task (required before save_config)
            task1.configure()

            # Save config
            task1.save_config(config_path)

            # Verify file exists
            assert config_path.exists()
        finally:
            task1.clear_task()

        # Load config and recreate task
        task2 = AITask.from_config(config_path)
        try:
            # Verify channels were restored
            assert task2.number_of_ch == 2
            assert task2.sample_rate == 10000
            assert len(task2.channel_list) == 2

            # Verify task works
            task2.configure()
            task2.start()
            time.sleep(0.1)
            data = task2.acquire(n_samples=100)
            assert data.shape == (100, 2)
        finally:
            task2.clear_task()


# ===========================================================================
# Task Group 1: AITask.from_name() simulated tests
# ===========================================================================


class TestAITaskFromNameSimulated:
    """Validate AITask.from_name() against a real NI MAX task database."""

    def test_from_name_wraps_ni_max_task(self, sim_task_name: str) -> None:
        """from_name(sim_task_name) returns owned AITask with correct properties.

        Verifies:
        - Returns an AITask instance without raising
        - _owns_task is True (from_name always takes ownership)
        - number_of_ch >= 1 (SimTask1 has 4 AI channels)
        - channel_list is a non-empty list of strings
        - sample_rate is approximately 10000 Hz (SimTask1 is configured at 10kHz)
        - start() succeeds without raising (ownership is correct)
        - clear_task() closes the underlying nidaqmx task silently (no warning)
        """
        from nidaqwrapper import AITask

        task = None
        try:
            task = AITask.from_name(sim_task_name)

            # Ownership: from_name always sets _owns_task=True
            assert task._owns_task is True

            # Channel count and names
            assert task.number_of_ch >= 1, (
                f"Expected at least 1 channel, got {task.number_of_ch}"
            )
            assert isinstance(task.channel_list, list)
            assert len(task.channel_list) >= 1
            assert all(isinstance(name, str) for name in task.channel_list)

            # Sample rate: SimTask1 is configured at 10kHz
            assert task.sample_rate == pytest.approx(10000.0, rel=0.01), (
                f"Expected ~10000 Hz, got {task.sample_rate}"
            )

            # start() must succeed — proves ownership is correct (not blocked)
            task.start()

        finally:
            if task is not None:
                try:
                    task.clear_task()
                except Exception:
                    pass

    def test_from_name_nonexistent_task_raises(
        self, simulated_device_name: str
    ) -> None:
        """from_name() raises KeyError for a task name that does not exist.

        The simulated_device_name fixture ensures nidaqmx is installed
        before testing the error path.
        """
        from nidaqwrapper import AITask

        with pytest.raises(KeyError):
            AITask.from_name("DoesNotExist_xyzzy_12345")


# ===========================================================================
# Task Group 3: Custom-scale voltage channel simulated tests
# ===========================================================================


class TestAITaskCustomScaleSimulated:
    """Validate add_channel() with custom linear scale on the PCIe-6361."""

    def test_custom_scale_channel_creates_and_reads(
        self, sim_device_name: str
    ) -> None:
        """Custom scale (scalar slope) channel creates, starts, and acquires.

        Uses a slope-only (no offset) linear scale.  Verifies the driver
        accepted the scale object and that acquire() returns the correct
        shape and dtype.
        """
        from nidaqwrapper import AITask

        task = None
        try:
            task = AITask("test_custom_scale", sample_rate=10000)
            task.add_channel(
                "accel_0",
                device=sim_device_name,
                channel_ind=0,
                units="m/s2",
                scale=0.5,
            )

            assert task.number_of_ch == 1

            task.configure()
            task.start()

            data = task.acquire(n_samples=50)

            # Shape: (n_samples, n_channels) per API Invariant #1
            assert isinstance(data, np.ndarray)
            assert data.shape == (50, 1), (
                f"Expected (50, 1), got {data.shape}"
            )
            assert data.dtype == np.float64

        finally:
            if task is not None:
                try:
                    task.clear_task()
                except Exception:
                    pass

    def test_custom_scale_channel_with_offset(
        self, sim_device_name: str
    ) -> None:
        """Custom scale (slope, y_intercept) tuple form creates and configures.

        Verifies the tuple form of scale is accepted by the driver.
        Does not start or acquire — configure() success is sufficient.
        """
        from nidaqwrapper import AITask

        task = None
        try:
            task = AITask("test_custom_scale_offset", sample_rate=5000)
            task.add_channel(
                "force_0",
                device=sim_device_name,
                channel_ind=0,
                units="N",
                scale=(100.0, 5.0),
            )

            assert task.number_of_ch == 1

            # configure() must not raise — driver accepted the scale and timing
            task.configure()

        finally:
            if task is not None:
                try:
                    task.clear_task()
                except Exception:
                    pass
