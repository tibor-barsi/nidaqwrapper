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

    def test_constructor_and_add_channel(self, sim_device_index):
        """Create AITask, add 2 voltage channels, verify properties."""
        from nidaqwrapper import AITask

        task = AITask("test_ai_construct", sample_rate=10000)
        try:
            # Add 2 voltage channels
            task.add_channel(
                "ai0", device_ind=sim_device_index, channel_ind=0, units="V"
            )
            task.add_channel(
                "ai1", device_ind=sim_device_index, channel_ind=1, units="V"
            )

            # Verify properties
            assert task.number_of_ch == 2
            assert len(task.channel_list) == 2
            assert "ai0" in task.channel_list
            assert "ai1" in task.channel_list
        finally:
            task.clear_task()

    def test_start_and_acquire_finite(self, sim_device_index):
        """Start task, acquire exactly n_samples, verify shape and data."""
        from nidaqwrapper import AITask

        task = AITask("test_ai_finite", sample_rate=10000)
        try:
            # Add 2 voltage channels
            task.add_channel(
                "ai0", device_ind=sim_device_index, channel_ind=0, units="V"
            )
            task.add_channel(
                "ai1", device_ind=sim_device_index, channel_ind=1, units="V"
            )

            # Start task with start_task=True
            task.start(start_task=True)

            # Acquire exactly 100 samples (blocking call)
            data = task.acquire(n_samples=100)

            # Verify shape: (n_channels, n_samples)
            assert isinstance(data, np.ndarray)
            assert data.dtype == np.float64
            assert data.shape == (2, 100), f"Expected (2, 100), got {data.shape}"

            # Verify data is non-zero (simulated devices generate noise)
            # At least some samples should be non-zero
            assert np.count_nonzero(data) > 0, "Expected non-zero data from simulated device"
        finally:
            task.clear_task()

    def test_continuous_acquire(self, sim_device_index):
        """Start task, sleep, call acquire() with no args, verify samples returned."""
        from nidaqwrapper import AITask

        task = AITask("test_ai_cont", sample_rate=10000)
        try:
            task.add_channel(
                "ai0", device_ind=sim_device_index, channel_ind=0, units="V"
            )

            task.start(start_task=True)

            # Sleep to let buffer fill
            time.sleep(0.1)

            # Drain buffer with acquire() (no args = READ_ALL_AVAILABLE)
            data = task.acquire()

            # Should return approximately 1000 samples (10000 Hz * 0.1s)
            # Allow generous tolerance for simulated device timing
            assert isinstance(data, np.ndarray)
            assert data.shape[0] == 1, f"Expected 1 channel, got {data.shape[0]}"
            assert 500 <= data.shape[1] <= 2000, (
                f"Expected ~1000 samples (500-2000), got {data.shape[1]}"
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
            assert data.shape == (2, 100), f"Expected (2, 100), got {data.shape}"

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

    def test_context_manager_cleanup(self, sim_device_index):
        """Use AITask in with block, verify cleanup, verify resources released."""
        from nidaqwrapper import AITask

        task_name = "test_ai_ctx"

        # First use in context manager
        with AITask(task_name, sample_rate=10000) as task:
            task.add_channel(
                "ai0", device_ind=sim_device_index, channel_ind=0, units="V"
            )
            task.start(start_task=True)
            time.sleep(0.1)
            data = task.acquire(n_samples=100)
            assert data.shape == (1, 100)

        # After exit, task should be None
        assert task.task is None

        # Verify resources released: can create new task with same name
        with AITask(task_name, sample_rate=10000) as task2:
            task2.add_channel(
                "ai0", device_ind=sim_device_index, channel_ind=0, units="V"
            )
            task2.start(start_task=False)

    def test_save_config_and_from_config_round_trip(
        self, sim_device_index, tmp_path
    ):
        """Create AITask, save_config to TOML, from_config to recreate, verify works."""
        from nidaqwrapper import AITask

        config_path = tmp_path / "test_ai_config.toml"

        # Create and configure task
        task1 = AITask("test_ai_save", sample_rate=10000)
        try:
            task1.add_channel(
                "ai0", device_ind=sim_device_index, channel_ind=0, units="V"
            )
            task1.add_channel(
                "ai1", device_ind=sim_device_index, channel_ind=1, units="V"
            )

            # Start task (required before save_config)
            task1.start(start_task=False)

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
            task2.start(start_task=True)
            time.sleep(0.1)
            data = task2.acquire(n_samples=100)
            assert data.shape == (2, 100)
        finally:
            task2.clear_task()
