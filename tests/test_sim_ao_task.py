"""Simulated AOTask lifecycle tests for nidaqwrapper.

All tests in this module run against NI-DAQmx simulated devices (not mocks).
They are marked with ``@pytest.mark.simulated`` and use the simulated device
fixtures from conftest.py.

Run with::

    uv run pytest tests/test_sim_ao_task.py -v -m simulated

Device Configuration
---------------------
- SimDev1 : PCIe-6361 (simulated) — 16 AI, 2 AO, 24 DI lines, 24 DO lines
"""

from __future__ import annotations

import nidaqmx
import numpy as np
import pytest

pytestmark = pytest.mark.simulated


@pytest.fixture
def sim_device_index(simulated_device_name):
    """Find the device index for SimDev1 in the system device list.

    Returns
    -------
    int
        Index of SimDev1 in nidaqmx.system.System.local().devices.
    """
    import nidaqmx.system

    system = nidaqmx.system.System.local()
    device_names = [d.name for d in system.devices]

    if simulated_device_name not in device_names:
        pytest.skip(f"Simulated device {simulated_device_name} not found")

    return device_names.index(simulated_device_name)


class TestAOTaskSimulated:
    """Full AOTask lifecycle tests with simulated device."""

    def test_constructor_add_channel_and_generate_single_channel(
        self, sim_device_index
    ):
        """Create AOTask, add 1 AO channel, call start/generate, verify no error.

        Tests single-channel generate with shape (n_samples,).
        """
        from nidaqwrapper import AOTask

        task = AOTask("test_ao_single", sample_rate=10000)
        try:
            # Add 1 AO voltage channel
            task.add_channel(
                "ao0", device_ind=sim_device_index, channel_ind=0,
                min_val=-10.0, max_val=10.0
            )

            # Start task
            task.start(start_task=False)

            # Generate single-channel data (1-D array)
            signal = np.sin(np.linspace(0, 2 * np.pi, 100))
            task.generate(signal)

            # No error means success
        finally:
            task.clear_task()

    def test_multi_channel_generate(self, sim_device_index):
        """Add 2 AO channels, generate with shape (n_samples, n_channels), verify no error.

        Tests the np.ascontiguousarray transposition path.
        """
        from nidaqwrapper import AOTask

        task = AOTask("test_ao_multi", sample_rate=10000)
        try:
            # Add 2 AO voltage channels (ao0, ao1)
            task.add_channel(
                "ao0", device_ind=sim_device_index, channel_ind=0,
                min_val=-10.0, max_val=10.0
            )
            task.add_channel(
                "ao1", device_ind=sim_device_index, channel_ind=1,
                min_val=-10.0, max_val=10.0
            )

            # Start task
            task.start(start_task=False)

            # Generate multi-channel data (2-D array, public format)
            t = np.linspace(0, 2 * np.pi, 100)
            signal = np.column_stack([
                np.sin(t),
                np.cos(t)
            ])
            assert signal.shape == (100, 2)

            task.generate(signal)

            # No error means success
        finally:
            task.clear_task()

    def test_from_task_wrapping(self, simulated_device_name):
        """Create raw nidaqmx.Task with AO channel, wrap via AOTask.from_task(), generate."""
        from nidaqwrapper import AOTask

        # Create raw nidaqmx task
        raw_task = nidaqmx.Task("test_ao_raw")
        try:
            # Configure raw task with AO channel
            raw_task.ao_channels.add_ao_voltage_chan(
                f"{simulated_device_name}/ao0",
                min_val=-10.0,
                max_val=10.0
            )
            raw_task.timing.cfg_samp_clk_timing(
                rate=10000,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            )

            # Wrap with AOTask.from_task()
            wrapped = AOTask.from_task(raw_task)

            # Verify ownership flag
            assert wrapped._owns_task is False

            # Verify generate works
            signal = np.sin(np.linspace(0, 2 * np.pi, 100))
            wrapped.generate(signal)

            # Clear wrapper — should NOT close the raw task
            wrapped.clear_task()
            assert wrapped.task is None

            # Verify raw task is still usable
            # Write a single sample to verify task is alive
            raw_task.write(0.0, auto_start=True)
        finally:
            # Caller must close raw task
            try:
                raw_task.close()
            except Exception:
                pass

    def test_context_manager_cleanup(self, sim_device_index):
        """Use AOTask in with block, verify cleanup."""
        from nidaqwrapper import AOTask

        task_name = "test_ao_ctx"

        # First use in context manager
        with AOTask(task_name, sample_rate=10000) as task:
            task.add_channel(
                "ao0", device_ind=sim_device_index, channel_ind=0,
                min_val=-10.0, max_val=10.0
            )
            task.start(start_task=False)

            signal = np.sin(np.linspace(0, 2 * np.pi, 100))
            task.generate(signal)

        # After exit, task should be None
        assert task.task is None

        # Verify resources released: can create new task with same name
        with AOTask(task_name, sample_rate=10000) as task2:
            task2.add_channel(
                "ao0", device_ind=sim_device_index, channel_ind=0,
                min_val=-10.0, max_val=10.0
            )
            task2.start(start_task=False)

    def test_save_config_and_from_config_round_trip(
        self, sim_device_index, tmp_path
    ):
        """Create AOTask, save_config to TOML, from_config to recreate, verify works."""
        from nidaqwrapper import AOTask

        config_path = tmp_path / "test_ao_config.toml"

        # Create and configure task
        task1 = AOTask("test_ao_save", sample_rate=10000)
        try:
            task1.add_channel(
                "ao0", device_ind=sim_device_index, channel_ind=0,
                min_val=-10.0, max_val=10.0
            )
            task1.add_channel(
                "ao1", device_ind=sim_device_index, channel_ind=1,
                min_val=-5.0, max_val=5.0
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
        task2 = AOTask.from_config(config_path)
        try:
            # Verify channels were restored
            assert task2.number_of_ch == 2
            assert task2.sample_rate == 10000
            assert len(task2.channel_list) == 2

            # Verify task works
            task2.start(start_task=False)
            t = np.linspace(0, 2 * np.pi, 100)
            signal = np.column_stack([np.sin(t), np.cos(t)])
            task2.generate(signal)
        finally:
            task2.clear_task()
