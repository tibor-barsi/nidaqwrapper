"""Simulated device tests for TOML config serialization.

Tests save_config() and from_config() round-trip functionality against SimDev1.
"""

from __future__ import annotations

import pathlib

import pytest

from nidaqwrapper import AITask, AOTask


@pytest.mark.simulated
class TestAITaskConfigRoundTrip:
    """Test TOML config persistence for AITask."""

    def test_save_config_produces_valid_toml(
        self, simulated_device_name, sim_device_index, tmp_path
    ):
        """Test that save_config() produces a valid TOML file.

        Creates AITask with 2 channels on SimDev1, starts it, saves config,
        reads the file, verifies [task], [devices], [[channels]] sections exist.
        """
        ai_task = AITask("test_save_config", sample_rate=10000)
        try:
            # Add 2 channels
            ai_task.add_channel(
                "ai0",
                device_ind=sim_device_index,
                channel_ind=0,
                units="V",
                min_val=-5.0,
                max_val=5.0,
            )
            ai_task.add_channel(
                "ai1",
                device_ind=sim_device_index,
                channel_ind=1,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task.start(start_task=False)

            # Save config
            config_path = tmp_path / "test_ai_config.toml"
            ai_task.save_config(config_path)

            # Verify file exists
            assert config_path.exists(), "Config file should be created"

            # Read and verify contents
            content = config_path.read_text(encoding="utf-8")

            # Verify key sections exist
            assert "[task]" in content, "Config should contain [task] section"
            assert "[devices]" in content, "Config should contain [devices] section"
            assert "[[channels]]" in content, "Config should contain [[channels]] sections"

            # Verify task name appears
            assert 'name = "test_save_config"' in content, (
                "Config should contain task name"
            )

            # Verify sample rate appears
            assert "sample_rate = 10000" in content, (
                "Config should contain sample rate"
            )

            # Verify device name appears (should be SimDev1)
            assert simulated_device_name in content, (
                "Config should contain the device name"
            )

            # Verify channel names appear
            assert 'name = "ai0"' in content, "Config should contain ai0 channel"
            assert 'name = "ai1"' in content, "Config should contain ai1 channel"
        finally:
            ai_task.clear_task()

    def test_from_config_round_trip(self, simulated_device_name, sim_device_index, tmp_path):
        """Test from_config() round-trip: save, load, verify task recreated.

        Creates AITask, saves config, loads with from_config(), starts the
        recreated task, acquires data, verifies data shape.
        """
        # Create original task
        ai_task_orig = AITask("test_round_trip", sample_rate=10000)
        try:
            ai_task_orig.add_channel(
                "ch0",
                device_ind=sim_device_index,
                channel_ind=0,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task_orig.add_channel(
                "ch1",
                device_ind=sim_device_index,
                channel_ind=1,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task_orig.start(start_task=False)

            # Save config
            config_path = tmp_path / "round_trip.toml"
            ai_task_orig.save_config(config_path)

            # Clean up original task
            ai_task_orig.clear_task()

            # Load from config
            ai_task_loaded = AITask.from_config(config_path)

            try:
                # Verify task attributes
                assert ai_task_loaded.task_name == "test_round_trip", (
                    "Task name should match original"
                )
                assert ai_task_loaded.sample_rate == pytest.approx(10000, rel=0.01), (
                    "Sample rate should match original"
                )
                assert ai_task_loaded.number_of_ch == 2, (
                    "Channel count should match original"
                )

                # Start the loaded task
                ai_task_loaded.start(start_task=True)

                # Acquire data
                data = ai_task_loaded.acquire(n_samples=50)

                # Verify data shape
                assert data.shape[0] == 2, "Should have 2 channels"
                assert data.shape[1] == 50, "Should have 50 samples"
            finally:
                ai_task_loaded.clear_task()
        finally:
            # Ensure original task is cleaned up if it still exists
            if hasattr(ai_task_orig, "task") and ai_task_orig.task is not None:
                ai_task_orig.clear_task()


@pytest.mark.simulated
class TestAOTaskConfigRoundTrip:
    """Test TOML config persistence for AOTask."""

    def test_ao_save_config_produces_valid_toml(
        self, simulated_device_name, sim_device_index, tmp_path
    ):
        """Test that AOTask.save_config() produces a valid TOML file.

        Creates AOTask with 1 channel, saves config, verifies file structure.
        """
        ao_task = AOTask("test_ao_save", sample_rate=10000)
        try:
            ao_task.add_channel(
                "ao0",
                device_ind=sim_device_index,
                channel_ind=0,
                min_val=-5.0,
                max_val=5.0,
            )
            ao_task.start(start_task=False)

            # Save config
            config_path = tmp_path / "test_ao_config.toml"
            ao_task.save_config(config_path)

            # Verify file exists
            assert config_path.exists(), "Config file should be created"

            # Read and verify contents
            content = config_path.read_text(encoding="utf-8")

            # Verify key sections
            assert "[task]" in content, "Config should contain [task] section"
            assert "[[channels]]" in content, "Config should contain [[channels]]"
            assert 'type = "output"' in content, (
                "AOTask config should specify type = output"
            )

            # Verify min/max values were written
            assert "min_val = -5.0" in content, "Config should contain min_val"
            assert "max_val = 5.0" in content, "Config should contain max_val"
        finally:
            ao_task.clear_task()

    def test_ao_from_config_round_trip(self, simulated_device_name, sim_device_index, tmp_path):
        """Test AOTask.from_config() round-trip.

        Creates AOTask, saves, loads, verifies attributes match.
        """
        # Create original task
        ao_task_orig = AOTask("test_ao_round_trip", sample_rate=10000)
        try:
            ao_task_orig.add_channel(
                "output0",
                device_ind=sim_device_index,
                channel_ind=0,
                min_val=-10.0,
                max_val=10.0,
            )
            ao_task_orig.start(start_task=False)

            # Save config
            config_path = tmp_path / "ao_round_trip.toml"
            ao_task_orig.save_config(config_path)

            # Clean up original
            ao_task_orig.clear_task()

            # Load from config
            ao_task_loaded = AOTask.from_config(config_path)

            try:
                # Verify attributes
                assert ao_task_loaded.task_name == "test_ao_round_trip", (
                    "Task name should match"
                )
                assert ao_task_loaded.sample_rate == pytest.approx(10000, rel=0.01), (
                    "Sample rate should match"
                )
                assert ao_task_loaded.number_of_ch == 1, (
                    "Channel count should match"
                )
            finally:
                ao_task_loaded.clear_task()
        finally:
            if hasattr(ao_task_orig, "task") and ao_task_orig.task is not None:
                ao_task_orig.clear_task()
