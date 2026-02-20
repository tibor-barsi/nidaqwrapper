"""Simulated device tests for MultiHandler.

Tests multi-task synchronization and trigger configuration against SimDev1.
All tests use the @pytest.mark.simulated marker.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from nidaqwrapper import AITask, MultiHandler


@pytest.mark.simulated
class TestMultiHandlerBasics:
    """Basic MultiHandler configuration and software trigger tests."""

    def test_single_task_software_trigger(self, simulated_device_name):
        """Test software trigger acquisition with a single AI task.

        Creates one AITask with 2 channels, starts it, configures MultiHandler,
        verifies trigger_type is 'software', sets a low-level trigger, and
        acquires data.
        """
        # Create AITask with 2 channels
        ai_task = AITask("test_multi_single", sample_rate=10000)
        try:
            ai_task.add_channel(
                "ai0",
                device_ind=0,
                channel_ind=0,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task.add_channel(
                "ai1",
                device_ind=0,
                channel_ind=1,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task.start(start_task=False)

            # Configure MultiHandler with the underlying nidaqmx task
            handler = MultiHandler()
            result = handler.configure(input_tasks=[ai_task.task])

            assert result is True, "configure() should return True"
            assert handler.trigger_type == "software", (
                "trigger_type should be 'software' when no hardware trigger is configured"
            )

            # Set trigger with low level (0.1V)
            handler.set_trigger(
                n_samples=100,
                trigger_channel=0,
                trigger_level=0.1,
            )

            # Acquire data
            data = handler.acquire()

            # Verify data structure
            assert isinstance(data, dict), "acquire() should return dict in software mode"
            assert "time" in data, "Result should contain 'time' key"

            # Data should have channels from the task
            channel_names = ai_task.channel_list
            for ch_name in channel_names:
                assert ch_name in data, f"Channel {ch_name} should be in result"
                assert isinstance(data[ch_name], np.ndarray), (
                    f"Channel {ch_name} data should be numpy array"
                )
        finally:
            ai_task.clear_task()
            handler.disconnect()

    def test_multi_task_validation(self, simulated_device_name):
        """Test validation when configuring multiple AI tasks.

        Creates two AITasks on different channel sets, both at 10kHz, starts
        both, configures MultiHandler with both, verifies validation runs
        (sample rate check, timing check).
        """
        # Create first AITask with channels ai0:1
        ai_task1 = AITask("test_multi_task1", sample_rate=10000)
        ai_task2 = AITask("test_multi_task2", sample_rate=10000)

        try:
            # Configure first task: ai0, ai1
            ai_task1.add_channel(
                "ai0",
                device_ind=0,
                channel_ind=0,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task1.add_channel(
                "ai1",
                device_ind=0,
                channel_ind=1,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task1.start(start_task=False)

            # Configure second task: ai2, ai3
            ai_task2.add_channel(
                "ai2",
                device_ind=0,
                channel_ind=2,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task2.add_channel(
                "ai3",
                device_ind=0,
                channel_ind=3,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task2.start(start_task=False)

            # Configure MultiHandler with both tasks
            handler = MultiHandler()
            result = handler.configure(
                input_tasks=[ai_task1.task, ai_task2.task]
            )

            # Validation should pass — both tasks have same sample rate and timing
            assert result is True, (
                "configure() should return True when tasks have matching sample rates"
            )

            # Both tasks should be stored
            assert len(handler.input_tasks) == 2, (
                "MultiHandler should store both input tasks"
            )

            # Verify sample rate was cached from first task
            assert handler.input_sample_rate == 10000, (
                "input_sample_rate should match the configured rate"
            )
        finally:
            ai_task1.clear_task()
            ai_task2.clear_task()
            handler.disconnect()

    def test_trigger_type_detection_no_hardware(self, simulated_device_name):
        """Test that trigger_type is set to 'software' when no hardware triggers exist.

        Creates a task without hardware triggers, configures MultiHandler,
        verifies trigger_type == 'software'.
        """
        ai_task = AITask("test_trigger_detect", sample_rate=10000)
        try:
            ai_task.add_channel(
                "ai0",
                device_ind=0,
                channel_ind=0,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task.start(start_task=False)

            # Configure MultiHandler
            handler = MultiHandler()
            result = handler.configure(input_tasks=[ai_task.task])

            assert result is True, "configure() should succeed"
            assert handler.trigger_type == "software", (
                "trigger_type should default to 'software' when no hardware "
                "trigger is configured (bug fix #2)"
            )
        finally:
            handler.disconnect()
            ai_task.clear_task()


@pytest.mark.simulated
class TestMultiHandlerSampleRateMismatch:
    """Test validation failure when tasks have different sample rates."""

    def test_sample_rate_mismatch_rejected(self, simulated_device_name):
        """Test that configure() returns False when tasks have different sample rates.

        Creates two AITasks with different sample rates, verifies configure() rejects them.
        """
        ai_task1 = AITask("test_mismatch1", sample_rate=10000)
        ai_task2 = AITask("test_mismatch2", sample_rate=20000)

        try:
            # Configure both tasks
            ai_task1.add_channel(
                "ai0",
                device_ind=0,
                channel_ind=0,
                units="V",
            )
            ai_task1.start(start_task=False)

            ai_task2.add_channel(
                "ai1",
                device_ind=0,
                channel_ind=1,
                units="V",
            )
            ai_task2.start(start_task=False)

            # Configure should fail due to sample rate mismatch
            handler = MultiHandler()
            result = handler.configure(
                input_tasks=[ai_task1.task, ai_task2.task]
            )

            assert result is False, (
                "configure() should return False when tasks have different sample rates"
            )
        finally:
            handler.disconnect()
            ai_task1.clear_task()
            ai_task2.clear_task()
