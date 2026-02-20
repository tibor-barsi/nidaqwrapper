"""Simulated device tests for raw nidaqmx Task passthrough.

Tests that verify the wrapper never interferes with raw nidaqmx.Task
operations when using the from_task() factory method.
"""

from __future__ import annotations

import pytest

import nidaqmx
from nidaqmx.constants import TerminalConfiguration

from nidaqwrapper import AITask


@pytest.mark.simulated
class TestRawTaskPassthrough:
    """Test wrapper behavior when wrapping externally-created tasks."""

    def test_custom_timing_preserved(self, simulated_device_name):
        """Test that custom timing configuration is preserved when wrapping.

        Creates raw nidaqmx.Task, configures timing at 25600Hz, wraps with
        AITask.from_task(), verifies wrapper.sample_rate == 25600.
        """
        raw_task = nidaqmx.Task("test_custom_timing")
        try:
            # Add AI channel
            raw_task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0",
                min_val=-10.0,
                max_val=10.0,
            )

            # Configure custom timing
            raw_task.timing.cfg_samp_clk_timing(rate=25600)

            # Wrap with AITask
            wrapper = AITask.from_task(raw_task)

            # Verify wrapper read the configured rate
            assert wrapper.sample_rate == 25600, (
                "Wrapper should preserve the custom sample rate from the raw task"
            )

            # Verify task_name was also read
            assert wrapper.task_name == "test_custom_timing", (
                "Wrapper should preserve the task name from the raw task"
            )

            # Verify wrapper.task points to the same object
            assert wrapper.task is raw_task, (
                "Wrapper should reference the same nidaqmx.Task object"
            )
        finally:
            # Wrapper does NOT close external tasks
            wrapper.clear_task()
            # We must close it manually
            raw_task.close()

    def test_non_default_terminal_config(self, simulated_device_name):
        """Test that non-default terminal configuration is preserved.

        Creates raw nidaqmx.Task, adds AI channel with DIFF terminal config,
        wraps with AITask.from_task(), starts raw task, reads through wrapper.
        """
        raw_task = nidaqmx.Task("test_terminal_config")
        try:
            # Add AI channel with differential terminal configuration
            raw_task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0",
                terminal_config=TerminalConfiguration.DIFF,
                min_val=-10.0,
                max_val=10.0,
            )

            # Configure timing
            raw_task.timing.cfg_samp_clk_timing(rate=10000)

            # Wrap with AITask
            wrapper = AITask.from_task(raw_task)

            # Start the raw task
            raw_task.start()

            # Read through wrapper
            data = wrapper.acquire(n_samples=50)

            # Verify data was returned
            assert data.shape[0] == 1, "Should have 1 channel"
            assert data.shape[1] == 50, "Should have 50 samples"

            # Stop task
            raw_task.stop()
        finally:
            wrapper.clear_task()
            raw_task.close()

    def test_clear_task_does_not_close_external(self, simulated_device_name):
        """Test that wrapper.clear_task() does NOT close externally-owned tasks.

        Wraps raw task via from_task(), calls wrapper.clear_task(), verifies
        raw task is still usable (task.read() works), then closes raw task manually.
        """
        raw_task = nidaqmx.Task("test_no_close")
        try:
            # Add AI channel
            raw_task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0",
                min_val=-10.0,
                max_val=10.0,
            )

            # Configure timing
            raw_task.timing.cfg_samp_clk_timing(rate=10000)

            # Wrap with AITask
            wrapper = AITask.from_task(raw_task)

            # Verify ownership flag
            assert wrapper._owns_task is False

            # Call wrapper.clear_task() — should NOT close the task
            wrapper.clear_task()

            # Verify wrapper.task was set to None
            assert wrapper.task is None, (
                "Wrapper.task should be None after clear_task()"
            )

            # Verify raw task is still usable
            raw_task.start()
            data = raw_task.read(number_of_samples_per_channel=10)
            raw_task.stop()

            # Data should be returned (task is still open)
            assert data is not None, (
                "Raw task should still be usable after wrapper.clear_task()"
            )
        finally:
            # Caller is responsible for closing the external task
            raw_task.close()


@pytest.mark.simulated
class TestExternalTaskRestrictions:
    """Test that add_channel() and start() are blocked for external tasks."""

    def test_add_channel_blocked_on_external_task(self, simulated_device_name):
        """Test that add_channel() raises RuntimeError for externally-owned tasks."""
        raw_task = nidaqmx.Task("test_add_channel_blocked")
        try:
            raw_task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0"
            )
            raw_task.timing.cfg_samp_clk_timing(rate=10000)

            wrapper = AITask.from_task(raw_task)

            # Attempt to add a channel — should raise RuntimeError
            with pytest.raises(RuntimeError, match="Cannot add channels"):
                wrapper.add_channel(
                    "ai1",
                    device_ind=0,
                    channel_ind=1,
                    units="V",
                )
        finally:
            wrapper.clear_task()
            raw_task.close()

    def test_start_blocked_on_external_task(self, simulated_device_name):
        """Test that start() raises RuntimeError for externally-owned tasks."""
        raw_task = nidaqmx.Task("test_start_blocked")
        try:
            raw_task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0"
            )
            raw_task.timing.cfg_samp_clk_timing(rate=10000)

            wrapper = AITask.from_task(raw_task)

            # Attempt to start via wrapper — should raise RuntimeError
            with pytest.raises(RuntimeError, match="Cannot start"):
                wrapper.start()
        finally:
            wrapper.clear_task()
            raw_task.close()
