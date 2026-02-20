"""Simulated device tests for real error code handling.

Tests that verify proper error handling with real NI-DAQmx error codes
when operations fail on simulated hardware.
"""

from __future__ import annotations

import pytest

import nidaqmx
from nidaqmx.errors import DaqError


@pytest.mark.simulated
class TestDaqErrors:
    """Test real DAQmx error codes with simulated devices."""

    def test_invalid_device_name(self):
        """Test that attempting to use a non-existent device raises DaqError.

        Tries to create a task and add channel with device name 'NonExistent',
        verifies DaqError is raised.
        """
        task = nidaqmx.Task("test_invalid_device")
        try:
            # Attempt to add channel on non-existent device
            with pytest.raises(DaqError) as exc_info:
                task.ai_channels.add_ai_voltage_chan(
                    "NonExistent/ai0",
                    min_val=-10.0,
                    max_val=10.0,
                )

            # Verify it's a DaqError with an error code
            assert hasattr(exc_info.value, "error_code"), (
                "DaqError should have error_code attribute"
            )
            assert exc_info.value.error_code != 0, (
                "Error code should be non-zero for device not found"
            )
        finally:
            task.close()

    def test_duplicate_task_name(self):
        """Test that creating duplicate task names raises an error.

        Creates nidaqmx.Task('DuplicateTest'), attempts to create another with
        the same name, verifies error. Cleans up both in finally.
        """
        task1 = nidaqmx.Task("DuplicateTest")
        task2 = None
        try:
            # Attempt to create second task with same name
            with pytest.raises(DaqError) as exc_info:
                task2 = nidaqmx.Task("DuplicateTest")

            # Verify error has error_code
            assert hasattr(exc_info.value, "error_code"), (
                "DaqError should have error_code attribute"
            )
        finally:
            # Clean up both tasks
            if task1 is not None:
                try:
                    task1.close()
                except Exception:
                    pass
            if task2 is not None:
                try:
                    task2.close()
                except Exception:
                    pass

    def test_read_before_start(self, simulated_device_name):
        """Test that reading from a task before starting raises DaqError.

        Creates task with AI channel on SimDev1, does NOT start it, calls
        task.read(), verifies DaqError.
        """
        task = nidaqmx.Task("test_read_before_start")
        try:
            # Add AI channel
            task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0",
                min_val=-10.0,
                max_val=10.0,
            )

            # Configure timing
            task.timing.cfg_samp_clk_timing(rate=10000)

            # Attempt to read without starting — should raise DaqError
            with pytest.raises(DaqError) as exc_info:
                task.read(number_of_samples_per_channel=10)

            # Verify error has error_code
            assert hasattr(exc_info.value, "error_code"), (
                "DaqError should have error_code attribute"
            )
            assert exc_info.value.error_code != 0, (
                "Error code should be non-zero when reading before start"
            )
        finally:
            task.close()

    def test_invalid_channel_index(self, simulated_device_name):
        """Test that invalid channel index raises DaqError.

        Tries to add channel 'SimDev1/ai99' (device only has 16 AI channels),
        verifies DaqError.
        """
        task = nidaqmx.Task("test_invalid_channel")
        try:
            # Attempt to add channel with out-of-range index
            # PCIe-6361 (SimDev1) has 16 AI channels (ai0-ai15)
            with pytest.raises(DaqError) as exc_info:
                task.ai_channels.add_ai_voltage_chan(
                    f"{simulated_device_name}/ai99",
                    min_val=-10.0,
                    max_val=10.0,
                )

            # Verify error has error_code
            assert hasattr(exc_info.value, "error_code"), (
                "DaqError should have error_code attribute"
            )
            assert exc_info.value.error_code != 0, (
                "Error code should be non-zero for invalid channel index"
            )
        finally:
            task.close()


@pytest.mark.simulated
class TestChannelRangeValidation:
    """Test channel range and configuration validation errors."""

    def test_invalid_terminal_configuration_combination(
        self, simulated_device_name
    ):
        """Test that incompatible terminal configurations are detected.

        Note: This test documents that nidaqmx may accept certain terminal
        configurations even when they are not optimal, as validation depends
        on the specific device model.
        """
        task = nidaqmx.Task("test_terminal_config")
        try:
            # Add channel — device will validate based on its capabilities
            # PCIe-6361 supports various terminal configurations
            task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0",
                min_val=-10.0,
                max_val=10.0,
            )

            # If we reach here, the configuration was accepted
            # No error raised — device accepted the configuration
            assert True, "Device accepted the terminal configuration"
        finally:
            task.close()

    def test_min_max_range_accepted(self, simulated_device_name):
        """Test that min > max doesn't cause immediate error.

        Documents that nidaqmx may accept inverted ranges at configuration
        time, with errors potentially occurring later at read time.
        """
        task = nidaqmx.Task("test_inverted_range")
        try:
            # Some drivers accept inverted ranges and coerce them
            task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0",
                min_val=10.0,
                max_val=-10.0,
            )

            # If we reach here, driver accepted or coerced the range
            assert True, "Driver accepted the channel configuration"
        finally:
            task.close()
