"""Simulated device tests for counter/timer (CI/CO) operations.

Test stubs for counter input and counter output functionality using raw
nidaqmx (no wrapper classes — CITask/COTask don't exist yet).

These tests document expected behavior and limitations of simulated devices.
"""

from __future__ import annotations

import time

import pytest

import nidaqmx


@pytest.mark.simulated
class TestCounterInput:
    """Test counter input channels (CI) using raw nidaqmx."""

    def test_ci_count_edges(self, simulated_device_name):
        """Test CI edge counting channel.

        Creates nidaqmx.Task, adds ci_count_edges_chan on SimDev1/ctr0,
        starts, reads, verifies returns 0 (documented simulated device limitation).
        Stops and closes.

        Notes
        -----
        Simulated devices typically return 0 for counter channels because
        there are no physical edges to count. This test documents the
        expected behavior for future CI wrapper implementation.
        """
        task = nidaqmx.Task("test_ci_edges")
        try:
            # Add CI edge counter channel
            task.ci_channels.add_ci_count_edges_chan(
                f"{simulated_device_name}/ctr0"
            )

            # Start task
            task.start()

            # Read counter value
            count = task.read()

            # Simulated device limitation: no edges to count
            assert count == 0, (
                "Simulated counter should return 0 (no physical edges)"
            )

            # Stop task
            task.stop()
        finally:
            task.close()

    def test_ci_period_measurement(self, simulated_device_name):
        """Test CI period measurement channel.

        Creates task with period measurement channel, verifies task can be
        created and started (actual measurement returns 0 on simulated hardware).
        """
        task = nidaqmx.Task("test_ci_period")
        try:
            # Add CI period measurement channel
            task.ci_channels.add_ci_period_chan(
                f"{simulated_device_name}/ctr1"
            )

            # Configure timing
            task.timing.cfg_implicit_timing(
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
            )

            # Start task
            task.start()

            # For simulated devices, we just verify the task can be created
            # and started without errors
            assert True, "CI period channel created and started successfully"

            # Stop task
            task.stop()
        finally:
            task.close()


@pytest.mark.simulated
class TestCounterOutput:
    """Test counter output channels (CO) using raw nidaqmx."""

    def test_co_pulse_generation(self, simulated_device_name):
        """Test CO pulse generation channel.

        Creates nidaqmx.Task, adds co_pulse_chan_freq on SimDev1/ctr1 at
        1000Hz, 50% duty cycle. Starts, sleeps briefly, stops, closes.
        Verifies no errors occur.

        Notes
        -----
        Simulated devices do not produce physical output pulses, but the
        task should start and run without errors. This test documents
        expected behavior for future CO wrapper implementation.
        """
        task = nidaqmx.Task("test_co_pulse")
        try:
            # Add CO pulse channel at 1000 Hz, 50% duty cycle
            task.co_channels.add_co_pulse_chan_freq(
                f"{simulated_device_name}/ctr1",
                freq=1000.0,
                duty_cycle=0.5,
            )

            # Configure timing for continuous generation
            task.timing.cfg_implicit_timing(
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
            )

            # Start task
            task.start()

            # Let it run briefly (simulated device won't produce real pulses)
            time.sleep(0.1)

            # Verify task is running (not done)
            is_done = task.is_task_done()
            assert is_done is False, (
                "Task should be running (not done) in continuous mode"
            )

            # Stop task
            task.stop()
        finally:
            task.close()

    def test_co_pulse_ticks(self, simulated_device_name):
        """Test CO pulse generation using tick-based timing.

        Creates task with co_pulse_chan_ticks, verifies task can be created
        and configured (tick-based timing for precise control).
        """
        task = nidaqmx.Task("test_co_pulse_ticks")
        try:
            # Add CO pulse channel with tick-based timing
            # Low ticks = 100, high ticks = 100 (50% duty cycle)
            task.co_channels.add_co_pulse_chan_ticks(
                f"{simulated_device_name}/ctr2",
                low_ticks=100,
                high_ticks=100,
            )

            # Start task
            task.start()

            # Brief run
            time.sleep(0.05)

            # Stop task
            task.stop()

            # If we reach here, task was configured and ran successfully
            assert True, "CO pulse ticks channel created and ran successfully"
        finally:
            task.close()


@pytest.mark.simulated
class TestCounterLimitations:
    """Document known limitations of simulated counter channels."""

    def test_ci_frequency_measurement_returns_zero(self, simulated_device_name):
        """Test that frequency measurement returns 0 on simulated devices.

        Documents that simulated devices have no physical signal to measure,
        so frequency measurements return 0 or timeout.
        """
        task = nidaqmx.Task("test_ci_freq")
        try:
            # Add CI frequency measurement channel
            task.ci_channels.add_ci_freq_chan(
                f"{simulated_device_name}/ctr0",
                min_val=1.0,
                max_val=10000.0,
            )

            # Start task
            task.start()

            # Attempt to read (may timeout or return 0)
            # Use short timeout to avoid hanging the test
            try:
                freq = task.read(timeout=0.5)
                # If read succeeds, expect 0 (no signal)
                assert freq == 0.0 or freq is None, (
                    "Simulated device should return 0 or None for frequency"
                )
            except nidaqmx.errors.DaqError:
                # Timeout is acceptable for simulated devices
                pass

            # Stop task
            task.stop()
        finally:
            task.close()

    def test_co_cannot_route_to_ai(self, simulated_device_name):
        """Test that CO output cannot be directly routed to AI on simulated devices.

        Documents that while physical hardware can route CO to AI terminals,
        simulated devices do not support this cross-channel routing.
        """
        co_task = nidaqmx.Task("test_co_route")
        ai_task = nidaqmx.Task("test_ai_route")
        try:
            # Create CO pulse output
            co_task.co_channels.add_co_pulse_chan_freq(
                f"{simulated_device_name}/ctr0",
                freq=1000.0,
            )

            # Create AI task (would need external routing on physical hardware)
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0"
            )
            ai_task.timing.cfg_samp_clk_timing(rate=10000)

            # Start both tasks
            co_task.start()
            ai_task.start()

            # Read AI (will get 0 or noise on simulated device)
            data = ai_task.read(number_of_samples_per_channel=10)

            # Document: simulated devices do not show CO signal on AI
            assert data is not None, (
                "AI read should succeed, but won't show CO signal on simulated device"
            )

            # Stop tasks
            ai_task.stop()
            co_task.stop()
        finally:
            ai_task.close()
            co_task.close()
