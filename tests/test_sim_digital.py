"""Simulated device tests for digital I/O classes.

All tests run against SimDev1 (PCIe-6361 simulated device) and are marked
with ``@pytest.mark.simulated``.

Run with::

    uv run pytest tests/test_sim_digital.py -v -m simulated

Requirements
------------
- SimDev1 must exist (created by scripts/setup_simulated_devices.sh)
- Tests use port0 for DI and port1 for DO to avoid conflicts
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.simulated


# ===========================================================================
# Group 6: DITask and DOTask Simulated Tests
# ===========================================================================


class TestDITaskSimulated:
    """Validate DITask on-demand and clocked modes with SimDev1."""

    def test_on_demand_read(self, simulated_device_name: str) -> None:
        """DITask on-demand read returns boolean/int data.

        Creates DITask without sample_rate (on-demand mode), adds 4 lines,
        starts, and reads. Verifies data is returned.
        """
        from nidaqwrapper import DITask

        di = DITask("test_sim_di_ondemand")
        try:
            # Add 4 lines on port0
            di.add_channel(
                "di_ch",
                lines=f"{simulated_device_name}/port0/line0:3"
            )
            di.start()

            data = di.read()

            assert isinstance(data, np.ndarray)
            assert data.size == 4, f"Expected 4 values, got {data.size}"
            # Values should be boolean-like (0 or 1)
            for val in data.flat:
                assert val in (True, False, 0, 1), f"Unexpected value: {val}"
        finally:
            di.clear_task()

    def test_clocked_read(self, simulated_device_name: str) -> None:
        """DITask clocked read returns array data.

        Creates DITask with sample_rate, adds channels, starts the task,
        waits for buffer to fill, then acquires data.
        """
        from nidaqwrapper import DITask

        di = DITask("test_sim_di_clocked", sample_rate=1000)
        try:
            # Add 4 lines on port0
            di.add_channel(
                "di_ch",
                lines=f"{simulated_device_name}/port0/line0:3"
            )
            di.start(start_task=True)

            # Let buffer fill
            time.sleep(0.15)

            data = di.acquire()

            assert isinstance(data, np.ndarray)
            assert data.ndim == 2, f"Expected 2D array, got shape {data.shape}"
            assert data.shape[1] == 4, f"Expected 4 lines, got {data.shape[1]}"
            assert data.shape[0] > 0, "Expected at least one sample"
        finally:
            di.clear_task()

    def test_clocked_blocking_read(self, simulated_device_name: str) -> None:
        """DITask acquire(n_samples) blocks until n_samples available."""
        from nidaqwrapper import DITask

        di = DITask("test_sim_di_blocking", sample_rate=1000)
        try:
            di.add_channel(
                "di_ch",
                lines=f"{simulated_device_name}/port0/line0:3"
            )
            di.start(start_task=True)

            # Blocking read for exactly 100 samples
            data = di.acquire(n_samples=100)

            assert isinstance(data, np.ndarray)
            assert data.shape == (100, 4), f"Expected (100, 4), got {data.shape}"
        finally:
            di.clear_task()


class TestDOTaskSimulated:
    """Validate DOTask on-demand and clocked modes with SimDev1."""

    def test_on_demand_write(self, simulated_device_name: str) -> None:
        """DOTask on-demand write succeeds without error.

        Creates DOTask without sample_rate, adds 4 lines, starts, and writes
        a list of boolean values.
        """
        from nidaqwrapper import DOTask

        do = DOTask("test_sim_do_ondemand")
        try:
            # Add 4 lines on port1 to avoid conflict with DI tests
            do.add_channel(
                "do_ch",
                lines=f"{simulated_device_name}/port1/line0:3"
            )
            do.start()

            # Write 4 boolean values (one per line)
            do.write([True, False, True, False])

            # No assertion possible — simulated device has no loopback to verify written data
        finally:
            do.clear_task()

    def test_clocked_write(self, simulated_device_name: str) -> None:
        """DOTask clocked write_continuous succeeds without error.

        Creates DOTask with sample_rate, adds channels, configures timing,
        then writes continuously (which pre-fills buffer and auto-starts).
        """
        from nidaqwrapper import DOTask

        do = DOTask("test_sim_do_clocked", sample_rate=1000)
        try:
            # Add 4 lines on port0 (port1 doesn't support buffered operations)
            do.add_channel(
                "do_ch",
                lines=f"{simulated_device_name}/port0/line4:7"
            )
            do.start(start_task=False)

            # Generate 200 samples of bool data (200, 4)
            data = np.random.randint(0, 2, size=(200, 4), dtype=bool)
            do.write_continuous(data)

            # Let it run briefly
            time.sleep(0.1)

            # No assertion needed — if no exception, write succeeded
        finally:
            do.clear_task()

    def test_single_line_write(self, simulated_device_name: str) -> None:
        """DOTask write handles single-line (scalar) data."""
        from nidaqwrapper import DOTask

        do = DOTask("test_sim_do_single")
        try:
            # Add single line on port1
            do.add_channel(
                "do_ch",
                lines=f"{simulated_device_name}/port1/line4"
            )
            do.start()

            # Write single bool value
            do.write(True)
            do.write(False)
            do.write(1)  # int also works
            do.write(0)

            # No assertion possible — simulated device has no loopback to verify written data
        finally:
            do.clear_task()
