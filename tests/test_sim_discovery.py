"""Simulated device discovery tests for nidaqwrapper.

All tests in this module run against NI-DAQmx simulated devices (not mocks).
They are marked with ``@pytest.mark.simulated`` and use the simulated device
fixtures from conftest.py.

Run with::

    uv run pytest tests/test_sim_discovery.py -v -m simulated

Device Configuration
---------------------
- SimDev1 : PCIe-6361 (simulated) — 16 AI, 2 AO, 24 DI lines, 24 DO lines
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.simulated


class TestSimulatedDeviceDiscovery:
    """Test device discovery functions with simulated hardware."""

    def test_list_devices_returns_simulated_device(self, simulated_device_name):
        """list_devices() returns list containing SimDev1 with correct structure.

        Each device dict must have 'name' and 'product_type' keys.
        """
        from nidaqwrapper import list_devices

        devices = list_devices()

        assert isinstance(devices, list)
        assert len(devices) > 0, "Expected at least one device"

        # Verify structure
        for dev in devices:
            assert "name" in dev, f"Device entry missing 'name': {dev}"
            assert "product_type" in dev, f"Device entry missing 'product_type': {dev}"

        # Verify SimDev1 is present
        device_names = [d["name"] for d in devices]
        assert simulated_device_name in device_names, (
            f"Expected {simulated_device_name} in device list, got: {device_names}"
        )

        # Verify SimDev1 has correct product type
        sim_dev = next(d for d in devices if d["name"] == simulated_device_name)
        assert sim_dev["product_type"] == "PCIe-6361", (
            f"Expected PCIe-6361, got {sim_dev['product_type']}"
        )

    def test_get_connected_devices_returns_sim_device(self, simulated_device_name):
        """get_connected_devices() returns set containing SimDev1."""
        from nidaqwrapper import get_connected_devices

        devices = get_connected_devices()

        assert isinstance(devices, set)
        assert len(devices) > 0, "Expected at least one device"
        assert simulated_device_name in devices, (
            f"Expected {simulated_device_name} in device set, got: {devices}"
        )

    def test_simulated_device_properties(self, sim_device):
        """Verify simulated device object has correct properties.

        The sim_device fixture provides the Device object for SimDev1.
        Verify it has the expected properties: is_simulated, product_type.
        """
        import nidaqmx.system

        assert isinstance(sim_device, nidaqmx.system.Device)
        assert sim_device.name == "SimDev1"
        assert sim_device.product_type == "PCIe-6361"

        # Verify it's marked as simulated
        assert sim_device.is_simulated is True

    def test_simulated_device_channel_counts(self, sim_device):
        """Verify simulated device has expected channel counts.

        PCIe-6361 spec: 16 AI, 2 AO, 24 DI lines, 24 DO lines.
        """
        # AI channels
        ai_chans = sim_device.ai_physical_chans
        assert len(ai_chans) == 16, f"Expected 16 AI channels, got {len(ai_chans)}"

        # AO channels
        ao_chans = sim_device.ao_physical_chans
        assert len(ao_chans) == 2, f"Expected 2 AO channels, got {len(ao_chans)}"

        # DI lines (PCIe-6361 has port0-port2, 8 lines each = 24 total)
        di_lines = sim_device.di_lines
        assert len(di_lines) == 24, f"Expected 24 DI lines, got {len(di_lines)}"

        # DO lines (PCIe-6361 has port0-port2, 8 lines each = 24 total)
        do_lines = sim_device.do_lines
        assert len(do_lines) == 24, f"Expected 24 DO lines, got {len(do_lines)}"
