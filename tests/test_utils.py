"""Tests for nidaqwrapper.utils module.

Notes
-----
nidaqmx 1.4.1 (installed) does not have
``AccelSensitivityUnits.M_VOLTS_PER_METERS_PER_SECOND_SQUARED``.
The enum only contains ``MILLIVOLTS_PER_G`` and ``VOLTS_PER_G``.

The intended bug fix (mapping ``mV/m/s**2`` to a dedicated m/s² sensitivity
constant) cannot be implemented until NI ships that constant in nidaqmx.
Until then, ``mV/m/s**2`` maps to ``MILLIVOLTS_PER_G`` — matching LDAQ behaviour
— with a TODO marking the correct fix for a future nidaqmx release.

See agent note: .claude-notes/agent-notes/units-nidaqmx-compat.md
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestUNITS:
    """Tests for the UNITS dictionary."""

    def test_units_has_exactly_seven_keys(self):
        """UNITS contains exactly 7 unit mappings."""
        from nidaqwrapper.utils import UNITS

        assert len(UNITS) == 7

    def test_units_keys(self):
        """UNITS contains all expected string keys."""
        from nidaqwrapper.utils import UNITS

        expected_keys = {"mV/g", "mV/m/s**2", "g", "m/s**2", "mV/N", "N", "V"}
        assert set(UNITS.keys()) == expected_keys

    def test_units_mV_per_g(self):
        """mV/g maps to AccelSensitivityUnits.MILLIVOLTS_PER_G."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["mV/g"] == constants.AccelSensitivityUnits.MILLIVOLTS_PER_G

    def test_units_mV_per_ms2_is_accel_sensitivity_unit(self):
        """mV/m/s**2 maps to an AccelSensitivityUnits member.

        Notes
        -----
        nidaqmx 1.4.1 has no ``M_VOLTS_PER_METERS_PER_SECOND_SQUARED`` constant.
        The intended fix (distinct m/s² sensitivity unit) is deferred until NI
        ships the constant.  This test ensures the key exists and resolves to
        a valid ``AccelSensitivityUnits`` member, not some arbitrary value.
        """
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["mV/m/s**2"] in list(constants.AccelSensitivityUnits)

    def test_units_g(self):
        """g maps to AccelUnits.G."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["g"] == constants.AccelUnits.G

    def test_units_ms2(self):
        """m/s**2 maps to AccelUnits.METERS_PER_SECOND_SQUARED."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["m/s**2"] == constants.AccelUnits.METERS_PER_SECOND_SQUARED

    def test_units_mV_per_N(self):
        """mV/N maps to ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert (
            UNITS["mV/N"]
            == constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON
        )

    def test_units_N(self):
        """N maps to ForceUnits.NEWTONS."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["N"] == constants.ForceUnits.NEWTONS

    def test_units_V(self):
        """V maps to VoltageUnits.VOLTS."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["V"] == constants.VoltageUnits.VOLTS

    def test_units_invalid_key_raises_keyerror(self):
        """Accessing an invalid key raises KeyError."""
        from nidaqwrapper.utils import UNITS

        with pytest.raises(KeyError):
            _ = UNITS["invalid_unit"]


class TestListDevices:
    """Tests for list_devices() function."""

    def test_list_devices_returns_list(self, mock_system):
        """list_devices() returns a list."""
        system = mock_system()
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert isinstance(result, list)

    def test_list_devices_two_devices(self, mock_system):
        """list_devices() returns correct dicts for two connected devices."""
        system = mock_system(
            devices=[("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        )
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert len(result) == 2
            assert result[0] == {"name": "cDAQ1Mod1", "product_type": "NI 9234"}
            assert result[1] == {"name": "cDAQ1Mod2", "product_type": "NI 9263"}

    def test_list_devices_no_devices(self, mock_system):
        """list_devices() returns empty list when no devices are connected."""
        system = mock_system(devices=[])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert result == []

    def test_list_devices_dict_keys(self, mock_system):
        """Each device dict has exactly 'name' and 'product_type' keys."""
        system = mock_system(devices=[("Dev1", "NI 9234")])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert set(result[0].keys()) == {"name", "product_type"}

    def test_list_devices_dict_values_are_strings(self, mock_system):
        """Device dict values are strings."""
        system = mock_system(devices=[("Dev1", "NI 9234")])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert isinstance(result[0]["name"], str)
            assert isinstance(result[0]["product_type"], str)

    def test_list_devices_raises_without_nidaqmx(self):
        """list_devices() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import list_devices

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                list_devices()
