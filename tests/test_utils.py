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


class TestListTasks:
    """Tests for list_tasks() function."""

    def test_list_tasks_returns_list(self, mock_system):
        """list_tasks() returns a list."""
        system = mock_system(task_names=["MyTask"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_tasks

            result = list_tasks()
            assert isinstance(result, list)

    def test_list_tasks_two_tasks(self, mock_system):
        """list_tasks() returns the correct names for two saved tasks."""
        system = mock_system(task_names=["MyInputTask", "MyOutputTask"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_tasks

            result = list_tasks()
            assert result == ["MyInputTask", "MyOutputTask"]

    def test_list_tasks_no_tasks(self, mock_system):
        """list_tasks() returns an empty list when no tasks are saved."""
        system = mock_system(task_names=[])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_tasks

            result = list_tasks()
            assert result == []

    def test_list_tasks_return_type_is_list_of_strings(self, mock_system):
        """Every element returned by list_tasks() is a str."""
        system = mock_system(task_names=["TaskA", "TaskB"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_tasks

            result = list_tasks()
            assert all(isinstance(name, str) for name in result)

    def test_list_tasks_raises_without_nidaqmx(self):
        """list_tasks() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import list_tasks

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                list_tasks()


class TestGetTaskByName:
    """Tests for get_task_by_name() function."""

    def test_get_task_by_name_success(self, mock_system):
        """get_task_by_name() returns the loaded task object on success."""
        system = mock_system(task_names=["MyTask"])
        loaded_task = system.tasks.__iter__().__next__().load.return_value
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            result = get_task_by_name("MyTask")
            assert result is loaded_task

    def test_get_task_by_name_not_found_raises_keyerror(self, mock_system):
        """get_task_by_name() raises KeyError when no task matches the name."""
        system = mock_system(task_names=["Other"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(KeyError) as exc_info:
                get_task_by_name("NonExistent")

            error_message = str(exc_info.value)
            assert "NonExistent" in error_message
            assert "Other" in error_message

    def test_get_task_by_name_error_200089_returns_none(self, mock_system):
        """get_task_by_name() returns None when error code -200089 is raised.

        Error -200089 means the task is already loaded elsewhere; the function
        swallows the exception and returns None.
        """
        from nidaqmx.errors import DaqError

        system = mock_system(task_names=["MyTask"])
        # Retrieve the pre-built mock task from the tasks collection
        mock_task_obj = list(system.tasks)[0]
        mock_task_obj.load.side_effect = DaqError("Task already loaded", -200089)

        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            result = get_task_by_name("MyTask")
            assert result is None

    def test_get_task_by_name_error_200089_logs_warning(
        self, mock_system, caplog
    ):
        """get_task_by_name() logs a WARNING when error code -200089 is raised."""
        import logging

        from nidaqmx.errors import DaqError

        system = mock_system(task_names=["MyTask"])
        mock_task_obj = list(system.tasks)[0]
        mock_task_obj.load.side_effect = DaqError("Task already loaded", -200089)

        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with caplog.at_level(logging.WARNING, logger="nidaqwrapper.utils"):
                get_task_by_name("MyTask")

            assert any(
                record.levelno == logging.WARNING for record in caplog.records
            )

    def test_get_task_by_name_error_201003_raises_connection_error(
        self, mock_system
    ):
        """get_task_by_name() raises ConnectionError when error code -201003 is raised.

        Error -201003 means the device is disconnected or in use by another
        application.
        """
        from nidaqmx.errors import DaqError

        system = mock_system(task_names=["MyTask"])
        mock_task_obj = list(system.tasks)[0]
        mock_task_obj.load.side_effect = DaqError("Device disconnected", -201003)

        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(ConnectionError) as exc_info:
                get_task_by_name("MyTask")

            error_message = str(exc_info.value)
            assert "disconnected" in error_message or "in use" in error_message

    def test_get_task_by_name_error_201003_logs_error(
        self, mock_system, caplog
    ):
        """get_task_by_name() logs an ERROR when error code -201003 is raised."""
        import logging

        from nidaqmx.errors import DaqError

        system = mock_system(task_names=["MyTask"])
        mock_task_obj = list(system.tasks)[0]
        mock_task_obj.load.side_effect = DaqError("Device disconnected", -201003)

        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with caplog.at_level(logging.ERROR, logger="nidaqwrapper.utils"):
                with pytest.raises(ConnectionError):
                    get_task_by_name("MyTask")

            assert any(
                record.levelno == logging.ERROR for record in caplog.records
            )

    def test_get_task_by_name_other_error_propagates(self, mock_system):
        """get_task_by_name() re-raises DaqError for unrecognised error codes."""
        from nidaqmx.errors import DaqError

        system = mock_system(task_names=["MyTask"])
        mock_task_obj = list(system.tasks)[0]
        original_error = DaqError("Unknown error", -99999)
        mock_task_obj.load.side_effect = original_error

        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(DaqError) as exc_info:
                get_task_by_name("MyTask")

            assert exc_info.value is original_error

    def test_get_task_by_name_raises_without_nidaqmx(self):
        """get_task_by_name() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                get_task_by_name("AnyTask")

    def test_get_task_by_name_empty_name_raises_keyerror(self, mock_system):
        """get_task_by_name('') raises KeyError — empty string is not a valid name."""
        system = mock_system(task_names=["RealTask"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(KeyError):
                get_task_by_name("")


class TestGetConnectedDevices:
    """Tests for get_connected_devices() function."""

    def test_get_connected_devices_two_devices(self, mock_system):
        """get_connected_devices() returns the name set for two connected devices."""
        system = mock_system(
            devices=[("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        )
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_connected_devices

            result = get_connected_devices()
            assert result == {"cDAQ1Mod1", "cDAQ1Mod2"}

    def test_get_connected_devices_no_devices(self, mock_system):
        """get_connected_devices() returns an empty set when no devices are connected."""
        system = mock_system(devices=[])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_connected_devices

            result = get_connected_devices()
            assert result == set()

    def test_get_connected_devices_return_type_is_set(self, mock_system):
        """get_connected_devices() always returns a set, never a list or tuple."""
        system = mock_system(devices=[("Dev1", "NI 9234")])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_connected_devices

            result = get_connected_devices()
            assert isinstance(result, set)

    def test_get_connected_devices_values_are_strings(self, mock_system):
        """Every element in the returned set is a str."""
        system = mock_system(
            devices=[("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        )
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_connected_devices

            result = get_connected_devices()
            assert all(isinstance(name, str) for name in result)

    def test_get_connected_devices_raises_without_nidaqmx(self):
        """get_connected_devices() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import get_connected_devices

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                get_connected_devices()
