"""Analog input task configuration for NI-DAQmx devices.

Provides the ``NITask`` class for programmatic creation and management
of analog input acquisition tasks.  Supports accelerometer (IEPE),
force (IEPE), and voltage channels with optional linear custom scales.

Notes
-----
nidaqmx is an optional dependency.  If absent, ``_NIDAQMX_AVAILABLE``
is ``False`` and construction raises ``RuntimeError``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .utils import UNITS, _require_nidaqmx

logger = logging.getLogger("nidaqwrapper.task_input")

try:
    import nidaqmx
    from nidaqmx import constants

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False


class NITask:
    """Programmatic analog input task for NI-DAQmx devices.

    Parameters
    ----------
    task_name : str
        Unique name for this task.  Must not collide with an existing
        task in NI MAX.
    sample_rate : float
        Sampling rate in Hz.
    settings_file : str, optional
        Path to an ``.xlsx`` or ``.csv`` file containing sensor calibration
        data (columns: ``serial_nr``, ``sensitivity``,
        ``sensitivity_units``, ``units``).

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX, or if ``settings_file``
        has an unsupported extension.
    TypeError
        If ``settings_file`` is not a string.
    RuntimeError
        If nidaqmx is not installed.

    Examples
    --------
    >>> task = NITask("vibration_test", sample_rate=25600)
    >>> task.add_channel("accel_x", device_ind=0, channel_ind=0,
    ...                  sensitivity=100.0, sensitivity_units="mV/g",
    ...                  units="g")
    >>> task.initiate()
    """

    def __init__(
        self,
        task_name: str,
        sample_rate: float,
        settings_file: str | None = None,
    ) -> None:
        _require_nidaqmx()

        self._logger = logging.getLogger("nidaqwrapper.task")
        self.task_name = task_name
        self.sample_rate = sample_rate

        # Device discovery
        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [d.name for d in system.devices]
        self.device_product_type: list[str] = [
            d.product_type for d in system.devices
        ]

        # Check for duplicate task name
        if task_name in system.tasks.task_names:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a different name or delete the existing task first."
            )

        self.sample_mode = constants.AcquisitionType.CONTINUOUS
        self.channels: dict[str, dict[str, Any]] = {}
        self.settings = None

        if settings_file is not None:
            self._read_settings_file(settings_file)

    # -- Introspection properties -------------------------------------------

    @property
    def channel_list(self) -> list[str]:
        """List of channel names in insertion order."""
        return list(self.channels.keys())

    @property
    def number_of_ch(self) -> int:
        """Number of configured channels."""
        return len(self.channels)

    @property
    def channel_info(self) -> dict[str, dict[str, Any]]:
        """Copy of the full channel configuration dict."""
        return {k: dict(v) for k, v in self.channels.items()}

    # -- Channel configuration -----------------------------------------------

    def add_channel(
        self,
        channel_name: str,
        device_ind: int,
        channel_ind: int,
        sensitivity: float | None = None,
        sensitivity_units: str | None = None,
        units: str | None = None,
        serial_nr: str | None = None,
        scale: float | tuple[float, float] | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """Add an analog input channel to this task.

        The channel is stored in the configuration but not added to a
        hardware task until :meth:`initiate` is called.

        Parameters
        ----------
        channel_name : str
            Unique name for this channel.
        device_ind : int
            Index into :attr:`device_list` identifying the target device.
        channel_ind : int
            Physical analog-input channel number on the device.
        sensitivity : float, optional
            Sensor sensitivity.  Required for accel/force channels
            when *scale* is not provided.
        sensitivity_units : str, optional
            Key into the ``UNITS`` dict for sensor sensitivity units
            (e.g. ``'mV/g'``, ``'mV/N'``).
        units : str
            Key into the ``UNITS`` dict for output measurement units
            (e.g. ``'g'``, ``'N'``, ``'V'``).  Always required.
        serial_nr : str, optional
            Sensor serial number.  When provided, *sensitivity*,
            *sensitivity_units*, and *units* are looked up from the
            settings file.
        scale : float or tuple[float, float], optional
            Linear custom scale.  Float → slope with y_intercept=0.
            Tuple → ``(slope, y_intercept)``.  When given, *sensitivity*
            and *sensitivity_units* are not required.
        min_val : float, optional
            Minimum expected value.  ``0.0`` is a valid value.
        max_val : float, optional
            Maximum expected value.  ``0.0`` is a valid value.

        Raises
        ------
        ValueError
            Duplicate channel name, duplicate physical channel, out-of-range
            device, invalid units, missing sensitivity, or missing units.
        TypeError
            Invalid *scale* type.
        """
        # -- Serial number lookup (settings file) ---------------------------
        if serial_nr is not None and sensitivity is None:
            if self.settings is None:
                raise ValueError(
                    "Cannot look up serial_nr: no settings file has been loaded. "
                    "Pass settings_file= to the NITask constructor."
                )
            sensitivity, sensitivity_units, units = self._lookup_serial_nr(
                serial_nr
            )

        # -- Basic validation -----------------------------------------------
        if units is None:
            raise ValueError(
                "units must be specified. "
                f"Valid units: {list(UNITS.keys())}"
            )

        if channel_name in self.channels:
            raise ValueError(
                f"Channel name '{channel_name}' already exists in this task."
            )

        if device_ind not in range(len(self.device_list)):
            raise ValueError(
                f"device_ind {device_ind} is out of range. "
                f"Available devices: {self.device_list}"
            )

        for ch in self.channels.values():
            if ch["device_ind"] == device_ind and ch["channel_ind"] == channel_ind:
                raise ValueError(
                    f"Physical channel ai{channel_ind} on device "
                    f"'{self.device_list[device_ind]}' is already in use."
                )

        # -- Scale type validation ------------------------------------------
        if scale is not None:
            if not isinstance(scale, (int, float, tuple)):
                raise TypeError(
                    f"scale must be a float or tuple, got {type(scale).__name__}."
                )

        # -- Units / sensitivity validation (skip when scale given) ---------
        if scale is None:
            if sensitivity_units is not None and sensitivity_units not in UNITS:
                raise ValueError(
                    f"Invalid sensitivity_units: '{sensitivity_units}'. "
                    f"Valid sensitivity_units: {list(UNITS.keys())}"
                )
            if units not in UNITS:
                raise ValueError(
                    f"Invalid units: '{units}'. "
                    f"Valid units: {list(UNITS.keys())}"
                )

        # -- Resolve constants from UNITS dict ------------------------------
        resolved_units = UNITS[units] if units in UNITS else units
        resolved_sens_units = (
            UNITS[sensitivity_units]
            if sensitivity_units is not None and sensitivity_units in UNITS
            else sensitivity_units
        )

        # -- Sensitivity required for non-voltage, non-scale channels -------
        if scale is None:
            is_voltage = (
                hasattr(resolved_units, "__objclass__")
                and resolved_units.__objclass__.__name__ == "VoltageUnits"
            )
            if not is_voltage:
                if sensitivity is None:
                    raise ValueError(
                        "sensitivity must be specified for non-voltage "
                        "channels when no scale is provided."
                    )
                if sensitivity_units is None:
                    raise ValueError(
                        "sensitivity_units must be specified for non-voltage "
                        "channels when no scale is provided."
                    )

        # -- Store channel configuration ------------------------------------
        self.channels[channel_name] = {
            "device_ind": device_ind,
            "channel_ind": channel_ind,
            "sensitivity": sensitivity,
            "sensitivity_units": resolved_sens_units,
            "units": resolved_units,
            "serial_nr": serial_nr,
            "scale": scale,
            "min_val": min_val,
            "max_val": max_val,
            "custom_scale_name": "",
        }

        self._logger.debug("Channel '%s' added to task '%s'.", channel_name, self.task_name)

    def _lookup_serial_nr(
        self, serial_nr: str
    ) -> tuple[float, str, str]:
        """Look up sensor calibration data by serial number.

        Parameters
        ----------
        serial_nr : str
            Sensor serial number to find in the settings DataFrame.

        Returns
        -------
        tuple[float, str, str]
            ``(sensitivity, sensitivity_units, units)`` from the matching row.

        Raises
        ------
        ValueError
            If required columns are missing, or serial_nr not found.
        """
        required_cols = {"serial_nr", "sensitivity", "sensitivity_units", "units"}
        missing = required_cols - set(self.settings.columns)
        if missing:
            raise ValueError(
                f"Settings file is missing required columns: {sorted(missing)}"
            )

        row = self.settings[self.settings["serial_nr"] == serial_nr]
        if len(row) == 0:
            raise ValueError(
                f"Serial number '{serial_nr}' not found in settings file."
            )

        first = row.iloc[0]
        return (
            float(first["sensitivity"]),
            str(first["sensitivity_units"]),
            str(first["units"]),
        )

    # -- Settings file loading ----------------------------------------------

    def _read_settings_file(self, file_name: str) -> None:
        """Load sensor calibration data from a settings file.

        Parameters
        ----------
        file_name : str
            Path to ``.xlsx`` or ``.csv`` settings file.

        Raises
        ------
        TypeError
            If *file_name* is not a string.
        ValueError
            If the file extension is not ``.xlsx`` or ``.csv``.
        ImportError
            If pandas is not installed.
        """
        if not isinstance(file_name, str):
            raise TypeError(
                f"settings_file must be a string, got {type(file_name).__name__}."
            )

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load settings files. "
                "Install with: pip install nidaqwrapper[settings]"
            ) from None

        if file_name.endswith(".xlsx"):
            self.settings = pd.read_excel(file_name)
        elif file_name.endswith(".csv"):
            self.settings = pd.read_csv(file_name)
        else:
            raise ValueError(
                f"Unsupported settings file extension: '{file_name}'. "
                "Only .xlsx and .csv files are supported."
            )
