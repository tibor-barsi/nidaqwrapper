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

    # -- Settings file loading (placeholder, implemented in Task 13) --------

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
