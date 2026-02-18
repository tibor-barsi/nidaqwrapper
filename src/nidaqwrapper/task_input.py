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

import warnings
from typing import Any

import numpy as np

from .utils import UNITS, _require_nidaqmx

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
            "units_str": units,  # original string for create_lin_scale
            "serial_nr": serial_nr,
            "scale": scale,
            "min_val": min_val,
            "max_val": max_val,
            "custom_scale_name": "",
        }

    # -- Task lifecycle ------------------------------------------------------

    def initiate(self, start_task: bool = False) -> None:
        """Create and configure the nidaqmx hardware task.

        Checks NI MAX for a colliding saved task, deletes it if found, then
        creates a fresh ``nidaqmx.task.Task``, adds all configured channels,
        and sets up timing.  Optionally starts the task immediately.

        Parameters
        ----------
        start_task : bool, optional
            If ``True``, call ``task.start()`` after configuration.
            Default is ``False`` — the caller is responsible for starting
            (e.g. via trigger or explicit ``task.start()``).

        Raises
        ------
        ValueError
            If the hardware driver coerces the sample rate to a different
            value than requested (some devices only support discrete rates).

        Notes
        -----
        Default is ``False``, unlike the original LDAQ implementation which
        defaulted to ``True``.  Explicit start is safer for triggered
        acquisition workflows.
        """
        # Remove any previously saved task with the same name from NI MAX
        # so nidaqmx does not raise on duplicate task name creation.
        system = nidaqmx.system.System.local()
        if self.task_name in system.tasks.task_names:
            self._delete_task()

        self.task = nidaqmx.task.Task(new_task_name=self.task_name)
        self._add_channels()
        self._setup_task()

        actual_rate = float(self.task._timing.samp_clk_rate)
        requested_rate = float(self.sample_rate)
        if actual_rate != requested_rate:
            self.clear_task()
            raise ValueError(
                f"Sample rate {requested_rate} Hz is not supported by this device. "
                f"The driver coerced it to {actual_rate} Hz. "
                "Use a rate that the device supports."
            )

        if start_task:
            self.task.start()

    def acquire_base(self) -> np.ndarray:
        """Read all available samples from the hardware buffer.

        Drains every sample currently in the on-board FIFO buffer.  The
        result is always a 2-D array with shape ``(n_channels, n_samples)``,
        matching nidaqmx's native channel-major layout.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(n_channels, n_samples)``.  For single-
            channel tasks nidaqmx returns a 1-D list; this method reshapes
            it to ``(1, n_samples)`` so callers always receive a consistent
            shape.

        Notes
        -----
        Uses ``number_of_samples_per_channel=-1`` which corresponds to
        ``nidaqmx.constants.READ_ALL_AVAILABLE``.
        """
        # -1 == nidaqmx.constants.READ_ALL_AVAILABLE — drain the full buffer
        raw = self.task.read(number_of_samples_per_channel=-1)
        data = np.array(raw)

        if data.ndim == 1:
            # Single-channel: nidaqmx returns a flat list; normalise to 2-D
            data = data.reshape(1, -1)

        return data

    def clear_task(self) -> None:
        """Release the hardware task handle.

        Closes the underlying ``nidaqmx.task.Task`` and sets ``self.task``
        to ``None``.  Safe to call on an un-initiated task or multiple times.
        """
        if hasattr(self, "task") and self.task is not None:
            self.task.close()
            self.task = None

    def save(self, clear_task: bool = True) -> None:
        """Save the task to NI MAX.

        If the task has not been initiated, it is initiated first with
        ``start_task=False``.  After saving, optionally closes the task.

        Parameters
        ----------
        clear_task : bool, optional
            If ``True`` (default), call :meth:`clear_task` after saving.

        Notes
        -----
        The LDAQ implementation checked ``hasattr(self, 'Task')`` (capital T)
        which never matched the actual attribute ``self.task`` (lowercase),
        causing spurious re-initiation on every call.  This is fixed here.
        """
        if not (hasattr(self, "task") and self.task is not None):
            self.initiate(start_task=False)

        self.task.save(overwrite_existing_task=True)

        if clear_task:
            self.clear_task()

    def __enter__(self) -> NITask:
        """Enter the runtime context; return ``self``."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the runtime context, releasing hardware resources.

        Calls :meth:`clear_task` unconditionally.  If ``clear_task`` raises,
        a warning is emitted and the exception is swallowed so it does not
        mask any exception that propagated from the ``with`` block body.

        Returns ``None`` so body exceptions are never suppressed.
        """
        try:
            self.clear_task()
        except Exception as exc:
            warnings.warn(str(exc), stacklevel=2)

    # -- Private helpers — channel wiring -----------------------------------

    def _add_channels(self) -> None:
        """Wire all configured channels into the live nidaqmx task."""
        for channel_name in self.channels:
            self._add_channel(channel_name)

    def _add_channel(self, channel_name: str) -> None:
        """Wire a single channel into the live nidaqmx task.

        Determines the channel type from the stored units constant's
        ``__objclass__.__name__``, builds the correct option dict, and
        dispatches to the matching ``add_ai_*`` method.

        Parameters
        ----------
        channel_name : str
            Key in ``self.channels`` for the channel to add.
        """
        cfg = self.channels[channel_name]
        units_const = cfg["units"]
        scale = cfg["scale"]

        # Determine channel mode from the units constant's enum class name.
        # When a custom scale is in play, force the voltage path regardless
        # of original units, because nidaqmx requires FROM_CUSTOM_SCALE on a
        # VoltageAIChannel.
        if scale is not None:
            mode = "VoltageUnits"
        elif hasattr(units_const, "__objclass__"):
            mode = units_const.__objclass__.__name__
        else:
            mode = "VoltageUnits"

        device_ind = cfg["device_ind"]
        channel_ind = cfg["channel_ind"]
        physical_channel = f"{self.device_list[device_ind]}/ai{channel_ind}"

        custom_scale_name = ""
        if scale is not None:
            # Create a linear custom scale in NI MAX.  The pre-scaled input
            # is always voltage; the scaled output uses the user's units string.
            if isinstance(scale, tuple):
                slope, y_intercept = scale[0], scale[1]
            else:
                slope, y_intercept = float(scale), 0

            scale_obj = nidaqmx.Scale.create_lin_scale(
                f"{channel_name}_scale",
                slope=slope,
                y_intercept=y_intercept,
                pre_scaled_units=constants.VoltageUnits.VOLTS,
                scaled_units=cfg["units_str"],
            )
            custom_scale_name = scale_obj.name
            # Persist so the channel config reflects the created scale name
            cfg["custom_scale_name"] = custom_scale_name

        # Build the option dict shared by all channel types
        options: dict = {
            "physical_channel": physical_channel,
            "name_to_assign_to_channel": channel_name,
            "terminal_config": constants.TerminalConfiguration.DEFAULT,
        }

        # Use is not None so that 0.0 is correctly forwarded (LDAQ bug fix)
        if cfg["min_val"] is not None:
            options["min_val"] = cfg["min_val"]
        if cfg["max_val"] is not None:
            options["max_val"] = cfg["max_val"]

        if mode == "AccelUnits":
            options["sensitivity"] = cfg["sensitivity"]
            options["sensitivity_units"] = cfg["sensitivity_units"]
            options["units"] = cfg["units"]
            self.task.ai_channels.add_ai_accel_chan(**options)

        elif mode == "ForceUnits":
            options["sensitivity"] = cfg["sensitivity"]
            options["sensitivity_units"] = cfg["sensitivity_units"]
            options["units"] = cfg["units"]
            self.task.ai_channels.add_ai_force_iepe_chan(**options)

        else:
            # VoltageUnits path — plain voltage or custom-scale channel
            if custom_scale_name:
                options["units"] = constants.VoltageUnits.FROM_CUSTOM_SCALE
                options["custom_scale_name"] = custom_scale_name
            self.task.ai_channels.add_ai_voltage_chan(**options)

    def _setup_task(self) -> None:
        """Configure sample-clock timing on the live nidaqmx task.

        Uses the module-level ``constants`` reference so that tests can patch
        it and verify the correct enum value is forwarded to nidaqmx.
        """
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=constants.AcquisitionType.CONTINUOUS,
        )

    def _delete_task(self) -> None:
        """Delete the saved task with :attr:`task_name` from NI MAX.

        Refreshes the system view so stale cached references are avoided,
        then locates and removes the matching saved task.
        """
        system = nidaqmx.system.System.local()
        for saved_task in system.tasks:
            if saved_task._name == self.task_name:
                saved_task.delete()
                return

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
