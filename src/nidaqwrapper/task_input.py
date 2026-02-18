"""Analog input task configuration for NI-DAQmx devices.

Provides the ``NITask`` class for programmatic creation and management
of analog input acquisition tasks.  Supports accelerometer (IEPE),
force (IEPE), and voltage channels with optional linear custom scales.

Architecture
------------
Direct delegation: the nidaqmx Task is created immediately in the
constructor. :meth:`add_channel` delegates straight to
``task.ai_channels.add_ai_*_chan()``.  The nidaqmx Task object is the
single source of truth; no intermediate channel dict is maintained.

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

    The nidaqmx hardware task is created immediately at construction.
    Channels are added via :meth:`add_channel` which delegates directly
    to the nidaqmx task.  Call :meth:`start` to configure timing and
    optionally start acquisition.

    Parameters
    ----------
    task_name : str
        Unique name for this task.  Must not collide with an existing
        task in NI MAX.
    sample_rate : float
        Sampling rate in Hz.

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX.
    RuntimeError
        If nidaqmx is not installed.

    Examples
    --------
    >>> task = NITask("vibration_test", sample_rate=25600)
    >>> task.add_channel("accel_x", device_ind=0, channel_ind=0,
    ...                  sensitivity=100.0, sensitivity_units="mV/g",
    ...                  units="g")
    >>> task.start()
    """

    def __init__(
        self,
        task_name: str,
        sample_rate: float,
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

        # Guard against duplicate task names before allocating a handle
        if task_name in system.tasks.task_names:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a different name or delete the existing task first."
            )

        self.sample_mode = constants.AcquisitionType.CONTINUOUS

        # Create the nidaqmx task immediately — it is the single source of truth
        self.task = nidaqmx.task.Task(new_task_name=task_name)

    # -- Introspection properties -------------------------------------------

    @property
    def channel_list(self) -> list[str]:
        """List of channel names registered with the nidaqmx task."""
        return list(self.task.channel_names)

    @property
    def number_of_ch(self) -> int:
        """Number of channels registered with the nidaqmx task."""
        return len(self.task.channel_names)

    # -- Channel configuration -----------------------------------------------

    def add_channel(
        self,
        channel_name: str,
        device_ind: int,
        channel_ind: int,
        sensitivity: float | None = None,
        sensitivity_units: str | None = None,
        units: str | None = None,
        scale: float | tuple[float, float] | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """Add an analog input channel to this task.

        The channel is configured immediately on the underlying nidaqmx
        task.  Channel type is determined from the units constant's
        ``__objclass__.__name__`` (``AccelUnits``, ``ForceUnits``, or
        ``VoltageUnits``).  Providing a *scale* forces the voltage path
        regardless of units.

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
        units : str, optional
            Key into the ``UNITS`` dict for output measurement units
            (e.g. ``'g'``, ``'N'``, ``'V'``).  Always required.
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
        # -- Basic validation -----------------------------------------------
        if units is None:
            raise ValueError(
                "units must be specified. "
                f"Valid units: {list(UNITS.keys())}"
            )

        # Duplicate name detection: check what nidaqmx already knows about
        if channel_name in self.task.channel_names:
            raise ValueError(
                f"Channel name '{channel_name}' already exists in this task."
            )

        if device_ind not in range(len(self.device_list)):
            raise ValueError(
                f"device_ind {device_ind} is out of range. "
                f"Available devices: {self.device_list}"
            )

        # Duplicate physical channel detection: iterate the live task channels
        physical_channel = (
            f"{self.device_list[device_ind]}/ai{channel_ind}"
        )
        for ch in self.task.ai_channels:
            if ch.physical_channel.name == physical_channel:
                raise ValueError(
                    f"Physical channel ai{channel_ind} on device "
                    f"'{self.device_list[device_ind]}' is already in use."
                )

        # -- Scale type validation ------------------------------------------
        if scale is not None and not isinstance(scale, (int, float, tuple)):
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

        # -- Resolve nidaqmx constants from UNITS dict ----------------------
        resolved_units = UNITS[units] if units in UNITS else units
        resolved_sens_units = (
            UNITS[sensitivity_units]
            if sensitivity_units is not None and sensitivity_units in UNITS
            else sensitivity_units
        )

        # -- Determine channel mode from the units constant's enum class ----
        # When a custom scale is provided, force the voltage path — nidaqmx
        # requires FROM_CUSTOM_SCALE on a VoltageAIChannel.
        if scale is not None:
            mode = "VoltageUnits"
        elif hasattr(resolved_units, "__objclass__"):
            mode = resolved_units.__objclass__.__name__
        else:
            mode = "VoltageUnits"

        # -- Sensitivity required for non-voltage, non-scale channels -------
        if scale is None and mode != "VoltageUnits":
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

        # -- Create linear custom scale in NI MAX when requested ------------
        custom_scale_name = ""
        if scale is not None:
            if isinstance(scale, tuple):
                slope, y_intercept = float(scale[0]), float(scale[1])
            else:
                slope, y_intercept = float(scale), 0.0

            # units_str is the original string, used as the scaled output unit
            scale_obj = nidaqmx.Scale.create_lin_scale(
                f"{channel_name}_scale",
                slope=slope,
                y_intercept=y_intercept,
                pre_scaled_units=constants.VoltageUnits.VOLTS,
                scaled_units=units,
            )
            custom_scale_name = scale_obj.name

        # -- Build the options dict shared by all channel types -------------
        options: dict[str, Any] = {
            "physical_channel": physical_channel,
            "name_to_assign_to_channel": channel_name,
            "terminal_config": constants.TerminalConfiguration.DEFAULT,
        }

        # Use is not None so that 0.0 is correctly forwarded (LDAQ bug fix)
        if min_val is not None:
            options["min_val"] = min_val
        if max_val is not None:
            options["max_val"] = max_val

        # -- Dispatch to the correct nidaqmx channel factory ---------------
        if mode == "AccelUnits":
            options["sensitivity"] = sensitivity
            options["sensitivity_units"] = resolved_sens_units
            options["units"] = resolved_units
            self.task.ai_channels.add_ai_accel_chan(**options)

        elif mode == "ForceUnits":
            options["sensitivity"] = sensitivity
            options["sensitivity_units"] = resolved_sens_units
            options["units"] = resolved_units
            self.task.ai_channels.add_ai_force_iepe_chan(**options)

        else:
            # VoltageUnits path — plain voltage or custom-scale channel
            if custom_scale_name:
                options["units"] = constants.VoltageUnits.FROM_CUSTOM_SCALE
                options["custom_scale_name"] = custom_scale_name
            else:
                options["units"] = resolved_units
            self.task.ai_channels.add_ai_voltage_chan(**options)

    # -- Task lifecycle ------------------------------------------------------

    def start(self, start_task: bool = False) -> None:
        """Configure timing and optionally start acquisition.

        Configures the sample-clock timing on the nidaqmx task, validates
        that the driver accepted the requested sample rate, and optionally
        starts the task.

        Parameters
        ----------
        start_task : bool, optional
            If ``True``, call ``task.start()`` after configuration.
            Default is ``False`` — the caller is responsible for starting
            (e.g. via hardware trigger or explicit ``task.start()``).

        Raises
        ------
        ValueError
            If the hardware driver coerces the sample rate to a different
            value than requested (some devices only support discrete rates).

        Notes
        -----
        Unlike the old ``initiate()``, a rate mismatch does NOT close the
        task handle.  The task remains valid and can be reconfigured or
        closed by the caller.
        """
        if not self.task.channel_names:
            raise ValueError(
                "Cannot start: no channels have been added to this task. "
                "Call add_channel() before start()."
            )

        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=constants.AcquisitionType.CONTINUOUS,
        )

        actual_rate = float(self.task.timing.samp_clk_rate)
        requested_rate = float(self.sample_rate)
        if actual_rate != requested_rate:
            raise ValueError(
                f"Sample rate {requested_rate} Hz is not supported by this "
                f"device. The driver coerced it to {actual_rate} Hz. "
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
        ``nidaqmx.constants.READ_ALL_AVAILABLE`` — drain the full buffer.
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
        to ``None``.  Safe to call on an already-cleared task or multiple
        times.
        """
        if hasattr(self, "task") and self.task is not None:
            try:
                self.task.close()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self.task = None

    def save(self, clear_task: bool = True) -> None:
        """Save the task to NI MAX.

        The task always exists in the direct-delegation architecture, so
        this method calls ``task.save()`` directly without auto-initiating.
        After saving, optionally closes the task.

        Parameters
        ----------
        clear_task : bool, optional
            If ``True`` (default), call :meth:`clear_task` after saving.
        """
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
