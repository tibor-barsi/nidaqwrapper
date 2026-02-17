"""Digital I/O classes for NI-DAQmx digital input and output.

Provides ``DigitalInput`` and ``DigitalOutput`` classes supporting both
on-demand (single-sample) and clocked (continuous/buffered) operation modes.
Digital channels use NI line specification strings rather than device/channel
indices.

Notes
-----
This is a new module — no LDAQ/OpenEOL source to consolidate. The design
follows the patterns from NITask and NITaskOutput for consistency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("nidaqwrapper.digital")

try:
    import nidaqmx
    from nidaqmx import constants

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False


def _expand_port_to_line_range(lines: str) -> str:
    """Expand a port-only spec to an explicit line range.

    ``CHAN_PER_LINE`` requires line-level specifications.  When the user
    passes a port-level spec (e.g. ``'Dev1/port0'``), this helper queries
    the device to determine how many lines the port has and returns an
    explicit line range string (e.g. ``'Dev1/port0/line0:7'``).

    If the spec already contains ``'/line'`` it is returned unchanged.
    """
    if "/line" in lines:
        return lines

    # Port-only spec — query device for line count
    # Extract device name (everything before the first '/')
    parts = lines.split("/")
    dev_name = parts[0]

    system = nidaqmx.system.System.local()
    dev = system.devices[dev_name]

    # Find all DI lines belonging to this port
    port_lines = [
        l.name for l in dev.di_lines if l.name.startswith(lines + "/")
    ]
    if not port_lines:
        # Try DO lines (for DigitalOutput)
        port_lines = [
            l.name for l in dev.do_lines if l.name.startswith(lines + "/")
        ]

    if not port_lines:
        return lines  # No expansion possible, let nidaqmx handle the error

    n_lines = len(port_lines)
    expanded = f"{lines}/line0:{n_lines - 1}"
    logger.debug("Expanded port spec '%s' → '%s'", lines, expanded)
    return expanded


class DigitalInput:
    """Programmatic digital input task supporting on-demand and clocked modes.

    Parameters
    ----------
    task_name : str
        Unique name for this digital input task.
    sample_rate : float or None, optional
        Sample rate in Hz. If ``None`` (default), the task operates in
        on-demand mode (single-sample reads). If a float is provided,
        the task operates in clocked (continuous) mode.

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX.

    Examples
    --------
    On-demand (single-sample) usage:

    >>> di = DigitalInput(task_name='switches')
    >>> di.add_channel('sw1', lines='Dev1/port0/line0:3')
    >>> di.initiate()
    >>> data = di.read()
    >>> di.clear_task()

    Clocked (continuous) usage:

    >>> di = DigitalInput(task_name='fast_di', sample_rate=1000)
    >>> di.add_channel('signals', lines='Dev1/port0/line0:7')
    >>> di.initiate()
    >>> data = di.read_all_available()
    >>> di.clear_task()
    """

    def __init__(self, task_name: str, sample_rate: float | None = None) -> None:
        self.logger = logging.getLogger("nidaqwrapper.digital")
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.mode: str = "on_demand" if sample_rate is None else "clocked"
        self.channels: dict[str, dict[str, Any]] = {}
        self.task: Any = None  # nidaqmx.Task instance after initiate()

        # Discover connected devices
        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]

        # Check for duplicate task name in NI MAX
        existing_tasks = system.tasks.task_names
        if task_name in existing_tasks:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a unique name."
            )

        self.logger.debug(
            "DigitalInput '%s' created (mode=%s, sample_rate=%s)",
            task_name,
            self.mode,
            sample_rate,
        )

    def add_channel(self, channel_name: str, lines: str) -> None:
        """Add a digital input channel by line specification.

        Parameters
        ----------
        channel_name : str
            Unique name for this channel.
        lines : str
            NI-DAQmx line specification (e.g. ``'Dev1/port0/line0'``,
            ``'Dev1/port0/line0:3'``, ``'Dev1/port0'``).

        Raises
        ------
        ValueError
            If ``channel_name`` is already used or ``lines`` are already
            assigned to another channel.
        """
        if channel_name in self.channels:
            raise ValueError(
                f"Channel name '{channel_name}' already exists. "
                "Use a unique name."
            )

        for existing_name, existing_config in self.channels.items():
            if existing_config["lines"] == lines:
                raise ValueError(
                    f"Lines '{lines}' are already assigned to channel "
                    f"'{existing_name}'. Each channel must use unique lines."
                )

        self.channels[channel_name] = {"lines": lines}
        self.logger.debug(
            "Channel '%s' added with lines='%s'", channel_name, lines
        )

    def initiate(self, start_task: bool = True) -> None:
        """Create the underlying nidaqmx task and configure channels/timing.

        Parameters
        ----------
        start_task : bool, optional
            If ``True`` (default) and mode is ``'clocked'``, the task is
            started after configuration. On-demand mode does not start
            the task regardless of this flag.
        """
        self.task = nidaqmx.Task()

        for ch_name, ch_config in self.channels.items():
            lines = _expand_port_to_line_range(ch_config["lines"])
            self.task.di_channels.add_di_chan(
                lines=lines,
                name_to_assign_to_lines=ch_name,
                line_grouping=constants.LineGrouping.CHAN_PER_LINE,
            )

        if self.mode == "clocked":
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=constants.AcquisitionType.CONTINUOUS,
            )
            if start_task:
                self.task.start()

        self.logger.debug(
            "DigitalInput '%s' initiated (mode=%s)", self.task_name, self.mode
        )

    def read(self) -> np.ndarray:
        """Read a single sample from all digital input lines (on-demand).

        Returns
        -------
        numpy.ndarray
            Array of bool-like values, one per line.
        """
        data = self.task.read()

        if isinstance(data, bool):
            return np.array([data])
        return np.array(data)

    def read_all_available(self) -> np.ndarray:
        """Read all available samples from the input buffer (clocked mode).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_lines)``.

        Raises
        ------
        RuntimeError
            If called in on-demand mode.
        """
        if self.mode != "clocked":
            raise RuntimeError(
                "read_all_available() requires clocked mode. "
                "Create DigitalInput with a sample_rate to enable "
                "continuous reading."
            )

        data = self.task.read(
            number_of_samples_per_channel=constants.READ_ALL_AVAILABLE
        )

        if not data:
            return np.array([]).reshape(0, 0)

        arr = np.array(data)

        # nidaqmx returns (n_lines, n_samples) for multi-line;
        # single-line returns a flat list → reshape to (n_samples, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr.T

    def clear_task(self) -> None:
        """Close the underlying nidaqmx task and release hardware resources.

        Safe to call multiple times or when no task has been initiated.
        """
        if self.task is not None:
            try:
                self.task.close()
            except Exception:
                self.logger.warning(
                    "Exception during clear_task() for '%s'", self.task_name,
                    exc_info=True,
                )
            self.task = None
            self.logger.debug("DigitalInput '%s' task cleared", self.task_name)

    def __enter__(self) -> DigitalInput:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.clear_task()


class DigitalOutput:
    """Programmatic digital output task supporting on-demand and clocked modes.

    Parameters
    ----------
    task_name : str
        Unique name for this digital output task.
    sample_rate : float or None, optional
        Sample rate in Hz. If ``None`` (default), the task operates in
        on-demand mode (single-sample writes). If a float is provided,
        the task operates in clocked (continuous) mode.

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX.

    Examples
    --------
    On-demand (single-sample) usage:

    >>> do = DigitalOutput(task_name='leds')
    >>> do.add_channel('led_1', lines='Dev1/port1/line0')
    >>> do.initiate()
    >>> do.write(True)
    >>> do.clear_task()

    Clocked (continuous) usage:

    >>> do = DigitalOutput(task_name='pattern_gen', sample_rate=1000)
    >>> do.add_channel('lines', lines='Dev1/port1/line0:3')
    >>> do.initiate()
    >>> do.write_continuous(data)
    >>> do.clear_task()
    """

    def __init__(self, task_name: str, sample_rate: float | None = None) -> None:
        self.logger = logging.getLogger("nidaqwrapper.digital")
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.mode: str = "on_demand" if sample_rate is None else "clocked"
        self.channels: dict[str, dict[str, Any]] = {}
        self.task: Any = None

        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]

        existing_tasks = system.tasks.task_names
        if task_name in existing_tasks:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a unique name."
            )

        self.logger.debug(
            "DigitalOutput '%s' created (mode=%s, sample_rate=%s)",
            task_name,
            self.mode,
            sample_rate,
        )

    def add_channel(self, channel_name: str, lines: str) -> None:
        """Add a digital output channel by line specification.

        Parameters
        ----------
        channel_name : str
            Unique name for this channel.
        lines : str
            NI-DAQmx line specification (e.g. ``'Dev1/port1/line0'``,
            ``'Dev1/port1/line0:7'``, ``'Dev1/port1'``).

        Raises
        ------
        ValueError
            If ``channel_name`` is already used or ``lines`` are already
            assigned to another channel.
        """
        if channel_name in self.channels:
            raise ValueError(
                f"Channel name '{channel_name}' already exists. "
                "Use a unique name."
            )

        for existing_name, existing_config in self.channels.items():
            if existing_config["lines"] == lines:
                raise ValueError(
                    f"Lines '{lines}' are already assigned to channel "
                    f"'{existing_name}'. Each channel must use unique lines."
                )

        self.channels[channel_name] = {"lines": lines}
        self.logger.debug(
            "Channel '%s' added with lines='%s'", channel_name, lines
        )

    def initiate(self, start_task: bool = True) -> None:
        """Create the underlying nidaqmx task and configure channels/timing.

        Parameters
        ----------
        start_task : bool, optional
            If ``True`` (default) and mode is ``'clocked'``, the task is
            started after configuration. On-demand mode does not start
            the task regardless of this flag.
        """
        self.task = nidaqmx.Task()

        for ch_name, ch_config in self.channels.items():
            lines = _expand_port_to_line_range(ch_config["lines"])
            self.task.do_channels.add_do_chan(
                lines=lines,
                name_to_assign_to_lines=ch_name,
                line_grouping=constants.LineGrouping.CHAN_PER_LINE,
            )

        if self.mode == "clocked":
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=constants.AcquisitionType.CONTINUOUS,
            )
            if start_task:
                self.task.start()

        self.logger.debug(
            "DigitalOutput '%s' initiated (mode=%s)", self.task_name, self.mode
        )

    def write(self, data: bool | int | list | np.ndarray) -> None:
        """Write a single sample to digital output lines (on-demand).

        Parameters
        ----------
        data : bool, int, list, or numpy.ndarray
            Data to write. For a single line, pass a bool or int.
            For multiple lines, pass a list or array of values.
        """
        if isinstance(data, np.ndarray):
            data = [bool(v) for v in data]
        elif isinstance(data, list):
            data = [bool(v) for v in data]
        elif isinstance(data, int) and not isinstance(data, bool):
            data = bool(data)
        self.task.write(data)
        self.logger.debug("DigitalOutput '%s' write complete", self.task_name)

    def write_continuous(self, data: np.ndarray) -> None:
        """Write buffered data to digital output lines (clocked mode).

        Parameters
        ----------
        data : numpy.ndarray
            For multi-line: shape ``(n_samples, n_lines)``, transposed
            internally to ``(n_lines, n_samples)`` for nidaqmx.
            For single-line: 1D array of shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If called in on-demand mode.
        """
        if self.mode != "clocked":
            raise RuntimeError(
                "write_continuous() requires clocked mode. "
                "Create DigitalOutput with a sample_rate to enable "
                "continuous writing."
            )

        if data.ndim == 2:
            # Transpose from (n_samples, n_lines) to (n_lines, n_samples)
            transposed = data.T
            write_data = [[bool(v) for v in row] for row in transposed]
        else:
            write_data = [bool(v) for v in data]

        self.task.write(write_data, auto_start=True)
        self.logger.debug(
            "DigitalOutput '%s' write_continuous complete", self.task_name
        )

    def clear_task(self) -> None:
        """Close the underlying nidaqmx task and release hardware resources.

        Safe to call multiple times or when no task has been initiated.
        """
        if self.task is not None:
            try:
                self.task.close()
            except Exception:
                self.logger.warning(
                    "Exception during clear_task() for '%s'", self.task_name,
                    exc_info=True,
                )
            self.task = None
            self.logger.debug(
                "DigitalOutput '%s' task cleared", self.task_name
            )

    def __enter__(self) -> DigitalOutput:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.clear_task()
