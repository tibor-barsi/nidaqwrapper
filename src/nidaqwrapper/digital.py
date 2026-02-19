"""Digital I/O classes for NI-DAQmx digital input and output.

Provides ``DITask`` and ``DOTask`` classes supporting both
on-demand (single-sample) and clocked (continuous/buffered) operation modes.
Digital channels use NI line specification strings rather than device/channel
indices.

Architecture
------------
Direct delegation: the nidaqmx Task is created immediately in the constructor.
:meth:`add_channel` delegates straight to ``task.di_channels.add_di_chan()``
(or ``do_channels.add_do_chan()``). The nidaqmx Task object is the single
source of truth; no intermediate channel dict is maintained.

Notes
-----
This is a new module — no LDAQ/OpenEOL source to consolidate. The design
follows the patterns from AITask and AOTask for consistency.
"""

from __future__ import annotations

import pathlib
import warnings
from datetime import datetime
from typing import Any

import numpy as np

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
        # Try DO lines (for DOTask)
        port_lines = [
            l.name for l in dev.do_lines if l.name.startswith(lines + "/")
        ]

    if not port_lines:
        return lines  # No expansion possible, let nidaqmx handle the error

    n_lines = len(port_lines)
    return f"{lines}/line0:{n_lines - 1}"


class DITask:
    """Programmatic digital input task supporting on-demand and clocked modes.

    The nidaqmx hardware task is created immediately at construction.
    Channels are added via :meth:`add_channel` which delegates directly to the
    nidaqmx task. Call :meth:`start` to configure timing and optionally start
    acquisition.

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

    >>> di = DITask(task_name='switches')
    >>> di.add_channel('sw1', lines='Dev1/port0/line0:3')
    >>> di.start()
    >>> data = di.read()
    >>> di.clear_task()

    Clocked (continuous) usage:

    >>> di = DITask(task_name='fast_di', sample_rate=1000)
    >>> di.add_channel('signals', lines='Dev1/port0/line0:7')
    >>> di.start(start_task=True)
    >>> data = di.acquire()
    >>> di.clear_task()
    """

    def __init__(self, task_name: str, sample_rate: float | None = None) -> None:
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.mode: str = "on_demand" if sample_rate is None else "clocked"

        # Discover connected devices
        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]

        # Check for duplicate task name in NI MAX before allocating a handle
        existing_tasks = system.tasks.task_names
        if task_name in existing_tasks:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a unique name."
            )

        # Track original add_channel() parameters for TOML serialisation.
        # The nidaqmx task stores resolved channel objects; we need the
        # original human-readable values to write a config file.
        self._channel_configs: list[dict[str, Any]] = []

        # Create the nidaqmx task immediately — it is the single source of truth
        self.task = nidaqmx.task.Task(new_task_name=task_name)

        # Track ownership — False when task is externally provided
        self._owns_task: bool = True

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

    def add_channel(self, channel_name: str, lines: str) -> None:
        """Add a digital input channel by line specification.

        Delegates directly to ``task.di_channels.add_di_chan()`` on the
        underlying nidaqmx task.

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
        RuntimeError
            If this task was created via :meth:`from_task` (externally provided).
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot add channels to an externally-provided task. "
                "Configure channels on the nidaqmx.Task before calling from_task()."
            )

        # Duplicate name detection: check what nidaqmx already knows about
        if channel_name in self.task.channel_names:
            raise ValueError(
                f"Channel name '{channel_name}' already exists. "
                "Use a unique name."
            )

        # Expand port-only spec to explicit line range before duplicate check
        expanded_lines = _expand_port_to_line_range(lines)

        # Duplicate lines detection: iterate the live task channels
        for ch in self.task.di_channels:
            if ch.physical_channel.name == expanded_lines:
                raise ValueError(
                    f"Lines '{lines}' are already assigned to channel "
                    f"'{ch.name}'. Each channel must use unique lines."
                )

        self.task.di_channels.add_di_chan(
            lines=expanded_lines,
            name_to_assign_to_lines=channel_name,
            line_grouping=constants.LineGrouping.CHAN_PER_LINE,
        )

        # Record original (pre-expansion) lines for human-readable TOML output
        self._channel_configs.append({"name": channel_name, "lines": lines})

    # -- Task lifecycle -------------------------------------------------------

    def start(self, start_task: bool = False) -> None:
        """Configure timing and optionally start acquisition.

        Parameters
        ----------
        start_task : bool, optional
            If ``True`` and mode is ``'clocked'``, the task is started after
            timing configuration. On-demand mode never starts the task
            regardless of this flag. Default is ``False``.

        Raises
        ------
        ValueError
            If no channels have been added to the task.
        RuntimeError
            If this task was created via :meth:`from_task` (externally provided).
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot start an externally-provided task. "
                "Start the nidaqmx.Task directly or pass an already-started task to from_task()."
            )

        if not self.task.channel_names:
            raise ValueError(
                "Cannot start: no channels have been added to this task. "
                "Call add_channel() before start()."
            )

        if self.mode == "clocked":
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=constants.AcquisitionType.CONTINUOUS,
            )
            if start_task:
                self.task.start()

    # -- Data acquisition ----------------------------------------------------

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

    def acquire(self, n_samples: int | None = None) -> np.ndarray:
        """Read samples from the input buffer (clocked mode).

        Parameters
        ----------
        n_samples : int, optional
            Number of samples per channel to read.  If provided, the call
            **blocks** until exactly *n_samples* are available — suitable
            for scripts and notebooks.  If ``None`` (default), drains every
            sample currently in the buffer without blocking
            (``READ_ALL_AVAILABLE``) — suitable for acquisition loops.

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
                "acquire() requires clocked mode. "
                "Create DITask with a sample_rate to enable "
                "continuous reading."
            )

        count = constants.READ_ALL_AVAILABLE if n_samples is None else n_samples
        data = self.task.read(number_of_samples_per_channel=count)

        if not data:
            return np.array([]).reshape(0, 0)

        arr = np.array(data)

        # nidaqmx returns (n_lines, n_samples) for multi-line;
        # single-line returns a flat list → reshape to (n_samples, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr.T

    # -- Cleanup -------------------------------------------------------------

    def clear_task(self) -> None:
        """Close the underlying nidaqmx task and release hardware resources.

        Safe to call multiple times or when no task has been initiated.

        Notes
        -----
        If this task was created via :meth:`from_task`, the underlying
        nidaqmx.Task is NOT closed. The caller retains ownership and must
        close it manually when done.
        """
        if hasattr(self, "task") and self.task is not None:
            if not self._owns_task:
                warnings.warn(
                    "Task was created externally — not closing. "
                    "Call task.close() when done.",
                    stacklevel=2,
                )
                self.task = None
                return

            try:
                self.task.close()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self.task = None

    # -- TOML config persistence ---------------------------------------------

    def save_config(self, path: str | pathlib.Path) -> None:
        """Serialise the task configuration to a TOML file.

        Writes a human-readable TOML file that can be loaded back with
        :meth:`from_config` to recreate the same task on any compatible
        hardware.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path. The file is created or overwritten.

        Notes
        -----
        TOML is generated with simple string formatting — no third-party
        library is required for writing. The ``sample_rate`` key is only
        written for clocked-mode tasks; on-demand tasks omit it entirely
        so that ``from_config`` correctly restores the mode.
        """
        lines: list[str] = []

        # Header comment with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"# Generated by nidaqwrapper 0.1.0 on {timestamp}")
        lines.append("")

        # [task] section
        lines.append("[task]")
        lines.append(f'name = "{self.task_name}"')
        lines.append('type = "digital_input"')
        if self.sample_rate is not None:
            lines.append(f"sample_rate = {self.sample_rate}")
        lines.append("")

        # [[channels]] entries — use original (pre-expansion) lines string
        for cfg in self._channel_configs:
            lines.append("[[channels]]")
            lines.append(f'name = "{cfg["name"]}"')
            lines.append(f'lines = "{cfg["lines"]}"')
            lines.append("")

        pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> DITask:
        """Create a :class:`DITask` from a TOML configuration file.

        Reads the TOML file produced by :meth:`save_config`, constructs a new
        task, and calls :meth:`add_channel` for every ``[[channels]]`` entry.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file produced by :meth:`save_config`.

        Returns
        -------
        DITask
            A fully configured task (channels added, not yet started).

        Raises
        ------
        ValueError
            If the ``[task]`` section is absent.
        tomllib.TOMLDecodeError
            On syntactically invalid TOML (propagated from the parser).
        """
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as fh:
            data = tomllib.load(fh)

        if "task" not in data:
            raise ValueError(
                "TOML file is missing required [task] section."
            )

        task_section = data["task"]
        sample_rate = task_section.get("sample_rate", None)

        task = cls(task_section["name"], sample_rate=sample_rate)

        for ch in data.get("channels", []):
            task.add_channel(channel_name=ch["name"], lines=ch["lines"])

        return task

    @classmethod
    def from_task(cls, task: nidaqmx.task.Task) -> DITask:
        """Wrap an externally-created nidaqmx.Task in a DITask.

        This classmethod provides an escape hatch for advanced users who need
        to configure task properties not exposed by the wrapper API. The task
        must already have DI channels configured before calling this method.

        Parameters
        ----------
        task : nidaqmx.task.Task
            A pre-configured nidaqmx.Task with at least one DI channel.

        Returns
        -------
        DITask
            A DITask instance wrapping the provided task. The wrapper does NOT
            take ownership of the task; the caller must close it manually.

        Raises
        ------
        ValueError
            If the task has no DI channels configured.

        Warnings
        --------
        Emits a UserWarning if the task is already running.

        Notes
        -----
        When created via this method:

        - :meth:`add_channel` and :meth:`start` will raise RuntimeError
        - :meth:`clear_task` and :meth:`__exit__` will NOT close the task
        - The caller retains full ownership and must manage the task lifecycle

        Examples
        --------
        >>> import nidaqmx
        >>> task = nidaqmx.Task()
        >>> task.di_channels.add_di_chan("Dev1/port0/line0:3")
        >>> task.timing.cfg_samp_clk_timing(rate=1000)
        >>> di = DITask.from_task(task)
        >>> data = di.acquire()
        >>> task.close()  # Caller must close
        """
        if len(task.di_channels) == 0:
            raise ValueError("Task has no DI channels.")

        # Warn if task is already running (check if task.is_task_done() exists)
        try:
            if hasattr(task, "is_task_done") and not task.is_task_done():
                warnings.warn("Task is already running.", stacklevel=2)
        except Exception:
            # Suppress errors from checking task state
            pass

        # Create instance without calling __init__ (bypass constructor checks)
        instance = object.__new__(cls)

        # Populate attributes from the live task
        instance.task = task
        instance.task_name = task.name
        # channel_list and number_of_ch are properties that read from task

        # Detect mode from timing configuration
        try:
            sample_rate = task.timing.samp_clk_rate
            if sample_rate and sample_rate > 0:
                instance.sample_rate = sample_rate
                instance.mode = "clocked"
            else:
                instance.sample_rate = None
                instance.mode = "on_demand"
        except Exception:
            # No timing configured — assume on-demand
            instance.sample_rate = None
            instance.mode = "on_demand"

        # Discover devices
        system = nidaqmx.system.System.local()
        instance.device_list = [dev.name for dev in system.devices]

        # Initialize _channel_configs as empty (can't reconstruct original params)
        instance._channel_configs = []

        # Mark as externally owned
        instance._owns_task = False

        return instance

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> DITask:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, calling clear_task() unconditionally."""
        self.clear_task()


class DOTask:
    """Programmatic digital output task supporting on-demand and clocked modes.

    The nidaqmx hardware task is created immediately at construction.
    Channels are added via :meth:`add_channel` which delegates directly to the
    nidaqmx task. Call :meth:`start` to configure timing and optionally begin
    output.

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

    >>> do = DOTask(task_name='leds')
    >>> do.add_channel('led_1', lines='Dev1/port1/line0')
    >>> do.start()
    >>> do.write(True)
    >>> do.clear_task()

    Clocked (continuous) usage:

    >>> do = DOTask(task_name='pattern_gen', sample_rate=1000)
    >>> do.add_channel('lines', lines='Dev1/port1/line0:3')
    >>> do.start(start_task=True)
    >>> do.write_continuous(data)
    >>> do.clear_task()
    """

    def __init__(self, task_name: str, sample_rate: float | None = None) -> None:
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.mode: str = "on_demand" if sample_rate is None else "clocked"

        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]

        existing_tasks = system.tasks.task_names
        if task_name in existing_tasks:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a unique name."
            )

        # Track original add_channel() parameters for TOML serialisation
        self._channel_configs: list[dict[str, Any]] = []

        # Create the nidaqmx task immediately — it is the single source of truth
        self.task = nidaqmx.task.Task(new_task_name=task_name)

        # Track ownership — False when task is externally provided
        self._owns_task: bool = True

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

    def add_channel(self, channel_name: str, lines: str) -> None:
        """Add a digital output channel by line specification.

        Delegates directly to ``task.do_channels.add_do_chan()`` on the
        underlying nidaqmx task.

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
        RuntimeError
            If this task was created via :meth:`from_task` (externally provided).
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot add channels to an externally-provided task. "
                "Configure channels on the nidaqmx.Task before calling from_task()."
            )

        # Duplicate name detection: check what nidaqmx already knows about
        if channel_name in self.task.channel_names:
            raise ValueError(
                f"Channel name '{channel_name}' already exists. "
                "Use a unique name."
            )

        # Expand port-only spec to explicit line range before duplicate check
        expanded_lines = _expand_port_to_line_range(lines)

        # Duplicate lines detection: iterate the live task channels
        for ch in self.task.do_channels:
            if ch.physical_channel.name == expanded_lines:
                raise ValueError(
                    f"Lines '{lines}' are already assigned to channel "
                    f"'{ch.name}'. Each channel must use unique lines."
                )

        self.task.do_channels.add_do_chan(
            lines=expanded_lines,
            name_to_assign_to_lines=channel_name,
            line_grouping=constants.LineGrouping.CHAN_PER_LINE,
        )

        # Record original (pre-expansion) lines for human-readable TOML output
        self._channel_configs.append({"name": channel_name, "lines": lines})

    # -- Task lifecycle -------------------------------------------------------

    def start(self, start_task: bool = False) -> None:
        """Configure timing and optionally start output generation.

        Parameters
        ----------
        start_task : bool, optional
            If ``True`` and mode is ``'clocked'``, the task is started after
            timing configuration. On-demand mode never starts the task
            regardless of this flag. Default is ``False``.

        Raises
        ------
        ValueError
            If no channels have been added to the task.
        RuntimeError
            If this task was created via :meth:`from_task` (externally provided).
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot start an externally-provided task. "
                "Start the nidaqmx.Task directly or pass an already-started task to from_task()."
            )

        if not self.task.channel_names:
            raise ValueError(
                "Cannot start: no channels have been added to this task. "
                "Call add_channel() before start()."
            )

        if self.mode == "clocked":
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=constants.AcquisitionType.CONTINUOUS,
            )
            if start_task:
                self.task.start()

    # -- Signal output -------------------------------------------------------

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
                "Create DOTask with a sample_rate to enable "
                "continuous writing."
            )

        if data.ndim == 2:
            # Transpose from (n_samples, n_lines) to (n_lines, n_samples)
            transposed = data.T
            write_data = [[bool(v) for v in row] for row in transposed]
        else:
            write_data = [bool(v) for v in data]

        self.task.write(write_data, auto_start=True)

    # -- Cleanup -------------------------------------------------------------

    def clear_task(self) -> None:
        """Close the underlying nidaqmx task and release hardware resources.

        Safe to call multiple times or when no task has been initiated.

        Notes
        -----
        If this task was created via :meth:`from_task`, the underlying
        nidaqmx.Task is NOT closed. The caller retains ownership and must
        close it manually when done.
        """
        if hasattr(self, "task") and self.task is not None:
            if not self._owns_task:
                warnings.warn(
                    "Task was created externally — not closing. "
                    "Call task.close() when done.",
                    stacklevel=2,
                )
                self.task = None
                return

            try:
                self.task.close()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self.task = None

    # -- TOML config persistence ---------------------------------------------

    def save_config(self, path: str | pathlib.Path) -> None:
        """Serialise the task configuration to a TOML file.

        Writes a human-readable TOML file that can be loaded back with
        :meth:`from_config` to recreate the same task on any compatible
        hardware.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path. The file is created or overwritten.

        Notes
        -----
        TOML is generated with simple string formatting — no third-party
        library is required for writing. The ``sample_rate`` key is only
        written for clocked-mode tasks; on-demand tasks omit it entirely
        so that ``from_config`` correctly restores the mode.
        """
        lines: list[str] = []

        # Header comment with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"# Generated by nidaqwrapper 0.1.0 on {timestamp}")
        lines.append("")

        # [task] section
        lines.append("[task]")
        lines.append(f'name = "{self.task_name}"')
        lines.append('type = "digital_output"')
        if self.sample_rate is not None:
            lines.append(f"sample_rate = {self.sample_rate}")
        lines.append("")

        # [[channels]] entries — use original (pre-expansion) lines string
        for cfg in self._channel_configs:
            lines.append("[[channels]]")
            lines.append(f'name = "{cfg["name"]}"')
            lines.append(f'lines = "{cfg["lines"]}"')
            lines.append("")

        pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> DOTask:
        """Create a :class:`DOTask` from a TOML configuration file.

        Reads the TOML file produced by :meth:`save_config`, constructs a new
        task, and calls :meth:`add_channel` for every ``[[channels]]`` entry.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file produced by :meth:`save_config`.

        Returns
        -------
        DOTask
            A fully configured task (channels added, not yet started).

        Raises
        ------
        ValueError
            If the ``[task]`` section is absent.
        tomllib.TOMLDecodeError
            On syntactically invalid TOML (propagated from the parser).
        """
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as fh:
            data = tomllib.load(fh)

        if "task" not in data:
            raise ValueError(
                "TOML file is missing required [task] section."
            )

        task_section = data["task"]
        sample_rate = task_section.get("sample_rate", None)

        task = cls(task_section["name"], sample_rate=sample_rate)

        for ch in data.get("channels", []):
            task.add_channel(channel_name=ch["name"], lines=ch["lines"])

        return task

    @classmethod
    def from_task(cls, task: nidaqmx.task.Task) -> DOTask:
        """Wrap an externally-created nidaqmx.Task in a DOTask.

        This classmethod provides an escape hatch for advanced users who need
        to configure task properties not exposed by the wrapper API. The task
        must already have DO channels configured before calling this method.

        Parameters
        ----------
        task : nidaqmx.task.Task
            A pre-configured nidaqmx.Task with at least one DO channel.

        Returns
        -------
        DOTask
            A DOTask instance wrapping the provided task. The wrapper does NOT
            take ownership of the task; the caller must close it manually.

        Raises
        ------
        ValueError
            If the task has no DO channels configured.

        Warnings
        --------
        Emits a UserWarning if the task is already running.

        Notes
        -----
        When created via this method:

        - :meth:`add_channel` and :meth:`start` will raise RuntimeError
        - :meth:`clear_task` and :meth:`__exit__` will NOT close the task
        - The caller retains full ownership and must manage the task lifecycle

        Examples
        --------
        >>> import nidaqmx
        >>> task = nidaqmx.Task()
        >>> task.do_channels.add_do_chan("Dev1/port1/line0:3")
        >>> task.timing.cfg_samp_clk_timing(rate=1000)
        >>> do = DOTask.from_task(task)
        >>> do.write_continuous(data)
        >>> task.close()  # Caller must close
        """
        if len(task.do_channels) == 0:
            raise ValueError("Task has no DO channels.")

        # Warn if task is already running
        try:
            if hasattr(task, "is_task_done") and not task.is_task_done():
                warnings.warn("Task is already running.", stacklevel=2)
        except Exception:
            # Suppress errors from checking task state
            pass

        # Create instance without calling __init__
        instance = object.__new__(cls)

        # Populate attributes from the live task
        instance.task = task
        instance.task_name = task.name
        # channel_list and number_of_ch are properties that read from task

        # Detect mode from timing configuration
        try:
            sample_rate = task.timing.samp_clk_rate
            if sample_rate and sample_rate > 0:
                instance.sample_rate = sample_rate
                instance.mode = "clocked"
            else:
                instance.sample_rate = None
                instance.mode = "on_demand"
        except Exception:
            # No timing configured — assume on-demand
            instance.sample_rate = None
            instance.mode = "on_demand"

        # Discover devices
        system = nidaqmx.system.System.local()
        instance.device_list = [dev.name for dev in system.devices]

        # Initialize _channel_configs as empty
        instance._channel_configs = []

        # Mark as externally owned
        instance._owns_task = False

        return instance

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> DOTask:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, calling clear_task() unconditionally."""
        self.clear_task()
