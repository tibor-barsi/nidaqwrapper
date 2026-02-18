"""NITaskOutput — programmatic analog output task configuration.

Provides the :class:`NITaskOutput` class for creating and managing NI-DAQmx
analog output tasks.  Channels are added programmatically (not from NI MAX),
and the output buffer supports continuous regeneration.

Architecture
------------
Direct delegation: the nidaqmx Task is created immediately in the
constructor. :meth:`add_channel` delegates straight to
``task.ao_channels.add_ao_voltage_chan()``.  The nidaqmx Task object is the
single source of truth; no intermediate channel dict is maintained.

Data Format
-----------
The public API accepts signal data in ``(n_samples, n_channels)`` format.
Internally, nidaqmx expects ``(n_channels, n_samples)``, so :meth:`generate`
transposes 2-D arrays automatically.  1-D arrays (single channel) are passed
through unchanged.

Examples
--------
>>> task = NITaskOutput("sig_gen", sample_rate=10000)
>>> task.add_channel("ao_0", device_ind=0, channel_ind=0)
>>> task.start()
>>> task.generate(signal_array)
>>> task.clear_task()

Or as a context manager::

    with NITaskOutput("sig_gen", 10000) as task:
        task.add_channel("ao_0", device_ind=0, channel_ind=0)
        task.start()
        task.generate(signal_array)
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any

import numpy as np

try:
    import nidaqmx
    from nidaqmx import constants

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False


class NITaskOutput:
    """Programmatic analog output task for NI-DAQmx devices.

    The nidaqmx hardware task is created immediately at construction.
    Channels are added via :meth:`add_channel` which delegates directly
    to the nidaqmx task.  Call :meth:`start` to configure timing and
    optionally start output generation.

    Parameters
    ----------
    task_name : str
        Unique name for the output task.  Must not collide with tasks
        already saved in NI MAX.
    sample_rate : float
        Output sample rate in Hz.
    samples_per_channel : int, optional
        Buffer size per channel.  Defaults to ``5 * int(sample_rate)``
        (5 seconds of buffer).

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX.
    """

    def __init__(
        self,
        task_name: str,
        sample_rate: float,
        samples_per_channel: int | None = None,
    ) -> None:
        self.task_name = task_name
        self.sample_rate = sample_rate

        if samples_per_channel is None:
            self.samples_per_channel = 5 * int(sample_rate)
        else:
            self.samples_per_channel = int(samples_per_channel)

        self.sample_mode = constants.AcquisitionType.CONTINUOUS

        # Discover connected devices
        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]

        # Reject duplicate task names in NI MAX before allocating a handle
        if task_name in system.tasks.task_names:
            raise ValueError(
                f"Task '{task_name}' already exists in NI MAX. "
                "Choose a different name."
            )

        # Track original add_channel() parameters for TOML serialisation.
        # The nidaqmx task stores resolved constants; we need the original
        # values (min_val, max_val) to write a human-readable config file.
        self._channel_configs: list[dict[str, Any]] = []

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
        min_val: float = -10.0,
        max_val: float = 10.0,
    ) -> None:
        """Add an analog output voltage channel to the task.

        The channel is configured immediately on the underlying nidaqmx task.

        Parameters
        ----------
        channel_name : str
            Logical name for the channel.
        device_ind : int
            Index into :attr:`device_list` identifying the target device.
        channel_ind : int
            AO channel number on the device (e.g. 0 for ``ao0``).
        min_val : float, optional
            Minimum output voltage, by default -10.0.
        max_val : float, optional
            Maximum output voltage, by default 10.0.

        Raises
        ------
        ValueError
            If ``channel_name`` is a duplicate, the ``(device_ind, channel_ind)``
            pair is already used, or ``device_ind`` is out of range.
        """
        # Duplicate name detection: check what nidaqmx already knows about
        if channel_name in self.task.channel_names:
            raise ValueError(
                f"Channel with duplicate name '{channel_name}' already exists."
            )

        # Reject out-of-range device_ind before building the physical channel string
        if device_ind not in range(len(self.device_list)):
            raise ValueError(
                f"device_ind={device_ind} is out of range. "
                f"Available devices ({len(self.device_list)}): {self.device_list}"
            )

        physical_channel = f"{self.device_list[device_ind]}/ao{channel_ind}"

        # Duplicate physical channel detection: iterate the live task channels
        for ch in self.task.ao_channels:
            if ch.physical_channel.name == physical_channel:
                raise ValueError(
                    f"Physical channel ao{channel_ind} on device "
                    f"'{self.device_list[device_ind]}' is already in use."
                )

        self.task.ao_channels.add_ao_voltage_chan(
            physical_channel=physical_channel,
            name_to_assign_to_channel=channel_name,
            min_val=min_val,
            max_val=max_val,
        )

        # Record original parameters after a successful nidaqmx call so that
        # save_config() can serialise human-readable values.
        self._channel_configs.append({
            "name": channel_name,
            "device_ind": device_ind,
            "channel_ind": channel_ind,
            "min_val": min_val,
            "max_val": max_val,
        })

    # -- Task lifecycle -------------------------------------------------------

    def start(self, start_task: bool = False) -> None:
        """Configure timing and optionally start output generation.

        Configures the sample-clock timing on the nidaqmx task, enables
        buffer regeneration, validates that the driver accepted the
        requested sample rate, and optionally starts the task.

        Parameters
        ----------
        start_task : bool, optional
            If ``True``, call ``task.start()`` after configuration.
            Default is ``False`` — the caller is responsible for starting
            (e.g. via hardware trigger or explicit ``task.start()``).

        Raises
        ------
        ValueError
            If no channels have been added, or if the hardware driver
            coerces the sample rate to a different value than requested
            (some devices only support discrete rates).

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
            samps_per_chan=self.samples_per_channel,
        )

        self.task._out_stream.regen_mode = constants.RegenerationMode.ALLOW_REGENERATION

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

    # -- Signal generation ---------------------------------------------------

    def generate(self, signal: np.ndarray) -> None:
        """Write signal data to the output task.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal data in public format:

            - ``(n_samples, n_channels)`` — multi-channel 2-D array
            - ``(n_samples,)`` — single-channel 1-D array
            - ``(n_samples, 1)`` — single-channel 2-D array

            2-D arrays are transposed to ``(n_channels, n_samples)``
            internally before writing.  A C-contiguous copy is made to
            satisfy the nidaqmx C layer requirement (``data.T`` returns
            Fortran-order).
        """
        if signal.ndim == 2:
            data = np.ascontiguousarray(signal.T)
            # nidaqmx requires a 1-D array for single-channel tasks;
            # a (1, N) 2-D array triggers a channel-count validation error.
            if data.shape[0] == 1:
                data = data[0]
        else:
            data = signal

        self.task.write(data, auto_start=True)

    # -- Cleanup -------------------------------------------------------------

    def clear_task(self) -> None:
        """Close the nidaqmx task and release hardware resources.

        Safe to call multiple times or when the task was never initiated.
        """
        if hasattr(self, "task") and self.task is not None:
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
        hardware.  Device names are replaced by short aliases in
        ``[devices]`` so that changing chassis enumeration only requires
        editing one line per module.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path.  The file is created or overwritten.

        Notes
        -----
        TOML is generated with simple string formatting — no third-party
        library is required for writing.  ``min_val`` and ``max_val`` are
        always written (they always have values: the defaults are -10.0
        and 10.0 respectively).

        Examples
        --------
        >>> task.save_config("/tmp/signal_gen.toml")
        """
        # Build alias → device_name map for every unique device used.
        # Aliases are assigned in the order channels were added: dev0, dev1 …
        device_alias: dict[int, str] = {}
        for cfg in self._channel_configs:
            ind = cfg["device_ind"]
            if ind not in device_alias:
                device_alias[ind] = f"dev{len(device_alias)}"

        lines: list[str] = []

        # [task] section
        lines.append("[task]")
        lines.append(f'name = "{self.task_name}"')
        lines.append(f"sample_rate = {self.sample_rate}")
        lines.append(f"samples_per_channel = {self.samples_per_channel}")
        lines.append('type = "output"')
        lines.append("")

        # [devices] section
        lines.append("[devices]")
        for ind, alias in device_alias.items():
            device_name = self.device_list[ind]
            lines.append(f'{alias} = "{device_name}"')
        lines.append("")

        # [[channels]] entries
        for cfg in self._channel_configs:
            alias = device_alias[cfg["device_ind"]]
            lines.append("[[channels]]")
            lines.append(f'name = "{cfg["name"]}"')
            lines.append(f'device = "{alias}"')
            lines.append(f'channel = {cfg["channel_ind"]}')
            lines.append(f"min_val = {cfg['min_val']}")
            lines.append(f"max_val = {cfg['max_val']}")
            lines.append("")

        pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> NITaskOutput:
        """Create an :class:`NITaskOutput` from a TOML configuration file.

        Reads the TOML file produced by :meth:`save_config`, constructs
        a new task, and calls :meth:`add_channel` for every ``[[channels]]``
        entry.  Device aliases are resolved to system device indices.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file.

        Returns
        -------
        NITaskOutput
            A fully configured task (channels added, not yet started).

        Raises
        ------
        ValueError
            If the ``[task]`` or ``[devices]`` section is absent, if a
            channel references an unknown device alias, or if a device
            name is not present on the current system.
        tomllib.TOMLDecodeError
            On syntactically invalid TOML (propagated from the parser).

        Examples
        --------
        >>> task = NITaskOutput.from_config("/tmp/signal_gen.toml")
        >>> task.start()
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
        if "devices" not in data:
            raise ValueError(
                "TOML file is missing required [devices] section."
            )

        task_section = data["task"]
        alias_to_name: dict[str, str] = data["devices"]

        samples_per_channel = task_section.get("samples_per_channel")

        task = cls(
            task_section["name"],
            sample_rate=task_section["sample_rate"],
            samples_per_channel=samples_per_channel,
        )

        # Build name → index map from the live device list
        name_to_ind: dict[str, int] = {
            name: ind for ind, name in enumerate(task.device_list)
        }

        for ch in data.get("channels", []):
            alias = ch["device"]
            if alias not in alias_to_name:
                raise ValueError(
                    f"Channel '{ch['name']}' references unknown device alias "
                    f"'{alias}'. Available aliases: {list(alias_to_name)}"
                )

            device_name = alias_to_name[alias]
            if device_name not in name_to_ind:
                raise ValueError(
                    f"Device '{device_name}' (alias '{alias}') was not found "
                    f"in the system. Available devices: {task.device_list}"
                )

            device_ind = name_to_ind[device_name]

            task.add_channel(
                channel_name=ch["name"],
                device_ind=device_ind,
                channel_ind=ch["channel"],
                min_val=ch.get("min_val", -10.0),
                max_val=ch.get("max_val", 10.0),
            )

        return task

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> NITaskOutput:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the context manager, ensuring task cleanup.

        Calls :meth:`clear_task` unconditionally.  If ``clear_task`` raises,
        a warning is emitted and the exception is swallowed so it does not
        mask any exception that propagated from the ``with`` block body.

        Returns ``None`` so body exceptions are never suppressed.
        """
        try:
            self.clear_task()
        except Exception as exc:
            warnings.warn(str(exc), stacklevel=2)
