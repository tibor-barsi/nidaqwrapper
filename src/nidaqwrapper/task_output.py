"""NITaskOutput — programmatic analog output task configuration.

Provides the :class:`NITaskOutput` class for creating and managing NI-DAQmx
analog output tasks.  Channels are added programmatically (not from NI MAX),
and the output buffer supports continuous regeneration.

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
>>> task.initiate()
>>> task.generate(signal_array)
>>> task.clear_task()

Or as a context manager::

    with NITaskOutput("sig_gen", 10000) as task:
        task.add_channel("ao_0", device_ind=0, channel_ind=0)
        task.initiate()
        task.generate(signal_array)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger("nidaqwrapper.task")

try:
    import nidaqmx
    from nidaqmx import constants

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False

if TYPE_CHECKING:
    import nidaqmx
    from nidaqmx import constants


class NITaskOutput:
    """Programmatic analog output task for NI-DAQmx devices.

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
        self.channels: dict[str, dict] = {}
        self.task: nidaqmx.task.Task | None = None

        if samples_per_channel is None:
            self.samples_per_channel = 5 * int(sample_rate)
        else:
            self.samples_per_channel = int(samples_per_channel)

        self.sample_mode = constants.AcquisitionType.CONTINUOUS

        # Discover connected devices
        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]

        # Reject duplicate task names in NI MAX
        if task_name in system.tasks.task_names:
            raise ValueError(
                f"Task '{task_name}' already exists in NI MAX. "
                "Choose a different name."
            )

        self._logger = logging.getLogger("nidaqwrapper.task")
        self._logger.debug("NITaskOutput '%s' created (rate=%s Hz)", task_name, sample_rate)

    # ------------------------------------------------------------------
    # Channel configuration
    # ------------------------------------------------------------------

    def add_channel(
        self,
        channel_name: str,
        device_ind: int,
        channel_ind: int,
        min_val: float = -10.0,
        max_val: float = 10.0,
    ) -> None:
        """Add an analog output voltage channel to the task.

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
        # Reject duplicate channel name
        if channel_name in self.channels:
            raise ValueError(
                f"Channel with duplicate name '{channel_name}' already exists."
            )

        # Reject duplicate physical channel (device_ind, channel_ind)
        for existing_name, cfg in self.channels.items():
            if cfg["device_ind"] == device_ind and cfg["channel_ind"] == channel_ind:
                raise ValueError(
                    f"Physical channel device_ind={device_ind}, channel_ind={channel_ind} "
                    f"is already assigned to channel '{existing_name}'."
                )

        # Reject out-of-range device_ind
        if device_ind < 0 or device_ind >= len(self.device_list):
            raise ValueError(
                f"device_ind={device_ind} is out of range. "
                f"Available devices ({len(self.device_list)}): {self.device_list}"
            )

        self.channels[channel_name] = {
            "device_ind": device_ind,
            "channel_ind": channel_ind,
            "min_val": min_val,
            "max_val": max_val,
        }
        self._logger.debug("Channel '%s' added (%s/ao%d)", channel_name, self.device_list[device_ind], channel_ind)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def initiate(self) -> None:
        """Create the nidaqmx task, add channels, and configure timing.

        Creates the underlying ``nidaqmx.Task``, adds all configured AO
        channels, sets up continuous sampling timing, and enables buffer
        regeneration.

        Raises
        ------
        ValueError
            If the hardware-reported sample rate does not match the
            requested rate.
        """
        self._create_task()
        self._add_channels()
        self._setup_task()

        # Validate actual sample rate matches requested
        actual_rate = float(self.task._timing.samp_clk_rate)
        if actual_rate != float(self.sample_rate):
            raise ValueError(
                f"Sample rate {self.sample_rate} Hz is not available for this device. "
                f"Next available sample rate is {actual_rate} Hz."
            )

        self._logger.debug("NITaskOutput '%s' initiated", self.task_name)

    def _create_task(self) -> None:
        """Create the underlying nidaqmx.Task."""
        self.task = nidaqmx.task.Task(new_task_name=self.task_name)

    def _add_channels(self) -> None:
        """Add all configured AO channels to the nidaqmx task."""
        for channel_name, cfg in self.channels.items():
            physical_channel = f"{self.device_list[cfg['device_ind']]}/ao{cfg['channel_ind']}"
            self.task.ao_channels.add_ao_voltage_chan(
                physical_channel=physical_channel,
                name_to_assign_to_channel=channel_name,
                min_val=cfg["min_val"],
                max_val=cfg["max_val"],
            )

    def _setup_task(self) -> None:
        """Configure timing and regeneration mode."""
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=self.sample_mode,
            samps_per_chan=self.samples_per_channel,
        )
        self.task._out_stream.regen_mode = constants.RegenerationMode.ALLOW_REGENERATION

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

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
            internally before writing.
        """
        if signal.ndim == 2:
            data = signal.T
        else:
            data = signal

        self.task.write(data, auto_start=True)
        self._logger.debug("Signal written (shape=%s)", signal.shape)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_task(self) -> None:
        """Close the nidaqmx task and release hardware resources.

        Safe to call multiple times or when the task was never initiated.
        """
        if self.task is not None:
            try:
                self.task.close()
            except Exception:
                self._logger.warning(
                    "Exception while closing task '%s'", self.task_name, exc_info=True
                )
            self.task = None
            self._logger.debug("Task '%s' cleared", self.task_name)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> NITaskOutput:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager, ensuring task cleanup."""
        try:
            self.clear_task()
        except Exception:
            self._logger.warning(
                "Exception during context manager cleanup for '%s'",
                self.task_name,
                exc_info=True,
            )
