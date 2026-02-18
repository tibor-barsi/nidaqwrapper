"""High-level single-task NI-DAQmx interface.

Provides :class:`NIDAQWrapper`, which consolidates OpenEOL's
``IONationalInstruments`` lifecycle, LDAQ's ``NIAcquisition`` / ``NIGeneration``
patterns, and pyTrigger-based software triggering into a single, unified class.

Supports NI MAX task name strings and programmatic :class:`NITask` /
:class:`NITaskOutput` objects.  Thread-safe via per-instance
:class:`threading.RLock`.

Examples
--------
Standalone quick-start::

    wrapper = NIDAQWrapper(task_in='MyInputTask')
    wrapper.connect()
    wrapper.set_trigger(n_samples=1000, trigger_channel=0, trigger_level=0.5)
    data = wrapper.acquire()
    wrapper.disconnect()

Framework integration (LDAQ / OpenEOL)::

    wrapper = NIDAQWrapper()
    wrapper.configure(task_in=ni_task_obj, task_out='OutputTask')
    wrapper.connect()
    ...
    wrapper.disconnect()
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Union

import numpy as np

from .task_input import NITask
from .task_output import NITaskOutput
from .utils import get_connected_devices, get_task_by_name

logger = logging.getLogger("nidaqwrapper.wrapper")

try:
    import nidaqmx
    from nidaqmx.constants import READ_ALL_AVAILABLE

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False

try:
    from pyTrigger import pyTrigger

    _PYTRIGGER_AVAILABLE = True
except ImportError:
    _PYTRIGGER_AVAILABLE = False


class NIDAQWrapper:
    """High-level single-task NI-DAQmx interface.

    Supports NI MAX task names (strings) and programmatic
    :class:`NITask` / :class:`NITaskOutput` objects.  Provides a
    ``configure → connect → acquire/generate → disconnect`` lifecycle
    with software-triggered acquisition via pyTrigger, continuous signal
    generation, single-sample I/O, auto-reconnection, and thread safety.

    Parameters
    ----------
    task_in : str or NITask, optional
        Input task — NI MAX name string or :class:`NITask` instance.
    task_out : str or NITaskOutput, optional
        Output task — NI MAX name string or :class:`NITaskOutput` instance.
    **kwargs
        Additional keyword arguments forwarded to :meth:`configure`.
        Includes ``acquisition_sleep`` and ``post_trigger_delay``.
    """

    def __init__(
        self,
        task_in: str | NITask | None = None,
        task_out: str | NITaskOutput | None = None,
        **kwargs: Any,
    ) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1)

        # State flags
        self._configured = False
        self._connected = False
        self._connect_called = False
        self._trigger_is_set = False
        self._state = "disconnected"

        # Task references (active nidaqmx task handles)
        self._task_in: Any | None = None
        self._task_out: Any | None = None
        self._task_in_obj_active: Any | None = None  # active NITask obj
        self._task_out_obj_active: Any | None = None  # active NITaskOutput obj

        # Type flags
        self._task_in_is_str = False
        self._task_in_is_obj = False
        self._task_out_is_str = False
        self._task_out_is_obj = False

        # Stored configuration for reconnection
        self._task_in_name: str | None = None
        self._task_in_obj: NITask | None = None
        self._task_in_channels: dict | None = None
        self._task_in_name_str: str | None = None
        self._task_in_sample_rate: float | None = None
        self._task_in_settings_file: str | None = None
        self._task_out_name: str | None = None
        self._task_out_obj: NITaskOutput | None = None
        self._task_out_channels: dict | None = None
        self._task_out_name_str: str | None = None
        self._task_out_sample_rate: float | None = None

        # Metadata (populated on connect)
        self._channel_names_in: list[str] = []
        self._channel_names_out: list[str] = []
        self._sample_rate_in: float | None = None
        self._sample_rate_out: float | None = None
        self._n_channels_in: int = 0
        self._n_channels_out: int = 0
        self._required_devices: set[str] = set()

        # Runtime flags
        self._acquire_running = False
        self._generation_running = False

        # Timing parameters (overridable via kwargs or configure)
        self.acquisition_sleep: float = kwargs.get("acquisition_sleep", 0.01)
        self.post_trigger_delay: float = kwargs.get("post_trigger_delay", 0.05)

        # If task kwargs provided, call configure (but NOT connect)
        if task_in is not None or task_out is not None:
            self.configure(task_in=task_in, task_out=task_out, **kwargs)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        task_in: str | NITask | None = None,
        task_out: str | NITaskOutput | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure input and/or output tasks.

        Accepts NI MAX task name strings or :class:`NITask` /
        :class:`NITaskOutput` objects.  Resets all internal state so the
        wrapper can be reconfigured without creating a new instance.

        Parameters
        ----------
        task_in : str or NITask, optional
            Input task specification.
        task_out : str or NITaskOutput, optional
            Output task specification.
        **kwargs
            ``acquisition_sleep``, ``post_trigger_delay``.
        """
        # Reset state
        self._trigger_is_set = False
        self._acquire_running = False
        self._generation_running = False
        self._task_in = None
        self._task_out = None
        self._task_in_obj_active = None
        self._task_out_obj_active = None

        # Timing parameters
        if "acquisition_sleep" in kwargs:
            self.acquisition_sleep = kwargs["acquisition_sleep"]
        if "post_trigger_delay" in kwargs:
            self.post_trigger_delay = kwargs["post_trigger_delay"]

        # -- Input task ------------------------------------------------
        self._task_in_is_str = False
        self._task_in_is_obj = False
        self._task_in_name = None
        self._task_in_obj = None
        self._task_in_channels = None
        self._task_in_name_str = None
        self._task_in_sample_rate = None
        self._task_in_settings_file = None

        if isinstance(task_in, str):
            self._task_in_is_str = True
            self._task_in_name = task_in
        elif isinstance(task_in, NITask):
            self._task_in_is_obj = True
            self._task_in_obj = task_in
            self._task_in_channels = copy.deepcopy(task_in.channels)
            self._task_in_name_str = task_in.task_name
            self._task_in_sample_rate = task_in.sample_rate
            self._task_in_settings_file = getattr(task_in, "settings_file", None)
        elif task_in is not None:
            raise TypeError(
                f"task_in must be a string or NITask, got {type(task_in).__name__}"
            )

        # -- Output task -----------------------------------------------
        self._task_out_is_str = False
        self._task_out_is_obj = False
        self._task_out_name = None
        self._task_out_obj = None
        self._task_out_channels = None
        self._task_out_name_str = None
        self._task_out_sample_rate = None

        if isinstance(task_out, str):
            self._task_out_is_str = True
            self._task_out_name = task_out
        elif isinstance(task_out, NITaskOutput):
            self._task_out_is_obj = True
            self._task_out_obj = task_out
            self._task_out_channels = copy.deepcopy(task_out.channels)
            self._task_out_name_str = task_out.task_name
            self._task_out_sample_rate = task_out.sample_rate
        elif task_out is not None:
            raise TypeError(
                f"task_out must be a string or NITaskOutput, got {type(task_out).__name__}"
            )

        self._configured = True
        logger.debug("Configured: task_in=%s, task_out=%s", task_in, task_out)

    # ------------------------------------------------------------------
    # Lifecycle: connect / disconnect
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to NI hardware by loading/creating tasks.

        For NI MAX name strings, loads the saved task via
        :func:`get_task_by_name`.  For :class:`NITask` objects, calls
        ``initiate()`` to create the underlying hardware task.

        Returns
        -------
        bool
            ``True`` if connection succeeded and :meth:`ping` passes,
            ``False`` otherwise.
        """
        with self._lock:
            self._connect_called = True

            try:
                # Close any existing tasks first
                self._close_task_in()
                self._close_task_out()

                # -- Input task ----------------------------------------
                if self._task_in_is_str:
                    loaded = get_task_by_name(self._task_in_name)
                    if loaded is None:
                        logger.error("Failed to load input task '%s'.", self._task_in_name)
                        self._connected = False
                        return False
                    self._task_in = loaded
                    self._extract_input_metadata_from_nidaqmx(loaded)

                elif self._task_in_is_obj:
                    ni_task = self._recreate_ni_task_in()
                    ni_task.initiate()
                    self._task_in = ni_task.task
                    self._task_in_obj_active = ni_task
                    self._extract_input_metadata_from_ni_task(ni_task)

                # -- Output task ---------------------------------------
                if self._task_out_is_str:
                    loaded = get_task_by_name(self._task_out_name)
                    if loaded is None:
                        logger.error("Failed to load output task '%s'.", self._task_out_name)
                        self._connected = False
                        return False
                    self._task_out = loaded
                    self._extract_output_metadata_from_nidaqmx(loaded)

                elif self._task_out_is_obj:
                    ni_task_out = self._recreate_ni_task_out()
                    ni_task_out.initiate()
                    self._task_out = ni_task_out.task
                    self._task_out_obj_active = ni_task_out
                    self._extract_output_metadata_from_ni_task_out(ni_task_out)

                # Build required devices set
                self._required_devices = set()
                if self._task_in is not None:
                    for dev in self._task_in.devices:
                        self._required_devices.add(dev.name)
                if self._task_out is not None:
                    for dev in self._task_out.devices:
                        self._required_devices.add(dev.name)

                # Verify connectivity
                if self.ping():
                    self._connected = True
                    self._state = "connected"
                    logger.info("Connected successfully.")
                    return True
                else:
                    self._connected = False
                    logger.warning("Connected but ping failed.")
                    return False

            except Exception:
                logger.exception("Connection failed.")
                self._connected = False
                return False

    def disconnect(self) -> bool:
        """Disconnect from NI hardware, closing all tasks.

        Idempotent — safe to call multiple times or when never connected.

        Returns
        -------
        bool
            Always ``True``.
        """
        with self._lock:
            # Stop any running generation safely
            if self._generation_running:
                try:
                    self._stop_generation_impl()
                except Exception:
                    logger.warning("Exception during generation stop in disconnect.", exc_info=True)

            self._close_task_in()
            self._close_task_out()
            self._connected = False
            self._state = "disconnected"
            logger.debug("Disconnected.")
            return True

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    def set_trigger(
        self,
        n_samples: int,
        trigger_channel: int,
        trigger_level: float,
        trigger_type: str = "abs",
        presamples: int = 0,
    ) -> None:
        """Set up software triggering via pyTrigger.

        Parameters
        ----------
        n_samples : int
            Number of samples to acquire after trigger.
        trigger_channel : int
            Channel index used for trigger detection.
        trigger_level : float
            Trigger level in the channel's native units.
        trigger_type : str, optional
            Trigger type: ``'abs'``, ``'up'``, or ``'down'``.
            Default is ``'abs'``.
        presamples : int, optional
            Number of pre-trigger samples. Default is ``0``.

        Raises
        ------
        ValueError
            If no input task is configured.
        """
        if not (self._task_in_is_str or self._task_in_is_obj):
            raise ValueError(
                "No input task is configured. Call configure() with task_in first."
            )

        self.trigger = pyTrigger(
            rows=n_samples,
            channels=self._n_channels_in,
            trigger_channel=trigger_channel,
            trigger_level=trigger_level,
            trigger_type=trigger_type,
            presamples=presamples,
        )
        self._trigger_is_set = True
        logger.debug(
            "Trigger set: n_samples=%d, channel=%d, level=%s, type=%s",
            n_samples, trigger_channel, trigger_level, trigger_type,
        )

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def acquire(
        self,
        return_dict: bool = False,
        blocking: bool = True,
    ) -> np.ndarray | dict | Future:
        """Perform software-triggered acquisition.

        Requires :meth:`set_trigger` to have been called first.

        Parameters
        ----------
        return_dict : bool, optional
            If ``True``, return a dict with channel names as keys and a
            ``'time'`` key.  Default is ``False`` (numpy array).
        blocking : bool, optional
            If ``True`` (default), block until acquisition completes.
            If ``False``, return a :class:`~concurrent.futures.Future`.

        Returns
        -------
        numpy.ndarray or dict or Future
            Acquired data in ``(n_samples, n_channels)`` format, or a
            dict, or a Future wrapping either.

        Raises
        ------
        ValueError
            If no input task is configured.
        RuntimeError
            If :meth:`set_trigger` has not been called.
        """
        if not (self._task_in_is_str or self._task_in_is_obj):
            raise ValueError(
                "No input task is configured. Call configure() with task_in first."
            )

        if not self._trigger_is_set:
            raise RuntimeError(
                "set_trigger() must be called before acquire(). "
                "Configure a trigger with set_trigger(n_samples, trigger_channel, trigger_level)."
            )

        if blocking:
            return self._acquire_impl(return_dict)
        else:
            return self._executor.submit(self._acquire_impl, return_dict)

    def _acquire_impl(self, return_dict: bool = False) -> np.ndarray | dict:
        """Execute the triggered acquisition loop.

        Internal method — called directly for blocking acquire, or
        submitted to the executor for non-blocking.

        Parameters
        ----------
        return_dict : bool
            Whether to return a dict with channel names.

        Returns
        -------
        numpy.ndarray or dict
            Acquired data.
        """
        with self._lock:
            # Reset trigger state for fresh acquisition
            self._reset_trigger()

            # Start the input task
            self._task_in.start()

            # Flush buffer (discard first read — driver init artifact)
            try:
                self._task_in.read(READ_ALL_AVAILABLE, timeout=0.5)
            except Exception:
                pass  # Flush may return empty; ignore

            # Main acquisition loop
            self._acquire_running = True
            while self._acquire_running:
                raw_data = np.array(
                    self._task_in.read(READ_ALL_AVAILABLE, timeout=0.5)
                )

                # Single-channel: nidaqmx returns 1D → reshape to (1, n_samples)
                if raw_data.ndim == 1:
                    raw_data = raw_data[None, :]

                # Transpose to (n_samples, n_channels) for pyTrigger
                data = raw_data.T
                self.trigger.add_data(data)

                if self.trigger.finished:
                    self._acquire_running = False
                else:
                    time.sleep(self.acquisition_sleep)

            # Post-trigger delay
            time.sleep(self.post_trigger_delay)

            # Stop the input task
            self._task_in.stop()

            # Get triggered data
            if return_dict:
                data_arr = self.trigger.get_data()
                result: dict[str, Any] = {}
                for i, ch_name in enumerate(self._channel_names_in):
                    result[ch_name] = data_arr[:, i]
                result["time"] = np.arange(data_arr.shape[0]) / self._sample_rate_in
                return result
            else:
                return self.trigger.get_data()

    def _reset_trigger(self) -> None:
        """Reset the pyTrigger instance for a fresh acquisition."""
        self.trigger.ringbuff.clear()
        self.trigger.triggered = False
        self.trigger.rows_left = self.trigger.rows
        self.trigger.finished = False
        self.trigger.first_data = True

    # ------------------------------------------------------------------
    # Read (LDAQ integration + single sample)
    # ------------------------------------------------------------------

    def read_all_available(self) -> np.ndarray:
        """Read all available samples without triggering (LDAQ pattern).

        Calls ``acquire_base()`` on the active input task and transposes
        the result to ``(n_samples, n_channels)`` public format.

        Returns
        -------
        numpy.ndarray
            Data in ``(n_samples, n_channels)`` format.  Returns
            ``(0, n_channels)`` if no data is available.
        """
        with self._lock:
            if self._task_in_obj_active is None:
                raise ValueError(
                    "No active input task. Call connect() first."
                )
            # acquire_base() returns (n_channels, n_samples)
            raw = self._task_in_obj_active.acquire_base()
            return raw.T

    def read(self) -> np.ndarray:
        """Read a single sample from each input channel.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_channels,)``.

        Raises
        ------
        ValueError
            If no input task is configured, or if a continuous acquisition
            is currently in progress.
        """
        with self._lock:
            if self._task_in is None:
                raise ValueError(
                    "No input task is configured. Call configure() and connect() first."
                )
            if self._acquire_running:
                raise ValueError(
                    "Cannot call read() while a continuous acquisition is running. "
                    "Wait for acquire() to complete or use read_all_available()."
                )
            data = np.array(self._task_in.read(1))
            # nidaqmx may return a scalar for single-channel tasks, producing
            # a 0-D array.  Reshape to (1,) so callers always receive 1-D output.
            if data.ndim == 0:
                data = data.reshape((1,))
            return data

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, data: np.ndarray, overwrite: bool = True) -> None:
        """Start continuous signal generation.

        Parameters
        ----------
        data : numpy.ndarray
            Signal data in ``(n_samples, n_channels)`` or ``(n_samples,)``
            format.
        overwrite : bool, optional
            If ``True`` (default), stop any current generation first.
            If ``False``, raise if generation is running.

        Raises
        ------
        ValueError
            If no output task configured, shape mismatch, or generation
            running with ``overwrite=False``.
        """
        with self._lock:
            if self._task_out is None:
                raise ValueError(
                    "No output task is configured. Call configure() with task_out first."
                )

            # Validate shape
            if data.ndim == 2 and data.shape[1] != self._n_channels_out:
                raise ValueError(
                    f"Data has {data.shape[1]} columns but {self._n_channels_out} "
                    f"output channels are configured."
                )

            if self._generation_running:
                if overwrite:
                    self._task_out.stop()
                    time.sleep(0.01)
                else:
                    raise ValueError(
                        "Output generation is currently running and overwrite=False."
                    )

            # Transpose for nidaqmx: (n_samples, n_channels) → (n_channels, n_samples)
            if data.ndim == 2:
                write_data = np.ascontiguousarray(data.T)
            else:
                # 1D single-channel — ensure C-contiguous for nidaqmx C layer
                write_data = np.ascontiguousarray(data)

            self._task_out.out_stream.output_buf_size = data.shape[0]
            self._task_out.write(write_data, auto_start=True)
            self._generation_running = True
            logger.debug("Generation started (shape=%s).", data.shape)

    def write(
        self,
        data: float | int | list | np.ndarray,
        overwrite: bool = True,
    ) -> None:
        """Write a single sample to each output channel.

        Handles nidaqmx's 2-sample minimum buffer requirement by
        duplicating the data to 2 samples internally.

        Parameters
        ----------
        data : float, int, list, or numpy.ndarray
            Single sample per channel.  Float/int for single-channel,
            array of shape ``(n_channels,)`` for multi-channel.
        overwrite : bool, optional
            If ``True`` (default), stop current generation first.

        Raises
        ------
        ValueError
            If shape doesn't match channel count, or generation running
            with ``overwrite=False``.
        """
        with self._lock:
            if self._task_out is None:
                raise ValueError(
                    "No output task is configured. Call configure() with task_out first."
                )

            # Normalize input
            if isinstance(data, (float, int)):
                data = np.array([data], dtype=float)
            elif isinstance(data, list):
                data = np.array(data, dtype=float)

            # Validate shape
            if data.shape[0] != self._n_channels_out:
                raise ValueError(
                    f"Data has {data.shape[0]} values but {self._n_channels_out} "
                    f"output channels are configured."
                )

            if self._generation_running:
                if overwrite:
                    self._task_out.stop()
                    time.sleep(0.01)
                else:
                    raise ValueError(
                        "Output generation is currently running and overwrite=False."
                    )

            # nidaqmx requires at least 2 samples
            if self._n_channels_out == 1:
                write_data = np.concatenate((data, data))
            else:
                # Stack to (n_channels, 2) then make C-contiguous
                write_data = np.ascontiguousarray(
                    np.array([data, data]).T
                )

            self._task_out.out_stream.output_buf_size = 2
            self._task_out.write(write_data, auto_start=True)
            self._task_out.stop()
            logger.debug("Single-sample write completed.")

    def stop_generation(self) -> None:
        """Stop signal generation and write zeros to all output channels.

        Follows the safe shutdown pattern: stop the task, then write
        zeros so the output doesn't hold the last voltage.
        """
        with self._lock:
            if self._task_out is None:
                return  # No-op if no output task

            self._stop_generation_impl()

    def _stop_generation_impl(self) -> None:
        """Internal stop generation — caller must hold lock."""
        self._generation_running = False
        self._task_out.stop()
        # Write zeros through the write() path (handles 2-sample minimum)
        self.write(np.zeros(self._n_channels_out))

    # ------------------------------------------------------------------
    # Connectivity: ping / check_state
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Test if all required NI devices are connected.

        Returns
        -------
        bool
            ``True`` if all required devices are in the connected set.
        """
        with self._lock:
            if not self._configured:
                return False

            if not self._required_devices:
                return False

            connected = get_connected_devices()
            for device in self._required_devices:
                if device not in connected:
                    logger.warning("Device '%s' is not connected.", device)
                    return False
            return True

    def check_state(self) -> str:
        """Check connection state with auto-reconnection.

        Returns
        -------
        str
            One of ``'connected'``, ``'reconnected'``,
            ``'connection lost'``, or ``'disconnected'``.
        """
        with self._lock:
            if not self._connected:
                if self._connect_called:
                    logger.info("Connection was lost, trying to reconnect...")
                    if self.connect():
                        self._state = "reconnected"
                        return "reconnected"
                    self._state = "connection lost"
                    return "connection lost"
                self._state = "disconnected"
                return "disconnected"

            # Connected — verify with ping
            if self.ping():
                self._state = "connected"
                return "connected"

            # Ping failed — attempt reconnect
            logger.warning("Connection to device lost. Attempting reconnect...")
            if self.connect():
                self._state = "reconnected"
                return "reconnected"

            self._state = "connection lost"
            return "connection lost"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_device_info(self) -> dict:
        """Return device/task metadata.

        Returns
        -------
        dict
            Keys ``'input'`` and/or ``'output'``, each containing
            ``'channel_names'`` and ``'sample_rate'``.
        """
        info: dict[str, Any] = {}
        if self._task_in_is_str or self._task_in_is_obj:
            if self._channel_names_in:
                info["input"] = {
                    "channel_names": self._channel_names_in,
                    "sample_rate": self._sample_rate_in,
                }
        if self._task_out_is_str or self._task_out_is_obj:
            if self._channel_names_out:
                info["output"] = {
                    "channel_names": self._channel_names_out,
                    "sample_rate": self._sample_rate_out,
                }
        return info

    def get_sample_rate(self) -> float:
        """Return the input task's sample rate.

        Returns
        -------
        float
            Sample rate in Hz.

        Raises
        ------
        ValueError
            If no input task is configured.
        """
        if self._sample_rate_in is None:
            raise ValueError(
                "No input task configured. Cannot determine sample rate."
            )
        return self._sample_rate_in

    def get_channel_names(self) -> list[str]:
        """Return input channel names.

        Returns
        -------
        list[str]
            Channel name strings, or empty list if no input task.
        """
        return list(self._channel_names_in)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def clear_buffer(self) -> None:
        """Drain and discard all buffered input data.

        Prevents buffer overflow by reading and discarding all samples
        currently in the hardware FIFO.
        """
        with self._lock:
            if self._task_in_obj_active is None:
                return  # No-op
            self._task_in_obj_active.acquire_base()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> NIDAQWrapper:
        """Enter the runtime context."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the runtime context, disconnecting from hardware."""
        self.disconnect()

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self.disconnect()
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_task_in(self) -> None:
        """Close the active input task (nidaqmx handle)."""
        if self._task_in is not None:
            try:
                self._task_in.close()
            except Exception:
                logger.warning("Exception closing input task.", exc_info=True)
            self._task_in = None

        if self._task_in_obj_active is not None:
            try:
                self._task_in_obj_active.clear_task()
            except Exception:
                logger.warning("Exception clearing NITask.", exc_info=True)
            self._task_in_obj_active = None

    def _close_task_out(self) -> None:
        """Close the active output task (nidaqmx handle)."""
        if self._task_out is not None:
            try:
                self._task_out.close()
            except Exception:
                logger.warning("Exception closing output task.", exc_info=True)
            self._task_out = None

        if self._task_out_obj_active is not None:
            try:
                self._task_out_obj_active.clear_task()
            except Exception:
                logger.warning("Exception clearing NITaskOutput.", exc_info=True)
            self._task_out_obj_active = None

    def _recreate_ni_task_in(self) -> NITask:
        """Recreate an NITask from stored channel configuration.

        Used on reconnection to get a fresh hardware task from the
        stored channel config (deep-copy pattern from LDAQ).
        """
        if self._task_in_channels is not None:
            new_task = NITask(
                self._task_in_name_str,
                self._task_in_sample_rate,
                self._task_in_settings_file,
            )
            for ch_name, ch_cfg in self._task_in_channels.items():
                new_task.add_channel(
                    ch_name,
                    ch_cfg["device_ind"],
                    ch_cfg["channel_ind"],
                    ch_cfg["sensitivity"],
                    ch_cfg.get("sensitivity_units"),
                    ch_cfg.get("units_str") or ch_cfg.get("units"),
                    ch_cfg.get("serial_nr"),
                    ch_cfg.get("scale"),
                    ch_cfg.get("min_val"),
                    ch_cfg.get("max_val"),
                )
            return new_task
        return self._task_in_obj

    def _recreate_ni_task_out(self) -> NITaskOutput:
        """Recreate an NITaskOutput from stored channel configuration."""
        if self._task_out_channels is not None:
            new_task = NITaskOutput(
                self._task_out_name_str,
                self._task_out_sample_rate,
            )
            new_task.channels = copy.deepcopy(self._task_out_channels)
            return new_task
        return self._task_out_obj

    def _extract_input_metadata_from_nidaqmx(self, task: Any) -> None:
        """Extract metadata from a loaded nidaqmx.Task (NI MAX)."""
        self._channel_names_in = list(task.channel_names)
        self._n_channels_in = task.number_of_channels
        self._sample_rate_in = float(task.timing.samp_clk_rate)

    def _extract_input_metadata_from_ni_task(self, ni_task: NITask) -> None:
        """Extract metadata from a programmatic NITask object."""
        self._channel_names_in = ni_task.channel_list
        self._n_channels_in = ni_task.number_of_ch
        self._sample_rate_in = ni_task.sample_rate

    def _extract_output_metadata_from_nidaqmx(self, task: Any) -> None:
        """Extract metadata from a loaded output nidaqmx.Task."""
        self._channel_names_out = list(task.channel_names)
        self._n_channels_out = task.number_of_channels
        self._sample_rate_out = float(task.timing.samp_clk_rate)

    def _extract_output_metadata_from_ni_task_out(
        self, ni_task_out: NITaskOutput
    ) -> None:
        """Extract metadata from a programmatic NITaskOutput object."""
        self._channel_names_out = list(ni_task_out.channels.keys())
        self._n_channels_out = len(ni_task_out.channels)
        self._sample_rate_out = ni_task_out.sample_rate
