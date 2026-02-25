"""Base class for NI-DAQmx task wrappers.

Provides shared lifecycle management, properties, and classmethods
inherited by all concrete task classes (AITask, AOTask, DITask, DOTask).

Design Constraint
-----------------
This module does NOT import nidaqmx.  All shared methods operate on
instance attributes (``self.task``, ``self._owns_task``, etc.) that are
set by subclass constructors.  This preserves existing test mock patterns
where nidaqmx references are patched per-subclass module.
"""

from __future__ import annotations

import warnings
from typing import Any

from .utils import _require_nidaqmx, get_task_by_name


class BaseTask:
    """Abstract base for all NI-DAQmx task wrappers.

    Subclasses must set the class attributes ``_channel_attr`` and
    ``_channel_type_label`` and implement their own ``__init__``,
    ``from_task``, ``configure``, ``add_channel``, and data I/O methods.

    Attributes
    ----------
    _channel_attr : str
        Name of the nidaqmx channel collection (e.g., ``"ai_channels"``).
    _channel_type_label : str
        Human-readable channel type for error messages (e.g., ``"AI"``).
    """

    _channel_attr: str
    _channel_type_label: str

    # -- Properties ----------------------------------------------------------

    @property
    def channel_list(self) -> list[str]:
        """Return the list of channel names in the task.

        Returns
        -------
        list[str]
            Channel name strings from the underlying nidaqmx task.
        """
        return list(self.task.channel_names)

    @property
    def number_of_ch(self) -> int:
        """Return the number of channels in the task.

        Returns
        -------
        int
            Count of channels configured on the underlying nidaqmx task.
        """
        return len(self.task.channel_names)

    # -- Lifecycle -----------------------------------------------------------

    def clear_task(self) -> None:
        """Release the hardware task handle.

        Closes the underlying ``nidaqmx.task.Task`` and sets ``self.task``
        to ``None``.  Safe to call on an already-cleared task or multiple
        times.

        Notes
        -----
        When this task was created via :meth:`from_task`, the nidaqmx task
        is NOT closed — the caller retains ownership and must call
        ``task.close()`` when done.  A warning is issued as a reminder.
        """
        if hasattr(self, "task") and self.task is not None:
            # If we don't own the task, skip close and warn the user
            if not self._owns_task:
                warnings.warn(
                    "Task was created externally — not closing. "
                    "Call task.close() when done.",
                    stacklevel=2,
                )
                self.task = None
                return

            # Normal path: we own the task, close it
            try:
                self.task.close()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self.task = None

    def _check_start_preconditions(self) -> None:
        """Validate that the task can be configured or started.

        Raises
        ------
        RuntimeError
            If the task was created via ``from_task()`` (externally owned).
        ValueError
            If no channels have been added to the task.
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot configure an externally-provided task. "
                "Configure the nidaqmx.Task directly or pass an "
                "already-configured task to from_task()."
            )
        if not self.task.channel_names:
            raise ValueError(
                "Cannot configure: no channels have been added to this task. "
                "Call add_channel() before configure()."
            )

    def start(self) -> None:
        """Start the hardware task.

        Begins acquisition or generation on the underlying nidaqmx task.
        Call :meth:`configure` first to set timing parameters.

        Raises
        ------
        RuntimeError
            If this task was created via :meth:`from_task`
            (externally owned — start the nidaqmx.Task directly).
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot start an externally-provided task. "
                "Start the nidaqmx.Task directly or pass an "
                "already-started task to from_task()."
            )
        self.task.start()

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> BaseTask:
        """Enter the runtime context; return ``self``."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the runtime context, releasing hardware resources.

        Calls :meth:`clear_task` unconditionally.  If ``clear_task`` raises,
        a warning is emitted and the exception is swallowed so it does not
        mask any exception that propagated from the ``with`` block body.
        """
        try:
            self.clear_task()
        except Exception as exc:
            warnings.warn(str(exc), stacklevel=2)

    # -- Class methods -------------------------------------------------------

    @classmethod
    def from_name(cls, task_name: str) -> BaseTask:
        """Load an NI MAX task by name and wrap it.

        Looks up the task in NI MAX via
        :func:`~nidaqwrapper.utils.get_task_by_name`, then wraps it using
        the subclass :meth:`from_task`.  Unlike ``from_task()``, the wrapper
        takes ownership of the loaded task and will close it on
        :meth:`clear_task` or ``__exit__``.

        Parameters
        ----------
        task_name : str
            The name of the task as saved in NI MAX.

        Returns
        -------
        BaseTask
            A task wrapper instance (concrete subclass type at runtime).

        Raises
        ------
        KeyError
            If no task named ``task_name`` exists in NI MAX.
        ConnectionError
            If the device associated with the task is disconnected.
        RuntimeError
            If the task is already loaded by another process, or if
            nidaqmx is not installed.
        ValueError
            If the loaded task has no channels of the expected type.

        Examples
        --------
        >>> task = AITask.from_name("MyInputTask")
        >>> task.start()
        >>> data = task.acquire()
        >>> task.clear_task()
        """
        _require_nidaqmx()
        loaded = get_task_by_name(task_name)
        if loaded is None:
            raise RuntimeError(
                f"Task '{task_name}' is already loaded by another process."
            )
        instance = cls.from_task(loaded)
        instance._owns_task = True
        return instance
