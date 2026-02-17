"""nidaqwrapper: Unified NI-DAQmx Python Wrapper."""

import logging

__version__ = "0.1.0"

logger = logging.getLogger("nidaqwrapper")

from .utils import (
    UNITS,
    get_connected_devices,
    get_task_by_name,
    list_devices,
    list_tasks,
)
from .task_output import NITaskOutput

__all__ = [
    "__version__",
    "NITaskOutput",
    "UNITS",
    "get_connected_devices",
    "get_task_by_name",
    "list_devices",
    "list_tasks",
]
