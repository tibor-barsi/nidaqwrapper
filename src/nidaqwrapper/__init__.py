"""nidaqwrapper: Unified NI-DAQmx Python Wrapper."""

import logging

__version__ = "0.1.0"

logger = logging.getLogger("nidaqwrapper")

try:
    import nidaqmx
    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False

from .utils import (
    UNITS,
    get_connected_devices,
    get_task_by_name,
    list_devices,
    list_tasks,
)

__all__ = [
    "__version__",
    "UNITS",
    "get_connected_devices",
    "get_task_by_name",
    "list_devices",
    "list_tasks",
]
