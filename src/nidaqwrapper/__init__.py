"""nidaqwrapper: Unified NI-DAQmx Python Wrapper."""

import logging

__version__ = "0.1.0"

logger = logging.getLogger("nidaqwrapper")

try:
    import nidaqmx
    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False

__all__ = ["__version__"]
