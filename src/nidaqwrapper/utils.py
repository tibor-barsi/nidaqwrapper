"""Utility functions and constants for nidaqwrapper.

Provides the UNITS constant dictionary mapping physical unit strings to
nidaqmx constant enum values, plus device and task discovery helpers.
All hardware-dependent functions require nidaqmx to be installed.

Notes
-----
nidaqmx 1.4.1 does not have ``AccelSensitivityUnits.M_VOLTS_PER_METERS_PER_SECOND_SQUARED``.
The ``mV/m/s**2`` key currently maps to ``MILLIVOLTS_PER_G`` as a temporary
compatibility measure.  When NI ships the dedicated m/s² sensitivity constant,
update the mapping here.  See .claude-notes/agent-notes/units-nidaqmx-compat.md.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("nidaqwrapper.utils")

try:
    import nidaqmx
    from nidaqmx import constants
    from nidaqmx.errors import DaqError

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False
    logger.warning(
        "nidaqmx not available. Hardware functions will raise RuntimeError."
    )


# ---------------------------------------------------------------------------
# UNITS — physical unit string → nidaqmx constant mapping
# ---------------------------------------------------------------------------

if _NIDAQMX_AVAILABLE:
    # TODO: Replace 'mV/m/s**2' mapping with
    #       AccelSensitivityUnits.M_VOLTS_PER_METERS_PER_SECOND_SQUARED once
    #       NI ships that constant.  nidaqmx 1.4.1 only has MILLIVOLTS_PER_G
    #       and VOLTS_PER_G in AccelSensitivityUnits.
    UNITS: dict[str, Any] = {
        "mV/g": constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
        "mV/m/s**2": constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
        "g": constants.AccelUnits.G,
        "m/s**2": constants.AccelUnits.METERS_PER_SECOND_SQUARED,
        "mV/N": constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,
        "N": constants.ForceUnits.NEWTONS,
        "V": constants.VoltageUnits.VOLTS,
    }
else:
    UNITS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_nidaqmx() -> None:
    """Raise RuntimeError when nidaqmx is not available.

    Raises
    ------
    RuntimeError
        If the nidaqmx package is not installed or NI-DAQmx drivers are absent.
    """
    if not _NIDAQMX_AVAILABLE:
        raise RuntimeError(
            "NI-DAQmx drivers are required for this operation. "
            "Install the package with: pip install nidaqmx"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_devices() -> list[dict[str, str]]:
    """List all NI-DAQmx compatible devices connected to the system.

    Parameters
    ----------
    None

    Returns
    -------
    list[dict[str, str]]
        A list of dicts, one per device.  Each dict has exactly two keys:

        - ``"name"`` — the device identifier string (e.g. ``"cDAQ1Mod1"``)
        - ``"product_type"`` — the product model string (e.g. ``"NI 9234"``)

        Returns an empty list when no devices are present.

    Raises
    ------
    RuntimeError
        If nidaqmx is not installed or NI-DAQmx drivers are unavailable.

    Examples
    --------
    >>> list_devices()
    [{'name': 'cDAQ1Mod1', 'product_type': 'NI 9234'}]

    >>> list_devices()  # no hardware connected
    []
    """
    _require_nidaqmx()
    system = nidaqmx.system.System.local()
    return [
        {"name": dev.name, "product_type": dev.product_type}
        for dev in system.devices
    ]
