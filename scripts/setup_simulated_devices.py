#!/usr/bin/env python3
"""Cross-platform script to set up simulated NI-DAQmx devices for testing.

This script handles simulated device creation for both Linux and Windows:

Linux
-----
- Generates the simulated_devices.ini file if needed
- Prints the exact nidaqmxconfig --import command to run (requires sudo)
- If device already exists, creates SimTask1 directly

Windows
-------
- UNTESTED: Windows path attempts to check if device exists
- If not found, prints step-by-step NI MAX instructions
- If device exists, creates SimTask1

The script can be run with --check to just report current state without
making any changes.

Usage
-----
    python scripts/setup_simulated_devices.py          # Normal setup
    python scripts/setup_simulated_devices.py --check  # Just check status

Requirements
------------
- NI-DAQmx driver installed (version 25.8 or later)
- nidaqmx Python package
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

# Device and task configuration
DEVICE_NAME = "SimDev1"
DEVICE_TYPE = "PCIe-6361"
DEVICE_SERIAL = "0x87654321"
TASK_NAME = "SimTask1"
SAMPLE_RATE = 10000
SAMPLES_PER_CHAN = 1000


def check_device_exists() -> bool:
    """Check if SimDev1 device exists."""
    try:
        import nidaqmx.system

        devices = [d.name for d in nidaqmx.system.System.local().devices]
        return DEVICE_NAME in devices
    except Exception as e:
        print(f"Warning: Could not query devices: {e}")
        return False


def check_task_exists() -> bool:
    """Check if SimTask1 task exists."""
    try:
        import nidaqmx.system

        tasks = [t._name for t in nidaqmx.system.System.local().tasks]
        return TASK_NAME in tasks
    except Exception as e:
        print(f"Warning: Could not query tasks: {e}")
        return False


def list_devices_and_tasks() -> None:
    """Print current devices and tasks."""
    try:
        import nidaqmx.system

        system = nidaqmx.system.System.local()

        print("\nCurrent NI-DAQmx devices:")
        devices = list(system.devices)
        if devices:
            for dev in devices:
                sim_flag = " (simulated)" if dev.is_simulated else ""
                print(f"  - {dev.name}: {dev.product_type}{sim_flag}")
        else:
            print("  (none)")

        print("\nCurrent persisted tasks:")
        tasks = list(system.tasks)
        if tasks:
            for task in tasks:
                print(f"  - {task._name}")
        else:
            print("  (none)")
        print()

    except Exception as e:
        print(f"Error listing devices/tasks: {e}")


def create_task() -> bool:
    """Create SimTask1 persisted task.

    Returns
    -------
    bool
        True if task created successfully, False otherwise.
    """
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration

        # Delete existing task if present
        try:
            system = nidaqmx.system.System.local()
            for task in system.tasks:
                if task._name == TASK_NAME:
                    print(f"Deleting existing task '{TASK_NAME}'...")
                    task.delete()
                    break
        except Exception as e:
            print(f"Warning: Could not check for existing task: {e}")

        # Create the task
        print(f"\nCreating task '{TASK_NAME}'...")
        with nidaqmx.Task(TASK_NAME) as task:
            task.ai_channels.add_ai_voltage_chan(
                f"{DEVICE_NAME}/ai0:3",
                terminal_config=TerminalConfiguration.RSE,
                min_val=-10.0,
                max_val=10.0,
            )
            task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=SAMPLES_PER_CHAN,
            )
            task.save(save_as=TASK_NAME, overwrite_existing_task=True)

        print(f"Task '{TASK_NAME}' created successfully.")
        print(f"  - 4 AI channels ({DEVICE_NAME}/ai0:3)")
        print(f"  - Sample rate: {SAMPLE_RATE} Hz")
        print(f"  - Buffer: {SAMPLES_PER_CHAN} samples/channel")
        print(f"  - Mode: Continuous acquisition")
        return True

    except Exception as e:
        print(f"ERROR: Failed to create task: {e}")
        return False


def setup_linux(check_only: bool = False) -> int:
    """Handle Linux setup.

    Parameters
    ----------
    check_only : bool
        If True, only report status without making changes.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    print("Platform: Linux")

    device_exists = check_device_exists()
    task_exists = check_task_exists()

    if check_only:
        list_devices_and_tasks()
        print("Status:")
        print(f"  SimDev1 device: {'EXISTS' if device_exists else 'NOT FOUND'}")
        print(f"  SimTask1 task:  {'EXISTS' if task_exists else 'NOT FOUND'}")
        return 0

    # If device doesn't exist, print instructions
    if not device_exists:
        print(f"\nDevice '{DEVICE_NAME}' not found.")
        print("\nTo create the simulated device, run:")
        print("  sudo nidaqmxconfig --import config/simulated_devices.ini")
        print("\nOr use the convenience script:")
        print("  sudo scripts/setup_simulated_devices.sh")
        return 1

    # Device exists, create task if needed
    print(f"\nDevice '{DEVICE_NAME}' found.")
    if task_exists:
        print(f"Task '{TASK_NAME}' already exists.")
        return 0

    return 0 if create_task() else 1


def setup_windows(check_only: bool = False) -> int:
    """Handle Windows setup (UNTESTED).

    Parameters
    ----------
    check_only : bool
        If True, only report status without making changes.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    print("Platform: Windows (UNTESTED)")

    device_exists = check_device_exists()
    task_exists = check_task_exists()

    if check_only:
        list_devices_and_tasks()
        print("Status:")
        print(f"  SimDev1 device: {'EXISTS' if device_exists else 'NOT FOUND'}")
        print(f"  SimTask1 task:  {'EXISTS' if task_exists else 'NOT FOUND'}")
        return 0

    # If device doesn't exist, print NI MAX instructions
    if not device_exists:
        print(f"\nDevice '{DEVICE_NAME}' not found.")
        print("\n" + "=" * 70)
        print("Create simulated device in NI MAX:")
        print("=" * 70)
        print("""
1. Open NI MAX (Measurement & Automation Explorer)
2. Right-click on 'Devices and Interfaces'
3. Select 'Create New...' -> 'Simulated NI-DAQmx Device or Modular Instrument'
4. Select device: 'PCIe-6361'
5. Click 'OK'
6. Right-click on the new device and select 'Rename'
7. Rename it to 'SimDev1'

After creating the device, run this script again to create the task.
""")
        print("=" * 70)
        return 1

    # Device exists, create task if needed
    print(f"\nDevice '{DEVICE_NAME}' found.")
    if task_exists:
        print(f"Task '{TASK_NAME}' already exists.")
        return 0

    return 0 if create_task() else 1


def main() -> int:
    """Main entry point.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Set up simulated NI-DAQmx devices for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check current device/task status without making changes",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("NI-DAQmx Simulated Device Setup")
    print("=" * 70)

    # Check nidaqmx is available
    try:
        import nidaqmx
    except ImportError:
        print("\nERROR: nidaqmx package not found.")
        print("Install with: pip install nidaqmx")
        return 1

    # Platform-specific setup
    system = platform.system()
    if system == "Linux":
        return setup_linux(check_only=args.check)
    elif system == "Windows":
        return setup_windows(check_only=args.check)
    else:
        print(f"\nERROR: Unsupported platform: {system}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
