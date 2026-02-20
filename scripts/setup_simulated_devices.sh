#!/usr/bin/env bash
set -euo pipefail

# setup_simulated_devices.sh — Create simulated NI-DAQmx devices for testing
#
# This script:
# 1. Checks that nidaqmxconfig is installed
# 2. Imports simulated_devices.ini to create SimDev1 (PCIe-6361)
# 3. Verifies the device was created
# 4. Creates the SimTask1 persisted task (4 AI channels, 10kHz, continuous)
#
# Must be run with sudo from the child repo root directory.

# ---------------------------------------------------------------------------
# Ensure script is run as root
# ---------------------------------------------------------------------------

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run with sudo."
    echo "Usage: sudo scripts/setup_simulated_devices.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Ensure script is run from repo root
# ---------------------------------------------------------------------------

if [[ ! -f "config/simulated_devices.ini" ]]; then
    echo "ERROR: config/simulated_devices.ini not found."
    echo "This script must be run from the nidaqwrapper repository root directory."
    exit 1
fi

# ---------------------------------------------------------------------------
# Check nidaqmxconfig exists
# ---------------------------------------------------------------------------

if ! command -v nidaqmxconfig &> /dev/null; then
    echo "ERROR: nidaqmxconfig not found in PATH."
    echo "Please install NI-DAQmx driver (version 25.8 or later)."
    exit 1
fi

echo "Found nidaqmxconfig: $(command -v nidaqmxconfig)"

# ---------------------------------------------------------------------------
# Import the simulated device INI
# ---------------------------------------------------------------------------

echo ""
echo "Importing simulated devices from config/simulated_devices.ini..."
if nidaqmxconfig --import config/simulated_devices.ini; then
    echo "Device import successful."
else
    echo "ERROR: Device import failed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify the device was created
# ---------------------------------------------------------------------------

echo ""
echo "Verifying SimDev1 device..."
if ! uv run python3 -c "
import nidaqmx.system
devices = [d.name for d in nidaqmx.system.System.local().devices]
if 'SimDev1' in devices:
    print('SimDev1 device found.')
    exit(0)
else:
    print('ERROR: SimDev1 device not found.')
    print('Available devices:', devices)
    exit(1)
"; then
    echo "ERROR: SimDev1 device verification failed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Create SimTask1 persisted task
# ---------------------------------------------------------------------------

echo ""
echo "Creating SimTask1 persisted task (4 AI channels, 10kHz, continuous)..."
if ! uv run python3 -c "
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# Check if task already exists and delete it
try:
    system = nidaqmx.system.System.local()
    for task in system.tasks:
        if task._name == 'SimTask1':
            print('Deleting existing SimTask1...')
            task.delete()
            break
except Exception as e:
    print(f'Warning: Could not check for existing task: {e}')

# Create the task
with nidaqmx.Task('SimTask1') as task:
    task.ai_channels.add_ai_voltage_chan(
        'SimDev1/ai0:3',
        terminal_config=TerminalConfiguration.RSE,
        min_val=-10.0,
        max_val=10.0
    )
    task.timing.cfg_samp_clk_timing(
        rate=10000,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=1000
    )
    task.save(save_as='SimTask1', overwrite_existing_task=True)
    print('SimTask1 created successfully.')
"; then
    echo "ERROR: SimTask1 creation failed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "Setup complete!"
echo "========================================================================"
echo ""
echo "Simulated device SimDev1 (PCIe-6361) created with:"
echo "  - 16 AI channels"
echo "  - 2 AO channels"
echo "  - 24 DI lines"
echo "  - 24 DO lines"
echo "  - 4 CI channels"
echo "  - 5 CO channels"
echo ""
echo "Persisted task SimTask1 created with:"
echo "  - 4 AI voltage channels (SimDev1/ai0:3)"
echo "  - 10 kHz sample rate"
echo "  - 1000 samples/channel buffer"
echo "  - Continuous acquisition mode"
echo ""
echo "Run simulated tests with:"
echo "  uv run pytest -m simulated -v"
echo ""
