"""GAP9 EVK process-based live-inference source.

The GAP9 has no USB-CDC serial console: firmware printf goes over JTAG
semihosting through OpenOCD, which runs inside the Deeploy Docker container
(platforms/gap9/WORKFLOW.md). The dashboard therefore launches the documented
docker command (build + flash + run) and parses its stdout instead of opening
a serial port. Flashing and monitoring are one step — JTAG is exclusive.

Host prerequisite: the ftdi_sio kernel driver must not claim the GAP9 FTDI
(0403:6011) interfaces. Either unbind manually after plugging the board in
(WORKFLOW.md step 2) or install the udev rule shipped next to the dashboard
(99-gap9-ftdi.rules, see README) once — then this works without a terminal.
"""
from __future__ import annotations

import shutil
from pathlib import Path

# mcu_stream/ -> dashboard/ -> platforms/
PLATFORMS_DIR = Path(__file__).resolve().parents[2]
DEEPLOY_DIR = PLATFORMS_DIR / "gap9" / "Deeploy"

IMAGE = "ghcr.io/pulp-platform/deeploy-gap9:latest"
CONTAINER = "kws-gap9-live"

# Mirrors WORKFLOW.md steps 3-5: env setup, openocd vid/pid fix, Deeploy
# install (container is ephemeral), then the live-inference runner.
# `python -u` + docker `-t` keep the semihosting output line-buffered so the
# dashboard sees lines as they happen.
_CONTAINER_SCRIPT = """\
source /app/install/gap9-sdk/.gap9-venv/bin/activate
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
export GAP_SDK_VERSION=dev
sed -i "s/^ftdi_vid_pid.*/ftdi_vid_pid 0x0403 0x6011/" \
  /app/install/gap9-sdk/utils/openocd_tools/tcl/gapuino_ftdi.cfg
cd /app/Deeploy && pip install -e . -q
cd DeeployTest
exec python -u deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board -D LIVE_INFERENCE=ON
"""


def gap9_live_command() -> list[str]:
    """docker run command whose stdout streams the live-inference output."""
    return [
        "docker", "run", "--rm", "-t", "--privileged", "--network", "host",
        "--name", CONTAINER,
        "-v", f"{DEEPLOY_DIR}:/app/Deeploy",
        "-v", "/dev/bus/usb:/dev/bus/usb",
        IMAGE, "/bin/bash", "-lc", _CONTAINER_SCRIPT,
    ]


def gap9_cleanup_command() -> list[str]:
    """Remove a stale container left over from a crashed session."""
    return ["docker", "rm", "-f", CONTAINER]


def gap9_stop_command() -> list[str]:
    """Stop the running container (SIGTERM, SIGKILL after 2 s)."""
    return ["docker", "stop", "-t", "2", CONTAINER]


def gap9_available() -> bool:
    return DEEPLOY_DIR.is_dir() and shutil.which("docker") is not None
