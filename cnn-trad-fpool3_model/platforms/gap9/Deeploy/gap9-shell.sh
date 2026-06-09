#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VID="0403"
PID="6011"
IMAGE="ghcr.io/pulp-platform/deeploy-gap9:latest"

echo "[1/4] Checking GAP9 FTDI device ${VID}:${PID}..."

if ! lsusb -d "${VID}:${PID}" >/dev/null 2>&1; then
  echo "[ERROR] GAP9 FTDI device not found."
  echo "Plug in the GAP9 board and check:"
  echo "  lsusb | grep -i \"${VID}:${PID}\""
  exit 1
fi

lsusb | grep -i "${VID}:${PID}" || true

echo
echo "[2/4] Unbinding ftdi_sio from GAP9 interfaces if needed..."

for dev in /sys/bus/usb/devices/*; do
  if [[ -f "$dev/idVendor" && -f "$dev/idProduct" ]]; then
    if [[ "$(cat "$dev/idVendor")" == "$VID" && "$(cat "$dev/idProduct")" == "$PID" ]]; then
      usbdev="$(basename "$dev")"

      for iface in "$dev":*; do
        [[ -e "$iface" ]] || continue
        iface_name="$(basename "$iface")"

        if [[ -L "$iface/driver" ]]; then
          driver="$(basename "$(readlink "$iface/driver")")"

          if [[ "$driver" == "ftdi_sio" ]]; then
            echo "  unbind $iface_name from ftdi_sio"
            echo "$iface_name" | sudo tee /sys/bus/usb/drivers/ftdi_sio/unbind >/dev/null
          else
            echo "  $iface_name uses driver $driver, leaving it alone"
          fi
        else
          echo "  $iface_name already has no driver"
        fi
      done
    fi
  fi
done

echo
echo "[3/4] Current USB tree:"
lsusb -t | sed -n '/0403\|6011/,+4p' || true

echo
echo "[4/4] Starting GAP9 Docker container with direct USB access..."
echo

docker run -it --rm --privileged --network host \
  -v "$PWD":/app/Deeploy \
  -v /dev/bus/usb:/dev/bus/usb \
  "$IMAGE" \
  /bin/bash -lc '
    set -e

    source /app/install/gap9-sdk/.gap9-venv/bin/activate
    source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
    export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
    export GAP_SDK_VERSION=dev

    sed -i "s/^ftdi_vid_pid.*/ftdi_vid_pid 0x0403 0x6011/" \
      /app/install/gap9-sdk/utils/openocd_tools/tcl/gapuino_ftdi.cfg

    echo
    echo "[READY] GAP9 container is ready."
    echo "Board config:"
    grep -n "ftdi_vid_pid" /app/install/gap9-sdk/utils/openocd_tools/tcl/gapuino_ftdi.cfg
    echo
    echo "Test command:"
    echo "  cd /app/install/gap9-sdk/examples/gap9/basic/helloworld"
    echo "  rm -rf build"
    echo "  cmake -B build"
    echo "  cmake --build build --target menuconfig"
    echo "  cmake --build build --target run"
    echo

    exec /bin/zsh
  '
