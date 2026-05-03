#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/kws20_measure_config.env"

echo "Starting KWS20 live benchmark"
echo "Project:  $KWS20_PROJECT"
echo "Board:    $KWS20_BOARD"
echo "Port:     $KWS20_PORT"
echo "Duration: ${KWS20_DURATION}s"
echo

if [ ! -f "$KWS20_SCRIPT" ]; then
    echo "ERROR: measurement script not found: $KWS20_SCRIPT"
    exit 1
fi

if [ ! -d "$KWS20_VENV" ]; then
    echo "ERROR: venv not found: $KWS20_VENV"
    exit 1
fi

if [ ! -e "$KWS20_PORT" ]; then
    echo "ERROR: serial port not found: $KWS20_PORT"
    ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
    exit 1
fi

sudo fuser -k "$KWS20_PORT" 2>/dev/null || true
sleep 1

source "$KWS20_VENV/bin/activate"

ARGS=(
  --mode live
  --project "$KWS20_PROJECT"
  --board "$KWS20_BOARD"
  --port "$KWS20_PORT"
  --baud "$KWS20_BAUD"
  --duration "$KWS20_DURATION"
  --sample-rate "$KWS20_SAMPLE_RATE"
  --sample-count "$KWS20_SAMPLE_COUNT"
  --clock-mhz "$KWS20_CLOCK_MHZ"
)

if [ -n "${KWS20_MAC_OPS}" ]; then
  ARGS+=(--mac-ops "$KWS20_MAC_OPS")
fi

if [ -n "${KWS20_VOLTAGE}" ] && [ -n "${KWS20_CURRENT_MA}" ]; then
  ARGS+=(--voltage "$KWS20_VOLTAGE" --current-ma "$KWS20_CURRENT_MA")
fi

"$KWS20_SCRIPT" "${ARGS[@]}"

echo
echo "Latest result folder:"
ls -td "$HOME/max78000/measurements/kws20_all_"* | head -1
