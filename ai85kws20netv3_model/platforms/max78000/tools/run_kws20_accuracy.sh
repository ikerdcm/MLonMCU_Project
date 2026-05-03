#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MANIFEST="$HOME/max78000/accuracy_test/manifest.csv"
PORT="/dev/ttyACM0"
VENV="$HOME/max78000/ai8x-synthesis/.venv"
SCRIPT="$SCRIPT_DIR/run_kws20_accuracy_playback.py"
RESPONSE_WINDOW="${1:-2.5}"

echo "Starting KWS20 board-level accuracy test"
echo "Manifest:        $MANIFEST"
echo "Port:            $PORT"
echo "Response window: ${RESPONSE_WINDOW}s"
echo

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Accuracy script not found: $SCRIPT"
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "ERROR: venv not found: $VENV"
    exit 1
fi

if [ ! -e "$PORT" ]; then
    echo "ERROR: Serial port not found: $PORT"
    echo "Available ports:"
    ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
    exit 1
fi

echo "Freeing serial port..."
sudo fuser -k "$PORT" 2>/dev/null || true
sleep 1

source "$VENV/bin/activate"

"$SCRIPT" \
  --manifest "$MANIFEST" \
  --port "$PORT" \
  --response-window "$RESPONSE_WINDOW" \
  --shuffle

echo
echo "Latest accuracy measurement:"
ls -td "$HOME/max78000/measurements/"*kws20_accuracy | head -1
