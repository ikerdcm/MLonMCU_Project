#!/usr/bin/env bash
# Build and flash the kws_live app (live keyword spotting via microphone).
#
# Usage:  ./scripts/build_and_flash_live.sh [path/to/coralmicro]
#
# If the board is unresponsive (crash-loop), put it in SDP mode first:
#   Hold USER button → press RESET → release USER button → run this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$CORAL_DIR/build"
CORALMICRO_ROOT="${1:-$CORAL_DIR/coralmicro}"

VENV_PYTHON="$CORAL_DIR/.venv/bin/python3"
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "ERROR: venv not found at $CORAL_DIR/.venv — see README.md"
    exit 1
fi

echo "=== Build kws_live ==="
cmake --build "$BUILD_DIR" -j"$(nproc)" --target kws_live
echo ""

echo "=== Flash kws_live to Coral Dev Board Micro ==="
"$VENV_PYTHON" "$CORALMICRO_ROOT/scripts/flashtool.py" \
    --build_dir "$BUILD_DIR" \
    --app kws_live
echo ""
echo "Flash complete. Connect serial terminal:"
echo "  screen /dev/ttyACM0 115200"
echo ""
echo "Speak a keyword (left, right, yes, no, go, stop, up, down, on, off)."
echo "Output every 500 ms. Confident detections are marked with >>>."
