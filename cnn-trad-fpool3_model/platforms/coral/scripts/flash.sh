#!/usr/bin/env bash
# Flash the kws_coral firmware to Coral Dev Board Micro via USB.
#
# Usage:  ./flash.sh [path/to/coralmicro]
#
# Prerequisites:
#   - Coral Dev Board Micro connected via USB
#   - coralmicro SDK cloned (see README.md)
#   - Firmware already built (see README.md build steps)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"

CORALMICRO_ROOT="${1:-$CORAL_DIR/coralmicro}"
BUILD_DIR="$CORAL_DIR/build"

if [[ ! -d "$CORALMICRO_ROOT" ]]; then
    echo "ERROR: coralmicro SDK not found at: $CORALMICRO_ROOT"
    echo "Clone it first: git clone --recurse-submodules https://github.com/google-coral/coralmicro"
    exit 1
fi

FLASHTOOL="$CORALMICRO_ROOT/scripts/flashtool.py"
if [[ ! -f "$FLASHTOOL" ]]; then
    echo "ERROR: flashtool.py not found at $FLASHTOOL"
    exit 1
fi

if [[ ! -f "$BUILD_DIR/apps/kws_coral/kws_coral" ]]; then
    echo "ERROR: firmware binary not found: $BUILD_DIR/apps/kws_coral/kws_coral"
    echo "Run the build steps in README.md first."
    exit 1
fi

# Use venv Python if available (has hidapi + flashtool deps)
VENV_PYTHON="$CORAL_DIR/.venv/bin/python3"
PYTHON="python3"
if [[ -f "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
fi

echo "=== Flashing kws_coral to Coral Dev Board Micro ==="
echo "Build  : $BUILD_DIR"
echo "Python : $PYTHON"
echo ""

"$PYTHON" "$FLASHTOOL" \
    --build_dir "$BUILD_DIR" \
    --app kws_coral

echo ""
echo "Flash complete. Connect a serial terminal at 115200 baud to see output."
