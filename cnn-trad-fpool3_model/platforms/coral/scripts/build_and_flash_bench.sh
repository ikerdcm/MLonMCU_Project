#!/usr/bin/env bash
# Build and flash the kws_bench app (50-inference offline benchmark).
#
# Usage:  ./scripts/build_and_flash_bench.sh [path/to/coralmicro]
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

echo "=== Build kws_bench ==="
cmake --build "$BUILD_DIR" -j"$(nproc)" --target kws_bench
echo ""

echo "=== Flash kws_bench to Coral Dev Board Micro ==="
DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH} \
"$VENV_PYTHON" "$CORALMICRO_ROOT/scripts/flashtool.py" \
    --build_dir "$BUILD_DIR" \
    --elf_path "$BUILD_DIR/kws_apps/kws_bench/kws_bench"
echo ""
echo "Flash complete. Connect serial terminal:"
echo "  screen \$(ls /dev/tty.usbmodem* 2>/dev/null | head -1) 115200"
echo ""
echo "Output: 50 BENCH CSV lines then summary stats."
