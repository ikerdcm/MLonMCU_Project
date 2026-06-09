#!/usr/bin/env bash
# Build and flash the kws_live_cpu app (live keyword spotting via microphone,
# FP32 on the M7 CPU — config: fp32-cpu, NO Edge TPU). Counterpart of
# build_and_flash_live.sh; uses models/ds_cnn_l_float.tflite.
#
# Usage:  ./scripts/build_and_flash_live_cpu.sh [path/to/coralmicro]
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

# --- Ensure build/ is configured for THIS worktree ---
# A CMakeCache.txt created in another checkout (e.g. a sibling git worktree)
# cannot be reused and makes cmake abort; detect that and (re)configure.
if [[ -d "$CORALMICRO_ROOT" ]]; then
    CORALMICRO_ROOT="$(cd "$CORALMICRO_ROOT" && pwd)"
fi
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]] || \
   ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=$CORALMICRO_ROOT" "$BUILD_DIR/CMakeCache.txt"; then
    echo "=== Configure (build/ missing or from another worktree) ==="
    rm -rf "$BUILD_DIR"
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    echo ""
fi

JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "=== Build kws_live_cpu ==="
# flashtool.py needs two infra artifacts besides the app: elf_loader
# (apps/elf_loader/image.srec) and flashloader (libs/nxp/flashloader/image.srec).
# Build them alongside the app so a fresh build/ has everything the flasher needs.
cmake --build "$BUILD_DIR" -j"$JOBS" --target elf_loader flashloader kws_live_cpu
echo ""

echo "=== Flash kws_live_cpu to Coral Dev Board Micro ==="
DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH} \
"$VENV_PYTHON" "$CORALMICRO_ROOT/scripts/flashtool.py" \
    --build_dir "$BUILD_DIR" \
    --elf_path "$BUILD_DIR/kws_apps/kws_live_cpu/kws_live_cpu"
echo ""
echo "Flash complete. Connect serial terminal (must assert DTR — screen/cat won't show output):"
echo "  $VENV_PYTHON -m serial.tools.miniterm \$(ls /dev/cu.usbmodem* 2>/dev/null | head -1) 115200"
echo ""
echo "Speak a keyword (left, right, yes, no, go, stop, up, down, on, off)."
echo "Confident detections are marked with >>>."
