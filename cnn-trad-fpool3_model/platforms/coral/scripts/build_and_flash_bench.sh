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

echo "=== Build kws_bench ==="
# flashtool.py needs two infra artifacts besides the app: elf_loader
# (apps/elf_loader/image.srec) and flashloader (libs/nxp/flashloader/image.srec).
# Build them alongside the app so a fresh build/ has everything the flasher needs.
cmake --build "$BUILD_DIR" -j"$JOBS" --target elf_loader flashloader kws_bench
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
