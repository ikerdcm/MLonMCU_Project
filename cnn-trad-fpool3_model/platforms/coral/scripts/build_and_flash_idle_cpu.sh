#!/usr/bin/env bash
# Build + flash kws_idle_cpu: ONE offline FP32 inference every 5 s with the M7 in
# WFI sleep between (config: fp32-cpu, NO Edge TPU). For power-meter duty-cycle
# measurement. Counterpart of build_and_flash_bench_cpu.sh.
#
# Usage:  ./scripts/build_and_flash_idle_cpu.sh [path/to/coralmicro]
#
# If the board is unresponsive (crash-loop), put it in SDP mode first:
#   Hold USER button -> press RESET -> release USER button -> run this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$CORAL_DIR/build"
CORALMICRO_ROOT="${1:-$CORAL_DIR/coralmicro}"

VENV_PYTHON="$CORAL_DIR/.venv/bin/python3"
[[ -f "$VENV_PYTHON" ]] || { echo "ERROR: venv not found at $CORAL_DIR/.venv — see README.md"; exit 1; }
[[ -d "$CORALMICRO_ROOT" ]] && CORALMICRO_ROOT="$(cd "$CORALMICRO_ROOT" && pwd)"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]] || \
   ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=$CORALMICRO_ROOT" "$BUILD_DIR/CMakeCache.txt"; then
    echo "=== Configure (fresh) ==="; rm -rf "$BUILD_DIR"
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
else
    echo "=== Re-configure (pick up new app) ==="
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" >/dev/null
fi

JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "=== Build kws_idle_cpu ==="
cmake --build "$BUILD_DIR" -j"$JOBS" --target elf_loader flashloader kws_idle_cpu

echo "=== Flash kws_idle_cpu ==="
DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH} \
"$VENV_PYTHON" "$CORALMICRO_ROOT/scripts/flashtool.py" \
    --build_dir "$BUILD_DIR" \
    --elf_path "$BUILD_DIR/kws_apps/kws_idle_cpu/kws_idle_cpu"
echo ""
echo "Flash complete. The board now runs 1 inference / 5 s, WFI sleep between."
echo "Watch the duty cycle:"
echo "  screen \$(ls /dev/tty.usbmodem* 2>/dev/null | head -1) 115200"
echo "Each cycle prints BENCH,event=inference (cnn_us) then BENCH,event=sleep."
