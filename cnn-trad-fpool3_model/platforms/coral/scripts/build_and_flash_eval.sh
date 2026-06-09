#!/usr/bin/env bash
# Build + flash the kws_eval app (on-device accuracy, device-in-the-loop, v2 int8).
# Usage:  ./scripts/build_and_flash_eval.sh [path/to/coralmicro]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$CORAL_DIR/build"
CORALMICRO_ROOT="${1:-$CORAL_DIR/coralmicro}"
VENV_PYTHON="$CORAL_DIR/.venv/bin/python3"
[[ -f "$VENV_PYTHON" ]] || { echo "ERROR: venv not found at $CORAL_DIR/.venv"; exit 1; }
[[ -d "$CORALMICRO_ROOT" ]] && CORALMICRO_ROOT="$(cd "$CORALMICRO_ROOT" && pwd)"
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]] || \
   ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=$CORALMICRO_ROOT" "$BUILD_DIR/CMakeCache.txt"; then
    echo "=== Configure ==="; rm -rf "$BUILD_DIR"
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
fi
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "=== Build kws_eval ==="
cmake --build "$BUILD_DIR" -j"$JOBS" --target elf_loader flashloader kws_eval
echo "=== Flash kws_eval ==="
DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH} \
"$VENV_PYTHON" "$CORALMICRO_ROOT/scripts/flashtool.py" \
    --build_dir "$BUILD_DIR" \
    --elf_path "$BUILD_DIR/kws_apps/kws_eval/kws_eval"
echo "Flash complete. Read EVAL lines on the CDC port."
