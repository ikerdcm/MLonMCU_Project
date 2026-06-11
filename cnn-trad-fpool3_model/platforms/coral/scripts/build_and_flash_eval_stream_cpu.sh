#!/usr/bin/env bash
# Build + flash kws_eval_stream_cpu (streaming device-in-the-loop accuracy,
# fp32 M7 CPU, no Edge TPU; receives raw audio from testbench.py).
# Usage:  ./scripts/build_and_flash_eval_stream_cpu.sh [path/to/coralmicro]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$CORAL_DIR/build"
CORALMICRO_ROOT="${1:-$CORAL_DIR/coralmicro}"
VENV_PYTHON="$CORAL_DIR/.venv/bin/python3"
[[ -f "$VENV_PYTHON" ]] || { echo "ERROR: venv not found at $CORAL_DIR/.venv"; exit 1; }
[[ -d "$CORALMICRO_ROOT" ]] && CORALMICRO_ROOT="$(cd "$CORALMICRO_ROOT" && pwd)"
# The app bakes eval/audio_set.bin into LittleFS — generate a default if missing.
if [[ ! -f "$CORAL_DIR/eval/audio_set.bin" ]]; then
    echo "=== baking eval audio set (per-class 12) ==="
    python3 "$CORAL_DIR/../testbench/make_eval_audio_set.py" --per-class 12
fi
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]] || \
   ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=$CORALMICRO_ROOT" "$BUILD_DIR/CMakeCache.txt"; then
    echo "=== Configure (fresh) ==="; rm -rf "$BUILD_DIR"
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
else
    echo "=== Re-configure (pick up new apps) ==="
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" >/dev/null
fi
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "=== Build kws_eval_stream_cpu ==="
cmake --build "$BUILD_DIR" -j"$JOBS" --target elf_loader flashloader kws_eval_stream_cpu
echo "=== Flash kws_eval_stream_cpu ==="
DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH} \
"$VENV_PYTHON" "$CORALMICRO_ROOT/scripts/flashtool.py" \
    --build_dir "$BUILD_DIR" \
    --elf_path "$BUILD_DIR/kws_apps/kws_eval_stream_cpu/kws_eval_stream_cpu"
echo "Flash complete. Run: python3 ../testbench/testbench.py --board coral --model v0 --port <CDC>"
