#!/usr/bin/env bash
# Build + flash kws_idle: ONE offline INT8 Edge-TPU inference every 5 s with the
# M7 in WFI sleep between. For power-meter duty-cycle measurement. Edge-TPU twin
# of build_and_flash_idle_cpu.sh (which is fixed to v0 / CPU).
#
# Usage:  ./scripts/build_and_flash_idle.sh [--version vNN | --model <basename>] [path/to/coralmicro]
#   --version selects ONE TPU network by its honest version id (keep in sync with
#   testbench.py REGISTRY). Default = v1 (int8-accel 6-block).
#     v1  int8-accel 6-blk | v21 prune f64b4 | v22 prune f32b6 | v23 prune f32b4
#     v31 distill f64b4    | v32 distill f32b4
#
# If the board is unresponsive (crash-loop), put it in SDP mode first:
#   Hold USER button -> press RESET -> release USER button -> run this script.
set -euo pipefail

# version id <-> edgetpu model basename. case-based (no associative arrays) so this
# runs on macOS's stock bash 3.2. ALL_VERSIONS is the selectable set.
ALL_VERSIONS="v1 v21 v22 v23 v31 v32"
version_to_model() {
    case "$1" in
        v1)  echo "ds_cnn_l_static_v2_edgetpu" ;;
        v21) echo "ds_cnn_l_pruned_f64b4_int8_edgetpu" ;;
        v22) echo "ds_cnn_l_pruned_f32b6_int8_edgetpu" ;;
        v23) echo "ds_cnn_l_pruned_f32b4_int8_edgetpu" ;;
        v31) echo "ds_cnn_l_distilled_f64b4_int8_edgetpu" ;;
        v32) echo "ds_cnn_l_distilled_f32b4_int8_edgetpu" ;;
        *)   echo "" ;;
    esac
}
model_to_version() {
    case "$1" in
        ds_cnn_l_static_v2_edgetpu)            echo "v1" ;;
        ds_cnn_l_pruned_f64b4_int8_edgetpu)    echo "v21" ;;
        ds_cnn_l_pruned_f32b6_int8_edgetpu)    echo "v22" ;;
        ds_cnn_l_pruned_f32b4_int8_edgetpu)    echo "v23" ;;
        ds_cnn_l_distilled_f64b4_int8_edgetpu) echo "v31" ;;
        ds_cnn_l_distilled_f32b4_int8_edgetpu) echo "v32" ;;
        *)                                     echo "unknown" ;;
    esac
}

IDLE_VERSION="v1"
IDLE_MODEL="$(version_to_model v1)"
if [[ "${1:-}" == "--version" ]]; then
    IDLE_VERSION="$2"; shift 2
    IDLE_MODEL="$(version_to_model "$IDLE_VERSION")"
    [[ -n "$IDLE_MODEL" ]] || { echo "ERROR: unknown --version '$IDLE_VERSION' (have: $ALL_VERSIONS)"; exit 1; }
elif [[ "${1:-}" == "--model" ]]; then
    IDLE_MODEL="$2"; shift 2
    IDLE_VERSION="$(model_to_version "$IDLE_MODEL")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$CORAL_DIR/build"
CORALMICRO_ROOT="${1:-$CORAL_DIR/coralmicro}"

VENV_PYTHON="$CORAL_DIR/.venv/bin/python3"
[[ -f "$VENV_PYTHON" ]] || { echo "ERROR: venv not found at $CORAL_DIR/.venv — see README.md"; exit 1; }
[[ -d "$CORALMICRO_ROOT" ]] && CORALMICRO_ROOT="$(cd "$CORALMICRO_ROOT" && pwd)"
[[ -f "$CORAL_DIR/model/${IDLE_MODEL}.tflite" ]] || \
    { echo "ERROR: model/${IDLE_MODEL}.tflite not found"; exit 1; }
echo "=== version ${IDLE_VERSION} -> ${IDLE_MODEL} ==="

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]] || \
   ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=$CORALMICRO_ROOT" "$BUILD_DIR/CMakeCache.txt"; then
    echo "=== Configure (fresh) ==="; rm -rf "$BUILD_DIR"
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DKWS_IDLE_MODEL="$IDLE_MODEL" -DKWS_IDLE_VERSION="$IDLE_VERSION"
else
    echo "=== Re-configure (pick up new app + idle model/version) ==="
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" \
        -DKWS_IDLE_MODEL="$IDLE_MODEL" -DKWS_IDLE_VERSION="$IDLE_VERSION" >/dev/null
fi

JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "=== Build kws_idle ==="
cmake --build "$BUILD_DIR" -j"$JOBS" --target elf_loader flashloader kws_idle

echo "=== Flash kws_idle ==="
DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH} \
"$VENV_PYTHON" "$CORALMICRO_ROOT/scripts/flashtool.py" \
    --build_dir "$BUILD_DIR" \
    --elf_path "$BUILD_DIR/kws_apps/kws_idle/kws_idle"
echo ""
echo "Flash complete: ${IDLE_VERSION} (Edge TPU, ${IDLE_MODEL}). 1 inference / 5 s, WFI sleep between."
echo "Watch the duty cycle:"
echo "  screen \$(ls /dev/tty.usbmodem* 2>/dev/null | head -1) 115200"
echo "Each cycle prints BENCH,event=inference (cnn_us) then BENCH,event=sleep."
