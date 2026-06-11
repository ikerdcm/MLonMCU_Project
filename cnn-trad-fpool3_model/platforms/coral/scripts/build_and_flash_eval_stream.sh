#!/usr/bin/env bash
# Build + flash the kws_eval_stream app (streaming device-in-the-loop accuracy,
# Edge-TPU int8; receives raw audio from testbench.py).
# Usage:  ./scripts/build_and_flash_eval_stream.sh [--version vNN | --model <basename>] [path/to/coralmicro]
#   --version selects ONE network by its honest version id (preferred; keep in
#   sync with testbench.py REGISTRY). --model takes a raw edgetpu basename instead.
#   Default = v1 (int8-accel 6-block).
#     v1  int8-accel 6-blk   | v21 prune f64b4 | v22 prune f32b6 | v23 prune f32b4
#     v31 distill f64b4      | v32 distill f32b4
set -euo pipefail

# version id -> edgetpu model basename (the ONLY network that gets flashed).
declare -A VERSION_MODEL=(
    [v1]="ds_cnn_l_static_v2_edgetpu"
    [v21]="ds_cnn_l_pruned_f64b4_int8_edgetpu"
    [v22]="ds_cnn_l_pruned_f32b6_int8_edgetpu"
    [v23]="ds_cnn_l_pruned_f32b4_int8_edgetpu"
    [v31]="ds_cnn_l_distilled_f64b4_int8_edgetpu"
    [v32]="ds_cnn_l_distilled_f32b4_int8_edgetpu"
)
# version id reported by the firmware (reverse lookup from basename).
declare -A MODEL_VERSION
for v in "${!VERSION_MODEL[@]}"; do MODEL_VERSION["${VERSION_MODEL[$v]}"]="$v"; done

EVAL_VERSION="v1"
EVAL_MODEL="${VERSION_MODEL[v1]}"
if [[ "${1:-}" == "--version" ]]; then
    EVAL_VERSION="$2"; shift 2
    EVAL_MODEL="${VERSION_MODEL[$EVAL_VERSION]:-}"
    [[ -n "$EVAL_MODEL" ]] || { echo "ERROR: unknown --version '$EVAL_VERSION' (have: ${!VERSION_MODEL[*]})"; exit 1; }
elif [[ "${1:-}" == "--model" ]]; then
    EVAL_MODEL="$2"; shift 2
    EVAL_VERSION="${MODEL_VERSION[$EVAL_MODEL]:-unknown}"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$CORAL_DIR/build"
CORALMICRO_ROOT="${1:-$CORAL_DIR/coralmicro}"
[[ -f "$CORAL_DIR/model/${EVAL_MODEL}.tflite" ]] || \
    { echo "ERROR: model/${EVAL_MODEL}.tflite not found"; exit 1; }
echo "=== version ${EVAL_VERSION} -> ${EVAL_MODEL} ==="
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
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DKWS_EVAL_STREAM_MODEL="$EVAL_MODEL" -DKWS_EVAL_STREAM_VERSION="$EVAL_VERSION"
else
    # Light re-configure so a newly-added app (add_subdirectory) is picked up AND
    # the selected model/version cache vars are refreshed (so swapping variants re-bakes).
    echo "=== Re-configure (pick up new apps + eval model) ==="
    cmake -S "$CORALMICRO_ROOT" -B "$BUILD_DIR" \
        -DKWS_EVAL_STREAM_MODEL="$EVAL_MODEL" -DKWS_EVAL_STREAM_VERSION="$EVAL_VERSION" >/dev/null
fi
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "=== Build kws_eval_stream ==="
cmake --build "$BUILD_DIR" -j"$JOBS" --target elf_loader flashloader kws_eval_stream
echo "=== Flash kws_eval_stream ==="
DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH} \
"$VENV_PYTHON" "$CORALMICRO_ROOT/scripts/flashtool.py" \
    --build_dir "$BUILD_DIR" \
    --elf_path "$BUILD_DIR/kws_apps/kws_eval_stream/kws_eval_stream"
echo "Flash complete: version ${EVAL_VERSION} (baked: ${EVAL_MODEL})."
echo "Read it with the SAME version id:"
echo "  python3 ../testbench/testbench.py --board coral --model ${EVAL_VERSION} --port <CDC>"
