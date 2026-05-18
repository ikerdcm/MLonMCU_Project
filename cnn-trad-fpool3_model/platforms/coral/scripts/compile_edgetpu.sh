#!/usr/bin/env bash
# Compile ds_cnn_l.tflite for the Edge TPU.
#
# Usage:  ./compile_edgetpu.sh [path/to/ds_cnn_l.tflite]
#
# Output: model/ds_cnn_l_edgetpu.tflite
#         model/ds_cnn_l_edgetpu.log
#
# Prerequisite: run ./setup.sh once to install edgetpu_compiler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$CORAL_DIR/model"

INPUT_MODEL="${1:-$(cd "$CORAL_DIR/../../models" && pwd)/ds_cnn_l.tflite}"

if ! command -v edgetpu_compiler &>/dev/null; then
    echo "ERROR: edgetpu_compiler not found. Run ./setup.sh first."
    exit 1
fi

if [[ ! -f "$INPUT_MODEL" ]]; then
    echo "ERROR: model not found: $INPUT_MODEL"
    exit 1
fi

MODEL_FILENAME="$(basename "$INPUT_MODEL")"
MODEL_STEM="${MODEL_FILENAME%.tflite}"

mkdir -p "$MODEL_DIR"

echo "=== Edge TPU Compiler ==="
edgetpu_compiler --version 2>&1 | head -1
echo "Input : $INPUT_MODEL"
echo "Output: $MODEL_DIR/${MODEL_STEM}_edgetpu.tflite"
echo ""

edgetpu_compiler \
    --out_dir "$MODEL_DIR" \
    --show_operations \
    "$INPUT_MODEL"

echo ""
echo "Done. Compiled model: $MODEL_DIR/${MODEL_STEM}_edgetpu.tflite"
echo ""

LOG="$MODEL_DIR/${MODEL_STEM}_edgetpu.log"
if [[ -f "$LOG" ]]; then
    ON_TPU=$(grep -c "Mapped to Edge TPU" "$LOG" 2>/dev/null || true)
    OFF_TPU=$(grep -c "Mapped to CPU"     "$LOG" 2>/dev/null || true)
    echo "Ops on Edge TPU      : $ON_TPU"
    echo "Ops on CPU (fallback): $OFF_TPU"
    if [[ "$OFF_TPU" -gt 0 ]]; then
        echo "WARNING: $OFF_TPU op(s) fell back to CPU — inference will be slower."
    fi
fi
