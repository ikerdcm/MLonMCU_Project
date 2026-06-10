#!/usr/bin/env bash
# §7.2 — Compile ds_cnn_l_static_v2.tflite (6-block, correct source) for Edge TPU.
#
# v1 bug: compile_edgetpu.sh defaulted to ds_cnn_l.tflite which has dynamic
#         batch shapes — the Edge TPU compiler rejects those.  The v1 static
#         model (ds_cnn_l_static.tflite) was produced from the 4-block
#         kws_ref_model, not the deployed 6-block DS-CNN-L.
#
# v2 fix: default to ds_cnn_l_static_v2.tflite (6-block, from convert_static_int8_v2.py).
#
# macOS compatibility: edgetpu_compiler is Linux x86_64 only.  When not found
# natively, this script falls back to Docker (linux/amd64 image with Rosetta
# emulation on Apple Silicon).  Docker Desktop must be running.
#
# Usage:
#   ./compile_edgetpu_v2.sh                       # uses ds_cnn_l_static_v2.tflite
#   ./compile_edgetpu_v2.sh path/to/model.tflite
#
# Prerequisite (Linux): run ./setup.sh once to install edgetpu_compiler.
# Prerequisite (macOS): Docker Desktop running.
#
# After compiling, evaluate accuracy (needs speech_commands dataset):
#   cd ../../training
#   python3 eval_quantized_model.py \
#       --tfl_file_name ../models/ds_cnn_l_static_v2.tflite \
#       --target_set test --data_dir ~/data/speech_commands_v002

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$CORAL_DIR/model"
MODELS_SHARED="$(cd "$CORAL_DIR/../../models" && pwd)"

INPUT_MODEL="${1:-$MODELS_SHARED/ds_cnn_l_static_v2.tflite}"

if [[ ! -f "$INPUT_MODEL" ]]; then
    echo "ERROR: model not found: $INPUT_MODEL"
    echo "Run convert_static_int8_v2.py first to produce ds_cnn_l_static_v2.tflite"
    exit 1
fi

MODEL_FILENAME="$(basename "$INPUT_MODEL")"
MODEL_STEM="${MODEL_FILENAME%.tflite}"
mkdir -p "$MODEL_DIR"

# ── Choose execution path: native vs Docker ──────────────────────────────────

if command -v edgetpu_compiler &>/dev/null; then
    echo "=== Edge TPU Compiler (native) ==="
    edgetpu_compiler --version 2>&1 | head -1
    echo "Input : $INPUT_MODEL"
    echo "Output: $MODEL_DIR/${MODEL_STEM}_edgetpu.tflite"
    echo ""
    edgetpu_compiler --out_dir "$MODEL_DIR" --show_operations "$INPUT_MODEL"

else
    echo "=== Edge TPU Compiler (Docker fallback — linux/amd64 via Rosetta) ==="
    if ! command -v docker &>/dev/null; then
        echo "ERROR: neither edgetpu_compiler nor docker found."
        echo "  Linux  : run ./setup.sh"
        echo "  macOS  : install Docker Desktop and ensure it is running"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        echo "ERROR: Docker daemon not running. Start Docker Desktop and retry."
        exit 1
    fi

    # Paths that Docker can mount (must be absolute)
    INPUT_ABS="$(cd "$(dirname "$INPUT_MODEL")" && pwd)/$(basename "$INPUT_MODEL")"
    MODEL_MOUNT="$(dirname "$INPUT_ABS")"
    OUT_MOUNT="$(cd "$MODEL_DIR" && pwd)"

    echo "Input : $INPUT_MODEL"
    echo "Output: $MODEL_DIR/${MODEL_STEM}_edgetpu.tflite"
    echo ""

    docker run --rm \
      --platform linux/amd64 \
      -v "$MODEL_MOUNT:/models_in:ro" \
      -v "$OUT_MOUNT:/out" \
      debian:bookworm-slim bash -c "
        set -e
        apt-get update -qq 2>&1 | tail -1
        apt-get install -y -qq curl gpg 2>&1 | tail -1
        curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
          | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg
        echo 'deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main' \
          > /etc/apt/sources.list.d/coral.list
        apt-get update -qq 2>&1 | tail -1
        apt-get install -y -qq edgetpu-compiler 2>&1 | tail -1
        edgetpu_compiler --out_dir /out --show_operations /models_in/$(basename "$INPUT_MODEL")
      "
fi

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

echo ""
echo "Evaluate accuracy (needs speech_commands dataset):"
echo "  cd ../../training"
echo "  python3 eval_quantized_model.py \\"
echo "      --tfl_file_name ../models/ds_cnn_l_static_v2.tflite \\"
echo "      --target_set test --data_dir ~/data/speech_commands_v002"
